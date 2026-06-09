# bfd_cnf

**Conditional Normalizing Flow prior for BFD weak-lensing shear estimation**

This package implements a learned normalizing-flow prior and variational posterior over galaxy moment statistics within the [BFD (Bayesian Fourier Domain)](https://arxiv.org/abs/1508.05655) weak-lensing shear estimation formalism. Rather than forward-modelling galaxy images at the pixel level, BFD works with pre-computed Fourier-domain moment summaries of each galaxy. This package replaces the original analytic moment-space priors with learned conditional flows, enabling more flexible modelling of the galaxy moment distribution and improved multiplicative bias performance.

---

## Overview

BFD estimates shear by marginalising over the intrinsic galaxy population at the level of **galaxy moments** (flux, size, and ellipticity moments in Fourier space), using Taylor-expanded shear derivatives (`dm/dg`, `d²m/dg²`) to propagate the effect of shear through the moment distribution — no image-level forward model is involved.

This package replaces the analytic BFD prior with two normalizing flows trained jointly via an ELBO objective:

- **Prior flow** `p(z | g, Σ_X)` — models the distribution of galaxy moments conditioned on the applied shear `g` and PSF noise covariance `Σ_X`.
- **Variational q flow** `q(z | y, Σ_y, g)` — approximates the posterior over latent moments given an observed noisy measurement `y` and its covariance `Σ_y`.

At inference time, the trained prior flow is used to compute per-object BFD **PQR** statistics (probability, first and second shear derivatives) via Randomized Quasi-Monte Carlo (RQMC) integration. These are then combined across a galaxy catalogue to estimate the shear vector and multiplicative bias.

---

## Model architecture

Both flows act on the 4-D **standardised** moment vector `z = φ(M_raw)`, where
`φ` maps the raw moments `[Mf, Mr, M1, M2]` to the whitened coordinates
`[log10 Mf, Mr/Mf, M1/Mr, M2/Mr]` (`RawMomentStandardize`). Noise covariances
are propagated through `φ` with a first- plus second-order Jacobian correction.

### Prior flow `p_θ(z | g, Σ_X)`

Built by `models.flows.build_flows` → `models.bijections.new_masked_autoregressive_flow`:

- **`prior_flow_layers − 1` equivariant layers** (`EquivariantAutoregressiveLayer`):
  each applies a spin-0 autoregressive step on the scalar moments `(z0, z1)`
  followed by a spin-2 coupling step on the ellipticity moments `(z2, z3)` that
  preserves the rotational (spin-2) symmetry. Early layers are separated by a
  *deterministic* permutation of the spin-0 pair (alternating); a single
  *random* spin-equivariant permutation follows the final equivariant layer.
- **Shear layer** (`ExplicitPolyLast`): a conditional last layer whose scale and
  shift coefficients are an at-most-quadratic polynomial in the reduced shear
  `g = (g1, g2)`, matching the BFD second-order Taylor expansion
  `P(M|g) ≈ P + Q·g + ½ g·R·g`.
- **PSF centroid layer** (`SigmaXCouplingLayer`): conditions on the PSF centroid
  noise covariance `C_X` via `[log_scale, e1, e2]` (with `log_scale = ½ log det C_X`),
  applying a spin-0 rescaling plus a spin-2 map `A = s·I + c·E`,
  `E = [[e1, e2], [e2, −e1]]`, that is first-order sensitive to PSF ellipticity.

The full prior condition vector is **`[g1, g2, log_scale, e1, e2]`** (the shear
layer reads only `g`; the PSF layer reads `[log_scale, e1, e2]`).

### Variational q flow `q_φ(z | y, Σ_y, g)`

A standard flowjax masked-autoregressive flow with a `BoundedAffine` transformer
(`q_flow_layers` layers), conditioned on the 16-D feature vector from
`cond_features_from_y_sigma`: standardised moments `ỹ` (4), off-diagonal
correlations of `Σ̃_y` (6), the log-diagonal of its Cholesky factor (4), and
`g / g_scale` (2).

### Training objective

The flows are trained jointly by an **importance-weighted ELBO** with three
ingredients: the Gaussian measurement likelihood `N(M_sheared | v, Σ̃)`, a flux
**selection correction** `−log P_sel` for the `Mf > target_flux_min` cut, and
**Σ_X (centroid-noise) marginalisation**. For each of `n_sx_train` PSF
covariances `C_X` drawn per step, the prior is trained to be the template
distribution *marginalised over the centroid offset* `X = [MX, MY]`, weighting
each template copy by its centroid likelihood `N(X; 0, C_X)`; the per-`C_X`
losses are averaged. See `FlowMath.tex` for the full per-layer equations and the
loss derivation.

---

## Repository structure

```
bfd_cnf/                   # repo root
├── bfd_cnf/               # Python package  →  run as `python -m bfd_cnf.run`
│   ├── config.py          # Hyperparameters, file paths, fixed matrices, PRNG key
│   ├── data.py            # Data loading, moment augmentation, standardisation
│   ├── models/
│   │   ├── bijections.py  # Custom flowjax bijections (BoundedAffine,
│   │   │                  #   RawMomentStandardize, SigmaXCouplingLayer, …)
│   │   └── flows.py       # Flow construction (build_flows), conditioning
│   │                      #   features, ELBO loss factory (make_elbo_loss)
│   ├── training.py        # Training loop wrappers, model save/load utilities
│   ├── inference.py       # RQMC PQR integration, flow probability helpers
│   ├── statistics.py      # PQR → shear estimation, multiplicative bias, bootstrap
│   ├── viz.py             # Diagnostic plots (corner plots, Q/R histograms, …)
│   ├── run.py             # End-to-end workflow script
│   ├── train_from_scratch.py / continue_train.py  # Training entry points
│   ├── integrate_grid.py  # Standalone RQMC PQR grid-integration CLI
│   ├── plot_corner.py / plot_shear_derivs.py      # Diagnostic plot CLIs
│   ├── validate_flow_vs_target.py                 # Flow-vs-target sanity check
│   └── make_tmpl_t04_joined.py                    # Template FITS builder
├── flows/                 # Saved flow weights (*.eqx) and timestamped backups
├── plots/                 # Generated figures (*.png)
├── data/                  # PQR grid outputs and cached standardiser stats (*.npz)
├── logs/                  # Training / integration run logs
└── attic/                 # Archived notebooks and one-off debug scripts
```

Run package commands from the **repo root** (`bfd_cnf/`), e.g. `python -m bfd_cnf.run`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `jax` / `jaxlib` | Array operations, JIT, autodiff |
| `equinox` | JAX neural networks and pytree serialisation |
| `flowjax` | Normalizing flow primitives |
| `optax` | AdamW optimiser and gradient clipping |
| `paramax` | `non_trainable` wrapper for frozen parameters |
| `jaxopt` | L-BFGS mode-finding for RQMC proposals |
| `numpy` | CPU-side array handling |
| `fitsio` | FITS table I/O for template catalogues |
| `bfd` | BFD moment utilities (`TemplateTable`, `MomentCovariance`, `stripMuPqr`, …) |
| `matplotlib` / `corner` | Visualisation |

Install dependencies with:

```bash
pip install jax equinox flowjax optax paramax jaxopt numpy fitsio matplotlib corner
```

> **Note:** `bfd` is not on PyPI — install it from its source repository.

---

## Quick start

### Run the full workflow

```bash
python -m bfd_cnf.run
```

This will:
1. Load the BFD template FITS catalogue (path set in `config.py`)
2. Train (or load) the prior and q flows
3. Run an RQMC PQR sanity check on a single object
4. Compute flow-based PQR for ±shear galaxy grids
5. Estimate the shear vector and bootstrap the multiplicative bias
6. Produce diagnostic plots

### Use individual modules

```python
import jax.random as jr
from bfd_cnf.data import load_data
from bfd_cnf.training import load_or_train
from bfd_cnf.inference import make_flow_prob_and_derivs, rqmc_pqr_grid
from bfd_cnf.statistics import bootstrap_total_mult_bias

key = jr.key(0)
data = load_data(key=key)

prior, q_flow, losses = load_or_train(
    key,
    data["moments_jnp"],
    data["centroid_moments_jnp"],
    data["cov_jnp"],
    data["dm_dg_jnp"],
    data["d2m_dg2_jnp"],
    data["weights"],
    data["raw2standard"],
)

flow_prob_and_derivs = make_flow_prob_and_derivs(prior)
```

---

## Configuration

All key settings live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `prior_flow_layers` | 10 | Coupling layers in the prior flow |
| `prior_early_nn_width` / `depth` | 32 / 2 | Width and depth of early-layer coefficient networks |
| `prior_last_nn_width` / `depth` | 32 / 4 | Width and depth of last-layer coefficient networks |
| `prior_sigmax_nn_width` / `depth` | 32 / 4 | Width and depth of Σ_X conditioning networks |
| `q_flow_layers` | 6 | Masked autoregressive layers in the q flow |
| `q_nn_width` / `depth` | 64 / 4 | Width and depth of q-flow coefficient networks |
| `batch_size` | 1024 | Training mini-batch size |
| `num_samples` | 4 | MC samples per batch element in ELBO |
| `n_sx_train` | 8 | Σ_X conditions sampled per gradient step (losses are averaged) |
| `log_scale_range` | (11.0, 15.0) | Range of 0.5 · log det(Σ_X) sampled per step for Σ_X conditioning; spans the corner-plot reference (≈11.9) through the grid targets' per-object scales (median ≈13.3) so the prior is in-range for both |
| `e_max` | 0.2 | Maximum PSF ellipticity magnitude for Σ_X conditioning; must be ≥ inference max |
| `target_flux_min` | 1500.0 | Minimum Mf for selection (must match catalogue and `fluxMin` in `TemplateTable`) |
| `FITS_PATH` | *(local path to `tmpl_t04_joined.fits`)* | Template FITS catalogue |
| `PRIOR_FLOW_PATH` | `flows/prior_flow_xy.eqx` | Saved prior flow weights |
| `Q_FLOW_PATH` | `flows/q_flow_xy.eqx` | Saved q flow weights |
| `GRID_P_PATH` / `GRID_M_PATH` | *(local paths)* | ±shear galaxy simulation grids |

---

## Model serialisation

Trained flow weights are saved and loaded with `equinox`:

```python
from bfd_cnf.training import save_models, load_models, load_or_train

# Save after training
save_models(prior_trained, q_trained)

# Load on subsequent runs (load_or_train does this automatically)
prior_trained, q_trained, _ = load_or_train(key, ...)
```

---

## License

*(Add license here)*
