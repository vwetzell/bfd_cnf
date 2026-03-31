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

## Repository structure

```
bfd_cnf/
├── config.py          # Hyperparameters, file paths, fixed matrices, PRNG key
├── data.py            # Data loading, moment augmentation, standardisation
├── models/
│   ├── bijections.py  # Custom flowjax bijections (BoundedAffine,
│   │                  #   RawMomentStandardize, SigmaXCouplingLayer, …)
│   └── flows.py       # Flow construction (build_flows), conditioning
│                      #   features, ELBO loss factory (make_elbo_loss)
├── training.py        # Training loop wrappers, model save/load utilities
├── inference.py       # RQMC PQR integration, flow probability helpers
├── statistics.py      # PQR → shear estimation, multiplicative bias,
│                      #   bootstrap uncertainty
├── viz.py             # Diagnostic plots (corner plots, Q/R histograms, …)
└── run.py             # End-to-end workflow script
```

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
| `astropy` | FITS table I/O for template catalogues |
| `bfd` | BFD moment utilities (`splitPqr`, `packPqr`, …) |
| `matplotlib` / `corner` | Visualisation |

Install dependencies with:

```bash
pip install jax equinox flowjax optax paramax jaxopt numpy astropy matplotlib corner
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
| `prior_flow_layers` | 13 | Coupling layers in the prior flow |
| `q_flow_layers` | 6 | Masked autoregressive layers in the q flow |
| `batch_size` | 2048 | Training mini-batch size |
| `num_samples` | 4 | MC samples per batch element in ELBO |
| `FITS_PATH` | *(local path)* | Template FITS catalogue |
| `PRIOR_FLOW_PATH` | `prior_flow_xy.eqx` | Saved prior flow weights |
| `Q_FLOW_PATH` | `q_flow_xy.eqx` | Saved q flow weights |

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
