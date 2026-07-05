# bfd_cnf

**Conditional normalizing-flow priors for BFD weak-lensing shear estimation.**

`bfd_cnf` replaces the template summing in the **BFD** (Bayesian
Fourier Domain; Bernstein & Armstrong 2014, [arXiv:1304.1843](https://arxiv.org/abs/1304.1843);
Bernstein et al. 2016, [arXiv:1508.05655](https://arxiv.org/abs/1508.05655)) weak-lensing
shear estimator with two **trained conditional normalizing flows** (JAX / Equinox / FlowJax),
enabling more robust galaxy-population modeling and improved multiplicative-bias performance.

For the full architecture, math, data formats, and script-by-script reference, see the
**[project wiki](../../wiki)**. This README covers orientation and quick start only.

---

## How BFD shear estimation works here

BFD never forward-models pixels. It works entirely in Fourier-domain **galaxy moment space**
— flux, size, and ellipticity moments `[Mf, Mr, M1, M2]` — and estimates shear by marginalizing
over the intrinsic (unsheared) galaxy population using Taylor-expanded shear derivatives to
propagate the effect of a small shear `g` through that population. The unsheared galaxy
population is derived from high signal to noise template galaxies measured in deep fields.

Two flows are trained jointly:

| Flow                             | Conditions on                                                   | Role                                                                       |
| -------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Prior** `p_θ(z \| g, Σ_X)`     | shear `g`, centroid-noise covariance `Σ_X`                      | Learned replacement for discrete template summation in BFD                 |
| **Q flow** `q_φ(z \| y, Σ_y, g)` | noisy template measurement `y`, its covariance `Σ_y`, shear `g` | Variational posterior, used only during training (for the IWAE-style ELBO) |

At inference time, only the trained **prior** flow is needed. For each target galaxy it is
integrated (via RQMC + a Laplace-mode proposal) against the target's Gaussian measurement
kernel to produce per-object **PQR** statistics:

- **P** — probability (marginal likelihood of the observed moments)
- **Q** — first shear derivative, `∂P/∂g`
- **R** — second shear derivative, `∂²P/∂g²`

`P`/`Q`/`R` are derivatives of the probability `P`, **not** of `log P` — Q and R can be
negative/indefinite, so there is no "log Q" or "log R". Per catalogue object these are
converted to the log-marginal quantities that actually drive shear, `Q_tot = Q/P` and
`R_tot = (Q⊗Q)/P² − R/P`, and summed over the catalogue; the maximum-likelihood shear is
then `ĝ = R_tot⁻¹·Q_tot`. Multiplicative bias is measured from `±g` simulation pairs as
`m = (ĝ₊ − ĝ₋)/Δg − 1`. See `statistics.pqr2g` for the implementation.

See the wiki's **[Architecture](../../wiki/Architecture)** and
**[Inference and Shear Estimation](../../wiki/Inference-and-Shear-Estimation)** pages for the
full derivation and implementation.

---

## Repository layout

```
bfd_cnf/                       # repo root
├── bfd_cnf/                   # Python package — run as `python -m bfd_cnf.<script>`
│   ├── config.py              # Hyperparameters, file paths, fixed matrices
│   ├── data.py                # FITS loading, quality cuts, nda-weighted sampling, standardization
│   ├── models/
│   │   ├── flows.py           # Flow construction (build_flows), ELBO/NLL loss factories
│   │   └── bijections.py      # Custom flowjax bijections (equivariant layers, shear layer, Σ_X layer)
│   ├── training.py            # Core training loop + checkpoint save/load
│   ├── pretrain_prior.py      # Prior-only NLL warm start
│   ├── converge_train.py      # Production trainer: blocked ELBO training with convergence-gated stopping
│   ├── continue_train.py      # Resume an existing checkpoint for a fixed number of steps
│   ├── train_from_scratch.py  # Debug entry point (jax_debug_nans forced on)
│   ├── inference.py           # RQMC PQR integration, shear-derivative assembly
│   ├── statistics.py          # pqr2g shear estimator, multiplicative-bias bootstrap
│   ├── run.py                 # Illustrative end-to-end pipeline (data → train → PQR → bias → plots)
│   ├── build_training_table.py / make_tmpl_t04_joined.py  # Template FITS builders
│   └── plot_*.py / analyze_*.py / validate_*.py / diag_*.py  # Diagnostic & validation CLIs
├── flows/                     # Saved flow weights (*.eqx) + standardiser sidecars (*.stats.npz) + backups
├── data/                      # Training FITS table, cached PQR grid results (*.npz) — gitignored
├── plots/                     # Generated figures — gitignored
├── logs/                      # Training / integration run logs — gitignored
├── dev/                       # Research/debug scripts: bias-source diagnostics, sweeps, smoke tests
├── tests/                     # Assert-based self-checks (bijection correctness, checkpoint sidecar)
└── .claude/skills/bfd-shear-flow-integration/  # Reference derivation + verified estimator for the
                                                  # inference-time integration problem (P/Q/R, RQMC, Laplace proposal)
```

Run package scripts from the **repo root** so the package import resolves, e.g.
`python -m bfd_cnf.converge_train`.

---

## Installation

No `requirements.txt`/`pyproject.toml` yet — install directly:

```bash
pip install jax jaxlib equinox flowjax optax paramax jaxopt fitsio astropy numpy matplotlib corner
```

> **`bfd`** (the underlying BFD moments/PQR library — `TemplateTable`, `MomentCovariance`,
> `momentcalc`, etc.) is not on PyPI; install it from its own source repository.

Large external inputs (template FITS tables, ± shear simulation grids) live outside the repo.
Point at them with environment variables so paths stay portable across machines:

```bash
export BFD_DATA_DIR=/path/to/BFD_cNF          # dir containing merged_masked_bfd_grid_{p,m}.npy
export BFD_TRAIN_FITS=/path/to/templates_train.fits
```

Both default to a local workstation path baked into `config.py` if unset.

---

## Quick start

### Train (or continue training) the flows

```bash
# One-time prior-only warm start (NLL loss, no Q flow yet)
python -m bfd_cnf.pretrain_prior

# Production joint-ELBO training with convergence-gated stopping
python -m bfd_cnf.converge_train --loss elbo

# Resume an existing checkpoint for a fixed number of extra steps
# (e.g. after widening the Σ_X training range in config.py)
python -m bfd_cnf.continue_train --steps 5000
```

Trained weights are saved to `flows/prior_flow_xy.eqx` / `flows/q_flow_xy.eqx`, each with a
`.stats.npz` sidecar recording the standardiser `(mean, std)` used to train it — required for
correct reloading, since the Σ_X layer's `c1` constant is derived from those stats at build time.

### Run PQR integration + shear/bias estimation on a trained flow

```bash
python -m bfd_cnf.integrate_grid       # RQMC PQR over the ± shear simulation grids
python -m bfd_cnf.plot_bias_table      # per-arm and total (m, c1, c2) bias vs. analytic BFD
```

### Full illustrative pipeline (data → train → integrate → bias → plots)

```bash
python -m bfd_cnf.run
```

`run.py` is a reference/demo script, not the production training path — see the wiki's
**[Training](../../wiki/Training)** page for how `pretrain_prior.py` → `converge_train.py` fit
together in practice.

---

## Documentation

Deep-dive documentation lives in the **[wiki](../../wiki)**:

- **[Architecture](../../wiki/Architecture)** — flow layer-by-layer math (equivariant layers, shear layer, Σ_X/PSF-centroid layer)
- **[Training](../../wiki/Training)** — loss functions, the `nda`-weighted sampling scheme, training entry points
- **[Data Pipeline](../../wiki/Data-Pipeline)** — FITS table formats, quality cuts, template building
- **[Inference and Shear Estimation](../../wiki/Inference-and-Shear-Estimation)** — RQMC integration, PQR assembly, bias estimation
- **[Configuration](../../wiki/Configuration)** — full `config.py` settings reference
- **[Flow Checkpoints](../../wiki/Flow-Checkpoints)** — how to read the `flows/*.eqx` naming conventions
- **[Scripts Reference](../../wiki/Scripts-Reference)** — one-line purpose of every CLI/diagnostic script in `bfd_cnf/`, `dev/`, and `tests/`

## License

MIT — see [LICENSE](LICENSE).
