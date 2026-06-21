# bfd_cnf — Project Overview Skill

Summarize the purpose, architecture, and key workflows of this repository for any contributor or collaborator who needs to get oriented quickly.

---

## Purpose

**bfd_cnf** implements learned normalizing-flow priors and posteriors for weak-lensing shear estimation inside the **BFD (Bayesian Fourier Domain)** framework. BFD operates in galaxy Fourier-domain moment space (flux Mf, size Mr, ellipticity M1/M2) rather than pixel space, marginalizing over the intrinsic galaxy population to produce per-object PQR statistics that are summed to yield a maximum-likelihood shear estimate.

The project replaces the original analytic moment-space priors with **trained conditional normalizing flows**, enabling more flexible modeling and improved multiplicative-bias performance — the primary metric in weak-lensing cosmology.

---

## Repository Layout

```
bfd_cnf/                   # repo root (artifacts + package)
├── bfd_cnf/               # Python package — run as `python -m bfd_cnf.run`
│   ├── config.py          # All hyperparameters, file paths, fixed matrices
│   ├── data.py            # FITS loading, quality cuts, density reweighting, coordinate transforms
│   ├── models/
│   │   ├── flows.py       # High-level flow construction and ELBO loss factory
│   │   └── bijections.py  # Custom flowjax bijection classes (standardize, bounded affine, etc.)
│   ├── training.py        # Training loop, serialization (load_or_train, save/load_models)
│   ├── inference.py       # RQMC integration, PQR assembly, shear-derivative computation
│   ├── statistics.py      # pqr2g shear estimator, multiplicative-bias bootstrap
│   ├── viz.py             # Diagnostic corner plots, SNR/moment/shear/bias figures
│   ├── run.py             # End-to-end pipeline orchestration (main entry point)
│   └── *.py               # CLI helpers: train_from_scratch, continue_train,
│                          #   integrate_grid, plot_corner, plot_shear_derivs, …
├── flows/                 # Saved flow weights (*.eqx) + timestamped backups
├── plots/                 # Generated figures (*.png)
├── data/                  # PQR grid outputs + cached standardiser stats (*.npz)
├── logs/                  # Training / integration run logs
└── attic/                 # Archived notebooks + one-off debug scripts
```

---

## Key Concepts

### Moment Space
Raw galaxy moments `[Mf, Mr, M1, M2]` are transformed through:
1. **Raw → Transformed**: `[log10(Mf), Mr/Mf, M1/Mr, M2/Mr]`
2. **Transformed → Standardized**: subtract mean, divide by std (per-coordinate)

Covariances are propagated through each step using Jacobians plus second-order Hessian corrections.

### The Two Flows

| Flow | Symbol | Conditions on | Role |
|------|--------|---------------|------|
| Prior | p(z \| g, Σ_X) | shear g, PSF noise Σ_X | Models the true galaxy moment distribution |
| Q flow | q(z \| y, Σ_y, g) | noisy measurement y, its cov Σ_y, shear g | Variational posterior for RQMC importance sampling |

Both flows are trained jointly via an **ELBO** that includes the measurement likelihood, shear Taylor-expansion terms, and per-object centroid weights.

### PQR Statistics
For each galaxy, RQMC integration computes:
- **P**: Probability (marginalization normalizer)
- **Q**: `∫ (∂ log p / ∂g) · p(m) dm` — first shear derivative
- **R**: `∫ (∂² log p / ∂g²) · p(m) dm` — second shear derivative

Summing over all galaxies: `ĝ = R⁻¹ · Q` (maximum-likelihood shear).

Multiplicative bias is estimated from ±shear simulation catalogues:
`m = (g₊ - g₋) / Δg − 1`

---

## Data Flow (run.py)

```
FITS catalogue
    └─► data.load_data()
            ├── SNR/ellipticity quality cuts
            ├── 2-D density histogram weighting (flat coverage)
            └── raw2standard bijection
                    └─► training.load_or_train()
                                ├── build prior flow + Q flow
                                ├── ELBO training (20K steps, AdamW)
                                └── save/load .eqx weights
                                        └─► inference.rqmc_pqr_grid()
                                                    ├── Halton RQMC sampling
                                                    ├── L-BFGS proposal optimization
                                                    └── PQR assembly
                                                            └─► statistics.bootstrap_total_mult_bias()
                                                                        └─► viz.* diagnostic plots
```

---

## Running the Pipeline

```bash
# Full end-to-end run (data → training → PQR → shear → plots)
# Run from the repo root so the package import resolves.
python -m bfd_cnf.run
```

Saved flow weights (`flows/prior_flow_xy.eqx`, `flows/q_flow_xy.eqx`) are reloaded automatically on subsequent runs, skipping re-training.

---

## Configuration (config.py)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `N_LAYERS_PRIOR` | 10 | Coupling layers in prior flow |
| `N_LAYERS_Q` | 6 | Autoregressive layers in Q flow |
| `N_MC` | 4 | MC samples per ELBO gradient step |
| `N_SX_TRAIN` | 8 | Σ_X conditions per gradient step |
| `LOG_SCALE_RANGE` | (11, 13) | log₁₀ PSF noise scale range |
| `E_MAX` | 0.2 | Max PSF ellipticity |
| Training steps | 20K | AdamW, lr=1e-4, wd=1e-5, grad clip 0.5 |

Key file paths are set in `config.py`: `FITS_PATH`, `PRIOR_FLOW_PATH`, `Q_FLOW_PATH`, `GRID_P_PATH`, `GRID_M_PATH`.

---

## Tech Stack

| Package | Role |
|---------|------|
| `jax` / `jaxlib` | JIT-compiled array ops, autodiff |
| `equinox` | JAX neural networks, pytree serialization |
| `flowjax` | Normalizing flow primitives |
| `optax` | AdamW optimizer, gradient clipping |
| `paramax` | Frozen-parameter wrappers (`non_trainable`) |
| `jaxopt` | L-BFGS for RQMC proposal optimization |
| `astropy` | FITS I/O |
| `bfd` | BFD library (moments, derivatives, PQR) |
| `corner` / `matplotlib` | Diagnostic plots |

---

## Key Files for Common Tasks

- **Change hyperparameters**: `config.py`
- **Modify flow architecture**: `models/flows.py` (`build_flows`)
- **Add/change bijections**: `models/bijections.py`
- **Adjust data cuts or weighting**: `data.py` (`load_data`, `filter_data`)
- **Debug RQMC integration**: `inference.py` (`rqmc_integrate_pqr_jax`)
- **Add diagnostic plots**: `viz.py`
- **Inspect shear estimator**: `statistics.py` (`pqr2g`, `pqr2multbias`)
