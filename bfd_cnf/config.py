"""
config.py
=========
JAX environment setup, PRNG key initialization, model hyperparameters,
file path constants, fixed matrices, and index arrays used throughout
the bfd_cnf package.
"""

from __future__ import annotations

import os

# Platform allocator reserving 85% of GPU memory. The environment can still
# override either var before importing this module.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import jax

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import jax.random as jr

assert jnp.array([1.0]).dtype == jnp.float32, "64-bit floats are not disabled!"

# ---------------------------------------------------------------------------
# PRNG key
# ---------------------------------------------------------------------------
key = jr.key(18061998)

# ---------------------------------------------------------------------------
# Hyperparameters / model architecture constants
# ---------------------------------------------------------------------------
prior_early_nn_width = 32
prior_early_nn_depth = 2
prior_last_nn_width = 32  # shear conditioner (ShearTaylorLast); was widened to 128 on 2026-07-11, reverted 2026-07-30
prior_last_nn_depth = 4
prior_sigmax_nn_width = 32
prior_sigmax_nn_depth = 4
prior_flow_layers = 10

# The prior's shear-conditioning layer is ShearTaylorLast (models/bijections.py): an
# additive Taylor-in-g moment displacement, data-adjacent (see EarlyChain), with an
# equivariant joint (M1,M2) coupling conditioned on the galaxy's own |e|^2 and an
# equivariant own-|e| correction on the (flux,size) second-order coefficients (see
# ShearTaylorLast's docstring and _owne_B_from_invariants). This is the only supported
# architecture -- earlier alternatives (a free-polynomial "poly" shear layer,
# base-adjacent layer ordering, a single-M1-only own-|e| correction, a shared A/B
# coefficient trunk) were removed 2026-08-01 once this configuration won on every axis;
# see memory shear-taylor-flux-size-blind-to-orientation for why the flux/size own-|e|
# correction needed to be equivariant. STATIC flow structure: a saved flow only
# deserialises against this exact architecture.

# Sobolev shear-derivative training weights (0 = off). When >0, the loss adds
# lambda * MSE between the flow's own moment shear-response (d m / d g and
# d^2 m / d g^2 of the decode map, by autodiff) and the template truth (a quadratic fit
# of the sheared moments over the g-grid). Works in both the NLL and ELBO paths.
# DEFAULT (matches nll_taylor_spin2_4Msub32); pass --sobolev-g1 0 --sobolev-g2 0 to
# disable.
sobolev_g1_weight = 1.0   # first-order  d m / d g   matching
sobolev_g2_weight = 1.0   # second-order d^2 m / d g^2 matching

import math as _math

# log_scale = 0.5 * log det(C_X) for the *centroid* noise covariance C_X = σ_xy² · I.
# The template catalogue's centroid noise is the FITS header SIG_XY = 391.4, so the
# physical centroid scale is 2*log(391.4) ≈ 11.94 (≈ the σ_xy = 400 used here).
prior_sigmax_log_scale_mean: float = 2.0 * _math.log(400.0)  # ≈ 11.98 (log_scale_n=0 at reference)
prior_sigmax_log_scale_std: float = 2.0  # maps log_scale_range (10.5, 13.0) to ~[-0.74, 0.51]

q_nn_width = 32
q_nn_depth = 2
q_flow_layers = 4

min_scale = 1e-2
max_scale = 100.0

batch_size = 512
num_samples = 8

# Number of gradient steps fused into a single ``jax.lax.scan`` dispatch during
# training.  The flows are many tiny kernels per step, so the training loop is
# dispatch/bubble-bound: fusing steps lets XLA run them back-to-back with no
# per-step host round-trip (≈1.7x faster, GPU SM 97%→100%, identical math).
# Set to 1 to fall back to the eager per-step loop (needed for jax_debug_nans,
# which cannot localise a NaN inside a scan).
train_chunk_size = 100

# ---------------------------------------------------------------------------
# Selection / flux threshold
# ---------------------------------------------------------------------------
# Minimum M_f for the target selection function (raw moment units).
# Must match the selection applied to the target catalogue AND the fluxMin
# passed to bfd.TemplateTable.
target_flux_min: float = 800.0

# Flux floor for the TRAINING-template population.  Kept equal to
# target_flux_min so the templates the flow trains on and the selection
# threshold applied in the loss / at integration share one cut.
template_flux_min: float = 800.0

# Flux ceiling for the TRAINING-template population (2026-08-06): quality_cut_mask
# previously had no upper flux bound. A vanishingly small number of templates
# (<0.03%) with Mf up into the hundreds of thousands to millions get astronomically
# amplified weight under detj=1/4(Mr^2-M1^2-M2^2) (size scales with flux), which
# collapses the effective sample size of the nda*detj-weighted training proposal
# from millions to ~3000 -- dominated by templates the flow never learned to
# extrapolate to. See memory m-tilt/offset investigation, 2026-08-06.
template_flux_max: float = 100000.0

# ---------------------------------------------------------------------------
# Template area/density (nda) weighting
# ---------------------------------------------------------------------------
# Weight each template copy by the BFD `nda = sky_density * da` factor (stored in
# the FITS `weight` column) so the flow learns the *nda-weighted* template prior,
# matching the traditional BFD integration `p = Σ nda·kernel`
# (bfd/probabilities_jax.py:236).  `nda` is the (sky-density × centroid-grid dX
# measure) prefactor; it multiplies the existing centroid Gaussian L(X|C_X) and is
# C_X-independent, so it does NOT double-count the Σ_X machinery or distort the
# learned C_X-response.  Set False to reproduce the pre-change (unweighted) loss
# exactly (control arm for A/B comparison).
use_nda_weight: bool = True

# ---------------------------------------------------------------------------
# Σ_X conditioning / training parameters
# ---------------------------------------------------------------------------
# log_scale = 0.5*log det(C_X) (centroid noise) sampled uniformly each step over
# [log_scale_median - log_scale_window, log_scale_median + log_scale_window].
# 2026-07-30: log_scale_median measured directly off imsims's data/targets_plus_4M_fix.fits
# (unpacking the stored covariance via bfd.moment.MomentCovariance) — EVERY target has the
# exact same, isotropic C_X: log_scale = 12.04249382, e1 = e2 = 0.0 (imsims uses one fixed
# global target-noise sigma, not per-object depth variation, unlike the grid-catalog
# calibration this range/e_max used to be tuned to). log_scale_window compressed to 0.1
# around that single point so training capacity concentrates where inference actually
# lands, while KEEPING the stencil/marginalisation structure (not collapsing to a point)
# so the Σ_X-conditioned layer still generalises to future target populations with genuine
# per-object log_scale/e spread. Widen back out before training on such a population.
# converge_train.py's --sigmax-layer none sets log_scale_window (and e_max) to exactly 0
# instead, since a model with no SigmaX layer has nothing to generalise FOR -- see there.
log_scale_median: float = 12.04249382
log_scale_window: float = 0.1
log_scale_range: tuple[float, float] = (
    log_scale_median - log_scale_window, log_scale_median + log_scale_window
)
# Maximum centroid-covariance ellipticity magnitude |e| sampled for Σ_X conditioning.
# 2026-07-30: compressed alongside log_scale_range for the same reason — imsims targets
# are exactly isotropic (e1=e2=0) today, so 0.05 was pure extrapolation padding for this
# dataset. 0.005 keeps a small nonzero ring (structure preserved for future non-isotropic
# populations) without diluting training signal onto e-directions never seen at inference.
e_max: float = 0.005

# ---------------------------------------------------------------------------
# File path constants
# ---------------------------------------------------------------------------
# Repo-root-relative artifact directories.  ``config.py`` lives in the
# ``bfd_cnf/`` package, so the repo root is the parent of the package dir.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
FLOWS_DIR = os.path.join(_REPO_ROOT, "flows")
PLOTS_DIR = os.path.join(_REPO_ROOT, "plots")
DATA_DIR = os.path.join(_REPO_ROOT, "data")

# External input data (not in the repo). Override the directory on a new
# machine (e.g. HPC) with BFD_DATA_DIR; falls back to the local workstation path.
_EXT_DIR = os.environ.get("BFD_DATA_DIR", "/home/vwetzell/Documents/BFD_cNF")
# Training table built by dev/make_template_file.py -> dev/subsample.py, already
# in trainer-native shapes: moments(4), cov(4,4), dm_dg(4,2), d2m_dg2(4,2,2),
# centroid(2), nda, id.
TRAIN_FITS_PATH = os.environ.get(
    "BFD_TRAIN_FITS", os.path.join(DATA_DIR, "templates_train.fits")
)
GRID_P_PATH = os.path.join(_EXT_DIR, "merged_masked_bfd_grid_p.npy")
GRID_M_PATH = os.path.join(_EXT_DIR, "merged_masked_bfd_grid_m.npy")

# Saved flow weights live in the repo's flows/ directory.  Defaults track the
# current w128 architecture (prior_last_nn_width=128); the older width-32
# prior_flow_xy.eqx no longer matches build_flows and fails to deserialise.
PRIOR_FLOW_PATH = os.path.join(FLOWS_DIR, "prior_flow_xy_w128.eqx")
Q_FLOW_PATH = os.path.join(FLOWS_DIR, "q_flow_xy_w128.eqx")

# ---------------------------------------------------------------------------
# Fixed matrices
# ---------------------------------------------------------------------------
# BFD B matrix – defined in raw moment space (Mf, Mr, M+, Mx)
B_RAW_JNP = jnp.diag(jnp.array([0.0, 0.25, -0.25, -0.25]))

# Reference noise covariance (diagonal, in raw moment units)
Sigma0 = jnp.diag(jnp.array([220.0, 1100.0, 750.0, 750.0]) ** 2)

# Shear scale used for conditioning
g_scale = jnp.array([0.01, 0.01])


# ---------------------------------------------------------------------------
# Lower-triangular index arrays (shared across modules)
# ---------------------------------------------------------------------------
r_idx, c_idx = jnp.tril_indices(4)
off_mask = r_idx != c_idx
r_off = r_idx[off_mask]
c_off = c_idx[off_mask]
