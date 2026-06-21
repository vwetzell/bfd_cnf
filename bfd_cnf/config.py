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
prior_last_nn_width = 32
prior_last_nn_depth = 4
prior_sigmax_nn_width = 32
prior_sigmax_nn_depth = 4
prior_flow_layers = 10

import math as _math

# log_scale = 0.5 * log det(C_X) for the *centroid* noise covariance C_X = σ_xy² · I.
# The template catalogue's centroid noise is the FITS header SIG_XY = 391.4, so the
# physical centroid scale is 2*log(391.4) ≈ 11.94 (≈ the σ_xy = 400 used here).
prior_sigmax_log_scale_mean: float = 2.0 * _math.log(400.0)  # ≈ 11.98 (log_scale_n=0 at reference)
prior_sigmax_log_scale_std: float = 2.0  # maps log_scale_range (10.5, 13.0) to ~[-0.74, 0.51]

# Standardised-coordinate offset of the size ratio, c1 = mean[1]/std[1] for
# z1 = (Mr/Mf - mean[1])/std[1].  Used by SigmaXCouplingLayer's locked size
# transform z1' = κ·z1 + c1·(κ-1)  (≡ scaling the un-centred Mr/Mf by κ), which
# turns the multiplicative size knob into the physical size *mean* shift.
# Provenance: data/raw2standard_stats.npz (must match the standardiser the flow
# is trained with).  Recompute if the standardiser changes.
prior_size_loc_c1: float = 5.784213542938232

q_nn_width = 32
q_nn_depth = 2
q_flow_layers = 4

min_scale = 1e-2
max_scale = 100.0

batch_size = 1024
num_samples = 4

# Number of gradient steps fused into a single ``jax.lax.scan`` dispatch during
# training.  The flows are many tiny kernels per step, so the training loop is
# dispatch/bubble-bound: fusing steps lets XLA run them back-to-back with no
# per-step host round-trip (≈1.7x faster, GPU SM 97%→100%, identical math).
# Set to 1 to fall back to the eager per-step loop (needed for jax_debug_nans,
# which cannot localise a NaN inside a scan).
train_chunk_size = 50

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
# Optional top-tail percentile clip on nda to bound per-batch gradient variance
# (nda spans ~282x p5→p95 with rare >p99 outliers).  None = no clip; 99.9 trims
# only the extreme tail with negligible effect on the bulk weighting.
nda_clip_percentile: float | None = 99.9

# ---------------------------------------------------------------------------
# Σ_X conditioning / training parameters
# ---------------------------------------------------------------------------
# Number of Σ_X conditions sampled per gradient step; losses are averaged.
n_sx_train: int = 8
# Range of log_scale = 0.5*log det(C_X) (centroid noise) sampled uniformly each step.
# Derived from the grid TARGETS' per-object centroid covariance C_X — the odd-moment
# covariance built from each target's even cov via models.flows.even_cov_to_CX (NOT the
# [M+, Mx] ellipticity sub-block, a different spin-2 quantity).  Across both ±shear grids
# the corrected C_X gives log_scale p0.1/p50/p99.5 ≈ 10.83/11.37/12.67, so (10.5, 13.0)
# covers ~99.7% of targets with margin AND contains the isotropic per-tier SIG_XY
# corner reference (≈11.94).  Normalisation (prior_sigmax_log_scale_mean/std = 11.98/2.0)
# is kept so log_scale_n=0 sits at the SIG_XY corner reference; the new range maps to
# log_scale_n ≈ [-0.74, +0.51].
# (Was (11,15): tuned to the WRONG [M+,Mx]-block C_X — median ≈13.27 — which left the
# true target bulk near 10.9-11 below the trained range and wasted capacity on 13-15.)
log_scale_range: tuple[float, float] = (10.5, 13.0)
# Maximum centroid-covariance ellipticity magnitude |e| sampled for Σ_X conditioning.
# Sized to the bulk of the real target C_X anisotropy (|e| median ≈0.012, p99 ≈0.041
# across both ±grids via even_cov_to_CX) rather than the full tail, so training capacity
# concentrates near the dipole magnitude targets actually have.  NOTE: ~1% of targets
# exceed this (rare max ≈0.27) and their e1,e2 conditioning mildly extrapolates at
# inference; widen if those high-|e| targets show miscalibration.
e_max: float = 0.05

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
FITS_PATH = os.path.join(_EXT_DIR, "tmpl_t04_joined.fits")
# Deep-field summary template library: one row per galaxy (1.37M), with no
# sub-pixel-shifted copies.  Used as the observed-template source for diagnostic
# corner plots so the shifted replicas in tmpl_t04_joined don't inflate density.
SUMMARY_FITS_PATH = os.path.join(_EXT_DIR, "summary_templates_new.fits")
# New template set, pre-joined into one lean training table by
# bfd_cnf.build_training_table (columns already in trainer-native shapes:
# moments(4), cov(4,4), dm_dg(4,2), d2m_dg2(4,2,2), centroid(2), nda, id).
TRAIN_FITS_PATH = os.environ.get(
    "BFD_TRAIN_FITS", os.path.join(DATA_DIR, "templates_train.fits")
)
# Training template-set selector (see bfd_cnf.data.load_training_dataset):
#   True  → the new pre-joined set (TRAIN_FITS_PATH via load_training_table)
#   False → the legacy tmpl_t04 set (FITS_PATH via load_data)
# Plot/inference scripts keep calling load_data directly so their raw2standard
# still matches the currently-saved flow weights.
train_on_new_templates: bool = True
GRID_P_PATH = os.path.join(_EXT_DIR, "merged_masked_bfd_grid_p.npy")
GRID_M_PATH = os.path.join(_EXT_DIR, "merged_masked_bfd_grid_m.npy")

# Saved flow weights live in the repo's flows/ directory.
PRIOR_FLOW_PATH = os.path.join(FLOWS_DIR, "prior_flow_xy.eqx")
Q_FLOW_PATH = os.path.join(FLOWS_DIR, "q_flow_xy.eqx")

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
