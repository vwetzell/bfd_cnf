"""
config.py
=========
JAX environment setup, PRNG key initialization, model hyperparameters,
file path constants, fixed matrices, and index arrays used throughout
the bfd_cnf package.
"""

from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"

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
# Config cell

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
prior_sigmax_log_scale_mean: float = 2.0 * _math.log(400.0)  # ≈ 11.98
prior_sigmax_log_scale_std: float = 1.0  # normalises the (11, 13) range to ~±1σ

q_nn_width = 64
q_nn_depth = 4
q_flow_layers = 6

min_scale = 1e-2
max_scale = 100.0

batch_size = 1024
num_samples = 4

# ---------------------------------------------------------------------------
# Selection / flux threshold
# ---------------------------------------------------------------------------
# Minimum M_f for the target selection function (raw moment units).
# Must match the selection applied to the target catalogue AND the fluxMin
# passed to bfd.TemplateTable.  The data.py quality cut uses the same value.
target_flux_min: float = 1500.0

# ---------------------------------------------------------------------------
# Σ_X conditioning / training parameters
# ---------------------------------------------------------------------------
# Number of Σ_X conditions sampled per gradient step; losses are averaged.
n_sx_train: int = 8
# Range of log_scale = 0.5*log det(C_X) (centroid noise) sampled uniformly each step.
# Spans the physical centroid/template scale (SIG_XY ⇒ ≈11.94, the corner-plot
# reference) up through the grid TARGETS' per-object C_X scales (median ≈13.27,
# p1-p99 ≈12.75-14.24 derived from each target covariance at inference) so the prior is
# trained in-range for both the marginal corner plot AND the PQR shear integration.
# Normalisation (prior_sigmax_log_scale_mean/std=11.98/1.0) is kept so log_scale_n=0 at
# the corner reference and continuation does not disturb the calibrated marginal.
# (Was (11,13): missed the target bulk → 88% out-of-range at inference.  Was (12,17):
# too broad — weak L(X|C_X) weighting → shift-contaminated marginal.)
log_scale_range: tuple[float, float] = (11.0, 15.0)
# Maximum PSF ellipticity magnitude |e| for Σ_X conditioning.
# Must be >= the maximum |e| seen in target PSF covariances at inference.
e_max: float = 0.2

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

# External input data (not in the repo).
FITS_PATH = "/home/vwetzell/Documents/BFD_cNF/tmpl_t04_joined.fits"
# Deep-field summary template library: one row per galaxy (1.37M), with no
# sub-pixel-shifted copies.  Used as the observed-template source for diagnostic
# corner plots so the shifted replicas in tmpl_t04_joined don't inflate density.
SUMMARY_FITS_PATH = "/home/vwetzell/Documents/BFD_cNF/summary_templates_new.fits"
GRID_P_PATH = "/home/vwetzell/Documents/BFD_cNF/merged_masked_bfd_grid_p.npy"
GRID_M_PATH = "/home/vwetzell/Documents/BFD_cNF/merged_masked_bfd_grid_m.npy"

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
def _make_lower_tri_index(D: int) -> tuple[jax.Array, jax.Array]:
    """Return row and column indices of the lower-triangular elements of a D×D matrix.

    Parameters
    ----------
    D : int
        Dimension of the square matrix.

    Returns
    -------
    tuple of jax.Array
        ``(row_indices, col_indices)`` each of length ``D*(D+1)//2``.
    """
    mask = jnp.tril(jnp.ones((D, D), dtype=bool))
    return jnp.where(mask)


r_idx, c_idx = _make_lower_tri_index(4)
off_mask = r_idx != c_idx
r_off = r_idx[off_mask]
c_off = c_idx[off_mask]
