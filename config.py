"""
config.py
=========
JAX environment setup, PRNG key initialization, model hyperparameters,
file path constants, fixed matrices, and index arrays used throughout
the bfd_cnf package.
"""

from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
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

# log_scale = 0.5 * log det(Σ_X); for a circular PSF with σ_xy = 400: 2*log(400)
prior_sigmax_log_scale_mean: float = 2.0 * _math.log(400.0)  # ≈ 11.98
prior_sigmax_log_scale_std: float = 1.0  # covers ~e^±1.5 variation in σ_xy

q_nn_width = 64
q_nn_depth = 4
q_flow_layers = 6

min_scale = 1e-2
max_scale = 100.0

batch_size = 1024
num_samples = 4

# ---------------------------------------------------------------------------
# Σ_X conditioning / training parameters
# ---------------------------------------------------------------------------
# Number of Σ_X conditions sampled per gradient step; losses are averaged.
n_sx_train: int = 8
# Range of 0.5 * log det(Σ_X) for the uniform prior over PSF noise scale.
log_scale_range: tuple[float, float] = (11.0, 13.0)
# Maximum PSF ellipticity magnitude |e| for Σ_X conditioning.
e_max: float = 0.2

# ---------------------------------------------------------------------------
# File path constants
# ---------------------------------------------------------------------------
FITS_PATH = "/home/vwetzell/Documents/BFD_cNF/tmpl_t04_joined.fits"
PRIOR_FLOW_PATH = "/home/vwetzell/gitrepos/bfd_cnf/prior_flow_xy.eqx"
Q_FLOW_PATH = "/home/vwetzell/gitrepos/bfd_cnf/q_flow_xy.eqx"
GRID_P_PATH = "/home/vwetzell/Documents/BFD_cNF/merged_masked_bfd_grid_p.npy"
GRID_M_PATH = "/home/vwetzell/Documents/BFD_cNF/merged_masked_bfd_grid_m.npy"

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
