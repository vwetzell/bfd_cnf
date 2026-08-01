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
prior_last_nn_width = 32  # shear conditioner (ExplicitPolyLast); was widened to 128 on 2026-07-11, reverted 2026-07-30
prior_last_nn_depth = 4
prior_sigmax_nn_width = 32
prior_sigmax_nn_depth = 4
prior_flow_layers = 10

# Shear-conditioning layer kind:
#   "poly"   -> ExplicitPolyLast: free polynomial-in-g affine coupling (the current
#              w128 .eqx flows were trained with this).
#   "taylor" -> ShearTaylorLast: the structural alternative — an additive Taylor-in-g
#              displacement, identity at g=0, whose 1st/2nd-order coefficients ARE the
#              moments' shear response (readable via .shear_derivs, cleanly Sobolev-
#              supervisable).
# The kind is a STATIC part of the flow structure, so a saved flow only deserialises
# with the kind it was trained with (like prior_last_nn_width above): switching this
# requires retraining and pointing PRIOR_FLOW_PATH at a matching .eqx.
# DEFAULT (matches the best-performing run, tag nll_taylor_spin2_4Msub32): a poly
# checkpoint at the default PRIOR_FLOW_PATH will fail to deserialise under this default
# (self-guarding — different param shapes) until PRIOR_FLOW_PATH points at a taylor .eqx;
# pass --shear-layer poly (+ --from-scratch or a matching poly checkpoint) for the old kind.
shear_layer_kind = "taylor"

# Where the conditional shear + Σ_X layers sit in the prior flow.
#   False -> base-adjacent (historical order the w128 .eqx flows were trained with):
#            the unconditional bulk is data-adjacent, so the data-space shear response
#            is the shear layer's coefficient threaded through the bulk decode Jacobian.
#   True  -> data-adjacent (default): generative stack base -> bulk -> Σ_X -> shear -> data,
#            so the shear layer's A IS the data-space dM/dg directly (cleanly Sobolev-
#            supervisable, decoupled from the bulk tail Jacobian). See EarlyChain.
# DANGER: this only reorders composition — it does NOT change the leaf set, so a flow
# trained with the other value deserialises WITHOUT error but silently mismatches (wrong
# order applied to the trained weights). It is NOT self-guarding like shear_layer_kind
# (which changes param shapes and fails loudly). Load a pre-flip .eqx with
# load_prior_flow(conditional_first=False) until a data-adjacent flow is retrained.
prior_conditional_first = True

# Own-|e| conditioning for the taylor shear layer (ShearTaylorLast). False (default) =
# the masked M1 dipole sees only (flux, size); True adds a second-stage M1 correction
# conditioned on the final M2, giving the shear response an own-ellipticity dependence
# (closed-form invertible, det=1) to fix the high-|e| tail under-response. STATIC flow
# structure (adds net_m1_e); a saved flow only deserialises with the value it trained with.
# Only meaningful with shear_layer_kind="taylor".
prior_shear_own_e = False

# Give the spin-2 SECOND-order shear coeff B its own coeff net instead of sharing the
# A trunk. A shared trunk is captured by the first-order NLL+sob1 gradient and starves
# the sob2 (2nd-order) supervision, pinning B at ~0; a dedicated B trunk lets sob2 train
# it. STATIC flow structure (adds net_m1_B/net_m2_B); a saved flow only deserialises with
# the value it trained with. Only meaningful with shear_layer_kind="taylor".
# NOW THE DEFAULT: pre-split-flip taylor .eqx must be loaded with shear_split_ab=False.
prior_shear_split_ab = True

# Condition the SPIN-2 (M1,M2) shear coefficients on the rotation-invariant |e|^2 =
# M1^2+M2^2, so the shift depends on the galaxy's OWN ellipticity — the ONLY way to
# represent the M1 second-order shear response (which is ~100% own-|e|: R^2=0 from
# flux,size, the masked layer's legal inputs). Turns the (M1,M2) block into a joint
# coupling with a real (non-unit) log-det + iterative inverse; equivariance kept because
# |e|^2 is a spin-0 invariant. STATIC structure. Only meaningful with taylor.
# DEFAULT (needed for 2nd-order fidelity); old taylor .eqx load with shear_spin2_owne=False.
prior_shear_spin2_owne = True

# Give the SPIN-0 (flux, size) shear coefficients the same own-|e| treatment
# spin2_owne/own_e already give M1,M2. net_flux/net_size are otherwise structurally
# blind to the galaxy's own ellipticity (masked order [flux,size,M1,M2] puts M1,M2
# strictly AFTER) -- this forces their g1*g2 cross-term to ~0 and their |g|^2
# curvature to a flux-only (or, for net_flux, fully global) compromise value,
# independent of how well-trained the flow is. Diagnosed 2026-08-01 on
# prior_flow_4Msub32sig0p3_w32_spin2default_narrowsx_elbo_ft.eqx: this structural
# gap produced a smooth +0.19-to--0.06 `m` gradient across the whole (flux,size)
# plane (see memory shear-taylor-flux-size-blind-to-orientation). STATIC flow
# structure (adds net_flux_e/net_size_e); a saved flow only deserialises with the
# value it trained with. Only meaningful with shear_layer_kind="taylor".
# DEFAULT (fixes the above); pre-fix taylor .eqx must be loaded with
# shear_flux_size_owne=False.
# 2026-08-01 (R12 follow-up): net_flux_e/net_size_e's own-|e| correction was
# initially built from an orientation-blind input (log1p|e|^2 only), which STILL
# cannot represent a nonzero B12 -- see the _owne_B_from_invariants docstring in
# bijections.py for the equivariant construction that replaced it. Any .eqx
# trained before this fix (including prior_flow_4Msub32sig0p3_w32_fsowne_narrowsx_
# elbo_ft.eqx) has net_flux_e/net_size_e with the OLD (3,)-output shape and needs
# a from-scratch retrain -- it will not deserialise into the new (2,)-output
# structure.
prior_shear_flux_size_owne = True

# Sobolev shear-derivative training weights (0 = off). When >0, the loss adds
# lambda * MSE between the flow's own moment shear-response (d m / d g and
# d^2 m / d g^2 of the decode map, by autodiff) and the template truth (a quadratic fit
# of the sheared moments over the g-grid). Works with either shear_layer_kind and in
# both the NLL and ELBO paths. DEFAULT (matches nll_taylor_spin2_4Msub32); pass
# --sobolev-g1 0 --sobolev-g2 0 to disable.
sobolev_g1_weight = 1.0   # first-order  d m / d g   matching
sobolev_g2_weight = 1.0   # second-order d^2 m / d g^2 matching

# Weight for the shear-coeff off-template penalty (0 = off): trains
# ShearTaylorLast's coefficient nets toward zero output on synthetic
# (flux, size, |e|) probes beyond where real templates live, instead of leaving
# their off-manifold extrapolation arbitrary. See ShearTaylorLast.coeff_ood_penalty.
shear_coeff_ood_weight = 0.01

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
# Range of log_scale = 0.5*log det(C_X) (centroid noise) sampled uniformly each step.
# 2026-07-30: measured directly off imsims's data/targets_plus_4M_fix.fits (unpacking
# the stored covariance via bfd.moment.MomentCovariance) — EVERY target has the exact
# same, isotropic C_X: log_scale = 12.04249382, e1 = e2 = 0.0 (imsims uses one fixed
# global target-noise sigma, not per-object depth variation, unlike the grid-catalog
# calibration this range/e_max used to be tuned to). Compressed to +/-0.1 around that
# single point so training capacity concentrates where inference actually lands, while
# KEEPING the stencil/marginalisation structure (not collapsing to a point) so the
# Σ_X-conditioned layer still generalises to future target populations with genuine
# per-object log_scale/e spread. Widen back out before training on such a population.
log_scale_range: tuple[float, float] = (11.94249382, 12.14249382)
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
