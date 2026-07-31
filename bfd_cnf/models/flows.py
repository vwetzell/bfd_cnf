"""
models/flows.py
===============
High-level flow construction (``build_flows``), conditioning-feature helpers,
shear utility, Cholesky/covariance helpers, and the ELBO loss factory
(``make_elbo_loss``).

Standardisation statistics (mean_log_diag, std_log_diag, mean_off, std_off)
are computed inside ``build_flows`` (or can be passed in explicitly) from the
training dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
from jax.scipy.special import logsumexp
import equinox as eqx
from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import MultivariateNormal
from paramax import non_trainable

from .bijections import (
    BoundedAffine,
    ShearTaylorLast,
    new_masked_autoregressive_flow,
)
from ..config import (
    prior_early_nn_width,
    prior_early_nn_depth,
    prior_last_nn_width,
    prior_last_nn_depth,
    prior_sigmax_nn_width,
    prior_sigmax_nn_depth,
    prior_sigmax_log_scale_mean,
    prior_sigmax_log_scale_std,
    prior_flow_layers,
    q_nn_width,
    q_nn_depth,
    q_flow_layers,
    min_scale,
    max_scale,
    r_off,
    c_off,
    g_scale,
    target_flux_min,
    e_max,
    shear_layer_kind as _shear_layer_kind,
    prior_conditional_first as _prior_conditional_first,
    prior_shear_own_e as _prior_shear_own_e,
    prior_shear_split_ab as _prior_shear_split_ab,
    prior_shear_spin2_owne as _prior_shear_spin2_owne,
    sobolev_g1_weight as _sobolev_g1_weight,
    sobolev_g2_weight as _sobolev_g2_weight,
    shear_coeff_ood_weight as _shear_coeff_ood_weight,
)

# ---------------------------------------------------------------------------
# Covariance helpers
# ---------------------------------------------------------------------------


def batch_cholesky_of_sym(sigmas: jax.Array, jitter: float = 1e-6) -> jax.Array:
    """Compute the Cholesky decomposition of a batch of symmetric matrices.

    Symmetrises the input by averaging with its transpose and adds a small
    diagonal jitter before calling ``jnp.linalg.cholesky``.

    Parameters
    ----------
    sigmas : jax.Array, shape (..., D, D)
        Batch of (approximately) symmetric positive-definite matrices.
    jitter : float, optional
        Small diagonal regularisation.  Default is ``1e-6``.

    Returns
    -------
    jax.Array, shape (..., D, D)
        Lower-triangular Cholesky factor ``L`` such that ``L @ L.T ≈ sigma``.
    """
    sigmas = (sigmas + sigmas.swapaxes(-1, -2)) * 0.5
    D = sigmas.shape[-1]
    sigmas = sigmas + jitter * jnp.eye(D, dtype=sigmas.dtype)
    return jnp.linalg.cholesky(sigmas)


def cov2corr(Sigma: jax.Array, eps: float = 1e-12) -> jax.Array:
    """Convert a batch of covariance matrices to correlation matrices.

    Parameters
    ----------
    Sigma : jax.Array, shape (..., D, D)
        Batch of covariance matrices.
    eps : float, optional
        Small value added to the diagonal before taking the square root to
        avoid division by zero.  Default is ``1e-12``.

    Returns
    -------
    jax.Array, shape (..., D, D)
        Corresponding correlation matrices with unit diagonal.
    """
    diag = jnp.sqrt(jnp.diagonal(Sigma, axis1=-2, axis2=-1) + eps)
    inv_diag = 1.0 / diag
    corr = Sigma * inv_diag[..., :, None] * inv_diag[..., None, :]
    return corr


# ---------------------------------------------------------------------------
# Conditioning features
# ---------------------------------------------------------------------------


def cond_features_from_y_sigma(
    y: jax.Array,
    Sigma: jax.Array,
    g: jax.Array,
    eps: float = 1e-12,
    mean_log_diag: jax.Array | None = None,
    std_log_diag: jax.Array | None = None,
    mean_off: jax.Array | None = None,
    std_off: jax.Array | None = None,
    g_scale: jax.Array | None = g_scale,
) -> tuple[jax.Array, jax.Array]:
    """Build conditioning feature vectors from standardised moments and covariances.

    Concatenates the standardised moments ``y``, the (optionally whitened)
    log-diagonal elements of the Cholesky factor, the off-diagonal
    correlation coefficients, and the (optionally scaled) shear vector.

    Parameters
    ----------
    y : jax.Array, shape (N, D)
        Standardised moment vectors.
    Sigma : jax.Array, shape (N, D, D)
        Standardised covariance matrices.
    g : jax.Array, shape (N, 2)
        Shear vectors.
    eps : float, optional
        Small value for numerical stability.  Default is ``1e-12``.
    mean_log_diag : jax.Array or None, optional
        Mean of ``log(diag(L))`` for whitening; if ``None`` no whitening is
        applied.
    std_log_diag : jax.Array or None, optional
        Std of ``log(diag(L))`` for whitening.
    mean_off : jax.Array or None, optional
        Mean of off-diagonal correlation elements for whitening.
    std_off : jax.Array or None, optional
        Std of off-diagonal correlation elements for whitening.
    g_scale : jax.Array or None, optional
        Scale applied to ``g`` before concatenation.  Defaults to
        ``config.g_scale``.

    Returns
    -------
    cond : jax.Array, shape (N, cond_dim)
        Conditioning feature vectors.
    L : jax.Array, shape (N, D, D)
        Cholesky factors of ``Sigma``.
    """
    L = batch_cholesky_of_sym(Sigma)
    diag = jnp.diagonal(L, axis1=-2, axis2=-1)
    log_diag = jnp.log(diag + eps)
    corr = cov2corr(Sigma, eps=eps)
    off = corr[..., r_off, c_off]

    if mean_log_diag is not None:
        log_diag = (log_diag - mean_log_diag) / (std_log_diag + 1e-8)
    if mean_off is not None:
        off = (off - mean_off) / (std_off + 1e-8)
    if g_scale is not None:
        g = g / g_scale

    return jnp.concatenate([y, off, log_diag, g], axis=-1), L


# ---------------------------------------------------------------------------
# Shear utility
# ---------------------------------------------------------------------------


def shear(x: jax.Array, g: jax.Array, dg: jax.Array, d2g: jax.Array) -> jax.Array:
    """Apply shear deformation to a batch of moments via second-order Taylor expansion.

    Computes ``x_sheared[b, g_i, :] = x[b] + dg[b] · g[g_i] + 0.5 * d2g[b] · g[g_i]⊗g[g_i]``.

    Parameters
    ----------
    x : jax.Array, shape (B, D)
        Batch of unsheared moment vectors.
    g : jax.Array, shape (1, G, 2) or broadcastable
        Grid of ``G`` shear vectors to apply.
    dg : jax.Array, shape (B, D, 2)
        First shear derivatives of the moments.
    d2g : jax.Array, shape (B, D, 2, 2)
        Second shear derivatives of the moments.

    Returns
    -------
    jax.Array, shape (B, G, D)
        Shear-deformed moments for each (batch element, shear value) pair.
    """
    g2d = jnp.reshape(g, (-1, g.shape[-1]))
    first_order = jnp.einsum("ndk,gk->ngd", dg, g2d)
    second_order = 0.5 * jnp.einsum("gk,ndkl,gl->ngd", g2d, d2g, g2d)
    return x[:, None, :] + first_order + second_order


# ---------------------------------------------------------------------------
# Gaussian likelihood helper
# ---------------------------------------------------------------------------


def log_gaussian_full(y: jax.Array, mu: jax.Array, L: jax.Array) -> jax.Array:
    """Compute the full-covariance Gaussian log-likelihood for a batch.

    Evaluates ``log N(y | mu_s, L_s L_s^T)`` for all combinations of data
    points ``y`` and sample means ``mu`` with corresponding Cholesky factors
    ``L``.

    Parameters
    ----------
    y : jax.Array, shape (BG, D)
        Observed data vectors.
    mu : jax.Array, shape (S, BG, D)
        Sample mean vectors (e.g. flow samples).
    L : jax.Array, shape (BG, D, D)
        Lower-triangular Cholesky factors of the covariance matrices.

    Returns
    -------
    jax.Array, shape (S, BG)
        Log-likelihood values for each (sample, data) pair.
    """
    BG, D = y.shape
    S = mu.shape[0]
    resid = y[None, :, :] - mu
    sol = jax.scipy.linalg.solve_triangular(L, resid.transpose(1, 2, 0), lower=True)
    sol = jax.scipy.linalg.solve_triangular(L.transpose(0, 2, 1), sol, lower=False)
    sol = sol.transpose(2, 0, 1)
    quad = jnp.sum(resid * sol, axis=-1)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    norm = D * jnp.log(2 * jnp.pi)
    return -0.5 * (quad + logdet[None, :] + norm)


# ---------------------------------------------------------------------------
# Σ_X helpers
# ---------------------------------------------------------------------------


def _moment_jacobian_det(m_raw: jax.Array) -> jax.Array:
    """Compute the BFD moment Jacobian determinant J(M_G).

    ``J(M_G) = (Mr² − M+² − M×²) / 4``

    Parameters
    ----------
    m_raw : jax.Array, shape (4,)
        Raw template moments ``[Mf, Mr, M+, M×]``.

    Returns
    -------
    jax.Array, scalar
        Jacobian determinant, clipped to a minimum of ``1e-10``.
    """
    Mr, Mp, Mx = m_raw[1], m_raw[2], m_raw[3]
    return jnp.maximum((Mr**2 - Mp**2 - Mx**2) / 4.0, 1e-10)


def _sample_sx_conds(
    key: jax.Array, log_scale_range: tuple[float, float], e_max: float
) -> jax.Array:
    """Build centering-bias (Σ_X) condition vectors on a fixed e-stencil.

    Each condition vector is ``[log_scale, e1, e2]`` where

    * ``log_scale = 0.5 * log det(Σ_X)`` (spin-0 noise level) — **random**,
    * ``(e1, e2)`` (spin-2 centering-bias ellipticity) — a **fixed stencil**: one centre
      ``e=0`` plus two 8-direction rings at ``|e| = e_max/2`` and ``|e| = e_max``
      (17 points total).

    Why a stencil (mirrors the fixed ``g``-ring in the ELBO loss): with a random
    ``φ`` draw the spin-2 dipole/quadrupole signal lands at a different azimuth
    every step, so the ``net_dipquad`` (D, c) gradient is azimuthally smeared and
    high-variance — the SigmaX centring response is small and was being starved by
    that noise.  Hitting the *same* e-directions every step gives a consistent,
    low-variance finite-difference of the e-response.  Two non-zero magnitudes
    (``e_max/2`` and ``e_max``, not just ``e_max``) sample the e-response *curvature*,
    not only its slope.  ``log_scale`` stays random: it's the smooth spin-0 ``T``
    magnitude the net interpolates well, and one random draw per stencil point keeps
    its coverage.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key (only ``log_scale`` is random now).
    log_scale_range : tuple of float
        ``(min, max)`` for the uniform distribution over ``log_scale``.
    e_max : float
        Centering-bias ellipticity magnitude of the outer stencil ring (inner ring = e_max/2).

    Returns
    -------
    jax.Array, shape (17, 3)
        Condition vectors ``[log_scale, e1, e2]``.
    """
    # Fixed e-stencil: centre e=0 plus 8-direction rings at |e| = e_max/2 and e_max.
    # Same 8 azimuths every step (low-variance finite-difference); two non-zero
    # magnitudes sample the e-response curvature, not just its slope.  The full 2φ
    # circle spans ±e1, ±e2 (Z₂ e→−e covered by antipodal pairs).
    n_dir = 8
    e_mags = jnp.array([0.5 * e_max, e_max])  # inner + outer ring
    two_phi = 2.0 * jnp.pi * jnp.arange(n_dir) / n_dir  # (8,)
    ring_e1 = (e_mags[:, None] * jnp.cos(two_phi)[None, :]).reshape(-1)  # (16,)
    ring_e2 = (e_mags[:, None] * jnp.sin(two_phi)[None, :]).reshape(-1)
    e1 = jnp.concatenate([jnp.zeros(1), ring_e1])  # (17,)
    e2 = jnp.concatenate([jnp.zeros(1), ring_e2])
    # One random log_scale per stencil point.
    log_scale = jr.uniform(
        key, shape=(e1.shape[0],), minval=log_scale_range[0], maxval=log_scale_range[1]
    )
    return jnp.stack([log_scale, e1, e2], axis=-1)  # (17, 3)


def _sx_cond_to_CX(sx_cond: jax.Array) -> jax.Array:
    """Reconstruct the 2×2 noise covariance C_X from ``[log_scale, e1, e2]``.

    The parameterisation follows :func:`_sample_sx_conds`:

    * ``log_scale = 0.5 * log det(C_X)``  (spin-0 noise level)
    * ``(e1, e2)`` = ellipticity of C_X

    C_X is reconstructed as

        C_X = (T/2) * [[1+e1, e2], [e2, 1−e1]]

    with  ``T/2 = exp(log_scale) / sqrt(1 − |e|²)``.

    Parameters
    ----------
    sx_cond : jax.Array, shape (3,)
        Condition vector ``[log_scale, e1, e2]``.

    Returns
    -------
    jax.Array, shape (2, 2)
        Positive-definite noise covariance matrix.
    """
    log_scale, e1, e2 = sx_cond[0], sx_cond[1], sx_cond[2]
    e_mag_sq = e1**2 + e2**2
    half_T = jnp.exp(log_scale) / jnp.sqrt(jnp.maximum(1.0 - e_mag_sq, 1e-8))
    return half_T * jnp.array([[1.0 + e1, e2], [e2, 1.0 - e1]])


def even_cov_to_CX(even_cov: jax.Array) -> jax.Array:
    """Centroid (odd-moment) noise covariance ``C_X`` from the even covariance.

    BFD does not store the odd-parity covariance separately: it is a fixed
    linear function of the even-moment covariance's flux row (see
    ``bfd.moment.MomentCovariance.__init__``).  With the even moments ordered
    ``[Mf, Mr, M+, Mx]`` (indices 0..3)::

        C_X[X, X] = ½ (Cov[Mf, Mr] + Cov[Mf, M+])
        C_X[Y, Y] = ½ (Cov[Mf, Mr] − Cov[Mf, M+])
        C_X[X, Y] = ½  Cov[Mf, Mx]

    This is the per-*target* centroid measurement covariance that conditions the
    prior flow (feed it through :func:`cx_to_sx_cond`).  It is NOT the
    ``[M+, Mx]`` ellipticity sub-block, which is a different (spin-2) quantity.

    Parameters
    ----------
    even_cov : jax.Array, shape (..., 4, 4) or (..., 5, 5)
        Even-moment covariance in raw moment units.  Only the flux-row entries
        ``[0, 1:4]`` are read, so either the 4×4 or full 5×5 block works.

    Returns
    -------
    jax.Array, shape (..., 2, 2)
        Symmetric (positive-definite for valid inputs) centroid covariance
        ``C_X``.
    """
    c_mr = even_cov[..., 0, 1]  # Cov[Mf, Mr]
    c_m1 = even_cov[..., 0, 2]  # Cov[Mf, M+]
    c_m2 = even_cov[..., 0, 3]  # Cov[Mf, Mx]
    cxx = 0.5 * (c_mr + c_m1)
    cyy = 0.5 * (c_mr - c_m1)
    cxy = 0.5 * c_m2
    row0 = jnp.stack([cxx, cxy], axis=-1)
    row1 = jnp.stack([cxy, cyy], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def cx_to_sx_cond(CX: jax.Array) -> jax.Array:
    """Convert a 2×2 centroid noise covariance C_X to ``[log_scale, e1, e2]``.

    Inverse of :func:`_sx_cond_to_CX`.  The parameterisation is::

        C_X = (T/2) * [[1+e1, e2], [e2, 1-e1]]

    so that::

        log_scale = 0.5 * log det(C_X)
        e1 = (C_X[0,0] - C_X[1,1]) / (C_X[0,0] + C_X[1,1])
        e2 = 2 * C_X[0,1]           / (C_X[0,0] + C_X[1,1])

    Parameters
    ----------
    CX : jax.Array, shape (2, 2)
        Centroid noise covariance (positive definite, symmetric).

    Returns
    -------
    jax.Array, shape (3,)
        ``[log_scale, e1, e2]`` condition vector.
    """
    _, logdet = jnp.linalg.slogdet(CX)
    log_scale = 0.5 * logdet
    trace = CX[0, 0] + CX[1, 1]
    e1 = (CX[0, 0] - CX[1, 1]) / jnp.maximum(trace, 1e-30)
    e2 = 2.0 * CX[0, 1]        / jnp.maximum(trace, 1e-30)
    return jnp.array([log_scale, e1, e2])


def _log_L_X(X_G: jax.Array, sx_cond: jax.Array) -> jax.Array:
    """Evaluate ``log L(X_G | C_X)`` — the Gaussian centroid weight from Eq. 36.

    Parameters
    ----------
    X_G : jax.Array, shape (2,)
        First-order Fourier moments of a single template copy.
    sx_cond : jax.Array, shape (3,)
        Condition vector ``[log_scale, e1, e2]``.

    Returns
    -------
    jax.Array, scalar
        Log of the zero-mean Gaussian density  N(X_G; 0, C_X).
    """
    log_scale = sx_cond[0]
    CX = _sx_cond_to_CX(sx_cond)

    # 2×2 explicit inverse: adj(A)/det
    det = jnp.exp(2.0 * log_scale)
    CX_inv = jnp.array([[CX[1, 1], -CX[0, 1]], [-CX[1, 0], CX[0, 0]]]) / det

    mahal = X_G @ CX_inv @ X_G
    return -0.5 * mahal - log_scale - jnp.log(2.0 * jnp.pi)


def _batch_log_L_X(X_batch: jax.Array, sx_cond: jax.Array) -> jax.Array:
    """Vectorised version of :func:`_log_L_X` over a batch of templates.

    Parameters
    ----------
    X_batch : jax.Array, shape (B, 2)
    sx_cond : jax.Array, shape (3,)

    Returns
    -------
    jax.Array, shape (B,)
    """
    return jax.vmap(lambda x: _log_L_X(x, sx_cond))(X_batch)


def _normalised_importance_weights(log_w: jax.Array) -> jax.Array:
    """Self-normalise log-weights and return linear-scale weights.

    Parameters
    ----------
    log_w : jax.Array, shape (B,)
        Unnormalised log-weights.

    Returns
    -------
    w : jax.Array, shape (B,)
        Normalised weights summing to ``B``  (so the mean is 1).
    ess : jax.Array, scalar
        Effective sample size ``1 / sum(w_tilde^2)`` where
        ``w_tilde = w / sum(w)``.
    """
    log_w_norm = log_w - logsumexp(log_w)  # log w̃ with Σ w̃ = 1
    w_tilde = jnp.exp(log_w_norm)  # (B,)  sums to 1
    B = log_w.shape[0]
    w = w_tilde * B  # sums to B — mean 1
    ess = 1.0 / jnp.sum(w_tilde**2)
    return w, ess


# ---------------------------------------------------------------------------
# Flux-selection correction (shared by make_elbo_loss and make_nll_loss)
# ---------------------------------------------------------------------------
# Floor on log P_sel so the selection correction -log P_sel stays bounded.
# σ_f = sqrt(var_mf) is tiny for high-SNR templates, so a sample drawn below
# the flux threshold makes logsf((threshold-mf)/σ_f) astronomically negative and
# -log P_sel explode (seen as a -2.9e9 loss spike at init).  The floor only
# engages for such pathological samples; legitimate near-threshold corrections
# are O(1-10) and unaffected.  P_sel >= exp(-30) ≈ 9e-14.
_LOG_PSEL_FLOOR = -30.0


def _log_p_select_given_x_raw(mf_true, var_mf, threshold):
    sigma = jnp.sqrt(jnp.maximum(var_mf, 0.0) + 1e-12)
    return jnp.maximum(jstats.norm.logsf((threshold - mf_true) / sigma), _LOG_PSEL_FLOOR)


# ---------------------------------------------------------------------------
# Sobolev shear-derivative term (shared by make_elbo_loss and make_nll_loss)
# ---------------------------------------------------------------------------
# "Sobolev training" = supervise the flow's DERIVATIVES w.r.t. a condition, not just
# its density.  Here we match the flow's moment shear-response (d m / d g and
# d^2 m / d g^2 at g=0) to the template truth.  The truth is a quadratic fit of the
# sheared-standardised template moments over the g-grid; the flow's response is the
# g-derivative of its decode map at fixed latent (autodiff, coordinate-agnostic — so
# it works for either shear layer kind).  Both are in standardised moment space and
# evaluated at the same unsheared template moments, so the match is pointwise.


def _sobolev_pinv(g2d: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Least-squares pseudo-inverse for a g-grid quadratic fit, and the fit scale.

    Fits ``f(g) = c0 + c1 u1 + c2 u2 + c3 u1^2 + c4 u2^2 + c5 u1 u2`` in the RESCALED
    shear ``u = g / s`` (``s`` = outer g-grid radius).  The rescale is essential: the
    raw grid has ``|g| ~ 1e-2``, so the ``g^2`` design columns are ``~1e-4`` against a
    unit constant column — a condition number that makes the un-scaled ``pinv`` lose
    all precision.  In ``u`` the columns are O(1) and the fit is exact for a quadratic.

    Returns ``(pinv, s)`` with ``pinv`` shape ``(6, G)``.  Fixed (depends only on the
    grid), so computed once in the loss factory.
    """
    s = jnp.sqrt(jnp.max(jnp.sum(g2d**2, axis=1)))  # outer ring radius
    u1, u2 = g2d[:, 0] / s, g2d[:, 1] / s
    Phi = jnp.stack(
        [jnp.ones_like(u1), u1, u2, u1**2, u2**2, u1 * u2], axis=1
    )  # (G, 6)
    return jnp.linalg.pinv(Phi), s  # (6, G), scalar


def _sobolev_target(y_std_grid: jax.Array, pinv: jax.Array, scale: jax.Array):
    """Template moment shear-response from a quadratic fit over the g-grid.

    ``y_std_grid`` : (B, G, D) sheared-standardised moments; ``pinv, scale`` from
    :func:`_sobolev_pinv`.  Coefficients are fit in ``u = g / scale`` then converted
    back to ``g`` derivatives: ``d/dg = (1/s) d/du``, ``d^2/dg^2 = (1/s^2) d^2/du^2``.
    Returns ``A`` (B, D, 2) = ``d m / d g`` and ``B`` (B, D, 2, 2) = ``d^2 m / d g^2``.

    The g=0 baseline is subtracted before the fit so we fit the O(g) *displacement*,
    not the moment plus its O(1) baseline: in float32 the g^2 curvature (~1e-4·B) would
    otherwise be swamped by that baseline and the second-order target come out as noise.
    """
    dy = y_std_grid - y_std_grid[:, 0:1, :]  # displacement from g=0 (float32-robust)
    coeffs = jnp.einsum("cg,bgd->bdc", pinv, dy)  # (B, D, 6) in u-space
    A = coeffs[..., 1:3] / scale  # (B, D, 2)
    s2 = scale**2
    c3, c4, c5 = coeffs[..., 3], coeffs[..., 4], coeffs[..., 5]
    B = jnp.stack(
        [
            jnp.stack([2.0 * c3, c5], axis=-1),
            jnp.stack([c5, 2.0 * c4], axis=-1),
        ],
        axis=-2,
    ) / s2  # (B, D, 2, 2)
    return A, B


def _flow_shear_derivs(prior_flow, m0_std: jax.Array, sx_ref: jax.Array):
    """Flow's own moment shear-response at each unsheared std moment, by autodiff.

    For each moment ``m0`` (at g=0): encode to the latent at ``[g=0, sx_ref]``, then
    differentiate the decode map w.r.t. ``g`` at fixed latent.  Since the encode uses
    g=0, ``decode(latent, g=0) == m0``, so ``A = d(decode)/dg`` and
    ``B = d^2(decode)/dg^2`` are the moment shear-response at ``m0``.  Coordinate-
    agnostic: it uses only the flow's decode/encode, not any layer internals.

    ``m0_std`` : (B, D).  ``sx_ref`` : (3,) reference ``[log_scale, e1, e2]``.
    Returns ``A`` (B, D, 2), ``B`` (B, D, 2, 2).
    """
    bij = prior_flow.bijection
    g0 = jnp.zeros(2)

    def per_example(m0):
        cond0 = jnp.concatenate([g0, sx_ref])
        z = bij.inverse(m0, cond0)  # encode data -> latent at g=0

        def decode(g):
            return bij.transform(z, jnp.concatenate([g, sx_ref]))  # latent -> data

        A = jax.jacfwd(decode)(g0)  # (D, 2)
        B = jax.jacfwd(jax.jacfwd(decode))(g0)  # (D, 2, 2)
        return A, B

    return jax.vmap(per_example)(m0_std)


def _find_shear_taylor_layer(prior_flow):
    """Return the flow's ``ShearTaylorLast`` instance, or ``None`` if it uses a
    different (e.g. ``poly``) shear layer."""
    layers = [
        n
        for n in jax.tree_util.tree_leaves(
            prior_flow, is_leaf=lambda x: isinstance(x, ShearTaylorLast)
        )
        if isinstance(n, ShearTaylorLast)
    ]
    return layers[0] if layers else None


def _shear_response(prior_flow, m0, sx_ref):
    """Flow's data-space moment shear-response ``(A, B)`` at each unsheared moment.

    The response is C_X-independent: ``_flow_shear_derivs`` encodes and decodes at the
    *same* Σ_X, so the SigmaX layer cancels in the round-trip (a single ``sx_ref``
    suffices — no averaging over C_X draws).

    Fast path: when the flow uses a **data-adjacent** ``ShearTaylorLast``
    (``conditional_first=True``), the response is that layer's analytic generative
    derivs (:meth:`ShearTaylorLast.shear_derivs_generative`) — no whole-flow autodiff.
    Otherwise (poly last layer, or base-adjacent) fall back to the autodiff round-trip.
    The kind/order are static pytree structure, so this branch resolves at trace time.
    """
    layer = _find_shear_taylor_layer(prior_flow)
    conditional_first = getattr(prior_flow.bijection.bijection, "conditional_first", False)
    # The analytic generative formula (A_gen=-A, B_gen=-B+corr) needs only shift(x,0)=0,
    # which the plain Taylor layer, spin2_owne, AND own_e all satisfy — shear_derivs
    # folds own_e's net_m1_e correction into A[2],B[2] (with u3 stop-gradiented so the
    # correction formula's ∂A/∂x Jacobian doesn't mistreat the fixed u3 as the unknown
    # m3; see ShearTaylorLast.shear_derivs_generative's docstring). spin2's shear_derivs
    # feeds it the joint (M1,M2) A,B and it reproduces the true inverse-map response to
    # 2e-7 (the non-unit det is irrelevant: det affects log_prob, not the moment
    # response m(g)).
    if layer is not None and conditional_first:
        return jax.vmap(layer.shear_derivs_generative)(m0)
    return _flow_shear_derivs(prior_flow, m0, sx_ref)


def _coeff_ood_penalty(prior_flow, key):
    """Mean squared shear-coeff magnitude at synthetic off-template probes.

    Trains ``ShearTaylorLast``'s coefficient nets toward zero output for
    (flux, size, |e|) combinations no real template ever visits (see
    :meth:`ShearTaylorLast.coeff_ood_penalty`). ``0.0`` for flows using a
    different shear layer (e.g. ``poly``).
    """
    layer = _find_shear_taylor_layer(prior_flow)
    return layer.coeff_ood_penalty(key) if layer is not None else jnp.zeros(())


def _sobolev_row_weights(
    X_b: jax.Array, sx_conds: jax.Array, max_weight_mult: float = 20.0
) -> jax.Array:
    """Per-template weight matching the density loss's L(X|C_X) SNIS correction.

    The density loss estimates ``E_{nda·detj·L(X|C_X)}[...]`` via templates drawn
    ∝ nda·detj (the shared batch draw — already identical between the density and
    Sobolev terms, no extra factor needed here) followed by a self-normalised
    softmax-over-batch correction for the residual ``L(X|C_X)`` factor, averaged
    over the C_X stencil (``loss = mean(losses_sx)``).  A_tgt/B_tgt/A_flow/B_flow
    (the Sobolev integrand) are C_X-independent, so for a c-independent integrand
    ``mean_c[softmax_i(logL(X_i|c))]`` would be *exactly* the weight the density
    loss's own combination collapses to (mean and sum commute).

    That uncapped identity caused a real training divergence (2026-07-30, prior
    ``_sobrw`` fine-tune, block 7 of 10, 824/1000 non-finite losses): a
    self-normalised softmax can concentrate almost entirely on one outlier
    template (classic SNIS effective-sample-size collapse — the same failure
    mode ``diagnostics/reweight_sensitivity_report.md`` hit and fixed by capping
    importance ratios to ``[1/20, 20]``), and unlike the (self-regularising)
    log-density term, the Sobolev MSE has no ceiling, so one bad draw produces a
    huge, badly-directed update.  Capping each stencil point's per-row weight at
    ``max_weight_mult / B`` before renormalising bounds that concentration (a
    small bias for a lot of stability) — mirroring the existing precedent. The
    main density loss's own ``w_bg`` is untouched (uncapped, proven stable
    across many prior runs); this cap applies ONLY to the Sobolev term.
    Evaluated at g=0 (``X_b`` is already the g=0 centroid moment, since
    ``shear`` at ``g=[0,0]`` is the identity).
    """
    B = X_b.shape[0]
    logL = jax.vmap(lambda c: _batch_log_L_X(X_b, c))(sx_conds)  # (n_sx, B)
    w_c = jax.nn.softmax(logL, axis=1)  # (n_sx, B), each row sums to 1
    w_c = jnp.minimum(w_c, max_weight_mult / B)
    w_c = w_c / jnp.sum(w_c, axis=1, keepdims=True)
    return jnp.mean(w_c, axis=0)  # (B,), sums to 1


def _sobolev_loss(prior_flow, y_std_grid, sob1_w, sob2_w, pinv, scale, sx_ref, row_w=None):
    """Sobolev penalty: MSE of (flow − template) 1st/2nd moment shear-response.

    Both the template target (A_tgt, B_tgt) and the flow's response are C_X-independent
    (see :func:`_shear_response`), so this is a single evaluation at ``sx_ref`` — no
    per-C_X averaging.

    ``row_w``, if given, is a (B,) population weight (summing to 1) applied across
    the batch axis instead of a plain mean — see :func:`_sobolev_row_weights`.  Pass
    ``None`` (plain mean) when the density loss itself has no C_X/L(X|C_X) term to
    match (e.g. ``use_sx=False``)."""
    A_tgt, B_tgt = _sobolev_target(y_std_grid, pinv, scale)  # (B,D,2), (B,D,2,2)
    m0 = y_std_grid[:, 0, :]  # (B, D) unsheared (g=0 is grid index 0)
    A_flow, B_flow = _shear_response(prior_flow, m0, sx_ref)
    if row_w is None:
        return sob1_w * jnp.mean((A_flow - A_tgt) ** 2) + sob2_w * jnp.mean(
            (B_flow - B_tgt) ** 2
        )
    a_err = jnp.mean((A_flow - A_tgt) ** 2, axis=tuple(range(1, A_flow.ndim)))
    b_err = jnp.mean((B_flow - B_tgt) ** 2, axis=tuple(range(1, B_flow.ndim)))
    return sob1_w * jnp.sum(row_w * a_err) + sob2_w * jnp.sum(row_w * b_err)


def warmstart_prior_B(prior_flow, raw2standard, moments, cov, dm_dg, d2m_dg2,
                      steps=800, lr=1e-2, batch=1024, seed=0):
    """Seed the flow's SECOND-order shear response B from the template truth.

    Minimises the sob2 term (``mean((B_flow − B_tgt)²)``) over the flow's params by
    Adam.  With a zero-initialised ShearTaylorLast the first-order coeff A (and thus the
    generative correction term) is 0, so the sob2 gradient w.r.t the A/bulk params
    vanishes at init and effectively ONLY the B coeff nets move — a direct regression of
    B onto the per-template target.  Run this on a fresh (``--from-scratch``) split_ab
    flow before the main NLL training so B starts near-correct instead of at zero (the
    zero it otherwise never leaves; see the sob2-B-untrained finding).

    Returns the warm-started ``prior_flow``.  Params: raw ``moments`` (N,4), ``cov``
    (N,4,4), ``dm_dg`` (N,6,2), ``d2m_dg2`` (N,6,2,2) — the trainer-native template
    arrays (even-4 slice is taken here)."""
    import optax
    from ..data import transform_dataset_to_standard

    g0 = jnp.array([[[0.0, 0.0]]])
    s2 = 1.0 / jnp.sqrt(2.0)
    ring = jnp.array([[0, 1], [s2, s2], [1, 0], [s2, -s2],
                      [0, -1], [-s2, -s2], [-1, 0], [-s2, s2]])[jnp.newaxis]
    g = jnp.concatenate([g0, 0.01 * ring, 0.02 * ring], axis=1)  # (1,G,2)
    G = g.shape[1]
    g2d = g.reshape(G, 2)
    pinv, scale = _sobolev_pinv(g2d)
    sx_ref = jnp.zeros(3)
    N = moments.shape[0]

    params, static = eqx.partition(prior_flow, eqx.is_inexact_array)

    # The batch is gathered OUTSIDE the jit and passed in as args — closing over the full
    # (millions-row) template arrays would inline them as multi-GB captured constants (OOM).
    def sob2(params, y_b, S_b, dg_b, d2g_b):
        flow = eqx.combine(params, static)
        B = y_b.shape[0]
        y_sheared = shear(y_b, g, dg_b, d2g_b)              # (B,G,4)
        y_flat = y_sheared.reshape(B * G, -1)
        S_flat = jnp.broadcast_to(S_b[:, None], (B, G, 4, 4)).reshape(B * G, 4, 4)
        y_std, _ = transform_dataset_to_standard(raw2standard, y_flat, S_flat)
        y_std_grid = y_std.reshape(B, G, -1)
        _, B_tgt = _sobolev_target(y_std_grid, pinv, scale)
        _, B_flow = _shear_response(flow, y_std_grid[:, 0, :], sx_ref)
        return jnp.mean((B_flow - B_tgt) ** 2)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step(params, opt_state, y_b, S_b, dg_b, d2g_b):
        loss, grads = eqx.filter_value_and_grad(sob2)(params, y_b, S_b, dg_b, d2g_b)
        updates, opt_state = opt.update(grads, opt_state, params)
        return eqx.apply_updates(params, updates), opt_state, loss

    key = jr.key(seed)
    bs = min(batch, N)
    print(f"Warm-start B: regressing 2nd-order response over {steps} steps (batch {bs})...")
    for i in range(steps):
        key, sk = jr.split(key)
        idx = jr.randint(sk, (bs,), 0, N)
        params, opt_state, loss = step(
            params, opt_state, moments[idx], cov[idx], dm_dg[idx, :4], d2m_dg2[idx, :4]
        )
        if i % 100 == 0 or i == steps - 1:
            print(f"  warmstart step {i:4d}  sob2={float(loss):.4f}")
    return eqx.combine(params, static)


def warmstart_prior_A(prior_flow, raw2standard, moments, cov, dm_dg, d2m_dg2,
                      steps=800, lr=1e-2, batch=1024, seed=0):
    """Seed the flow's FIRST-order shear response A from the template truth.

    Minimises the sob1 term (``mean((A_flow − A_tgt)²)``) over the flow's params by
    Adam.  ``A_gen = -A`` (see :func:`_shear_response`) is the derivative at ``g=0``
    of a shift that is (at least locally) linear-plus-quadratic in ``g``, so it depends
    only on the layer's own linear-in-g coefficient — never on the quadratic-in-g
    coefficient B or any bulk flow parameter, since ``d/dg[A·g + 1/2 B·g²]|_{g=0} = A``
    regardless of B.  So this is a direct, isolated regression of A onto the
    per-template target, independent of :func:`warmstart_prior_B` — the two touch
    disjoint parameters and may be run in either order.

    Run this on a fresh (``--from-scratch``) taylor flow before the main NLL/ELBO
    training so A starts near the template-derived truth instead of at its zero init —
    the same treatment :func:`warmstart_prior_B` already gives the second-order
    coefficient (see the sob2-B-untrained finding this mirrors for first order).

    Returns the warm-started ``prior_flow``.  Params: raw ``moments`` (N,4), ``cov``
    (N,4,4), ``dm_dg`` (N,6,2), ``d2m_dg2`` (N,6,2,2) — the trainer-native template
    arrays (even-4 slice is taken here)."""
    import optax
    from ..data import transform_dataset_to_standard

    g0 = jnp.array([[[0.0, 0.0]]])
    s2 = 1.0 / jnp.sqrt(2.0)
    ring = jnp.array([[0, 1], [s2, s2], [1, 0], [s2, -s2],
                      [0, -1], [-s2, -s2], [-1, 0], [-s2, s2]])[jnp.newaxis]
    g = jnp.concatenate([g0, 0.01 * ring, 0.02 * ring], axis=1)  # (1,G,2)
    G = g.shape[1]
    g2d = g.reshape(G, 2)
    pinv, scale = _sobolev_pinv(g2d)
    sx_ref = jnp.zeros(3)
    N = moments.shape[0]

    params, static = eqx.partition(prior_flow, eqx.is_inexact_array)

    # The batch is gathered OUTSIDE the jit and passed in as args — closing over the full
    # (millions-row) template arrays would inline them as multi-GB captured constants (OOM).
    def sob1(params, y_b, S_b, dg_b, d2g_b):
        flow = eqx.combine(params, static)
        B = y_b.shape[0]
        y_sheared = shear(y_b, g, dg_b, d2g_b)              # (B,G,4)
        y_flat = y_sheared.reshape(B * G, -1)
        S_flat = jnp.broadcast_to(S_b[:, None], (B, G, 4, 4)).reshape(B * G, 4, 4)
        y_std, _ = transform_dataset_to_standard(raw2standard, y_flat, S_flat)
        y_std_grid = y_std.reshape(B, G, -1)
        A_tgt, _ = _sobolev_target(y_std_grid, pinv, scale)
        A_flow, _ = _shear_response(flow, y_std_grid[:, 0, :], sx_ref)
        return jnp.mean((A_flow - A_tgt) ** 2)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step(params, opt_state, y_b, S_b, dg_b, d2g_b):
        loss, grads = eqx.filter_value_and_grad(sob1)(params, y_b, S_b, dg_b, d2g_b)
        updates, opt_state = opt.update(grads, opt_state, params)
        return eqx.apply_updates(params, updates), opt_state, loss

    key = jr.key(seed)
    bs = min(batch, N)
    print(f"Warm-start A: regressing 1st-order response over {steps} steps (batch {bs})...")
    for i in range(steps):
        key, sk = jr.split(key)
        idx = jr.randint(sk, (bs,), 0, N)
        params, opt_state, loss = step(
            params, opt_state, moments[idx], cov[idx], dm_dg[idx, :4], d2m_dg2[idx, :4]
        )
        if i % 100 == 0 or i == steps - 1:
            print(f"  warmstart step {i:4d}  sob1={float(loss):.4f}")
    return eqx.combine(params, static)


def warmstart_prior_AB(prior_flow, raw2standard, moments, cov, dm_dg, d2m_dg2,
                       sob1_w=1.0, sob2_w=1.0, steps=800, lr=1e-2, batch=1024, seed=0):
    """Jointly seed the flow's 1st- AND 2nd-order shear response (A, B) from template
    truth, in a single pass — the combination-safe alternative to running
    :func:`warmstart_prior_A` and :func:`warmstart_prior_B` back to back.

    Minimises the SAME combined loss used in main training (:func:`_sobolev_loss`:
    ``sob1_w·mean((A_flow−A_tgt)²) + sob2_w·mean((B_flow−B_tgt)²)``) over the flow's
    params by Adam.

    Why not just run A then B (or B then A)?  The generative 2nd-order response is
    ``B_gen = -B + correction(A, ∂A/∂x)`` (see
    :meth:`ShearTaylorLast.shear_derivs_generative`) — built entirely from this layer's
    own coefficient nets, so a sob2-ONLY step's gradient can flow back into A's coeff
    net too, once A ≠ 0.  :func:`warmstart_prior_B`'s "only B moves" guarantee relies on
    A being exactly 0 (true only at a fresh, from-scratch init); on an existing/resumed
    flow A is already nonzero, so a sob2-only step can nudge A away from A_tgt again
    right after (or before) an A-only step set it correctly — the two steps can fight
    each other. Minimising both terms simultaneously has no such issue: A_tgt is right
    there in the same loss, opposing any pull away from it via the B term. Safe to run
    on either a fresh (``--from-scratch``) OR an already-trained (resumed, via
    ``--prior-in``) flow — nothing about the argument depends on the checkpoint's
    current A/B values.

    Returns the warm-started ``prior_flow``.  Params: raw ``moments`` (N,4), ``cov``
    (N,4,4), ``dm_dg`` (N,6,2), ``d2m_dg2`` (N,6,2,2) — the trainer-native template
    arrays (even-4 slice is taken here)."""
    import optax
    from ..data import transform_dataset_to_standard

    g0 = jnp.array([[[0.0, 0.0]]])
    s2 = 1.0 / jnp.sqrt(2.0)
    ring = jnp.array([[0, 1], [s2, s2], [1, 0], [s2, -s2],
                      [0, -1], [-s2, -s2], [-1, 0], [-s2, s2]])[jnp.newaxis]
    g = jnp.concatenate([g0, 0.01 * ring, 0.02 * ring], axis=1)  # (1,G,2)
    G = g.shape[1]
    g2d = g.reshape(G, 2)
    pinv, scale = _sobolev_pinv(g2d)
    sx_ref = jnp.zeros(3)
    N = moments.shape[0]

    params, static = eqx.partition(prior_flow, eqx.is_inexact_array)

    # The batch is gathered OUTSIDE the jit and passed in as args — closing over the full
    # (millions-row) template arrays would inline them as multi-GB captured constants (OOM).
    def sob_ab(params, y_b, S_b, dg_b, d2g_b):
        flow = eqx.combine(params, static)
        B = y_b.shape[0]
        y_sheared = shear(y_b, g, dg_b, d2g_b)              # (B,G,4)
        y_flat = y_sheared.reshape(B * G, -1)
        S_flat = jnp.broadcast_to(S_b[:, None], (B, G, 4, 4)).reshape(B * G, 4, 4)
        y_std, _ = transform_dataset_to_standard(raw2standard, y_flat, S_flat)
        y_std_grid = y_std.reshape(B, G, -1)
        return _sobolev_loss(flow, y_std_grid, sob1_w, sob2_w, pinv, scale, sx_ref)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step(params, opt_state, y_b, S_b, dg_b, d2g_b):
        loss, grads = eqx.filter_value_and_grad(sob_ab)(params, y_b, S_b, dg_b, d2g_b)
        updates, opt_state = opt.update(grads, opt_state, params)
        return eqx.apply_updates(params, updates), opt_state, loss

    key = jr.key(seed)
    bs = min(batch, N)
    print(f"Warm-start A+B: jointly regressing 1st+2nd-order response over {steps} steps "
          f"(batch {bs})...")
    for i in range(steps):
        key, sk = jr.split(key)
        idx = jr.randint(sk, (bs,), 0, N)
        params, opt_state, loss = step(
            params, opt_state, moments[idx], cov[idx], dm_dg[idx, :4], d2m_dg2[idx, :4]
        )
        if i % 100 == 0 or i == steps - 1:
            print(f"  warmstart step {i:4d}  sob1+sob2={float(loss):.4f}")
    return eqx.combine(params, static)


# ---------------------------------------------------------------------------
# ELBO loss factory
# ---------------------------------------------------------------------------


def make_elbo_loss(
    N: int,
    *,
    batch_size: int = 128,
    num_samples: int = 8,
    weights: jax.Array | None = None,
    log_scale_range: tuple[float, float] | None = None,
    e_max: float = 0.0,
    use_sx: bool = True,
    raw2standard: Any = None,
    mean_log_diag: jax.Array | None = None,
    std_log_diag: jax.Array | None = None,
    mean_off: jax.Array | None = None,
    std_off: jax.Array | None = None,
    sobolev_g1_weight: float = _sobolev_g1_weight,
    sobolev_g2_weight: float = _sobolev_g2_weight,
    coeff_ood_weight: float = _shear_coeff_ood_weight,
) -> Callable:
    """Build the ELBO loss function for training the (prior_flow, q_flow) pair.

    The returned function has signature
    ``(model_tuple, data_y, data_Sigma, data_dg, data_d2g, data_X, key) -> scalar``.
    The large data arrays are passed as dynamic arguments rather than captured
    in the closure, so they are never embedded as XLA constants in the compiled
    CUBIN — only the small ``weights_cdf`` array (proportional to N floats) is
    kept in the closure.

    Parameters
    ----------
    N : int
        Number of training examples.
    batch_size : int, optional
        Mini-batch size.  Default is 128.
    num_samples : int, optional
        Number of Monte Carlo samples drawn from q per batch element.
        Default is 8.
    weights : array-like or None, optional
        Batch-sampling proposal ∝ the BFD per-copy prior weight ``nda·detj`` (nda =
        sky_density·da, HT-corrected; detj = ¼(Mr²−M1²−M2²) is the |dX/dx| Jacobian from
        BFD's kernel; see ``_finalize_dataset``).  Sampling ∝ nda·detj makes the proposal
        match the objective's static part, so the only residual per-copy weight is the
        centroid marginalisation ``L(X|C_X)``.  ``None`` ⇒ uniform sampling.
    log_scale_range : tuple of float or None, optional
        Range of ``0.5 * log det(Σ_X)`` for Σ_X marginalisation.  Pass
        ``None`` (default) to disable Σ_X conditioning.
    e_max : float, optional
        Maximum centering-bias ellipticity magnitude.  Default is 0.
    use_sx : bool, optional
        Whether to use Σ_X marginalisation.  Must be consistent with
        ``log_scale_range``.  Default is ``True``.
    raw2standard : RawMomentStandardize or None, optional
        Bijection from raw to standardised moment coordinates.
    mean_log_diag, std_log_diag, mean_off, std_off : jax.Array or None
        Whitening statistics for Cholesky conditioning features.
    sobolev_g1_weight, sobolev_g2_weight : float, optional
        Weights for the Sobolev shear-derivative term (0 = off).  When either is
        > 0 the loss adds ``λ · MSE`` between the prior flow's own moment shear-
        response (1st / 2nd ``d m / d g`` of its decode map, by autodiff) and the
        template truth (a quadratic fit of the sheared moments over the g-grid).
    coeff_ood_weight : float, optional
        Weight for the shear-coeff off-template penalty (0 = off; see
        ``ShearTaylorLast.coeff_ood_penalty``). Trains the shear layer's
        coefficient nets toward zero on synthetic (flux, size, |e|) probes beyond
        where real templates live, independent of the training batch.

    Returns
    -------
    callable
        ``(model_tuple, data_y, data_Sigma, data_dg, data_d2g, data_X, key) -> scalar``
    """
    _use_sx = use_sx and (log_scale_range is not None)
    _sob_on = (sobolev_g1_weight > 0.0) or (sobolev_g2_weight > 0.0)
    _ood_on = coeff_ood_weight > 0.0

    # ``weights`` is the batch-sampling proposal = nda (BFD template weight, HT-corrected;
    # see _finalize_dataset).  Sampling ∝ nda·detj ⇒ residual per-copy weight is L(X|C_X) only.
    if weights is not None:
        weights = jnp.asarray(weights)
        # Guard against float32 underflow when most raw weights are near zero:
        # if mean underflows to 0, dividing produces inf/NaN in the CDF.
        weights = weights / jnp.maximum(jnp.mean(weights), jnp.finfo(jnp.float32).tiny)
        weights_cdf = jnp.cumsum(weights / jnp.maximum(jnp.sum(weights), jnp.finfo(jnp.float32).tiny))
    else:
        weights_cdf = None

    g0 = jnp.array([[[0.0, 0.0]]])
    sqrt2 = 1.0 / jnp.sqrt(2.0)
    g_grid = jnp.array(
        [
            [0.0, 1.0],
            [sqrt2, sqrt2],
            [1.0, 0.0],
            [sqrt2, -sqrt2],
            [0.0, -1.0],
            [-sqrt2, -sqrt2],
            [-1.0, 0.0],
            [-sqrt2, sqrt2],
        ]
    )[jnp.newaxis, :, :]
    g = jnp.concatenate([g0, 0.01 * g_grid, 0.02 * g_grid], axis=1)  # (1, G, 2) centre + 2 rings
    G = g.shape[1]
    g2d = g.reshape(G, -1)  # (G, 2)

    # Sobolev term precompute (fixed g-grid quadratic-fit pinv + reference C_X).
    if _sob_on:
        _sob_pinv, _sob_scale = _sobolev_pinv(g2d)
        _ls_ref = 0.5 * (log_scale_range[0] + log_scale_range[1]) if _use_sx else 0.0
        _sob_sx_ref = jnp.array([_ls_ref, 0.0, 0.0])  # single reference Σ_X for the (C_X-independent) Sobolev shear-response

    # Import here to avoid circular at module level
    from ..data import transform_dataset_to_standard

    def plain_loss(model_tuple, data_y, data_Sigma, data_dg, data_d2g, data_X, key):
        prior_flow, q_flow = model_tuple

        # ── sample batch indices ∝ nda·detj (proposal == objective's static part) ──
        key, subkey = jr.split(key)
        if weights_cdf is not None:
            u = jr.uniform(subkey, shape=(batch_size,))
            idx = jnp.clip(jnp.searchsorted(weights_cdf, u, side="right"), 0, N - 1)
        else:
            idx = jr.choice(subkey, N, shape=(batch_size,), replace=True)

        y_b = data_y[idx]  # (B, D)   raw template moments
        S_b = data_Sigma[idx]  # (B, D, D)
        dg_b = data_dg[idx]   # (B, 6, 2)   [Mf,Mr,M1,M2,MX,MY] derivs
        d2g_b = data_d2g[idx] # (B, 6, 2, 2)

        BG = batch_size * G
        S = num_samples

        # ── shear EVEN moments + flatten ───────────────────────────────
        y_sheared = shear(y_b, g, dg_b[:, :4], d2g_b[:, :4])  # (B, G, D)
        y_flat = y_sheared.reshape(BG, -1)  # (BG, D)
        D_dim = S_b.shape[-1]
        Sigma_flat = jnp.broadcast_to(
            S_b[:, None, :, :], (batch_size, G, D_dim, D_dim)
        ).reshape(BG, D_dim, D_dim)
        g_flat_batch = jnp.broadcast_to(g2d[None, :, :], (batch_size, G, 2)).reshape(
            BG, 2
        )

        # ── standardise ───────────────────────────────────────────────
        y_std, Sigma_std = transform_dataset_to_standard(
            raw2standard, y_flat, Sigma_flat
        )

        # ── cond features + Cholesky ──────────────────────────────────
        cond, L = cond_features_from_y_sigma(
            y_std,
            Sigma_std,
            g_flat_batch,
            mean_log_diag=mean_log_diag,
            std_log_diag=std_log_diag,
            mean_off=mean_off,
            std_off=std_off,
            g_scale=g_scale,
        )  # (BG, cond_dim), (BG, D, D)

        # ── sample m ~ q(m | cond) ────────────────────────────────────
        key, subkey = jr.split(key)
        m, log_q = q_flow.sample_and_log_prob(
            subkey,
            sample_shape=(S,),
            condition=cond,
        )  # m: (S, BG, D),  log_q: (S, BG)
        log_q = log_q.reshape(S, batch_size, G)

        # ── likelihood + selection correction (independent of Σ_X) ──────────
        log_p_y = log_gaussian_full(y_std, m.reshape(S, BG, -1), L).reshape(
            S, batch_size, G
        )

        log10mf_destd = m.reshape(S * BG, -1)[:, 0] * raw2standard.std[0] + raw2standard.mean[0]
        mf_true = jnp.power(10.0, log10mf_destd).reshape(S, BG)
        # var_mf uses the template's own flux variance (S_b[:, 0, 0]) rather than a
        # fixed target noise. This makes the selection correction per-template: bright
        # low-noise templates get P_sel ≈ 1 (negligible correction); templates near
        # the flux threshold get a smooth, gradient-friendly correction. The physical
        # target noise would be larger (shallower survey), making the true transition
        # wider. If you need to match B16 Eq. (34) exactly, replace with a fixed
        # representative target noise (e.g., config.target_flux_var = (f_min/SNR_min)^2).
        var_mf = jnp.broadcast_to(S_b[:, 0, 0][:, None], (batch_size, G)).reshape(BG)
        log_p_sel = _log_p_select_given_x_raw(mf_true, var_mf[None, :], target_flux_min).reshape(
            S, batch_size, G
        )

        log_p_y_minus_sel = log_p_y - log_p_sel  # (S, B, G)

        # ── Σ_X marginalisation: learn the X-marginal per drawn C_X ──────────
        # For each randomly drawn centroid covariance C_X, the prior p(m|g,C_X) is
        # trained to be the template distribution *marginalised over the centroid
        # offset* X, weighting each copy by L(X|C_X)=N(X;0,C_X) computed from the TRUE
        # 1st-order centroid moments X=[MX,MY].  (Copies with large |X| have lower M0
        # and lower MR/M0 by the BFD convexity requirement, so down-weighting them
        # recovers the near-centre distribution.)  We therefore condition the prior on
        # C_X, form the per-copy ELBO, and take the L(X|C_X)-weighted batch mean — NOT
        # the old per-template softmax *over* C_X draws.  Losses for the n_sx random
        # C_X draws are then averaged.
        if _use_sx:
            key, k_sx = jr.split(key)
            sx_conds = _sample_sx_conds(k_sx, log_scale_range, e_max)  # (17, 3)
            X_b = data_X[idx]  # (B, 2)  true centroid moments [MX, MY] at g=0
            # Shear the centroid by its own derivatives [4:6] so L(X(g)|C_X) tracks
            # the BFD centroid shear response (was held at g=0).  See make_nll_loss.
            X_bg = shear(X_b, g, dg_b[:, 4:6], d2g_b[:, 4:6])  # (B, G, 2)
            X_bg_flat = X_bg.reshape(BG, 2)
            m_SBG = m.reshape(S, BG, -1)  # (S, BG, D)

            def _loss_for_sx(sx_cond):
                # ELBO of every copy under the prior conditioned on this C_X.
                sx_tiled = jnp.broadcast_to(sx_cond[None, :], (BG, 3))
                cond_p = jnp.concatenate([g_flat_batch, sx_tiled], axis=-1)  # (BG, 5)
                log_pz = jax.vmap(
                    lambda m_s: prior_flow.log_prob(m_s, condition=cond_p)
                )(m_SBG).reshape(S, batch_size, G)  # (S, B, G)
                elbo = log_p_y_minus_sel + log_pz - log_q  # (S, B, G)
                lse = logsumexp(elbo, axis=0) - jnp.log(S)  # (B, G)
                # Batch drawn ∝ nda·detj ⇒ the ONLY residual per-copy weight is the BFD
                # centroid marginalisation L(X(g)|C_X), self-normalised per g ⇒ the
                # nda·detj·L X-marginal for this C_X.  detj is in the proposal, not here.
                logL_bg = _batch_log_L_X(X_bg_flat, sx_cond).reshape(batch_size, G)
                w_bg = jax.nn.softmax(logL_bg, axis=0)  # (B, G) per-g, cols sum to 1
                return -jnp.mean(jnp.sum(w_bg * lse, axis=0))  # mean over g

            # jax.checkpoint keeps lax.map from holding all n_sx residuals at once.
            losses_sx = jax.lax.map(jax.checkpoint(_loss_for_sx), sx_conds)  # (n_sx,)
            loss = jnp.mean(losses_sx)
        else:
            # prior conditioned on [g1, g2, log_scale=0, e1=0, e2=0]
            cond_p = jnp.concatenate(
                [g_flat_batch, jnp.zeros((BG, 3), dtype=g_flat_batch.dtype)], axis=-1
            )  # (BG, 5)
            log_p_z = jax.vmap(lambda m_s: prior_flow.log_prob(m_s, condition=cond_p))(
                m.reshape(S, BG, -1)
            ).reshape(
                S, batch_size, G
            )  # (S, B, G)

            elbo = log_p_y_minus_sel + log_p_z - log_q
            lse = logsumexp(elbo, axis=0) - jnp.log(S)  # (B, G)
            # No C_X ⇒ no L; batch is already ∝ nda·detj ⇒ plain mean ELBO over the batch.
            loss = -jnp.mean(lse)

        if _sob_on:
            # Shear-response is C_X-independent (see _shear_response), so a single
            # reference sx_ref suffices — no averaging over the sx_conds draws.
            #
            # The target/anchor must be the DENOISED moment, not y_std (the noisy
            # observed template): ELBO mode exists precisely to avoid the "m≈y"
            # shortcut (see make_nll_loss's docstring), and the raw->standard map
            # is nonlinear (log flux, moment ratios), so its local Jacobian/Hessian
            # evaluated at a noisy point differs from the denoised one — a bias
            # that hits the 2nd-order (curvature) Sobolev term hardest. Build the
            # sheared/standardised grid from q's own denoised estimate at g=0
            # (mean of its S samples there) instead of the noisy y_b/y_std.
            m0_std = jnp.mean(m.reshape(S, batch_size, G, -1)[:, :, 0, :], axis=0)  # (B, D)
            # ponytail: q is freshly-reinitialised at the start of elbo fine-tuning and can
            # emit few-sigma-outlier denoised means before it converges; raw2standard.inverse
            # exponentiates the log10(Mf) coord with no bound, so an outlier here blows Mf up
            # to inf/nan and poisons the whole batch loss (mean vs median divergence + NaNs
            # seen in early elbo_ft blocks). Clip to a generous 8-sigma box — no real template
            # is out here, only pre-convergence q garbage.
            m0_std = jnp.clip(m0_std, -8.0, 8.0)
            x0_raw = jax.vmap(raw2standard.inverse)(m0_std)  # (B, D)
            x0_sheared = shear(x0_raw, g, dg_b[:, :4], d2g_b[:, :4])  # (B, G, D)
            denoised_std_grid = jax.vmap(raw2standard.transform)(
                x0_sheared.reshape(BG, -1)
            ).reshape(batch_size, G, -1)
            # row-weighting rolled back 2026-07-30: _sobolev_row_weights reuses the
            # density term's centroid-marginalisation softmax (see _batch_log_L_X) as an
            # importance weight on a per-template SUPERVISED regression target, not an
            # expectation — that reallocates fitting capacity toward small-|X| (bright/
            # small) templates instead of correcting an estimator, and produced a large
            # sign-flipping m-bias regression (verified: clean flows all used row_w=None).
            row_w = None
            loss = loss + _sobolev_loss(
                prior_flow,
                denoised_std_grid,
                sobolev_g1_weight,
                sobolev_g2_weight,
                _sob_pinv,
                _sob_scale,
                _sob_sx_ref,
                row_w,
            )

        if _ood_on:
            key, k_ood = jr.split(key)
            loss = loss + coeff_ood_weight * _coeff_ood_penalty(prior_flow, k_ood)

        return loss

    return plain_loss


# ---------------------------------------------------------------------------
# NLL loss factory (prior pre-training, m ≈ y)
# ---------------------------------------------------------------------------


def make_nll_loss(
    N: int,
    *,
    batch_size: int = 128,
    weights: jax.Array | None = None,
    log_scale_range: tuple[float, float] | None = None,
    e_max: float = 0.0,
    use_sx: bool = True,
    raw2standard: Any = None,
    sobolev_g1_weight: float = _sobolev_g1_weight,
    sobolev_g2_weight: float = _sobolev_g2_weight,
    coeff_ood_weight: float = _shear_coeff_ood_weight,
) -> Callable:
    """Direct NLL loss for prior-only pre-training (m ≈ y approximation).

    Treats each template's moments y as a direct sample from the prior p(m|g,C_X),
    valid when templates have high SNR.  Unlike make_elbo_loss, no Q flow is
    involved — the prior is supervised directly on log p(y_std | g, C_X), averaged
    over the shear g-grid and the 17-point C_X stencil, weighted by L(X|C_X), with
    the same flux-selection correction ``-log P_sel(mf)`` make_elbo_loss applies
    (here evaluated at the template's own sheared flux and its own flux variance,
    since m ≈ y stands in for a q-sample).

    The BFD per-copy prior weight is nda·detj·L(X|C_X): the template weight nda
    (= sky-density·da, momentcalc.py:556; HT-corrected for our subsample), the |dX/dx|
    Jacobian detj = ¼(Mr²−M1²−M2²) carried in BFD's kernel (probabilities_jax.py:200),
    and the centroid marginalisation L (probabilities_jax.py:198).  nda·detj enters ONLY
    as the batch SAMPLING proposal (``weights`` ∝ nda·detj), so sampling ∝ nda·detj makes
    the proposal match the objective's static part and the residual per-copy weight in the
    softmax is L alone.  No is_corr (proposal == objective's static part).

    ``sobolev_g1_weight`` / ``sobolev_g2_weight`` add the same Sobolev shear-derivative
    term as :func:`make_elbo_loss` (0 = off): ``λ · MSE`` between the prior flow's own
    moment shear-response and the template quadratic-fit truth.

    Returned signature:
        (prior_flow, data_y, data_Sigma, data_dg, data_d2g, data_X, key) -> scalar
    """
    _use_sx = use_sx and (log_scale_range is not None)
    _sob_on = (sobolev_g1_weight > 0.0) or (sobolev_g2_weight > 0.0)
    _ood_on = coeff_ood_weight > 0.0

    # ``weights`` is the batch-sampling proposal = nda (BFD template weight, HT-corrected;
    # see _finalize_dataset).  Sampling ∝ nda·detj ⇒ residual per-copy weight is L(X|C_X) only.
    if weights is not None:
        weights = jnp.asarray(weights)
        weights = weights / jnp.maximum(jnp.mean(weights), jnp.finfo(jnp.float32).tiny)
        weights_cdf = jnp.cumsum(
            weights / jnp.maximum(jnp.sum(weights), jnp.finfo(jnp.float32).tiny)
        )
    else:
        weights_cdf = None

    g0 = jnp.array([[[0.0, 0.0]]])
    sqrt2 = 1.0 / jnp.sqrt(2.0)
    g_grid = jnp.array(
        [
            [0.0, 1.0], [sqrt2, sqrt2], [1.0, 0.0], [sqrt2, -sqrt2],
            [0.0, -1.0], [-sqrt2, -sqrt2], [-1.0, 0.0], [-sqrt2, sqrt2],
        ]
    )[jnp.newaxis, :, :]
    g = jnp.concatenate([g0, 0.01 * g_grid, 0.02 * g_grid], axis=1)  # (1, G, 2) centre + 2 rings
    G = g.shape[1]
    g2d = g.reshape(G, -1)  # (G, 2)

    # Sobolev term precompute (fixed g-grid quadratic-fit pinv + reference C_X).
    if _sob_on:
        _sob_pinv, _sob_scale = _sobolev_pinv(g2d)
        _ls_ref = 0.5 * (log_scale_range[0] + log_scale_range[1]) if _use_sx else 0.0
        _sob_sx_ref = jnp.array([_ls_ref, 0.0, 0.0])  # single reference Σ_X for the (C_X-independent) Sobolev shear-response

    def nll_loss(prior_flow, data_y, data_Sigma, data_dg, data_d2g, data_X, key):
        # ── batch sampling ∝ nda·detj (proposal == objective's static part) ────
        key, subkey = jr.split(key)
        if weights_cdf is not None:
            u = jr.uniform(subkey, shape=(batch_size,))
            idx = jnp.clip(jnp.searchsorted(weights_cdf, u, side="right"), 0, N - 1)
        else:
            idx = jr.choice(subkey, N, shape=(batch_size,), replace=True)

        y_b = data_y[idx]     # (B, 4)
        S_b = data_Sigma[idx] # (B, 4, 4)
        dg_b = data_dg[idx]   # (B, 6, 2)    [Mf,Mr,M1,M2,MX,MY] derivs
        d2g_b = data_d2g[idx] # (B, 6, 2, 2)

        BG = batch_size * G

        # ── shear EVEN copies + standardise (m ≈ y) ──────────────────────
        y_sheared = shear(y_b, g, dg_b[:, :4], d2g_b[:, :4])  # (B, G, 4)
        # detj is held at g=0 ON PURPOSE: BFD's detj is the TARGET's (g-constant),
        # not the copy's — shearing it tripled the grid m-bias (+0.13→+0.32, 28σ,
        # 2026-06-29).  Only the centroid weight L is g-dependent (BFD's chiO is).
        y_flat = y_sheared.reshape(BG, -1)        # (BG, 4)
        g_flat_batch = jnp.broadcast_to(g2d[None, :, :], (batch_size, G, 2)).reshape(BG, 2)

        transform_and_logdet = raw2standard.transform_and_log_det
        y_std = jax.vmap(transform_and_logdet)(y_flat)[0]  # (BG, 4)

        # ── flux-selection correction (mirrors make_elbo_loss; m ≈ y ⇒ mf_true is
        # the template's own sheared raw flux, var_mf its own flux variance) ──────
        mf_true = y_flat[:, 0]  # (BG,) raw flux, already sheared
        var_mf = jnp.broadcast_to(S_b[:, 0, 0][:, None], (batch_size, G)).reshape(BG)
        log_p_sel = _log_p_select_given_x_raw(mf_true, var_mf, target_flux_min).reshape(
            batch_size, G
        )

        # ── Σ_X marginalisation ───────────────────────────────────────────
        if _use_sx:
            key, k_sx = jr.split(key)
            sx_conds = _sample_sx_conds(k_sx, log_scale_range, e_max)  # (17, 3)
            X_b = data_X[idx]  # (B, 2)  centroid at g=0
            # Shear the centroid by its own derivatives [4:6] so the weight
            # L(X(g)|C_X) tracks the BFD centroid shear response (was held at g=0).
            X_bg = shear(X_b, g, dg_b[:, 4:6], d2g_b[:, 4:6])  # (B, G, 2)
            X_bg_flat = X_bg.reshape(BG, 2)

            def _loss_for_sx(sx_cond):
                sx_tiled = jnp.broadcast_to(sx_cond[None, :], (BG, 3))
                cond_p = jnp.concatenate([g_flat_batch, sx_tiled], axis=-1)  # (BG, 5)
                log_p = prior_flow.log_prob(y_std, condition=cond_p).reshape(
                    batch_size, G
                )  # (B, G)
                log_p_minus_sel = log_p - log_p_sel
                logL_bg = _batch_log_L_X(X_bg_flat, sx_cond).reshape(batch_size, G)
                # Batch drawn ∝ nda·detj ⇒ the ONLY residual per-copy weight is the BFD
                # centroid marginalisation L(X(g)|C_X), self-normalised per g ⇒ the
                # nda·detj·L X-marginal for this C_X.  detj is in the proposal, not here;
                # no is_corr (proposal == nda·detj == objective's static part).
                w_bg = jax.nn.softmax(logL_bg, axis=0)  # (B, G) per-g, cols sum to 1
                return -jnp.mean(jnp.sum(w_bg * log_p_minus_sel, axis=0))  # mean over g

            losses_sx = jax.lax.map(jax.checkpoint(_loss_for_sx), sx_conds)
            loss = jnp.mean(losses_sx)
        else:
            # No C_X ⇒ no L; batch is already ∝ nda·detj ⇒ plain mean NLL over the batch.
            cond_p = jnp.concatenate(
                [g_flat_batch, jnp.zeros((BG, 3), dtype=g_flat_batch.dtype)], axis=-1
            )
            log_p = prior_flow.log_prob(y_std, condition=cond_p).reshape(batch_size, G)
            loss = -jnp.mean(log_p - log_p_sel)

        if _sob_on:
            # Shear-response is C_X-independent (see _shear_response), so a single
            # reference sx_ref suffices — no averaging over the sx_conds draws.
            # row-weighting rolled back 2026-07-30: see the make_nll_loss call site for
            # why (_sobolev_row_weights misapplies an SNIS importance weight to a fixed
            # per-template regression target; caused a large sign-flipping m-bias).
            row_w = None
            loss = loss + _sobolev_loss(
                prior_flow,
                y_std.reshape(batch_size, G, -1),
                sobolev_g1_weight,
                sobolev_g2_weight,
                _sob_pinv,
                _sob_scale,
                _sob_sx_ref,
                row_w,
            )

        if _ood_on:
            key, k_ood = jr.split(key)
            loss = loss + coeff_ood_weight * _coeff_ood_penalty(prior_flow, k_ood)

        return loss

    return nll_loss


# ---------------------------------------------------------------------------
# Flow builder
# ---------------------------------------------------------------------------


def build_flows(
    key: jax.Array,
    latent_dim: int,
    cond_dim: int,
    prior_flow_layers: int = prior_flow_layers,
    prior_early_nn_width: int = prior_early_nn_width,
    prior_early_nn_depth: int = prior_early_nn_depth,
    prior_last_nn_width: int = prior_last_nn_width,
    prior_last_nn_depth: int = prior_last_nn_depth,
    prior_sigmax_nn_width: int = prior_sigmax_nn_width,
    prior_sigmax_nn_depth: int = prior_sigmax_nn_depth,
    prior_sigmax_log_scale_mean: float = prior_sigmax_log_scale_mean,
    prior_sigmax_log_scale_std: float = prior_sigmax_log_scale_std,
    prior_size_loc_c1: float | None = None,
    raw2standard: Any = None,
    prior_e_max: float = e_max,
    q_flow_layers: int = q_flow_layers,
    q_nn_width: int = q_nn_width,
    q_nn_depth: int = q_nn_depth,
    min_scale: float = min_scale,
    max_scale: float = max_scale,
    shear_layer_kind: str = _shear_layer_kind,
    prior_conditional_first: bool = _prior_conditional_first,
    prior_shear_own_e: bool = _prior_shear_own_e,
    prior_shear_split_ab: bool = _prior_shear_split_ab,
    prior_shear_spin2_owne: bool = _prior_shear_spin2_owne,
) -> tuple[Any, Any]:
    """Construct the prior and variational (q) normalizing flows.

    The **prior** is a :func:`~bfd_cnf.models.bijections.new_masked_autoregressive_flow`
    conditioned on ``[g1, g2, log_scale, e1, e2]`` and equipped with a
    :class:`~bfd_cnf.models.bijections.SigmaXCouplingLayer`.

    The **q flow** is a standard flowjax masked autoregressive flow
    conditioned on the full ``cond_dim``-dimensional feature vector built by
    :func:`cond_features_from_y_sigma`.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key (split into two sub-keys).
    latent_dim : int
        Dimensionality of the latent space (typically 4 for galaxy moments).
    cond_dim : int
        Conditioning dimensionality for the q flow.
    prior_flow_layers : int, optional
        Number of coupling/autoregressive layers in the prior.  Default from
        ``config.prior_flow_layers``.
    prior_early_nn_width : int, optional
        Hidden width of the early-layer MAF networks in the prior.  Default
        from ``config.prior_early_nn_width``.
    prior_early_nn_depth : int, optional
        Depth of the early-layer MAF networks in the prior.  Default from
        ``config.prior_early_nn_depth``.
    prior_last_nn_width : int, optional
        Hidden width of the final coupling-layer networks in the prior.  Default
        from ``config.prior_last_nn_width``.
    prior_last_nn_depth : int, optional
        Depth of the final coupling-layer networks in the prior.  Default from
        ``config.prior_last_nn_depth``.
    prior_sigmax_nn_width : int, optional
        Hidden width of the SigmaX coupling network in the prior.  Default from
        ``config.prior_sigmax_nn_width``.
    prior_sigmax_nn_depth : int, optional
        Depth of the SigmaX coupling network in the prior.  Default from
        ``config.prior_sigmax_nn_depth``.
    prior_sigmax_log_scale_mean : float, optional
        Mean of the log-scale prior for SigmaX conditioning.  Default from
        ``config.prior_sigmax_log_scale_mean``.
    prior_sigmax_log_scale_std : float, optional
        Std of the log-scale prior for SigmaX conditioning.  Default from
        ``config.prior_sigmax_log_scale_std``.
    q_flow_layers : int, optional
        Number of masked autoregressive layers in the q flow.  Default from
        ``config.q_flow_layers``.
    q_nn_width : int, optional
        Hidden width of the MAF networks in the q flow.  Default from
        ``config.q_nn_width``.
    q_nn_depth : int, optional
        Depth of the MAF networks in the q flow.  Default from
        ``config.q_nn_depth``.
    min_scale : float, optional
        Minimum scale for the BoundedAffine transformer.  Default from
        ``config.min_scale``.
    max_scale : float, optional
        Maximum scale for the BoundedAffine transformer.  Default from
        ``config.max_scale``.

    Returns
    -------
    prior : Transformed
        Prior normalizing flow conditioned on shear and centering-bias (Σ_X) parameters.
    q_flow : Transformed
        Variational q flow conditioned on the moment + covariance features.
    """
    k1, k2 = jr.split(key)

    # The SigmaX locked-size constant c1 MUST equal the standardiser's mean[1]/std[1]
    # (Mr/Mf); see SigmaXCouplingLayer.  Always DERIVE it from the live standardiser so
    # it is recomputed per training run and travels with the flow via the sidecar stats
    # (<flow>.eqx.stats.npz), never a hardcoded constant.  c1 is a *static* field, so a
    # flow must be retrained if this value differs from what it was trained with.
    if raw2standard is not None:
        prior_size_loc_c1 = float(raw2standard.mean[1] / raw2standard.std[1])
    elif prior_size_loc_c1 is None:
        raise ValueError(
            "build_flows needs raw2standard to derive the SigmaX size constant "
            "c1 = mean[1]/std[1] (or pass prior_size_loc_c1 explicitly)."
        )

    prior = new_masked_autoregressive_flow(
        k1,
        base_dist=non_trainable(
            MultivariateNormal(jnp.zeros(latent_dim), jnp.eye(latent_dim))
        ),
        flow_layers=prior_flow_layers,
        nn_activation=jax.nn.silu,
        nn_depth=prior_early_nn_depth,
        nn_width=prior_early_nn_width,
        last_layer_nn_depth=prior_last_nn_depth,
        last_layer_nn_width=prior_last_nn_width,
        last_layer_cond_dim=5,  # was 2 — full [g1, g2, log_scale, e1, e2]
        invert=True,
        quadratic_last=True,
        sigmax_cond_dim=5,  # enables SigmaXCouplingLayer
        sigmax_nn_width=prior_sigmax_nn_width,
        sigmax_nn_depth=prior_sigmax_nn_depth,
        sigmax_log_scale_mean=prior_sigmax_log_scale_mean,
        sigmax_log_scale_std=prior_sigmax_log_scale_std,
        sigmax_e_max=prior_e_max,
        sigmax_size_loc=prior_size_loc_c1,
        shear_layer_kind=shear_layer_kind,
        conditional_first=prior_conditional_first,
        shear_own_e=prior_shear_own_e,
        shear_split_ab=prior_shear_split_ab,
        shear_spin2_owne=prior_shear_spin2_owne,
    )

    q_flow = masked_autoregressive_flow(
        k2,
        base_dist=MultivariateNormal(jnp.zeros(latent_dim), jnp.eye(latent_dim)),
        transformer=BoundedAffine(min_scale=min_scale, max_scale=max_scale),
        flow_layers=q_flow_layers,
        nn_activation=jax.nn.silu,
        nn_depth=q_nn_depth,
        nn_width=q_nn_width,
        cond_dim=cond_dim,
        invert=True,
    )
    return prior, q_flow
