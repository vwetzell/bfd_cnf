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
    SigmaXCouplingLayer,
    SigmaXBlockLayer,
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
    sobolev_g1_weight as _sobolev_g1_weight,
    sobolev_g2_weight as _sobolev_g2_weight,
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
    """Return the flow's ``ShearTaylorLast`` instance."""
    layers = [
        n
        for n in jax.tree_util.tree_leaves(
            prior_flow, is_leaf=lambda x: isinstance(x, ShearTaylorLast)
        )
        if isinstance(n, ShearTaylorLast)
    ]
    return layers[0] if layers else None


def _find_sigmax_layer(prior_flow):
    """Return the flow's ``SigmaXCouplingLayer``/``SigmaXBlockLayer`` instance (or ``None``)."""
    is_sigmax = lambda x: isinstance(x, (SigmaXCouplingLayer, SigmaXBlockLayer))
    layers = [
        n
        for n in jax.tree_util.tree_leaves(prior_flow, is_leaf=is_sigmax)
        if is_sigmax(n)
    ]
    return layers[0] if layers else None


def _shear_response(prior_flow, m0, sx_ref):
    """Flow's data-space moment shear-response ``(A, B)`` at each unsheared moment.

    ``ShearTaylorLast`` sits right after the unconditional bulk, with
    :class:`~.bijections.SigmaXCouplingLayer` chained after it — i.e. Σ_X is now
    the data-adjacent layer (see :class:`~.bijections.EarlyChain`). So the shear
    layer's own local generative response
    (:meth:`~.bijections.ShearTaylorLast.shear_derivs_generative`) is the response
    at ITS coordinate — one step before Σ_X — not yet the data-space response.
    Since Σ_X does not depend on ``g``, the data-space response is obtained by the
    (2nd-order) chain rule through Σ_X's own local transform, evaluated at
    ``sx_ref``::

        A_data        = J_sx . A_shear
        B_data[j,a,b] = J_sx . B_shear[.,a,b] + H_sx[j,.,.] : A_shear[.,a] A_shear[.,b]

    ``J_sx``/``H_sx`` are the Jacobian/Hessian of Σ_X's ``transform`` at the shear
    layer's own g=0 point (recovered from ``m0`` via Σ_X's closed-form inverse) —
    LOCAL derivatives of Σ_X's small coeff nets, same cost class as
    ``shear_derivs_generative``'s own ``∂A/∂x``; no whole-flow autodiff.

    Unlike before the shear/Σ_X swap, the response is now genuinely C_X-dependent
    (Σ_X's Jacobian is a function of ``sx_ref``), so evaluating at a single
    reference ``sx_ref`` (as callers do, for a cheap Sobolev term) is an
    approximation rather than an exact simplification — see the call sites in
    :func:`_sobolev_loss`/``make_elbo_loss``.
    """
    shear_layer = _find_shear_taylor_layer(prior_flow)
    sigmax_layer = _find_sigmax_layer(prior_flow)
    if sigmax_layer is None:
        return jax.vmap(shear_layer.shear_derivs_generative)(m0)

    sx_cond_full = jnp.concatenate([jnp.zeros(2), sx_ref])  # [g1=0,g2=0,log_scale,e1,e2]

    def _one(m0_i):
        # Within the flow's decode direction, Σ_X's OWN generative step is its
        # `.inverse_and_log_det` (v -> m; see EarlyChain/Chain composition order),
        # so recovering the shear layer's fixed point v0 from m0 needs Σ_X's
        # `.transform_and_log_det` (the inverse of that inverse).
        v0, _ = sigmax_layer.transform_and_log_det(m0_i, sx_cond_full)
        A_shear, B_shear = shear_layer.shear_derivs_generative(v0)

        def _sx_generative(v):
            return sigmax_layer.inverse_and_log_det(v, sx_cond_full)[0]  # v -> m

        J_sx = jax.jacfwd(_sx_generative)(v0)               # (4,4) d(m)/d(v)
        H_sx = jax.jacfwd(jax.jacfwd(_sx_generative))(v0)   # (4,4,4) d^2(m)/d(v)^2

        A_data = jnp.einsum("ji,ia->ja", J_sx, A_shear)
        B_data = jnp.einsum("ji,iab->jab", J_sx, B_shear) + jnp.einsum(
            "jik,ia,kb->jab", H_sx, A_shear, A_shear
        )
        return A_data, B_data

    return jax.vmap(_one)(m0)


def _sobolev_row_weights(
    X_b: jax.Array, sx_conds: jax.Array, max_weight_mult: float = 20.0
) -> jax.Array:
    """Per-template weight matching the density loss's L(X|C_X) SNIS correction.

    The density loss estimates ``E_{nda·detj·L(X|C_X)}[...]`` via templates drawn
    ∝ nda·detj (the shared batch draw — already identical between the density and
    Sobolev terms, no extra factor needed here) followed by a self-normalised
    softmax-over-batch correction for the residual ``L(X|C_X)`` factor, averaged
    over the C_X stencil (``loss = mean(losses_sx)``).  A_tgt/B_tgt are C_X-
    independent, and A_flow/B_flow were too before the shear/Σ_X layer swap (see
    ``_shear_response``); treating the Sobolev integrand as c-independent here is
    now an approximation, not exact, but for a (nearly) c-independent integrand
    ``mean_c[softmax_i(logL(X_i|c))]`` is still close to the weight the density
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

    The template target (A_tgt, B_tgt) is C_X-independent; the flow's response is
    now genuinely C_X-dependent since the shear/Σ_X layer swap (see
    :func:`_shear_response`), so evaluating at a single ``sx_ref`` — rather than
    averaging over the C_X stencil like the density term does — is a deliberate
    cheap approximation, not an exact simplification.

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

    Returns
    -------
    callable
        ``(model_tuple, data_y, data_Sigma, data_dg, data_d2g, data_X, key) -> scalar``
    """
    _use_sx = use_sx and (log_scale_range is not None)
    # A degenerate stencil (log_scale_range collapsed to a single point, e_max=0
    # -- converge_train.py's --sigmax-layer none) makes every one of the 17
    # stencil draws land on the SAME sx_cond, so _sample_sx_conds + lax.map
    # would recompute the identical log_pz 17x for nothing. Skip the stencil
    # entirely and evaluate once at that single point (L(X|C_X) still applied
    # -- this is not the same as _use_sx=False, which drops L(X|C_X) too).
    _sx_fixed_point = (
        _use_sx and log_scale_range[0] == log_scale_range[1] and e_max == 0.0
    )
    _sob_on = (sobolev_g1_weight > 0.0) or (sobolev_g2_weight > 0.0)

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
    # 9 radii -- restored 2026-08-05 (see make_nll_loss's matching comment).
    _radii = jnp.array([0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05])[:, None, None]
    g = jnp.concatenate([g0] + [r * g_grid for r in _radii], axis=1)  # (1, G, 2)
    G = g.shape[1]
    g2d = g.reshape(G, -1)  # (G, 2)

    # Sobolev term precompute (fixed g-grid quadratic-fit pinv + reference C_X).
    if _sob_on:
        _sob_pinv, _sob_scale = _sobolev_pinv(g2d)
        _ls_ref = 0.5 * (log_scale_range[0] + log_scale_range[1]) if _use_sx else 0.0
        _sob_sx_ref = jnp.array([_ls_ref, 0.0, 0.0])  # single reference Σ_X the Sobolev shear-response is evaluated at (approximation, see _shear_response)

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

            if _sx_fixed_point:
                # Every stencil point is the same real (log_scale, 0, 0) -- evaluate
                # once, no stencil/lax.map, no key spent (nothing random to draw).
                loss = _loss_for_sx(jnp.array([log_scale_range[0], 0.0, 0.0]))
            else:
                key, k_sx = jr.split(key)
                sx_conds = _sample_sx_conds(k_sx, log_scale_range, e_max)  # (17, 3)
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
            # Shear-response is now C_X-dependent post shear/Σ_X swap (see
            # _shear_response); evaluating at a single reference sx_ref instead of
            # averaging over the sx_conds draws is a deliberate cheap approximation.
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
            # 2026-08-05 retry -- see make_nll_loss's matching comment for the full
            # reasoning (previously rolled back 2026-07-30 for a sign-flipping
            # m-bias; retrying now that the implicit loss's own B-identifiability
            # is dramatically better, so starving low-L(X|C_X) regions of Sobolev
            # supervision should be far less costly than it was back then).
            row_w = _sobolev_row_weights(X_b, _sob_sx_ref[None, :]) if _use_sx else None
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
    freeze_centroid_reweight_at_g0: bool = False,
    disable_centroid_reweight: bool = False,
    stencil_radii: jax.Array | None = None,
    curvature_weight: float = 0.0,
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

    ``freeze_centroid_reweight_at_g0`` (default False, EXPERIMENTAL/diagnostic):
    normally the per-copy softmax weight w_bg = softmax(L(X(g)|C_X)) is recomputed
    at EACH g-stencil point from the SHEARED centroid X(g) -- i.e. which copies
    dominate the batch average changes from one stencil point to the next, since
    each copy's OWN centroid response dX/dg differs (real spread, unlike each
    copy's m-channel dm/dg which is fairly homogeneous at fixed flux/size -- see
    the 2026-08-05 investigation). ShearTaylorLast is a PURE bijective transport:
    it can only move mass along smooth per-point trajectories, it has no
    mechanism to represent this independent g-dependent REWEIGHTING of which
    samples matter. The hypothesis: the optimizer, unable to represent this
    reweighting effect any other way, partially "fakes" it via anomalously large
    B (2nd-order) coefficients concentrated at the outer stencil ring, where the
    reweighting-vs-g0 divergence is largest. Setting this True computes w_bg ONCE
    from the g=0 centroid X_b and reuses it for every stencil column (freezing
    "which copies count" at its g=0 value) -- a deliberate simplification, not
    obviously more physically correct, but a clean ablation: if B's blowup
    shrinks substantially with this on, that's strong evidence the reweighting
    mechanism (not weak identifiability alone) is the driver.

    ``disable_centroid_reweight`` (default False, EXPERIMENTAL/diagnostic):
    a stronger ablation than the freeze above -- drops the SNIS L(X|C_X)
    softmax entirely (flat/uniform mean over the batch, ignoring L(X|C_X)
    altogether), rather than just freezing its g-dependence. 2026-08-05
    ESS investigation found a real, flux-localized 2-3% bias in the
    softmax's finite-batch estimate (loss_lx_ess_check.py); fixing that via
    an 8x larger batch left the m-tilt unchanged (see
    snis-batch-size-fix-does-not-fix-tilt memory). This tests the stronger
    claim -- does the reweighting MECHANISM itself (not just its ESS bias)
    play any causal role at all.

    Returned signature:
        (prior_flow, data_y, data_Sigma, data_dg, data_d2g, data_X, key) -> scalar
    """
    _use_sx = use_sx and (log_scale_range is not None)
    # A degenerate stencil (log_scale_range collapsed to a single point, e_max=0
    # -- converge_train.py's --sigmax-layer none) makes every one of the 17
    # stencil draws land on the SAME sx_cond, so _sample_sx_conds + lax.map
    # would recompute the identical log_pz 17x for nothing. Skip the stencil
    # entirely and evaluate once at that single point (L(X|C_X) still applied
    # -- this is not the same as _use_sx=False, which drops L(X|C_X) too).
    _sx_fixed_point = (
        _use_sx and log_scale_range[0] == log_scale_range[1] and e_max == 0.0
    )
    _sob_on = (sobolev_g1_weight > 0.0) or (sobolev_g2_weight > 0.0)

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
    # 9 radii -- separating a linear (A) from a quadratic (B) response from only 2
    # non-zero radii is the bare minimum (3 points for 3 unknowns f0,A,B: exactly
    # determined, zero redundancy to average out per-step batch noise, and a short
    # lever arm for the g^2 term's curvature to stand out against the g term).
    # More, wider-spaced radii give the implicit density loss genuine statistical
    # redundancy + a longer lever arm for B, without any explicit A/B regression
    # -- see 2026-08-05 investigation (this alone, tried as a std-2-radii-only
    # rescale, did not fix the nosob B blowup; a wider/denser radial stencil is a
    # different, untried lever). Round 2: widened/densified further after the
    # 5-radius version gave a real, consistent ~2-2.5x reduction in B's mismatch
    # (not full convergence) -- 0.05 stays within the docstring's own noted
    # physically-relevant |g| range. Round 3 (same day): dropped back to 2 radii
    # to free up compute for --batch-size 4096 (SNIS ESS investigation). Round 4
    # (same day): restored to 9 -- that batch-size test showed the m-tilt
    # unchanged either way (snis-batch-size-fix-does-not-fix-tilt memory), so
    # there's no reason to keep trading stencil accuracy for batch size.
    #
    # ``stencil_radii`` overrides the default 9-radius stencil (diagnostic knob,
    # 2026-08-06): pass ``jnp.array([])`` to train on ONLY the g0 point, isolating
    # whether the shared-parameter multi-g averaging itself (vs. the shear layer's
    # mere presence, or the flux-selection term) is what pulls the bulk layers'
    # g=0 fit away from a clean unit normal -- see the bulk-only isolation test.
    _radii = (
        stencil_radii
        if stencil_radii is not None
        else jnp.array([0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05])
    )[:, None, None]
    g = jnp.concatenate([g0] + [r * g_grid for r in _radii], axis=1)  # (1, G, 2)
    G = g.shape[1]
    g2d = g.reshape(G, -1)  # (G, 2)

    # Sobolev term precompute (fixed g-grid quadratic-fit pinv + reference C_X).
    if _sob_on:
        _sob_pinv, _sob_scale = _sobolev_pinv(g2d)
        _ls_ref = 0.5 * (log_scale_range[0] + log_scale_range[1]) if _use_sx else 0.0
        _sob_sx_ref = jnp.array([_ls_ref, 0.0, 0.0])  # single reference Σ_X the Sobolev shear-response is evaluated at (approximation, see _shear_response)

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
            X_b = data_X[idx]  # (B, 2)  centroid at g=0
            # Shear the centroid by its own derivatives [4:6] so the weight
            # L(X(g)|C_X) tracks the BFD centroid shear response (was held at g=0).
            X_bg = shear(X_b, g, dg_b[:, 4:6], d2g_b[:, 4:6])  # (B, G, 2)
            X_bg_flat = X_bg.reshape(BG, 2)
            # EXPERIMENTAL (see docstring): reuse the g=0 centroid for every
            # stencil column instead of the per-g sheared one, so w_bg no longer
            # varies with g -- ablation for whether the g-dependent REWEIGHTING
            # (not just weak identifiability) is driving the 2nd-order blowup.
            X_bg_flat_for_L = (
                jnp.broadcast_to(X_b[:, None, :], (batch_size, G, 2)).reshape(BG, 2)
                if freeze_centroid_reweight_at_g0 else X_bg_flat
            )

            def _loss_for_sx(sx_cond):
                sx_tiled = jnp.broadcast_to(sx_cond[None, :], (BG, 3))
                cond_p = jnp.concatenate([g_flat_batch, sx_tiled], axis=-1)  # (BG, 5)
                log_p = prior_flow.log_prob(y_std, condition=cond_p).reshape(
                    batch_size, G
                )  # (B, G)
                log_p_minus_sel = log_p - log_p_sel
                if disable_centroid_reweight:
                    # FULL ablation (distinct from freeze_centroid_reweight_at_g0, which
                    # still applies the softmax but freezes its g-dependence): skip the
                    # SNIS L(X|C_X) correction entirely -- flat mean over the batch, same
                    # as the _use_sx=False branch below. Tests whether the reweighting
                    # MECHANISM itself (not just its finite-batch ESS bias, see
                    # loss_lx_ess_check.py / snis-batch-size-fix-does-not-fix-tilt memory)
                    # has any causal role in the m-tilt, since fixing the ESS bias via a
                    # bigger batch (2026-08-05) left the tilt unchanged.
                    return -jnp.mean(jnp.mean(log_p_minus_sel, axis=0))  # mean over g
                logL_bg = _batch_log_L_X(X_bg_flat_for_L, sx_cond).reshape(batch_size, G)
                # Batch drawn ∝ nda·detj ⇒ the ONLY residual per-copy weight is the BFD
                # centroid marginalisation L(X(g)|C_X), self-normalised per g ⇒ the
                # nda·detj·L X-marginal for this C_X.  detj is in the proposal, not here;
                # no is_corr (proposal == nda·detj == objective's static part).
                w_bg = jax.nn.softmax(logL_bg, axis=0)  # (B, G) per-g, cols sum to 1
                return -jnp.mean(jnp.sum(w_bg * log_p_minus_sel, axis=0))  # mean over g

            if _sx_fixed_point:
                # Every stencil point is the same real (log_scale, 0, 0) -- evaluate
                # once, no stencil/lax.map, no key spent (nothing random to draw).
                loss = _loss_for_sx(jnp.array([log_scale_range[0], 0.0, 0.0]))
            else:
                key, k_sx = jr.split(key)
                sx_conds = _sample_sx_conds(k_sx, log_scale_range, e_max)  # (17, 3)
                losses_sx = jax.lax.map(jax.checkpoint(_loss_for_sx), sx_conds)
                loss = jnp.mean(losses_sx)
        else:
            # No C_X ⇒ no L; batch is already ∝ nda·detj ⇒ plain mean NLL over the batch.
            cond_p = jnp.concatenate(
                [g_flat_batch, jnp.zeros((BG, 3), dtype=g_flat_batch.dtype)], axis=-1
            )
            log_p = prior_flow.log_prob(y_std, condition=cond_p).reshape(batch_size, G)
            loss = -jnp.mean(log_p - log_p_sel)

            # ``curvature_weight`` (2026-08-06, EXPERIMENTAL, _use_sx=False path only):
            # direct penalty on d^2[log p(y_fixed; g)]/dg1^2 having the WRONG (positive/
            # convex) sign at FIXED y (only the CONDITIONING g varies) -- this is the
            # inference-relevant quantity (BFD's own R=-d2(logP)/dg2 MLE curvature),
            # NOT the training-stencil quantity log_p[:, col] used for the main loss
            # above (there y ALSO moves with g, via `shear(y_b, g, ...)`, which mixes
            # in the response dy/dg and is a different, already-well-fit object -- an
            # earlier version of this penalty reused those columns and, correctly,
            # had ZERO effect on the broken quantity). Costs 2 extra prior_flow.log_prob
            # calls (cheap vs. the existing G=73-point stencil). Motivated by repeated
            # empirical failure of Sobolev supervision (which only fixes ShearTaylorLast's
            # OWN A,B coefficients, not the WHOLE flow's fixed-y log p(g) curvature --
            # these decouple because log p also depends on the base density's score) to
            # reliably give log p(y_fixed; g) a real peak near the true shear: across
            # seeds/window-reweighting variants, the sign of this curvature was
            # essentially a coin flip. A well-specified P(m|g) must be concave in g at
            # fixed y for BFD's MLE estimator to recover sane shears -- this term
            # supervises that necessary condition directly.
            if curvature_weight > 0.0 and _radii.shape[0] > 0:
                _k = _radii.shape[0] - 1  # widest radius: largest curvature lever arm
                _r = _radii[_k, 0, 0]  # _radii is (K,1,1) -- squeeze to a scalar
                _y_g0 = y_std.reshape(batch_size, G, 4)[:, 0, :]  # fixed (unsheared) data
                _cond_p1 = jnp.concatenate(
                    [jnp.broadcast_to(jnp.array([_r, 0.0]), (batch_size, 2)),
                     jnp.zeros((batch_size, 3))], axis=-1,
                )
                _cond_m1 = jnp.concatenate(
                    [jnp.broadcast_to(jnp.array([-_r, 0.0]), (batch_size, 2)),
                     jnp.zeros((batch_size, 3))], axis=-1,
                )
                _lp_p = prior_flow.log_prob(_y_g0, condition=_cond_p1)
                _lp_m = prior_flow.log_prob(_y_g0, condition=_cond_m1)
                _d2_g1 = (_lp_p - 2.0 * log_p[:, 0] + _lp_m) / (_r * _r)
                loss = loss + curvature_weight * jnp.mean(jax.nn.relu(_d2_g1))

        if _sob_on:
            # Shear-response is now C_X-dependent post shear/Σ_X swap (see
            # _shear_response); evaluating at a single reference sx_ref instead of
            # averaging over the sx_conds draws is a deliberate cheap approximation.
            # 2026-08-05 retry (previously rolled back 2026-07-30, see git/memory
            # history: "_sobolev_row_weights misapplies an SNIS importance weight
            # to a fixed per-template regression target, starving supervision in
            # low-L(X|C_X) [low-flux] regions and letting the (back then, much
            # more poorly identified) implicit density loss fill the gap badly
            # -- caused a large sign-flipping m-bias"). Retrying now that the
            # implicit loss's own B-identifiability is dramatically better
            # (real-only coefficients, wide stencil, bounded loc1, full LR decay
            # -- see that day's investigation): the specific failure mode this
            # was rolled back for should be much less severe as a fallback now,
            # so this is a genuine re-test, not a blind repeat. Empirically
            # motivated by the CONFIRMED low-flux-concentrated bias in the
            # UNWEIGHTED Sobolev target vs the properly nda*detj*L(X|C_X)-
            # weighted truth (shear_derivs_channel_audit.py): e.g. r1's dA/dg1
            # off by 15-35% at low flux, ~0% at high flux -- same footprint as
            # the standard m-tilt's positive (low-flux) lobe.
            row_w = _sobolev_row_weights(X_b, _sob_sx_ref[None, :]) if _use_sx else None
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
    sigmax_layer_kind: str = "autoregressive",
    q_flow_layers: int = q_flow_layers,
    q_nn_width: int = q_nn_width,
    q_nn_depth: int = q_nn_depth,
    min_scale: float = min_scale,
    max_scale: float = max_scale,
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
    sigmax_layer_kind : {"autoregressive", "block", "none"}, optional
        Which SigmaX layer to build -- see
        :func:`~.bijections.new_masked_autoregressive_flow`. Default
        ``"autoregressive"`` (the original :class:`SigmaXCouplingLayer`).
        ``"none"`` omits the layer entirely (``sigmax_cond_dim=None``): only
        correct for a homoscedastic dataset whose training data is already
        weighted at the real (single) Sigma_X.
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
        # "none" disables the SigmaX layer entirely (sigmax_cond_dim=None) --
        # for a homoscedastic dataset (every target shares one Sigma_X) the
        # layer marginalises over nothing; correct only if the TRAINING DATA's
        # own nda*detj*L(X|C_X) weighting already reflects that one real
        # Sigma_X (see imsims.templates.per_copy_weight), not a placeholder.
        sigmax_cond_dim=(None if sigmax_layer_kind == "none" else 5),
        sigmax_nn_width=prior_sigmax_nn_width,
        sigmax_nn_depth=prior_sigmax_nn_depth,
        sigmax_log_scale_mean=prior_sigmax_log_scale_mean,
        sigmax_log_scale_std=prior_sigmax_log_scale_std,
        sigmax_e_max=prior_e_max,
        sigmax_size_loc=prior_size_loc_c1,
        sigmax_layer_kind=sigmax_layer_kind,
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
