"""
inference.py
============
Flow-based PQR inference utilities:

  - make_flow_prob_and_derivs()   — build batched log-prob + shear derivatives
  - prob()                        — probability at a point given shear condition
  - prob_template()               — template-matching probability kernel
  - sample_trunc_mvn_noise_first_lower()  — truncated MVN sampler
  - rqmc_pqr_grid()               — batched RQMC PQR integration over a grid
  - rqmc_integrate_pqr_jax()      — single-object RQMC PQR integrator
  - Template loading and per-object PQR assembly
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import ndtr, ndtri
import equinox as eqx
import jaxopt

from .config import (
    B_RAW_JNP,
    Sigma0,
    GRID_P_PATH,
    GRID_M_PATH,
    prior_sigmax_log_scale_mean,
    log_scale_range,
    e_max,
    target_flux_min,
)
from .models.bijections import (
    _raw2std_jacobian_single_jax,
    propagate_cov_to_std_jax,
)
from .models.flows import cx_to_sx_cond
from .data import (
    augment_moments_raw_jax,
    make_augmentation_noise_raw_jax,
    transform_dataset_to_standard,
)

# ---------------------------------------------------------------------------
# Flow probability and shear derivatives
# ---------------------------------------------------------------------------


def make_flow_prob_and_derivs(
    prior_flow: Any,
    sx_cond: jax.Array | None = None,
) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    """Build a batched function returning log-prob and shear derivatives at g=0.

    Parameters
    ----------
    prior_flow : Transformed
        Trained prior normalizing flow conditioned on ``[g1, g2, log_scale, e1, e2]``.
    sx_cond : jax.Array, shape (3,), optional
        Fixed Σ_X condition ``[log_scale, e1, e2]`` to use when evaluating
        log-prob and its shear derivatives.  Defaults to
        ``[prior_sigmax_log_scale_mean, 0, 0]`` — the reference circular PSF
        at the training prior mean scale.

    Returns
    -------
    log_prob_and_derivs : callable
        A function ``(x_batch: (N, D)) -> tuple`` returning six arrays each of
        shape ``(N,)``:

        ``(log p(x|g=0),``
        ``d log p / dg1,``
        ``d log p / dg2,``
        ``d² log p / dg1²,``
        ``d² log p / dg2²,``
        ``d² log p / dg1 dg2)``

        all evaluated at g = (0, 0).
    """
    if sx_cond is None:
        sx_cond = jnp.array([prior_sigmax_log_scale_mean, 0.0, 0.0])

    def _log_prob(x_single, g1, g2):
        condition = jnp.concatenate([jnp.array([g1, g2]), sx_cond])
        return prior_flow.log_prob(x_single, condition=condition)

    _dlp_dg1 = jax.grad(_log_prob, argnums=1)
    _dlp_dg2 = jax.grad(_log_prob, argnums=2)
    _d2lp_dg1dg1 = jax.grad(_dlp_dg1, argnums=1)
    _d2lp_dg2dg2 = jax.grad(_dlp_dg2, argnums=2)
    _d2lp_dg1dg2 = jax.grad(_dlp_dg1, argnums=2)

    def _single(xi):
        lp = _log_prob(xi, 0.0, 0.0)
        dlp1 = _dlp_dg1(xi, 0.0, 0.0)
        dlp2 = _dlp_dg2(xi, 0.0, 0.0)
        d2lp11 = _d2lp_dg1dg1(xi, 0.0, 0.0)
        d2lp22 = _d2lp_dg2dg2(xi, 0.0, 0.0)
        d2lp12 = _d2lp_dg1dg2(xi, 0.0, 0.0)
        return lp, dlp1, dlp2, d2lp11, d2lp22, d2lp12

    def log_prob_and_derivs(x_batch):
        return jax.vmap(_single)(x_batch)

    return log_prob_and_derivs


# ---------------------------------------------------------------------------
# Simple probability and template-matching functions
# ---------------------------------------------------------------------------


def prob(
    x: jax.Array,
    g1: float,
    g2: float,
    prior_flow: Any,
    log_scale: float = 12.0,
    e1: float = 0.0,
    e2: float = 0.0,
) -> jax.Array:
    """Evaluate the prior flow probability at a point given shear and PSF conditions.

    Parameters
    ----------
    x : jax.Array, shape (N, D) or (D,)
        Standardised moment vector(s) at which to evaluate the probability.
    g1 : float
        First shear component.
    g2 : float
        Second shear component.
    prior_flow : Transformed
        Trained prior normalizing flow.
    log_scale : float, optional
        ``0.5 * log det(Σ_X)`` — PSF noise scale condition.  Default is 12.
    e1 : float, optional
        First PSF ellipticity component.  Default is 0.
    e2 : float, optional
        Second PSF ellipticity component.  Default is 0.

    Returns
    -------
    probs : jax.Array
        Probability values ``exp(log p(x | g, Σ_X))``.
    """
    condition = jnp.array([g1, g2, log_scale, e1, e2], dtype=jnp.float32)
    # Compute the log probabilities
    log_probs = prior_flow.log_prob(x, condition=condition)
    # Convert log probabilities to probabilities
    probs = jnp.exp(log_probs)
    return probs


def prob_template(
    targ: tuple[jax.Array, jax.Array],
    template: jax.Array,
) -> jax.Array:
    """Compute the template-matching probability kernel.

    Evaluates ``exp(-0.5 * (target - template)^T Σ^{-1} (target - template))``
    and returns zero when the chi-squared value exceeds 20.

    Parameters
    ----------
    targ : tuple of jax.Array
        ``(target, target_cov)`` where ``target`` has shape ``(D,)`` and
        ``target_cov`` has shape ``(D, D)``.
    template : jax.Array, shape (D,)
        Template moment vector to match against.

    Returns
    -------
    jax.Array, scalar
        Template-matching probability (zero when ``chisq >= 20``).
    """
    target, target_cov = targ
    dE = target - template
    inv_cov = jnp.linalg.inv(target_cov)
    chisq = jnp.einsum("i,ij,j", dE, inv_cov, dE)
    p = jnp.exp(-0.5 * chisq)
    return jnp.where(chisq < 20.0, p, 0.0)


# ---------------------------------------------------------------------------
# Truncated MVN sampler
# ---------------------------------------------------------------------------


def sample_trunc_mvn_noise_first_lower(
    key: jax.Array,
    cov: jax.Array,
    lower0: jax.Array,
    jitter: float = 1e-9,
) -> jax.Array:
    """Sample ε ~ N(0, Σ) conditioned on ε₀ ≥ lower0 for a batch of covariances.

    Uses the inverse-CDF method to sample the first component from a truncated
    normal, then samples the remaining components from the Gaussian conditional.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    cov : jax.Array, shape (N, 4, 4)
        Per-object noise covariance matrices.
    lower0 : jax.Array, shape (N,)
        Lower bound for the first noise component ``ε₀``.
    jitter : float, optional
        Small diagonal jitter added to the conditional covariance for
        numerical stability.  Default is ``1e-9``.

    Returns
    -------
    jax.Array, shape (N, 4)
        Sampled noise vectors satisfying ``ε₀ ≥ lower0``.
    """
    cov = jnp.asarray(cov)
    lower0 = jnp.asarray(lower0)
    N = cov.shape[0]
    keys = jr.split(key, N)

    def _one(k, S, lo):
        k_u, k_z = jr.split(k, 2)

        s00 = S[0, 0]
        s0r = S[0, 1:]  # (3,)
        sr0 = S[1:, 0]  # (3,)
        Srr = S[1:, 1:]  # (3,3)

        # --- 1) sample ε0 ~ N(0,s00) truncated below lo ---
        sigma0 = jnp.sqrt(jnp.maximum(s00, 1e-12))
        alpha = lo / sigma0
        u_min = ndtr(alpha)  # Φ(alpha)
        u = jr.uniform(k_u, (), minval=u_min, maxval=1.0)
        u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
        eps0 = sigma0 * ndtri(u)

        # --- 2) sample ε_rest | ε0 (Gaussian conditional) ---
        # mean:  E[εr | ε0] = sr0/s00 * ε0
        mu_r = (sr0 / jnp.maximum(s00, 1e-12)) * eps0

        # cov:   Cov(εr | ε0) = Srr - sr0*s0r/s00
        outer = jnp.outer(sr0, s0r) / jnp.maximum(s00, 1e-12)
        C = Srr - outer
        C = 0.5 * (C + C.T) + jitter * jnp.eye(3, dtype=C.dtype)

        L = jnp.linalg.cholesky(C)
        z = jr.normal(k_z, (3,))
        eps_r = mu_r + L @ z

        return jnp.concatenate(
            [jnp.array([eps0], dtype=S.dtype), eps_r.astype(S.dtype)], axis=0
        )

    return jax.vmap(_one)(keys, cov, lower0)


# ---------------------------------------------------------------------------
# RQMC low-discrepancy sampler
# ---------------------------------------------------------------------------


def _halton_sequence(key: jax.Array, n_points: int, dim: int) -> jax.Array:
    """Generate a scrambled Halton low-discrepancy sequence in JAX.

    Builds a base Halton sequence using the first ``dim`` primes, then
    applies a random Cranley-Patterson rotation (shift mod 1) for
    scrambling.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key used for the random shift.
    n_points : int
        Number of quasi-random points to generate.
    dim : int
        Dimensionality of each point.  Must be ≤ 12.

    Returns
    -------
    jax.Array, shape (n_points, dim)
        Scrambled Halton points in ``[0, 1)^dim``.
    """
    # Use small primes as bases
    primes = jnp.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37][:dim])

    def _halton_1d(base, n):
        """Van der Corput sequence in given base."""
        result = jnp.zeros(n)
        f = 1.0 / base
        i_arr = jnp.arange(1, n + 1)

        def body(carry, _):
            res, f_val, i_val = carry
            digit = i_val % base
            res = res + digit * f_val
            i_val = i_val // base
            f_val = f_val / base
            return (res, f_val, i_val), None

        # Unroll enough iterations for the number of points
        max_digits = 30  # enough for 2^30 points
        init = (jnp.zeros(n), jnp.full(n, 1.0 / base), i_arr)
        (result, _, _), _ = jax.lax.scan(body, init, None, length=max_digits)
        return result

    # Build base Halton sequence
    cols = []
    for d in range(dim):
        cols.append(_halton_1d(primes[d], n_points))
    base_seq = jnp.stack(cols, axis=-1)  # (n_points, dim)

    # Scramble: random shift mod 1 (Cranley-Patterson rotation)
    shift = jr.uniform(key, shape=(dim,))
    scrambled = (base_seq + shift[None, :]) % 1.0

    return scrambled


# ---------------------------------------------------------------------------
# Mode-finding helper
# ---------------------------------------------------------------------------


def _find_mode_jax(
    neg_log_integrand: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    x0: jax.Array,
    n_steps: int = 100,
) -> tuple[jax.Array, jax.Array]:
    """Find the mode of a log-density using L-BFGS (JIT-compatible).

    Parameters
    ----------
    neg_log_integrand : callable
        Function ``x -> (neg_log_val, grad)`` returning the negative
        log-integrand and its gradient.
    x0 : jax.Array, shape (D,)
        Initial point for the optimiser.
    n_steps : int, optional
        Maximum number of L-BFGS iterations.  Default is 100.

    Returns
    -------
    best_x : jax.Array, shape (D,)
        Parameter vector at the mode.
    best_val : jax.Array, scalar
        Log-integrand value at the mode.
    """
    solver = jaxopt.LBFGS(
        fun=neg_log_integrand,
        maxiter=n_steps,
        tol=1e-3,
        value_and_grad=True,
    )
    result = solver.run(x0)
    best_x = result.params
    best_val, _ = neg_log_integrand(best_x)
    return best_x, -best_val


def _finite_diff_hessian(
    f: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    x: jax.Array,
    eps: float = 0.05,
) -> jax.Array:
    """Compute the symmetric finite-difference Hessian of a scalar function.

    Uses the standard four-point mixed-derivative formula
    ``H[i,j] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4 eps²)``
    evaluated via a single batched function call for efficiency.

    Parameters
    ----------
    f : callable
        Function ``x -> (value, grad)`` where ``value`` is a scalar.
    x : jax.Array, shape (D,)
        Point at which to evaluate the Hessian.
    eps : float, optional
        Finite-difference step size.  Default is 0.05.

    Returns
    -------
    jax.Array, shape (D, D)
        Symmetrised Hessian matrix.
    """
    dim = x.shape[0]
    idx = jnp.arange(dim)
    ii, jj = jnp.meshgrid(idx, idx, indexing="ij")  # (D,D)
    ii_flat = ii.reshape(-1)  # (D^2,)
    jj_flat = jj.reshape(-1)

    # Build all 4*D^2 perturbed points at once -> (4*D^2, D)
    def make_pts(sign_i, sign_j):
        ei = (
            jnp.zeros((dim * dim, dim))
            .at[jnp.arange(dim * dim), ii_flat]
            .set(sign_i * eps)
        )
        ej = (
            jnp.zeros((dim * dim, dim))
            .at[jnp.arange(dim * dim), jj_flat]
            .set(sign_j * eps)
        )
        return x[None] + ei + ej  # (D^2, D)

    pts_pp = make_pts(+1, +1)
    pts_pm = make_pts(+1, -1)
    pts_mp = make_pts(-1, +1)
    pts_mm = make_pts(-1, -1)
    all_pts = jnp.concatenate([pts_pp, pts_pm, pts_mp, pts_mm], axis=0)  # (4*D^2, D)

    # Single batched flow call instead of 4*D^2 sequential ones
    vals = jax.vmap(lambda xi: f(xi)[0])(all_pts)  # (4*D^2,)
    n = dim * dim
    fpp, fpm, fmp, fmm = vals[:n], vals[n : 2 * n], vals[2 * n : 3 * n], vals[3 * n :]

    H_flat = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
    H = H_flat.reshape(dim, dim)
    return 0.5 * (H + H.T)


# ---------------------------------------------------------------------------
# Jacobian log correction
# ---------------------------------------------------------------------------


def _jacobian_log_correction_std_jax(
    x_samples: jax.Array,
    B_std: jax.Array,
    trace_correction: jax.Array,
    J_bfd_base: jax.Array,
) -> jax.Array:
    """Compute the log BFD Jacobian correction for a batch of standardised samples.

    The correction accounts for the change in the BFD selection Jacobian
    when noise augmentation is applied.

    Parameters
    ----------
    x_samples : jax.Array, shape (N, D)
        Standardised moment samples.
    B_std : jax.Array, shape (D, D)
        BFD B matrix propagated to standardised coordinates.
    trace_correction : jax.Array, scalar
        ``tr(B_RAW @ CA_raw)`` computed during noise augmentation.
    J_bfd_base : jax.Array, scalar
        BFD Jacobian at the template centroid (before augmentation).

    Returns
    -------
    jax.Array, shape (N,)
        Log ratio ``log J_augmented - log J_base`` for each sample.
    """
    J_vals = jnp.einsum("ni,ij,nj->n", x_samples, B_std, x_samples)
    J_augmented = jnp.maximum(J_vals + trace_correction, 1e-30)
    J_base = jnp.maximum(J_bfd_base, 1e-30)
    return jnp.log(J_augmented) - jnp.log(J_base)


# ---------------------------------------------------------------------------
# Single-object RQMC PQR integrator
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 1, 5, 6, 12))
def rqmc_integrate_pqr_jax(
    log_flow_fn,
    flow_prob_and_derivs_fn,
    mu_std,
    cov_std,
    key,
    n_points=2**14,
    n_replicates=32,
    # --- noise augmentation ---
    mu_raw=None,
    CM_raw=None,
    CA_raw=None,
    std_scales=None,
    hessian_scale=1.0,
    adapt_proposal=True,
):
    """Compute PQR and standard errors for a single galaxy via RQMC integration.

    Fully JAX-native and JIT-compatible.  Optionally applies noise augmentation
    to the measurement covariance.  Uses an adaptive Laplace proposal (mode
    finding + finite-difference Hessian) to reduce variance.

    Parameters
    ----------
    log_flow_fn : callable
        JIT-able function ``(x: (1, D),) -> (log_p: (1,),)`` evaluating
        ``log p(x | g=0)``.
    flow_prob_and_derivs_fn : callable
        Batched function returned by :func:`make_flow_prob_and_derivs`.
    mu_std : jax.Array, shape (D,)
        Standardised mean (observed galaxy moments).
    cov_std : jax.Array, shape (D, D)
        Standardised measurement covariance.
    key : jax.Array
        JAX PRNG key.
    n_points : int, optional
        Number of RQMC quadrature points per replicate.  Default is 2¹⁴.
    n_replicates : int, optional
        Number of independent RQMC replicates (used to estimate SE).
        Default is 32.
    mu_raw : jax.Array or None, optional
        Raw template moments; required for noise augmentation.
    CM_raw : jax.Array or None, optional
        Raw measurement covariance; required for noise augmentation.
    CA_raw : jax.Array or None, optional
        Augmentation noise covariance; required for noise augmentation.
    std_scales : jax.Array or None, optional
        Standardisation scales from ``raw2standard.std``; required for
        noise augmentation.
    hessian_scale : float, optional
        Scalar multiplier applied to the Laplace proposal covariance.
        Default is 1.0.
    adapt_proposal : bool, optional
        If ``True`` (default), adapt the proposal to the posterior mode via
        L-BFGS and finite-difference Hessian.

    Returns
    -------
    P_estimate, P_se : jax.Array, scalar
        Estimated P and its standard error.
    Q1_estimate, Q1_se : jax.Array, scalar
        Estimated Q₁ = dP/dg₁ and its standard error.
    Q2_estimate, Q2_se : jax.Array, scalar
        Estimated Q₂ = dP/dg₂ and its standard error.
    R11_estimate, R11_se : jax.Array, scalar
        Estimated R₁₁ = d²P/dg₁² and its standard error.
    R22_estimate, R22_se : jax.Array, scalar
        Estimated R₂₂ = d²P/dg₂² and its standard error.
    R12_estimate, R12_se : jax.Array, scalar
        Estimated R₁₂ = d²P/dg₁dg₂ and its standard error.
    proposal_mu : jax.Array, shape (D,)
        Centre of the Laplace proposal (mode of the integrand).
    x_samples : jax.Array, shape (n_points, D)
        RQMC sample points from the last replicate.
    """
    dim = mu_std.shape[0]

    # ------------------------------------------------------------------
    # Noise augmentation
    # ------------------------------------------------------------------
    augment = (
        (mu_raw is not None)
        and (CM_raw is not None)
        and (CA_raw is not None)
        and (std_scales is not None)
    )

    if augment:
        not_zero = jnp.any(CA_raw != 0.0)
    else:
        not_zero = False

    B_std = jnp.zeros((dim, dim))
    trace_correction = 0.0
    J_bfd_base = 1.0

    if augment:
        C_raw, trace_corr, J_base = augment_moments_raw_jax(mu_raw, CM_raw, CA_raw)
        cov_std_aug = propagate_cov_to_std_jax(C_raw, mu_raw, std_scales)

        A = _raw2std_jacobian_single_jax(mu_raw, std_scales)
        A_inv = jnp.linalg.inv(A)
        B_std_computed = A_inv.T @ B_RAW_JNP @ A_inv

        cov_std = jnp.where(not_zero, cov_std_aug, cov_std)
        B_std = jnp.where(not_zero, B_std_computed, B_std)
        trace_correction = jnp.where(not_zero, trace_corr, 0.0)
        J_bfd_base = jnp.where(not_zero, J_base, 1.0)

    do_augment = augment and True

    # ------------------------------------------------------------------
    # Gaussian factor
    # ------------------------------------------------------------------
    cov_inv = jnp.linalg.inv(cov_std)
    cov_logdet = jnp.linalg.slogdet(cov_std)[1]
    gauss_const = -0.5 * (dim * jnp.log(2.0 * jnp.pi) + cov_logdet)

    def log_gaussian(x):
        diff = x - mu_std
        return gauss_const - 0.5 * jnp.einsum("ij,jk,ik->i", diff, cov_inv, diff)

    def log_gaussian_single(x):
        diff = x - mu_std
        return gauss_const - 0.5 * jnp.dot(diff, cov_inv @ diff)

    # ------------------------------------------------------------------
    # Augmentation correction
    # ------------------------------------------------------------------
    def log_jac_corr_single(x):
        if do_augment:
            J_val = jnp.dot(x, B_std @ x)
            J_augmented = jnp.maximum(J_val + trace_correction, 1e-30)
            J_base_val = jnp.maximum(J_bfd_base, 1e-30)
            return jnp.log(J_augmented) - jnp.log(J_base_val)
        else:
            return 0.0

    def jac_corr_ratio(x):
        if do_augment:
            log_c = _jacobian_log_correction_std_jax(
                x, B_std, trace_correction, J_bfd_base
            )
            return jnp.exp(log_c)
        else:
            return jnp.ones(x.shape[0])

    # ------------------------------------------------------------------
    # Neg-log-integrand for mode finding (uses P only)
    # ------------------------------------------------------------------
    def neg_log_integrand(x):
        def obj(x_):
            return -(
                log_gaussian_single(x_)
                + log_flow_fn(x_[None, :])[0]
                + log_jac_corr_single(x_)
            )

        neg_val, grad = jax.value_and_grad(obj)(x)
        neg_val = jnp.where(jnp.isfinite(neg_val), neg_val, jnp.array(1e10))
        grad = jnp.where(jnp.isfinite(grad), grad, jnp.zeros_like(grad))
        return neg_val, grad

    # ------------------------------------------------------------------
    # Laplace proposal
    # ------------------------------------------------------------------
    if adapt_proposal:
        mode, best_obj = _find_mode_jax(neg_log_integrand, mu_std, n_steps=50)

        hess = _finite_diff_hessian(neg_log_integrand, mode, eps=0.05)
        hess = 0.5 * (hess + hess.T)

        eigvals_h, eigvecs_h = jnp.linalg.eigh(hess)
        eigvals_h = jnp.maximum(eigvals_h, 1e-6)
        proposal_cov = (eigvecs_h / eigvals_h) @ eigvecs_h.T
        proposal_cov = proposal_cov * hessian_scale**2
        proposal_mu = mode
    else:
        proposal_mu = mu_std
        proposal_cov = cov_std

    # ------------------------------------------------------------------
    # Proposal quantities
    # ------------------------------------------------------------------
    L_prop = jnp.linalg.cholesky(proposal_cov)
    prop_inv = jnp.linalg.inv(proposal_cov)
    prop_logdet = jnp.linalg.slogdet(proposal_cov)[1]
    prop_const = -0.5 * (dim * jnp.log(2.0 * jnp.pi) + prop_logdet)

    def log_proposal(x):
        diff = x - proposal_mu
        return prop_const - 0.5 * jnp.einsum("ij,jk,ik->i", diff, prop_inv, diff)

    # ------------------------------------------------------------------
    # RQMC loop – returns (P, Q1, Q2, R11, R22, R12) per replicate
    # ------------------------------------------------------------------
    def single_replicate(key_r):
        u = _halton_sequence(key_r, n_points, dim)
        u = jnp.clip(u, 1e-8, 1.0 - 1e-8)
        z = jax.scipy.stats.norm.ppf(u)
        x = proposal_mu + z @ L_prop.T

        log_gauss = log_gaussian(x)
        jac_corr = jac_corr_ratio(x)

        # Flow probability and all shear derivatives
        lp_flow, dlp_dg1, dlp_dg2, d2lp_dg1dg1, d2lp_dg2dg2, d2lp_dg1dg2 = (
            flow_prob_and_derivs_fn(x)
        )
        p_flow = jnp.exp(lp_flow)

        log_prop = log_proposal(x)
        gauss = jnp.exp(log_gauss)
        prop = jnp.exp(log_prop)
        common = gauss * jac_corr / prop

        P_est = jnp.mean(common * p_flow)
        Q1_est = jnp.mean(common * p_flow * dlp_dg1)
        Q2_est = jnp.mean(common * p_flow * dlp_dg2)
        R11_est = jnp.mean(common * p_flow * (d2lp_dg1dg1 + dlp_dg1 * dlp_dg1))
        R22_est = jnp.mean(common * p_flow * (d2lp_dg2dg2 + dlp_dg2 * dlp_dg2))
        R12_est = jnp.mean(common * p_flow * (d2lp_dg1dg2 + dlp_dg1 * dlp_dg2))

        return P_est, Q1_est, Q2_est, R11_est, R22_est, R12_est, x

    keys = jr.split(key, n_replicates)
    (
        P_ests,
        Q1_ests,
        Q2_ests,
        R11_ests,
        R22_ests,
        R12_ests,
        x_samples_all,
    ) = jax.vmap(
        single_replicate
    )(keys)

    def _mean_se(vals):
        return jnp.mean(vals), jnp.std(vals, ddof=1) / jnp.sqrt(n_replicates)

    P_estimate, P_se = _mean_se(P_ests)
    Q1_estimate, Q1_se = _mean_se(Q1_ests)
    Q2_estimate, Q2_se = _mean_se(Q2_ests)
    R11_estimate, R11_se = _mean_se(R11_ests)
    R22_estimate, R22_se = _mean_se(R22_ests)
    R12_estimate, R12_se = _mean_se(R12_ests)

    x_samples_out = x_samples_all[-1]

    return (
        P_estimate,
        P_se,
        Q1_estimate,
        Q1_se,
        Q2_estimate,
        Q2_se,
        R11_estimate,
        R11_se,
        R22_estimate,
        R22_se,
        R12_estimate,
        R12_se,
        proposal_mu,
        x_samples_out,
    )


# ---------------------------------------------------------------------------
# Batched grid RQMC integrator
# ---------------------------------------------------------------------------


def rqmc_pqr_grid(
    mu_std_all,
    cov_std_all,
    mu_raw_all,
    CM_raw_all,
    sx_conds_all=None,
    n_points=2**10,
    n_replicates=16,
    batch_size=128,
    *,
    raw2standard,
    prior_flow,
):
    """Compute RQMC PQR for a grid of galaxy templates in batches.

    Halton points are generated once and shared across all templates for
    efficiency.  Noise augmentation and PSF conditioning are applied per object.

    Parameters
    ----------
    mu_std_all : jax.Array, shape (N, D)
        Standardised observed moments for all grid objects.
    cov_std_all : jax.Array, shape (N, D, D)
        Standardised measurement covariances.
    mu_raw_all : jax.Array, shape (N, D)
        Raw observed moments (needed for noise augmentation).
    CM_raw_all : jax.Array, shape (N, D, D)
        Raw measurement covariances (needed for noise augmentation).
    sx_conds_all : jax.Array, shape (N, 3) or None
        Per-target ``[log_scale, e1, e2]`` PSF condition vectors computed via
        :func:`~models.flows.cx_to_sx_cond`.  If ``None``, a fixed reference
        circular PSF (``prior_sigmax_log_scale_mean``, e=0) is used for all
        targets — reproducing the old fixed-reference behaviour.
    n_points : int, optional
        RQMC quadrature points per replicate.  Default is 2¹⁰.
    n_replicates : int, optional
        Independent replicates for SE estimation.  Default is 16.
    batch_size : int, optional
        Number of templates processed per ``lax.map`` call.  Default is 128.
    raw2standard : RawMomentStandardize
        Bijection from raw to standardised moment coordinates.
    prior_flow : equinox pytree
        Trained prior normalizing flow.  Passed as a traced (non-static)
        argument so that per-target ``sx_cond`` can be vmapped over it.

    Returns
    -------
    P, P_se : jax.Array, shape (N,)
        Per-object P estimates and standard errors.
    Q1, Q1_se : jax.Array, shape (N,)
        Per-object Q₁ estimates and standard errors.
    Q2, Q2_se : jax.Array, shape (N,)
        Per-object Q₂ estimates and standard errors.
    R11, R11_se : jax.Array, shape (N,)
        Per-object R₁₁ estimates and standard errors.
    R22, R22_se : jax.Array, shape (N,)
        Per-object R₂₂ estimates and standard errors.
    R12, R12_se : jax.Array, shape (N,)
        Per-object R₁₂ estimates and standard errors.
    """
    N = mu_std_all.shape[0]
    dim = mu_std_all.shape[1]

    if sx_conds_all is None:
        sx_conds_all = jnp.broadcast_to(
            jnp.array([prior_sigmax_log_scale_mean, 0.0, 0.0]), (N, 3)
        )

    n_batches = (N + batch_size - 1) // batch_size
    pad = n_batches * batch_size - N

    def pad_front(a):
        pad_cfg = ((0, pad),) + ((0, 0),) * (a.ndim - 1)
        return jnp.pad(a, pad_cfg)

    mu_std_pad = pad_front(mu_std_all)
    cov_std_pad = pad_front(cov_std_all)
    mu_raw_pad = pad_front(mu_raw_all)
    CM_raw_pad = pad_front(CM_raw_all)
    sx_conds_pad = pad_front(sx_conds_all)

    mu_std_b = mu_std_pad.reshape(n_batches, batch_size, dim)
    cov_std_b = cov_std_pad.reshape(n_batches, batch_size, dim, dim)
    mu_raw_b = mu_raw_pad.reshape(n_batches, batch_size, dim)
    CM_raw_b = CM_raw_pad.reshape(n_batches, batch_size, dim, dim)
    sx_conds_b = sx_conds_pad.reshape(n_batches, batch_size, 3)

    # ── Generate ALL Halton points once, outside lax.map ──────────────────────
    # Shape: (n_replicates, n_points, dim)
    # Each replicate gets its own scrambling key, same as before —
    # but now this happens once for the entire grid rather than N_grid times.
    rqmc_keys = jr.split(jr.key(42), n_replicates)
    u_all = jax.vmap(lambda k: _halton_sequence(k, n_points, dim))(rqmc_keys)
    u_all = jnp.clip(u_all, 1e-8, 1.0 - 1e-8)  # clip once here too
    # ──────────────────────────────────────────────────────────────────────────

    def _single(prior_flow, mu_std, cov_std, mu_raw, CM_raw, sx_cond):

        # jax.debug.print(60 * "=")

        current_snr = mu_raw[0] / jnp.sqrt(CM_raw[0, 0])

        CA_raw = make_augmentation_noise_raw_jax(
            CM_raw, current_snr=current_snr, target_snr=20.0
        )

        # CA_raw = jnp.zeros_like(CM_raw)

        dim = mu_std.shape[0]
        not_zero = jnp.any(CA_raw != 0.0)

        std_scales = jnp.asarray(raw2standard.std)
        eye_dim = jnp.eye(dim, dtype=cov_std.dtype)

        def _aug_branch(_):
            C_raw, trace_corr, J_base = augment_moments_raw_jax(mu_raw, CM_raw, CA_raw)
            cov_std_aug = propagate_cov_to_std_jax(C_raw, mu_raw, std_scales)
            cov_std_use = cov_std_aug + 1e-8 * eye_dim

            A = _raw2std_jacobian_single_jax(mu_raw, std_scales)

            # B_std = A^{-T} B_RAW_JNP A^{-1}, without explicit inverse
            tmp = jax.scipy.linalg.solve(A.T, B_RAW_JNP, assume_a="gen")
            B_std = jax.scipy.linalg.solve(A.T, tmp.T, assume_a="gen").T

            return cov_std_use, B_std, trace_corr, J_base

        def _noaug_branch(_):
            return (
                cov_std + 1e-8 * eye_dim,
                jnp.zeros((dim, dim), dtype=cov_std.dtype),
                0.0,
                1.0,
            )

        cov_std_use, B_std, trace_correction, J_bfd_base = jax.lax.cond(
            not_zero, _aug_branch, _noaug_branch, operand=None
        )

        # Gaussian factor (Cholesky form; no explicit inverse)
        L_cov = jnp.linalg.cholesky(cov_std_use)
        cov_logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L_cov)))
        gauss_const = -0.5 * (dim * jnp.log(2.0 * jnp.pi) + cov_logdet)

        def log_gaussian(x):
            # x shape: (N, dim)
            diff = x - mu_std
            v = jax.scipy.linalg.solve_triangular(L_cov, diff.T, lower=True)
            return gauss_const - 0.5 * jnp.sum(v * v, axis=0)

        def log_gaussian_single(x):
            # x shape: (dim,)
            diff = x - mu_std
            v = jax.scipy.linalg.solve_triangular(L_cov, diff, lower=True)
            return gauss_const - 0.5 * jnp.dot(v, v)

        def log_jac_corr_single(x):
            J_val = jnp.dot(x, B_std @ x)
            J_augmented = jnp.maximum(J_val + trace_correction, 1e-30)
            J_base_val = jnp.maximum(J_bfd_base, 1e-30)
            return jnp.where(not_zero, jnp.log(J_augmented) - jnp.log(J_base_val), 0.0)

        def log_jac_corr_batched(x):
            # x: (N, dim)
            J_vals = jnp.einsum("ni,ij,nj->n", x, B_std, x)
            J_augmented = jnp.maximum(J_vals + trace_correction, 1e-30)
            J_base_val = jnp.maximum(J_bfd_base, 1e-30)
            return jnp.where(
                not_zero,
                jnp.log(J_augmented) - jnp.log(J_base_val),
                jnp.zeros(x.shape[0]),
            )

        # ── Per-target flow evaluation closures ───────────────────────────────
        # The 5-dim condition is [g1, g2, log_scale, e1, e2].
        # At g=0 the prior gives log p(z | g=0, Σ_X_target).
        # jax.grad differentiates the log-prob w.r.t. g1 and g2.
        def _log_p_g(x_single, g1, g2):
            cond = jnp.concatenate([jnp.array([g1, g2]), sx_cond])
            return prior_flow.log_prob(x_single, condition=cond)

        _dlp_dg1     = jax.grad(_log_p_g, argnums=1)
        _dlp_dg2     = jax.grad(_log_p_g, argnums=2)
        _d2lp_dg1dg1 = jax.grad(_dlp_dg1, argnums=1)
        _d2lp_dg2dg2 = jax.grad(_dlp_dg2, argnums=2)
        _d2lp_dg1dg2 = jax.grad(_dlp_dg1, argnums=2)

        def flow_prob_and_derivs(x_batch):
            """log p + 5 shear derivatives at g=0 for each row of x_batch."""
            def _one(xi):
                return (
                    _log_p_g(xi, 0.0, 0.0),
                    _dlp_dg1(xi, 0.0, 0.0),
                    _dlp_dg2(xi, 0.0, 0.0),
                    _d2lp_dg1dg1(xi, 0.0, 0.0),
                    _d2lp_dg2dg2(xi, 0.0, 0.0),
                    _d2lp_dg1dg2(xi, 0.0, 0.0),
                )
            return jax.vmap(_one)(x_batch)

        def log_flow_fn(x):
            """log p(x | g=0, Σ_X_target) for a single point; x shape (1, D) → (1,)."""
            cond = jnp.concatenate([jnp.zeros(2), sx_cond])
            return prior_flow.log_prob(x[0], condition=cond)[None]

        # ─────────────────────────────────────────────────────────────────────

        def neg_log_integrand(x):
            def base_fn(x_):
                return -(
                    log_gaussian_single(x_)
                    + log_flow_fn(x_[None, :])[0]
                    + log_jac_corr_single(x_)
                )

            neg_val, grad = jax.value_and_grad(base_fn)(x)
            neg_val = jnp.where(jnp.isfinite(neg_val), neg_val, jnp.array(1e10))
            grad = jnp.where(jnp.isfinite(grad), grad, jnp.zeros_like(grad))
            return neg_val, grad

        # Laplace proposal
        mode, _ = _find_mode_jax(neg_log_integrand, mu_std, n_steps=20)

        # jax.debug.print("Target mu (std space): {m}", m=mu_std)
        # jax.debug.print("Mode found at: {m}", m=mode)

        mode = jnp.where(jnp.isfinite(mode), mode, mu_std)
        hess = _finite_diff_hessian(neg_log_integrand, mode, eps=0.05)
        hess = 0.5 * (hess + hess.T)
        eigvals_h, eigvecs_h = jnp.linalg.eigh(hess)
        eigvals_h = jnp.maximum(eigvals_h, 1e-6)
        proposal_cov = (eigvecs_h / eigvals_h) @ eigvecs_h.T
        hessian_scale = 3.0
        proposal_cov = proposal_cov * hessian_scale**2
        proposal_cov = proposal_cov + 1e-6 * jnp.eye(dim)
        proposal_mu = mode

        # proposal_mu = mu_std
        # proposal_cov = cov_std_use

        # jax.debug.print("Proposal std: {std}", std=jnp.sqrt(jnp.diag(proposal_cov)))

        L_prop = jnp.linalg.cholesky(proposal_cov)
        prop_logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L_prop)))
        prop_const = -0.5 * (dim * jnp.log(2.0 * jnp.pi) + prop_logdet)

        def log_proposal(x):
            diff = x - proposal_mu
            v = jax.scipy.linalg.solve_triangular(L_prop, diff.T, lower=True)
            return prop_const - 0.5 * jnp.sum(v**2, axis=0)

        # single_replicate now receives pre-clipped u instead of a key
        def single_replicate(u):  # u: (n_points, dim)
            z = jax.scipy.stats.norm.ppf(u)
            x = proposal_mu + z @ L_prop.T

            log_gauss = log_gaussian(x)
            log_jc = log_jac_corr_batched(x)
            lp_flow, dlp1, dlp2, d2lp11, d2lp22, d2lp12 = flow_prob_and_derivs(x)
            p_flow = jnp.exp(lp_flow)
            log_prop = log_proposal(x)

            log_w = log_gauss + log_jc - log_prop
            log_w = jnp.where(
                jnp.isfinite(log_w), log_w, jnp.full_like(log_w, -jnp.inf)
            )
            w_raw = jnp.exp(log_w)
            w_raw = jnp.where(jnp.isfinite(w_raw), w_raw, jnp.zeros_like(w_raw))

            return (
                jnp.nanmean(w_raw * p_flow),
                jnp.nanmean(w_raw * p_flow * dlp1),
                jnp.nanmean(w_raw * p_flow * dlp2),
                jnp.nanmean(w_raw * p_flow * (d2lp11 + dlp1 * dlp1)),
                jnp.nanmean(w_raw * p_flow * (d2lp22 + dlp2 * dlp2)),
                jnp.nanmean(w_raw * p_flow * (d2lp12 + dlp1 * dlp2)),
            )

        # Process replicates sequentially with lax.map to avoid materialising
        # a (batch_size, n_replicates, n_points, dim) tensor through the flow.
        # Each map step allocates (batch_size, n_points, dim) intermediates instead.
        stacked = jax.lax.map(single_replicate, u_all)
        P_ests, Q1_ests, Q2_ests, R11_ests, R22_ests, R12_ests = stacked

        def _mean_se(vals):
            return jnp.nanmean(vals), jnp.nanstd(vals, ddof=1) / jnp.sqrt(n_replicates)

        P, P_se = _mean_se(P_ests)
        Q1, Q1_se = _mean_se(Q1_ests)
        Q2, Q2_se = _mean_se(Q2_ests)
        R11, R11_se = _mean_se(R11_ests)
        R22, R22_se = _mean_se(R22_ests)
        R12, R12_se = _mean_se(R12_ests)

        return P, P_se, Q1, Q1_se, Q2, Q2_se, R11, R11_se, R22, R22_se, R12, R12_se

    # eqx.filter_jit handles the non-array leaves in prior_flow (e.g. triangular.fn)
    # by treating them as static cache keys rather than traced arrays.
    _single_batch = eqx.filter_jit(
        eqx.filter_vmap(_single, in_axes=(None, 0, 0, 0, 0, 0))
    )

    results = []
    for i in range(n_batches):
        result = _single_batch(
            prior_flow,
            mu_std_b[i], cov_std_b[i], mu_raw_b[i], CM_raw_b[i], sx_conds_b[i],
        )
        results.append(result)
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *results)

    # Flatten batches and remove padding
    out = jax.tree.map(lambda x: x.reshape(-1)[:N], batched)
    return out


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


def load_grid_data(
    grid_p_path: str = GRID_P_PATH,
    grid_m_path: str = GRID_M_PATH,
) -> np.ndarray:
    """Load the +/- shear galaxy grid files and join them on the object ID.

    Parameters
    ----------
    grid_p_path : str, optional
        Path to the +shear numpy structured array file.  Defaults to
        ``config.GRID_P_PATH``.
    grid_m_path : str, optional
        Path to the -shear numpy structured array file.  Defaults to
        ``config.GRID_M_PATH``.

    Returns
    -------
    numpy.ndarray (structured)
        Joined structured array keyed on ``'id'`` with ``_p`` / ``_m`` field
        suffixes for the + and - shear catalogues respectively.
    """
    from numpy.lib.recfunctions import join_by

    grid_gal_p = np.load(grid_p_path)
    grid_gal_m = np.load(grid_m_path)

    joined_grid = join_by(
        "id", grid_gal_p, grid_gal_m, usemask=False, r1postfix="_p", r2postfix="_m"
    )
    return joined_grid


def assemble_pqr_from_flow(
    P_p: jax.Array,
    Q1_p: jax.Array,
    Q2_p: jax.Array,
    R11_p: jax.Array,
    R22_p: jax.Array,
    R12_p: jax.Array,
    P_m: jax.Array,
    Q1_m: jax.Array,
    Q2_m: jax.Array,
    R11_m: jax.Array,
    R22_m: jax.Array,
    R12_m: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Stack per-object PQR scalar arrays into ``(N, 6)`` matrices.

    Parameters
    ----------
    P_p, Q1_p, Q2_p, R11_p, R22_p, R12_p : jax.Array, shape (N,)
        PQR components for the +shear catalogue.
    P_m, Q1_m, Q2_m, R11_m, R22_m, R12_m : jax.Array, shape (N,)
        PQR components for the -shear catalogue.

    Returns
    -------
    pqr_arr_p : jax.Array, shape (N, 6)
        Stacked PQR array for +shear with columns
        ``[P, Q1, Q2, R11, R22, R12]``.
    pqr_arr_m : jax.Array, shape (N, 6)
        Stacked PQR array for -shear (same layout).
    """
    pqr_arr_p = jnp.stack([P_p, Q1_p, Q2_p, R11_p, R22_p, R12_p], axis=-1)
    pqr_arr_m = jnp.stack([P_m, Q1_m, Q2_m, R11_m, R22_m, R12_m], axis=-1)
    return pqr_arr_p, pqr_arr_m


# ---------------------------------------------------------------------------
# Grid covariance unpacking + end-to-end grid integration
# ---------------------------------------------------------------------------


def unpack_packed_cov(
    packed: np.ndarray, n: int = 5, keep: int = 4
) -> np.ndarray:
    """Unpack BFD packed upper-triangular covariance rows to dense sub-blocks.

    The grid ``covariance`` field stores each object's symmetric ``n×n`` moment
    covariance as the ``n*(n+1)//2`` upper-triangular entries in row-major order
    (the layout produced by ``bfd.MomentCovariance``).  This expands them to a
    dense ``(N, keep, keep)`` array, keeping the leading ``keep×keep`` sub-block
    (``keep=4`` drops the 5th moment Mc).

    Parameters
    ----------
    packed : numpy.ndarray, shape (N, n*(n+1)//2)
        Packed covariance rows.
    n : int, optional
        Dimension of the packed symmetric matrix.  Default 5.
    keep : int, optional
        Size of the leading sub-block to return.  Default 4.

    Returns
    -------
    numpy.ndarray, shape (N, keep, keep)
        Dense symmetric covariance sub-blocks.
    """
    packed = np.asarray(packed)
    N = packed.shape[0]
    C = np.zeros((N, n, n), dtype=float)
    j = 0
    for i in range(n):
        nvals = n - i
        C[:, i, i:] = packed[:, j : j + nvals]
        C[:, i:, i] = packed[:, j : j + nvals]
        j += nvals
    return C[:, :keep, :keep]


def integrate_grid_pqr(
    joined_grid: np.ndarray,
    raw2standard: Any,
    prior_flow: Any,
    *,
    key: jax.Array,
    n_targets: int | None = None,
    flux_min: float = target_flux_min,
    flux_max: float = 90000.0,
    mr_mf_lo: float = 2.2,
    mr_mf_hi: float = 3.5,
    n_points: int = 2**10,
    n_replicates: int = 16,
    batch_size: int = 512,
    fixed_sx_cond: tuple[float, float, float] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run flow-based RQMC PQR over the ±shear simulation-grid targets.

    Reproduces the grid-integration step of the BFD cNF workflow:

    1. Apply the moment selection cuts (``flux_min < Mf < flux_max`` and
       ``mr_mf_lo < Mr/Mf < mr_mf_hi``) plus a PQR-positivity mask
       (``P_sim >= 1e-10`` on both shear sides).
    2. Optionally subsample to ``n_targets`` objects.
    3. Derive a per-target Σ_X condition from the 2×2 ``[2:4, 2:4]`` sub-block of
       each object's moment covariance (the ``[M+, Mx]`` noise block) via
       :func:`~bfd_cnf.models.flows.cx_to_sx_cond` → ``[log_scale, e1, e2]``.
    4. Integrate both shear sides with :func:`rqmc_pqr_grid`.

    Selection, masking and subsampling are applied with a single shared index so
    the returned ``_p`` / ``_m`` arrays are row-aligned.

    Parameters
    ----------
    joined_grid : numpy.ndarray (structured)
        Output of :func:`load_grid_data` (``_p`` / ``_m`` suffixed fields and a
        shared ``id``).
    raw2standard : RawMomentStandardize
        Raw→standardised bijection; must match the one the flow was trained with.
    prior_flow : equinox pytree
        Trained prior flow conditioned on ``[g1, g2, log_scale, e1, e2]``.
    key : jax.Array
        JAX PRNG key for the optional target subsample.
    n_targets : int or None, optional
        Subsample this many selected targets (without replacement).  ``None``
        (default) keeps all selected targets.
    flux_min, flux_max : float, optional
        ``Mf`` selection bounds.  Defaults ``config.target_flux_min`` and 90000.
    mr_mf_lo, mr_mf_hi : float, optional
        ``Mr/Mf`` selection bounds.  Defaults 2.2 and 3.5.
    n_points, n_replicates, batch_size : int, optional
        RQMC controls forwarded to :func:`rqmc_pqr_grid`.
    fixed_sx_cond : tuple of float or None, optional
        If given, use this single ``(log_scale, e1, e2)`` Σ_X condition for
        **every** target instead of deriving a per-target Σ_X from the
        covariance block.  Useful as a diagnostic — e.g. ``(13.0, 0.0, 0.0)``
        keeps every target inside the trained ``log_scale`` range.  The
        per-target Σ_X correspondence self-check is skipped in this mode.
    verbose : bool, optional
        Print per-target ``log_scale`` / ``|e|`` ranges against the training
        config (``config.log_scale_range`` / ``config.e_max``), including a
        count of targets whose ``log_scale`` falls outside the trained range.

    Returns
    -------
    dict
        ``ids`` (N,); ``targets_p`` / ``targets_m`` (N, 4) raw moments;
        ``sx_conds_p`` / ``sx_conds_m`` (N, 3); ``pqr_p`` / ``pqr_m`` (N, 6) flow
        PQR in ``[P, Q1, Q2, R11, R22, R12]`` order; ``pqr_sim_p`` /
        ``pqr_sim_m`` (N, 6) the simulation's analytic BFD PQR in the **same**
        column order; ``components_p`` / ``components_m`` — the 12-tuple
        ``(P, P_se, Q1, Q1_se, Q2, Q2_se, R11, R11_se, R22, R22_se, R12, R12_se)``
        returned by :func:`rqmc_pqr_grid`.
    """
    import bfd

    # ── moments + selection (cuts use the +shear catalogue, like the notebook) ─
    mom_p = np.asarray(joined_grid["moments_p"][:, :4])
    mom_m = np.asarray(joined_grid["moments_m"][:, :4])

    mf = mom_p[:, 0]
    mr_mf = mom_p[:, 1] / mom_p[:, 0]
    sel = (mf > flux_min) & (mf < flux_max) & (mr_mf > mr_mf_lo) & (mr_mf < mr_mf_hi)

    # Simulation's own analytic PQR (magnification terms stripped) →
    # BFD column order [P, Q1, Q2, R11, R12, R22].
    pqr_sim_p_all = np.asarray(bfd.stripMuPqr(joined_grid["pqr_p"]))
    pqr_sim_m_all = np.asarray(bfd.stripMuPqr(joined_grid["pqr_m"]))
    sel = sel & (pqr_sim_p_all[:, 0] >= 1e-10) & (pqr_sim_m_all[:, 0] >= 1e-10)

    sel_idx = np.where(sel)[0]

    # ── optional subsample (shared across both shear sides) ───────────────────
    if n_targets is not None and n_targets < sel_idx.shape[0]:
        sub = np.asarray(
            jr.choice(key, sel_idx.shape[0], shape=(n_targets,), replace=False)
        )
        sel_idx = np.sort(sel_idx[sub])

    ids = np.asarray(joined_grid["id"][sel_idx])
    targets_p = jnp.asarray(mom_p[sel_idx])
    targets_m = jnp.asarray(mom_m[sel_idx])

    # Reorder BFD [P,Q1,Q2,R11,R12,R22] → flow [P,Q1,Q2,R11,R22,R12].
    _to_flow = [0, 1, 2, 3, 5, 4]
    pqr_sim_p = jnp.asarray(pqr_sim_p_all[sel_idx][:, _to_flow])
    pqr_sim_m = jnp.asarray(pqr_sim_m_all[sel_idx][:, _to_flow])

    # ── covariance blocks (4×4 measurement covariance per object) ─────────────
    CM_raw_p = jnp.asarray(unpack_packed_cov(joined_grid["covariance_p"][sel_idx]))
    CM_raw_m = jnp.asarray(unpack_packed_cov(joined_grid["covariance_m"][sel_idx]))

    # ── standardise ───────────────────────────────────────────────────────────
    mu_std_p, sigma_std_p = transform_dataset_to_standard(
        raw2standard, targets_p, CM_raw_p
    )
    mu_std_m, sigma_std_m = transform_dataset_to_standard(
        raw2standard, targets_m, CM_raw_m
    )

    # ── per-target Σ_X from the [M+, Mx] (index 2,3) covariance sub-block ──────
    # (or a single fixed Σ_X for every target, when fixed_sx_cond is given).
    if fixed_sx_cond is not None:
        _fix = jnp.asarray(fixed_sx_cond, dtype=jnp.float32)
        sx_conds_p = jnp.broadcast_to(_fix, (targets_p.shape[0], 3))
        sx_conds_m = jnp.broadcast_to(_fix, (targets_m.shape[0], 3))
    else:
        sx_conds_p = jax.vmap(cx_to_sx_cond)(CM_raw_p[:, 2:4, 2:4])  # (N, 3)
        sx_conds_m = jax.vmap(cx_to_sx_cond)(CM_raw_m[:, 2:4, 2:4])

    # ── ±shear pairing integrity ──────────────────────────────────────────────
    # Shape-noise cancellation requires the +shear and -shear targets to be the
    # SAME galaxy.  load_grid_data joins on 'id', so each joined row pairs the
    # +/- versions, and the single shared sel_idx above keeps
    # targets_p[i] <-> targets_m[i] id-matched.  Unique ids => the join is
    # strictly 1:1 (no cartesian-product duplicates from join_by).
    if np.unique(ids).size != ids.size:
        raise ValueError(
            "Duplicate ids among selected targets: +/- pairing is not 1:1, so "
            "shape-noise cancellation would be invalid."
        )

    # In per-target mode, independently confirm each Σ_X corresponds to that
    # target's own covariance block, by looking the id back up in joined_grid
    # (rather than reusing sel_idx) — a guard against any index misalignment.
    if fixed_sx_cond is None:
        grid_ids = np.asarray(joined_grid["id"])
        _sample = np.random.default_rng(0).choice(
            ids.shape[0], size=int(min(8, ids.shape[0])), replace=False
        )
        for k in _sample:
            row = int(np.flatnonzero(grid_ids == ids[k])[0])
            for side, packed_field, sx in (
                ("p", "covariance_p", sx_conds_p),
                ("m", "covariance_m", sx_conds_m),
            ):
                blk = unpack_packed_cov(
                    joined_grid[packed_field][row][None]
                )[0, 2:4, 2:4]
                expect = np.asarray(cx_to_sx_cond(jnp.asarray(blk)))
                got = np.asarray(sx[k])
                if not np.allclose(expect, got, atol=1e-4, rtol=1e-4):
                    raise AssertionError(
                        f"Sigma_X mismatch (side {side}) at selected row {k} "
                        f"(id {int(ids[k])}): expected {expect}, got {got}"
                    )

    if verbose:
        if fixed_sx_cond is None:
            print(
                f"Selected {sel_idx.shape[0]} grid targets "
                f"(+/- id-matched 1:1; per-target Sigma_X, cross-check passed)."
            )
        else:
            print(
                f"Selected {sel_idx.shape[0]} grid targets (+/- id-matched 1:1).  "
                f"FIXED Sigma_X for all targets: log_scale={float(_fix[0]):.2f}, "
                f"e=({float(_fix[1]):.3f}, {float(_fix[2]):.3f})."
            )
        for lbl, sx in (("+shear", sx_conds_p), ("-shear", sx_conds_m)):
            ls = sx[:, 0]
            em = jnp.hypot(sx[:, 1], sx[:, 2])
            n_oor = int(
                jnp.sum((ls < log_scale_range[0]) | (ls > log_scale_range[1]))
            )
            print(
                f"  [{lbl}] log_scale [{float(ls.min()):.2f}, {float(ls.max()):.2f}] "
                f"(train {log_scale_range}; {n_oor} out-of-range)   "
                f"|e| [{float(em.min()):.4f}, {float(em.max()):.4f}] (e_max {e_max})"
            )

    # ── integrate both shear sides ────────────────────────────────────────────
    components_p = rqmc_pqr_grid(
        mu_std_p, sigma_std_p, targets_p, CM_raw_p, sx_conds_p,
        n_points=n_points, n_replicates=n_replicates, batch_size=batch_size,
        raw2standard=raw2standard, prior_flow=prior_flow,
    )
    components_m = rqmc_pqr_grid(
        mu_std_m, sigma_std_m, targets_m, CM_raw_m, sx_conds_m,
        n_points=n_points, n_replicates=n_replicates, batch_size=batch_size,
        raw2standard=raw2standard, prior_flow=prior_flow,
    )

    (P_p, _, Q1_p, _, Q2_p, _, R11_p, _, R22_p, _, R12_p, _) = components_p
    (P_m, _, Q1_m, _, Q2_m, _, R11_m, _, R22_m, _, R12_m, _) = components_m
    pqr_p, pqr_m = assemble_pqr_from_flow(
        P_p, Q1_p, Q2_p, R11_p, R22_p, R12_p,
        P_m, Q1_m, Q2_m, R11_m, R22_m, R12_m,
    )

    return dict(
        ids=ids,
        targets_p=targets_p,
        targets_m=targets_m,
        sx_conds_p=sx_conds_p,
        sx_conds_m=sx_conds_m,
        pqr_p=pqr_p,
        pqr_m=pqr_m,
        pqr_sim_p=pqr_sim_p,
        pqr_sim_m=pqr_sim_m,
        components_p=components_p,
        components_m=components_m,
    )
