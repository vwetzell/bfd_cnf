"""
statistics.py
=============
Pure math / statistics helpers for BFD shear estimation:

  - pqr2g()                    — estimate shear vector from PQR array
  - pqr2multbias()             — compute multiplicative bias from ± PQR
  - clipR()                    — eigenvalue-clip the R matrix
  - bootstrap_total_mult_bias() — bootstrap uncertainty on multiplicative bias

Notation — the per-object ``pqr`` array is PROBABILITY-space ``[P, Q1, Q2, R11,
R22, R12]`` where Q = ∇_g P and R = ∇²_g P (derivatives of P, NOT of log P).
The estimators here form the LOG-MARGINAL quantities
    Q_tot ≡ ∇_g log P     = Q / P
    R_tot ≡ −∇²_g log P   = (Q⊗Q)/P² − R/P
and the ML shear ĝ = (Σ R_tot)⁻¹ (Σ Q_tot) (per-object P-normalised, summed with
EQUAL target weight).  Consequence: any per-object constant factor on [P,Q,R]
(e.g. the BFD target-detj prefactor) cancels in Q_tot/R_tot ⇒ no effect on ĝ.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import bfd

# ---------------------------------------------------------------------------
# pqr2g  (final version from cell 36 — uses P_mask)
# ---------------------------------------------------------------------------


def pqr2g(pqr_arr: jax.Array) -> jax.Array:
    """Estimate the shear vector from a per-object PQR array.

    Computes the maximum-likelihood shear by summing the per-object
    Q (first-derivative) and R (second-derivative) tensors and solving
    the resulting linear system.

    Parameters
    ----------
    pqr_arr : array-like, shape (N, 6)
        Per-object PQR values with columns
        ``[P, Q1, Q2, R11, R22, R12]``.
        Objects with ``P < 1e-10`` are masked out.

    Returns
    -------
    g_est : jax.Array, shape (2,)
        Estimated shear vector ``[g1, g2]``.
    """
    pqr_arr = jnp.asarray(pqr_arr)

    P_mask = pqr_arr[:, 0] >= 1e-10
    pqr_arr = pqr_arr[P_mask]

    R_arr = jnp.array(
        [
            [pqr_arr[:, 3], pqr_arr[:, 5]],
            [pqr_arr[:, 5], pqr_arr[:, 4]],
        ]
    ).T
    Q_tot = jnp.nansum(pqr_arr[:, 1:3] / pqr_arr[:, 0][..., None], axis=0)
    R_tot = jnp.nansum(
        pqr_arr[:, 1:3][..., :, None]
        * pqr_arr[:, 1:3][..., None, :]
        / pqr_arr[:, 0][..., None, None] ** 2
        - R_arr / pqr_arr[:, 0][..., None, None],
        axis=0,
    )

    g_est = jnp.einsum("ij,j->i", jnp.linalg.inv(R_tot), Q_tot)
    return g_est


# ---------------------------------------------------------------------------
# pqr2multbias
# ---------------------------------------------------------------------------


def pqr2multbias(pqr_pm_arr: jax.Array) -> jax.Array:
    """Compute the multiplicative bias from stacked +/- shear PQR arrays.

    Parameters
    ----------
    pqr_pm_arr : array-like, shape (N, 12)
        Concatenated PQR arrays; the first 6 columns are the +shear PQR and
        the last 6 columns are the -shear PQR, each with columns
        ``[P, Q1, Q2, R11, R22, R12]``.

    Returns
    -------
    mult_bias : jax.Array, scalar
        Multiplicative bias ``m = (g1_plus - g1_minus) / delta_g - 1``
        where ``delta_g = 0.04``.
    """
    pqr_pm_arr = jnp.asarray(pqr_pm_arr)
    pqr_arr_p = pqr_pm_arr[:, :6]
    pqr_arr_m = pqr_pm_arr[:, 6:]

    g_p = pqr2g(pqr_arr_p)
    g_m = pqr2g(pqr_arr_m)
    mult_bias = ((g_p[0] - g_m[0]) / 0.04) - 1.0
    return mult_bias


# ---------------------------------------------------------------------------
# clipR
# ---------------------------------------------------------------------------


def clipR(pqr: np.ndarray, lower: float = -100, upper: float = 100) -> np.ndarray:
    """Clip the eigenvalues of the R matrix in each per-object PQR entry.

    Transforms ``(Q, R)`` so that the resulting constraint on the shear has the
    same mode but bounded curvature (inverse uncertainty).  Eigenvalues of R
    outside ``[lower, upper]`` are clipped; Q is adjusted accordingly to
    preserve the probability maximum.

    Parameters
    ----------
    pqr : array-like, shape (N, 6)
        Per-object PQR array as returned by the BFD library.
    lower : float, optional
        Minimum allowed eigenvalue of R.  Default is -100.
    upper : float, optional
        Maximum allowed eigenvalue of R.  Default is 100.

    Returns
    -------
    array-like
        PQR array with clipped R eigenvalues, packed in the same format as the
        input.
    """
    p, q, r = bfd.splitPqr(pqr)
    evals, evecs = np.linalg.eigh(r)
    # Clip both eigenvalues
    eval2 = np.clip(evals, a_min=lower, a_max=upper)
    # Reconstruct r with new eigenvalues
    r = np.einsum("nij, nj, nkj -> nik", evecs, eval2, evecs)
    # Shift q to give same probability max as before
    q = np.einsum("nij, nj, nkj,nk->ni", evecs, eval2 / evals, evecs, q)
    return bfd.packPqr(p, q, r)


# ---------------------------------------------------------------------------
# bootstrap_total_mult_bias
# ---------------------------------------------------------------------------


def bootstrap_total_mult_bias(
    pqr_arr_p: jax.Array,
    pqr_arr_m: jax.Array,
    n_boot: int = 2000,
    delta_g: float = 0.04,
    ridge: float = 1e-10,
    key: jax.Array = jr.PRNGKey(0),
    return_samples: bool = False,
) -> dict[str, Any]:
    """Bootstrap the multiplicative bias and its uncertainty from per-object PQR arrays.

    Parameters
    ----------
    pqr_arr_p : array-like, shape (N, 6)
        PQR array for the +shear catalogue with columns
        ``[P, Q1, Q2, R11, R22, R12]``.
    pqr_arr_m : array-like, shape (N, 6)
        PQR array for the -shear catalogue (same layout as ``pqr_arr_p``).
    n_boot : int, optional
        Number of bootstrap resamples.  Default is 2000.
    delta_g : float, optional
        ``g_plus - g_minus``.  For ±0.02 catalogues use 0.04 (default).
    ridge : float, optional
        Small diagonal regularisation added to the summed R matrix before
        solving for the shear.  Default is ``1e-10``.
    key : jax.Array, optional
        JAX PRNG key used for bootstrap resampling.
    return_samples : bool, optional
        If ``True``, include the full array of bootstrap samples in the
        returned dict.  Default is ``False``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        m_point : jax.Array, scalar
            Full-sample point estimate of the multiplicative bias.
        m_mean : jax.Array, scalar
            Bootstrap mean of the multiplicative bias.
        m_std : jax.Array, scalar
            Bootstrap 1-σ uncertainty (``ddof=1``).
        m_p16, m_p84 : jax.Array, scalar
            16th and 84th bootstrap percentiles.
        n_used : int
            Number of objects after filtering.
        m_boot : jax.Array, shape (n_boot,)
            Bootstrap samples (only present when ``return_samples=True``).
    """

    pqr_arr_p = jnp.asarray(pqr_arr_p)
    pqr_arr_m = jnp.asarray(pqr_arr_m)

    if pqr_arr_p.shape != pqr_arr_m.shape:
        raise ValueError("pqr_arr_p and pqr_arr_m must have the same shape")
    if pqr_arr_p.ndim != 2 or pqr_arr_p.shape[1] != 6:
        raise ValueError("Each PQR array must have shape (N, 6)")

    # SHARED validity mask applied to BOTH arms up front, so the + and - catalogues
    # keep exactly the same matched objects (a proper *paired* bootstrap).
    #
    # The previous version applied the P>=1e-10 cut to each arm separately *inside*
    # pqr_to_qr_totals and never actually applied a shared mask (the `[finite_mask]`
    # slice was commented out), with n_obj left at the unmasked N.  When the two arms
    # dropped different objects (common — thousands of targets have tiny P in only one
    # arm) this (a) misaligned Qp[i] vs Qm[i] so the resample paired *different*
    # galaxies across ±, and (b) indexed [0, N) into the shorter masked arrays, which
    # JAX silently clamps.  Both corrupt the bootstrap (its mean became input-order /
    # pre-filter dependent and biased low).  Masking both arms together fixes it.
    keep = (
        jnp.all(jnp.isfinite(pqr_arr_p), axis=1)
        & jnp.all(jnp.isfinite(pqr_arr_m), axis=1)
        & (pqr_arr_p[:, 0] >= 1e-10)
        & (pqr_arr_m[:, 0] >= 1e-10)
    )
    p = pqr_arr_p[keep]
    m = pqr_arr_m[keep]
    n_obj = p.shape[0]

    if n_obj < 2:
        raise ValueError("Not enough valid objects after filtering")

    def pqr_to_qr_totals(pqr):
        # No per-arm masking here: the shared `keep` mask above already guarantees
        # P>=1e-10 and keeps the two arms paired (same rows, same length = n_obj).
        P = pqr[:, 0]
        Q = pqr[:, 1:3]
        R = jnp.stack(
            [
                jnp.stack([pqr[:, 3], pqr[:, 5]], axis=-1),
                jnp.stack([pqr[:, 5], pqr[:, 4]], axis=-1),
            ],
            axis=-2,
        )  # (N, 2, 2)

        Q_tot = Q / P[:, None]
        R_tot = (
            jnp.einsum("ni,nj->nij", Q, Q) / P[:, None, None] ** 2
            - R / P[:, None, None]
        )
        return Q_tot, R_tot

    Qp, Rp = pqr_to_qr_totals(p)
    Qm, Rm = pqr_to_qr_totals(m)

    eye2 = jnp.eye(2)

    def g_from_totals(Q_tot, R_tot):
        Q_sum = jnp.nansum(Q_tot, axis=0)
        R_sum = jnp.nansum(R_tot, axis=0) + ridge * eye2
        return jnp.linalg.solve(R_sum, Q_sum)

    # Full-sample point estimate
    g_p_point = g_from_totals(Qp, Rp)
    g_m_point = g_from_totals(Qm, Rm)
    m_point = (g_p_point[0] - g_m_point[0]) / delta_g - 1.0

    # Bootstrap using per-iteration keys (no giant idx matrix)
    key, sub = jr.split(key)
    keys = jr.split(sub, n_boot)

    def one_boot(k):
        ii = jr.randint(k, shape=(n_obj,), minval=0, maxval=n_obj)
        g_p = g_from_totals(Qp[ii], Rp[ii])
        g_m = g_from_totals(Qm[ii], Rm[ii])
        return (g_p[0] - g_m[0]) / delta_g - 1.0

    m_boot = jax.lax.map(one_boot, keys)

    out = {
        "m_point": m_point,
        "m_mean": jnp.nanmean(m_boot),
        "m_std": jnp.nanstd(m_boot, ddof=1),
        "m_p16": jnp.nanpercentile(m_boot, 16.0),
        "m_p84": jnp.nanpercentile(m_boot, 84.0),
        "n_used": n_obj,
    }
    if return_samples:
        out["m_boot"] = m_boot
    return out


# ---------------------------------------------------------------------------
# bootstrap_independent_mult_bias
# ---------------------------------------------------------------------------


def bootstrap_independent_mult_bias(
    pqr_arr_p: jax.Array,
    pqr_arr_m: jax.Array,
    n_boot: int = 2000,
    delta_g: float = 0.04,
    ridge: float = 1e-10,
    key: jax.Array = jr.PRNGKey(0),
    return_samples: bool = False,
) -> dict[str, Any]:
    """Bootstrap the multiplicative bias from two *independent* PQR ensembles.

    Unlike :func:`bootstrap_total_mult_bias` (which assumes ``pqr_arr_p`` and
    ``pqr_arr_m`` are the SAME galaxies row-paired, and resamples them jointly for
    shape-noise cancellation), this treats the +shear and -shear catalogues as
    independent samples — different objects, possibly different lengths — and
    resamples each arm independently.  This is the correct estimator when the two
    catalogues are independent injection realisations rather than ring pairs (see
    :func:`bfd_cnf.inference.load_grid_data`).

    The shear of each arm is the aggregate sum-PQR estimate and
    ``m = (g1_p - g1_m) / delta_g - 1``.

    Parameters
    ----------
    pqr_arr_p, pqr_arr_m : array-like, shape (Np, 6) / (Nm, 6)
        Per-object PQR for the + and - catalogues; the two lengths may differ.
    n_boot, delta_g, ridge, key, return_samples
        As in :func:`bootstrap_total_mult_bias`.

    Returns
    -------
    dict
        Same keys as :func:`bootstrap_total_mult_bias`, with ``n_used`` replaced
        by ``n_used_p`` / ``n_used_m``.
    """
    pqr_arr_p = jnp.asarray(pqr_arr_p)
    pqr_arr_m = jnp.asarray(pqr_arr_m)
    if pqr_arr_p.ndim != 2 or pqr_arr_p.shape[1] != 6 or pqr_arr_m.shape[1] != 6:
        raise ValueError("Each PQR array must have shape (N, 6)")

    def _clean(pqr):
        keep = jnp.all(jnp.isfinite(pqr), axis=1) & (pqr[:, 0] >= 1e-10)
        return pqr[keep]

    p = _clean(pqr_arr_p)
    m = _clean(pqr_arr_m)
    n_p, n_m = p.shape[0], m.shape[0]
    if n_p < 2 or n_m < 2:
        raise ValueError("Not enough valid objects after filtering")

    def pqr_to_qr_totals(pqr):
        P = pqr[:, 0]
        Q = pqr[:, 1:3]
        R = jnp.stack(
            [
                jnp.stack([pqr[:, 3], pqr[:, 5]], axis=-1),
                jnp.stack([pqr[:, 5], pqr[:, 4]], axis=-1),
            ],
            axis=-2,
        )
        Q_tot = Q / P[:, None]
        R_tot = (
            jnp.einsum("ni,nj->nij", Q, Q) / P[:, None, None] ** 2
            - R / P[:, None, None]
        )
        return Q_tot, R_tot

    Qp, Rp = pqr_to_qr_totals(p)
    Qm, Rm = pqr_to_qr_totals(m)
    eye2 = jnp.eye(2)

    def g_from_totals(Q_tot, R_tot):
        Q_sum = jnp.nansum(Q_tot, axis=0)
        R_sum = jnp.nansum(R_tot, axis=0) + ridge * eye2
        return jnp.linalg.solve(R_sum, Q_sum)

    g_p_point = g_from_totals(Qp, Rp)
    g_m_point = g_from_totals(Qm, Rm)
    m_point = (g_p_point[0] - g_m_point[0]) / delta_g - 1.0

    key, sub = jr.split(key)
    keys = jr.split(sub, n_boot)

    def one_boot(k):
        kp, km = jr.split(k)
        ip = jr.randint(kp, shape=(n_p,), minval=0, maxval=n_p)
        im = jr.randint(km, shape=(n_m,), minval=0, maxval=n_m)
        g_p = g_from_totals(Qp[ip], Rp[ip])
        g_m = g_from_totals(Qm[im], Rm[im])
        return (g_p[0] - g_m[0]) / delta_g - 1.0

    m_boot = jax.lax.map(one_boot, keys)

    out = {
        "m_point": m_point,
        "m_mean": jnp.nanmean(m_boot),
        "m_std": jnp.nanstd(m_boot, ddof=1),
        "m_p16": jnp.nanpercentile(m_boot, 16.0),
        "m_p84": jnp.nanpercentile(m_boot, 84.0),
        "n_used_p": n_p,
        "n_used_m": n_m,
    }
    if return_samples:
        out["m_boot"] = m_boot
    return out
