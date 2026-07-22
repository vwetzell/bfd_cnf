"""
statistics.py
=============
Pure math / statistics helpers for BFD shear estimation:

  - to_bfd() / from_bfd()      — flow<->bfd PQR column-order interop
  - split_qr()                 — P, Q (N,2), R (N,2,2) via bfd.splitPqr
  - qr_log_totals()            — per-object log-space Q_tot, R_tot via bfd.logPqr
  - pqr2g()                    — estimate shear vector from PQR array
  - pqr2multbias()             — compute multiplicative bias from ± PQR
  - clipR()                    — eigenvalue-clip the R matrix
  - bootstrap_independent_mult_bias() — bootstrap uncertainty on multiplicative
    bias from two independent (unpaired, possibly unequal-length) PQR ensembles

All PQR unpacking / shear math routes through the ``bfd`` package so there is a
SINGLE definition of the packing convention (see ``to_bfd``).

Notation — the per-object ``pqr`` array is PROBABILITY-space ``[P, Q1, Q2, R11,
R22, R12]`` where Q = ∇_g P and R = ∇²_g P (derivatives of P, NOT of log P).
The estimators here form the LOG-MARGINAL quantities
    Q_tot ≡ ∇_g log P     = Q / P
    R_tot ≡ −∇²_g log P   = (Q⊗Q)/P² − R/P
and the ML shear ĝ = (Σ R_tot)⁻¹ (Σ Q_tot) (per-object P-normalised, summed with
EQUAL target weight).  Consequence: any per-object constant factor on [P,Q,R]
(e.g. the BFD target-detj prefactor) cancels in Q_tot/R_tot ⇒ no effect on ĝ.
This is exactly ``bfd.meanShear(Σ bfd.logPqr(pqr))``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import bfd

# ---------------------------------------------------------------------------
# BFD PQR interop  (single source of truth for the column convention)
# ---------------------------------------------------------------------------
# The flow catalogs pack the R block as [R11, R22, R12]; bfd.pqr (PqrNoMu) uses
# the canonical [R11, R12, R22].  This permutation swaps those two columns and
# is its OWN INVERSE, so one call maps flow->bfd AND bfd->flow.
_FLOW_BFD = [0, 1, 2, 3, 5, 4]


def to_bfd(pqr: np.ndarray) -> np.ndarray:
    """Reorder flow PQR ``[...,6]=[P,Q1,Q2,R11,R22,R12]`` to bfd's
    ``[P,Q1,Q2,R11,R12,R22]`` (involution: the same call maps bfd->flow)."""
    return np.asarray(pqr)[..., _FLOW_BFD]


from_bfd = to_bfd  # the permutation is its own inverse


def split_qr(pqr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``P (...,)``, ``Q (...,2)``, ``R (...,2,2)`` from flow-order PQR, via
    :func:`bfd.splitPqr` (so the R block is assembled once, correctly)."""
    return bfd.splitPqr(to_bfd(pqr))


def qr_log_totals(pqr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-object log-space accumulation terms used to form the ML shear:
    ``Q_tot = Q/P`` (...,2) and ``R_tot = Q Qᵀ/P² − R/P`` (...,2,2), from
    :func:`bfd.logPqr`.  Rows with ``P<=0`` come back as zeros (as in bfd)."""
    _, q, r = bfd.splitPqr(bfd.logPqr(to_bfd(np.asarray(pqr))))
    # bfd.logPqr stores ∇²log P = R/P − Q Qᵀ/P² = −R_tot; negate to get R_tot.
    return q, -r


# ---------------------------------------------------------------------------
# pqr2g
# ---------------------------------------------------------------------------


def pqr2g(pqr_arr: np.ndarray) -> np.ndarray:
    """Estimate the shear vector from a per-object PQR array.

    Computes the maximum-likelihood shear by summing the per-object log-space
    Q and R tensors (``bfd.logPqr``) and solving the resulting linear system
    (``bfd.meanShear``).

    Parameters
    ----------
    pqr_arr : array-like, shape (N, 6)
        Per-object PQR values with columns
        ``[P, Q1, Q2, R11, R22, R12]``.
        Objects with ``P < 1e-10`` are masked out.

    Returns
    -------
    g_est : np.ndarray, shape (2,)
        Estimated shear vector ``[g1, g2]``.
    """
    pqr_arr = np.asarray(pqr_arr)
    keep = pqr_arr[:, 0] >= 1e-10
    total = bfd.logPqr(to_bfd(pqr_arr[keep])).sum(axis=0)
    return bfd.meanShear(total)[0]  # xmax = (Σ R_tot)⁻¹ (Σ Q_tot)


# ---------------------------------------------------------------------------
# Selection correction (apply to an already-integrated catalogue)
# ---------------------------------------------------------------------------


def net_qr_totals(
    pqr_arm: np.ndarray,
    sel_qtot: np.ndarray | None = None,
    sel_rtot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-object log-totals with the BFD selection correction subtracted per object.

    ``Q_tot, R_tot = qr_log_totals(pqr_arm)`` then, if given, subtract the *per-object*
    selection log-totals ``sel_qtot`` (N,2) / ``sel_rtot`` (N,2,2) — each galaxy's own
    ``σ_f``-bin selection response (gather ``q_tot_b[bin_idx]`` / ``r_tot_b[bin_idx]``
    from :func:`bfd_cnf.inference.selection_pqr_binned`).  Attaching the correction
    per object (rather than as one fixed sum) is what makes the bootstrap correct: a
    resample moves the measured and selection sums together.
    """
    q, r = qr_log_totals(pqr_arm)
    if sel_qtot is not None:
        q = q - np.asarray(sel_qtot)
        r = r - np.asarray(sel_rtot)
    return q, r


def apply_selection(
    pqr_arm: np.ndarray,
    sel_qtot: np.ndarray | None = None,
    sel_rtot: np.ndarray | None = None,
) -> np.ndarray:
    """ML shear from an integrated arm's PQR with the selection correction applied.

    ``ĝ = (Σ_i R_tot,i − Σ_i R_tot,sel,i)⁻¹ (Σ_i Q_tot,i − Σ_i Q_tot,sel,i)``.  With
    ``sel_*=None`` this reduces exactly to :func:`pqr2g`.  ``sel_qtot``/``sel_rtot`` are
    row-aligned to ``pqr_arm`` (filtered here to the same ``P>=1e-10`` rows as
    :func:`pqr2g`).  Apply per arm; mult-bias is ``(ĝ_p[0]−ĝ_m[0])/Δg − 1``.
    """
    pqr_arm = np.asarray(pqr_arm)
    keep = pqr_arm[:, 0] >= 1e-10
    q, r = net_qr_totals(
        pqr_arm[keep],
        None if sel_qtot is None else np.asarray(sel_qtot)[keep],
        None if sel_rtot is None else np.asarray(sel_rtot)[keep],
    )
    return np.linalg.solve(r.sum(axis=0), q.sum(axis=0))


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
    p, q, r = split_qr(pqr)  # flow-order in; bfd handles the R assembly
    evals, evecs = np.linalg.eigh(r)
    # Clip both eigenvalues
    eval2 = np.clip(evals, a_min=lower, a_max=upper)
    # Reconstruct r with new eigenvalues
    r = np.einsum("nij, nj, nkj -> nik", evecs, eval2, evecs)
    # Shift q to give same probability max as before
    q = np.einsum("nij, nj, nkj,nk->ni", evecs, eval2 / evals, evecs, q)
    return from_bfd(bfd.packPqr(p, q, r))  # repack, then bfd-order -> flow-order


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
    sel_qtot_p: jax.Array | None = None,
    sel_rtot_p: jax.Array | None = None,
    sel_qtot_m: jax.Array | None = None,
    sel_rtot_m: jax.Array | None = None,
) -> dict[str, Any]:
    """Bootstrap the multiplicative bias from two *independent* PQR ensembles.

    The +shear and -shear grid catalogues are independent injection realisations
    over the same footprint — different objects, possibly different lengths, with
    only a small fraction landing at coincident sky positions — so each arm is
    resampled independently rather than jointly.  The shear of each arm is the
    aggregate sum-PQR estimate and ``m = (g1_p - g1_m) / delta_g - 1``.

    Parameters
    ----------
    pqr_arr_p, pqr_arr_m : array-like, shape (Np, 6) / (Nm, 6)
        Per-object PQR for the + and - catalogues; the two lengths may differ.
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
    sel_qtot_p, sel_rtot_p, sel_qtot_m, sel_rtot_m : array-like or None, optional
        Per-object BFD selection log-totals (``(Np,2)``/``(Np,2,2)`` etc.), row-aligned
        to ``pqr_arr_p``/``pqr_arr_m`` *before* filtering, gathered from
        :func:`bfd_cnf.inference.selection_pqr_binned` by each galaxy's σ_f bin
        (``q_tot_b[bin_idx]`` / ``r_tot_b[bin_idx]``).  Subtracted per object so the
        resample captures the measured↔selection correlation.  ``None`` (default) ⇒ no
        correction (unchanged behaviour).

    Returns
    -------
    dict
        ``m_point``, ``m_mean``, ``m_std``, ``m_p16``, ``m_p84`` (as in
        :func:`pqr2multbias`'s bootstrap); the additive bias ``c1_point``/``c1_std``
        (``=(g1_p+g1_m)/2``) and ``c2_point``/``c2_std`` (``=(g2_p+g2_m)/2``); plus
        ``n_used_p`` / ``n_used_m`` (object counts after filtering, one per arm), and
        ``m_boot`` / ``c1_boot`` / ``c2_boot`` when ``return_samples=True``.
    """
    pqr_arr_p = jnp.asarray(pqr_arr_p)
    pqr_arr_m = jnp.asarray(pqr_arr_m)
    if pqr_arr_p.ndim != 2 or pqr_arr_p.shape[1] != 6 or pqr_arr_m.shape[1] != 6:
        raise ValueError("Each PQR array must have shape (N, 6)")

    def _clean(pqr):
        keep = jnp.all(jnp.isfinite(pqr), axis=1) & (pqr[:, 0] >= 1e-10)
        return pqr[keep], keep

    p, keep_p = _clean(pqr_arr_p)
    m, keep_m = _clean(pqr_arr_m)
    n_p, n_m = p.shape[0], m.shape[0]
    if n_p < 2 or n_m < 2:
        raise ValueError("Not enough valid objects after filtering")

    def pqr_to_qr_totals(pqr):
        # jax mirror of qr_log_totals (kept in jax for the lax.map bootstrap);
        # flow order [P,Q1,Q2,R11,R22,R12] -> R=[[R11,R12],[R12,R22]] (see to_bfd).
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

    # Per-object BFD selection correction: each galaxy carries its own σ_f-bin
    # selection log-total, so a resample moves the measured and selection sums
    # together (correct, correlation-aware error bars).  Filter to the kept rows.
    def _sub_sel(Q_tot, R_tot, sel_q, sel_r, keep):
        if sel_q is None:
            return Q_tot, R_tot
        return (Q_tot - jnp.asarray(sel_q)[keep],
                R_tot - jnp.asarray(sel_r)[keep])

    Qp, Rp = _sub_sel(Qp, Rp, sel_qtot_p, sel_rtot_p, keep_p)
    Qm, Rm = _sub_sel(Qm, Rm, sel_qtot_m, sel_rtot_m, keep_m)
    eye2 = jnp.eye(2)

    def g_from_totals(Q_tot, R_tot):
        Q_sum = jnp.nansum(Q_tot, axis=0)
        R_sum = jnp.nansum(R_tot, axis=0) + ridge * eye2
        return jnp.linalg.solve(R_sum, Q_sum)

    def _mcc(g_p, g_m):
        # (m, c1, c2) from the ± lever: m from g1 difference, additive c from the mean.
        m = (g_p[0] - g_m[0]) / delta_g - 1.0
        c1 = (g_p[0] + g_m[0]) / 2.0
        c2 = (g_p[1] + g_m[1]) / 2.0
        return jnp.array([m, c1, c2])

    g_p_point = g_from_totals(Qp, Rp)
    g_m_point = g_from_totals(Qm, Rm)
    m_point, c1_point, c2_point = _mcc(g_p_point, g_m_point)

    key, sub = jr.split(key)
    keys = jr.split(sub, n_boot)

    def one_boot(k):
        kp, km = jr.split(k)
        ip = jr.randint(kp, shape=(n_p,), minval=0, maxval=n_p)
        im = jr.randint(km, shape=(n_m,), minval=0, maxval=n_m)
        g_p = g_from_totals(Qp[ip], Rp[ip])
        g_m = g_from_totals(Qm[im], Rm[im])
        return _mcc(g_p, g_m)

    boot = jax.lax.map(one_boot, keys)         # (n_boot, 3) = [m, c1, c2]
    m_boot, c1_boot, c2_boot = boot[:, 0], boot[:, 1], boot[:, 2]

    out = {
        "m_point": m_point,
        "m_mean": jnp.nanmean(m_boot),
        "m_std": jnp.nanstd(m_boot, ddof=1),
        "m_p16": jnp.nanpercentile(m_boot, 16.0),
        "m_p84": jnp.nanpercentile(m_boot, 84.0),
        "c1_point": c1_point,
        "c1_std": jnp.nanstd(c1_boot, ddof=1),
        "c2_point": c2_point,
        "c2_std": jnp.nanstd(c2_boot, ddof=1),
        "n_used_p": n_p,
        "n_used_m": n_m,
    }
    if return_samples:
        out["m_boot"] = m_boot
        out["c1_boot"] = c1_boot
        out["c2_boot"] = c2_boot
    return out
