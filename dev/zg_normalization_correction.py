#!/usr/bin/env python
"""dev/zg_normalization_correction.py
Test the missing centroid-normalization Z(g,C_X) hypothesis for the flow m-bias.

BFD's master formula (bfd2.tex eq. pMsG1) keeps the centroid factor L[X^G(g)|C_X]
*inside* the template sum, unnormalized:

    P_BFD(target,g) ∝ |J_target| · Σ_copies nda · L[M_target−M^G(g)] · L[X^G(g)|C_X]

The flow folds the SAME L[X|C_X] into its copy weights but then renormalizes per
g-column (flows.py:915  w_bg = softmax(log_w_bg, axis=0)).  A normalizing flow is a
normalized density: ∫ p_flow(m|g) dm = 1 for *every* g, so it structurally cannot
carry the g-dependence of the centroid-marginalization total mass

    Z(g,C_X) ≡ Σ_copies nda · L[X^G(g)|C_X].

Consequence (exact for a perfectly fit flow):

    P_flow(target,g) = P_BFD(target,g) / ( Z(g,C_X) · |J_target| )
    ⇒ (Q/P)_BFD = (Q/P)_flow + ∂_g logZ            [Q_tot correction]
    ⇒ R_tot,BFD = R_tot,flow − ∂²_g logZ           [R_tot = −∇²logP correction]
    ⇒ flow misses  m ≈ −⟨∂²_g logZ(g,C_X)⟩ / ⟨R_BFD⟩,  with C_X structure.

|J_target| is g-constant → cancels in ĝ.  Z(g) is NOT g-constant (the centroid
moments shear) → its curvature is an unmodeled multiplicative bias.

This script computes ∂_g logZ and ∂²_g logZ from the training catalog over a
log_scale grid (the C_X axis the residual localizes to; e≈0), applies them as an
inference-time correction to an existing flow-PQR npz (bfd_cnf.integrate_grid
output), and re-measures m before vs after.  No retraining.

Run the self-test:   python dev/zg_normalization_correction.py --self-test
Run the analysis:    python dev/zg_normalization_correction.py --npz data/pqr_indep_nll_ndaL.npz
"""
from __future__ import annotations

import argparse
import os

# Light CPU-only math (logsumexps over the catalog); leave the GPU for the flow
# integration that produced the npz.  Must precede the jax import.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# float64 is REQUIRED: ∂²_g logZ·h² ~ 1e-6 sits at the float32 noise floor against
# logZ ~ -14 (catastrophic cancellation in the 2nd difference).
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp

from bfd_cnf.models.flows import shear, _batch_log_L_X

# bfd_cnf.config (imported above) forces x64 OFF; re-enable it here — the 2nd
# difference for ∂²_g logZ needs float64 (see note above).
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Z(g, C_X) and its shear derivatives, from the training-catalog copies
# ---------------------------------------------------------------------------

# 9-point central-difference stencil for ∂_g and ∂²_g of a 2D scalar field.
def _stencil(h: float) -> np.ndarray:
    return np.array(
        [[0, 0], [h, 0], [-h, 0], [0, h], [0, -h],
         [h, h], [h, -h], [-h, h], [-h, -h]], dtype=np.float64
    )


def _fd_derivs(lz: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference ∂_g logZ (2,) and ∂²_g logZ (2,2) from the 9 stencil values."""
    d1 = np.array([(lz[1] - lz[2]) / (2 * h), (lz[3] - lz[4]) / (2 * h)])
    d11 = (lz[1] - 2 * lz[0] + lz[2]) / h**2
    d22 = (lz[3] - 2 * lz[0] + lz[4]) / h**2
    d12 = (lz[5] - lz[6] - lz[7] + lz[8]) / (4 * h**2)
    d2 = np.array([[d11, d12], [d12, d22]])
    return d1, d2


@jax.jit
def _logZ_over_grid(X, dgX, d2gX, log_nda, g_grid, sx_cond):
    """log Z(g) = logsumexp_copies( log(nda) + log L[X(g)|C_X] )  for each g in g_grid.

    X (N,2) centroid moments; dgX (N,2,2), d2gX (N,2,2,2) its shear derivatives
    (moment-component, shear-component); log_nda (N,); g_grid (1,G,2); sx_cond (3,).
    Returns (G,).
    """
    Xg = shear(X, g_grid, dgX, d2gX)  # (N,G,2) — identical shear() the loss uses
    logL = jax.vmap(lambda s: _batch_log_L_X(s, sx_cond), in_axes=1, out_axes=1)(Xg)
    return jax.scipy.special.logsumexp(log_nda[:, None] + logL, axis=0)  # (G,)


def logZ_derivs_over_log_scale(
    X, dgX, d2gX, nda, log_scale_grid, *, e1=0.0, e2=0.0, h=0.01
):
    """∂_g logZ (K,2) and ∂²_g logZ (K,2,2) on a grid of log_scale values (e fixed)."""
    X = jnp.asarray(X, jnp.float64)
    dgX = jnp.asarray(dgX, jnp.float64)
    d2gX = jnp.asarray(d2gX, jnp.float64)
    log_nda = jnp.log(jnp.maximum(jnp.asarray(nda, jnp.float64), 1e-300))
    g_grid = jnp.asarray(_stencil(h))[None, :, :]

    d1_list, d2_list = [], []
    for ls in np.asarray(log_scale_grid):
        sx = jnp.asarray([ls, e1, e2], jnp.float64)
        lz = np.asarray(_logZ_over_grid(X, dgX, d2gX, log_nda, g_grid, sx))
        d1, d2 = _fd_derivs(lz, h)
        d1_list.append(d1)
        d2_list.append(d2)
    return np.stack(d1_list), np.stack(d2_list)


def interp_corrections(log_scale_obj, log_scale_grid, d1_grid, d2_grid):
    """Interpolate the per-component ∂_g logZ / ∂²_g logZ to each object's log_scale.

    np.interp clamps outside the grid (targets in the C_X wings get the endpoint
    correction), which is the conservative choice.
    """
    lso = np.asarray(log_scale_obj, float)
    d1 = np.stack([np.interp(lso, log_scale_grid, d1_grid[:, a]) for a in range(2)], -1)
    d2 = np.empty((lso.size, 2, 2))
    for a in range(2):
        for b in range(2):
            d2[:, a, b] = np.interp(lso, log_scale_grid, d2_grid[:, a, b])
    return d1, d2  # (N,2), (N,2,2)


# ---------------------------------------------------------------------------
# Shear estimator with the optional Z(g) correction (mirrors statistics.pqr2g)
# ---------------------------------------------------------------------------


def pqr2g_np(pqr, dlogZ=None, d2logZ=None):
    """ĝ = (Σ R_tot)⁻¹ (Σ Q_tot) from per-object [P,Q1,Q2,R11,R22,R12].

    With dlogZ (N,2) / d2logZ (N,2,2) supplied, applies the BFD-recovering
    correction  Q_tot += ∂_g logZ ,  R_tot −= ∂²_g logZ  per object.  With both
    None this reproduces bfd_cnf.statistics.pqr2g exactly.
    """
    pqr = np.asarray(pqr, float)
    P = pqr[:, 0]
    mask = P >= 1e-10
    pqr, P = pqr[mask], P[mask]
    Q = pqr[:, 1:3]
    R = np.empty((len(pqr), 2, 2))
    R[:, 0, 0], R[:, 1, 1] = pqr[:, 3], pqr[:, 4]
    R[:, 0, 1] = R[:, 1, 0] = pqr[:, 5]

    qlog = Q / P[:, None]                                            # ∇ log P
    rtot = (Q[:, :, None] * Q[:, None, :]) / (P**2)[:, None, None] \
        - R / P[:, None, None]                                       # −∇² log P
    if dlogZ is not None:
        qlog = qlog + np.asarray(dlogZ)[mask]
        rtot = rtot - np.asarray(d2logZ)[mask]
    Q_tot = np.nansum(qlog, axis=0)
    R_tot = np.nansum(rtot, axis=0)
    return np.linalg.solve(R_tot, Q_tot)


def mult_bias(pqr_p, pqr_m, corr_p=None, corr_m=None, delta_g=0.04):
    cp = corr_p or (None, None)
    cm = corr_m or (None, None)
    g_p = pqr2g_np(pqr_p, *cp)
    g_m = pqr2g_np(pqr_m, *cm)
    return float((g_p[0] - g_m[0]) / delta_g - 1.0), g_p, g_m


def per_bin_m(pqr_p, ls_p, pqr_m, ls_m, corr_p, corr_m, n_bins=8):
    """m per log_scale bin, before (corr=None) handled by caller via two calls."""
    edges = np.quantile(np.concatenate([ls_p, ls_m]), np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mp = (ls_p >= lo) & (ls_p < hi if i < n_bins - 1 else ls_p <= hi)
        mm = (ls_m >= lo) & (ls_m < hi if i < n_bins - 1 else ls_m <= hi)
        if mp.sum() < 50 or mm.sum() < 50:
            rows.append((0.5 * (lo + hi), mp.sum() + mm.sum(), np.nan))
            continue
        cp = (corr_p[0][mp], corr_p[1][mp]) if corr_p else None
        cm = (corr_m[0][mm], corr_m[1][mm]) if corr_m else None
        m, _, _ = mult_bias(pqr_p[mp], pqr_m[mm], cp, cm)
        rows.append((0.5 * (lo + hi), int(mp.sum() + mm.sum()), m))
    return rows


# ---------------------------------------------------------------------------
# Self-test: single-copy Z(g) has a closed form
# ---------------------------------------------------------------------------


def _check():
    """For ONE copy, Z(g)=nda·N(X0+A g; 0,C) ⇒ logZ is exactly quadratic in g:
        ∂_g logZ|0  = −Aᵀ C⁻¹ X0 ,   ∂²_g logZ|0 = −Aᵀ C⁻¹ A .
    Assert the finite-difference field matches these closed forms."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=2) * 300.0
    A = rng.normal(size=(2, 2)) * 50.0           # dX/dg  (moment-comp, shear-comp)
    ls = 11.9
    # C_X for sx_cond=[ls,0,0] is exp(ls)·I  (half_T = exp(ls), e=0).
    C = np.exp(ls) * np.eye(2)
    Cinv = np.linalg.inv(C)
    d1_exact = -(A.T @ Cinv @ X0)
    d2_exact = -(A.T @ Cinv @ A)

    d1g, d2g = logZ_derivs_over_log_scale(
        X0[None, :], A[None, :, :], np.zeros((1, 2, 2, 2)),
        np.array([1.0]), [ls], h=0.01,
    )
    d1, d2 = d1g[0], d2g[0]
    assert np.allclose(d1, d1_exact, rtol=1e-3, atol=1e-4), (d1, d1_exact)
    assert np.allclose(d2, d2_exact, rtol=1e-3, atol=1e-4), (d2, d2_exact)

    # pqr2g_np with no correction must reproduce statistics.pqr2g.
    from bfd_cnf.statistics import pqr2g
    pqr = rng.normal(size=(200, 6))
    pqr[:, 0] = np.abs(pqr[:, 0]) + 0.5
    g_ref = np.asarray(pqr2g(jnp.asarray(pqr)))
    assert np.allclose(pqr2g_np(pqr), g_ref, rtol=1e-4, atol=1e-6)
    print("self-test OK: FD ∂logZ/∂²logZ match closed form; pqr2g_np matches statistics.pqr2g")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="Run the assert self-test and exit.")
    ap.add_argument("--z-only", action="store_true",
                    help="Compute+print the Z(g) derivative grid from the catalog and exit "
                         "(no npz needed) — early read on ∂²_g logZ sign/magnitude.")
    ap.add_argument("--npz", type=str, help="Flow-PQR npz from bfd_cnf.integrate_grid.")
    ap.add_argument("--subsample", type=int, default=20,
                    help="Keep ~1/this of the catalog for the Z(g) sum (default 20).")
    ap.add_argument("--h", type=float, default=0.01, help="Shear finite-difference step.")
    ap.add_argument("--z-scale", type=float, default=1.0,
                    help="Scale the Z(g) correction by this factor (diagnostic: the flow "
                         "may already encode part of Z, so the missing fraction is <1).")
    ap.add_argument("--n-ls", type=int, default=30, help="log_scale grid points.")
    ap.add_argument("--ls-lo", type=float, default=10.0)
    ap.add_argument("--ls-hi", type=float, default=13.5)
    ap.add_argument("--n-bins", type=int, default=8, help="log_scale bins for per-bin m.")
    args = ap.parse_args()

    if args.self_test:
        _check()
        return
    if not (args.npz or args.z_only):
        ap.error("--npz is required (or use --self-test / --z-only)")

    _check()  # always validate the math before trusting the numbers

    # ── catalog copies (the prior population the flow normalized) ─────────────
    from bfd_cnf.data import load_training_dataset
    data = load_training_dataset(subsample=args.subsample)
    X = np.asarray(data["centroid_moments_jnp"])             # (N,2)
    dgX = np.asarray(data["dm_dg_jnp"])[:, 4:6, :]           # (N,2,2)  centroid derivs
    d2gX = np.asarray(data["d2m_dg2_jnp"])[:, 4:6, :, :]     # (N,2,2,2)
    nda = np.asarray(data["nda"])                            # (N,)
    print(f"Z(g) catalog: {X.shape[0]:,} copies")

    log_scale_grid = np.linspace(args.ls_lo, args.ls_hi, args.n_ls)
    d1_grid, d2_grid = logZ_derivs_over_log_scale(
        X, dgX, d2gX, nda, log_scale_grid, h=args.h
    )
    print("\n  log_scale   ∂logZ/∂g1    ∂²logZ/∂g1²   ∂²logZ/∂g2²   ∂²logZ/∂g1∂g2")
    for ls, d1, d2 in zip(log_scale_grid, d1_grid, d2_grid):
        print(f"  {ls:7.2f}   {d1[0]:+10.4f}   {d2[0,0]:+11.4f}   {d2[1,1]:+11.4f}   {d2[0,1]:+11.4f}")

    if args.z_only:
        return

    # ── existing flow PQR + per-object Σ_X ────────────────────────────────────
    z = np.load(args.npz)
    pqr_p, pqr_m = z["pqr_p"], z["pqr_m"]
    ls_p, ls_m = z["sx_conds_p"][:, 0], z["sx_conds_m"][:, 0]

    c_p = interp_corrections(ls_p, log_scale_grid, d1_grid * args.z_scale, d2_grid * args.z_scale)
    c_m = interp_corrections(ls_m, log_scale_grid, d1_grid * args.z_scale, d2_grid * args.z_scale)
    if args.z_scale != 1.0:
        print(f"\n(Z correction scaled by {args.z_scale})")

    m0, gp0, gm0 = mult_bias(pqr_p, pqr_m)
    m1, gp1, gm1 = mult_bias(pqr_p, pqr_m, c_p, c_m)

    print("\n=== Aggregate multiplicative bias (flow) ===")
    print(f"  BEFORE  m = {m0:+.4f}   g(+)={gp0[0]:+.4f} g(-)={gm0[0]:+.4f}")
    print(f"  AFTER   m = {m1:+.4f}   g(+)={gp1[0]:+.4f} g(-)={gm1[0]:+.4f}   (Z(g) correction)")
    if "pqr_sim_p" in z:
        ms, gps, gms = mult_bias(z["pqr_sim_p"], z["pqr_sim_m"])
        print(f"  BFD ref m = {ms:+.4f}   g(+)={gps[0]:+.4f} g(-)={gms[0]:+.4f}")

    print("\n=== Per-log_scale-bin m (before → after) ===")
    rows_b = per_bin_m(pqr_p, ls_p, pqr_m, ls_m, None, None, args.n_bins)
    rows_a = per_bin_m(pqr_p, ls_p, pqr_m, ls_m, c_p, c_m, args.n_bins)
    print("  log_scale    n      m_before    m_after")
    for (lc, n, mb), (_, _, ma) in zip(rows_b, rows_a):
        print(f"  {lc:8.2f}  {n:7d}   {mb:+8.4f}   {ma:+8.4f}")


if __name__ == "__main__":
    main()
