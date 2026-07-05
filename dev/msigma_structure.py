#!/usr/bin/env python
"""Per-sigma-bin m structure: m_flow and the paired excess (m_flow-m_sim), binned by
C_X noise level sigma=exp(log_scale/2), for baseline vs end-of-continued-training.
Galaxies common+valid to both files; SAME bootstrap resample for A and B within a bin,
so Δexc=(exc_B-exc_A) isolates the flow change (sim/population scatter cancels).
Each arm binned by its own targets' sigma (arms are independent populations).
Usage: python dev/msigma_structure.py A.npz B.npz  (A=baseline, B=end-of-cont)
"""
import sys
import numpy as np

A = sys.argv[1] if len(sys.argv) > 1 else "data/pqr_grid_100k_nll.npz"
B = sys.argv[2] if len(sys.argv) > 2 else "data/pqr_grid_100k_nll_cont90000.npz"
DG, NB, NBIN = 0.04, 2000, 6


def qr_tot(pqr):
    P = pqr[:, 0]; Q = pqr[:, 1:3]
    R = np.stack([np.stack([pqr[:, 3], pqr[:, 5]], -1),
                  np.stack([pqr[:, 5], pqr[:, 4]], -1)], -2)
    return Q / P[:, None], np.einsum("ni,nj->nij", Q, Q) / P[:, None, None] ** 2 - R / P[:, None, None]


def valid(pqr):
    return np.all(np.isfinite(pqr), 1) & (pqr[:, 0] > 1e-10)


def g1(Qt, Rt, idx):
    Rs = Rt[idx].sum(0) + 1e-10 * np.eye(2)
    return np.linalg.solve(Rs, Qt[idx].sum(0))[0]


da, db = np.load(A), np.load(B)


def align(arm):
    """Common+valid galaxies; returns sig, qr_flowA, qr_flowB, qr_sim (sim from baseline)."""
    ida, idb = da[f"ids_{arm}"], db[f"ids_{arm}"]
    _, ia, ib = np.intersect1d(ida, idb, return_indices=True)
    pfa, pfb = da[f"pqr_{arm}"][ia], db[f"pqr_{arm}"][ib]
    ps = da[f"pqr_sim_{arm}"][ia]
    sig = np.exp(da[f"sx_conds_{arm}"][ia, 0] / 2.0)
    keep = valid(pfa) & valid(pfb) & valid(ps)
    return (sig[keep], qr_tot(pfa[keep]), qr_tot(pfb[keep]), qr_tot(ps[keep]))


arms = {arm: align(arm) for arm in ("p", "m")}
# fixed sigma edges: fine in the bulk, resolve the sparse wide-sigma tail (crossover ~390, Q-collapse beyond)
edges = np.array([-np.inf, 270, 290, 310, 340, 390, 460, 560, np.inf])
NBIN = len(edges) - 1
binlab = {arm: np.clip(np.digitize(arms[arm][0], edges[1:-1]), 0, NBIN - 1) for arm in ("p", "m")}

print(f"A (baseline)    = {A}")
print(f"B (end-of-cont) = {B}\n")
print(f"{'sigma':>7} {'n+':>6} | {'mF_A':>7} {'mF_B':>7} | "
      f"{'exc_A':>7} {'exc_B':>7} | {'Δexc (B-A, paired)':>22}")

rng = np.random.default_rng(0)
for k in range(NBIN):
    pk = np.where(binlab["p"] == k)[0]
    mk = np.where(binlab["m"] == k)[0]
    smed = np.median(arms["p"][0][pk])
    mFA = np.empty(NB); mFB = np.empty(NB); eA = np.empty(NB); eB = np.empty(NB)
    for bi in range(NB):
        ip = pk[rng.integers(0, pk.size, pk.size)]
        im = mk[rng.integers(0, mk.size, mk.size)]
        _, qfA_p, qfB_p, qs_p = arms["p"]
        _, qfA_m, qfB_m, qs_m = arms["m"]
        gFA = (g1(*qfA_p, ip) - g1(*qfA_m, im)) / DG - 1
        gFB = (g1(*qfB_p, ip) - g1(*qfB_m, im)) / DG - 1
        gS = (g1(*qs_p, ip) - g1(*qs_m, im)) / DG - 1
        mFA[bi] = gFA; mFB[bi] = gFB; eA[bi] = gFA - gS; eB[bi] = gFB - gS
    dexc = eB - eA
    print(f"{smed:7.0f} {pk.size:6d} | {mFA.mean():+7.3f} {mFB.mean():+7.3f} | "
          f"{eA.mean():+7.3f} {eB.mean():+7.3f} | "
          f"{dexc.mean():+8.4f}±{dexc.std(ddof=1):.4f} ({abs(dexc.mean()/dexc.std(ddof=1)):4.1f}σ)")
