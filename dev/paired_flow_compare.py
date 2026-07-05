#!/usr/bin/env python
"""Paired comparison of TWO flow integrations on the SAME galaxies (aligned by ID).
Bootstraps both flows on the same resampled galaxies so the shared scatter cancels in
Δm = m_B − m_A — measures the m SHIFT between flow versions far below each one's own error.
Usage: python dev/paired_flow_compare.py A.npz B.npz   (A=before, B=after)
"""
import sys
import numpy as np

A, B = sys.argv[1], sys.argv[2]
DG = 0.04
NB = 4000


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


def align(arm):  # arm = 'p' or 'm'; return (pqrA, pqrB, pqrSimA) on common, valid galaxies
    ida, idb = da[f"ids_{arm}"], db[f"ids_{arm}"]
    _, ia, ib = np.intersect1d(ida, idb, return_indices=True)
    pa, pb = da[f"pqr_{arm}"][ia], db[f"pqr_{arm}"][ib]
    ps = da[f"pqr_sim_{arm}"][ia]
    keep = valid(pa) & valid(pb) & valid(ps)
    return pa[keep], pb[keep], ps[keep]


print(f"A (before) = {A}\nB (after)  = {B}\n")
arms = {arm: align(arm) for arm in ("p", "m")}
qr = {}
for arm, (pa, pb, ps) in arms.items():
    qr[arm] = dict(A=qr_tot(pa), B=qr_tot(pb), S=qr_tot(ps), n=pa.shape[0])
    print(f"{arm}-arm common+valid galaxies: {pa.shape[0]}")

rng = np.random.default_rng(0)
mA = np.empty(NB); mB = np.empty(NB); mS = np.empty(NB)
for b in range(NB):
    ip = rng.integers(0, qr["p"]["n"], qr["p"]["n"])
    im = rng.integers(0, qr["m"]["n"], qr["m"]["n"])
    def m_of(key):
        gp = g1(*qr["p"][key], ip); gm = g1(*qr["m"][key], im)
        return (gp - gm) / DG - 1
    mA[b] = m_of("A"); mB[b] = m_of("B"); mS[b] = m_of("S")

d = mB - mA
print(f"\nm_A (before)      = {mA.mean():+.4f} ± {mA.std(ddof=1):.4f}")
print(f"m_B (after)       = {mB.mean():+.4f} ± {mB.std(ddof=1):.4f}")
print(f"m_analytic (ref)  = {mS.mean():+.4f} ± {mS.std(ddof=1):.4f}")
print(f"Δm = m_B − m_A    = {d.mean():+.4f} ± {d.std(ddof=1):.4f}   "
      f"[{abs(d.mean()/d.std(ddof=1)):.1f}σ shift; "
      f"paired error {np.hypot(mA.std(ddof=1), mB.std(ddof=1))/d.std(ddof=1):.0f}× below unpaired]")
