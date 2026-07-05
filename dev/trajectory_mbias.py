#!/usr/bin/env python
"""Paired Δm trajectory: align every continued-training checkpoint to the baseline
by galaxy ID and bootstrap them JOINTLY (same resampled galaxies across all steps)
so the curve is internally consistent. Δm(step) = m(step) − m(baseline), paired.
Usage: python dev/trajectory_mbias.py
"""
import numpy as np

BASE = "data/pqr_grid_100k_nll.npz"
STEPS = [  # (label, npz)
    ("10k", "data/pqr_grid_100k_nll_cont10k.npz"),
    ("30k", "data/pqr_grid_100k_nll_cont30000.npz"),
    ("50k", "data/pqr_grid_100k_nll_cont50000.npz"),
    ("70k", "data/pqr_grid_100k_nll_cont70000.npz"),
    ("90k", "data/pqr_grid_100k_nll_cont90000.npz"),
]
DG, NB = 0.04, 4000


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


base = np.load(BASE)
checks = [(lbl, np.load(p)) for lbl, p in STEPS]

# Align all checkpoints + baseline + analytic to common valid IDs per arm.
arms = {}
for arm in ("p", "m"):
    idb = base[f"ids_{arm}"]
    common = idb
    for _, d in checks:
        common = np.intersect1d(common, d[f"ids_{arm}"])
    # index maps
    def idx_of(d):
        order = np.argsort(d[f"ids_{arm}"])
        pos = np.searchsorted(d[f"ids_{arm}"], common, sorter=order)
        return order[pos]
    pqr_base = base[f"pqr_{arm}"][idx_of(base)]
    pqr_sim = base[f"pqr_sim_{arm}"][idx_of(base)]
    pqr_ck = [d[f"pqr_{arm}"][idx_of(d)] for _, d in checks]
    keep = valid(pqr_base) & valid(pqr_sim)
    for p in pqr_ck:
        keep &= valid(p)
    arms[arm] = dict(
        base=qr_tot(pqr_base[keep]),
        sim=qr_tot(pqr_sim[keep]),
        ck=[qr_tot(p[keep]) for p in pqr_ck],
        n=int(keep.sum()),
    )
    print(f"{arm}-arm common+valid: {keep.sum()} (of {idb.size})")

rng = np.random.default_rng(0)
nlbl = len(STEPS)
m_base = np.empty(NB); m_sim = np.empty(NB)
m_ck = np.empty((nlbl, NB))
for b in range(NB):
    ip = rng.integers(0, arms["p"]["n"], arms["p"]["n"])
    im = rng.integers(0, arms["m"]["n"], arms["m"]["n"])
    def m_of(key, j=None):
        if j is None:
            qp, qm = arms["p"][key], arms["m"][key]
        else:
            qp, qm = arms["p"]["ck"][j], arms["m"]["ck"][j]
        return (g1(*qp, ip) - g1(*qm, im)) / DG - 1
    m_base[b] = m_of("base")
    m_sim[b] = m_of("sim")
    for j in range(nlbl):
        m_ck[j, b] = m_of(None, j)

print(f"\nBFD analytic (ref)  m = {m_sim.mean():+.4f} ± {m_sim.std(ddof=1):.4f}")
print(f"baseline (0 steps)  m = {m_base.mean():+.4f} ± {m_base.std(ddof=1):.4f}\n")
print(f"{'step':>6} {'m_flow':>14}  {'Δm vs base (paired)':>26}  {'Δm vs BFD':>12}")
print(f"{'0':>6} {m_base.mean():+.4f}±{m_base.std(ddof=1):.4f}  {'(reference)':>26}  "
      f"{(m_base-m_sim).mean():+.4f}")
for j, (lbl, _) in enumerate(STEPS):
    dvb = m_ck[j] - m_base
    dvbfd = m_ck[j] - m_sim
    print(f"{lbl:>6} {m_ck[j].mean():+.4f}±{m_ck[j].std(ddof=1):.4f}  "
          f"{dvb.mean():+.4f}±{dvb.std(ddof=1):.4f} ({abs(dvb.mean()/dvb.std(ddof=1)):4.1f}σ)  "
          f"{dvbfd.mean():+.4f}")
