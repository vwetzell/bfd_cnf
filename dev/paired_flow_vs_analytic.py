#!/usr/bin/env python
"""Paired flow-vs-analytic m: bootstrap flow and analytic on the SAME resampled galaxies
(per arm) so the shared population scatter cancels in the difference Δm=m_flow−m_sim.
Beats down the noise on the FLOW's bias relative to BFD vs quoting m_flow, m_sim separately.
Usage: python dev/paired_flow_vs_analytic.py [npz]  (default data/pqr_grid_100k_nll.npz)
"""
import sys
import numpy as np

NPZ = sys.argv[1] if len(sys.argv) > 1 else "data/pqr_grid_100k_nll.npz"
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


d = np.load(NPZ)
# ID overlap between arms (ring-pair-by-id feasibility check)
ip_ids, im_ids = d["ids_p"], d["ids_m"]
inter = np.intersect1d(ip_ids, im_ids)
print(f"{NPZ}")
print(f"+arm {ip_ids.size}  -arm {im_ids.size}  shared IDs: {inter.size} "
      f"({100*inter.size/ip_ids.size:.1f}% of +arm)")

# keep arms where BOTH flow and analytic are valid (paired within arm)
vp = valid(d["pqr_p"]) & valid(d["pqr_sim_p"])
vm = valid(d["pqr_m"]) & valid(d["pqr_sim_m"])
Qtf_p, Rtf_p = qr_tot(d["pqr_p"][vp]); Qts_p, Rts_p = qr_tot(d["pqr_sim_p"][vp])
Qtf_m, Rtf_m = qr_tot(d["pqr_m"][vm]); Qts_m, Rts_m = qr_tot(d["pqr_sim_m"][vm])
np_, nm_ = vp.sum(), vm.sum()
print(f"valid paired: +{np_}  -{nm_}")

rng = np.random.default_rng(0)
mf = np.empty(NB); ms = np.empty(NB); dm = np.empty(NB)
for b in range(NB):
    ip = rng.integers(0, np_, np_); im = rng.integers(0, nm_, nm_)  # SAME idx for flow & sim
    g1pf = g1(Qtf_p, Rtf_p, ip); g1mf = g1(Qtf_m, Rtf_m, im)
    g1ps = g1(Qts_p, Rts_p, ip); g1ms = g1(Qts_m, Rts_m, im)
    mf[b] = (g1pf - g1mf) / DG - 1
    ms[b] = (g1ps - g1ms) / DG - 1
    dm[b] = mf[b] - ms[b]

print(f"\nm_flow      = {mf.mean():+.4f} ± {mf.std(ddof=1):.4f}")
print(f"m_analytic  = {ms.mean():+.4f} ± {ms.std(ddof=1):.4f}")
print(f"Δm (paired) = {dm.mean():+.4f} ± {dm.std(ddof=1):.4f}   "
      f"[flow bias vs BFD; {abs(dm.mean()/dm.std(ddof=1)):.1f}σ]")
print(f"  (unpaired Δm error would be √(σf²+σs²) = "
      f"{np.hypot(mf.std(ddof=1), ms.std(ddof=1)):.4f}; pairing shrinks it "
      f"{np.hypot(mf.std(ddof=1), ms.std(ddof=1))/dm.std(ddof=1):.1f}×)")
