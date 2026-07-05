#!/usr/bin/env python
"""Conditioning-sensitivity sweep: SAME 1k galaxies, vary only the log_scale fed to the
flow (fixed_sx_cond); integration kernel stays at each target's TRUE covariance and the
analytic pqr_sim is unchanged.  Isolates whether the flow's shear response (Q) and
curvature (R) drift with the C_X conditioning alone — no population noise.  See
project_mbias_logscale_qr_imbalance.
"""
import sys

import numpy as np
import jax
import jax.random as jr

from bfd_cnf.integrate_grid import GRID_P_PATH, GRID_M_PATH, load_stats, load_prior_flow

from bfd_cnf.inference import integrate_catalog_pqr

# usage: python dev/sweep_logscale_1k.py [N] [--coarse]
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000
LS = ([10.5, 11.4, 11.94, 13.0] if "--coarse" in sys.argv
      else [10.5, 11.0, 11.4, 11.7, 11.94, 12.2, 12.5, 13.0])


def qr_tot(pqr):
    P = pqr[:, 0]; Q = pqr[:, 1:3]
    R = np.stack([np.stack([pqr[:, 3], pqr[:, 5]], -1),
                  np.stack([pqr[:, 5], pqr[:, 4]], -1)], -2)
    Qt = Q / P[:, None]
    Rt = np.einsum("ni,nj->nij", Q, Q) / P[:, None, None] ** 2 - R / P[:, None, None]
    return Qt, Rt


def g_of(pqr, mask):
    Qt, Rt = qr_tot(pqr)
    Qs = np.nansum(Qt[mask], 0); Rs = np.nansum(Rt[mask], 0) + 1e-10 * np.eye(2)
    return np.linalg.solve(Rs, Qs), np.nansum(Qt[mask, 0]), np.nansum(Rt[mask, 0, 0])


def valid(pqr):
    return np.isfinite(pqr).all(1) & (pqr[:, 0] > 1e-10)


def main():
    key = jr.PRNGKey(0)
    k_flow, kp, km = jr.split(key, 3)          # kp/km fixed -> SAME 1k at every ls
    raw2standard = load_stats("flows/prior_flow_xy_nll.eqx")
    prior_flow = load_prior_flow(k_flow, "flows/prior_flow_xy_nll.eqx", "flows/q_flow_xy.eqx")
    cat_p = np.load(GRID_P_PATH); cat_m = np.load(GRID_M_PATH)

    common = dict(n_targets=N, flux_min=1500.0, flux_max=90000.0, verbose=False)

    print(f"{'log_scale':>9} {'sigma':>6} {'m_flow':>8} {'m_sim':>8} "
          f"{'ΣQ1 f/a':>8} {'ΣR11 f/a':>9} {'g1p_flow':>9} {'g1p_sim':>8} {'n':>5}")
    for ls in LS:
        fx = (ls, 0.0, 0.0)
        rp = integrate_catalog_pqr(cat_p, raw2standard, prior_flow, key=kp, fixed_sx_cond=fx, **common)
        rm = integrate_catalog_pqr(cat_m, raw2standard, prior_flow, key=km, fixed_sx_cond=fx, **common)
        pf, mf_ = np.asarray(rp["pqr"]), np.asarray(rm["pqr"])
        ps, ms_ = np.asarray(rp["pqr_sim"]), np.asarray(rm["pqr_sim"])
        v = valid(pf) & valid(ps); vm = valid(mf_) & valid(ms_)
        gpf, q1f, r1f = g_of(pf, v); gmf, _, _ = g_of(mf_, vm)
        gps, q1s, r1s = g_of(ps, v); gms, _, _ = g_of(ms_, vm)
        m_flow = (gpf[0] - gmf[0]) / 0.04 - 1
        m_sim = (gps[0] - gms[0]) / 0.04 - 1
        print(f"{ls:9.2f} {np.exp(ls/2):6.0f} {m_flow:+8.3f} {m_sim:+8.3f} "
              f"{q1f/q1s:8.3f} {r1f/r1s:9.3f} {gpf[0]:+9.4f} {gps[0]:+8.4f} {int(v.sum()):5d}")

    print("\nm_sim is ~flat (analytic pqr unchanged by fixed_sx_cond); the m_flow swing and the "
          "ΣQ1 f/a collapse vs log_scale = the flow's conditioning sensitivity.")


if __name__ == "__main__":
    main()
