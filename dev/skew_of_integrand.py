"""Test: does the skewness of the integrand p_flow*N(x;M,Sigma) drive the m-bias?

The integrator is already proven skew-ROBUST (dev/test_skew_integrator.py), so any
skew effect must be the FLOW giving a wrongly-skewed integrand => wrong Q/R. This
measures the integrand's per-target skewness (production path, return_ess now also
returns per-dim weighted 3rd moment) and asks whether the flow's shear response
(relative to the analytic BFD truth on the SAME objects/selection) grows with it.

x-moment axes: [Mf, Mr, M1, M2]; g1 shears M1 (axis 2), g2 shears M2 (axis 3).

Run:  PYTHONPATH=. JAX_PLATFORMS=cuda python dev/skew_of_integrand.py
"""
from __future__ import annotations

import numpy as np
import jax.random as jr

from bfd_cnf.config import GRID_P_PATH, key as base_key
from bfd_cnf.integrate_grid import load_raw2standard, load_prior_flow
from bfd_cnf.inference import integrate_catalog_pqr
from bfd_cnf.statistics import pqr2g

raw2standard = load_raw2standard("data/raw2standard_stats_retest.npz", rebuild=False)
k_flow, k_int = jr.split(base_key, 2)
prior_flow = load_prior_flow(k_flow)  # default flows/prior_flow_xy.eqx

res = integrate_catalog_pqr(
    np.load(GRID_P_PATH), raw2standard, prior_flow, key=k_int,
    n_targets=100000, flux_min=1500, flux_max=90000,
    n_points=4096, n_replicates=16, batch_size=512,
    hessian_scale=3.0, return_ess=True, verbose=True,
)
skew = np.asarray(res["skew"])       # (N,5): [log10Mf, Mr/Mf, M1/Mr, M2/Mr, skew2d_fluxsize]
pqr = np.asarray(res["pqr"])         # flow   [P,Q1,Q2,R11,R22,R12]
pqr_sim = np.asarray(res["pqr_sim"]) # analytic, same order
mf = np.asarray(res["targets"][:, 0])
np.savez("data/skew_of_integrand.npz", skew=skew, pqr=pqr, pqr_sim=pqr_sim, mf=mf)

# Standardized coords are [log10(Mf), Mr/Mf, M1/Mr, M2/Mr]; col 4 = 2D directional
# skew in the (log10Mf, Mr/Mf) plane (catches diagonal/banana skew the marginals miss).
axes = ["log10Mf", "Mr/Mf", "M1/Mr", "M2/Mr", "skew2D(f,s)"]
print("\n=== integrand p_flow*N skewness, median |skew|, by observed Mf ===")
print(f'{"Mf bin":>14} {"n":>7} ' + " ".join(f"{a:>11}" for a in axes))
edges = [1500, 2000, 2500, 3000, 4000, 6000, 10000, 90000]
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (mf >= lo) & (mf < hi)
    med = np.nanmedian(np.abs(skew[s]), axis=0)
    print(f'{f"{lo}-{hi}":>14} {s.sum():>7} ' + " ".join(f"{v:>11.3f}" for v in med))
med = np.nanmedian(np.abs(skew), axis=0)
print(f'{"ALL":>14} {len(skew):>7} ' + " ".join(f"{v:>11.3f}" for v in med))


def g1_ratio(mask):
    """flow g1 / analytic g1 on the SAME objects+selection (selection cancels in ratio)."""
    pf, ps = pqr[mask], pqr_sim[mask]
    if mask.sum() < 200:
        return np.nan, np.nan, np.nan
    g1f = float(pqr2g(pf)[0]); g1s = float(pqr2g(ps)[0])
    return g1f, g1s, (g1f / g1s if abs(g1s) > 1e-6 else np.nan)

# THE TEST: bin objects by the integrand's 2D skewness in the (log10Mf, Mr/Mf)
# plane and see whether the flow over-responds (g1_flow/g1_sim > 1) more where the
# integrand is more skewed in that plane.  Quintiles of skew2D.
sk = skew[:, 4]  # 2D directional skew in the flux-size plane (>=0)
good = np.isfinite(sk)
qs = np.nanquantile(sk[good], [0, .2, .4, .6, .8, 1.0])
print("\n=== shear response flow-vs-analytic, binned by integrand 2D skew (log10Mf,Mr/Mf) ===")
print(f'{"skew2D bin":>16} {"n":>7} {"g1_flow":>9} {"g1_sim":>9} {"flow/sim":>9}')
for lo, hi in zip(qs[:-1], qs[1:]):
    m = good & (sk >= lo) & (sk <= hi)
    g1f, g1s, r = g1_ratio(m)
    print(f'{f"{lo:.2f}-{hi:.2f}":>16} {m.sum():>7} {g1f:>9.4f} {g1s:>9.4f} {r:>9.3f}')
g1f, g1s, r = g1_ratio(good)
print(f'{"ALL":>16} {good.sum():>7} {g1f:>9.4f} {g1s:>9.4f} {r:>9.3f}')
print("\nIf flow/sim rises with skew2D => integrand skewness in the flux-size plane "
      "tracks the over-response => hypothesis supported.")
