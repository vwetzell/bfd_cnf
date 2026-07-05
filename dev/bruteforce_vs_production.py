"""PROOF test: is the m-bias the integration scheme or the flow?

Runs the SAME flow on the SAME targets with two integrators that share zero
proposal/sampler machinery:
  - production : x-space mode-find + Gaussian Laplace proposal + scrambled-Halton
                 with a hard +/-5.6 sigma clip (the suspect).
  - bruteforce : plain Monte Carlo -- draw straight from the (augmented) noise
                 kernel with ordinary normals, no proposal, no low-discrepancy,
                 no clip. Same integrand otherwise (same augmentation, jac-corr,
                 flow eval).
Both selected with the SAME key => identical target set, identical analytic
pqr_sim. We compare the flow/analytic g1 over-response, split by the integrand's
2D flux-size-plane skewness (skew2D, from the production run).

If bruteforce reproduces production's ~1.45x over-response on skewed targets, the
production scheme is unbiased and the defect is the flow. If bruteforce gives ~1.0
where production gives ~1.45, the scheme is the culprit.

Run:  PYTHONPATH=. JAX_PLATFORMS=cuda python dev/bruteforce_vs_production.py
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
prior_flow = load_prior_flow(k_flow)

cat = np.load(GRID_P_PATH)
common = dict(flux_min=1500, flux_max=90000, n_targets=5000)
# Same k_int => same subsample => same targets & pqr_sim for both integrators.

print(">>> production (mode-find + Gaussian proposal + Halton clip), np=4096 x16")
prod = integrate_catalog_pqr(cat, raw2standard, prior_flow, key=k_int,
                             n_points=4096, n_replicates=16, hessian_scale=3.0,
                             batch_size=512, return_ess=True, bruteforce=False, **common)

# bruteforce needs a big point count (no variance reduction); shrink the batch so
# the per-step (batch, n_points, dim) flow intermediates match the production footprint.
print(">>> bruteforce (plain kernel MC, no proposal/clip), np=16384 x8")
bf = integrate_catalog_pqr(cat, raw2standard, prior_flow, key=k_int,
                           n_points=16384, n_replicates=8,
                           batch_size=128, return_ess=False, bruteforce=True, **common)

pf_prod, pf_bf, ps = np.asarray(prod["pqr"]), np.asarray(bf["pqr"]), np.asarray(prod["pqr_sim"])
skew2d = np.asarray(prod["skew"])[:, 4]
assert np.array_equal(np.asarray(prod["ids"]), np.asarray(bf["ids"])), "target sets differ!"
np.savez("data/bruteforce_vs_production.npz",
         pqr_prod=pf_prod, pqr_bf=pf_bf, pqr_sim=ps, skew2d=skew2d,
         ids=np.asarray(prod["ids"]))


def g1(pqr, m):
    return float(pqr2g(pqr[m])[0]) if m.sum() >= 200 else np.nan


good = np.isfinite(skew2d)
qs = np.nanquantile(skew2d[good], [0, .2, .4, .6, .8, 1.0])
print("\n=== flow g1 over-response vs analytic: PRODUCTION vs BRUTEFORCE, by skew2D ===")
print(f'{"skew2D bin":>16} {"n":>6} | {"g1_sim":>8} {"g1_prod":>8} {"prod/sim":>8} '
      f"| {'g1_bf':>8} {'bf/sim':>8} | {'prod/bf':>8}")
for lo, hi in zip(qs[:-1], qs[1:]):
    m = good & (skew2d >= lo) & (skew2d <= hi)
    gs, gp, gb = g1(ps, m), g1(pf_prod, m), g1(pf_bf, m)
    print(f'{f"{lo:.2f}-{hi:.2f}":>16} {m.sum():>6} | {gs:>8.4f} {gp:>8.4f} '
          f'{gp/gs:>8.3f} | {gb:>8.4f} {gb/gs:>8.3f} | {gp/gb:>8.3f}')
m = good
gs, gp, gb = g1(ps, m), g1(pf_prod, m), g1(pf_bf, m)
print(f'{"ALL":>16} {m.sum():>6} | {gs:>8.4f} {gp:>8.4f} {gp/gs:>8.3f} | '
      f'{gb:>8.4f} {gb/gs:>8.3f} | {gp/gb:>8.3f}')

# Per-target agreement of the two integrators (the direct proof).
both = good & (pf_prod[:, 0] > 1e-12) & (pf_bf[:, 0] > 1e-12)
for nm, ci in [("P", 0), ("Q1", 1), ("R11", 3)]:
    a, b = pf_prod[both, ci], pf_bf[both, ci]
    rel = np.abs(a - b) / (np.abs(b) + 1e-30)
    print(f"per-target {nm:>3}: median |prod-bf|/|bf| = {np.nanmedian(rel):.3e}  "
          f"corr = {np.corrcoef(a, b)[0,1]:.4f}")
print("\nIf prod/bf ~ 1 in every skew2D bin => the two schemes AGREE => the "
      "over-response is the FLOW's, not the integrator's.")
