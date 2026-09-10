"""Choose the defensive floor's `eps`, and measure what it actually catches.

The floor is inert wherever `eps * P_broad << P_flow` and takes over wherever
`eps * P_broad >> P_flow`.  So the whole choice is set by the distribution of
`lp - lb` (log P_flow - log P_broad):

  * on the TRAINING PRIOR it must stay well above `log eps` -- otherwise the
    floor is rewriting the density where the flow is right;
  * on the poisoned KERNEL DRAWS it is hugely negative, which is what the floor
    is for.

Prints both distributions and, for a ladder of eps, the fraction of each set the
floor takes over.  A good eps has ~0% takeover on the prior and ~all of the
poisoned draws.

    PYTHONPATH=. python dev/floor_scan.py
"""
import os
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias                                                      # noqa: E402
import bulk                                                      # noqa: E402
import shear as sm                                               # noqa: E402
from models.bijections import POINT_SOURCE, in_support           # noqa: E402

D = "../bfd_cnf_imsims/data"
FLOW = os.environ.get("FLOW", "flows/bulk_v3.eqx")
N = 20000
EPS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]


def gap(w, x):
    """lp - lb, in-support rows only."""
    x = jnp.asarray(x, jnp.float32)
    lp = np.asarray(jax.vmap(w.flow.log_prob)(x))
    lb = np.asarray(jax.vmap(w._log_broad)(x))
    ok = np.asarray(in_support(x)) & np.isfinite(lb)
    lp = np.where(np.isfinite(lp), lp, -np.inf)
    return (lp - lb)[ok], lp[ok]


def report(name, g, lp):
    print(f"  {name:<22s} n {len(g):7d}   lp-lb: p0.1 {np.percentile(g, 0.1):9.2f}"
          f"  p1 {np.percentile(g, 1):8.2f}  p50 {np.percentile(g, 50):8.2f}"
          f"  min {g.min():11.4g}")
    row = "      takeover fraction:  " + "  ".join(
        f"eps {e:<7g} {np.mean(g < np.log(e)):6.2%}" for e in EPS)
    print(row)


def main():
    m, _, _ = sm.load(f"{D}/moments_bulgedisc_v3.fits")
    m = np.asarray(m, np.float64)
    tg = fitsio.read(f"{D}/targets_v3_g0_200k.fits", rows=np.arange(N))
    M = np.asarray(tg["moments"], np.float64)
    cov = np.asarray(bias.load_cov(f"{D}/targets_v3_g0_200k.fits"), np.float64)
    if cov.ndim == 3:
        cov = cov[0]

    flow = eqx.tree_deserialise_leaves(FLOW, bulk.build_flow(jr.key(0), m))
    w = bulk.SupportedFlow(flow, eps=1e-3, broad_std=4.0)

    print(f"{FLOW}, broad_std = {w.broad_std}\n")
    print("where the floor takes over (lp - lb < log eps):")
    g, lp = gap(w, m[:N])
    report("training prior", g, lp)
    g, lp = gap(w, M)
    report("g=0 targets (noisy)", g, lp)

    # kernel draws, split by how close the target is to the point-source edge
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(0)
    S = 512
    a = np.zeros(5)
    a[0], a[1] = POINT_SOURCE, -1.0
    t = M @ a / np.sqrt(a @ cov @ a)
    for lo, hi, name in ((-9, 1, "draws, target <1 sig"), (1, 3, "draws, target 1-3"),
                         (3, 99, "draws, target >3")):
        idx = np.where((t >= lo) & (t < hi))[0][:60]
        if len(idx) == 0:
            continue
        dr = np.concatenate([M[i] + rng.standard_normal((S, 5)) @ L.T
                             for i in idx])
        g, lp = gap(w, dr)
        pois = lp < -1e4
        report(name, g, lp)
        print(f"      of these, {pois.mean():6.2%} are poisoned (lp < -1e4); "
              f"floor catches {np.mean(g[pois] < np.log(1e-3)):6.2%} of those "
              f"at eps 1e-3")


if __name__ == "__main__":
    main()
