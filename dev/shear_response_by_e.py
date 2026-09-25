"""Shear layer's per-galaxy response against the exact dm/dg, by galaxy |e|.

The shear layer's transport IS a per-galaxy shift: a sheared galaxy's chart
coordinates move by -Q(z) g + O(g^2) (`ShearResponse.unshear` = z + Q g + ...),
with Q read in closed form off `response_tensors`.  It is fitted to the
conditional mean E[dz/dg | z], so on held-out prior galaxies the exact
dz/dg = J_chart dm/dg (jvp of the chart) should scatter around it with zero
mean in every bin.  Per |e| bin this reports the regression slope of the
flow's -Q on the exact response (1 = unbiased) for the spin-0 and spin-2 blocks,
and how close the five spin-2 coefficients (A, B, mu, nu, rho) sit to their
+/-_COEFF_MAX bound.
"""
import sys
sys.path.insert(0, ".")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import bias as B
import bulk
import shear
from models import shear as S

PRIOR = f"../bfd_cnf_imsims/data/{B.TRAIN_DATA['gauss2_v4n']}"
FLOW = sys.argv[1] if len(sys.argv) > 1 else "flows/shear_g2v4n.eqx"
EBINS = [0, 0.05, 0.08, 0.11, 0.15, 0.2, np.inf]
NAMES = ["A", "B", "mu", "nu", "rho"]


def main():
    m, dm, d2m = shear.load(PRIOR)
    n_tr = int(0.9 * len(m))
    flow = eqx.tree_deserialise_leaves(
        FLOW, bulk.build_flow(jr.key(0), m[:n_tr][:20000], shear=True, centroid=False))
    chain = flow.bijection.bijection.bijections
    chart, layer = chain[0], chain[1]
    assert type(layer).__name__ == "ShearResponse", [type(l).__name__ for l in chain]
    mh, dmh = m[n_tr:], dm[n_tr:]                      # held out of bulk and shear training

    @jax.jit
    def one(mi, dmi):
        z = chart.transform(mi)
        dz = jnp.stack([jax.jvp(chart.transform, (mi,), (dmi[a],))[1] for a in range(2)], -1)  # (5,2)
        Q, _ = layer.response_tensors(z)
        f, a_, b_, q, _ = S._invariants(z)
        return dz, -Q, layer.coeffs(f, a_, b_, q)

    out = [jax.vmap(one)(jnp.asarray(mh[i:i + 8192], jnp.float32), jnp.asarray(dmh[i:i + 8192], jnp.float32))
           for i in range(0, len(mh), 8192)]
    ex, fl, co = (np.concatenate([np.asarray(o[k], np.float64) for o in out]) for k in range(3))
    e = np.hypot(mh[:, 2], mh[:, 3]) / mh[:, 1]
    bound = np.r_[np.full(9, S._COEFF_MAX_SPIN0), np.full(5, S._COEFF_MAX)]

    def slope(a, b):
        return (a * b).sum() / (b * b).sum()

    def r2(a, b):
        return 1 - ((a - b) ** 2).sum() / ((b - b.mean(0)) ** 2).sum()

    print(f"{FLOW}: {len(mh)} held-out prior galaxies")
    print(f"{'|e| bin':>12s} {'n':>6s}  spin0 slope   spin2 slope  spin2 R2(flow vs cond. scatter)   "
          + "  ".join(f"{k:>11s}" for k in NAMES) + "   max|c|/bound")
    for lo, hi in zip(EBINS[:-1], EBINS[1:]):
        k = (e >= lo) & (e < hi)
        s0 = slope(fl[k, :3], ex[k, :3]); s2 = slope(fl[k, 3:], ex[k, 3:])
        # bootstrap errors on the slopes
        idx = np.flatnonzero(k); rng = np.random.default_rng(0)
        bs = np.array([(slope(fl[b, :3], ex[b, :3]), slope(fl[b, 3:], ex[b, 3:]))
                       for b in (rng.choice(idx, len(idx)) for _ in range(200))]).std(0)
        cs = co[k, 9:]
        sat = np.abs(co[k] / bound).max()
        cells = "  ".join(f"{cs[:, i].mean():+.3f}/{cs[:, i].std():.2f}" for i in range(5))
        print(f"{lo:5.2f}-{hi:<6.2f} {k.sum():6d}  {s0:.4f}+/-{bs[0]:.4f}  {s2:.4f}+/-{bs[1]:.4f}  "
              f"{r2(fl[k, 3:], ex[k, 3:]):+.3f}                         {cells}   {sat:.3f}")


if __name__ == "__main__":
    main()
