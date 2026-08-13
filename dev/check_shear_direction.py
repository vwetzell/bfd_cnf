"""Is the g1 - g2 split real, or did the bootstrap underestimate its error?

`dev/check_noiseless_shear_recovery.py` measures g1 - g2 = -0.0074 +/- 0.0006 in
the top Mr/Mf octile.  `dev/check_isotropy.py` then showed the FLOW is exactly
isotropic (log P invariant to bitwise zero under frame rotation, |Q| and R's
invariants to float32 epsilon), and the catalog population has no net
orientation.  So the split cannot be an equivariance violation -- it has to come
from the finite sample.

The two arms are not measuring the same per-galaxy quantity: arm g1 weights the
second-derivative columns [g1g1, g1g2, g2g2] as [0.5, 0, 0] g^2 and arm g2 as
[0, 0, 0.5] g^2.  Those are different numbers for a given galaxy, equal only in
distribution, so a finite sample splits them.

This scans the shear direction continuously.  For an isotropic flow and an
isotropic population, m(phi) is constant in expectation, and its variation
across phi IS the null distribution of the g1 - g2 statistic -- exactly, with no
bootstrap assumption, because rotating the shear direction is a symmetry of the
problem.  If the observed split sits inside that spread, there is nothing to
explain and the bootstrap was simply too small.

    python dev/check_shear_direction.py [--flow flows/shear.eqx] [--nphi 12]
"""
import argparse
import sys

import equinox as eqx
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402
from check_noiseless_shear_recovery import lensed    # noqa: E402

DATA = "../bfd_cnf_imsims/data/moments.fits"
G = 0.02
NBINS = 8


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--nphi", type=int, default=12)
    a = p.parse_args()

    m, dm, d2m = (np.asarray(v, np.float64) for v in shear_top.load(DATA))
    n90 = int(0.9 * len(m))
    flow = bulk.build_flow(jr.key(0), m[:n90], shear=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)

    # phi over a half turn: the shear direction is a spin-2 object, so pi is a
    # full period and anything beyond it repeats.
    phis = np.linspace(0.0, np.pi, a.nphi, endpoint=False)
    est = {}
    for i, phi in enumerate(phis):
        n = np.array([np.cos(phi), np.sin(phi)])
        for sign in (+1, -1):
            est[i, sign] = tuple(
                np.asarray(v, np.float64)
                for v in bias.pqr(flow, lensed(m, dm, d2m, sign * G * n)))

    def m_of(i, sel):
        (qp, rp), (qm, rm) = est[i, +1], est[i, -1]
        d = (bias.ghat(qp, rp, sel) - bias.ghat(qm, rm, sel)) / (2 * G)
        return d @ np.array([np.cos(phis[i]), np.sin(phis[i])]) - 1.0

    r = m[:, 1] / m[:, 0]
    qs = np.quantile(r, np.linspace(0, 1, NBINS + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    print(f"{a.flow}: m(phi) over {a.nphi} shear directions, |g| = {G}\n")
    print(f"{'<Mr/Mf>':>9s} {'mean':>9s} {'sd over phi':>12s} "
          f"{'ptp':>9s} | {'m(0)-m(90) [= g1-g2]':>21s}")
    for k in range(NBINS):
        sel = (r >= qs[k]) & (r < qs[k + 1])
        v = np.array([m_of(i, sel) for i in range(a.nphi)])
        half = a.nphi // 2
        print(f"{r[sel].mean():9.3f} {v.mean():+9.4f} {v.std(ddof=1):12.4f} "
              f"{np.ptp(v):9.4f} | {v[0] - v[half]:+21.4f}")
    print(f"\nsd over phi is the EXACT null spread of any single-direction "
          f"measurement;\nan honest error bar on m(g1) is that, not the "
          f"galaxy bootstrap.")


if __name__ == "__main__":
    main()
