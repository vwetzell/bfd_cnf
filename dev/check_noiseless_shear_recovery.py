"""Does the flow recover shear from NOISELESS galaxies?  Binned by resolution.

This strips away everything the noisy-target path adds -- kernel importance
sampling, the C_M convolution, the centroid layer -- and leaves only the flow's
density.  Targets are the exact catalog moments, sheared with the catalog's own
exact `dm_dg` / `d2m_dg2`, so the applied shear is right by construction and PQR
is a plain point evaluation (`bias.pqr` with no draws).

Any m that survives here is the flow's P(m|g) being wrong, and nothing else.
The last 10% is the split `shear.py train` held out, so it is reported apart
from the 90% the flow actually saw.

Three arms, because ONE SHEAR DIRECTION CANNOT SEE THE WHOLE RESPONSE.  A pure
g1 shear puts weight [0.5 g^2, 0, 0] on the second-derivative columns
[g1g1, g1g2, g2g2], so the mixed column never enters -- and that column is the
worst-fit part of the layer by an order of magnitude (36-51% against 1.6-6% on
the diagonals, `shear.py check` split by column).  The diagonal arm weights it
[0.25, 0.5, 0.25] g^2, i.e. more than either diagonal column, so it is the arm
that can see it.  Measured: `diag - mean(g1,g2)` is consistent with ZERO in
every octile, so that column carries no bias despite being badly fitted.

**g1 - g2 is NOT an equivariance test.**  The flow is isotropic to float32
(`dev/check_isotropy.py`), but the two arms still weight the second-derivative
columns differently -- [0.5,0,0] g^2 against [0,0,0.5] g^2 -- and those are
different numbers for a given galaxy, equal only in distribution.  A finite
sample therefore splits them, and in the EXTREME octiles that split is the
dominant error: the bootstrap below resamples galaxies but not shear DIRECTION,
and understates the error on any single-direction m1 by 2-5x there (top octile:
+/- 0.0028 against a bootstrap 0.0006; the six middle octiles are fine at 1e-4).
Use `dev/check_shear_direction.py` for an honest error bar on the edge bins.

Reported per arm: the multiplicative bias ALONG the applied direction, the
leakage into the perpendicular one (zero by symmetry), and the additive pair.

    python dev/check_noiseless_shear_recovery.py [--flow flows/shear.eqx]
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

DATA = "../bfd_cnf_imsims/data"
G = 0.02
NBINS = 8
S = 1.0 / np.sqrt(2.0)
# name -> unit vector in (g1, g2).  The applied shear is +/- G * this.
ARMS = {"g1": (1.0, 0.0), "g2": (0.0, 1.0), "diag": (S, S)}


def lensed(m, dm, d2m, g):
    """Exact moments under a single global shear g, through second order.

    `d2m` columns are [g1g1, g1g2, g2g2] and carry no factor of 1/2, matching
    `shear.lens`: m + Q.g + 1/2 g.R.g expands to the quad weights below.
    """
    quad = np.array([0.5 * g[0] ** 2, g[0] * g[1], 0.5 * g[1] ** 2])
    return (m + np.einsum("i,bij->bj", np.asarray(g), dm)
            + np.einsum("i,bij->bj", quad, d2m))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--boot", type=int, default=200)
    a = p.parse_args()

    m, dm, d2m = shear_top.load(f"{DATA}/moments.fits")
    m, dm, d2m = (np.asarray(v, np.float64) for v in (m, dm, d2m))
    n90 = int(0.9 * len(m))
    flow = bulk.build_flow(jr.key(0), m[:n90], shear=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    print(f"{a.flow} on {len(m)} exact noiseless galaxies, |g| = {G}")

    pqr = {}
    for name, n in ARMS.items():
        for sign in (+1, -1):
            g = sign * G * np.asarray(n)
            pqr[name, sign] = tuple(
                np.asarray(v, np.float64)
                for v in bias.pqr(flow, lensed(m, dm, d2m, g)))

    def est(name, sel):
        """(m along the arm, m leaked perpendicular, c1, c2)."""
        (qp, rp), (qm, rm) = pqr[name, +1], pqr[name, -1]
        gp, gm = bias.ghat(qp, rp, sel), bias.ghat(qm, rm, sel)
        n = np.asarray(ARMS[name])
        d = (gp - gm) / (2 * G)
        return (d @ n - 1.0, d @ np.array([-n[1], n[0]]),
                *(0.5 * (gp + gm)))

    rng = np.random.default_rng(0)

    def err(name, sel, which=0):
        i0 = np.flatnonzero(sel)
        draws = [est(name, rng.choice(i0, len(i0)))[which]
                 for _ in range(a.boot)]
        return float(np.std(draws))

    r = m[:, 1] / m[:, 0]
    for lab, base in (("train 90%", np.arange(len(m)) < n90),
                      ("held-out 10%", np.arange(len(m)) >= n90)):
        print(f"\n{lab}  (N={base.sum()})")
        print(f"{'arm':>6s} {'m (along)':>18s} {'m (perp)':>18s} "
              f"{'c1':>10s} {'c2':>10s}")
        for name in ARMS:
            v = est(name, base)
            print(f"{name:>6s} {f'{v[0]:+.4f} +/- {err(name, base):.4f}':>18s} "
                  f"{f'{v[1]:+.4f} +/- {err(name, base, 1):.4f}':>18s} "
                  f"{v[2]:+10.5f} {v[3]:+10.5f}")

        q = np.quantile(r[base], np.linspace(0, 1, NBINS + 1))
        q[0], q[-1] = -np.inf, np.inf
        # Both contrasts are PAIRED: one galaxy resample per draw, every arm
        # evaluated on it, so the population scatter that dominates each arm's
        # own error cancels and the difference is far better determined.
        def paired(sel, f):
            i0 = np.flatnonzero(sel)
            d = [f({n: est(n, i)[0] for n in ARMS})
                 for i in (rng.choice(i0, len(i0)) for _ in range(a.boot))]
            return float(np.std(d))

        # `mix` is the part of the bias only a MIXED shear can produce, which is
        # where the g1g2 column lives; `aniso` is a pure equivariance test, zero
        # for any isotropic flow.
        mix = lambda v: v["diag"] - 0.5 * (v["g1"] + v["g2"])
        aniso = lambda v: v["g1"] - v["g2"]
        print(f"\n{'<Mr/Mf>':>9s} {'N':>7s}" +
              "".join(f"{'m ' + k:>10s}" for k in ARMS) +
              f"{'diag-mean':>19s} {'g1-g2':>19s}")
        for k in range(NBINS):
            s = base & (r >= q[k]) & (r < q[k + 1])
            v = {name: est(name, s)[0] for name in ARMS}
            print(f"{r[s].mean():9.3f} {s.sum():7d}" +
                  "".join(f"{v[name]:+10.4f}" for name in ARMS) +
                  f"{f'{mix(v):+.4f} +/- {paired(s, mix):.4f}':>19s}"
                  f"{f'{aniso(v):+.4f} +/- {paired(s, aniso):.4f}':>19s}")


if __name__ == "__main__":
    main()
