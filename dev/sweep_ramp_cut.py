"""The Mr/Mf ramp, judged only where a real analysis would keep the galaxies.

Everything above `Mr/Mf = 3.5` is out of scope here: that tail is a known,
separately diagnosed flow-density failure at the point-source boundary
(memory: `resolution-edge-bias-is-real`), and cutting it is legitimate because
the cut is spin-0 and evaluated on the SAME galaxies in the +g and -g arms, so
`dP(s|g)/dg = 0` and the eq. (45)-(46) selection terms stay negligible.  What is
left below 3.5 is a monotone ramp -- m1 running -1% to +3% with resolution --
and that is what this measures.

Metric: `dev/check_noiseless_shear_recovery.py` restricted to `Mr/Mf < 3.5`,
re-binned into octiles WITHIN the cut, summarised by the slope of m1 against
Mr/Mf and by RMS(m1) across those bins.  Global m1 is reported but must not be
optimised -- it cancels along the ramp.  This is the noiseless path, so it sees
the flow's density and response and nothing of the noisy-target machinery.

    python dev/sweep_ramp_cut.py --eval-only        # baseline flow on disk
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

DATA = "../bfd_cnf_imsims/data/moments.fits"
G = 0.02
CUT = 3.5          # evaluation boundary: nothing above this is measured
NBINS = 8


def recovery(flow, m, dm, d2m, n90):
    """m1 in octiles of Mr/Mf BELOW the cut, plus its slope and RMS.

    One shear direction (g1) is enough here: the direction split only dominates
    in the extreme octiles, which the cut has already removed.
    """
    shear_m = lambda s: m + s * G * dm[:, 0, :] + 0.5 * (s * G) ** 2 * d2m[:, 0, :]
    p = [np.asarray(v, np.float64)
         for s in (+1, -1) for v in bias.pqr(flow, shear_m(s))]
    m1 = lambda sel: bias.bias(p[0], p[1], p[2], p[3], G, sel)[0]

    r = m[:, 1] / m[:, 0]
    rng = np.random.default_rng(0)

    def err(sel, n=200):
        i0 = np.flatnonzero(sel)
        return float(np.std([m1(rng.choice(i0, len(i0))) for _ in range(n)]))

    out = {}
    for lab, base in (("train", np.arange(len(m)) < n90),
                      ("heldout", np.arange(len(m)) >= n90)):
        base = base & (r < CUT)
        e = np.quantile(r[base], np.linspace(0, 1, NBINS + 1))
        e[0], e[-1] = -np.inf, np.inf
        binned = [(float(r[s].mean()), float(m1(s)), err(s))
                  for k in range(NBINS)
                  for s in [base & (r >= e[k]) & (r < e[k + 1])]]
        x = np.array([b[0] for b in binned])
        y = np.array([b[1] for b in binned])
        # Slope of the ramp itself -- the number that says whether a tomographic
        # bin with a different size distribution gets a different m1.
        slope = float(np.polyfit(x, y, 1)[0])
        out[lab] = {"all": float(m1(base)), "all_err": err(base),
                    "slope": slope, "rms": float(np.sqrt((y ** 2).mean())),
                    "ptp": float(y.max() - y.min()), "bins": binned}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-only", default="flows/shear.eqx", nargs="?",
                   const="flows/shear.eqx",
                   help="score an existing checkpoint")
    a = p.parse_args()

    m, dm, d2m = (np.asarray(v, np.float64) for v in shear_top.load(DATA))

    n90 = int(0.9 * len(m))
    flow = bulk.build_flow(jr.key(0), m[:n90], shear=True)
    flow = eqx.tree_deserialise_leaves(a.eval_only, flow)
    rec = recovery(flow, m, dm, d2m, n90)
    for lab in ("train", "heldout"):
        print(f"\n{a.eval_only}  {lab}  Mr/Mf < {CUT}")
        print(f"  m1 {rec[lab]['all']:+.4f} +/- {rec[lab]['all_err']:.4f}  "
              f"slope {rec[lab]['slope']:+.4f}  RMS {rec[lab]['rms']:.4f}  "
              f"ptp {rec[lab]['ptp']:.4f}")
        for c, v, e in rec[lab]["bins"]:
            print(f"    {c:6.3f}  {v:+.4f} +/- {e:.4f}")


if __name__ == "__main__":
    main()
