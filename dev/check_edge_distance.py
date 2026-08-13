"""Is the size-axis bias set by the CHART ceiling, or by the population's edge?

Both are boundaries on the same axis, and on one population they are impossible
to tell apart.  Two populations separate them, because the chart ceiling
(`POINT_SOURCE`, z0 = +inf) is shared while the support edge is not: bulgedisc
reaches z0 = +5.5, gauss2 only +3.5.

    chart-ceiling hypothesis -> the two m1 curves line up in z0
    support-edge hypothesis  -> they line up in (z0_edge - z0)

Reads the cached noiseless PQR from `dev/plot_bias_flux_size.py` (one per
population), so this costs no flow evaluation.

    python dev/check_edge_distance.py
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
from models.bijections import POINT_SOURCE           # noqa: E402

G, NBOOT = 0.02, 60
CACHES = {"bulgedisc": "dev/bias_flux_size_cache.npz",
          "gauss2": "dev/bias_flux_size_cache_gauss2.npz"}
# The support edge as a high quantile, not the max: one stray target should not
# define where a population stops.
EDGE_Q = 0.999


def profile(z, qp, rp, qm, rm, edges, rng):
    """m1 in the given bins of `z`, with a paired galaxy bootstrap."""
    out = {}
    for k in range(len(edges) - 1):
        s = (z >= edges[k]) & (z < edges[k + 1])
        if s.sum() < 200:
            continue
        idx = np.flatnonzero(s)
        v = bias.bias(qp[s], rp[s], qm[s], rm[s], G)[0]
        boot = [bias.bias(qp[i], rp[i], qm[i], rm[i], G)[0]
                for i in (rng.choice(idx, len(idx)) for _ in range(NBOOT))]
        out[k] = (v, float(np.std(boot)), int(s.sum()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nbin", type=int, default=28)
    a = p.parse_args()
    rng = np.random.default_rng(0)

    runs = {}
    for pop, path in CACHES.items():
        d = np.load(path)
        z = np.log(d["y"] / (POINT_SOURCE - d["y"]))
        runs[pop] = (z, np.quantile(z, EDGE_Q),
                     *(d[k] for k in ("qp", "rp", "qm", "rm")))
        print(f"{pop:10s}: {len(z)} targets, z0 in [{z.min():+.2f}, {z.max():+.2f}], "
              f"edge (q={EDGE_Q}) = {runs[pop][1]:+.3f}")

    for label, fn in (("z0  (shared chart ceiling at +inf)", lambda z, e: z),
                      ("z0_edge - z0  (own support edge at 0)", lambda z, e: e - z)):
        print(f"\n=== m1 vs {label} ===")
        cols = {pop: fn(z, e) for pop, (z, e, *_) in runs.items()}
        lo, hi = (min(np.quantile(c, 0.002) for c in cols.values()),
                  max(np.quantile(c, 0.998) for c in cols.values()))
        edges = np.linspace(lo, hi, a.nbin + 1)
        prof = {pop: profile(cols[pop], *runs[pop][2:], edges, rng) for pop in runs}
        print(f"  {'bin centre':>11s} " + " ".join(f"{p:>22s}" for p in runs))
        for k in range(a.nbin):
            row = [prof[pop].get(k) for pop in runs]
            if not any(row):
                continue
            cells = [f"{v:+.4f}+/-{e:.4f}({n // 1000:3d}k)" if r else " " * 22
                     for r, (v, e, n) in ((r, r or (0, 0, 0)) for r in row)]
            print(f"  {0.5 * (edges[k] + edges[k + 1]):+11.3f} " + " ".join(cells))


if __name__ == "__main__":
    main()
