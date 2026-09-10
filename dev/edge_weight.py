"""Does the estimator's per-target weight diverge at the point-source limit?

The chart's slot 1 is logit(u), u = Mr/(POINT_SOURCE Mf).  Its Jacobian carries
1/(1-u), so the shear response IN CHART COORDINATES is

    d logit(u) / dg  =  [1/(1-u)] * d ln(Mr/Mf) / dg

and the catalog's own dm_dg says the physical factor does NOT vanish as u -> 1.
So d z / dg diverges at the boundary, and since Q ~ dz/dg and R ~ (dz/dg)^2,
the per-target contributions should grow like 1/(1-u) and 1/(1-u)^2.

This checks that against the real per-target Q and R of the 20k run, binned on
the g = 0 arm (independent of both arms' noise and of g).

    PYTHONPATH=. python dev/edge_weight.py
"""
import os

import numpy as np

from models.bijections import POINT_SOURCE

PQR = os.environ.get("PQR", "dev/pqr_v3_s3_20k.npz")


def main():
    d = np.load(PQR)
    ok = np.ones(len(d["plus_q"]), dtype=bool)
    for k in ("plus", "minus"):
        ok &= np.isfinite(d[f"{k}_q"]).all(1)
        ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    q = d["plus_q"][ok][:, 0]
    r = -d["plus_r"][ok][:, 0, 0]
    m0 = d["moments"][ok]
    size = m0[:, 1] / m0[:, 0]
    u = size / POINT_SOURCE
    amp = 1.0 / (1.0 - np.clip(u, None, 1 - 1e-9))
    n = len(q)
    print(f"{n} targets, g=0-arm Mr/Mf: p50 {np.median(size):.3f}  "
          f"p99 {np.percentile(size, 99):.3f}  max {size.max():.4f}  "
          f"(POINT_SOURCE {POINT_SOURCE})")
    print(f"fraction above the ceiling on the NOISY g=0 arm: "
          f"{(size >= POINT_SOURCE).mean():.2%}\n")

    edges = [0, 2.8, 3.0, 3.2, 3.4, 3.5, 3.6, POINT_SOURCE, 99]
    print(f"{'Mr/Mf band':>14s} {'n':>6s} {'%n':>6s} {'1/(1-u)':>9s} "
          f"{'mean|q1|':>10s} {'mean -R11':>11s} {'%|q1| sum':>10s} "
          f"{'%-R11 sum':>10s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (size >= lo) & (size < hi)
        if b.sum() == 0:
            continue
        a = amp[b].mean() if hi <= POINT_SOURCE else np.inf
        print(f"  {lo:6.3f}-{hi:6.3f} {int(b.sum()):6d} {b.mean():6.2%} "
              f"{a:9.1f} {np.abs(q[b]).mean():10.3g} {r[b].mean():11.3g} "
              f"{np.abs(q[b]).sum() / np.abs(q).sum():10.2%} "
              f"{r[b].sum() / r.sum():10.2%}")

    print(f"\ntop-|q1| targets, and where they sit:")
    idx = np.argsort(-np.abs(q))[:10]
    print(f"  {'Mr/Mf':>8s} {'1/(1-u)':>9s} {'Mf':>9s} {'q1':>12s} {'-R11':>12s}")
    for i in idx:
        print(f"  {size[i]:8.4f} {amp[i]:9.1f} {m0[i, 0]:9.0f} "
              f"{q[i]:12.4g} {r[i]:12.4g}")

    inb = size < POINT_SOURCE
    for lab, v in (("|q1|", np.abs(q)), ("-R11", r)):
        c = np.corrcoef(np.log(amp[inb]), np.log(np.abs(v[inb]) + 1e-30))[0, 1]
        sl = np.polyfit(np.log(amp[inb]), np.log(np.abs(v[inb]) + 1e-30), 1)[0]
        print(f"\nlog {lab} vs log 1/(1-u): slope {sl:+.3f}, corr {c:+.3f}"
              f"   (divergence predicts {1.0 if lab == '|q1|' else 2.0:+.1f})")


if __name__ == "__main__":
    main()
