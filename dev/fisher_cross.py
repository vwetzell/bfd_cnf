"""Truth-free Fisher check, cross-fit for free from two draw-seed runs.

`E[q q^T] = E[-R]` holds for the true density; the estimator's own metric is
`-SUM r`, so `SUM q^2 / SUM -r` per flux bin says whether the model's second
derivative matches the score it produces -- no ground truth needed.

`SUM qhat^2` is biased up by the Monte-Carlo noise in `qhat`.  Two runs over
the SAME galaxies and noise with different `--draw-seed` give independent
`qhat`, so `SUM qhat_a qhat_b` is that same quantity CROSS-FIT: the noise is
independent between runs and drops out of the expectation.  No rerun needed.

    PQR_A=dev/pqr_auto_20k.npz PQR_B=dev/pqr_auto_20k_d7.npz python dev/fisher_cross.py
"""
import os
import numpy as np

A = os.environ.get("PQR_A", "dev/pqr_auto_20k.npz")
B = os.environ.get("PQR_B", "dev/pqr_auto_20k_d7.npz")
G = float(os.environ.get("G", 0.02))
NB = int(os.environ.get("NB", 5))
BOOT = int(os.environ.get("BOOT", 200))
BINBY = os.environ.get("BINBY", "flux")
FLUX_LO = float(os.environ.get("FLUX_LO", 0))
GUARD = float(os.environ.get("GUARD", 1000))    # sane_targets factor, 0 = off
PCT = os.environ.get("PCT", "")                 # explicit percentile edges


def align(a, b):
    """Rows of both runs, matched on the clean truth moments (runs may drop a
    different handful of targets)."""
    key = lambda d: [tuple(np.round(r, 9)) for r in d["moments"]]
    ka, kb = key(a), key(b)
    pos = {k: i for i, k in enumerate(kb)}
    ia = [i for i, k in enumerate(ka) if k in pos]
    ib = [pos[ka[i]] for i in ia]
    fin = lambda d, idx: np.all([np.isfinite(d[f"{k}_{n}"][idx]).reshape(
        len(idx), -1).all(1) for k in ("plus", "minus") for n in "qr"], axis=0)
    ok = fin(a, np.array(ia)) & fin(b, np.array(ib))
    ia, ib = np.array(ia)[ok], np.array(ib)[ok]
    ok = a["flux"][ia] >= FLUX_LO
    if GUARD:                                   # bias.sane_targets, both runs
        for d, i in ((a, ia), (b, ib)):
            for n, sl in (("q", (slice(None), 0)), ("r", (slice(None), 0, 0))):
                v = [np.abs(d[f"{k}_{n}"][i][sl]) for k in ("plus", "minus")]
                med = np.median(np.concatenate(v))
                for x in v:
                    ok &= x < GUARD * med
    return ia[ok], ib[ok]


def stats(a, b, ia, ib):
    """(m1_newton, m1_fisher, SUM q_a q_b / SUM -r, plain/cross inflation)."""
    qa = {k: a[f"{k}_q"][ia, 0] for k in ("plus", "minus")}
    qb = {k: b[f"{k}_q"][ib, 0] for k in ("plus", "minus")}
    ra = {k: a[f"{k}_r"][ia, 0, 0] for k in ("plus", "minus")}
    num = (qa["plus"].sum() - qa["minus"].sum()) / (2 * G)
    den = 0.5 * (-ra["plus"].sum() + -ra["minus"].sum())
    cross = 0.5 * sum((qa[k] * qb[k]).sum() for k in qa)
    plain = 0.5 * sum((qa[k] ** 2).sum() for k in qa)
    return num / den - 1, num / cross - 1, cross / den, plain / cross


if __name__ == "__main__":
    a, b = np.load(A), np.load(B)
    ia, ib = align(a, b)
    m = a["moments"][ia]
    f = {"flux": a["flux"][ia], "size": m[:, 1] / m[:, 0],
         "e": np.hypot(m[:, 2], m[:, 3]) / m[:, 1]}[BINBY]
    print(f"{len(ia)} targets kept (flux >= {FLUX_LO:g}, guard {GUARD:g}) across\n  {A}\n  {B}")
    q = ([float(x) for x in PCT.split()] if PCT
         else list(np.linspace(0, 100, NB + 1)))
    edges = np.percentile(f, q)
    rng = np.random.default_rng(0)
    print(f"\nbinned by {BINBY}, edges "
          + " ".join(f"{e:.4g}" for e in edges))
    print(f"\n{'bin':>5s}{'n':>7s}{'m1 (-SUM r)':>20s}{'m1 (SUM qq cross)':>24s}"
          f"{'qq/-r':>9s}{'plain/cross':>13s}")
    rows = [("all", np.arange(len(ia)))]
    rows += [(f"{q[i]:.0f}-{q[i+1]:.0f}",
              np.flatnonzero((f >= edges[i]) & (f <= edges[i + 1])))
             for i in range(len(edges) - 1)]
    for name, sub in rows:
        v = stats(a, b, ia[sub], ib[sub])
        s = np.std([stats(a, b, ia[j], ib[j]) for j in
                    (rng.choice(sub, len(sub)) for _ in range(BOOT))],
                   axis=0, ddof=1)
        print(f"{name:>5s}{len(sub):7d}{v[0]:+12.4f} +/- {s[0]:.4f}"
              f"{v[1]:+15.4f} +/- {s[1]:.4f}{v[2]:9.4f}{v[3]:13.5f}")
