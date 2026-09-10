"""Per-target Monte-Carlo noise on `Q` and `R`, from two draw-seed runs, by size.

Two runs over the SAME galaxies and noise with different `--draw-seed` give
independent `qhat`, `rhat` per target, so `|a - b| / sqrt(2)` IS the per-target
MC error -- no rerun, no ESS bookkeeping.  That matters because `rhat` is a
ratio estimator: its noise does not average out, it BIASES `|R|` up (see the
`is-ratio-bias-in-R` note), which drives `m1` negative.  If the relative noise
on `R` explodes at the size edge, the edge bias is an estimator artifact and
more draws fix it; if it does not, the edge is a real density error.
"""
import os
import numpy as np

A = os.environ.get("PQR_A", "dev/pqr_auto_20k.npz")
B = os.environ.get("PQR_B", "dev/pqr_auto_20k_d7.npz")
G = float(os.environ.get("G", 0.02))
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
PCT = [float(x) for x in os.environ.get("PCT", "0 25 50 75 90 98 100").split()]
GUARD = float(os.environ.get("GUARD", 1000))

from dev.fisher_cross import align  # noqa: E402  (same matching + guard)

if __name__ == "__main__":
    a, b = np.load(A), np.load(B)
    ia, ib = align(a, b)
    m = a["moments"][ia]
    c = m[:, 1] / m[:, 0]
    edges = np.percentile(c, PCT)
    print(f"{len(ia)} targets, flux >= {FLUX_LO:g}\n"
          f"  MC error from the {A} / {B} draw-seed pair\n")
    print(f"{'size pct':>10s}{'n':>7s}{'Mr/Mf':>8s}{'|R|':>10s}{'sd(R)':>9s}"
          f"{'sd/|R|':>8s}{'|Q|':>10s}{'sd(Q)':>9s}{'sd/|Q|':>8s}{'m1':>9s}")
    for i in range(len(edges) - 1):
        s = np.flatnonzero((c >= edges[i]) & (c <= edges[i + 1]))
        ja, jb = ia[s], ib[s]
        out = []
        for n, sl in (("r", (slice(None), 0, 0)), ("q", (slice(None), 0))):
            va = np.concatenate([a[f"{k}_{n}"][ja][sl] for k in ("plus", "minus")])
            vb = np.concatenate([b[f"{k}_{n}"][jb][sl] for k in ("plus", "minus")])
            sd = np.sqrt(np.mean((va - vb) ** 2) / 2)
            out += [np.sqrt(np.mean(va ** 2)), sd]
        num = (a["plus_q"][ja, 0].sum() - a["minus_q"][ja, 0].sum()) / (2 * G)
        den = 0.5 * (-a["plus_r"][ja, 0, 0].sum() - a["minus_r"][ja, 0, 0].sum())
        print(f"{PCT[i]:4.0f}-{PCT[i+1]:<5.0f}{len(s):7d}{np.median(c[s]):8.2f}"
              f"{out[0]:10.2f}{out[1]:9.2f}{out[1] / out[0]:8.2f}"
              f"{out[2]:10.2f}{out[3]:9.2f}{out[3] / out[2]:8.2f}"
              f"{num / den - 1:+9.4f}")
