"""Is the size-edge `m1` the known O(1/S) bias of the R estimator, or something
bigger?

`R = C/A - (B/A)(B/A)^T` (`bias._merge_finish`), so the squared term inflates
`-R` by `Var_MC(Qhat)` -- and the draw-seed PAIR measures `Var_MC(Qhat)`
directly, per target.  The predicted shift in `m1` from that term alone is

    dm1 = -SUM Var_MC(Qhat) / SUM(-R)

which is a hard number, not a guess.  Compare it with the m1 actually seen.  A
big gap means the square term is not the story: the `C/A` ratio bias at low
ESS, the jackknife falling back to the plain estimator, or a real density
error.  Also reports how concentrated the two sums are, since a bin whose sum
is carried by five targets is not measuring a population.
"""
import os
import numpy as np

A = os.environ.get("PQR_A", "dev/pqr_auto_20k.npz")
B = os.environ.get("PQR_B", "dev/pqr_auto_20k_d7.npz")
G = float(os.environ.get("G", 0.02))
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
PCT = [float(x) for x in os.environ.get("PCT", "0 25 50 75 90 98 100").split()]

from dev.fisher_cross import align  # noqa: E402

if __name__ == "__main__":
    a, b = np.load(A), np.load(B)
    ia, ib = align(a, b)
    m = a["moments"][ia]
    c = m[:, 1] / m[:, 0]
    edges = np.percentile(c, PCT)
    arms = ("plus", "minus")
    print(f"{len(ia)} targets, flux >= {FLUX_LO:g}\n")
    print(f"{'size pct':>10s}{'n':>7s}{'mean -R':>10s}{'VarQ':>9s}"
          f"{'dm1 pred':>10s}{'m1 seen':>10s}{'top5 -R':>9s}{'top5 num':>10s}")
    for i in range(len(edges) - 1):
        s = np.flatnonzero((c >= edges[i]) & (c <= edges[i + 1]))
        ja, jb = ia[s], ib[s]
        negr = np.concatenate([-a[f"{k}_r"][ja, 0, 0] for k in arms])
        vq = np.concatenate([(a[f"{k}_q"][ja, 0] - b[f"{k}_q"][jb, 0]) ** 2 / 2
                             for k in arms])
        num = (a["plus_q"][ja, 0].sum() - a["minus_q"][ja, 0].sum()) / (2 * G)
        den = 0.5 * negr.sum()
        top = lambda v: np.sort(np.abs(v))[-5:].sum() / np.abs(v).sum()
        print(f"{PCT[i]:4.0f}-{PCT[i+1]:<5.0f}{len(s):7d}{negr.mean():10.2f}"
              f"{vq.mean():9.3f}{-0.5 * vq.sum() / den:10.4f}"
              f"{num / den - 1:+10.4f}{top(negr):9.2%}"
              f"{top(a['plus_q'][ja, 0] - a['minus_q'][ja, 0]):10.2%}")
