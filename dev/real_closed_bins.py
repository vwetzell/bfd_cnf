"""Where does the sn8 windowed m1 excess sit?  Sims minus closed loop, per observed bin. (2026-10-01)

Same flow (cv2 s0) on both sides, so the selection terms are shared and the
uncorrected difference IS the corrected difference.  Closed loop = targets drawn
from the flow's own J-tilted density (dev/closed_sn8r_J.sh, m1 +0.0002); sims =
sn8r (fixed conditions) and varobs renders.  Per bin of each arm's OWN observed
Mr/Mf or Mf (as window_mask does): uncorrected m1 and its R-weight share w_b, so
the window value is ~ sum_b w_b m1_b and bin b contributes w_b (sims_b - closed_b).
Offline numpy.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bias import bias, window_mask

S, F = (2.2, 3.2), (3000, 20000)
RUNS = {"closedJ": "pqr/closedJ_sn8r_J.npz", "sn8r": "pqr/cv2_s0_sn8r_J.npz", "varobs": "pqr/cv2_s0_J.npz"}
RUNS.update({a: f"pqr/{a}.npz" for a in sys.argv[1:]})      # extra runs, e.g. cv4_s2 (different flow: not paired to closedJ)
BINS = {"Mr/Mf": (lambda o: o[:, 1] / o[:, 0], [2.2, 2.3, 2.45, 2.7, 2.95, 3.1, 3.2]),
        "Mf": (lambda o: o[:, 0], [3000, 4000, 5000, 7000, 10000, 20000])}


def load(path):
    d = np.load(path)
    k = {x: d[x] for x in ("plus_q", "plus_r", "minus_q", "minus_r", "obs_plus", "obs_minus")}
    for a in ("plus", "minus"):
        ok = np.isfinite(k[a + "_q"]).all(1) & np.isfinite(k[a + "_r"]).all((1, 2))
        k["in_" + a] = window_mask(k["obs_" + a], S, F) & ok
    return k


def stat(k, sp, sm, i=slice(None)):
    m1 = bias(k["plus_q"][i], k["plus_r"][i], k["minus_q"][i], k["minus_r"][i], 0.02, sel=(sp[i], sm[i]))[0]
    w = -(k["plus_r"][i][sp[i], 0, 0].sum() + k["minus_r"][i][sm[i], 0, 0].sum())
    return m1, w


rng = np.random.default_rng(0)
R = {n: load(p) for n, p in RUNS.items()}
for bname, (f, edges) in BINS.items():
    res = {}
    for n, k in R.items():
        N = len(k["plus_q"])
        boot = [rng.integers(0, N, N) for _ in range(100)]
        tot = stat(k, k["in_plus"], k["in_minus"])[1]
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sp = k["in_plus"] & (f(k["obs_plus"]) > lo) & (f(k["obs_plus"]) < hi)
            sm = k["in_minus"] & (f(k["obs_minus"]) > lo) & (f(k["obs_minus"]) < hi)
            m1, w = stat(k, sp, sm)
            e = np.std([stat(k, sp, sm, b)[0] for b in boot])
            rows.append((m1, e, w / tot, sp.sum()))
        res[n] = rows
    print(f"\n{bname} bin        " + "".join(f"{n:>22s}" for n in R) + "   sims-closed contribution (w_b * diff)")
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        line = f"{lo:>6g}-{hi:<6g} w {res['closedJ'][j][2]:.3f} "
        line += "".join(f"  {res[n][j][0]:+.4f}+/-{res[n][j][1]:.4f}" for n in R)
        c = res["closedJ"][j]
        for n in [x for x in R if x != "closedJ"]:
            s = res[n][j]
            line += f"  {n} {s[2] * (s[0] - c[0]):+.4f}+/-{s[2] * np.hypot(s[1], c[1]):.4f}"
        print(line)
    for n in [x for x in R if x != "closedJ"]:
        tot = sum(res[n][j][2] * (res[n][j][0] - res["closedJ"][j][0]) for j in range(len(edges) - 1))
        print(f"  sum of contributions {n}: {tot:+.4f}")
