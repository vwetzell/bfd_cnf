"""Corrected m1 with selection terms averaged over several draw-seed logs of ONE flow -- the
k x 2^24-draw estimate bias.py cannot hold on the card at once.  (2026-10-14)  numpy only.
Solve as bias.py's ghat with N_ns: per arm, Q - N_ns Q_s/(1-P_s), R + N_ns (Q_s Q_s/(1-P_s)^2 + R_s/(1-P_s)).
Usage: python dev/sel_seed_combine.py PQR SIZE_LO SIZE_HI FLUX_LO FLUX_HI LOG [LOG ...]"""
import re, sys, os
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B

D, N, G = "../bfd_cnf_imsims/data/", 300000, 0.02
B.WINDOW_TAPER = (0.05, 150.0)
pqr, S, F, logs = sys.argv[1], tuple(map(float, sys.argv[2:4])), tuple(map(float, sys.argv[4:6])), sys.argv[6:]


def terms(log):
    t = open(log).read()
    ps = float(re.search(r"P_s = ([-+.\de]+)", t).group(1))
    qs = np.array(re.search(r"Q_s = \(([-+.\de]+), ([-+.\de]+)\)", t).groups(), float)
    r = np.array(re.search(r"R_s = \[\[([-+.\de]+), ([-+.\de]+)\], \[([-+.\de]+), ([-+.\de]+)\]\]", t).groups(), float)
    m = float(re.search(r"windowed, corrected:\s+m1 = ([-+.\de]+)", t).group(1))
    return ps, qs, r.reshape(2, 2), m


d = np.load(pqr)
cat = {a: fitsio.read(D + f"targets_bdg2_{a}_1m_desv.fits", columns=["moments", "badcenter"], rows=range(N))
       for a in ("g1p02", "g1m02", "g0")}
keep = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
hdr = fitsio.read_header(D + "targets_bdg2_g1p02_1m_desv.fits", ext=1)
npop = round(hdr["NPOP"] * N / hdr["NAXIS2"])


def solve(ps, qs, rs):
    g = {}
    for arm, a in (("plus", "g1p02"), ("minus", "g1m02")):
        w = np.asarray(B.window_mask(d["obs_" + arm], S, F), float)
        n_in = round(float(np.asarray(B.window_mask(np.asarray(cat[a]["moments"], np.float64)[keep], S, F), float).sum()))
        nns = npop - n_in
        Q = (d[arm + "_q"] * w[:, None]).sum(0) - nns * qs / (1 - ps)
        R = -(d[arm + "_r"] * w[:, None, None]).sum(0) + nns * (np.outer(qs, qs) / (1 - ps) ** 2 + rs / (1 - ps))
        g[arm] = np.linalg.solve(R, Q)
    return (g["plus"][0] - g["minus"][0]) / (2 * G) - 1


T = [terms(l) for l in logs]
for l, t in zip(logs, T):
    print(f"  {os.path.basename(l):48s} logged m1 {t[3]:+.5f}  re-solved {solve(*t[:3]):+.5f}  R_s11 {t[2][0, 0]:+.5f}")
avg = [np.mean([t[i] for t in T], axis=0) for i in range(3)]
ms = np.array([solve(*t[:3]) for t in T])
print(f"  {len(T)}-seed terms: m1 {solve(*avg):+.5f}   per-seed sd {ms.std(ddof=1) if len(T) > 1 else 0:.5f} "
      f"-> MC error of the combined {ms.std(ddof=1) / np.sqrt(len(T)) if len(T) > 1 else 0:.5f}")
