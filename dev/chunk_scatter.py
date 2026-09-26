"""Real vs closed loop with an honest error: chunk scatter.

Each closed-loop PQR file is split into population-sized chunks (65372 = the
real 22k catalog's NPOP), and windowed m1 (corrected, uncorrected, and the
uncorrected Mr/Mf edge bins) is computed per chunk.  The chunk sd is the
sampling error of ONE real-22k-sized catalog; a real catalog of K populations
has sd/sqrt(K).  Selection terms (P_s, Q_s, R_s) are parsed from each run's
bias.py log (same flow and window everywhere).

usage: python dev/chunk_scatter.py
"""
import re
import sys
sys.path.insert(0, ".")
import numpy as np

import bias as B

NPOP = 65372
SIZE, FLUX = (2.2, 3.2), (3000.0, 20000.0)
BINS = {"lower edge 2.2-2.3": (2.2, 2.3), "interior 2.45-2.95": (2.45, 2.95), "upper edge 3.1-3.2": (3.1, 3.2)}
CLOSED = [("pqr/closed_g2v4n_Ke_4x.npz", "logs/bias_closed_g2v4n_Ke_4x.log", 4),
          ("pqr/closed8_g2v4n_Ke.npz", "logs/bias_closed8_g2v4n_Ke.log", 8)]
REAL = [("real 22k", "gauss2_v4n", "pqr/g2v4n_Ke_20k.npz", "logs/bias_g2v4n_Ke_20k.log"),
        ("real 176k", "gauss2_v4n_176k", "pqr/g2v4n_Ke_176k.npz", "logs/bias_g2v4n_Ke_176k.log")]


def sel_terms(log):
    t = open(log).read()
    num = r"([-+]?[\d.]+(?:e[-+]?\d+)?)"
    ps = float(re.search(r"P_s = " + num, t).group(1))
    qs = np.array(re.search(r"Q_s = \(" + num + r", " + num, t).groups(), float)
    rs = np.array(re.search(r"R_s = \[\[" + num + r", " + num + r"\], \[" + num + r", " + num, t).groups(),
                  float).reshape(2, 2)
    return ps, qs, rs


def stats(d, idx, npop, st):
    q = tuple(d[k][idx] for k in ("plus_q", "plus_r", "minus_q", "minus_r"))
    sp, sm = B.window_mask(d["obs_plus"][idx], SIZE, FLUX), B.window_mask(d["obs_minus"][idx], SIZE, FLUX)
    ns = ((npop - sp.sum(),) + st, (npop - sm.sum(),) + st)
    out = {"corrected": B.bias(*q, 0.02, sel=(sp, sm), ns=ns)[0],
           "uncorrected": B.bias(*q, 0.02, sel=(sp, sm))[0]}
    r = d["moments"][idx, 1] / d["moments"][idx, 0]
    for name, (lo, hi) in BINS.items():
        s = (r >= lo) & (r < hi)
        out[name] = B.bias(*q, 0.02, sel=(sp & s, sm & s))[0]
    return out


chunks = []
for f, log, k in CLOSED:
    try:
        d = np.load(f)
    except FileNotFoundError:
        print(f"(missing {f})")
        continue
    st, n = sel_terms(log), len(d["plus_q"])
    chunks += [stats(d, np.arange(i * n // k, (i + 1) * n // k), NPOP, st) for i in range(k)]
keys = list(chunks[0])
C = np.array([[c[k] for k in keys] for c in chunks])
mu, sd = C.mean(0), C.std(0, ddof=1)
print(f"{len(C)} closed chunks of one real-22k population each")
print(f"{'':22s}" + "".join(f"{k:>20s}" for k in keys))
print(f"{'closed mean':22s}" + "".join(f"{x:>+20.4f}" for x in mu))
print(f"{'chunk sd (1 pop)':22s}" + "".join(f"{x:>20.4f}" for x in sd))
for name, pop, f, log in REAL:
    try:
        d = np.load(f)
    except FileNotFoundError:
        print(f"(missing {f})")
        continue
    import fitsio
    npop = fitsio.read_header(f"../bfd_cnf_imsims/data/{B.CATALOGS[pop]['plus']}.fits", 1)["NPOP"]
    k = npop / NPOP
    r = stats(d, slice(None), npop, sel_terms(log))
    err = np.hypot(sd / np.sqrt(k), sd / np.sqrt(len(C)))
    print(f"{name:22s}" + "".join(f"{r[x]:>+20.4f}" for x in keys))
    print(f"{'  real - closed':22s}" + "".join(f"{r[x] - m:>+12.4f}+/-{e:.4f}" for x, m, e in zip(keys, mu, err)))
