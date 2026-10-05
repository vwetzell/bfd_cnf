"""Paired m1: varied conditions (varobs) minus fixed (sn8r), same galaxies, J model (2026-09-29).
Rows mapped back to catalog index via exact observed moments; each run keeps its own
selection terms and N_ns; bootstrap over catalog rows."""
import os, re, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B
from bias import bias, window_mask

D, S, F = "../bfd_cnf_imsims/data/", (2.2, 3.2), (3000, 20000)
RUNS = {"sn8r": ("pqr/cv2_s0_sn8r_J.npz", "logs/cv2/bias_cv2_s0_sn8r_J.log", "sn8r"),
        "varobs": ("pqr/cv2_s0_J.npz", "logs/cv2/bias_cv2_s0_J.log", "varobs_sn8")}
# Usage: pair_sn8r_varobs.py [--taper] [SN8R.npz SN8R.log VAROBS.npz VAROBS.log]
#        pair_sn8r_varobs.py [--taper] A.npz A.log A_TAG B.npz B.log B_TAG   (prints B - A)
# --taper: Gaussian-CDF window edges (bias.WINDOW_TAPER = 0.05, 150), float weights,
#          N_ns = NPOP - sum w; the logs must then hold TAPERED selection terms.
if "--taper" in sys.argv:
    sys.argv.remove("--taper")
    B.WINDOW_TAPER = (0.05, 150.0)
if len(sys.argv) == 5:
    RUNS = {"sn8r": (sys.argv[1], sys.argv[2], "sn8r"), "varobs": (sys.argv[3], sys.argv[4], "varobs_sn8")}
elif len(sys.argv) == 7:
    RUNS = {"A": tuple(sys.argv[1:4]), "B": tuple(sys.argv[4:7])}


def sel_terms(log):
    t = open(log).read()
    ps = float(re.search(r"P_s = ([-+.\de]+)", t).group(1))
    qs = np.array(re.search(r"Q_s = \(([-+.\de]+), ([-+.\de]+)\)", t).groups(), float)
    r = np.array(re.search(r"R_s = \[\[([-+.\de]+), ([-+.\de]+)\], \[([-+.\de]+), ([-+.\de]+)\]\]", t).groups(), float)
    return ps, qs, r.reshape(2, 2)


R = {}
for k, (pqr, log, tag) in RUNS.items():
    d = np.load(pqr)
    cat = {a: fitsio.read(D + f"targets_bdg2_{a}_176k_{tag}.fits", columns=["moments", "badcenter"])
           for a in ("g1p02", "g1m02", "g0")}
    n = len(cat["g0"])
    npop = fitsio.read_header(D + f"targets_bdg2_g1p02_176k_{tag}.fits", ext=1)["NPOP"]
    keep = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
    pos = {tuple(r): i for i, r in enumerate(np.asarray(cat["g1p02"]["moments"], np.float64))}
    row = np.array([pos.get(tuple(r), -1) for r in d["obs_plus"]])
    assert (row >= 0).all(), (k, (row < 0).sum())
    q = {s: np.zeros((n, 2)) for s in "pm"}; rr = {s: np.zeros((n, 2, 2)) for s in "pm"}
    q["p"][row], rr["p"][row], q["m"][row], rr["m"][row] = d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"]
    integ = np.zeros(n, bool); integ[row] = True
    sel = {s: (keep & integ) * window_mask(np.asarray(cat[a]["moments"], np.float64), S, F)
           for s, a in (("p", "g1p02"), ("m", "g1m02"))}
    if B.WINDOW_TAPER is None:
        sel = {s: v.astype(bool) for s, v in sel.items()}
    R[k] = dict(q=q, r=rr, sel=sel, npop=npop, st=sel_terms(log), n=n)
    print(k, n, "rows;", round(float(sel["p"].sum())), "in window (plus)")

n = min(R[k]["n"] for k in R)


def m1(k, i):
    x = R[k]; ps, qs, rs = x["st"]
    sp, sm = x["sel"]["p"][i], x["sel"]["m"][i]
    # N_ns scaled to the resampled count: npop - in-window
    ns = ((x["npop"] - sp.sum(), ps, qs, rs), (x["npop"] - sm.sum(), ps, qs, rs))
    return np.array(bias(x["q"]["p"][i], x["r"]["p"][i], x["q"]["m"][i], x["r"]["m"][i], 0.02,
                         sel=(sp, sm), ns=ns))   # (m1, c1, c2)


base = np.arange(n)
ka, kb = list(R)
a, b = m1(ka, base), m1(kb, base)
rng = np.random.default_rng(0)
d = np.array([m1(kb, i) - m1(ka, i) for i in (rng.integers(0, n, n) for _ in range(200))])
e = d.std(0)
print(f"{'taper' if B.WINDOW_TAPER else 'hard'}: {ka} m1 {a[0]:+.5f}  {kb} m1 {b[0]:+.5f}  paired diff "
      f"m1 {b[0] - a[0]:+.5f} +/- {e[0]:.5f}  c1 {b[1] - a[1]:+.2e} +/- {e[1]:.1e}  c2 {b[2] - a[2]:+.2e} +/- {e[2]:.1e}")
