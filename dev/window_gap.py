"""varobs - sn8r corrected-m1 gap across selection windows, paired over rows.  (2026-10-03)

Per-target Q, R from the full cv6 s2 runs (re-windowed offline; the 1/J target weight is
window-independent); P_s, Q_s, R_s per window from dev/window_shift.py logs (same seed,
so common selection draws).  Rows are the same galaxies in both catalogs, so the gap is
bootstrapped over rows of the full catalog (N_ns = NPOP - replicate in-window count).
A constant gap across windows = population-wide offset; a gap that tracks the size of
the selection correction = the flow's selection model under varied conditions.
Usage: python dev/window_gap.py      (after dev/wshift_gap.sh)
"""
import os, re, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bias import bias, window_mask

D, G = "../bfd_cnf_imsims/data/", 0.02
WINDOWS = {"flux6k": ((2.2, 3.2), (6000, 20000))}   # then the upper size edge, in order
WINDOWS.update({f"size{int(round(10 * e))}" if e != 3.2 else "base": ((2.2, e), (3000, 20000))
                for e in (2.8, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5)})
RUNS = {"A": ("pqr/cv6_s2_sn8r_nopool.npz", "sn8r", "logs/cv6/wshift_sn8r"),
        "B": ("pqr/cv6_s2.npz", "varobs_sn8", "logs/cv6/wshift_varobs")}


def sel_terms(log):
    t = open(log).read()
    ps = float(re.search(r"P_s = ([-+.\de]+)", t).group(1))
    qs = np.array(re.search(r"Q_s = \(([-+.\de]+), ([-+.\de]+)\)", t).groups(), float)
    r = np.array(re.search(r"R_s = \[\[([-+.\de]+), ([-+.\de]+)\], \[([-+.\de]+), ([-+.\de]+)\]\]", t).groups(), float)
    return ps, qs, r.reshape(2, 2)


def load(pqr, tag, logs):
    d = np.load(pqr)
    cat = {a: fitsio.read(D + f"targets_bdg2_{a}_176k_{tag}.fits", columns=["moments", "badcenter"])
           for a in ("g1p02", "g1m02", "g0")}
    n = len(cat["g0"])
    keep = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
    pos = {tuple(r): i for i, r in enumerate(np.asarray(cat["g1p02"]["moments"], np.float64))}
    row = np.array([pos.get(tuple(r), -1) for r in d["obs_plus"]])
    assert (row >= 0).all()
    x = {k: np.zeros((n,) + d[k].shape[1:]) for k in ("plus_q", "plus_r", "minus_q", "minus_r")}
    for k in x:
        x[k][row] = d[k]
    integ = np.zeros(n, bool); integ[row] = True
    x["npop"] = fitsio.read_header(D + f"targets_bdg2_g1p02_176k_{tag}.fits", ext=1)["NPOP"]
    x["sel"] = {}
    for w, (s, f) in WINDOWS.items():
        mp, mm = (window_mask(np.asarray(cat[a]["moments"], np.float64), s, f) & keep for a in ("g1p02", "g1m02"))
        assert (mp & ~integ).sum() < 50 and (mm & ~integ).sum() < 50, (tag, w)   # inside the pre-filter pad
        x["sel"][w] = (mp & integ, mm & integ, sel_terms(f"{logs}/{w}.log"))
    return x


def m1(x, w, i=slice(None), corr=True):
    sp, sm, (ps, qs, rs) = x["sel"][w]
    sp, sm = sp[i], sm[i]
    ns = ((x["npop"] - sp.sum(), ps, qs, rs), (x["npop"] - sm.sum(), ps, qs, rs)) if corr else None
    return bias(x["plus_q"][i], x["plus_r"][i], x["minus_q"][i], x["minus_r"][i], G, sel=(sp, sm), ns=ns)[0]


R = {k: load(*v) for k, v in RUNS.items()}
n = len(R["A"]["plus_q"])
rng = np.random.default_rng(0)
idx = [rng.integers(0, n, n) for _ in range(200)]
res = {}
for w in WINDOWS:
    v = {(k, c): m1(R[k], w, corr=c) for k in "AB" for c in (False, True)}
    b = {(k, c): np.array([m1(R[k], w, i, c) for i in idx]) for k in "AB" for c in (False, True)}
    res[w] = v, b
    print(w, "done", flush=True)
print(f"\n{'window':>7s} {'P_s A/B':>13s} {'R_s11 A/B':>15s} {'corr A':>8s} {'corr B':>8s} "
      f"{'m1 A':>9s} {'m1 B':>9s} {'gap uncorr':>18s} {'gap corr':>18s} {'d(gap) vs base':>18s}")
gb = res["base"][1][("B", True)] - res["base"][1][("A", True)]
for w, (v, b) in res.items():
    sa, sb = R["A"]["sel"][w][2], R["B"]["sel"][w][2]
    gu, gc = b[("B", False)] - b[("A", False)], b[("B", True)] - b[("A", True)]
    g0 = res["base"][0][("B", True)] - res["base"][0][("A", True)]
    print(f"{w:>7s} {sa[0]:.4f}/{sb[0]:.4f} {sa[2][0, 0]:+.4f}/{sb[2][0, 0]:+.4f} "
          f"{v['A', True] - v['A', False]:+8.4f} {v['B', True] - v['B', False]:+8.4f} "
          f"{v['A', True]:+9.5f} {v['B', True]:+9.5f} "
          f"{v['B', False] - v['A', False]:+.5f}+/-{gu.std():.5f} {v['B', True] - v['A', True]:+.5f}+/-{gc.std():.5f} "
          f"{v['B', True] - v['A', True] - g0:+.5f}+/-{(gc - gb).std():.5f}")
# Steps between ADJACENT upper size edges, paired: a model error at the edge moves the gap
# smoothly; a fluctuation carried by the edge-crossers moves it in independent jumps.
edges = [w for w in WINDOWS if w != "flux6k"]
gap = lambda w, k=None: (res[w][0][("B", True)] - res[w][0][("A", True)]) if k is None else \
    (res[w][1][("B", True)] - res[w][1][("A", True)])
print("\nupper-edge step      d(gap)            ")
steps = []
for u, v in zip(edges, edges[1:]):
    d, e = gap(v) - gap(u), (gap(v, 1) - gap(u, 1)).std()
    steps.append((d, e))
    print(f"  {u:>7s} -> {v:<7s} {d:+.5f}+/-{e:.5f}  ({d / e:+.1f} sigma)")
d, e = np.array(steps).T
print(f"  chi2 of steps vs 0: {((d / e)**2).sum():.1f}/{len(d)}   (steps share no rows' window flips -> ~independent)")
print("(errors: row bootstrap, paired; selection-term MC error not included)")
