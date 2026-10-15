"""Where in the window does the desv - noiseless-template m1 gap live?  CPU, offline.  (2026-10-10)

  JAX_PLATFORMS=cpu python dev/strip_pair.py [A N]      (default s3 n15c_s3; any two of RUNS)

Paired runs on the SAME 300k targets (dev/desv_noiseless.sh): A = desv s3 (noisy bdg2d templates),
N = n15c s3 (bdg2n templates, everything else identical).  Swap decomposition of Delta = m(A) - m(N):
  strip S:   m(A's Q,R inside S, N's elsewhere; N's selection terms) - m(N)
  selection: m(N's Q,R; A's selection terms) - m(N)
Strips are regions of the g = 0 arm's moments (`moments`, independent of both arms' noise, so binning
on it selects nothing): faint Mf < 3000, bright Mf > 20000, large Mr/Mf > 3.2, small Mr/Mf < 2.2,
core = the rest (= the cv6win box).  Selection terms (P_s, Q_s, R_s, N_ns) are read from the bias logs,
mean over the 4 selection seeds.  Error: paired target bootstrap with the selection terms held fixed
(the galaxy-noise part of each difference; flow-retrain scatter, ~0.001 per flow, is NOT in it).
"""
import os
import re
import sys
import numpy as np

sys.path.insert(0, ".")
import bias as B

B.WINDOW_TAPER = (0.05, 150.0)
G, NBOOT = 0.02, 200
ALL = {**{f"s{i}": (f"pqr/desv_tnc15c_s{i}_sub.npz", f"logs/desv/bias_s{i}sub_{{w}}_sd{{s}}.log") for i in range(2, 7)},
       "n15c_s3": ("pqr/desv_n15c_s3_sub.npz", "logs/desv/bias_n15c_s3sub_{w}_sd{s}.log"),
       "s3dsx": ("pqr/desv_tnc15c_s3dsx_sub.npz", "logs/desv/bias_s3dsxsub_{w}_sd{s}.log"),
       **{k: (f"pqr/desv_tnc15c_{k}_sub.npz", f"logs/desv/bias_{k}sub_{{w}}_sd{{s}}.log") for k in ("k0", "k0g")}}
# desv s2-s6 pairs share templates and differ only by training seed: the null for one A - N pair.
a_, n_ = sys.argv[1:3] if len(sys.argv) > 2 else ("s3", "n15c_s3")
RUNS = {"A": ALL[a_], "N": ALL[n_]}
print(f"A = {a_}, N = {n_}")
WINDOWS = {"full90k": ((2.2, 3.5), (2200, 90000)), "sel20k": ((2.2, 3.5), (2200, 20000)),
           "cv6win": ((2.2, 3.2), (3000, 20000))}
WINDOWS = {w: v for w, v in WINDOWS.items()     # k-split runs have sel20k re-solves only
           if all(os.path.exists(p.format(w=w, s=0)) for _, p in RUNS.values())}
FLT = r"([-+.\de]+)"


def sel_terms(log):
    t = open(log).read()
    ps = float(re.search(rf"P_s = {FLT}", t).group(1))
    qs = np.array(re.search(rf"Q_s = \({FLT}, {FLT}\)", t).groups(), float)
    rs = np.array(re.search(rf"R_s = \[\[{FLT}, {FLT}\], \[{FLT}, {FLT}\]\]", t).groups(), float).reshape(2, 2)
    nn = [int(x) for x in re.search(r"excluded \(drives the correction\): plus (\d+), minus (\d+)", t).groups()]
    m1 = float(re.search(rf"windowed, corrected:\s+m1 = {FLT}", t).group(1))
    return ps, qs, rs, nn, m1


d = {k: np.load(f) for k, (f, _) in RUNS.items()}
for key in ("moments", "obs_plus", "obs_minus", "prefilter_w"):
    assert np.array_equal(d["A"][key], d["N"][key]), f"runs are not paired on {key}"
qr = {k: (v["plus_q"], v["plus_r"], v["minus_q"], v["minus_r"]) for k, v in d.items()}
m0 = d["A"]["moments"]
ratio, flux = m0[:, 1] / m0[:, 0], m0[:, 0]
strips = {"faint": flux < 3000, "bright": flux > 20000}
mid = ~strips["faint"] & ~strips["bright"]
strips |= {"large": mid & (ratio > 3.2), "small": mid & (ratio < 2.2)}
strips["core"] = mid & (ratio >= 2.2) & (ratio <= 3.2)
assert sum(s.astype(int) for s in strips.values()).max() == 1


def m1(q, sel, ns):
    return B.bias(*q, G, sel=sel, ns=ns)[0]


def mix(mask):
    """A's per-target Q, R where `mask`, N's elsewhere."""
    a, n = qr["A"], qr["N"]
    return tuple(np.where(mask.reshape((-1,) + (1,) * (x.ndim - 1)), x, y) for x, y in zip(a, n))


for w, (size, fl) in WINDOWS.items():
    sel = (B.window_mask(d["A"]["obs_plus"], size, fl), B.window_mask(d["A"]["obs_minus"], size, fl))
    ns, logged = {}, {}
    for k, (_, pat) in RUNS.items():
        t = [sel_terms(pat.format(w=w, s=s)) for s in range(4)]
        assert all(x[3] == t[0][3] for x in t)
        ps, qs, rs = (np.mean([x[i] for x in t], 0) for i in range(3))
        ns[k] = tuple((t[0][3][j], ps, qs, rs) for j in (0, 1))
        logged[k] = np.mean([x[4] for x in t])
    base = {k: m1(qr[k], sel, ns[k]) for k in RUNS}
    print(f"\n== {w}  size {size} flux {fl}")
    for k in RUNS:   # mean-of-seed terms vs mean of the logged per-seed solves (should agree to ~1e-5)
        print(f"  m1({k}) = {base[k]:+.5f}   (logged 4-seed mean {logged[k]:+.5f})")
    terms = {s: (lambda q, idx, s=s: m1(mix(strips[s]) if idx is None else tuple(x[idx] for x in mix(strips[s])),
                                        sel if idx is None else (sel[0][idx], sel[1][idx]), ns["N"]))
             for s in strips}
    terms["selection"] = lambda q, idx: m1(qr["N"] if idx is None else tuple(x[idx] for x in qr["N"]),
                                           sel if idx is None else (sel[0][idx], sel[1][idx]), ns["A"])
    terms["TOTAL"] = lambda q, idx: m1(qr["A"] if idx is None else tuple(x[idx] for x in qr["A"]),
                                       sel if idx is None else (sel[0][idx], sel[1][idx]), ns["A"])
    rng = np.random.default_rng(0)
    idxs = [rng.integers(0, len(m0), len(m0)) for _ in range(NBOOT)]

    def mN(idx):
        return m1(tuple(x[idx] for x in qr["N"]), (sel[0][idx], sel[1][idx]), ns["N"])
    bN = np.array([mN(i) for i in idxs])
    wsum = 0.5 * (sel[0] + sel[1])
    tot = 0.0
    for name, f in terms.items():
        c = f(None, None) - base["N"]
        e = np.std([f(None, i) for i in idxs] - bN)
        share = "" if name in ("selection", "TOTAL") else f"   window weight {wsum[strips[name]].sum() / wsum.sum():6.1%}"
        print(f"  {name:9s} {c:+.5f} +/- {e:.5f}{share}")
        tot += c if name != "TOTAL" else 0.0
    print(f"  sum of strips + selection {tot:+.5f}  (interaction {base['A'] - base['N'] - tot:+.5f})")
