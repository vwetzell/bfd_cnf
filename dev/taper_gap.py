"""Corrected windowed m1, hard vs TAPERED window, both catalogs, paired.  (2026-10-03)

Per-target Q, R from the full cv6 s2 runs (window-independent); per-target weight w =
`bias.window_mask` (0/1 hard, Gaussian-CDF edges under `bias.WINDOW_TAPER`) on each arm's
own moments; P_s, Q_s, R_s from bias.py logs (hard: dev/window_shift.py base; tapered:
dev/taper_sel.sh); N_ns = NPOP - sum w.  Poisson row bootstrap, paired across catalogs.
Usage: python dev/taper_gap.py      (after dev/taper_sel.sh)
"""
import os, re, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B

D, G, S, F, TAPER = "../bfd_cnf_imsims/data/", 0.02, (2.2, 3.2), (3000, 20000), (0.05, 150.0)
RUNS = {"A": ("pqr/cv6_s2_sn8r_nopool.npz", "sn8r", "sn8r"),
        "B": ("pqr/cv6_s2.npz", "varobs_sn8", "varobs")}
LOGS = {"hard": "logs/cv6/wshift_{}/base.log", "taper": "logs/cv6/taper_sel_{}.log"}
TAPER_TAG = {"sn8r": "sn8r", "varobs": "varobs_sn8"}


def sel_terms(log):
    t = open(log).read()
    ps = float(re.search(r"P_s = ([-+.\de]+)", t).group(1))
    qs = np.array(re.search(r"Q_s = \(([-+.\de]+), ([-+.\de]+)\)", t).groups(), float)
    r = np.array(re.search(r"R_s = \[\[([-+.\de]+), ([-+.\de]+)\], \[([-+.\de]+), ([-+.\de]+)\]\]", t).groups(), float)
    return ps, qs, r.reshape(2, 2)


def load(pqr, tag, short):
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
    x["w"], x["sel"] = {}, {}
    for kind in LOGS:
        B.WINDOW_TAPER = TAPER if kind == "taper" else None
        for arm, a in (("plus", "g1p02"), ("minus", "g1m02")):
            w = np.asarray(B.window_mask(np.asarray(cat[a]["moments"], np.float64), S, F), float)
            assert ((w > 1e-4) & ~integ).sum() < 50, (tag, kind, "past the pre-filter pad")
            x["w"][kind, arm] = w * (keep & integ)
        x["sel"][kind] = sel_terms(LOGS[kind].format(short if kind == "hard" else TAPER_TAG[short]))
    B.WINDOW_TAPER = None
    return x, integ


def m1(x, kind, c):
    """(uncorrected, corrected) m1; c = Poisson counts (n_rep, n) or None."""
    ps, qs, rs = x["sel"][kind]
    out = {}
    for arm in ("plus", "minus"):
        w = x["w"][kind, arm]
        q, r = x[arm + "_q"] * w[:, None], x[arm + "_r"] * w[:, None, None]
        if c is None:
            Q, Rm, sw = q.sum(0)[None], -r.sum(0)[None], np.array([w.sum()])
        else:
            Q, Rm, sw = c @ q, -np.einsum("bn,nij->bij", c, r), c @ w
        nns = (x["npop"] - sw)[:, None]
        Qc = Q - nns * qs / (1 - ps)
        Rc = Rm + nns[..., None] * (np.outer(qs, qs) / (1 - ps) ** 2 + rs / (1 - ps))
        out[arm] = (np.linalg.solve(Rm, Q[..., None])[:, 0, 0], np.linalg.solve(Rc, Qc[..., None])[:, 0, 0])
    return [(out["plus"][i] - out["minus"][i]) / (2 * G) - 1 for i in (0, 1)]


L = {k: load(*v) for k, v in RUNS.items()}
use = np.flatnonzero(L["A"][1] | L["B"][1])   # rows never integrated carry w = 0
R = {}
for k, (x, _) in L.items():
    for f in ("plus_q", "plus_r", "minus_q", "minus_r"):
        x[f] = x[f][use]
    x["w"] = {kk: v[use] for kk, v in x["w"].items()}
    R[k] = x
c = np.random.default_rng(0).poisson(1.0, (300, len(use))).astype(float)
print(f"taper (tau_size, tau_flux) = {TAPER};  selection terms: hard {LOGS['hard']}, taper {LOGS['taper']}")
print(f"{'window':>6s} {'P_s A/B':>13s} {'R_s11 A/B':>15s} {'corr A':>8s} {'corr B':>8s}  "
      f"{'m1 A (sn8r)':>18s} {'m1 B (varobs)':>18s} {'gap uncorr':>18s} {'gap corr':>18s}")
for kind in LOGS:
    (ua, ca), (ub, cb) = (m1(R[k], kind, None) for k in "AB")
    (bua, bca), (bub, bcb) = (m1(R[k], kind, c) for k in "AB")
    sa, sb = R["A"]["sel"][kind], R["B"]["sel"][kind]
    print(f"{kind:>6s} {sa[0]:.4f}/{sb[0]:.4f} {sa[2][0, 0]:+.4f}/{sb[2][0, 0]:+.4f} "
          f"{(ca - ua)[0]:+8.4f} {(cb - ub)[0]:+8.4f}  "
          f"{ca[0]:+.5f}+/-{bca.std():.5f} {cb[0]:+.5f}+/-{bcb.std():.5f} "
          f"{(ub - ua)[0]:+.5f}+/-{(bub - bua).std():.5f} {(cb - ca)[0]:+.5f}+/-{(bcb - bca).std():.5f}")
print("(errors: Poisson row bootstrap, paired; selection-term MC error not included)")

# Tapered selection terms at seeds 0..3 (dev/taper_sel.sh + dev/taper_selseed.sh): per-seed
# corrected m1, then seed-MEAN P_s, Q_s, R_s; selection MC error = seed sd / sqrt(n_seed).
SEEDLOG = "logs/cv6/taper_selseed/{}_seed{}.log"
seeds = [0] + [s for s in (1, 2, 3) if os.path.exists(SEEDLOG.format("sn8r", s))
                                     and os.path.exists(SEEDLOG.format("varobs_sn8", s))]
per = []
for s in seeds:
    for k, short in (("A", "sn8r"), ("B", "varobs")):
        R[k]["sel"]["taper"] = sel_terms(LOGS["taper"].format(TAPER_TAG[short]) if s == 0
                                         else SEEDLOG.format(TAPER_TAG[short], s))
    ca, cb = m1(R["A"], "taper", None)[1][0], m1(R["B"], "taper", None)[1][0]
    per.append((ca, cb, cb - ca, R["A"]["sel"]["taper"][0], R["B"]["sel"]["taper"][0]))
    print(f"  seed {s}: P_s {per[-1][3]:.4f}/{per[-1][4]:.4f}  m1 A {ca:+.5f}  m1 B {cb:+.5f}  gap {cb - ca:+.5f}")
per = np.array(per)
if len(seeds) > 1:
    sm = per[:, :3].std(0, ddof=1) / np.sqrt(len(seeds))
    for k, short in (("A", "sn8r"), ("B", "varobs")):
        st = [sel_terms(LOGS["taper"].format(TAPER_TAG[short]) if s == 0 else SEEDLOG.format(TAPER_TAG[short], s))
              for s in seeds]
        R[k]["sel"]["taper"] = tuple(np.mean([t[i] for t in st], axis=0) for i in range(3))
    (_, ca), (_, cb) = (m1(R[k], "taper", None) for k in "AB")
    (_, bca), (_, bcb) = (m1(R[k], "taper", c) for k in "AB")
    bt = np.array([bca.std(), bcb.std(), (bcb - bca).std()])
    tot = np.hypot(bt, sm)
    print(f"seed-mean selection terms ({len(seeds)} seeds), tapered, corrected:")
    for lab, v, b, e, t in zip(("m1 A (sn8r)", "m1 B (varobs)", "gap B-A"), (ca[0], cb[0], cb[0] - ca[0]), bt, sm, tot):
        print(f"  {lab:>14s} = {v:+.5f}  +/- {b:.5f} (targets) +/- {e:.5f} (sel MC) = +/- {t:.5f}")
