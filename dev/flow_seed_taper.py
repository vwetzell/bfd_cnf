"""Flow-training scatter under the TAPERED window: cv6 s3 flow minus s2 flow, same galaxies,
paired.  (2026-10-04)  Offline, CPU only.

Per-target Q, R of both flows exist (dev/cv6_seed.sh) and are window-independent.  The s2
flow has tapered selection terms (dev/taper_sel.sh, seed 0); the s3 flow has only the
hard-window ones from its full run, so its tapered terms are APPROXIMATED as
  s3_taper ~= s3_hard + (s2_taper - s2_hard)
-- the taper's change to the terms (P_s -2e-4, R_s11 -2%), carried over from s2.  Both
s2 terms come from the same seed-0 prior draws, so that difference is nearly noise-free;
the approximation errs only by how differently the two flows respond to the taper.
Gate: the HARD rows must reproduce dev/cv6_seed.sh's paired s3 - s2 (sn8r -0.00002,
varobs -0.00092).
Usage: JAX_PLATFORMS=cpu python dev/flow_seed_taper.py
"""
import os, re, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B

D, S, F, G, TAPER = "../bfd_cnf_imsims/data/", (2.2, 3.2), (3000, 20000), 0.02, (0.05, 150.0)
CASES = {"sn8r": ("sn8r", "pqr/cv6_{}_sn8r.npz", "logs/cv6/bias_cv6_{}_sn8r.log", "logs/cv6/taper_sel_sn8r.log"),
         "varobs": ("varobs_sn8", "pqr/cv6_{}.npz", "logs/cv6/bias_cv6_{}.log", "logs/cv6/taper_sel_varobs_sn8.log")}


def sel_terms(log):
    t = open(log).read()
    ps = float(re.search(r"P_s = ([-+.\de]+)", t).group(1))
    qs = np.array(re.search(r"Q_s = \(([-+.\de]+), ([-+.\de]+)\)", t).groups(), float)
    r = np.array(re.search(r"R_s = \[\[([-+.\de]+), ([-+.\de]+)\], \[([-+.\de]+), ([-+.\de]+)\]\]", t).groups(), float)
    return ps, qs, r.reshape(2, 2)


def m1(x, w, nin, npop, sel, c=None):
    ps, qs, rs = sel
    g = {}
    for arm in ("plus", "minus"):
        q, r, wa = x[arm + "_q"] * w[arm][:, None], x[arm + "_r"] * w[arm][:, None, None], w[arm]
        if c is None:
            Q, Rm, nns = q.sum(0)[None], -r.sum(0)[None], np.array([npop - nin[arm]])
        else:
            Q, Rm = c @ q, -np.einsum("bn,nij->bij", c, r)
            nns = npop - (nin[arm] - wa.sum() + c @ wa)
        nns = nns[:, None]
        Qc = Q - nns * qs / (1 - ps)
        Rc = Rm + nns[..., None] * (np.outer(qs, qs) / (1 - ps) ** 2 + rs / (1 - ps))
        g[arm] = np.linalg.solve(Rc, Qc[..., None])[:, 0, 0]
    return (g["plus"] - g["minus"]) / (2 * G) - 1


for name, (tag, pqr, log, tlog) in CASES.items():
    cat = {a: fitsio.read(D + f"targets_bdg2_{a}_176k_{tag}.fits", columns=["moments", "badcenter"])
           for a in ("g1p02", "g1m02", "g0")}
    n = len(cat["g0"])
    keep = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
    npop = fitsio.read_header(D + f"targets_bdg2_g1p02_176k_{tag}.fits", ext=1)["NPOP"]
    pos = {tuple(r): i for i, r in enumerate(np.asarray(cat["g1p02"]["moments"], np.float64))}
    X = {}
    for fl in ("s2", "s3"):
        d = np.load(pqr.format(fl))
        row = np.array([pos.get(tuple(r), -1) for r in d["obs_plus"]])
        assert (row >= 0).all()
        x = {k: np.zeros((n,) + d[k].shape[1:]) for k in ("plus_q", "plus_r", "minus_q", "minus_r")}
        for k in x:
            x[k][row] = d[k]
        x["integ"] = np.zeros(n, bool); x["integ"][row] = True
        X[fl] = x
    use = np.flatnonzero(X["s2"]["integ"] | X["s3"]["integ"])
    for x in X.values():
        for k in ("plus_q", "plus_r", "minus_q", "minus_r"):
            x[k] = x[k][use]
    hard = {fl: sel_terms(log.format(fl)) for fl in ("s2", "s3")}
    t2 = sel_terms(tlog)
    taper = {"s2": t2, "s3": tuple(h3 + (a - b) for h3, a, b in zip(hard["s3"], t2, hard["s2"]))}
    c = np.random.default_rng(0).poisson(1.0, (300, len(use))).astype(float)
    for kind, sel in (("hard", hard), ("taper", taper)):
        B.WINDOW_TAPER = TAPER if kind == "taper" else None
        mom = {arm: np.asarray(cat[a]["moments"], np.float64) for arm, a in (("plus", "g1p02"), ("minus", "g1m02"))}
        nin = {arm: float(B.window_mask(v[keep], S, F).sum()) for arm, v in mom.items()}
        w = {arm: np.asarray(B.window_mask(v, S, F), float)[use] * keep[use] for arm, v in mom.items()}
        v = {fl: m1(X[fl], w, nin, npop, sel[fl])[0] for fl in X}
        b = {fl: m1(X[fl], w, nin, npop, sel[fl], c) for fl in X}
        print(f"{name:>6s} {kind:>5s}: s2 {v['s2']:+.5f}  s3 {v['s3']:+.5f}  "
              f"paired s3 - s2 = {v['s3'] - v['s2']:+.5f} +/- {(b['s3'] - b['s2']).std():.5f}")
    B.WINDOW_TAPER = None
