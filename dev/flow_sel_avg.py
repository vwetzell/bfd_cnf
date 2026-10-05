"""Flow-averaged SELECTION terms on fixed targets.  (2026-10-04)  Offline, CPU only.

s3 vs s2 showed the flows' targets agree (uncorrected dm ~1e-4) while their selection terms
move corrected m by -0.0006; dev/flow_pool.py s4 re-checks the target side for a third flow.
So: the s2 flow's per-target Q, R over the 4 sn8r batches (as dev/sn8r_pool.py), solved with
each trained flow's tapered selection terms (all seed-0 draws; s2 = its 4-seed mean), then
the mean over flows.  Also P_s closure: each flow's P_s against the catalogs' own in-window
fraction sum(w)/NPOP (the paper's P(s) check; second-order g bias ~3e-5, negligible).
CAVEAT (2026-10-05): s4 FALSIFIED the premise -- its target side moved -0.00081 +/- 0.00025
against a selection-side +0.00093 (own-flow total only +0.00013).  Mixing one flow's targets
with another's selection terms breaks the first-order cancellation; use the P_s closure only.
Usage: JAX_PLATFORMS=cpu python dev/flow_sel_avg.py
"""
import os, re, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B

D, S, F, G = "../bfd_cnf_imsims/data/", (2.2, 3.2), (3000, 20000), 0.02
B.WINDOW_TAPER = (0.05, 150.0)
BATCHES = [("pqr/cv6_s2_sn8r_nopool.npz", "sn8r")] + \
          [(f"pqr/cv6_s2_sn8r_s{s}.npz", f"sn8r_s{s}") for s in (3, 4, 5)]
SEL = {"s2": ["logs/cv6/taper_sel_sn8r.log"] + [f"logs/cv6/taper_selseed/sn8r_seed{s}.log" for s in (1, 2, 3)],
       "s3": ["logs/cv6/bias_cv6_s3_sn8r_s3.log"],
       **{f"s{s}": [f"logs/cv6/taper_sel_sn8r_flow_s{s}.log"] for s in (4, 5, 6, 7)}}
SEL = {k: v for k, v in SEL.items() if all(os.path.exists(f) for f in v)}
LAB = ("m1", "c1", "c2")


def sel_terms(log):
    t = open(log).read()
    ps = float(re.search(r"P_s = ([-+.\de]+)", t).group(1))
    qs = np.array(re.search(r"Q_s = \(([-+.\de]+), ([-+.\de]+)\)", t).groups(), float)
    r = np.array(re.search(r"R_s = \[\[([-+.\de]+), ([-+.\de]+)\], \[([-+.\de]+), ([-+.\de]+)\]\]", t).groups(), float)
    return ps, qs, r.reshape(2, 2)


def load(pqr, tag):
    d = np.load(pqr)
    cat = {a: fitsio.read(D + f"targets_bdg2_{a}_176k_{tag}.fits", columns=["moments", "badcenter"])
           for a in ("g1p02", "g1m02", "g0")}
    keep = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
    x = {"npop": fitsio.read_header(D + f"targets_bdg2_g1p02_176k_{tag}.fits", ext=1)["NPOP"]}
    for arm, a in (("plus", "g1p02"), ("minus", "g1m02")):
        w = np.asarray(B.window_mask(d["obs_" + arm], S, F), float)
        x[arm] = (d[arm + "_q"] * w[:, None], d[arm + "_r"] * w[:, None, None], w)
        x["n_in_" + arm] = float(B.window_mask(np.asarray(cat[a]["moments"], np.float64)[keep], S, F).sum())
    return x


def solve(xs, sel, cs=None):
    ps, qs, rs = sel
    g = {}
    for arm in ("plus", "minus"):
        Q = Rm = nns = 0.0
        for b, x in enumerate(xs):
            q, r, w = x[arm]
            if cs is None:
                Q, Rm, nns = Q + q.sum(0)[None], Rm - r.sum(0)[None], nns + x["npop"] - x["n_in_" + arm]
            else:
                Q, Rm = Q + cs[b] @ q, Rm - np.einsum("bn,nij->bij", cs[b], r)
                nns = nns + x["npop"] - (x["n_in_" + arm] - w.sum() + cs[b] @ w)
        nns = np.atleast_1d(nns)[:, None]
        Qc = Q - nns * qs / (1 - ps)
        Rc = Rm + nns[..., None] * (np.outer(qs, qs) / (1 - ps) ** 2 + rs / (1 - ps))
        g[arm] = np.linalg.solve(Rc, Qc[..., None])[..., 0]
    gp, gm = g["plus"], g["minus"]
    return np.stack([(gp[:, 0] - gm[:, 0]) / (2 * G) - 1, *(0.5 * (gp + gm)).T], -1)


X = [load(p, t) for p, t in BATCHES]
# data P(s): in-window fraction, both arms, every batch (independent galaxies per batch)
frac = np.array([[x["n_in_plus"] / x["npop"], x["n_in_minus"] / x["npop"]] for x in X]).mean(1)
p_dat, p_err = frac.mean(), np.sqrt(frac.mean() * (1 - frac.mean()) / sum(x["npop"] for x in X))
print(f"data P(s) = {p_dat:.5f} +/- {p_err:.5f}  (per batch {np.round(frac, 5)})")
rng = np.random.default_rng(0)
cs = [rng.poisson(1.0, (300, len(x["plus"][2]))).astype(float) for x in X]
res = {}
for fl, logs in SEL.items():
    st = [sel_terms(f) for f in logs]
    sel = tuple(np.mean([s[i] for s in st], axis=0) for i in range(3))
    v, e = solve(X, sel)[0], solve(X, sel, cs).std(0)
    res[fl] = v
    print(f"  flow {fl}: P_s {sel[0]:.4f} ({(sel[0] - p_dat) / p_err:+.1f} sigma vs data)  R_s11 {sel[2][0, 0]:.5f}  "
          + "  ".join(f"{l} {a:+.5f}" for l, a in zip(LAB, v)) + f"   (targets +/- {e[0]:.5f} on m1)")
V = np.array(list(res.values()))
nf = len(V)
mean, sd = V.mean(0), V.std(0, ddof=1) if nf > 1 else np.zeros(3)
e_t = solve(X, tuple(np.mean([sel_terms(SEL[f][0])[i] for f in SEL], axis=0) for i in range(3)), cs).std(0)
print(f"flow-averaged selection terms ({nf} flows, s2 targets):")
for i, l in enumerate(LAB):
    tot = np.hypot(e_t[i], sd[i] / np.sqrt(nf))
    print(f"  {l} = {mean[i]:+.5f} +/- {e_t[i]:.5f} (targets) +/- {sd[i] / np.sqrt(nf):.5f} (flow mean; "
          f"per-flow sd {sd[i]:.5f}) = +/- {tot:.5f}   [{mean[i] / tot:+.1f} sigma]")
