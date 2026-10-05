"""Two-flow pooled tapered (m1, c1, c2) over the sn8r batches: cv6 s2 and s3 flows on the
SAME galaxies, paired.  (2026-10-04)  Offline, CPU only.

Batches: render seeds 2-5.  Flow s2: pqr/cv6_s2_sn8r{_nopool,_s3..5}.npz, selection terms
= mean of its 4 tapered seeds (as dev/sn8r_pool.py).  Flow s3: pqr/cv6_s3_sn8r{,_s3..5}.npz
(dev/cv6_seed.sh, dev/cond_batches.sh flow), tapered selection terms from its own run
(seed 0).  Rows mapped to catalog index so one Poisson count matrix resamples both flows:
the s3 - s2 difference is paired (flow-training scatter), the mean is the quoted number.
Usage: JAX_PLATFORMS=cpu python dev/flow_pool.py [FLOW]   (FLOW = s3 default, s4, ...; batches
       without that flow's pqr are skipped; "target side" re-solves both flows with s2's terms)
"""
import os, re, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B

D, S, F, G = "../bfd_cnf_imsims/data/", (2.2, 3.2), (3000, 20000), 0.02
B.WINDOW_TAPER = (0.05, 150.0)
FL = sys.argv[1] if len(sys.argv) > 1 else "s3"
BATCHES = {"sn8r": {"s2": "pqr/cv6_s2_sn8r_nopool.npz", "s3": f"pqr/cv6_{FL}_sn8r.npz"},
           **{f"sn8r_s{s}": {"s2": f"pqr/cv6_s2_sn8r_s{s}.npz", "s3": f"pqr/cv6_{FL}_sn8r_s{s}.npz"}
              for s in (3, 4, 5)}}
BATCHES = {t: p for t, p in BATCHES.items() if all(os.path.exists(v) for v in p.values())}
SEL = {"s2": ["logs/cv6/taper_sel_sn8r.log"] + [f"logs/cv6/taper_selseed/sn8r_seed{s}.log" for s in (1, 2, 3)],
       "s3": ["logs/cv6/bias_cv6_s3_sn8r_s3.log" if FL == "s3" else f"logs/cv6/taper_sel_sn8r_flow_{FL}.log"]}
NB = 300
LAB = ("m1", "c1", "c2")


def sel_terms(log):
    t = open(log).read()
    ps = float(re.search(r"P_s = ([-+.\de]+)", t).group(1))
    qs = np.array(re.search(r"Q_s = \(([-+.\de]+), ([-+.\de]+)\)", t).groups(), float)
    r = np.array(re.search(r"R_s = \[\[([-+.\de]+), ([-+.\de]+)\], \[([-+.\de]+), ([-+.\de]+)\]\]", t).groups(), float)
    return ps, qs, r.reshape(2, 2)


def load(tag, pqrs):
    cat = {a: fitsio.read(D + f"targets_bdg2_{a}_176k_{tag}.fits", columns=["moments", "badcenter"])
           for a in ("g1p02", "g1m02", "g0")}
    n = len(cat["g0"])
    keep = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
    npop = fitsio.read_header(D + f"targets_bdg2_g1p02_176k_{tag}.fits", ext=1)["NPOP"]
    pos = {tuple(r): i for i, r in enumerate(np.asarray(cat["g1p02"]["moments"], np.float64))}
    X, integ = {}, np.zeros(n, bool)
    for fl, p in pqrs.items():
        d = np.load(p)
        row = np.array([pos.get(tuple(r), -1) for r in d["obs_plus"]])
        assert (row >= 0).all(), p
        x = {k: np.zeros((n,) + d[k].shape[1:]) for k in ("plus_q", "plus_r", "minus_q", "minus_r")}
        for k in x:
            x[k][row] = d[k]
        integ[row] = True
        X[fl] = x
    use = np.flatnonzero(integ)
    out = {"npop": npop, "use": len(use)}
    for arm, a in (("plus", "g1p02"), ("minus", "g1m02")):
        mom = np.asarray(cat[a]["moments"], np.float64)
        w = np.asarray(B.window_mask(mom, S, F), float) * keep
        out["n_in_" + arm] = w.sum()          # whole catalog, as bias.py main
        out["w_" + arm] = w[use]
    for fl, x in X.items():
        out[fl] = {k: v[use] for k, v in x.items()}
    return out


def solve(bs, fl, sel, cs=None):
    """(n_rep, 3) corrected (m1, c1, c2) pooled over batches bs for flow fl."""
    ps, qs, rs = sel
    g = {}
    for arm in ("plus", "minus"):
        Q = Rm = nns = 0.0
        for b, x in enumerate(bs):
            w = x["w_" + arm]
            q, r = x[fl][arm + "_q"] * w[:, None], x[fl][arm + "_r"] * w[:, None, None]
            if cs is None:
                Q, Rm = Q + q.sum(0)[None], Rm - r.sum(0)[None]
                nns = nns + x["npop"] - x["n_in_" + arm]
            else:
                Q, Rm = Q + cs[b] @ q, Rm - np.einsum("bn,nij->bij", cs[b], r)
                nns = nns + x["npop"] - (x["n_in_" + arm] - w.sum() + cs[b] @ w)
        nns = np.atleast_1d(nns)[:, None]
        Qc = Q - nns * qs / (1 - ps)
        Rc = Rm + nns[..., None] * (np.outer(qs, qs) / (1 - ps) ** 2 + rs / (1 - ps))
        g[arm] = np.linalg.solve(Rc, Qc[..., None])[..., 0]
    gp, gm = g["plus"], g["minus"]
    return np.stack([(gp[:, 0] - gm[:, 0]) / (2 * G) - 1, *(0.5 * (gp + gm)).T], -1)


X = {t: load(t, p) for t, p in BATCHES.items()}
sel = {fl: tuple(np.mean([sel_terms(f)[i] for f in fs], axis=0) for i in range(3)) for fl, fs in SEL.items()}
rng = np.random.default_rng(0)
print(f"flow '{FL}' (labelled s3 below) vs s2; taper {B.WINDOW_TAPER}; selection terms: s2 = mean of "
      f"{len(SEL['s2'])} seeds, {FL} = {SEL['s3']}")
print("per batch: s2, s3, paired s3 - s2 (m1);  target side = both flows' targets with s2's terms")
diffs = []
for t, x in X.items():
    cs = [rng.poisson(1.0, (NB, x["use"])).astype(float)]
    v2, v3 = solve([x], "s2", sel["s2"])[0], solve([x], "s3", sel["s3"])[0]
    b2, b3 = solve([x], "s2", sel["s2"], cs), solve([x], "s3", sel["s3"], cs)
    e = (b3 - b2).std(0)
    diffs.append((v3[0] - v2[0], e[0]))
    tv, tb = solve([x], "s3", sel["s2"])[0], solve([x], "s3", sel["s2"], cs)
    print(f"  {t:>9s}: {v2[0]:+.5f}  {v3[0]:+.5f}  {v3[0] - v2[0]:+.5f} +/- {e[0]:.5f}   "
          f"target side {tv[0] - v2[0]:+.5f} +/- {(tb - b2).std(0)[0]:.5f}")
dd = np.array(diffs)
chi2 = 0.0 if len(dd) < 2 else np.sum(((dd[:, 0] - np.average(dd[:, 0], weights=dd[:, 1] ** -2)) / dd[:, 1]) ** 2)
bs = list(X.values())
cs = [rng.poisson(1.0, (NB, x["use"])).astype(float) for x in bs]
v2, v3 = solve(bs, "s2", sel["s2"])[0], solve(bs, "s3", sel["s3"])[0]
b2, b3 = solve(bs, "s2", sel["s2"], cs), solve(bs, "s3", sel["s3"], cs)
d, ed = v3 - v2, (b3 - b2).std(0)
mean, em = 0.5 * (v2 + v3), (0.5 * (b2 + b3)).std(0)
print(f"pooled ({len(bs)} batches), paired s3 - s2 per-batch chi2 = {chi2:.1f} / {len(bs) - 1}:")
for i, l in enumerate(LAB):
    # flow-training scatter per flow ~ |s3 - s2| / sqrt(2); the 2-flow mean carries half its variance
    fs = abs(d[i]) / np.sqrt(2)
    tot = np.hypot(em[i], fs / np.sqrt(2))
    print(f"  {l}: s2 {v2[i]:+.5f}  s3 {v3[i]:+.5f}  s3 - s2 {d[i]:+.5f} +/- {ed[i]:.5f}   "
          f"2-flow mean {mean[i]:+.5f} +/- {em[i]:.5f} (targets) +/- {fs / np.sqrt(2):.5f} (flow) = +/- {tot:.5f}")
