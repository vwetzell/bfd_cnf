"""Pooled tapered corrected m1 over the sn8r target batches, cv6 s2.  (2026-10-03)  CPU only.

Batches: seed 2 (pqr/cv6_s2_sn8r_nopool.npz) + seeds 3-5 (dev/sn8r_batches.sh).  Same
population, round PSF, fixed C_M -> one set of selection terms for all: the seed-mean of
the 4 tapered selection seeds (dev/taper_sel.sh + dev/taper_selseed.sh).  One shear solve
per arm over every batch's targets, N_ns = sum_b (NPOP_b - sum w_b), as bias.ghat.
Errors: Poisson row bootstrap (independent per batch); selection MC from the seed scatter.
Usage: JAX_PLATFORMS=cpu python dev/sn8r_pool.py
"""
import os, re, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B

D, S, F, G = "../bfd_cnf_imsims/data/", (2.2, 3.2), (3000, 20000), 0.02
B.WINDOW_TAPER = (0.05, 150.0)
BATCHES = [("pqr/cv6_s2_sn8r_nopool.npz", "sn8r")] + \
          [(f"pqr/cv6_s2_sn8r_s{s}.npz", f"sn8r_s{s}") for s in (3, 4, 5)]
SEL = ["logs/cv6/taper_sel_sn8r.log"] + [f"logs/cv6/taper_selseed/sn8r_seed{s}.log" for s in (1, 2, 3)]


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
    npop = fitsio.read_header(D + f"targets_bdg2_g1p02_176k_{tag}.fits", ext=1)["NPOP"]
    x = {"npop": npop}
    for arm, a in (("plus", "g1p02"), ("minus", "g1m02")):
        w = np.asarray(B.window_mask(d["obs_" + arm], S, F), float)
        x[arm] = (d[arm + "_q"] * w[:, None], d[arm + "_r"] * w[:, None, None], w)
        # in-window count over the WHOLE catalog (as bias.py main), not the integrated set
        x["n_in_" + arm] = float(B.window_mask(np.asarray(cat[a]["moments"], np.float64)[keep], S, F).sum())
    return x


def m1(xs, sel, cs=None):
    """Corrected (m1, c1, c2), shape (n_rep, 3), pooled over batches xs; cs = per-batch
    Poisson counts (n_rep, n_b) or None (-> n_rep = 1)."""
    ps, qs, rs = sel
    g = {}
    for arm in ("plus", "minus"):
        Q, Rm, nns = 0.0, 0.0, 0.0
        for b, x in enumerate(xs):
            q, r, w = x[arm]
            if cs is None:
                Q, Rm = Q + q.sum(0)[None], Rm - r.sum(0)[None]
                nns = nns + x["npop"] - x["n_in_" + arm]
            else:   # resample the integrated rows; out-of-pad rows (w = 0) count as themselves
                c = cs[b]
                Q, Rm = Q + c @ q, Rm - np.einsum("bn,nij->bij", c, r)
                nns = nns + x["npop"] - (x["n_in_" + arm] - w.sum() + c @ w)
        nns = np.atleast_1d(nns)[:, None]
        Qc = Q - nns * qs / (1 - ps)
        Rc = Rm + nns[..., None] * (np.outer(qs, qs) / (1 - ps) ** 2 + rs / (1 - ps))
        g[arm] = np.linalg.solve(Rc, Qc[..., None])[..., 0]
    gp, gm = g["plus"], g["minus"]
    return np.stack([(gp[:, 0] - gm[:, 0]) / (2 * G) - 1, *(0.5 * (gp + gm)).T], -1)


have = [(p, t) for p, t in BATCHES if os.path.exists(p)]
X = [load(p, t) for p, t in have]
sels = [sel_terms(f) for f in SEL]
sel = tuple(np.mean([s[i] for s in sels], axis=0) for i in range(3))
rng = np.random.default_rng(0)
NB = 300
print(f"taper {B.WINDOW_TAPER}; selection terms = mean of {len(sels)} seeds")
LAB = ("m1", "c1", "c2")
per = []
for (p, t), x in zip(have, X):
    cs = [rng.poisson(1.0, (NB, len(x["plus"][2]))).astype(float)]
    v, e = m1([x], sel)[0], m1([x], sel, cs).std(0)
    per.append((v, e))
    print(f"  {t:>9s}: N_int {len(x['plus'][2])}  sum w {x['plus'][2].sum():.0f}  "
          + "  ".join(f"{l} = {a:+.5f} +/- {b:.5f}" for l, a, b in zip(LAB, v, e)))
pv, pe = np.array([p[0] for p in per]), np.array([p[1] for p in per])
cs = [rng.poisson(1.0, (NB, len(x["plus"][2]))).astype(float) for x in X]
v, e = m1(X, sel)[0], m1(X, sel, cs).std(0)
sm = np.std([m1(X, s)[0] for s in sels], axis=0, ddof=1) / np.sqrt(len(sels))
wm = np.sum(pv / pe ** 2, 0) / np.sum(1 / pe ** 2, 0)
chi2 = np.sum(((pv - wm) / pe) ** 2, 0)
print(f"pooled ({len(X)} batches), errors (targets) (sel MC) = total:")
for i, l in enumerate(LAB):
    t = np.hypot(e[i], sm[i])
    print(f"  {l} = {v[i]:+.5f} +/- {e[i]:.5f} +/- {sm[i]:.5f} = +/- {t:.5f}   [{v[i] / t:+.1f} sigma]"
          f"   batch chi2 = {chi2[i]:.1f} / {len(X) - 1}")
