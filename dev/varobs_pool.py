"""Pooled (m1, c1, c2) over the six 176k varied-conditions batches (seed 2 from
dev/psfe_g2v4.sh TAG=varobs, seeds 3-7 from dev/more_176k_batches.sh).

Each batch has its own condition means, hence its own selection terms, so the
pool is ONE shear solve per arm with every batch's non-selection term added:
Q = sum_b [sum q_b - N_ns_b Q_s_b / (1 - P_s_b)], R likewise (bias.ghat's
formula summed over batches).  Galaxy bootstrap is stratified by batch;
selection-term MC error is not in the bar.  Also fits the per-batch c against
the batch's mean PSF e (c_i = a_i + b_i e_psf_i).

usage: python dev/varobs_pool.py [--sel-scan]
  --sel-scan: selection-term MC error from dev/selseed_varobs.sh's reruns.
  Pooled point estimates with every batch's terms at seed r (r = 0..5, the
  production correlation: one seed shared by all batches), then with batch b
  at seed (r + b) % 6.  sd over r of the shared scan is the MC error.  The cyclic
  scan is a Latin square (every row uses each seed once), so it reproduces the
  6-seed mean; its near-zero sd says a seed's MC error is common to all batches,
  not that independent seeds would remove it.
"""
import re
import sys
sys.path.insert(0, ".")
import fitsio
import numpy as np

import bias as B

SIZE, FLUX, G = (2.2, 3.2), (3000.0, 20000.0), 0.02
BATCHES = [("gauss2_v4n_176k_varobs", "g2v4n_Ke_176k_varobs")] + \
          [(f"gauss2_v4n_176k_s{s}_varobs", f"g2v4n_Ke_176k_s{s}_varobs") for s in range(3, 8)]


def sel_terms(log):   # same parse as dev/chunk_scatter.py (which runs on import)
    t = open(log).read()
    num = r"([-+]?[\d.]+(?:e[-+]?\d+)?)"
    ps = float(re.search(r"P_s = " + num, t).group(1))
    qs = np.array(re.search(r"Q_s = \(" + num + r", " + num, t).groups(), float)
    rs = np.array(re.search(r"R_s = \[\[" + num + r", " + num + r"\], \[" + num + r", " + num, t).groups(),
                  float).reshape(2, 2)
    return ps, qs, rs


def load(pop, tag):
    path = f"../bfd_cnf_imsims/data/{B.CATALOGS[pop]['plus']}.fits"
    h = fitsio.read_header(path, 1)
    e = fitsio.read(path, ext="OBS")[["psf_e1", "psf_e2"]]
    d = np.load(f"pqr/{tag}.npz")
    k = {x: d[x] for x in ("plus_q", "plus_r", "minus_q", "minus_r")}
    k["sp"] = B.window_mask(d["obs_plus"], SIZE, FLUX)
    k["sm"] = B.window_mask(d["obs_minus"], SIZE, FLUX)
    return k, h["NPOP"], sel_terms(f"logs/bias_{tag}.log"), (e["psf_e1"].mean(), e["psf_e2"].mean())


def arm(batches, idx, a, corr):
    Q, R = 0.0, 0.0
    for (k, npop, (ps, qs, rs), _), i in zip(batches, idx):
        s = k["s" + a[0]][i]
        q, r = k[a + "_q"][i][s], k[a + "_r"][i][s]
        Q, R = Q + q.sum(0), R - r.sum(0)
        if corr:
            n_ns = npop - s.sum()
            Q = Q - n_ns * qs / (1 - ps)
            R = R + n_ns * (np.outer(qs, qs) / (1 - ps) ** 2 + rs / (1 - ps))
    return np.linalg.solve(R, Q)


def est(batches, idx):
    out = []
    for corr in (True, False):
        gp, gm = arm(batches, idx, "plus", corr), arm(batches, idx, "minus", corr)
        out += [(gp[0] - gm[0]) / (2 * G) - 1, *(0.5 * (gp + gm))]
    return np.array(out)


def sel_log(tag, r):
    return f"logs/bias_{tag}.log" if r == 0 else f"logs/selseed/bias_{tag}_seed{r}.log"


Bs = [load(*b) for b in BATCHES]
if "--sel-scan" in sys.argv:
    full = [np.arange(len(b[0]["plus_q"])) for b in Bs]
    for name, pick in (("shared seed", lambda r, j: r), ("independent", lambda r, j: (r + j) % 6)):
        P = np.array([est([(k, npop, sel_terms(sel_log(tag, pick(r, j))), e)
                           for j, ((k, npop, _, e), (_, tag)) in enumerate(zip(Bs, BATCHES))], full)
                      for r in range(6)])
        print(f"{name:12s} m1/c1/c2 corr per r:")
        for r, p in enumerate(P):
            print(f"  r={r}  {p[0]:+.5f} {p[1]:+.2e} {p[2]:+.2e}")
        print(f"  sd over r   {P[:, 0].std(ddof=1):.5f} {P[:, 1].std(ddof=1):.1e} {P[:, 2].std(ddof=1):.1e}")
    sys.exit()
full = [np.arange(len(b[0]["plus_q"])) for b in Bs]
rng = np.random.default_rng(0)
lab = ["m1 corr", "c1 corr", "c2 corr", "m1 uncorr", "c1 uncorr", "c2 uncorr"]
per = []
print(f"{'batch':26s}{'<e1>':>8s}{'<e2>':>8s}" + "".join(f"{x:>20s}" for x in lab[:3]))
for (pop, _), b, f in zip(BATCHES, Bs, full):
    pt = est([b], [f])
    bs = np.array([est([b], [rng.choice(f, len(f))]) for _ in range(200)]).std(0)
    per.append((b[3], pt, bs))
    print(f"{pop:26s}{b[3][0]:>+8.3f}{b[3][1]:>+8.3f}" +
          "".join(f"{pt[c]:>+11.5f}+/-{bs[c]:.5f}" for c in range(3)))
pt = est(Bs, full)
boot = np.array([est(Bs, [rng.choice(f, len(f)) for f in full]) for _ in range(400)]).std(0)
n = sum(len(f) for f in full)
print(f"\nPOOLED ({len(Bs)} batches, {n} integrated targets, NPOP {sum(b[1] for b in Bs)})")
for c in range(6):
    print(f"  {lab[c]:10s} {pt[c]:+.5f} +/- {boot[c]:.5f}")
E = np.array([p[0] for p in per])
for j, cname in ((1, "c1"), (2, "c2")):
    y, s = np.array([p[1][j] for p in per]), np.array([p[2][j] for p in per])
    X = np.c_[np.ones(len(E)), E[:, j - 1]] / s[:, None]
    cov = np.linalg.inv(X.T @ X)
    a, b = cov @ X.T @ (y / s)
    chi2 = (((y - a - b * E[:, j - 1]) / s) ** 2).sum()
    print(f"  {cname} = a + b e{j}_psf over batches: a {a:+.2e} +/- {cov[0, 0] ** .5:.1e}   "
          f"b {b:+.4f} +/- {cov[1, 1] ** .5:.4f}   chi2 {chi2:.1f}/{len(E) - 2}")
