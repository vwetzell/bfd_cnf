"""Paired PSF-ellipticity test on the current flow: (m1, c1, c2) of the 22k
real targets under psf_e1 = 0.05 minus the same targets under a round PSF.

Both renders share seed, galaxies and pixel noise (dev/psfe_g2v4.sh), so each
saved-PQR row is mapped back to its catalog row (exact bytes of the + arm's
observed moments) and the two runs are bootstrapped over the SAME rows -- the
population scatter that dominates either run alone cancels.  A row a run did
not integrate (outside its pre-filter pad, or a sane_targets drop) counts as
out of that run's window, which is what the estimator does with it.  N_ns per
arm = NPOP - n_in; selection terms from each run's bias.py log.

usage: python dev/psfe_g2v4_paired.py [tag [22k|176k]]      (default psfe1p05 22k)
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "dev")
import fitsio
import numpy as np

import bias as B

TAG = sys.argv[1] if len(sys.argv) > 1 else "psfe1p05"
SET = sys.argv[2] if len(sys.argv) > 2 else "22k"
SIZE, FLUX = (2.2, 3.2), (3000.0, 20000.0)
G = {"round": 0.02, TAG: 0.01 if TAG == "g01" else 0.02}   # g01: m(0.01) - m(0.02) = -3e-4 alpha
RUNS = {"22k": {"round": ("gauss2_v4n", "pqr/g2v4n_Ke_20k.npz", "logs/bias_g2v4n_Ke_20k.log"),
                TAG: (f"gauss2_v4n_{TAG}", f"pqr/g2v4n_Ke_22k_{TAG}.npz", f"logs/bias_g2v4n_Ke_22k_{TAG}.log")},
        "176k": {"round": ("gauss2_v4n_176k", "pqr/g2v4n_Ke_176k.npz", "logs/bias_g2v4n_Ke_176k.log"),
                 TAG: (f"gauss2_v4n_176k_{TAG}", f"pqr/g2v4n_Ke_176k_{TAG}.npz",
                       f"logs/bias_g2v4n_Ke_176k_{TAG}.log")}}[SET]


def sel_terms(log):
    import re
    t = open(log).read()
    num = r"([-+]?[\d.]+(?:e[-+]?\d+)?)"
    ps = float(re.search(r"P_s = " + num, t).group(1))
    qs = np.array(re.search(r"Q_s = \(" + num + r", " + num, t).groups(), float)
    rs = np.array(re.search(r"R_s = \[\[" + num + r", " + num + r"\], \[" + num + r", " + num, t).groups(),
                  float).reshape(2, 2)
    return ps, qs, rs


def load(pop, f, log):
    path = f"../bfd_cnf_imsims/data/{B.CATALOGS[pop]['plus']}.fits"
    cat = fitsio.read(path, columns=["moments"])["moments"].astype("<f8")
    npop = fitsio.read_header(path, 1)["NPOP"]
    where = {r.tobytes(): i for i, r in enumerate(cat)}
    d = np.load(f)
    row = np.array([where[r.astype("<f8").tobytes()] for r in d["obs_plus"]])
    n = len(cat)
    full = {}
    for k in ("plus_q", "plus_r", "minus_q", "minus_r", "obs_plus", "obs_minus"):
        a = np.zeros((n,) + d[k].shape[1:])
        a[row] = d[k]
        full[k] = a
    have = np.zeros(n, bool)
    have[row] = True
    sp = B.window_mask(np.where(have[:, None], full["obs_plus"], 1.0), SIZE, FLUX) & have
    sm = B.window_mask(np.where(have[:, None], full["obs_minus"], 1.0), SIZE, FLUX) & have
    return full, sp, sm, npop, sel_terms(log)


def est(r, i, g):
    full, sp, sm, npop, st = r
    q = tuple(full[k][i] for k in ("plus_q", "plus_r", "minus_q", "minus_r"))
    p, m = sp[i], sm[i]
    ns = ((npop - p.sum(),) + st, (npop - m.sum(),) + st)
    return np.array([B.bias(*q, g, sel=(p, m), ns=ns), B.bias(*q, g, sel=(p, m))]).ravel()


R = {k: load(*v) for k, v in RUNS.items()}
n = len(R["round"][1])
assert all(len(r[1]) == n for r in R.values())
for k, r in R.items():
    print(f"{k:10s} NPOP {r[3]}  n_in +/- {r[1].sum()}/{r[2].sum()}  P_s {r[4][0]:.4f}  "
          f"Q_s {r[4][1]}  R_s diag {np.diag(r[4][2])}")
print(f"window membership differs on {(R['round'][1] != R[TAG][1]).sum()} (+) / "
      f"{(R['round'][2] != R[TAG][2]).sum()} (-) of {n} rows")
idx = np.arange(n)
pt = {k: est(r, idx, G[k]) for k, r in R.items()}
rng = np.random.default_rng(0)
boot = []
for _ in range(400):
    i = rng.choice(idx, n)
    boot.append([est(r, i, G[k]) for k, r in R.items()])
boot = np.array(boot)                                  # (nboot, 2 runs, 6)
lab = ["m1 corr", "c1 corr", "c2 corr", "m1 uncorr", "c1 uncorr", "c2 uncorr"]
print(f"\n{'':10s}" + "".join(f"{x:>22s}" for x in lab))
for j, k in enumerate(R):
    print(f"{k:10s}" + "".join(f"{pt[k][c]:>+13.5f}+/-{boot[:, j, c].std():.5f}" for c in range(6)))
d = pt[TAG] - pt["round"]
sd = (boot[:, 1] - boot[:, 0]).std(0)
print(f"{'diff':10s}" + "".join(f"{d[c]:>+13.5f}+/-{sd[c]:.5f}" for c in range(6)))
print(f"{'sigma':10s}" + "".join(f"{d[c] / sd[c]:>22.1f}" for c in range(6)))
if TAG == "g01":
    print(f"alpha (m = m0 + alpha g^2): corr {-d[0] / 3e-4:+.2f} +/- {sd[0] / 3e-4:.2f}   "
          f"uncorr {-d[3] / 3e-4:+.2f} +/- {sd[3] / 3e-4:.2f}")
