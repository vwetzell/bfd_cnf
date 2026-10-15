"""Paired PSF-anisotropy test on the desv chain: (m1, c1, c2) of the first 300k desv targets rendered with
mean PSF e1 +0.05 minus the same rows under the baseline PSF (dev/desv_x6r_psfe.sh).  (2026-10-14)

Same seed, galaxies, conditions and pixel noise row for row, so each run's PQR rows are mapped back to their
catalog row (exact bytes of the + arm's moments) and both runs are bootstrapped over the SAME rows.  Window
weights are the tapered full90k window on each catalog's own moments; N_ns = NPOP_slice - sum w over the
slice (badcenter rows of either catalog dropped, as bias.py does).  Selection terms from each run's log.
Pure numpy (no bias.py import, so it can run beside a GPU job).
Usage: python dev/psfe_desv_paired.py BASE.npz BASE.log BASE_POP_TAG PSFE.npz PSFE.log PSFE_POP_TAG [DE1]
       tags are the catalog suffixes, e.g. 1m_desv and 1m_desv_psfe1p05; DE1 = PSF e1 shift (0.05)
"""
import re, sys
import numpy as np, fitsio
from scipy.stats import norm

D, N, G = "../bfd_cnf_imsims/data/", 300000, 0.02
SIZE, FLUX, TAU = (2.2, 3.5), (2200.0, 90000.0), (0.05, 150.0)
args = sys.argv[1:]
DE1 = float(args[6]) if len(args) > 6 else 0.05


def wmask(m):   # = bias.window_mask with --window-taper TAU
    r = m[:, 1] / m[:, 0]
    return ((norm.cdf((r - SIZE[0]) / TAU[0]) - norm.cdf((r - SIZE[1]) / TAU[0]))
            * (norm.cdf((m[:, 0] - FLUX[0]) / TAU[1]) - norm.cdf((m[:, 0] - FLUX[1]) / TAU[1])))


def terms(log):
    t, num = open(log).read(), r"([-+]?[\d.]+(?:e[-+]?\d+)?)"
    ps = float(re.search(r"P_s = " + num, t).group(1))
    qs = np.array(re.search(r"Q_s = \(" + num + r", " + num, t).groups(), float)
    rs = np.array(re.search(r"R_s = \[\[" + num + r", " + num + r"\], \[" + num + r", " + num, t).groups(),
                  float).reshape(2, 2)
    return ps, qs, rs


def load(pqr, log, tag):
    cat = {a: fitsio.read(D + f"targets_bdg2_{a}_{tag}.fits", columns=["moments", "badcenter"], rows=range(N))
           for a in ("g1p02", "g1m02", "g0")}
    hdr = fitsio.read_header(D + f"targets_bdg2_g1p02_{tag}.fits", ext=1)
    d = np.load(pqr)
    where = {r.tobytes(): i for i, r in enumerate(cat["g1p02"]["moments"].astype("<f8"))}
    row = np.array([where[r.astype("<f8").tobytes()] for r in d["obs_plus"]])
    full = {}
    for arm in ("plus", "minus"):
        for k in ("q", "r"):
            a = np.zeros((N,) + d[f"{arm}_{k}"].shape[1:])
            a[row] = d[f"{arm}_{k}"]
            full[f"{arm}_{k}"] = a
    have = np.zeros(N, bool); have[row] = True
    w = {arm: wmask(cat[a]["moments"].astype(np.float64)) for arm, a in (("plus", "g1p02"), ("minus", "g1m02"))}
    bad = np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
    return dict(full=full, have=have, w=w, bad=bad, npop=round(hdr["NPOP"] * N / hdr["NAXIS2"]), st=terms(log))


def est(r, i):
    ps, qs, rs = r["st"]
    g = {}
    for arm in ("plus", "minus"):
        w = r["w"][arm][i] * r["keep"][i]
        nns = r["npop"] - w.sum() * N / len(i)
        Q = (r["full"][f"{arm}_q"][i] * w[:, None]).sum(0) * N / len(i) - nns * qs / (1 - ps)
        R = (-(r["full"][f"{arm}_r"][i] * w[:, None, None]).sum(0) * N / len(i)
             + nns * (np.outer(qs, qs) / (1 - ps) ** 2 + rs / (1 - ps)))
        g[arm] = np.linalg.solve(R, Q)
    return np.array([(g["plus"][0] - g["minus"][0]) / (2 * G) - 1, *(0.5 * (g["plus"] + g["minus"]))])


R = {"base": load(*args[0:3]), "psfe": load(*args[3:6])}
bad = R["base"]["bad"] | R["psfe"]["bad"]
for k, r in R.items():
    r["keep"] = (~bad).astype(float)
    lost = sum((r["w"][a] * (~bad) * (~r["have"])).sum() for a in r["w"])
    print(f"{k}: NPOP_slice {r['npop']}  rows integrated {r['have'].sum()}  in-window weight NOT integrated {lost:.2f}  "
          f"P_s {r['st'][0]:.4f}")
print(f"badcenter-dropped rows (either catalog): {bad.sum()}")
idx = np.arange(N)
pt = {k: est(r, idx) for k, r in R.items()}
rng = np.random.default_rng(0)
boot = np.array([[est(r, i) for r in R.values()] for i in (rng.choice(idx, N) for _ in range(300))])
lab = ("m1", "c1", "c2")
for j, k in enumerate(R):
    print(f"{k:6s} " + "  ".join(f"{lab[c]} {pt[k][c]:+.5f} +/- {boot[:, j, c].std():.5f}" for c in range(3)))
d, sd = pt["psfe"] - pt["base"], (boot[:, 1] - boot[:, 0]).std(0)
print("diff   " + "  ".join(f"{lab[c]} {d[c]:+.6f} +/- {sd[c]:.6f} ({d[c] / sd[c]:+.1f} sigma)" for c in range(3)))
print(f"leakage alpha = dc1 / d(psf e1) = {d[1] / DE1:+.5f} +/- {sd[1] / DE1:.5f}   "
      f"(paper: >3000x suppression, i.e. |alpha| < ~3e-4)")
