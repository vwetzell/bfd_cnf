"""Does the varobs-minus-sn8r m1 excess grow with distance from the reference condition? (2026-09-29)

Same galaxies in both catalogs (rows match), so the conditions are a random
label on the galaxies.  Per quintile of each varobs row's Sigma_X scale
(sqrt det, relative to the sn8r reference) and of its C_ff, paired over rows:
  - Fisher ratio sum q1^2 / sum(-r11) over in-window targets (health check);
  - uncorrected windowed m1.  Caveat: the selection effect itself varies with
    the condition, so only the TREND across bins is informative, not the level.
Offline; reuses pqr/cv2_s0_J.npz (varobs) and pqr/cv2_s0_sn8r_J.npz (sn8r).
"""
import os, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bias import bias, window_mask

D, S, F = "../bfd_cnf_imsims/data/", (2.2, 3.2), (3000, 20000)
RUNS = {"A": ("pqr/cv2_s0_sn8r_J.npz", "sn8r"), "B": ("pqr/cv2_s0_J.npz", "varobs_sn8")}
# Usage: condition_split.py [A.npz A_TAG B.npz B_TAG]   (diff = B - A; tags sn8r | varobs_sn8)
if len(sys.argv) == 5:
    RUNS = {"A": tuple(sys.argv[1:3]), "B": tuple(sys.argv[3:5])}


def load(pqr, tag):
    d = np.load(pqr)
    cat = {a: fitsio.read(D + f"targets_bdg2_{a}_176k_{tag}.fits", columns=["moments", "badcenter"])
           for a in ("g1p02", "g1m02", "g0")}
    n = len(cat["g0"])
    keep = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
    pos = {tuple(r): i for i, r in enumerate(np.asarray(cat["g1p02"]["moments"], np.float64))}
    row = np.array([pos.get(tuple(r), -1) for r in d["obs_plus"]])
    assert (row >= 0).all()
    out = {k: np.zeros((n,) + d[k].shape[1:]) for k in ("plus_q", "plus_r", "minus_q", "minus_r")}
    for k in out:
        out[k][row] = d[k]
    integ = np.zeros(n, bool); integ[row] = True
    out["sp"] = keep & integ & window_mask(np.asarray(cat["g1p02"]["moments"], np.float64), S, F)
    out["sm"] = keep & integ & window_mask(np.asarray(cat["g1m02"]["moments"], np.float64), S, F)
    return out


R = {k: load(*v) for k, v in RUNS.items()}
ref = fitsio.read(D + "targets_bdg2_g0_176k_sn8r.fits", columns=["cov_odd", "cov"], rows=[0])
c = fitsio.read(D + "targets_bdg2_g0_176k_varobs_sn8.fits", columns=["cov_odd", "cov"])
det = lambda s: s[..., 0] * s[..., 2] - s[..., 1] ** 2
cond = {"Sigma_X scale": np.sqrt(np.sqrt(det(c["cov_odd"]) / det(ref["cov_odd"][0]))),
        "C_ff ratio": c["cov"][:, 0] / ref["cov"][0][0],
        "Sigma_X |e|": np.hypot(c["cov_odd"][:, 0] - c["cov_odd"][:, 2], 2 * c["cov_odd"][:, 1])
                       / (c["cov_odd"][:, 0] + c["cov_odd"][:, 2])}
# The varobs render's TRUE per-row conditions (imsims sample_obs, dev/bdg2n.sh DEPTH=sn8): the
# mean PSF is sigma 0.42, e (0.03, -0.02), not sn8r's round 0.40, so no Sigma_X/C_ff bin sits
# at the sn8r reference; these bin on what was actually varied.
sys.path.insert(0, "../bfd_cnf_imsims")
from imsims.sim import sample_obs
o = sample_obs(len(c), 2, (0.03, -0.02, 0.42, 3.674), (0.02, 0.02, 0.012, 0.1837))
cond.update({"psf sigma": o["psf_sigma"], "psf |e|": np.hypot(o["psf_e1"], o["psf_e2"]),
             "noise sigma": o["noise_sigma"]})
print({k: np.percentile(v, [0, 5, 50, 95, 100]).round(3).tolist() for k, v in cond.items()},
      "corr", np.corrcoef(list(cond.values())).round(3).tolist())


def stats(x, i):
    sp, sm = x["sp"][i], x["sm"][i]
    fr = ((x["plus_q"][i][sp, 0] ** 2).sum() + (x["minus_q"][i][sm, 0] ** 2).sum()) / \
         (-(x["plus_r"][i][sp, 0, 0]).sum() - (x["minus_r"][i][sm, 0, 0]).sum())
    m = bias(x["plus_q"][i], x["plus_r"][i], x["minus_q"][i], x["minus_r"][i], 0.02, sel=(sp, sm))[0]
    return np.array([fr, m])


rng = np.random.default_rng(0)
for name, v in cond.items():
    edges = np.percentile(v, np.linspace(0, 100, 6))
    print(f"\n{name} quintile   N_in   Fisher A     B        diff             "
          f"m1 unc A       B        diff")
    dm = []
    for b in range(5):
        rows = np.flatnonzero((v >= edges[b]) & (v <= edges[b + 1]))
        a, o = stats(R["A"], rows), stats(R["B"], rows)
        bs = []
        for _ in range(100):
            i = rows[rng.integers(0, len(rows), len(rows))]
            bs.append(stats(R["B"], i) - stats(R["A"], i))
        e = np.std(bs, 0)
        print(f"  {edges[b]:.3f}-{edges[b + 1]:.3f} {R['B']['sp'][rows].sum():>6d}   "
              f"{a[0]:.4f}  {o[0]:.4f}  {o[0] - a[0]:+.4f}+/-{e[0]:.4f}   "
              f"{a[1]:+.4f}  {o[1]:+.4f}  {o[1] - a[1]:+.4f}+/-{e[1]:.4f}")
        dm.append((o[1] - a[1], e[1]))
    d, e = np.array(dm).T   # bins share no rows -> independent
    w = 1 / e**2; mean = (w * d).sum() / w.sum(); x = np.arange(5.0); x -= (w * x).sum() / w.sum()
    slope = (w * x * (d - mean)).sum() / (w * x * x).sum()
    print(f"  m1 diff: weighted mean {mean:+.4f}+/-{w.sum()**-.5:.4f}   slope/quintile "
          f"{slope:+.4f}+/-{(w * x * x).sum()**-.5:.4f}   chi2 flat {(w * (d - mean)**2).sum():.1f}/4")
