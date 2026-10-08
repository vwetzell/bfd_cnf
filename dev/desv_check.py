"""Pre-flight for dev/desv.sh: do varied-PSF targets stay inside what the flow was built for?  CPU.

1. Sigma_X coverage: centroid train-sigmax --multi-scale --anisotropic saw Sigma_X = base * s^2,
   s in [1/MARGIN, MARGIN], with ellipticity |e| <= --aniso-max 0.10 (base = the copies' cov_odd).
   Each target's cov_odd -> trace ratio to base and |e| = sqrt((C00-C11)^2 + 4 C01^2) / (C00+C11).
2. Template deconvolution: C_M - <C_tmpl> must stay positive definite for every target
   (C_M's shape now varies with the PSF; <C_tmpl> is the fixed deep-field one).
Fails (exit 1) if more than 1% of targets fall outside the Sigma_X cloud or any kernel is not PD.
Usage: python dev/desv_check.py TARGETS.fits COPIES.fits TEMPLATES.fits [MARGIN]
"""
import sys
import numpy as np, fitsio, bfd

EMAX = 0.10
tg, cp, tm = sys.argv[1:4]
MARGIN = float(sys.argv[4]) if len(sys.argv) > 4 else 1.4   # = centroid --scale-margin = copies --sigma-range
t = fitsio.read(tg, columns=["cov", "cov_odd"])
base = fitsio.read(cp, ext="GALAXIES", columns=["cov_odd"], rows=[0])["cov_odd"][0].astype(float)
sx = np.asarray(t["cov_odd"], float)
ratio = (sx[:, 0] + sx[:, 2]) / (base[0] + base[2])
e = np.hypot(sx[:, 0] - sx[:, 2], 2 * sx[:, 1]) / (sx[:, 0] + sx[:, 2])
out = (ratio < MARGIN ** -2) | (ratio > MARGIN ** 2) | (e > EMAX)
P = (0.1, 1, 16, 50, 84, 99, 99.9)
print(f"Sigma_X trace / base   p{P}: {np.percentile(ratio, P).round(3)}   trained [{MARGIN ** -2:.3f}, {MARGIN ** 2:.3f}]")
print(f"Sigma_X |e|            p{P}: {np.percentile(e, P).round(4)}   trained <= {EMAX}")
print(f"outside the trained Sigma_X cloud: {out.mean():.4%}")
cm = bfd.MomentCovariance.bulkUnpack(np.asarray(t["cov"], float))
ct = bfd.MomentCovariance.bulkUnpack(np.asarray(fitsio.read(tm, columns=["cov"])["cov"], float)).mean(0)
ev = np.linalg.eigvalsh(cm - ct).min(1) / np.linalg.eigvalsh(cm).min(1)
print(f"C_M - <C_tmpl>: min-eigenvalue ratio p0/p1/p50 {np.percentile(ev, (0, 1, 50)).round(3)}; "
      f"C_tmpl/C_M (Mf) p1/50/99 {np.percentile(ct[0, 0] / cm[:, 0, 0], (1, 50, 99)).round(4)}")
# sim --nproc once dropped psf_e silently (memory sim-nproc-dropped-psf-e): the PSF spread must reach cov_odd
spread = np.median(e) > 1e-3 and ratio.std() / ratio.mean() > 0.05
print(f"PSF variation reached the renders (median |e| > 1e-3, trace sd > 5%): {spread}")
ok = out.mean() <= 0.01 and ev.min() > 0 and spread
print("desv pre-flight", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
