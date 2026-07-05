"""Does the flux-size-plane m-bias correlate with (a) training-template SNIS/ESS
(convergence) or (b) prior curvature (selection)?

Maps three things on the SAME (log10 Mf, Mr/Mf) grid:
  bias m        : from the independent-ensemble PQR npz (per-cell summed g+/-).
  train density : nda*detj-weighted template density + per-cell SNIS ESS, from
                  templates_train.fits (the weight the flow is trained to match;
                  low ESS / low density => few effective templates => under-trained).
  curvature     : Laplacian of the nda*detj-weighted log template-density (the
                  prior's curvature; binning/selection bias scales with it).
Then Spearman-correlates |m| and m/sigma against each.

Run:  PYTHONPATH=. JAX_PLATFORMS=cpu python dev/bias_vs_training_curvature.py
"""
from __future__ import annotations

import sys
import numpy as np
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.ndimage import laplace

from bfd_cnf.config import PLOTS_DIR, TRAIN_FITS_PATH

# ---- grid (selection box) -------------------------------------------------
MF = (1500.0, 90000.0); MRMF = (2.2, 3.5)
NX, NY = 22, 16
xe = np.linspace(np.log10(MF[0]), np.log10(MF[1]), NX + 1)
ye = np.linspace(MRMF[0], MRMF[1], NY + 1)
xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])


def cellidx(u, v):
    ix = np.clip(np.digitize(u, xe) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(v, ye) - 1, 0, NY - 1)
    inside = (u >= xe[0]) & (u < xe[-1]) & (v >= ye[0]) & (v < ye[-1])
    return ix, iy, inside


def g1_cells(pqr, u, v, min_arm):
    """Per-cell summed-PQR g1 (NX,NY); NaN where < min_arm targets."""
    ix, iy, inside = cellidx(u, v)
    g = np.full((NX, NY), np.nan)
    P, Q, R = pqr[:, 0], pqr[:, 1:3], pqr[:, 3:6]
    for i in range(NX):
        for j in range(NY):
            m = inside & (ix == i) & (iy == j) & (P >= 1e-10)
            if m.sum() < min_arm:
                continue
            Pi, Qi, Ri = P[m], Q[m], R[m]
            Qtot = np.nansum(Qi / Pi[:, None], axis=0)
            Rmat = np.empty((m.sum(), 2, 2))
            Rmat[:, 0, 0] = Ri[:, 0]; Rmat[:, 1, 1] = Ri[:, 1]
            Rmat[:, 0, 1] = Rmat[:, 1, 0] = Ri[:, 2]
            Rtot = np.nansum(np.einsum("ni,nj->nij", Qi, Qi) / Pi[:, None, None] ** 2
                             - Rmat / Pi[:, None, None], axis=0)
            try:
                g = g.copy() if False else g
                g[i, j] = np.linalg.solve(Rtot, Qtot)[0]
            except np.linalg.LinAlgError:
                pass
    return g


# ---- bias map -------------------------------------------------------------
INP = sys.argv[1] if len(sys.argv) > 1 else "data/pqr_grid_indep_1500_90000.npz"
print(f"bias npz: {INP}")
d = np.load(INP)
def arm(pfx):
    t = d[f"targets_{pfx}"]; return np.log10(t[:, 0]), t[:, 1] / t[:, 0], d[f"pqr_{pfx}"].astype(np.float64)
up, vp, pqrp = arm("p"); um, vm, pqrm = arm("m")
MIN_ARM = 40
g1p = g1_cells(pqrp, up, vp, MIN_ARM)
g1m = g1_cells(pqrm, um, vm, MIN_ARM)
mmap = (g1p - g1m) / 0.04 - 1.0  # (NX,NY)

# ---- training template density / SNIS ESS / curvature ---------------------
F = fitsio.FITS(TRAIN_FITS_PATH)[1]
ntot = F.get_nrows()
# read ~24 contiguous chunks spread across the file to dodge any row ordering.
nchunk, csz = 24, 80_000
starts = np.linspace(0, ntot - csz, nchunk).astype(int)
us, vs, ws = [], [], []
for s in starts:
    r = F.read(rows=np.arange(s, s + csz), columns=["moments", "nda"])
    mom = np.asarray(r["moments"]).reshape(-1, 5)      # flatten 3 copies
    nda = np.asarray(r["nda"]).reshape(-1)
    Mf, Mr, M1, M2 = mom[:, 0], mom[:, 1], mom[:, 2], mom[:, 3]
    detj = 0.25 * (Mr**2 - M1**2 - M2**2)
    w = nda * np.maximum(detj, 0.0)
    ok = (Mf > 0) & (Mr > 0) & np.isfinite(w) & (w > 0)
    us.append(np.log10(Mf[ok])); vs.append(Mr[ok] / Mf[ok]); ws.append(w[ok])
u = np.concatenate(us); v = np.concatenate(vs); w = np.concatenate(ws)
print(f"templates read: {u.size:,} copies across {nchunk} chunks")

ix, iy, inside = cellidx(u, v)
flat = ix * NY + iy
sumw = np.bincount(flat[inside], weights=w[inside], minlength=NX * NY).astype(float)
sumw2 = np.bincount(flat[inside], weights=w[inside] ** 2, minlength=NX * NY)
cnt = np.bincount(flat[inside], minlength=NX * NY).astype(float)
ess = np.where(sumw2 > 0, sumw**2 / sumw2, np.nan)        # effective # templates / cell
dens = (sumw / sumw.sum()).reshape(NX, NY)                 # nda*detj-weighted density
ess = ess.reshape(NX, NY); cnt = cnt.reshape(NX, NY)
logdens = np.log(np.where(dens > 0, dens, np.nan))
curv = np.abs(laplace(np.nan_to_num(logdens, nan=np.nanmin(logdens))))  # |Laplacian log-density|

# ---- per-cell TARGET counts (per arm) -- needed to control the noise confound ---
def cellcount(u, v):
    ix, iy, inside = cellidx(u, v)
    return np.bincount((ix * NY + iy)[inside], minlength=NX * NY).reshape(NX, NY)
ncp = cellcount(up, vp); ncm = cellcount(um, vm)
ntcell = np.minimum(ncp, ncm).astype(float)   # min over arms (the binding constraint)

# ---- correlations ---------------------------------------------------------
def partial_spearman(y, x, z, mask):
    """Spearman of y vs x after regressing the RANKS of both on rank(z)."""
    from scipy.stats import rankdata
    ry, rx, rz = rankdata(y[mask]), rankdata(x[mask]), rankdata(z[mask])
    Z = np.c_[np.ones(rz.size), rz]
    ry_r = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    rx_r = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    rho, p = spearmanr(ry_r, rx_r)
    return rho, p

flat_ok = np.isfinite(mmap) & np.isfinite(ess) & np.isfinite(curv) & (ntcell > 0)
am = np.abs(mmap)
logcnt = np.log(ntcell + 1)
print(f"\nvalid cells: {flat_ok.sum()}   (target-count/cell: "
      f"median {np.median(ntcell[flat_ok]):.0f}, "
      f"p10 {np.percentile(ntcell[flat_ok],10):.0f})")

preds = {"log train density": np.log(dens + 1e-30),
         "SNIS ESS (eff #tmpl)": ess,
         "|curvature| log-dens": curv,
         "log target count": logcnt}
print("\n--- raw Spearman |m| vs predictor (CONFOUNDED by per-cell noise) ---")
for nm, x in preds.items():
    rho, p = spearmanr(am[flat_ok], x[flat_ok])
    print(f"  |m| vs {nm:<22}: rho={rho:+.3f}  p={p:.1e}")

print("\n--- PARTIAL Spearman |m| vs predictor, controlling for log target count ---")
print("    (removes the 'low count => noisy m => big |m|' artifact)")
for nm, x in preds.items():
    if nm == "log target count":
        continue
    rho, p = partial_spearman(am, x, logcnt, flat_ok)
    print(f"  |m| vs {nm:<22} | count: rho={rho:+.3f}  p={p:.1e}")

# Cross-check: well-sampled cells only (per-cell m is trustworthy here).
hi = flat_ok & (ntcell >= np.nanpercentile(ntcell[flat_ok], 60))
print(f"\n--- well-sampled cells only (count >= p60 = "
      f"{np.nanpercentile(ntcell[flat_ok],60):.0f}; n={hi.sum()}) ---")
for nm, x in preds.items():
    rho, p = spearmanr(am[hi], x[hi])
    print(f"  |m| vs {nm:<22}: rho={rho:+.3f}  p={p:.1e}")

# ---- figure ---------------------------------------------------------------
ext = [xe[0], xe[-1], ye[0], ye[-1]]
def show(ax, M, title, cmap, **kw):
    im = ax.imshow(M.T, origin="lower", extent=ext, aspect="auto", cmap=cmap, **kw)
    ax.set_title(title, fontsize=10); ax.set_xlabel("log10 Mf"); ax.set_ylabel("Mr/Mf")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
mx = np.nanpercentile(np.abs(mmap), 95)
show(ax[0, 0], mmap, "m (bias)", "RdBu_r", vmin=-mx, vmax=mx)
show(ax[0, 1], np.log(dens + 1e-30), "log nda*detj train density", "viridis")
show(ax[0, 2], ess, "SNIS ESS (eff # templates/cell)", "magma")
show(ax[1, 0], curv, "|curvature| of log prior", "cividis")
show(ax[1, 1], np.log(cnt + 1), "log target count", "viridis")
ax[1, 2].scatter(np.log(dens + 1e-30)[flat_ok], am[flat_ok], s=10, alpha=.6, label="vs logdens")
ax[1, 2].scatter(curv[flat_ok], am[flat_ok], s=10, alpha=.6, label="vs curv")
ax[1, 2].set_xlabel("predictor"); ax[1, 2].set_ylabel("|m|"); ax[1, 2].legend(fontsize=8)
fig.tight_layout()
import os
tag = os.path.basename(INP).replace(".npz", "")
out = f"{PLOTS_DIR}/bias_vs_training_curvature_{tag}.png"
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
