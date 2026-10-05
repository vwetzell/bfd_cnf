"""Paper eq. 36 detection vs the sims' recentring, galaxy by galaxy. (2026-10-01)

Truegal drops 7.8% of the sn8r galaxies (v < sum of copy weights), the sims only
2.9% (badcenter).  Same galaxies row for row (seed-2 copies = sn8r targets), so
per noiseless bin compare P_det = sum_u w(u; g) (|J| lensed) with the sims'
detected fraction, and the shear response of detection
  R_det = < e1_0 (det(+g) - det(-g)) > / 2g      (e1_0: noiseless e1 at g=0)
which is detection's contribution to the population's d<e1>/dg.  Numpy only.
  python dev/detect_vs_recenter.py
"""
import sys
import numpy as np, fitsio
sys.path.insert(0, "../bfd_cnf_imsims")
from imsims.copies import log_weights, log_jacobian

D = "../bfd_cnf_imsims/data/"
H = 0.02
CF = D + "copies_bulgedisc_g2_bdg2n_sn8_seed2.fits"
SIMS = {g: f"{D}targets_bdg2_{t}_176k_sn8r.fits" for g, t in ((H, "g1p02"), (-H, "g1m02"), (0., "g0"))}

g0 = fitsio.read(CF, ext="GALAXIES", columns=["moments", "badcenter"])
m0 = g0["moments"].astype(np.float64)
ng = len(m0)
sx = fitsio.read(SIMS[H], columns=["cov_odd"], rows=[0])["cov_odd"][0]
P = {g: np.zeros(ng) for g in SIMS}
f = fitsio.FITS(CF)["COPIES"]
n, step = f.get_nrows(), 3_000_000
for i in range(0, n, step):
    c = f.read(columns=["gal", "moments", "xy", "da", "dm_dg", "d2m_dg2", "dxy_dg", "d2xy_dg2"], rows=np.arange(i, min(i + step, n)))
    mc = np.asarray(c["moments"], np.float64)
    for g in SIMS:
        ml = mc + g * c["dm_dg"][:, 0] + 0.5 * g * g * c["d2m_dg2"][:, 0]
        lw = log_weights(c, sx, g=np.array([g, 0.0])) + log_jacobian(ml) - log_jacobian(mc)
        P[g] += np.bincount(c["gal"], np.where(np.isfinite(lw), np.exp(lw), 0.0), ng)
Ds = {g: ~fitsio.read(p, columns=["badcenter"])["badcenter"] for g, p in SIMS.items()}

f0, r0, e1 = m0[:, 0], m0[:, 1] / m0[:, 0], m0[:, 2] / m0[:, 1]
print(f"all: P_det {P[0.].mean():.4f}  sims detected {Ds[0.].mean():.4f}   (copies-build badcenter {g0['badcenter'].mean():.4f})")
print(f"corr(P_det, sims det) {np.corrcoef(P[0.], Ds[0.])[0, 1]:.3f}")
for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, .99), (.99, 2)):
    k = (P[0.] >= lo) & (P[0.] < hi)
    print(f"  P_det {lo:.2f}-{hi:.2f}: N {k.sum():>6d}  mean P {P[0.][k].mean():.3f}  sims detected {Ds[0.][k].mean():.3f}")


def rows(name, v, edges, extra):
    print(f"\nnoiseless {name} bin ({extra[0]})       N   P_det  sims_det    R_det model     R_det sims   diff")
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (v >= lo) & (v < hi) & extra[1]
        rm = e1[k] * (P[H][k] - P[-H][k]) / (2 * H)
        rs = e1[k] * (Ds[H][k].astype(float) - Ds[-H][k]) / (2 * H)
        d = rm - rs
        se = lambda x: x.std() / np.sqrt(k.sum())
        print(f"{lo:>8g}-{hi:<8g} {k.sum():>8d} {P[0.][k].mean():.4f} {Ds[0.][k].mean():.4f}"
              f"  {rm.mean():+.4f}+/-{se(rm):.4f} {rs.mean():+.4f}+/-{se(rs):.4f} {d.mean():+.4f}+/-{se(d):.4f}")


rows("Mf", f0, [0, 2000, 3000, 4000, 5000, 7000, 10000, 20000, np.inf], ("size 2.2-3.2", (r0 > 2.2) & (r0 < 3.2)))
rows("Mr/Mf", r0, [0, 2.2, 2.3, 2.45, 2.7, 2.95, 3.1, 3.2, np.inf], ("Mf 3000-20000", (f0 > 3000) & (f0 < 20000)))
