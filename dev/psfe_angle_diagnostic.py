"""Does the per-galaxy PSF-ellipticity leak depend on the galaxy's OWN
ellipticity (magnitude / relative angle to the PSF), or only on the PSF's?

`SigmaXBlockLayer.net_dipquad` outputs D,c as isotropic functions of
(flux, size, |e_psf|^2) only -- never seeing the galaxy's own ellipticity
(x3,x4). The composition itself (`m3,m4` in `_ellipticity`) IS a proper
bilinear spin-2 map so relative angle already enters through x3,x4 times
(e1,e2) -- but D,c's *strength* can't adapt to it. This checks, empirically,
whether the per-galaxy moment shift induced by turning on PSF ellipticity
correlates with the galaxy's own zero-shear e1,e2 (from psfe00) -- via
|e_gal|^2 and via the joint invariant e_gal . e_psf.

Uses the SAME matched-pairs alignment as dev/psfe_paired_diff.py (nearest-
neighbor on the saved `moments` field), but instead of feeding aligned rows
into bias.py's PQR sum, looks directly at the per-galaxy DELTA in the
zero-shear moments field (M1/Mr, M2/Mr) between psfe00 and each psfeXX
config -- this delta IS is the leak at the single-galaxy level, before any
population-level PQR summation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree

CONFIGS = {
    "psfe1p02": (0.02, 0.0), "psfe1p05": (0.05, 0.0), "psfe1p10": (0.10, 0.0),
    "psfe2p02": (0.0, 0.02), "psfe2p05": (0.0, 0.05), "psfe2p10": (0.0, 0.10),
    "psfe1m02": (-0.02, 0.0), "psfe1m05": (-0.05, 0.0), "psfe1m10": (-0.10, 0.0),
}


def load(tag):
    return np.load(f"pqr/g2v3d_{tag}.npz")


def align(a, b):
    tree = cKDTree(a["moments"])
    dist, j = tree.query(b["moments"])
    self_dist, _ = cKDTree(b["moments"]).query(b["moments"], k=2)
    scale = np.median(self_dist[:, 1])
    ok = dist < 0.5 * scale
    i = j[ok]
    bidx = np.flatnonzero(ok)
    uniq, counts = np.unique(i, return_counts=True)
    keep_i = set(uniq[counts == 1])
    sel = np.array([k for k, ii in enumerate(i) if ii in keep_i])
    return i[sel], bidx[sel]


def egal(d, idx):
    m = d["moments"][idx]
    return m[:, 2] / m[:, 1], m[:, 3] / m[:, 1]


def main():
    base = load("psfe00")
    rows = []
    for tag, (e1p, e2p) in CONFIGS.items():
        b = load(tag)
        ia, ib = align(base, b)
        e1g, e2g = egal(base, ia)          # galaxy's OWN zero-shear ellipticity (baseline)
        e1b, e2b = egal(b, ib)
        de1, de2 = e1b - e1g, e2b - e2g    # per-galaxy leak vector
        e_mag_sq = e1g**2 + e2g**2
        dot = e1g * e1p + e2g * e2p        # joint invariant: e_gal . e_psf
        # component of the leak ALONG e_psf direction (the physically relevant one)
        e_psf_mag = np.hypot(e1p, e2p)
        if e_psf_mag > 0:
            along = (de1 * e1p + de2 * e2p) / e_psf_mag
        else:
            along = np.zeros_like(de1)
        rows.append((tag, len(ia), e_mag_sq, dot, along))
        r_mag = np.corrcoef(e_mag_sq, along)[0, 1] if len(ia) > 10 else np.nan
        r_dot = np.corrcoef(dot, along)[0, 1] if len(ia) > 10 else np.nan
        print(f"{tag:>10s}  n={len(ia):6d}  "
              f"mean(along)={along.mean():+.3e}  "
              f"corr(along, |e_gal|^2)={r_mag:+.3f}  "
              f"corr(along, e_gal.e_psf)={r_dot:+.3f}")

    # pool across all configs (normalize e_psf-amplitude sign/scale out by
    # using the 'along' component and 'dot' directly -- both already carry
    # the PSF direction)
    all_mag = np.concatenate([r[2] for r in rows])
    all_dot = np.concatenate([r[3] for r in rows])
    all_along = np.concatenate([r[4] for r in rows])
    print(f"\npooled (n={len(all_along)}): "
          f"corr(along, |e_gal|^2)={np.corrcoef(all_mag, all_along)[0,1]:+.4f}  "
          f"corr(along, e_gal.e_psf)={np.corrcoef(all_dot, all_along)[0,1]:+.4f}")

    # simple linear regression of 'along' on [1, |e_gal|^2, e_gal.e_psf] pooled,
    # to see if either term has a nonzero slope with a rough error bar
    X = np.column_stack([np.ones_like(all_along), all_mag, all_dot])
    beta, *_ = np.linalg.lstsq(X, all_along, rcond=None)
    resid = all_along - X @ beta
    dof = len(all_along) - 3
    cov = np.linalg.inv(X.T @ X) * (resid @ resid / dof)
    se = np.sqrt(np.diag(cov))
    names = ["const", "|e_gal|^2", "e_gal.e_psf"]
    print("\nOLS  along ~ const + |e_gal|^2 + e_gal.e_psf  (pooled, all configs, all amplitudes)")
    for n, b_, s_ in zip(names, beta, se):
        print(f"  {n:>12s}  {b_:+.4e} +/- {s_:.4e}  ({b_/s_:+.2f} sigma)")


if __name__ == "__main__":
    main()
