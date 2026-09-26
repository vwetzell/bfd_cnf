"""Centroid-jitter broadening the deterministic centroid layer cannot carry.

Per galaxy, the eq.-36-weighted copies are the distribution of its measured
(noise-free) moments over the recentring offset u.  The layer maps each
galaxy to ONE point (the weighted mean); the flow adds only C_M noise on
top.  The missing per-galaxy covariance V = Cov_w[M^G(u)] would inflate the
real targets' score variance relative to the model's curvature by about

    excess ~ sum_a v_a^T V v_a / sum_a v_a^T C_M v_a,

with v_a the galaxy's own shear response dM/dg_a (weighted copy mean of
dm_dg).  Compared per |e| bin with the paired real-minus-closed Fisher
excess (dev/paired_closed_loop.py: -0.005, +0.018, +0.029, +0.040, +0.083).
Numpy only.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "../bfd_cnf_imsims")

import fitsio
import numpy as np

import bias as B
from imsims.copies import log_weights

D = "../bfd_cnf_imsims/data"
COPIES = f"{D}/copies_gauss2_fwd_g2v4n.fits"
ROWS = 20_000_000
OBS = [-0.0048, 0.0175, 0.0294, 0.0397, 0.0828]


def main():
    tgt = f"{D}/{B.CATALOGS['gauss2_v4n']['plus']}.fits"
    sx = fitsio.read(tgt, rows=[0])["cov_odd"][0]
    CM = B.load_cov(tgt)
    c = fitsio.FITS(COPIES)["COPIES"].read(
        columns=["gal", "moments", "xy", "da", "dm_dg"], rows=np.arange(ROWS))
    c = c[c["gal"] < c["gal"][-1]]
    lw = log_weights(c, sx)
    gal, inv = np.unique(c["gal"], return_inverse=True)
    w = np.exp(lw - np.maximum.reduceat(lw, np.r_[0, np.flatnonzero(np.diff(inv)) + 1])[inv])
    W = np.bincount(inv, w)
    m = np.asarray(c["moments"], np.float64)
    mean = np.stack([np.bincount(inv, w * m[:, i]) for i in range(5)], 1) / W[:, None]
    dmc = m - mean[inv]
    V = np.stack([np.stack([np.bincount(inv, w * dmc[:, i] * dmc[:, j]) for j in range(5)], 1)
                  for i in range(5)], 1) / W[:, None, None]                        # (G,5,5)
    dm = np.asarray(c["dm_dg"], np.float64)                                      # (n,2,5)
    v = np.stack([np.stack([np.bincount(inv, w * dm[:, a, i]) for i in range(5)], 1)
                  for a in range(2)], 1) / W[:, None, None]                        # (G,2,5)
    num = np.einsum("gai,gij,gaj->g", v, V, v)
    den = np.einsum("gai,ij,gaj->g", v, CM, v)

    e = np.hypot(mean[:, 2], mean[:, 3]) / mean[:, 1]
    win = B.window_mask(mean, (2.2, 3.2), (3000.0, 20000.0))
    q = np.quantile(e[win], np.linspace(0, 1, 6)); q[-1] += 1
    print(f"{len(gal)} galaxies, {win.sum()} in the window (by their weighted-mean moments)")
    print(f"median V/C_M on the diagonal (in window): "
          + "  ".join(f"{np.median(V[win, i, i] / CM[i, i]):.3e}" for i in range(5)))
    print(f"\n{'|e| bin':>13s} {'n':>6s}  {'predicted excess':>16s}  {'observed (paired)':>17s}")
    for (lo, hi), o in zip(zip(q[:-1], q[1:]), OBS):
        k = win & (e >= lo) & (e < hi)
        print(f"{lo:.3f}-{hi:.3f} {k.sum():6d}  {num[k].sum() / den[k].sum():+16.4f}  {o:+17.4f}")


if __name__ == "__main__":
    main()
