"""Z(g): the q-mass SUM_u L(X^G(g,u)) d2u per galaxy, averaged over the population,
from the copies (truth, numpy only).  Under the consistent J model the estimator's
P(M|g) = J(M) INT p_flow(m|g)/J(m) N(M;m,C) dm carries Z(g) = E_flow(g)[1/J(m)] in
every target, so the flow must reproduce d2 log Z/dg2.  Also the J-weighted mass
(the detection probability, should be flat in g) as a sanity check, and the
galaxy-centre proxy E_G[1/J(M^G(g))].  sn8 copies at the sn8r Sigma_X.
Usage: python dev/z_of_g.py
"""
import numpy as np, fitsio
D = "../bfd_cnf_imsims/data/"
sx = fitsio.read(D + "targets_bdg2_g0_176k_sn8r.fits", columns=["cov_odd"], rows=[0])["cov_odd"][0]
c00, c01, c11 = sx; det = c00 * c11 - c01 ** 2
H = 0.02
GS = [(-H, 0.0), (0.0, 0.0), (H, 0.0), (0.0, -H), (0.0, H)]


def lens(v, dv, d2v, g):
    g1, g2 = g
    return v + g1 * dv[:, 0] + g2 * dv[:, 1] + np.tensordot([0.5 * g1 * g1, g1 * g2, 0.5 * g2 * g2], d2v, axes=(0, 1))


J = lambda m: 0.25 * (m[..., 1] ** 2 - m[..., 2] ** 2 - m[..., 3] ** 2)
f = fitsio.FITS(D + "copies_bulgedisc_g2_bdg2n_sn8.fits")
n, ngal = f["COPIES"].get_nrows(), f["GALAXIES"].get_nrows()
zq, zp = np.zeros(len(GS)), np.zeros(len(GS))
for i in range(0, n, 3_000_000):
    c = f["COPIES"].read(columns=["moments", "xy", "da", "dm_dg", "d2m_dg2", "dxy_dg", "d2xy_dg2"],
                         rows=np.arange(i, min(i + 3_000_000, n)))
    for k, g in enumerate(GS):
        x = lens(c["xy"], c["dxy_dg"], c["d2xy_dg2"], g)
        chi = (c11 * x[:, 0] ** 2 - 2 * c01 * x[:, 0] * x[:, 1] + c00 * x[:, 1] ** 2) / det
        lx = c["da"] * np.exp(-0.5 * chi) / (2 * np.pi * np.sqrt(det))
        zq[k] += lx.sum()
        zp[k] += (lx * J(lens(c["moments"], c["dm_dg"], c["d2m_dg2"], g))).sum()
zq /= ngal; zp /= ngal
gal = f["GALAXIES"].read(columns=["moments", "dm_dg", "d2m_dg2"])
zc = np.array([np.mean(1 / J(lens(gal["moments"], gal["dm_dg"], gal["d2m_dg2"], g))) for g in GS])
for name, z in (("q-mass Z(g) = SUM L(X) d2u", zq), ("detection prob SUM J L d2u", zp),
                ("centre proxy E[1/J(M^G(g))]", zc)):
    lz = np.log(z)
    print(f"{name:32s} Z(0) = {z[1]:.6e}   d2lnZ/dg1^2 = {(lz[0] - 2 * lz[1] + lz[2]) / H**2:+.4f}"
          f"   d2lnZ/dg2^2 = {(lz[3] - 2 * lz[1] + lz[4]) / H**2:+.4f}   dlnZ/dg1 = {(lz[2] - lz[0]) / (2 * H):+.2e}")
