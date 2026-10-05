"""Where does the copies' d<e1>/dg1 come from? (2026-09-29; numpy only)

Splits the copies' band-mean e1 response (dev/flow_tilt_response.py's reference)
into moment lensing (weights at g=0) and weight change (moments at g=0), plus a
variant that also lenses the copy's |J(u)| (log_weights keeps it at g=0), and
the parent galaxies' own response (unit weight, no centroid) for comparison.
Size 2.0-3.4, Mf bands as there, g1 = +/-H.
"""
import sys
import numpy as np, fitsio
sys.path.insert(0, "../bfd_cnf_imsims")
from imsims.copies import log_weights, log_jacobian

D = "../bfd_cnf_imsims/data/"
H, SIZE = 0.02, (2.0, 3.4)
BANDS = [(2000, 3000), (3000, 3500), (3500, 4000), (4000, 5000), (5000, 8000), (8000, 20000)]


def stats(m, w):
    s, e1 = m[:, 1] / m[:, 0], m[:, 2] / m[:, 1]
    out = np.zeros((len(BANDS), 2))
    for b, (lo, hi) in enumerate(BANDS):
        k = (m[:, 0] >= lo) & (m[:, 0] < hi) & (s > SIZE[0]) & (s < SIZE[1])
        out[b] = w[k].sum(), (w[k] * e1[k]).sum()
    return out


resp = lambda p, m: (p[:, 1] / p[:, 0] - m[:, 1] / m[:, 0]) / (2 * H)
sx = fitsio.read(D + "targets_bdg2_g0_176k_sn8r.fits", columns=["cov_odd"], rows=[0])["cov_odd"][0]
V = ("full", "moments only", "weights only", "full + lensed J")
acc = {v: [0, 0] for v in V}
f = fitsio.FITS(D + "copies_bulgedisc_g2_bdg2n_sn8.fits")["COPIES"]
n, step = f.get_nrows(), 3_000_000
for i in range(0, n, step):
    c = f.read(columns=["moments", "xy", "da", "dm_dg", "d2m_dg2", "dxy_dg", "d2xy_dg2"],
               rows=np.arange(i, min(i + step, n)))
    m0 = np.asarray(c["moments"], np.float64)
    w0 = np.exp(log_weights(c, sx))
    for j, sg in enumerate((1, -1)):
        g = np.array([sg * H, 0.0])
        ml = m0 + sg * H * c["dm_dg"][:, 0] + 0.5 * H * H * c["d2m_dg2"][:, 0]
        wl = np.exp(log_weights(c, sx, g=g))
        wj = wl * np.exp(log_jacobian(ml) - log_jacobian(m0))
        for v, (mm, ww) in zip(V, ((ml, wl), (ml, w0), (m0, wl), (ml, wj))):
            acc[v][j] = acc[v][j] + stats(mm, ww)
print("copies done", flush=True)

pm, pdm, pd2m = (fitsio.read(D + "moments_bulgedisc_g2_bdg2n.fits", columns=[k])[k].astype(np.float64)
                 for k in ("moments", "dm_dg", "d2m_dg2"))
par = [stats(pm + sg * H * pdm[:, 0] + 0.5 * H * H * pd2m[:, 0], np.ones(len(pm))) for sg in (1, -1)]

print(f"size {SIZE}, d<e1>/dg1:")
print(f"{'Mf band':>11s} " + " ".join(f"{v:>16s}" for v in V) + f" {'parents':>10s}")
for b, (lo, hi) in enumerate(BANDS):
    print(f"{lo:>5d}-{hi:<5d} " + " ".join(f"{resp(*acc[v])[b]:>16.4f}" for v in V)
          + f" {resp(*par)[b]:>10.4f}")
