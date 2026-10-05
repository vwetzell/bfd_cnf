"""Flow vs the truegal galaxies' copies, in the window's own bins. (2026-10-01)

Truegal (dev/truegal_loop.py, m1 +0.0179) and the closed loop (+0.0002) share the
noise model and the flow; only the galaxies differ (seed-2 copies vs flow draws).
So the flow's density disagrees with the copies-defined one somewhere -- where?
  noisy:      closedJ vs truegal catalogs, binned on each arm's observed moments
  noiseless:  flow (sn8r Sigma_X) vs seed-2 copies weighted by eq. 36 with |J|
              lensed (the flow's own training target, CopySampler's weighting)
Per bin: mass fraction (g=0 arm) and d<e1>/dg1 (+/- arms, own-arm membership),
jackknife over row chunks; diff = flow/closed - copies/truegal.
  BFD_LEGACY_CHART=1 python dev/flow_vs_truegal.py flows/cv2/centroid_s0.eqx
"""
import sys
import numpy as np, fitsio
sys.path.insert(0, "."); sys.path.insert(0, "../bfd_cnf_imsims")
import bias as B

D = "../bfd_cnf_imsims/data/"
H, S, F = 0.02, (2.2, 3.2), (3000, 20000)
BINS = [("window", "f", 0, np.inf)]
BINS += [("Mf", "f", lo, hi) for lo, hi in zip([3000, 4000, 5000, 7000, 10000], [4000, 5000, 7000, 10000, 20000])]
BINS += [("Mr/Mf", "r", lo, hi) for lo, hi in zip([2.2, 2.3, 2.45, 2.7, 2.95, 3.1], [2.3, 2.45, 2.7, 2.95, 3.1, 3.2])]


def stats(m, w):
    """(bins+1, 2): per bin sum w, sum w e1; last row total w."""
    win = B.window_mask(m, S, F)
    v = {"f": m[:, 0], "r": m[:, 1] / m[:, 0]}
    e1 = m[:, 2] / m[:, 1]
    out = np.zeros((len(BINS) + 1, 2))
    for b, (_, k, lo, hi) in enumerate(BINS):
        s = win & (v[k] >= lo) & (v[k] < hi)
        out[b] = w[s].sum(), (w[s] * e1[s]).sum()
    out[-1, 0] = w.sum()
    return out


def summarize(p, m, z):
    def val(P, M, Z):
        return np.stack([Z[:-1, 0] / Z[-1, 0], (P[:-1, 1] / P[:-1, 0] - M[:-1, 1] / M[:-1, 0]) / (2 * H)], 1)
    P, M, Z = p.sum(0), m.sum(0), z.sum(0)
    jk = np.array([val(P - p[i], M - m[i], Z - z[i]) for i in range(len(p))])
    n = len(p)
    return val(P, M, Z), np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))


def report(title, a, b, na, nb):
    (va, ea), (vb, eb) = a, b
    print(f"\n{title}   diff = {na} - {nb}")
    print(f"{'bin':>18s} {'frac ' + na:>14s} {nb:>8s} {'diff':>16s}   {'de1/dg ' + na:>16s} {nb:>8s} {'diff':>16s}")
    for i, (name, _, lo, hi) in enumerate(BINS):
        lab = name if name == "window" else f"{name} {lo:g}-{hi:g}"
        print(f"{lab:>18s} {va[i, 0]:>14.5f} {vb[i, 0]:>8.5f} {va[i, 0] - vb[i, 0]:>+8.5f}+/-{np.hypot(ea[i, 0], eb[i, 0]):.5f}"
              f"   {va[i, 1]:>16.4f} {vb[i, 1]:>8.4f} {va[i, 1] - vb[i, 1]:>+8.4f}+/-{np.hypot(ea[i, 1], eb[i, 1]):.4f}")


def catalog(pop):
    arms = [fitsio.read(f"{D}{B.CATALOGS[pop][a]}.fits", columns=["moments", "badcenter"]) for a in ("plus", "minus", "zero")]
    acc = [[], [], []]
    for idx in np.array_split(np.arange(len(arms[0])), 32):
        for a, c in zip(acc, arms):
            m = c["moments"][idx].astype(np.float64)
            a.append(stats(m, (~c["badcenter"][idx]).astype(float)))
    return summarize(*map(np.array, acc))


report("NOISY catalogs (same noise model, same flow's selection)", catalog("bdg2n_sn8r_closedJ"),
       catalog("bdg2n_sn8r_truegal"), "closedJ", "truegal")
sys.stdout.flush()
if len(sys.argv) < 2:
    sys.exit()

import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
import bulk, shear
from imsims.copies import log_weights, log_jacobian

sx = fitsio.read(f"{D}{B.CATALOGS['bulgedisc_g2n_176k_sn8r']['plus']}.fits", columns=["cov_odd"], rows=[0])["cov_odd"][0]
mt = shear.load(D + "moments_bulgedisc_g2_bdg2n.fits")[0]
fl = bulk.load_flow(sys.argv[1], mt, key=jr.key(0), shear=True, centroid=True)
tr = eqx.filter_jit(jax.vmap(lambda z, c: fl.bijection.transform(z, c), in_axes=(0, None)))
acc = [[], [], []]
for i in range(32):
    z = fl.base_dist.sample(jr.key(500 + i), (500_000,))
    out = [np.asarray(tr(z, B.condition(jnp.array([g, 0.]), jnp.asarray(sx, jnp.float32))), np.float64) for g in (H, -H, 0.)]
    ok = np.all([np.isfinite(o).all(1) for o in out], 0)
    for a, o in zip(acc, out):
        a.append(stats(o[ok], np.ones(ok.sum())))
flow = summarize(*map(np.array, acc))

f = fitsio.FITS(D + "copies_bulgedisc_g2_bdg2n_sn8_seed2.fits")["COPIES"]
n, step = f.get_nrows(), 3_000_000
acc = [[], [], []]
for i in range(0, n, step):        # ponytail: block edges split a galaxy; fine for jackknife errors
    c = f.read(columns=["moments", "xy", "da", "dm_dg", "d2m_dg2", "dxy_dg", "d2xy_dg2"], rows=np.arange(i, min(i + step, n)))
    m0 = np.asarray(c["moments"], np.float64)
    for a, g in zip(acc, (H, -H, 0.)):
        ml = m0 + g * c["dm_dg"][:, 0] + 0.5 * g * g * c["d2m_dg2"][:, 0]
        lw = log_weights(c, sx, g=np.array([g, 0.0])) + log_jacobian(ml) - log_jacobian(m0)
        a.append(stats(ml, np.where(np.isfinite(lw), np.exp(lw), 0.0)))
report(f"NOISELESS {sys.argv[1]} (sn8r Sigma_X) vs seed-2 copies, eq. 36 + lensed |J|", flow,
       summarize(*map(np.array, acc)), "flow", "copies")
