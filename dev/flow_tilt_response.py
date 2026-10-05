"""Flow vs copies: shear response of e1, plain and 1/Mr^2-tilted, at the faint end (2026-09-29).

The consistent-J target weight is effectively 1/J ~ 4/Mr^2 (dev/flow_jedge.py:
t = 1-|e|^2 ~ 0.99 in the window), and its m1 shift sits at Mf 3000-5000.  If
the flow's size density or its shear response is wrong there, the tilt exposes
it.  Per (Mf band, size 2.0-3.4), on lensed moments at g1 = +/-H (CRN for both):
  <tau>             tau = 1e7/Mr^2, the tilt (density check)
  d<e1>/dg1         plain band mean
  d<e1>_tau/dg1     tau-weighted band mean
Flow: same base draws transformed at +/-H (sn8r Sigma_X).  Copies: each copy
lensed with its own dm_dg/d2m_dg2 and weighted by log_weights(g=+/-H).
Errors: std over chunks / sqrt(n_chunks).
"""
import sys
import numpy as np, fitsio, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
sys.path.insert(0, "."); sys.path.insert(0, "../bfd_cnf_imsims")
import bias as B, bulk, shear
from imsims.copies import log_weights

D = "../bfd_cnf_imsims/data/"
FLOW, H, NCH, CH = "flows/cv2/centroid_s0.eqx", 0.02, 32, 500_000
BANDS = [(2000, 3000), (3000, 3500), (3500, 4000), (4000, 5000), (5000, 8000), (8000, 20000)]
SIZE = (2.0, 3.4)


def stats(m, w):
    """per band: sum w, sum w e1, sum w tau, sum w tau e1."""
    s, e1, tau = m[:, 1] / m[:, 0], m[:, 2] / m[:, 1], 1e7 / m[:, 1] ** 2
    out = np.zeros((len(BANDS), 4))
    for b, (lo, hi) in enumerate(BANDS):
        k = (m[:, 0] >= lo) & (m[:, 0] < hi) & (s > SIZE[0]) & (s < SIZE[1])
        wk = w[k]
        out[b] = wk.sum(), (wk * e1[k]).sum(), (wk * tau[k]).sum(), (wk * tau[k] * e1[k]).sum()
    return out


def summarize(p, m):
    """p, m: (chunks, bands, 4) sums at +H, -H -> per-chunk (tau, resp, resp_tau)."""
    P, M = p.sum(0), m.sum(0)
    val = lambda P, M: np.stack([(P[:, 2] + M[:, 2]) / (P[:, 0] + M[:, 0]),
                                 (P[:, 1] / P[:, 0] - M[:, 1] / M[:, 0]) / (2 * H),
                                 (P[:, 3] / P[:, 2] - M[:, 3] / M[:, 2]) / (2 * H)], 1)
    v = val(P, M)
    jk = np.array([val(P - p[i], M - m[i]) for i in range(len(p))])   # delete-one jackknife
    n = len(p)
    return v, np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))


sx = fitsio.read(D + "targets_bdg2_g0_176k_sn8r.fits", columns=["cov_odd"], rows=[0])["cov_odd"][0]
sxj = jnp.asarray(sx, jnp.float32)
flow = eqx.tree_deserialise_leaves(FLOW, bulk.build_flow(
    jr.key(0), shear.load(D + "moments_bulgedisc_g2_bdg2n.fits")[0], shear=True, centroid=True))
tr = eqx.filter_jit(jax.vmap(lambda z, c: flow.bijection.transform(z, c), in_axes=(0, None)))
cp, cm = B.condition(jnp.array([H, 0.]), sxj), B.condition(jnp.array([-H, 0.]), sxj)
fp, fm = [], []
for i in range(NCH):
    z = flow.base_dist.sample(jr.key(500 + i), (CH,))
    a, b = np.asarray(tr(z, cp), np.float64), np.asarray(tr(z, cm), np.float64)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    fp.append(stats(a[ok], np.ones(ok.sum()))); fm.append(stats(b[ok], np.ones(ok.sum())))
print("flow done", flush=True)

f = fitsio.FITS(D + "copies_bulgedisc_g2_bdg2n_sn8.fits")["COPIES"]
n, step = f.get_nrows(), 3_000_000
cols = ["moments", "xy", "da", "dm_dg", "d2m_dg2", "dxy_dg", "d2xy_dg2"]
kp, km = [], []
for i in range(0, n, step):
    c = f.read(columns=cols, rows=np.arange(i, min(i + step, n)))
    m0 = np.asarray(c["moments"], np.float64)
    d1, d2 = np.asarray(c["dm_dg"][:, 0], np.float64), np.asarray(c["d2m_dg2"][:, 0], np.float64)
    for sgn, acc in ((1, kp), (-1, km)):
        g = np.array([sgn * H, 0.0])
        acc.append(stats(m0 + sgn * H * d1 + 0.5 * H * H * d2, np.exp(log_weights(c, sx, g=g))))
print("copies done", flush=True)

vf, ef = summarize(np.array(fp), np.array(fm))
vc, ec = summarize(np.array(kp), np.array(km))
print(f"\nsize {SIZE}, H = {H}.  flow / copies (+/- jackknife)")
print(f"{'Mf band':>11s} {'<tau> flow':>11s} {'copies':>8s} {'ratio':>7s}   "
      f"{'de1/dg flow':>12s} {'copies':>8s} {'ratio':>14s}   {'tilted flow':>12s} {'copies':>8s} {'ratio':>14s}")
for b, (lo, hi) in enumerate(BANDS):
    r = vf[b] / vc[b]
    re = np.abs(r) * np.hypot(ef[b] / vf[b], ec[b] / vc[b])
    print(f"{lo:>5d}-{hi:<5d} {vf[b, 0]:>11.4f} {vc[b, 0]:>8.4f} {r[0]:>7.4f}   "
          f"{vf[b, 1]:>12.4f} {vc[b, 1]:>8.4f} {r[1]:>7.4f}+/-{re[1]:.4f}   "
          f"{vf[b, 2]:>12.4f} {vc[b, 2]:>8.4f} {r[2]:>7.4f}+/-{re[2]:.4f}")
