"""Per galaxy: which part of the centroid layer's shear response disagrees with the copies? (2026-10-01)

dev/stage_response.py showed the shear stage matching the templates, while the
full flow under-responds against the J-lensed copies at faint flux (0.91 at
Mf 2-3k).  Here the comparison is per GALAXY and deterministic, so no population
or band-migration confound.  For each copies galaxy (noiseless m0, sn8r Sigma_X),
d e1/dg1 by +/-H finite difference of:
  copies  full      weighted copy mean, copies' moments AND |J| lensed (eq. 36 weights at g)
          moments   lensed moments, g=0 weights
          weights   g=0 moments, lensed weights (incl. J)
  parent            m0 lensed analytically (no centroid)
  flow    shear     chart^-1(shear(g) z0)                       -- the bulk+shear part
          total     centroid(g, Sigma) after shear(g)
          implicit  centroid(0, Sigma) after shear(g)           -- layer carried along by its input
          explicit  centroid(g, Sigma) on the g=0 input         -- the layer's own g-conditioning (net_K)
Band means over galaxies with noiseless Mf in band and size 2.0-3.4.
  python dev/centroid_g_split.py flows/cv4/centroid_s2.eqx
"""
import sys
import numpy as np, fitsio, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
sys.path.insert(0, "."); sys.path.insert(0, "../bfd_cnf_imsims")
import bulk, shear
from imsims.copies import log_weights, log_jacobian

D = "../bfd_cnf_imsims/data/"
H, SIZE = 0.02, (2.0, 3.4)
BANDS = [(2000, 3000), (3000, 3500), (3500, 4000), (4000, 5000), (5000, 8000), (8000, 20000)]
CF = D + "copies_bulgedisc_g2_bdg2n_sn8.fits"
e1 = lambda m: m[:, 2] / m[:, 1]

gal = fitsio.read(CF, ext="GALAXIES", columns=["moments", "dm_dg", "d2m_dg2"])
m0 = gal["moments"].astype(np.float64)
ng = len(m0)
sx = fitsio.read(D + "targets_bdg2_g0_176k_sn8r.fits", columns=["cov_odd"], rows=[0])["cov_odd"][0]

# ---- copies: per-galaxy weighted means at +H, -H for three variants
V = ("full", "moments", "weights")
den = {v: np.zeros((2, ng)) for v in V}
num = {v: np.zeros((2, ng, 2)) for v in V}          # (M1, Mr) sums
f = fitsio.FITS(CF)["COPIES"]
n, step = f.get_nrows(), 3_000_000
for i in range(0, n, step):
    c = f.read(columns=["gal", "moments", "xy", "da", "dm_dg", "d2m_dg2", "dxy_dg", "d2xy_dg2"],
               rows=np.arange(i, min(i + step, n)))
    mc = np.asarray(c["moments"], np.float64)
    w0 = np.exp(log_weights(c, sx))
    for j, sg in enumerate((1, -1)):
        ml = mc + sg * H * c["dm_dg"][:, 0] + 0.5 * H * H * c["d2m_dg2"][:, 0]
        lw = log_weights(c, sx, g=np.array([sg * H, 0.0])) + log_jacobian(ml) - log_jacobian(mc)
        wl = np.where(np.isfinite(lw), np.exp(lw), 0.0)
        for v, (mm, ww) in zip(V, ((ml, wl), (ml, w0), (mc, wl))):
            den[v][j] += np.bincount(c["gal"], ww, ng)
            for k, col in enumerate((2, 1)):
                num[v][j, :, k] += np.bincount(c["gal"], ww * mm[:, col], ng)
print("copies done", flush=True)
resp = {}
for v in V:
    r = num[v] / den[v][..., None]
    resp["copies " + v] = (r[0, :, 0] / r[0, :, 1] - r[1, :, 0] / r[1, :, 1]) / (2 * H)
lens = lambda sg: m0 + sg * H * gal["dm_dg"][:, 0] + 0.5 * H * H * gal["d2m_dg2"][:, 0]
resp["parent"] = (e1(lens(1)) - e1(lens(-1))) / (2 * H)

# ---- flow: deterministic transport of each galaxy
mt = shear.load(D + "moments_bulgedisc_g2_bdg2n.fits")[0]
fl = eqx.tree_deserialise_leaves(sys.argv[1], bulk.build_flow(jr.key(0), mt, shear=True, centroid=True))
chart, cent, sh = fl.bijection.bijection.bijections[:3]
sxj = jnp.asarray(sx, jnp.float32)
cnd = lambda g: jnp.concatenate([jnp.array([g, 0.0], jnp.float32), sxj])


@eqx.filter_jit
def flow_m(m, gs, gc, use_cent):
    def one(mi):
        z0 = chart.transform(mi)
        zs = sh.inverse(sh.transform(z0, cnd(0.0)), cnd(gs))
        zc = cent.inverse(zs, cnd(gc)) if use_cent else zs
        return chart.inverse(zc)
    return jax.vmap(one)(m)


mj = jnp.asarray(m0, jnp.float32)
F = lambda gs, gc, uc: np.concatenate([np.asarray(flow_m(mj[i:i + 50000], gs, gc, uc), np.float64)
                                       for i in range(0, ng, 50000)])
for name, (gs, gc, uc) in {"flow shear": (1, 0, False), "flow total": (1, 1, True),
                           "flow implicit": (1, 0, True), "flow explicit": (0, 1, True)}.items():
    resp[name] = (e1(F(gs * H, gc * H, uc)) - e1(F(-gs * H, -gc * H, uc))) / (2 * H)
print("flow done\n", flush=True)

s = m0[:, 1] / m0[:, 0]
keys = list(resp)
print(f"size {SIZE}, H {H}: band-mean d e1/dg1 per galaxy (sem in last digit col)")
print(f"{'Mf band':>11s} {'N':>6s} " + " ".join(f"{k:>15s}" for k in keys) + "   total/copies full")
for lo, hi in BANDS:
    k = (m0[:, 0] >= lo) & (m0[:, 0] < hi) & (s > SIZE[0]) & (s < SIZE[1])
    k &= np.all([np.isfinite(resp[q]) for q in keys], 0)
    mean = {q: resp[q][k].mean() for q in keys}
    d = resp["flow total"][k] - resp["copies full"][k]
    print(f"{lo:>5d}-{hi:<5d} {k.sum():>6d} " + " ".join(f"{mean[q]:>15.4f}" for q in keys)
          + f"   {mean['flow total'] / mean['copies full']:.4f}  (diff {d.mean():+.4f}+/-{d.std() / np.sqrt(k.sum()):.4f})")
