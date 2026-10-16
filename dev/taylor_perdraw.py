"""Per-draw (value, grad, Hessian) of the peeled log p in g: full autodiff vs bias.log_conv_is_taylor.
Draws = template rows + target-sized Gaussian noise (so junk/edge points are included).  Diagnostic.
Usage: python dev/taylor_perdraw.py FLOW [N]"""
import sys; sys.path.insert(0, ".")
import fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
D = "../bfd_cnf_imsims/data"
cat = f"{D}/{B.CATALOGS['bulgedisc_g2n_1m_desv']['zero']}.fits"
m_train = shear.load(f"{D}/{B.TRAIN_DATA['bulgedisc_g2n_1m_desv']}")[0]
sx = jnp.asarray(np.asarray(fitsio.read(cat, columns=["cov_odd"], rows=[0])["cov_odd"])[0], jnp.float32)
import bfd
cov = np.asarray(bfd.MomentCovariance.bulkUnpack(fitsio.read(cat, columns=["cov"], rows=[0])["cov"]), np.float64)[0]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
flow = bulk.load_flow(sys.argv[1], m_train, key=jr.key(0), shear=True, centroid=True)
flow_g, layer = B.split_centroid(flow)
tl = B.taylor_layers(flow_g)
rng = np.random.default_rng(0)
C = cov
m = m_train[rng.choice(len(m_train), N)]
if C is not None:
    m = m + rng.multivariate_normal(np.zeros(5), C, N)
m = jnp.asarray(m, jnp.float32)
ok = jnp.asarray(B.in_domain(m) & B.in_support(m))
d, _ = B.centroid_transform(layer, jnp.where(ok[:, None], m, jnp.asarray(m_train[:1], jnp.float32)),
                            np.broadcast_to(np.asarray(sx), (N, 3)), to_host=False)
one_ok = jnp.ones(1, bool); z = jnp.zeros(1)
full = lambda x: B._val_grad_hess(lambda g: B.log_conv_is(flow_g, None, x[None], z, B.condition(g, sx), one_ok))
tay = lambda x: B._val_grad_hess(lambda g: B.log_conv_is_taylor(tl, flow_g.base_dist, None, x[None], z, g, sx, one_ok))
F = jax.jit(jax.vmap(full))(d); T = jax.jit(jax.vmap(tay))(d)
okn = np.asarray(ok)
for name, a, b in zip(("val", "grad", "hess"), F, T):
    a, b = np.asarray(a).reshape(N, -1), np.asarray(b).reshape(N, -1)
    fa, fb = np.isfinite(a).all(1), np.isfinite(b).all(1)
    rel = np.abs(a - b).max(1) / (np.abs(a).max(1) + 1e-6)
    sel = okn & fa & fb
    print(f"{name}: in-domain {okn.sum()}/{N}  nonfinite full {(okn & ~fa).sum()} taylor {(okn & ~fb).sum()}  "
          f"rel diff p50 {np.median(rel[sel]):.1e} p99.9 {np.percentile(rel[sel], 99.9):.1e} max {rel[sel].max():.1e}  n>1e-3 {(rel[sel] > 1e-3).sum()}")
    if name == "hess":
        bad = np.where(okn & (~fb | (rel > 1e-3)))[0][:10]
        x = np.asarray(m)
        for i in bad:
            print(f"   Mf {x[i,0]:9.1f} Mr/Mf {x[i,1]/x[i,0]:6.3f} Mc/Mr {x[i,4]/x[i,1]:6.3f} |e| {np.hypot(x[i,2],x[i,3])/x[i,1]:.3f}"
                  f"  z {np.asarray(d[i]).round(2)}  full {a[i].round(2)}  taylor {b[i].round(2)}")
