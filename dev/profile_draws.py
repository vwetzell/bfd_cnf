"""Cost of bias.py's proposal-draw stage, piece by piece (fp32, GPU).  Diagnostic only.
One chunk at the x6rA config: T targets x S draws, alpha 0.5 (half kernel, half flow).
Times: mixture_draws as called, then its parts -- kernel RNG, flow.sample, flow.log_prob on all
draws, and the separate centroid_transform pqr_streamed applies afterwards; then per-layer cost
of the sampling (base -> data) and log_prob (data -> base) directions.
Usage: python dev/profile_draws.py FLOW [T S]"""
import sys, time; sys.path.insert(0, ".")
import bfd, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
jax.config.update("jax_default_matmul_precision", "highest")
D = "../bfd_cnf_imsims/data"
path = sys.argv[1]
T, S = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (16, 4096)
cat = fitsio.read(f"{D}/{B.CATALOGS['bulgedisc_g2n_1m_desv']['plus']}.fits", rows=np.arange(1000, 1000 + T),
                  columns=["moments", "cov", "cov_odd"])
m = np.asarray(cat["moments"][:, :5], np.float64)
cov = np.asarray(bfd.MomentCovariance.bulkUnpack(cat["cov"]), np.float64)[:, :5, :5]
sx = np.asarray(cat["cov_odd"], np.float32)
m_train = shear.load(f"{D}/{B.TRAIN_DATA['bulgedisc_g2n_1m_desv']}")[0]
flow = bulk.load_flow(path, m_train, key=jr.key(0), shear=True, centroid=True)
flow_g, peel = B.split_centroid(flow)
n_f = S // 2


def timed(fn, *a, reps=5):
    jax.block_until_ready(fn(*a)); t = time.perf_counter()
    for _ in range(reps): jax.block_until_ready(fn(*a))
    return (time.perf_counter() - t) / reps / T * 8192 / S * 1e3   # ms per target-arm (8192 draws)


md = lambda: B.mixture_draws(flow, m, cov, S, 0.5, 7, batch=T, sigma_x=sx, to_host=False)
t_mix = timed(md)
d, lw = md()
print(f"mixture_draws (as called): {t_mix:.3f} ms/target-arm", flush=True)
t_k = timed(lambda: B.kernel_draws(cov, T, S // 2, 7))
cond0 = jax.vmap(B.condition, in_axes=(None, 0))(jnp.zeros(2), jnp.asarray(sx))
samp = jax.jit(lambda k: flow.sample(k, (n_f,), condition=cond0))
t_s = timed(samp, jr.key(1))
condf = jnp.broadcast_to(cond0[:, None, :], d.shape[:-1] + (5,))
lp = jax.jit(lambda x: flow.log_prob(x, condition=condf))
t_lp = timed(lp, d)
t_ct = timed(lambda x: B.centroid_transform(peel, x, sx, to_host=False), d)
print(f"  kernel RNG {t_k:.3f} | flow.sample (half the draws) {t_s:.3f} | flow.log_prob (all draws) {t_lp:.3f} "
      f"| centroid_transform after (all draws) {t_ct:.3f}", flush=True)

# per-layer: sampling direction (base -> data = each layer's inverse, chain reversed)
bij = list(flow.bijection.bijection.bijections)
u = jax.vmap(lambda c: flow.base_dist.sample(jr.key(2), (n_f,)), in_axes=0)(cond0)       # (T, n_f, 5)
cs = jnp.broadcast_to(cond0[:, None, :], u.shape[:-1] + (5,))
fx = jax.vmap(jax.vmap(lambda v, c: flow.bijection.transform(v, c)))
x_s = fx(u, cs)
print("sampling direction (inverse, value only), ms per target-arm:", flush=True)
y = u
for l in reversed(bij):
    c = cs if l.cond_shape is not None else None
    f = jax.jit(jax.vmap(jax.vmap(lambda v, cc, l=l: l.inverse(v, cc))) if c is not None else
                jax.vmap(jax.vmap(lambda v, l=l: l.inverse(v, None))))
    t = timed(f, y, c) if c is not None else timed(f, y)
    print(f"   {type(l).__name__:32s} {t * 2:.3f}  (x2: per 8192 sampled)", flush=True)
    y = f(y, c) if c is not None else f(y)
print("log_prob direction (transform + log-det), ms per target-arm:", flush=True)
y = d
cf = condf
for l in bij:
    c = cf if l.cond_shape is not None else None
    f = jax.jit(jax.vmap(jax.vmap(lambda v, cc, l=l: l.transform_and_log_det(v, cc))) if c is not None else
                jax.vmap(jax.vmap(lambda v, l=l: l.transform_and_log_det(v, None))))
    t = timed(f, y, c) if c is not None else timed(f, y)
    print(f"   {type(l).__name__:32s} {t:.3f}", flush=True)
    y = (f(y, c) if c is not None else f(y))[0]
