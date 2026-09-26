"""Per-layer marginal cost of the flow's g-Hessian (fp32): time and compiled temp memory of
linearize(value_and_grad(f_k)) where f_k = sum of log-dets of layers[:k] + 0.5|x_k|^2.
Marginal cost of layer k = cost(prefix k+1) - cost(prefix k).   Diagnostic only."""
import sys, time; sys.path.insert(0, ".")
import equinox as eqx, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
D = "../bfd_cnf_imsims/data"
m_all = np.asarray(shear.load(f"{D}/{B.TRAIN_DATA['gauss2_v4n']}")[0])
flow = eqx.tree_deserialise_leaves("flows/centroid_g2v4n.eqx",
    bulk.build_flow(jr.key(0), m_all[:20000], shear=True, centroid=True))
sx = jnp.asarray(np.asarray(fitsio.read(f"{D}/{B.CATALOGS['gauss2_v4n']['zero']}.fits")["cov_odd"])[0], jnp.float32)
layers = list(flow.bijection.bijection.bijections)
N = 512
mm = jnp.asarray(m_all[np.random.default_rng(0).choice(len(m_all), N, replace=False)], jnp.float32)
z0, e0, e1 = jnp.zeros(2), jnp.array([1., 0.]), jnp.array([0., 1.])
def make(k):
    def f_of(m, g):
        c = B.condition(g, sx); x = m; tot = 0.0
        for l in layers[:k]:
            x, ld = l.transform_and_log_det(x, c if l.cond_shape is not None else None)
            tot = tot + ld
        return tot + 0.5 * jnp.sum(x ** 2)
    def one(m):
        f = lambda g: f_of(m, g)
        (_, q), lin = jax.linearize(jax.value_and_grad(f), z0)
        return q, jnp.stack([lin(e0)[1], lin(e1)[1]], -1)
    return jax.jit(jax.vmap(one))
prev_t = prev_mem = 0.0
print(f"{'k':>2} {'layer':28s} {'prefix ms/draw':>15} {'marg ms':>9} {'temp MiB/draw':>14} {'marg MiB':>9}", flush=True)
for k in range(1, len(layers) + 1):
    fn = make(k)
    mem = fn.lower(mm).compile().memory_analysis().temp_size_in_bytes / 2**20 / N
    jax.block_until_ready(fn(mm)); t = time.time()
    for _ in range(3): jax.block_until_ready(fn(mm))
    ms = (time.time() - t) / 3 / N * 1e3
    print(f"{k:2d} {type(layers[k-1]).__name__:28s} {ms:15.4f} {ms-prev_t:9.4f} {mem:14.3f} {mem-prev_mem:9.3f}", flush=True)
    prev_t, prev_mem = ms, mem
