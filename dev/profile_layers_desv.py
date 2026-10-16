"""Per-layer cost of bias.py's g-Hessian on a desv centroid flow (fp32, GPU).  Diagnostic only.
Mirrors the peeled path: draws go through the chart + centroid layer outside the autodiff, then
linearize(value_and_grad) over the rest of the chain, layer prefix by prefix, so
marginal cost of layer k = cost(prefix k+1) - cost(prefix k).  Also times value-only and value+grad
of the full chain, so the Hessian's multiplier is visible.
Usage: python dev/profile_layers_desv.py FLOW [FLOW ...]"""
import sys, time; sys.path.insert(0, ".")
import fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
jax.config.update("jax_default_matmul_precision", "highest")
D = "../bfd_cnf_imsims/data"
m_train = shear.load(f"{D}/{B.TRAIN_DATA['bulgedisc_g2n_1m_desv']}")[0]
sx = jnp.asarray(np.asarray(fitsio.read(f"{D}/{B.CATALOGS['bulgedisc_g2n_1m_desv']['zero']}.fits",
                                        columns=["cov_odd"], rows=[0])["cov_odd"])[0], jnp.float32)
N, S = 32768, 8192          # draws per call (= --batch-budget 32768), draws per target-arm
rows = np.random.default_rng(0).choice(len(m_train), N, replace=False)
z0, e0, e1 = jnp.zeros(2), jnp.array([1., 0.]), jnp.array([0., 1.])


def timed(fn, x):
    jax.block_until_ready(fn(x)); t = time.perf_counter()
    for _ in range(5): jax.block_until_ready(fn(x))
    return (time.perf_counter() - t) / 5 / N * S * 1e3      # ms per target-arm


for path in sys.argv[1:]:
    flow = bulk.load_flow(path, m_train, key=jr.key(0), shear=True, centroid=True)
    bij = list(flow.bijection.bijection.bijections)
    chart, cent, layers = bij[0], bij[1], bij[2:]
    cond = lambda g: B.condition(g, sx)
    m = jnp.asarray(m_train[rows], jnp.float32)
    x = jax.vmap(lambda v: chart.transform_and_log_det(v, None)[0])(m)
    x = jax.vmap(lambda v: cent.transform_and_log_det(v, cond(z0))[0])(x)

    def f_of(k):
        def f(v, g):
            c, tot = cond(g), 0.0
            for l in layers[:k]:
                v, ld = l.transform_and_log_det(v, c if l.cond_shape is not None else None)
                tot = tot + ld
            return tot - 0.5 * jnp.sum(v ** 2)
        return f

    def hess(k):
        f = f_of(k)
        def one(v):
            (_, q), lin = jax.linearize(jax.value_and_grad(lambda g: f(v, g)), z0)
            return q, jnp.stack([lin(e0)[1], lin(e1)[1]], -1)
        return jax.jit(jax.vmap(one))

    print(f"\n== {path}  ({len(layers)} layers after the peel; ms per target-arm = {S} draws)", flush=True)
    fk = f_of(len(layers))
    t_val = timed(jax.jit(jax.vmap(lambda v: fk(v, z0))), x)
    t_grad = timed(jax.jit(jax.vmap(lambda v: jax.value_and_grad(lambda g: fk(v, g))(z0))), x)
    print(f"full chain: value {t_val:.3f} ms, value+grad {t_grad:.3f} ms", flush=True)
    print(f"{'k':>2} {'layer':32s} {'prefix ms':>10} {'marg ms':>8} {'temp MiB/draw':>14}", flush=True)
    prev = 0.0
    for k in range(1, len(layers) + 1):
        fn = hess(k)
        mem = fn.lower(x).compile().memory_analysis().temp_size_in_bytes / 2**20 / N
        ms = timed(fn, x)
        print(f"{k:2d} {type(layers[k-1]).__name__:32s} {ms:10.3f} {ms - prev:8.3f} {mem:14.4f}", flush=True)
        prev = ms
    print(f"Hessian / value = {prev / t_val:.1f}x", flush=True)
