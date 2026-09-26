"""Why is ShearResponse 24x heavier (memory) in the centroid flow?  g-Hessian cost of
sum(logdet)+0.5|x|^2 through ShearResponse ALONE, crossing: which flow's layer, which input
(chart(m) vs SigmaXBlock(chart(m))), which condition width.  fp32, N=512, compiled temp MiB."""
import sys, time; sys.path.insert(0, ".")
import equinox as eqx, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
D = "../bfd_cnf_imsims/data"
m_all = np.asarray(shear.load(f"{D}/{B.TRAIN_DATA['gauss2_v4n']}")[0])
def load(p, c): return eqx.tree_deserialise_leaves(p, bulk.build_flow(jr.key(0), m_all[:20000], shear=True, centroid=c))
fs, fc = load("flows/shear_g2v4n.eqx", False), load("flows/centroid_g2v4n.eqx", True)
ls, lc = list(fs.bijection.bijection.bijections), list(fc.bijection.bijection.bijections)
sx = jnp.asarray(np.asarray(fitsio.read(f"{D}/{B.CATALOGS['gauss2_v4n']['zero']}.fits")["cov_odd"])[0], jnp.float32)
N = 512
mm = jnp.asarray(m_all[np.random.default_rng(0).choice(len(m_all), N, replace=False)], jnp.float32)
z0, e0, e1 = jnp.zeros(2), jnp.array([1., 0.]), jnp.array([0., 1.])
chart, sigx = lc[0], lc[1]
def cell(name, layer, use_sigx, cond5):
    def one(m):
        def f(g):
            c = B.condition(g, sx) if cond5 else g
            x = chart.transform_and_log_det(m, None)[0]
            if use_sigx: x = sigx.transform_and_log_det(x, B.condition(g, sx))[0]
            y, ld = layer.transform_and_log_det(x, c)
            return ld + 0.5 * jnp.sum(y ** 2)
        (_, q), lin = jax.linearize(jax.value_and_grad(f), z0)
        return q, jnp.stack([lin(e0)[1], lin(e1)[1]], -1)
    fn = jax.jit(jax.vmap(one))
    mem = fn.lower(mm).compile().memory_analysis().temp_size_in_bytes / 2**20 / N
    jax.block_until_ready(fn(mm)); t = time.time()
    for _ in range(3): jax.block_until_ready(fn(mm))
    print(f"{name:52s} temp {mem:7.3f} MiB/draw  {(time.time()-t)/3/N*1e3:7.4f} ms/draw", flush=True)
print("cond_shape shear-flow layer:", ls[1].cond_shape, " centroid-flow layer:", lc[2].cond_shape)
# cell("shear-flow layer,    chart input, cond 2", ls[1], False, False)
# cell("centroid-flow layer, chart input, cond 5", lc[2], False, True)
# cell("centroid-flow layer, chart input, cond 2 (g only)", lc[2], False, False)
cell("shear-flow layer,    SigmaX input, cond 2", ls[1], True, False)
cell("centroid-flow layer, SigmaX input, cond 5", lc[2], True, True)
