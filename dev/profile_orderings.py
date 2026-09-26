"""TIMING PROXY (existing weights, not physics): g-Hessian cost of three layer orderings.
  V0 baseline  : chart, SigmaX(real g), Shear, AR...
  VA           : chart, SigmaX(g detached), Shear, [g-coupled dipole translation], AR...
  VB           : chart, Shear, SigmaX(real g), AR...
fp32; compiled temp MiB/draw and ms/draw at b256/512/1024."""
import sys, time; sys.path.insert(0, ".")
import equinox as eqx, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
D = "../bfd_cnf_imsims/data"
m_all = np.asarray(shear.load(f"{D}/{B.TRAIN_DATA['gauss2_v4n']}")[0])
flow = eqx.tree_deserialise_leaves("flows/centroid_g2v4n.eqx",
    bulk.build_flow(jr.key(0), m_all[:20000], shear=True, centroid=True))
sx = jnp.asarray(np.asarray(fitsio.read(f"{D}/{B.CATALOGS['gauss2_v4n']['zero']}.fits")["cov_odd"])[0], jnp.float32)
L = list(flow.bijection.bijection.bijections)
chart, sigx, shr, rest = L[0], L[1], L[2], L[3:]
mm = jnp.asarray(m_all[np.random.default_rng(0).choice(len(m_all), 1024, replace=False)], jnp.float32)
z0, e0, e1 = jnp.zeros(2), jnp.array([1., 0.]), jnp.array([0., 1.])
def tl(l, x, c):
    return l.transform_and_log_det(x, c if l.cond_shape is not None else None)
def gdipole(x, g):
    """the g-coupled dipole translation of (y3,y4) as a stand-alone step (uses sigx's trained net)"""
    ls, e1_, e2_, _, ems, T, _, _ = sigx._unpack(B.condition(g, sx))
    d3, d4 = sigx._g_shift(g[0], g[1], e1_, e2_, ls, ems, T)
    return x.at[3].add(d3).at[4].add(d4)
def order(name):
    def f(m, g):
        c = B.condition(g, sx); cd = B.condition(jax.lax.stop_gradient(g), sx); tot = 0.0
        x, ld = tl(chart, m, None); tot += ld
        if name == "V0":
            x, ld = tl(sigx, x, c); tot += ld; x, ld = tl(shr, x, c); tot += ld
        elif name == "VA":
            x, ld = tl(sigx, x, cd); tot += ld; x, ld = tl(shr, x, c); tot += ld; x = gdipole(x, g)
        else:
            x, ld = tl(shr, x, c); tot += ld; x, ld = tl(sigx, x, c); tot += ld
        for l in rest:
            x, ld = tl(l, x, None); tot += ld
        return tot + 0.5 * jnp.sum(x ** 2)
    def one(m):
        fg = lambda g: f(m, g)
        (_, q), lin = jax.linearize(jax.value_and_grad(fg), z0)
        return q, jnp.stack([lin(e0)[1], lin(e1)[1]], -1)
    return jax.jit(jax.vmap(one))
for name in ("V0", "VA", "VB"):
    fn = order(name)
    mem = fn.lower(mm[:256]).compile().memory_analysis().temp_size_in_bytes / 2**20 / 256
    line = f"{name} temp {mem:6.3f} MiB/draw |"
    for bs in (256, 512, 1024):
        jax.block_until_ready(fn(mm[:bs])); t = time.time()
        for _ in range(3): jax.block_until_ready(fn(mm[:bs]))
        line += f" b{bs} {(time.time()-t)/3/bs*1e3:.4f}"
    print(line + " ms/draw", flush=True)
