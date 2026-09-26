import sys, time
sys.path.insert(0, ".")
import equinox as eqx, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
D = "../bfd_cnf_imsims/data"
m_train = np.asarray(shear.load(f"{D}/{B.TRAIN_DATA['gauss2_v4n']}")[0])[:2000]
flow = eqx.tree_deserialise_leaves("flows/centroid_g2v4n.eqx",
    bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True))
s = jnp.asarray(np.asarray(fitsio.read(f"{D}/{B.CATALOGS['gauss2_v4n']['zero']}.fits")["cov_odd"])[0], jnp.float32)
zero = jnp.zeros(2); e0, e1 = jnp.array([1., 0.]), jnp.array([0., 1.])
def mk(f_of):
    def one(m):
        f = lambda g: f_of(m, g)
        (_, q), lin = jax.linearize(jax.value_and_grad(f), zero)
        return q, jnp.stack([lin(e0)[1], lin(e1)[1]], -1)
    return jax.jit(jax.vmap(one))
full = lambda m, g: flow.log_prob(m, condition=B.condition(g, s))
nold = lambda m, g: flow.base_dist.log_prob(flow.bijection.transform(m, B.condition(g, s)))
m_all = jnp.asarray(np.asarray(shear.load(f"{D}/{B.TRAIN_DATA['gauss2_v4n']}")[0])[:4096])
for name, f in (("full", full), ("no log-det", nold)):
    fn = mk(f)
    for n in (64, 256, 1024, 2048):
        m = m_all[:n]
        try:
            jax.block_until_ready(fn(m)); t = time.time()
            for _ in range(3): jax.block_until_ready(fn(m))
            print(f"{name:11s} n={n:5d}  {(time.time()-t)/3/n*1e3:7.3f} ms/draw", flush=True)
        except Exception as e:
            print(f"{name:11s} n={n:5d}  FAILED {type(e).__name__}", flush=True)
