"""Per-draw cost of the flow's (Q, R) Hessian call: shear vs centroid stage,
float32 vs float64, plus peak GPU memory per draw.  Diagnostic only."""
import sys, time
sys.path.insert(0, ".")
import equinox as eqx, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
D = "../bfd_cnf_imsims/data"
m_train = np.asarray(shear.load(f"{D}/{B.TRAIN_DATA['gauss2_v4n']}")[0])[:20000]
sx = np.asarray(fitsio.read(f"{D}/{B.CATALOGS['gauss2_v4n']['zero']}.fits")["cov_odd"])[0]

def load(path, centroid):
    return eqx.tree_deserialise_leaves(path, bulk.build_flow(
        jr.key(0), m_train, shear=True, centroid=centroid))

def make(flow, dt, centroid):
    s = jnp.asarray(sx, dtype=dt) if centroid else None
    zero = jnp.zeros(2, dt); e0, e1 = jnp.array([1., 0.], dt), jnp.array([0., 1.], dt)
    def one(m):
        f = lambda g: flow.log_prob(m, condition=B.condition(g, s))
        (_, q), lin = jax.linearize(jax.value_and_grad(f), zero)
        return q, jnp.stack([lin(e0)[1], lin(e1)[1]], -1)
    return eqx.filter_jit(jax.vmap(one))

def bench(name, fn, dt, n):
    m = jnp.asarray(m_train[:n], dtype=dt)
    jax.block_until_ready(fn(m))                      # compile
    dev = jax.devices()[0]
    t = time.time()
    for _ in range(3): jax.block_until_ready(fn(m))
    ms = (time.time() - t) / 3 / n * 1e3
    peak = dev.memory_stats().get("peak_bytes_in_use", 0) / 2**20
    print(f"{name:28s} n={n:5d}  {ms:8.3f} ms/draw   peak {peak:8.0f} MiB", flush=True)

sh32, ce32 = load("flows/shear_g2v4n.eqx", False), load("flows/centroid_g2v4n.eqx", True)
bench("shear    fp32", make(sh32, jnp.float32, False), jnp.float32, 512)
bench("centroid fp32", make(ce32, jnp.float32, True), jnp.float32, 256)
jax.config.update("jax_enable_x64", True)
up = lambda f: jax.tree_util.tree_map(
    lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, f)
bench("shear    fp64", make(up(sh32), jnp.float64, False), jnp.float64, 512)
bench("centroid fp64", make(up(ce32), jnp.float64, True), jnp.float64, 256)
