"""ShearResponse log-det: autodiff slogdet(jacfwd) vs the exact g=0 Taylor `_log_det_g2`.
fp32, g-Hessian of flow.log_prob: Q,R agreement and ms/draw (b256/512/1024)."""
import sys, time; sys.path.insert(0, ".")
import equinox as eqx, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
from models.shear import ShearResponse as S
D = "../bfd_cnf_imsims/data"
m_all = np.asarray(shear.load(f"{D}/{B.TRAIN_DATA['gauss2_v4n']}")[0])
flow = eqx.tree_deserialise_leaves("flows/centroid_g2v4n.eqx",
    bulk.build_flow(jr.key(0), m_all[:20000], shear=True, centroid=True))
sx = jnp.asarray(np.asarray(fitsio.read(f"{D}/{B.CATALOGS['gauss2_v4n']['zero']}.fits")["cov_odd"])[0], jnp.float32)
mm = jnp.asarray(m_all[np.random.default_rng(0).choice(len(m_all), 2048, replace=False)], jnp.float32)
z0, e0, e1 = jnp.zeros(2), jnp.array([1., 0.]), jnp.array([0., 1.])
def qr():
    def one(m):
        f = lambda g: flow.log_prob(m, condition=B.condition(g, sx))
        (_, q), lin = jax.linearize(jax.value_and_grad(f), z0)
        return q, jnp.stack([lin(e0)[1], lin(e1)[1]], -1)
    return jax.jit(jax.vmap(one))
def run(fn, bs):
    out = [fn(mm[i:i + bs]) for i in range(0, len(mm), bs)]
    jax.block_until_ready(out[-1])
    return [np.concatenate([np.asarray(o[j], np.float64) for o in out]) for j in (0, 1)]
from models.bijections import SigmaXBlockLayer as L
import jax
orig_gs = L._g_shift
def peak(f):
    return f.lower(mm[:256]).compile().memory_analysis().temp_size_in_bytes / 2**20 / 256
for name, gs in [("_g_shift = 0*e (g-free, TIMING ONLY)", lambda self, g1, g2, e1, e2, ls, es, T: (0.0 * e1, 0.0 * e2))]:
    L._g_shift = gs
    f = qr(); run(f, 256)
    line = f"{name:30s} temp {peak(f):6.3f} MiB/draw |"
    for bs in (256, 512, 1024):
        run(f, bs); t = time.time(); run(f, bs)
        line += f" b{bs} {(time.time()-t)/len(mm)*1e3:.3f}"
    print(line + " ms/draw", flush=True)
