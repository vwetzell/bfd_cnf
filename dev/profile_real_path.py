"""REAL-PATH timing (flow.log_prob g-Hessian, fp32) of anisotropic-capable designs, via monkeypatch:
  baseline | iso-detach (SigmaX sees detached condition, NO dipole: exact only at E=0) |
  design: SigmaX detached + polynomial g-dipole (translation of y3,y4, coefficients placeholders,
          Sigma_X features from the condition) applied right after ShearResponse."""
import sys, time; sys.path.insert(0, ".")
import equinox as eqx, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
from models.bijections import SigmaXBlockLayer as SX, cx_to_sx_cond
from models.shear import ShearResponse as SR
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
    jax.block_until_ready(out[-1]); return out
un0, tl0 = SX._unpack, SR.transform_and_log_det
def detach(self, c): return un0(self, jax.lax.stop_gradient(c))
def with_dipole(self, x, condition=None):
    y, ld = tl0(self, x, condition)
    g, CX = condition[:2], jnp.array([[condition[2], condition[3]], [condition[3], condition[4]]])
    _, e1_, e2_ = cx_to_sx_cond(jax.lax.stop_gradient(CX))
    ge = g[0] * e1_ + g[1] * e2_; gm = g[0] ** 2 + g[1] ** 2
    Dg = 0.01 + 0.3 * ge + 0.2 * gm + 0.1 * ge ** 2       # placeholder coefficients (timing only)
    return y.at[3].add(Dg * e1_).at[4].add(Dg * e2_), ld
for name, unp, tl in (("baseline", un0, tl0), ("iso-detach (no dipole)", detach, tl0),
                      ("detach + polynomial g-dipole", detach, with_dipole)):
    SX._unpack, SR.transform_and_log_det = unp, tl
    f = qr(); run(f, 256)
    mem = f.lower(mm[:256]).compile().memory_analysis().temp_size_in_bytes / 2**20 / 256
    line = f"{name:30s} temp {mem:6.3f} MiB/draw |"
    for bs in (256, 512, 1024):
        run(f, bs); t = time.time(); run(f, bs)
        line += f" b{bs} {(time.time()-t)/len(mm)*1e3:.4f}"
    print(line + " ms/draw", flush=True)
