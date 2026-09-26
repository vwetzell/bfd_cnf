"""Acceptance test: SigmaXBlockLayer._log_det_analytic vs the jacfwd+slogdet
log-det -- value, and the g-gradient/Hessian that feed Q and R.

    python dev/check_analytic_logdet.py --flow flows/centroid_g2v4n.eqx [--x64]
"""
import argparse, sys
import equinox as eqx, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
sys.path.insert(0, ".")
import bias as B, bulk, shear
from models.bijections import SigmaXBlockLayer

p = argparse.ArgumentParser()
p.add_argument("--flow", default="flows/centroid_g2v4n.eqx")
p.add_argument("--pop", default="gauss2_v4n")
p.add_argument("--n", type=int, default=20000)
p.add_argument("--x64", action="store_true")
a = p.parse_args()
D = "../bfd_cnf_imsims/data"
m_train = np.asarray(shear.load(f"{D}/{B.TRAIN_DATA[a.pop]}")[0])
flow = eqx.tree_deserialise_leaves(a.flow, bulk.build_flow(
    jr.key(0), m_train, shear=True, centroid=True))
dt = jnp.float32
if a.x64:
    jax.config.update("jax_enable_x64", True); dt = jnp.float64
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(dt) if eqx.is_inexact_array(x) else x, flow)
layers = [l for l in jax.tree_util.tree_leaves(
    flow.bijection, is_leaf=lambda n: isinstance(n, SigmaXBlockLayer))
    if isinstance(l, SigmaXBlockLayer)]
print(f"{len(layers)} SigmaXBlockLayer(s); dtype {dt.__name__}")
sx = jnp.asarray(np.asarray(fitsio.read(f"{D}/{B.CATALOGS[a.pop]['zero']}.fits")["cov_odd"])[0], dt)
rng = np.random.default_rng(0)
mt = m_train[rng.choice(len(m_train), a.n, replace=False)]
chart = bulk.chart_of(flow)
xs = jax.vmap(lambda m: chart.transform_and_log_det(m, None)[0])(jnp.asarray(mt, dt))
xs = xs[jnp.isfinite(xs).all(-1)]
# also stress: widen the cloud (edge/off-support points)
xs = jnp.concatenate([xs, xs * 1.5 + 0.3 * jnp.asarray(rng.standard_normal(xs.shape), dt)])
zero = jnp.zeros(2, dt)
for k, L in enumerate(layers):
    ref = lambda x, g: L.transform_and_log_det(x, B.condition(g, sx))[1]
    ana = lambda x, g: L._log_det_analytic(x, B.condition(g, sx))
    def chunked(op):   # the reference Hessian is the memory hog this test exists to avoid
        def run(f):
            v = jax.jit(jax.vmap(lambda x: op(f, x)))
            return jnp.concatenate([v(xs[i:i + 1024]) for i in range(0, len(xs), 1024)])
        return run
    for name, fn in (("value", chunked(lambda f, x: f(x, zero))),
                     ("d/dg", chunked(lambda f, x: jax.grad(f, 1)(x, zero))),
                     ("d2/dg2", chunked(lambda f, x: jax.hessian(f, 1)(x, zero))),
                     # the g-dependence that matters enters through x (upstream
                     # shear layers), so check the x-derivatives too
                     ("d/dx", chunked(lambda f, x: jax.grad(f, 0)(x, zero))),
                     ("d2/dx2", chunked(lambda f, x: jax.hessian(f, 0)(x, zero)))):
        r, q = np.asarray(fn(ref), np.float64), np.asarray(fn(ana), np.float64)
        ok = np.isfinite(r).reshape(len(r), -1).all(1) & np.isfinite(q).reshape(len(q), -1).all(1)
        err = np.abs(r[ok] - q[ok]); scale = np.abs(r[ok]).max() + 1e-30
        print(f"layer {k} {name:7s} n={ok.sum()}/{len(r)}  max|err| {err.max():.3e}  "
              f"median {np.median(err):.3e}  (max|ref| {scale:.3g})")

# End to end: full-flow Q, R with the layer's log-det swapped for the analytic one.
import time
def qr_fn():
    z0, e0, e1 = jnp.zeros(2, dt), jnp.array([1., 0.], dt), jnp.array([0., 1.], dt)
    def one(m):
        f = lambda g: flow.log_prob(m, condition=B.condition(g, sx))
        (_, q), lin = jax.linearize(jax.value_and_grad(f), z0)
        return q, jnp.stack([lin(e0)[1], lin(e1)[1]], -1)
    return jax.jit(jax.vmap(one))
def run(fn, m, bs):
    out = [fn(m[i:i + bs]) for i in range(0, len(m), bs)]
    jax.block_until_ready(out[-1]); return [np.concatenate([np.asarray(o[j], np.float64) for o in out]) for j in (0, 1)]
mm = jnp.asarray(mt[:2048], dt)
q_ref, r_ref = run(qr_fn(), mm, 256)
orig = SigmaXBlockLayer._fwd_log_det   # flow.log_prob runs inverse_and_log_det, which calls this
SigmaXBlockLayer._fwd_log_det = lambda self, x, condition: self._log_det_analytic(x, condition)
fn = qr_fn(); q_an, r_an = run(fn, mm, 256)
for nm, a_, b_ in (("Q", q_ref, q_an), ("R", r_ref, r_an)):
    e = np.abs(a_ - b_); print(f"end-to-end {nm}: max|err| {e.max():.3e}  max|ref| {np.abs(a_).max():.3e}  max rel {(e/(np.abs(a_)+1e-12)).max():.2e}")
for bs in (256, 512, 1024):
    try:
        run(fn, mm, bs); t = time.time(); run(fn, mm, bs)
        print(f"analytic  batch {bs:5d}: {(time.time()-t)/len(mm)*1e3:.4f} ms/draw", flush=True)
    except Exception as ex:
        print(f"analytic  batch {bs:5d}: FAILED {type(ex).__name__}")
SigmaXBlockLayer._fwd_log_det = orig
fn0 = qr_fn()
for bs in (256, 512, 1024):
    try:
        run(fn0, mm, bs); t = time.time(); run(fn0, mm, bs)
        print(f"autodiff  batch {bs:5d}: {(time.time()-t)/len(mm)*1e3:.4f} ms/draw", flush=True)
    except Exception as ex:
        print(f"autodiff  batch {bs:5d}: FAILED {type(ex).__name__}")
