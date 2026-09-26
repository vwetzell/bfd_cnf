"""Which SigmaXBlockLayer methods does flow.log_prob actually call?"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
from models import bijections as bj
D = "../bfd_cnf_imsims/data"
m = np.asarray(shear.load(f"{D}/{B.TRAIN_DATA['gauss2_v4n']}")[0])[:64]
flow = eqx.tree_deserialise_leaves("flows/centroid_g2v4n.eqx",
    bulk.build_flow(jr.key(0), m, shear=True, centroid=True))
calls = []
for name in ("_fwd_log_det", "inverse_and_log_det", "transform_and_log_det"):
    orig = getattr(bj.SigmaXBlockLayer, name)
    def mk(orig, name):
        def w(self, *a, **k):
            calls.append(name); return orig(self, *a, **k)
        return w
    setattr(bj.SigmaXBlockLayer, name, mk(orig, name))
sx = jnp.array([1.2e4, 0.0, 1.2e4], jnp.float32)
flow.log_prob(jnp.asarray(m[0]), condition=B.condition(jnp.zeros(2), sx))
print("calls:", calls)
print("SigmaX layers:", [type(l).__name__ for l in jax.tree_util.tree_leaves(
    flow.bijection, is_leaf=lambda n: isinstance(n, bj.SigmaXBlockLayer))
    if isinstance(l, bj.SigmaXBlockLayer)])
print("bijection type:", type(flow.bijection).__name__)

# direct swap test
ms = jnp.asarray(m[:8]); cond = B.condition(jnp.zeros(2), sx)
ref = jax.vmap(lambda x: flow.log_prob(x, condition=cond))(ms)
hit = []
bj.SigmaXBlockLayer._fwd_log_det = lambda self, x, c: (hit.append(1), self._log_det_analytic(x, c))[1]
ana = jax.vmap(lambda x: flow.log_prob(x, condition=cond))(ms)
print("analytic hook hit:", len(hit), " max|logp diff|:", float(jnp.abs(ref - ana).max()))
