"""Does the shear layer run away when the bulk is EXACTLY right?

The runaway (log-det at g=0.02 going 0.024 -> 2.4 nats past ~96k steps, with
held-out NLL still improving) has a candidate explanation: the bulk is frozen
and slightly wrong, and the layer spends its g-freedom partly correcting the
bulk's density error.  That correction generalises, which is why held-out NLL
improves while the physical response degrades.

gauss2 cannot test this -- its analytic P0 is in a stale chart (sim.py slot 2
is a bare Mc/Mr, bulk.to_coords uses logit) AND it is 30% accepted, so the
catalog is a truncated Gaussian either way.

So generate the templates FROM a known flow instead:

  * templates m ~ G's own bulk, so the frozen bulk is EXACTLY the data density
  * responses q, r = G's own shear layer, so the target IS representable by an
    identically-shaped fresh layer

Every source of model mismatch is removed by construction.  If the runaway
persists here it is intrinsic to the objective; if it vanishes, the cure is a
better bulk rather than a constraint on the layer.

`lens` applies the 2nd-order Taylor of G's map rather than the map itself; at
|g| <= 0.02 that differs by O(g^3) ~ 1e-5 relative, negligible against a
runaway measured in nats.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
from models.shear import ShearResponse, dm_dg

D = "../bfd_cnf_imsims/data"
N = 100_000
real = shear.load(f"{D}/moments.fits")[0]

G = bulk.build_flow(jr.key(0), np.asarray(real), shear=True)
G = eqx.tree_deserialise_leaves("flows/shear.eqx", G)
g0 = jnp.zeros(2)

# 1. templates drawn from G's own bulk (shear layer is identity at g=0)
m_syn = G.sample(jr.key(7), (N,), condition=g0)
# 2. their responses under G's own layer
layerG, chart = shear._shear_layer(G), shear._chart(G)
qs, rs = [], []
for i in range(0, N, 20000):
    q, r = jax.vmap(dm_dg, in_axes=(None, 0, None))(layerG, m_syn[i:i+20000], chart)
    qs.append(q); rs.append(r)
data = (np.asarray(m_syn), np.asarray(jnp.concatenate(qs)),
        np.asarray(jnp.concatenate(rs)))
print(f"synthetic templates: {data[0].shape}", flush=True)

m_e = m_syn[:4000]
q_true = jnp.concatenate(qs)[:4000]

def report(tag, flow):
    layer = shear._shear_layer(flow)
    q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(layer, m_e, chart)
    s = shear._scale(m_e)[:, None, :]
    p, t = np.asarray(q / s), np.asarray(q_true / s)
    a = ((np.mean((p + t) ** 2, (0, 1)) - np.mean((p - t) ** 2, (0, 1)))
         / (4 * np.mean(t ** 2, (0, 1))))
    _, ld = jax.vmap(lambda mi: layer.transform_and_log_det(
        chart.transform(mi), jnp.array([0.02, 0.0])))(m_e)
    print(f"{tag:>9}{float(jnp.mean(ld)):14.4f}" + "".join(f"{v:8.2f}" for v in a),
          flush=True)

print(f"\n{'steps':>9}{'log-det':>14}" + "".join(f"{n:>8}" for n in
      ["Mf", "Mr", "M1", "M2", "Mc"]) + "   (alpha vs GENERATOR, 1.0 = recovered)")
for steps in (6_000, 24_000, 96_000, 400_000):
    fresh = bulk.build_flow(jr.key(0), np.asarray(real), shear=True)
    keep = [i for i, b in enumerate(fresh.bijection.bijection.bijections)
            if not isinstance(b, ShearResponse)]
    src = [b for b in G.bijection.bijection.bijections
           if not isinstance(b, ShearResponse)]
    fresh = eqx.tree_at(lambda f: [f.bijection.bijection.bijections[i] for i in keep],
                        fresh, src)
    out = shear.train(fresh, data, jr.key(1), steps=steps, batch=1024)
    eqx.tree_serialise_leaves(f"flows/probe_fixed/selfcons{steps}.eqx", out)
    report(str(steps), out)
