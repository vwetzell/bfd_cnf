"""Do Mf and Mc converge with more training, or plateau?

Continues the ladder in `dev/gain_vs_steps.py`, which got as far as 96k steps:

    steps  batch     Mf    Mr    M1    M2    Mc
     1500   1024   0.54  0.32  0.99  0.98  0.18
     6000   1024   0.53  0.31  1.01  1.00  0.14
    24000   1024   1.60  1.00  1.00  1.00  0.33
    96000   1024   1.45  1.03  0.99  0.99  0.45

Spin-2 is pinned at 1.00 and Mr has converged, so the only live question is
whether Mf (overshot to ~1.5) settles back and whether Mc (creeping 0.14 ->
0.33 -> 0.45) is slow or stuck.

Each run gets its own cosine schedule over its own step count, so these are
independent trainings-to-convergence, not one trajectory sampled at intervals.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
from models.shear import dm_dg

D = "../bfd_cnf_imsims/data"
data = shear.load(f"{D}/moments.fits")
m_e, q_e = jnp.asarray(data[0][:4000]), jnp.asarray(data[1][:4000])

def alpha(flow):
    layer, chart = shear._shear_layer(flow), shear._chart(flow)
    q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(layer, m_e, chart)
    s = shear._scale(m_e)[:, None, :]
    p, t = np.asarray(q / s), np.asarray(q_e / s)
    return ((np.mean((p + t) ** 2, (0, 1)) - np.mean((p - t) ** 2, (0, 1)))
            / (4 * np.mean(t ** 2, (0, 1))))

bulk_only = eqx.tree_deserialise_leaves(
    "flows/ladder/bulk_60000.eqx", bulk.build_flow(jr.key(0), np.asarray(data[0])))

print(f"{'steps':>9} {'val nll':>9}" + "".join(f"{n:>8}" for n in
      ["Mf", "Mr", "M1", "M2", "Mc"]) + "   (1.0 = correct)", flush=True)
for steps in (192_000, 384_000, 768_000):
    full = bulk.build_flow(jr.key(0), np.asarray(data[0]), shear=True)
    keep = [i for i, b in enumerate(full.bijection.bijection.bijections)
            if not isinstance(b, type(shear._shear_layer(full)))]
    full = eqx.tree_at(lambda f: [f.bijection.bijection.bijections[i] for i in keep],
                       full, list(bulk_only.bijection.bijection.bijections))
    out = shear.train(full, data, jr.key(1), steps=steps, batch=1024)
    eqx.tree_serialise_leaves(f"flows/probe_fixed/long{steps}.eqx", out)
    v = float(shear.val_nll(out, data, jr.key(99)))
    print(f"{steps:>9} {v:>9.4f}" + "".join(f"{x:8.2f}" for x in alpha(out)),
          flush=True)
