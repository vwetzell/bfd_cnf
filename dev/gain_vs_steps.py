"""Is the response gain a function of BATCH, or just of how far the optimiser got?

`dev/batch_probe*.sh` found the layer learns the response in the right direction
with the wrong gain, and that the gain grows monotonically with batch:
alpha = 0.55 at batch 1024, 2.56 at 65536, passing through the correct 1.0 on
the way.  At fixed steps, a larger batch means a cleaner gradient and so more
progress down the NLL, so batch and steps should be interchangeable IF this is
just "amount of optimisation".

  alpha rises with steps too  -> not batch-specific; the NLL optimum itself has
                                 the wrong gain, and gradient noise at small
                                 batch was accidentally regularising it
  alpha flat in steps         -> something batch-specific, i.e. a bug

alpha is read off from  ||pred+truth||^2 - ||pred-truth||^2 = 4 alpha ||truth||^2,
which is exact for pred = alpha*truth + (noise uncorrelated with truth).
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
from models.shear import dm_dg

D = "../bfd_cnf_imsims/data"
data = shear.load(f"{D}/moments.fits")
train, val = data[:], data[:]
m_e, q_e = jnp.asarray(data[0][:4000]), jnp.asarray(data[1][:4000])

def alpha(flow):
    """PER MOMENT.  Pooling across all five is useless: spin-2 sits at 1.00
    always and dominates the average, hiding the spin-0 failure entirely."""
    layer, chart = shear._shear_layer(flow), shear._chart(flow)
    q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(layer, m_e, chart)
    s = shear._scale(m_e)[:, None, :]
    p, t = np.asarray(q / s), np.asarray(q_e / s)
    return ((np.mean((p + t) ** 2, (0, 1)) - np.mean((p - t) ** 2, (0, 1)))
            / (4 * np.mean(t ** 2, (0, 1))))

print(f"{'steps':>8} {'batch':>8}" + "".join(f"{n:>8}" for n in
      ["Mf", "Mr", "M1", "M2", "Mc"]) + "   (1.0 = correct gain)")
for steps, batch in [(1500, 1024), (6000, 1024), (24000, 1024), (96000, 1024),
                     (24000, 16384)]:
    flow = bulk.build_flow(jr.key(0), np.asarray(data[0]), shear=True)
    flow = eqx.tree_deserialise_leaves("flows/ladder/bulk_60000.eqx",
                                       bulk.build_flow(jr.key(0), np.asarray(data[0])))
    full = bulk.build_flow(jr.key(0), np.asarray(data[0]), shear=True)
    keep = [i for i, b in enumerate(full.bijection.bijection.bijections)
            if not isinstance(b, type(shear._shear_layer(full)))]
    full = eqx.tree_at(lambda f: [f.bijection.bijection.bijections[i] for i in keep],
                       full, list(flow.bijection.bijection.bijections))
    out = shear.train(full, train, jr.key(1), steps=steps, batch=batch)
    eqx.tree_serialise_leaves(f"flows/probe_fixed/steps{steps}_b{batch}.eqx", out)
    print(f"{steps:>8} {batch:>8}" + "".join(f"{v:8.2f}" for v in alpha(out)),
          flush=True)
