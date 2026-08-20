"""How much of the ladder is seed noise?

Every "trend" read off the steps ladders this session came from ONE training run
per step count, and `shear.train` is documented as not reproducible run to run.
The window map made that impossible to ignore: Mc read 0.70 at 96k, 0.09 at
150k, 0.86 at 200k.  Either step count matters enormously and non-monotonically,
or the spread between independent runs at a FIXED step count is comparable to
the differences being interpreted.

Fixed steps, fixed data, fixed init; only the training seed varies.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
from models.shear import ShearResponse, dm_dg

D = "../bfd_cnf_imsims/data"
data = shear.load(f"{D}/moments.fits")
n_train = int(0.9 * len(data[0]))
train = tuple(x[:n_train] for x in data)
val = tuple(x[n_train:] for x in data)
m_e, q_e = jnp.asarray(val[0][:4000]), jnp.asarray(val[1][:4000])
bulk_only = eqx.tree_deserialise_leaves(
    "flows/ladder/bulk_60000.eqx", bulk.build_flow(jr.key(0), np.asarray(train[0])))

def alpha(flow):
    layer, chart = shear._shear_layer(flow), shear._chart(flow)
    q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(layer, m_e, chart)
    s = shear._scale(m_e)[:, None, :]
    p, t = np.asarray(q / s), np.asarray(q_e / s)
    return ((np.mean((p + t) ** 2, (0, 1)) - np.mean((p - t) ** 2, (0, 1)))
            / (4 * np.mean(t ** 2, (0, 1))))

STEPS = 96_000
print(f"{STEPS} steps, 5 seeds\n")
print(f"{'seed':>6}{'val nll':>10}" + "".join(f"{n:>8}" for n in
      ["Mf", "Mr", "M1", "M2", "Mc"]), flush=True)
rows = []
for seed in (1, 2, 3, 4, 5):
    full = bulk.build_flow(jr.key(0), np.asarray(train[0]), shear=True)
    keep = [i for i, b in enumerate(full.bijection.bijection.bijections)
            if not isinstance(b, ShearResponse)]
    full = eqx.tree_at(lambda f: [f.bijection.bijection.bijections[i] for i in keep],
                       full, list(bulk_only.bijection.bijection.bijections))
    out = shear.train(full, train, jr.key(seed), steps=STEPS, batch=1024)
    a = alpha(out); rows.append(a)
    v = float(shear.val_nll(out, val, jr.key(99)))
    print(f"{seed:>6}{v:>10.4f}" + "".join(f"{x:8.2f}" for x in a), flush=True)
r = np.array(rows)
print(f"\n{'mean':>6}{'':>10}" + "".join(f"{x:8.2f}" for x in r.mean(0)))
print(f"{'sd':>6}{'':>10}" + "".join(f"{x:8.2f}" for x in r.std(0)))
