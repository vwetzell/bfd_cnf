"""The three fixes stacked, against the old bound, paired by seed.

  * _COEFF_MAX 12 -> 100   (12 was stale from the raw-moment era and destroyed
                            31% of the true Mr/Mc first-order response)
  * g_max 0.02 -> 0.10     (closes the second-order log-det leak)
  * whitened coefficient-net inputs (condition number 2625 -> 1.17)

Paired: identical seeds, data, init and steps; the arms differ only in the
bound.  Both use g_max=0.10 and whitening, so this isolates the recalibration.

Noise floor from `dev/seed_spread.py` at fixed steps: Mf sd 0.54, Mr 0.40,
Mc 0.28 -- differences smaller than that mean nothing, which is why this runs
seeds rather than single points.

Physical targets: alpha = 1.00 all five; log-det odd -0.00079, even +0.00033.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import models.shear as ms

BOUND = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
ms._COEFF_MAX = BOUND                       # read at call time; set before build
import bulk, shear
from models.shear import ShearResponse, dm_dg

D = "../bfd_cnf_imsims/data"
data = shear.load(f"{D}/moments.fits")
n_train = int(0.9 * len(data[0]))
train = tuple(x[:n_train] for x in data); val = tuple(x[n_train:] for x in data)
m_e, q_e = jnp.asarray(val[0][:20000]), jnp.asarray(val[1][:20000])
bulk_only = eqx.tree_deserialise_leaves(
    "flows/ladder/bulk_60000.eqx", bulk.build_flow(jr.key(0), np.asarray(train[0])))

def build():
    f = bulk.build_flow(jr.key(0), np.asarray(train[0]), shear=True)
    keep = [i for i, b in enumerate(f.bijection.bijection.bijections)
            if not isinstance(b, ShearResponse)]
    return eqx.tree_at(lambda ff: [ff.bijection.bijection.bijections[i] for i in keep],
                       f, list(bulk_only.bijection.bijection.bijections))

print(f"_COEFF_MAX = {ms._COEFF_MAX}, g_max = 0.10, whitening ON", flush=True)
print(f"{'seed':>5}{'val nll':>10}{'odd':>10}{'even':>10}"
      + "".join(f"{n:>8}" for n in ["Mf", "Mr", "M1", "M2", "Mc"]), flush=True)
rows = []
for seed in (1, 2, 3, 4, 5):
    f = shear.train(build(), train, jr.key(seed), steps=96_000, batch=1024, g_max=0.10)
    lay, ch = shear._shear_layer(f), shear._chart(f)
    z = jax.vmap(ch.transform)(m_e)
    p = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([0.02, 0.]))[1])(z)
    mm = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([-0.02, 0.]))[1])(z)
    q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(lay, m_e, ch)
    s = shear._scale(m_e)[:, None, :]
    pp, tt = np.asarray(q / s), np.asarray(q_e / s)
    a = ((np.mean((pp + tt) ** 2, (0, 1)) - np.mean((pp - tt) ** 2, (0, 1)))
         / (4 * np.mean(tt ** 2, (0, 1))))
    v = float(shear.val_nll(f, val, jr.key(99)))
    rows.append(a)
    print(f"{seed:>5}{v:>10.4f}{float(jnp.mean(p-mm)/2):>10.5f}"
          f"{float(jnp.mean(p+mm)/2):>10.5f}" + "".join(f"{x:8.2f}" for x in a), flush=True)
r = np.array(rows)
print(f"\n{'mean':>25}" + "".join(f"{x:8.2f}" for x in r.mean(0)))
print(f"{'SD':>25}" + "".join(f"{x:8.2f}" for x in r.std(0)))
