"""Does widening the training shear range close the second-order log-det leak?

The runaway is entirely SECOND order in g (`dev/logdet_parity.py`: odd part is
~0 in every flow, even part goes 0.0016 -> 1.686).  Diagnosis: `p2 = |g|^2` is a
genuine spin-0 invariant so isotropy does not protect it, AND training at
|g| <= 0.02 means |g|^2 <= 4e-4, so the data barely constrains those
coefficients.  A nearly-free direction the likelihood spends on log-det.

If that is right, widening g_max should shrink the even part -- the g^2 terms
get real leverage -- with no architectural change at all.

Both `lens` and the layer are exact second-order Taylor models in g, so
training at larger g is self-consistent: we are fitting a second-order model to
second-order data.  Lensed templates stay on-chart out to g_max = 0.2 (checked).

Physical target: even part +0.00033, odd -0.00079 at g=0.02.
Log-det and alpha are always evaluated at g=0.02 so the columns are comparable.
Whitening ON throughout (kept per instruction).
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
m_e, q_e = jnp.asarray(val[0][:20000]), jnp.asarray(val[1][:20000])
bulk_only = eqx.tree_deserialise_leaves(
    "flows/ladder/bulk_60000.eqx", bulk.build_flow(jr.key(0), np.asarray(train[0])))

def build():
    f = bulk.build_flow(jr.key(0), np.asarray(train[0]), shear=True)
    keep = [i for i, b in enumerate(f.bijection.bijection.bijections)
            if not isinstance(b, ShearResponse)]
    return eqx.tree_at(lambda ff: [ff.bijection.bijection.bijections[i] for i in keep],
                       f, list(bulk_only.bijection.bijection.bijections))

def measure(f):
    lay, ch = shear._shear_layer(f), shear._chart(f)
    z = jax.vmap(ch.transform)(m_e)
    ldp = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([0.02, 0.]))[1])(z)
    ldm = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([-0.02, 0.]))[1])(z)
    q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(lay, m_e, ch)
    s = shear._scale(m_e)[:, None, :]
    p, t = np.asarray(q / s), np.asarray(q_e / s)
    a = ((np.mean((p + t) ** 2, (0, 1)) - np.mean((p - t) ** 2, (0, 1)))
         / (4 * np.mean(t ** 2, (0, 1))))
    return (float(jnp.mean(ldp - ldm) / 2), float(jnp.mean(ldp + ldm) / 2), a)

print(f"{'g_max':>7}{'seed':>5}{'val nll':>10}{'odd':>10}{'even':>10}"
      + "".join(f"{n:>7}" for n in ["Mf", "Mr", "M1", "M2", "Mc"]), flush=True)
print(f"{'physical':>7}{'':>5}{'':>10}{-0.00079:>10.5f}{0.00033:>10.5f}"
      + "".join(f"{1.0:>7.2f}" for _ in range(5)))
summ = {}
for gm in (0.02, 0.05, 0.10, 0.20):
    rows = []
    for seed in (1, 2, 3):
        f = shear.train(build(), train, jr.key(seed), steps=96_000,
                        batch=1024, g_max=gm)
        odd, even, a = measure(f)
        v = float(shear.val_nll(f, val, jr.key(99)))
        rows.append((odd, even, a))
        print(f"{gm:>7.2f}{seed:>5}{v:>10.4f}{odd:>10.5f}{even:>10.5f}"
              + "".join(f"{x:7.2f}" for x in a), flush=True)
    summ[gm] = rows
print(f"\n{'g_max':>7}{'even mean':>12}{'even sd':>10}"
      + "".join(f"{n:>8}" for n in ["Mf", "Mr", "M1", "M2", "Mc"]) + "   (mean)")
for gm, rows in summ.items():
    ev = np.array([r[1] for r in rows]); al = np.array([r[2] for r in rows])
    print(f"{gm:>7.2f}{ev.mean():>12.5f}{ev.std():>10.5f}"
          + "".join(f"{x:8.2f}" for x in al.mean(0)))
