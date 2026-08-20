"""The steps ladder, re-run against the non-saturating coefficient bound.

Before (tanh bound, gradient dies exponentially once saturated):

    steps     Mf    Mr    M1    M2    Mc   log-det@g=.02  median attenuation
     6000   0.53  0.31  1.01  1.00  0.14      0.0019          0.996
    24000   1.60  1.00  1.00  1.00  0.33      0.0108          0.958
    96000   1.45  1.03  0.99  0.99  0.45      0.0240          0.837
   400000    --    --    --    --    --       ~2.4            3.2e-4

The NLL pins the spin-0 scale to sigma ~0.03 with gradient SNR ~36 at the point
training converged to (`dev/spin0_profile.py`, `dev/spin0_gradient_snr.py`), so
the signal was always there.  If gradient death was the cause, the spin-0 gains
should now climb toward 1.0 with steps instead of stalling near 0.4, and the
attenuation should stay healthy.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
from models.shear import ShearResponse, dm_dg, _invariants, _COEFF_MAX

D = "../bfd_cnf_imsims/data"
data = shear.load(f"{D}/moments.fits")
n_train = int(0.9 * len(data[0]))
train = tuple(x[:n_train] for x in data)
val = tuple(x[n_train:] for x in data)
m_e, q_e = jnp.asarray(val[0][:4000]), jnp.asarray(val[1][:4000])

bulk_only = eqx.tree_deserialise_leaves(
    "flows/ladder/bulk_60000.eqx", bulk.build_flow(jr.key(0), np.asarray(train[0])))

def stats(flow):
    layer, chart = shear._shear_layer(flow), shear._chart(flow)
    q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(layer, m_e, chart)
    s = shear._scale(m_e)[:, None, :]
    p, t = np.asarray(q / s), np.asarray(q_e / s)
    a = ((np.mean((p + t) ** 2, (0, 1)) - np.mean((p - t) ** 2, (0, 1)))
         / (4 * np.mean(t ** 2, (0, 1))))
    z = jax.vmap(chart.transform)(m_e)
    _, ld = jax.vmap(lambda zi: layer.transform_and_log_det(
        zi, jnp.array([0.02, 0.0])))(z)
    raw = jax.vmap(lambda zi: layer.coeffs.net(jnp.stack(
        [_invariants(zi)[0], _invariants(zi)[1], _invariants(zi)[2],
         (_invariants(zi)[3] - 2.0) / 2.0])))(z)
    att = np.asarray((1.0 + (np.asarray(raw) / _COEFF_MAX) ** 2) ** -1.5)
    return a, float(jnp.mean(ld)), np.median(att), float((att < 1e-2).mean())

print(f"{'steps':>8}{'val nll':>10}{'log-det':>11}{'med att':>10}{'dead':>7}"
      + "".join(f"{n:>7}" for n in ["Mf", "Mr", "M1", "M2", "Mc"]), flush=True)
for steps in (150_000, 200_000, 250_000, 300_000):
    full = bulk.build_flow(jr.key(0), np.asarray(train[0]), shear=True)
    keep = [i for i, b in enumerate(full.bijection.bijection.bijections)
            if not isinstance(b, ShearResponse)]
    full = eqx.tree_at(lambda f: [f.bijection.bijection.bijections[i] for i in keep],
                       full, list(bulk_only.bijection.bijection.bijections))
    out = shear.train(full, train, jr.key(1), steps=steps, batch=1024)
    eqx.tree_serialise_leaves(f"flows/probe_fixed/rational{steps}.eqx", out)
    a, ld, med, dead = stats(out)
    v = float(shear.val_nll(out, val, jr.key(99)))
    print(f"{steps:>8}{v:>10.4f}{ld:>11.4f}{med:>10.3f}{dead:>7.3f}"
          + "".join(f"{x:7.2f}" for x in a), flush=True)
