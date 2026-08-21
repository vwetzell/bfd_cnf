"""The three spin-0 parameterisations, paired inside ONE process.

`_chart_spin0_jac` returns M = D L, and each arm is one monkeypatch of it:

    I   M = identity      the pre-16ece5c-era parameterisation: the network
                          emits chart-space coefficients directly.  This is
                          EXACTLY the old code path, reproduced without
                          checking out the old code.
    D   M = diag(D)       the divergence divided out, the differences left in.
                          The network emits d log X/dg DIFFERENCES.
    DL  M = D L           the shipped one: the network emits the raw c_X.

Why all three here rather than against numbers already in the logs: XLA picks
matmul precision per process, and on this flow that is worth ~1 nat.  Two
processes running the SAME code at the SAME seed landed at even log-dets of
0.019 and 1.430 -- different basins, not scatter -- which is why `shear.py` and
`bulk.py` now pin `jax_default_matmul_precision` and why every number in
`dev/reparam_paired.py` and `dev/gmax_sweep.py` is unusable as a baseline.
Anything compared across processes before that pin is void.

3 seeds x 3 arms x 96k steps, ~3 h.  Physical targets: odd -0.00079, even
+0.00033, alpha 1.00 on every moment.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
import models.shear as ms
from models.shear import ShearResponse, dm_dg

D_ = "../bfd_cnf_imsims/data"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 96_000
G_MAX = 0.02

data = shear.load(f"{D_}/moments.fits")
n_train = int(0.9 * len(data[0]))
train = tuple(x[:n_train] for x in data)
val = tuple(x[n_train:] for x in data)
m_e, q_e = jnp.asarray(val[0][:20000]), jnp.asarray(val[1][:20000])
bulk_only = eqx.tree_deserialise_leaves(
    "flows/ladder/bulk_60000.eqx", bulk.build_flow(jr.key(0), np.asarray(train[0])))

_FULL = ms._chart_spin0_jac
ARMS = {
    "I":  lambda z, loc, scale: jnp.eye(3),
    "D":  lambda z, loc, scale: jnp.diag(jnp.diag(_FULL(z, loc, scale))),
    "DL": _FULL,
}


def build():
    f = bulk.build_flow(jr.key(0), np.asarray(train[0]), shear=True)
    keep = [i for i, b in enumerate(f.bijection.bijection.bijections)
            if not isinstance(b, ShearResponse)]
    return eqx.tree_at(lambda ff: [ff.bijection.bijection.bijections[i] for i in keep],
                       f, list(bulk_only.bijection.bijection.bijections))


def measure(f):
    lay, ch = shear._shear_layer(f), shear._chart(f)
    z = jax.vmap(ch.transform)(m_e)
    ldp = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([G_MAX, 0.]))[1])(z)
    ldm = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([-G_MAX, 0.]))[1])(z)
    q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(lay, m_e, ch)
    s = shear._scale(m_e)[:, None, :]
    p, t = np.asarray(q / s), np.asarray(q_e / s)
    a = ((np.mean((p + t) ** 2, (0, 1)) - np.mean((p - t) ** 2, (0, 1)))
         / (4 * np.mean(t ** 2, (0, 1))))
    cf = np.asarray(jax.vmap(lambda zi: lay.coeffs(zi[0], zi[1], zi[2],
                                                   zi[3]**2 + zi[4]**2))(z))
    return (float(jnp.mean(ldp - ldm) / 2), float(jnp.mean(ldp + ldm) / 2), a,
            100 * np.mean(np.abs(cf[:, [0, 3, 6]]) > 11.88, axis=0))


print(f"{'arm':>4}{'seed':>5}{'val nll':>10}{'odd':>10}{'even':>10}"
      + "".join(f"{n:>7}" for n in ["Mf", "Mr", "M1", "M2", "Mc"]), flush=True)
print(f"{'phys':>4}{'':>5}{'':>10}{-0.00079:>10.5f}{0.00033:>10.5f}"
      + "".join(f"{1.0:>7.2f}" for _ in range(5)), flush=True)
out = {}
for arm, jac in ARMS.items():
    ms._chart_spin0_jac = jac
    rows = []
    for seed in (1, 2, 3):
        f = shear.train(build(), train, jr.key(seed), steps=STEPS, batch=1024,
                        g_max=G_MAX)
        eqx.tree_serialise_leaves(f"flows/arm{arm}_s{seed}.eqx", f)
        odd, even, a, pin = measure(f)
        v = float(shear.val_nll(f, val, jr.key(99)))
        rows.append((v, even, a, pin))
        print(f"{arm:>4}{seed:>5}{v:>10.4f}{odd:>10.5f}{even:>10.5f}"
              + "".join(f"{x:7.2f}" for x in a), flush=True)
    out[arm] = rows
ms._chart_spin0_jac = _FULL

print(f"\n{'arm':>4}{'val nll':>10}{'even':>21}"
      + "".join(f"{n:>15}" for n in ["Mf", "Mr", "M1", "M2", "Mc"]))
for arm, rows in out.items():
    ev = np.array([r[1] for r in rows]); al = np.array([r[2] for r in rows])
    v = np.array([r[0] for r in rows])
    print(f"{arm:>4}{v.mean():>10.4f}{ev.mean():>13.5f} +/-{ev.std():7.5f}"
          + "".join(f"{al[:,k].mean():10.2f} +/-{al[:,k].std():4.2f}" for k in range(5)))
print()
for arm, rows in out.items():
    print(f"{arm:>4}: spin-0 p1 coefficients pinned at the bound "
          + np.array2string(np.mean([r[3] for r in rows], axis=0), precision=2) + " %")
