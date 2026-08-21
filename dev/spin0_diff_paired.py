"""Does the chart Jacobian's DIFFERENCE structure cost the Mc response?

`_chart_spin0_jac` is M = D L, with

    D = diag(1/(sd0 ln10), 1/(sd1 (1-u)), 1/(sd2 (1-v)))     the divergence
    L = [[1,0,0], [-1,1,0], [0,-1,1]]                        the differences

D is the part that fixes the real problem: it is what makes the chart-space
coefficient diverge at the point-source ceiling, and dividing it out is what
took the even log-det from 0.495 to 0.0226 and unpinned the bound
(`dev/reparam_paired.py`).  L is not load-bearing for that.  It comes along
because dz1/dg genuinely carries d log Mr/dg - d log Mf/dg, and it lets the
network emit the raw c_X that `_COEFF_MAX` was sized for.

The suspicion it is on trial for: z1 and z2 correlate at 0.997, so
c_Mc - c_Mr is a small difference of two numbers near 3-4, and with L in place
the raw coefficients come back by cumulative sum -- Mc last, inheriting the
error in Mf and Mr.  Mc is exactly what regressed, 0.45 +/- 0.07 -> 0.13 +/- 0.05
(both tight across seeds, so not noise).

Arm "D" drops L by monkeypatching `_chart_spin0_jac` to its diagonal.  The
network then emits the DIFFERENCE combinations directly -- physical range
c_Mr - c_Mf in [-0.14, 1.52] and c_Mc - c_Mr in [-0.68, 1.11], both far inside
the bound -- and each z slot's coefficient is independent again.

Both arms run under the e_scale fix (380fe55), so "DL" here is a re-measurement
and not the number in `dev/reparam_paired.py`'s docstring.
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


def _diag_only(z, loc, scale):
    """M with L dropped: the divergence removed, the differences left in."""
    return jnp.diag(jnp.diag(_FULL(z, loc, scale)))


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
      + "".join(f"{1.0:>7.2f}" for _ in range(5)))
print(f"{'old':>4}{'':>5}{'':>10}{'':>10}{0.49544:>10.5f}"
      + "".join(f"{x:7.2f}" for x in [1.00, 0.55, 0.87, 0.87, 0.45])
      + "   (pre-reparameterisation, 3 seeds)")
out = {}
for arm, jac in (("DL", _FULL), ("D", _diag_only)):
    ms._chart_spin0_jac = jac
    rows = []
    for seed in (1, 2, 3):
        f = shear.train(build(), train, jr.key(seed), steps=STEPS, batch=1024,
                        g_max=G_MAX)
        eqx.tree_serialise_leaves(f"flows/jac{arm}_s{seed}.eqx", f)
        odd, even, a, pin = measure(f)
        v = float(shear.val_nll(f, val, jr.key(99)))
        rows.append((v, even, a, pin))
        print(f"{arm:>4}{seed:>5}{v:>10.4f}{odd:>10.5f}{even:>10.5f}"
              + "".join(f"{x:7.2f}" for x in a), flush=True)
    out[arm] = rows
ms._chart_spin0_jac = _FULL

print(f"\n{'arm':>4}{'val nll':>10}{'even':>20}"
      + "".join(f"{n:>14}" for n in ["Mf", "Mr", "M1", "M2", "Mc"]))
for arm, rows in out.items():
    ev = np.array([r[1] for r in rows]); al = np.array([r[2] for r in rows])
    v = np.array([r[0] for r in rows])
    print(f"{arm:>4}{v.mean():>10.4f}{ev.mean():>12.5f} +/-{ev.std():6.5f}"
          + "".join(f"{al[:,k].mean():9.2f} +/-{al[:,k].std():4.2f}" for k in range(5)))
for arm, rows in out.items():
    print(f"{arm}: spin-0 p1 coefficients pinned at the bound: "
          + np.array2string(np.mean([r[3] for r in rows], axis=0), precision=2) + " %")
