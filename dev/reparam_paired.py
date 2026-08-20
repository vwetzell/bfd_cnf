"""Does pulling the chart Jacobian out of the spin-0 coefficients close the leak?

The layer's spin-0 network now emits the RAW response coefficients c_X of
`dX/dg = X c_X Re(ebar.g)` and `models.shear._chart_spin0_jac` carries them into
z analytically.  Two things should follow, and this is the paired test of both:

  * `_COEFF_MAX = 12` stops binding.  The physical c stay inside [-2, 7] where
    the chart-space a they replace ran to a p99 of 253.
  * the even (second-order in g) log-det shrinks toward its physical +0.00033.
    The target function is flat now -- c_Mr climbs 0.84 -> 3.63 along z1 where
    a_Mr climbed 0.84 -> 61 -- so the network no longer has to be steep to fit
    it, and steepness is what `dev/flexibility_audit.py` showed buys the leak.

Same arms as `dev/gmax_sweep.py`'s g_max = 0.02 row, which is the baseline:
96k steps, batch 1024, bulk_60000, whitening on, seeds 1/2/3.  Three seeds
because `dev/seed_spread.py` measured sd 0.28-0.54 in the spin-0 alphas at
fixed steps -- a single run cannot tell these arms apart.

    baseline (old parameterisation, dev/gmax_sweep.py):
        even 0.49544 +/- 0.12672
        alpha  Mf 1.00   Mr 0.55   M1 0.87   M2 0.87   Mc 0.45

RESULT (3 seeds, 96k steps each)

    seed   val nll       odd      even     Mf     Mr     M1     M2     Mc
    phys            -0.00079   0.00033   1.00   1.00   1.00   1.00   1.00
       1   40.3934   0.00046   0.02031   1.48   0.66   1.03   1.04   0.06
       2   40.3944   0.00027   0.02411   1.65   0.81   1.03   1.03   0.17
       3   40.3917   0.00087   0.02327   1.70   0.81   1.04   1.04   0.15

    even  0.02257 +/- 0.00163   against  0.49544 +/- 0.12672

The leak is 22x smaller and its seed spread 78x smaller, and the bound stopped
binding: a_Mr and a_Mc are pinned for 0.00% of templates where they were pinned
for 61% and 53%.  Spin-2 came along for the ride, 0.87 -> 1.04.

Two things this did NOT fix.

Mf and Mc got worse (1.00 -> 1.61, 0.45 -> 0.13) and the fitted coefficient
functions are still the wrong shape -- fitted c_Mf runs -7.0 -> +0.6 along z1
where the truth wants a flat -1.0 -> -2.1, and c_Mc changes sign.  The layer is
no longer PREVENTED from representing the spin-0 response; the likelihood still
does not identify it, which is [[nll-learns-spin2-not-spin0]] and a separate
problem.

The network is still far steeper than it needs to be: median |dc/dz| 20-80,
against physical raw-coefficient slopes of ~0.25 (c_Mf) to ~0.6 (c_Mr).  That
residual steepness is the remaining 0.022 of leak, and bounding it is the next
change, not this one.

Val nll rose 40.11 -> 40.39.  Expected: 0.47 nats of the old arm's likelihood
was the volume collapse itself, and this arm does not take it.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
from models.shear import ShearResponse, dm_dg

D = "../bfd_cnf_imsims/data"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 96_000
G_MAX = 0.02

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
    ldp = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([G_MAX, 0.]))[1])(z)
    ldm = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([-G_MAX, 0.]))[1])(z)
    q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(lay, m_e, ch)
    s = shear._scale(m_e)[:, None, :]
    p, t = np.asarray(q / s), np.asarray(q_e / s)
    a = ((np.mean((p + t) ** 2, (0, 1)) - np.mean((p - t) ** 2, (0, 1)))
         / (4 * np.mean(t ** 2, (0, 1))))
    cf = np.asarray(jax.vmap(lambda zi: lay.coeffs(zi[0], zi[1], zi[2],
                                                   zi[3]**2 + zi[4]**2))(z))
    pinned = 100 * np.mean(np.abs(cf) > 11.88, axis=0)
    G = jax.vmap(jax.jacfwd(lambda zi: lay.coeffs(zi[0], zi[1], zi[2],
                                                  zi[3]**2 + zi[4]**2)))(z)
    steep = np.asarray(jnp.sqrt(jnp.sum(G ** 2, -1)))
    return (float(jnp.mean(ldp - ldm) / 2), float(jnp.mean(ldp + ldm) / 2), a,
            pinned, np.median(steep, axis=0))


print(f"{'seed':>5}{'val nll':>10}{'odd':>10}{'even':>10}"
      + "".join(f"{n:>7}" for n in ["Mf", "Mr", "M1", "M2", "Mc"]), flush=True)
print(f"{'phys':>5}{'':>10}{-0.00079:>10.5f}{0.00033:>10.5f}"
      + "".join(f"{1.0:>7.2f}" for _ in range(5)))
rows = []
for seed in (1, 2, 3):
    f = shear.train(build(), train, jr.key(seed), steps=STEPS, batch=1024, g_max=G_MAX)
    eqx.tree_serialise_leaves(f"flows/reparam_s{seed}.eqx", f)
    odd, even, a, pinned, steep = measure(f)
    v = float(shear.val_nll(f, val, jr.key(99)))
    rows.append((even, a, pinned, steep))
    print(f"{seed:>5}{v:>10.4f}{odd:>10.5f}{even:>10.5f}"
          + "".join(f"{x:7.2f}" for x in a), flush=True)

names = ["a_Mf", "p2_Mf", "p3_Mf", "a_Mr", "p2_Mr", "p3_Mr", "a_Mc", "p2_Mc",
         "p3_Mc", "A", "B", "mu", "nu", "rho"]
ev = np.array([r[0] for r in rows]); al = np.array([r[1] for r in rows])
print(f"\neven {ev.mean():.5f} +/- {ev.std():.5f}   (baseline 0.49544 +/- 0.12672)")
print("alpha" + "".join(f"{x:8.2f}" for x in al.mean(0))
      + "   (baseline 1.00 0.55 0.87 0.87 0.45)")
print(f"\n{'coeff':>8}{'% pinned at bound':>20}{'median |dc/dz|':>16}")
for i, n_ in enumerate(names):
    print(f"{n_:>8}{np.mean([r[2][i] for r in rows]):20.2f}"
          f"{np.mean([r[3][i] for r in rows]):16.2f}")
