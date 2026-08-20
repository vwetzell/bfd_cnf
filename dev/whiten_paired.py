"""Does whitening the coefficient net's inputs reduce the spin-0 seed spread?

`dev/seed_spread.py` measured the noise floor at fixed 96k steps: Mf sd 0.54,
Mr 0.40, Mc 0.28, while held-out val nll sd is 7e-4.  The response is not
learned reproducibly; the density is.  Whitening is the candidate cause --
inputs `a` and `b` correlate at 0.997 and the input condition number is 2625,
and Adam's diagonal preconditioner cannot fix input CORRELATION.

PAIRED: same seeds, same data, same init, same steps in both arms.  The control
resets the whitening statistics to identity, which is exactly the pre-change
behaviour, so the arms differ in nothing else.

What would count as success: a drop in the SD of the spin-0 columns.  A change
in the mean alone proves nothing at this noise level.
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

def build(whiten):
    f = bulk.build_flow(jr.key(0), np.asarray(train[0]), shear=True)
    keep = [i for i, b in enumerate(f.bijection.bijection.bijections)
            if not isinstance(b, ShearResponse)]
    f = eqx.tree_at(lambda ff: [ff.bijection.bijection.bijections[i] for i in keep],
                    f, list(bulk_only.bijection.bijection.bijections))
    if not whiten:                       # control: statistics back to identity
        lay = shear._shear_layer(f)
        f = eqx.tree_at(lambda ff: [shear._shear_layer(ff).coeffs.u_mean,
                                    shear._shear_layer(ff).coeffs.u_white],
                        f, [type(lay.coeffs.u_mean)(jnp.zeros(4)),
                            type(lay.coeffs.u_white)(jnp.eye(4))])
    return f

STEPS, SEEDS = 96_000, (1, 2, 3, 4, 5)
out = {}
for whiten in (False, True):
    tag = "whitened" if whiten else "control "
    rows = []
    print(f"\n=== {tag} ===", flush=True)
    print(f"{'seed':>6}{'val nll':>10}" + "".join(f"{n:>8}" for n in
          ["Mf", "Mr", "M1", "M2", "Mc"]), flush=True)
    for seed in SEEDS:
        f = shear.train(build(whiten), train, jr.key(seed), steps=STEPS, batch=1024)
        a = alpha(f); rows.append(a)
        v = float(shear.val_nll(f, val, jr.key(99)))
        print(f"{seed:>6}{v:>10.4f}" + "".join(f"{x:8.2f}" for x in a), flush=True)
    out[tag] = np.array(rows)

print(f"\n{'':>16}" + "".join(f"{n:>8}" for n in ["Mf", "Mr", "M1", "M2", "Mc"]))
for tag, r in out.items():
    print(f"{tag+' mean':>16}" + "".join(f"{x:8.2f}" for x in r.mean(0)))
    print(f"{tag+' SD':>16}" + "".join(f"{x:8.2f}" for x in r.std(0)))
