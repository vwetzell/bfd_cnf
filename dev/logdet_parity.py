"""Which order in g carries the log-det runaway?

The physical mean log-det is 0.0000 +/- 0.0013 (`dev/physical_logdet.py`), and
it is zero for a REASON: the first-order response goes as Re(ebar.g), which is
parity-odd in e, so its divergence averages to zero over an isotropic
population.  That protection is exact and needs no fitted constant.

But it only covers the ODD (first-order) part.  The even part is second order in
g -- the `p2 = |g|^2` column of s0, which is spin-0 and survives the average.
If the runaway lives there, the constraint is specific and cheap: it is about
the p2 block, not the whole layer.

Split by parity: odd = (ld(+g) - ld(-g))/2, even = (ld(+g) + ld(-g))/2.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
from models.shear import ShearResponse

D = "../bfd_cnf_imsims/data"
data = shear.load(f"{D}/moments.fits")
n_train = int(0.9 * len(data[0]))
train = tuple(x[:n_train] for x in data)
val = tuple(x[n_train:] for x in data)
m_e = jnp.asarray(val[0][:20000])
bulk_only = eqx.tree_deserialise_leaves(
    "flows/ladder/bulk_60000.eqx", bulk.build_flow(jr.key(0), np.asarray(train[0])))

def build(whiten):
    f = bulk.build_flow(jr.key(0), np.asarray(train[0]), shear=True)
    keep = [i for i, b in enumerate(f.bijection.bijection.bijections)
            if not isinstance(b, ShearResponse)]
    f = eqx.tree_at(lambda ff: [ff.bijection.bijection.bijections[i] for i in keep],
                    f, list(bulk_only.bijection.bijection.bijections))
    if not whiten:
        lay = shear._shear_layer(f)
        f = eqx.tree_at(lambda ff: [shear._shear_layer(ff).coeffs.u_mean,
                                    shear._shear_layer(ff).coeffs.u_white],
                        f, [type(lay.coeffs.u_mean)(jnp.zeros(4)),
                            type(lay.coeffs.u_white)(jnp.eye(4))])
    return f

print(f"{'arm':>10}{'steps':>8}{'val nll':>10}{'log-det':>10}{'odd':>10}{'even':>10}")
print(f"{'physical':>10}{'':>8}{'':>10}{0.0:>10.4f}{0.0:>10.4f}{0.0:>10.4f}   (+/- 0.0013)")
for whiten, steps in ((False, 6_000), (False, 96_000), (False, 400_000),
                      (True, 96_000)):
    f = shear.train(build(whiten), train, jr.key(1), steps=steps, batch=1024)
    lay, ch = shear._shear_layer(f), shear._chart(f)
    z = jax.vmap(ch.transform)(m_e)
    ldp = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([0.02, 0.]))[1])(z)
    ldm = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([-0.02, 0.]))[1])(z)
    v = float(shear.val_nll(f, val, jr.key(99)))
    print(f"{'whitened' if whiten else 'plain':>10}{steps:>8}{v:>10.4f}"
          f"{float(jnp.mean(ldp)):>10.4f}{float(jnp.mean(ldp-ldm)/2):>10.4f}"
          f"{float(jnp.mean(ldp+ldm)/2):>10.4f}", flush=True)
