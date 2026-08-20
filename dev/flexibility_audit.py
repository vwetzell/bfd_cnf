"""Which block of the shear layer buys the unphysical log-det at g_max = 0.02?

The layer's mean log-det at |g| = 0.02 should be ~0: the physical value is
+0.00033 (`dev/physical_logdet.py`), and the ODD part is protected exactly by
isotropy.  What training produces instead is an EVEN part of 0.4-1.7 nats
(`dev/logdet_parity.py`) -- a volume collapse three orders of magnitude too
large, and the reason `dev/gmax_sweep.py` found g_max = 0.10 "fixes" it.

Even means SECOND order in g, which invites the obvious reading: the g^2
coefficient blocks (p2, p3, mu, nu, rho) are barely constrained at |g| <= 0.02,
so the likelihood spends them on density.  This script tests that reading by
zeroing one block at a time and re-measuring.  The alternative is that the leak
is -1/2 g^2 tr((dv/dz)^2), which is second order in g but built from the FIRST
order coefficients' steepness in z -- in which case bounding the g^2 blocks
fixes nothing.

Also prints |d coeff / dz| per block, against the physical value inverted from
bfd's own dm/dg by `dev/physical_steepness.py`.

RESULT (96k steps, batch 1024, g_max 0.02, bulk_60000, whitening on)

    mean log-det at |g| = 0.02      physical: odd -0.00079, even +0.00033
      block zeroed         odd        even   even removed
              none     0.00747     0.50906
       a (p1, 1st)     0.00097     0.29445        0.21461
    A (spin2, 1st)     0.00197     0.08561        0.42346
    B (spin2, 1st)     0.00678     0.50809        0.00097
   p2 (|g|^2, 2nd)     0.00712     0.43796        0.07111
          p3 (2nd)     0.00748     0.50904        0.00003
          mu (2nd)     0.00741     0.50806        0.00100
          nu (2nd)     0.00743     0.50437        0.00469
         rho (2nd)     0.00747     0.50906        0.00000
     ALL 2nd order     0.00706     0.43205        0.07702

The even part is 1500x its physical value, so the collapse is real -- but the
nine genuinely second-order coefficients carry 15% of it and the two spin-2
FIRST-order ones carry 83%.  Bounding the g^2 blocks would not have worked.
The leak is -1/2 g^2 tr((dv/dz)^2): even in g, sourced by the first-order
response's slope in z, and invisible to training at g_max = 0.02 because the
likelihood penalty for that slope is itself O(g^2).

Steepness, trained (median / p99 / max of |d coeff / dz|):

     a_Mf  121.6 /  5367 /   8797        a_Mr    1.8 / 10286 / 130294
        A    5.1 /  1938 /   2673           B   10.6 / 13071 /  45026

against physical values inverted from bfd's own dm/dg by robust binning:
a_Mf slope ~0.25, a_Mr ~10-20, A ~0.15, B ~0.7 over the whole population.

And the opposite failure in the same layer -- `_COEFF_MAX` is too SMALL in this
chart.  The physical coefficients, binned in z1 (duodeciles, 200k templates,
weight |e|^2), in the raw parameterisation the bound was sized on and in the
chart the layer actually uses:

       z1   Mr/Mf     c_Mf     a_Mf     c_Mr     a_Mr     c_Mc     a_Mc
    -2.54    2.13     0.98     1.41     0.84    -0.15     0.16    -1.73
    -0.48    3.10     1.74     2.50     2.74     6.62     3.29     4.74
    -0.01    3.29     1.88     2.71     3.10    11.83     3.89     9.94
     0.55    3.44     2.00     2.87     3.35    20.99     4.27    18.82
     1.10    3.54     2.07     2.97     3.51    35.99     4.53    33.64
     2.36    3.60     2.11     3.04     3.63    61.35     4.74    59.38

c stays inside [-2, 7]; a runs to 253 at p99 because z1's logit Jacobian
diverges at the point-source ceiling.  See `models/shear.py`'s `_COEFF_MAX`.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
from models.shear import ShearResponse

D = "../bfd_cnf_imsims/data"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 96_000
G_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
OUT = f"flows/audit_g{G_MAX}_s{STEPS}.eqx"

data = shear.load(f"{D}/moments.fits")
n_train = int(0.9 * len(data[0]))
train = tuple(x[:n_train] for x in data)
val = tuple(x[n_train:] for x in data)
m_e = jnp.asarray(val[0][:20000])

flow = bulk.build_flow(jr.key(0), np.asarray(train[0]), shear=True)
bulk_only = eqx.tree_deserialise_leaves(
    "flows/ladder/bulk_60000.eqx", bulk.build_flow(jr.key(0), np.asarray(train[0])))
keep = [i for i, b in enumerate(flow.bijection.bijection.bijections)
        if not isinstance(b, ShearResponse)]
flow = eqx.tree_at(lambda f: [f.bijection.bijection.bijections[i] for i in keep],
                   flow, list(bulk_only.bijection.bijection.bijections))

import os
if os.path.exists(OUT):
    flow = eqx.tree_deserialise_leaves(OUT, flow)
    print(f"loaded {OUT}")
else:
    flow = shear.train(flow, train, jr.key(1), steps=STEPS, batch=1024, g_max=G_MAX)
    eqx.tree_serialise_leaves(OUT, flow)
    print(f"wrote {OUT}")
print(f"val nll {shear.val_nll(flow, val, jr.key(99)):.4f}", flush=True)

layer, chart = shear._shear_layer(flow), shear._chart(flow)
z = jax.vmap(chart.transform)(m_e)


class Masked(eqx.Module):
    """`_Coeffs` with some of the fourteen outputs forced to zero."""
    coeffs: eqx.Module
    mask: jax.Array

    def __call__(self, f, a, b, q):
        return self.coeffs(f, a, b, q) * self.mask


# s0 is coeffs[:9].reshape(3,3): row = (z0, z1, z2), col = (p1, p2, p3).
# p1 = Re(ebar.g) is FIRST order; p2 = |g|^2 and p3 = Re(ebar^2 g^2) are second.
BLOCKS = {
    "a (p1, 1st)":      [0, 3, 6],
    "A (spin2, 1st)":   [9],
    "B (spin2, 1st)":   [10],
    "p2 (|g|^2, 2nd)":  [1, 4, 7],
    "p3 (2nd)":         [2, 5, 8],
    "mu (2nd)":         [11],
    "nu (2nd)":         [12],
    "rho (2nd)":        [13],
}


def parity(lay):
    ldp = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([G_MAX, 0.]))[1])(z)
    ldm = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([-G_MAX, 0.]))[1])(z)
    return float(jnp.mean(ldp - ldm) / 2), float(jnp.mean(ldp + ldm) / 2)


def with_mask(idx):
    mask = jnp.ones(14).at[jnp.array(idx)].set(0.0)
    return eqx.tree_at(lambda l: l.coeffs, layer, Masked(layer.coeffs, mask),
                       is_leaf=lambda x: x is layer.coeffs)


odd0, even0 = parity(layer)
print(f"\nmean log-det at |g| = {G_MAX}  (physical: odd -0.00079, even +0.00033)")
print(f"{'block zeroed':>18}{'odd':>12}{'even':>12}{'even removed':>15}")
print(f"{'none':>18}{odd0:>12.5f}{even0:>12.5f}")
for name, idx in BLOCKS.items():
    o, e = parity(with_mask(idx))
    print(f"{name:>18}{o:>12.5f}{e:>12.5f}{even0 - e:>15.5f}", flush=True)
o, e = parity(with_mask([i for n, ix in BLOCKS.items() if "2nd" in n for i in ix]))
print(f"{'ALL 2nd order':>18}{o:>12.5f}{e:>12.5f}{even0 - e:>15.5f}")

# steepness of each coefficient in z, on the same templates
G = jax.vmap(jax.jacfwd(lambda zi: layer.coeffs(zi[0], zi[1], zi[2],
                                                zi[3]**2 + zi[4]**2)))(z)   # (n,14,5)
nrm = np.asarray(jnp.sqrt(jnp.sum(G ** 2, -1)))
names = ["a_Mf", "p2_Mf", "p3_Mf", "a_Mr", "p2_Mr", "p3_Mr", "a_Mc", "p2_Mc",
         "p3_Mc", "A", "B", "mu", "nu", "rho"]
print(f"\n|d coeff / dz| of the trained network")
print(f"{'coeff':>8}{'median':>10}{'p99':>10}{'max':>12}")
for i, n_ in enumerate(names):
    print(f"{n_:>8}{np.median(nrm[:,i]):10.2f}{np.percentile(nrm[:,i],99):10.2f}"
          f"{nrm[:,i].max():12.1f}")
