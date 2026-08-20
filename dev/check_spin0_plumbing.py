"""Feed the layer CONSTANT raw coefficients and ask bfd's own formula for dm/dg.

If `_chart_spin0_jac` is plumbed correctly then a layer whose spin-0 network
emits a constant c must produce exactly dX/dg = X c_X Re(ebar.g) in RAW moment
space, for every X and every galaxy -- no fitting involved.

Result 2026-08-20: exact to 1.4e-3 (float32), once `e_scale` was fixed.  Before
that it was off by a uniform 1.85% across all three moments -- a common scale
factor, which is what said the error was not structural: `build_flow` passed
`t.std(0)[3]` where the chart divides z3 and z4 by the symmetrised
sqrt((s3^2 + s4^2)/2).  Predicted 1.018385 for this script's 1000-row chart
against a measured 1.0185.
"""
import sys; sys.path.insert(0, "/home/vwetzell/gitrepos/bfd_cnf")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
from models.shear import dm_dg
from paramax import unwrap

D = "../bfd_cnf_imsims/data"
m_all, _, _ = shear.load(f"{D}/moments.fits")
m = jnp.asarray(m_all[:200])
flow = bulk.build_flow(jr.key(0), np.asarray(m_all[:1000]), shear=True)
chart, layer = shear._chart(flow), shear._shear_layer(flow)

C = jnp.array([1.7, -0.9, 2.6])          # arbitrary constant raw coefficients


class Const(eqx.Module):
    """`_Coeffs` replaced by a constant: p1 column = C, everything else zero."""
    c: jax.Array

    def __call__(self, f, a, b, q):
        return jnp.zeros(14).at[jnp.array([0, 3, 6])].set(self.c)


lay = eqx.tree_at(lambda l: l.coeffs, layer, Const(C),
                  is_leaf=lambda x: x is layer.coeffs)
q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(lay, m, chart)   # (n,2,5) dm/dg

mm = np.asarray(m)
Mr = mm[:, 1]
e = np.stack([mm[:, 2] / Mr, mm[:, 3] / Mr], -1)                 # physical e
X = mm[:, [0, 1, 4]]
# unshear carries data -> base, so the GENERATIVE response is minus the
# coefficient: dX/dg_k = -X c_X e_k.
want = -X[:, None, :] * np.asarray(C)[None, None, :] * e[:, :, None]
got = np.asarray(q)[:, :, [0, 1, 4]]
rel = np.abs(got - want) / np.abs(want).max(axis=(1, 2), keepdims=True)
print(f"constant-coefficient plumbing check, {len(mm)} galaxies")
print(f"  max relative error in dm/dg over (Mf, Mr, Mc): {rel.max():.2e}")
for k, n_ in enumerate(["Mf", "Mr", "Mc"]):
    print(f"    {n_}: got/want median {np.median(got[:,:,k]/want[:,:,k]):.6f}")
