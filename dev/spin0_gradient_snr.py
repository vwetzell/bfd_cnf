"""Why does training miss a spin-0 optimum the data pins to 3%?

`dev/spin0_profile.py` showed the NLL has a clean convex minimum at the true
spin-0 scale (Fisher info ~0.011/galaxy, so sigma_alpha ~0.03 over 100k), yet
training converges to alpha ~0.43.  So the signal is present and the optimiser
is not using it.  Candidates, measured here per minibatch:

  1. SNR of the spin-0 directional gradient vs the spin-2 one.
  2. Whether `clip_by_global_norm(1.0)` is active -- if the global norm is
     routinely >1, clipping rescales the WHOLE gradient, shrinking an already
     tiny spin-0 component along with everything else.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
from models.shear import ShearResponse, dm_dg

D = "../bfd_cnf_imsims/data"
real = shear.load(f"{D}/moments.fits")[0]
G = bulk.build_flow(jr.key(0), np.asarray(real), shear=True)
G = eqx.tree_deserialise_leaves("flows/shear.eqx", G)
chart, layerG = shear._chart(G), shear._shear_layer(G)

N = 100_000
m_syn = G.sample(jr.key(7), (N,), condition=jnp.zeros(2))
qs, rs = [], []
for i in range(0, N, 20000):
    q, r = jax.vmap(dm_dg, in_axes=(None, 0, None))(layerG, m_syn[i:i+20000], chart)
    qs.append(q); rs.append(r)
q_all, r_all = jnp.concatenate(qs), jnp.concatenate(rs)


class ScaledCoeffs(eqx.Module):
    inner: eqx.Module
    scale: jax.Array
    def __call__(self, f, a, b, q):
        return self.inner(f, a, b, q) * self.scale


IDX = {"spin-0 (1st order)": [0, 3, 6], "spin-2 (1st order)": [9, 10]}

def make(alpha, idx):
    sc = jnp.ones(14).at[jnp.array(idx)].set(alpha)
    return eqx.tree_at(
        lambda f: [b.coeffs for b in f.bijection.bijection.bijections
                   if isinstance(b, ShearResponse)],
        G, [ScaledCoeffs(layerG.coeffs, sc)])

@eqx.filter_jit
def dnll(alpha, idx_arr, xb, gb):
    def f(a):
        sc = jnp.ones(14).at[idx_arr].set(a)
        flow = eqx.tree_at(
            lambda ff: [b.coeffs for b in ff.bijection.bijection.bijections
                        if isinstance(b, ShearResponse)],
            G, [ScaledCoeffs(layerG.coeffs, sc)])
        return -jnp.mean(flow.log_prob(xb, condition=gb))
    return jax.grad(f)(alpha)

for batch in (1024, 16384):
    print(f"\n=== minibatch {batch} (antithetic, as in training) ===")
    for name, idx in IDX.items():
        ia = jnp.array(idx)
        gs = []
        for s in range(40):
            k1, k2 = jr.split(jr.key(100 + s))
            sel = jr.randint(k1, (batch // 2,), 0, N)
            gh = shear.sample_g(k2, batch // 2, shear.G_MAX)
            gb = jnp.concatenate([gh, -gh])
            tk = jnp.concatenate([sel, sel])
            xb = shear.lens(m_syn[tk], q_all[tk], r_all[tk], gb)
            gs.append(float(dnll(1.0, ia, xb, gb)))
        gs = np.array(gs)
        print(f"  {name:22s} mean {gs.mean():+.3e}  sd {gs.std():.3e}"
              f"  |SNR| {abs(gs.mean())/gs.std():.4f}")
