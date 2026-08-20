"""Is the spin-0 shear response actually IN the likelihood, or not?

"NLL training fails to recover it" does not imply "the data does not constrain
it".  This separates the two with no optimiser involved: take the generator
flow, scale ONLY its first-order spin-0 coefficients by a factor, and evaluate
the NLL of data generated at scale 1.0.

  clear minimum at 1.0 -> the signal IS present and training is at fault
  flat                 -> genuinely unidentified

`response` is linear in `coeffs`, and the first-order spin-0 block is the p1
column of `s0 = coeffs[:9].reshape(3,3)`, i.e. flat indices 0, 3, 6.  (p2 and
p3 are second order in g; A, B at 9, 10 are the first-order SPIN-2 block, held
fixed here so this isolates spin-0.)
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

# antithetic evaluation set, exactly as training builds its batches
g_half = shear.sample_g(jr.key(3), N // 2, shear.G_MAX)
g = jnp.concatenate([g_half, -g_half])
take = jnp.concatenate([jnp.arange(N // 2), jnp.arange(N // 2)])
x = shear.lens(m_syn[take], q_all[take], r_all[take], g)


class ScaledCoeffs(eqx.Module):
    inner: eqx.Module
    scale: jax.Array
    def __call__(self, f, a, b, q):
        return self.inner(f, a, b, q) * self.scale


def nll_at(alpha):
    sc = jnp.ones(14).at[jnp.array([0, 3, 6])].set(alpha)
    flow = eqx.tree_at(
        lambda f: [b.coeffs for b in f.bijection.bijection.bijections
                   if isinstance(b, ShearResponse)],
        G, [ScaledCoeffs(layerG.coeffs, sc)])
    tot, n = 0.0, 0
    for i in range(0, len(x), 25000):
        lp = flow.log_prob(x[i:i+25000], condition=g[i:i+25000])
        lp = lp[jnp.isfinite(lp)]
        tot += float(jnp.sum(lp)); n += int(lp.shape[0])
    return -tot / n


base = nll_at(1.0)
print(f"NLL at the TRUE spin-0 scale (alpha=1): {base:.6f}")
print(f"\n{'alpha':>7}{'NLL':>14}{'NLL - NLL(1)':>16}")
for a in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0):
    v = nll_at(a)
    print(f"{a:7.2f}{v:14.6f}{v - base:16.3e}", flush=True)
