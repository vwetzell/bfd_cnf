"""What SHOULD the shear layer's log-det be?

The runaway is diagnosed as the layer buying likelihood with an unphysical
volume contraction at g != 0 (log-det 0.02 -> 2.4 nats over training).  Before
constraining it, measure the physical value -- an earlier guess of ~1e-4 came
from `det A = 1` for the image-space shear, which is the WRONG Jacobian: the
moment map m -> m(g) is not the image map, and its transport divergence need
not vanish.

Exact route, no regression.  For a velocity field v and density P0,
integration by parts gives

    E[div v] = -E[v . grad log P0]

(boundary term vanishes because P0 -> 0 at the chart edges), and the mean
log-det of the transport at shear g is  ~ g * E[div v].  v comes from bfd's
exact per-template dm/dg pushed through the chart; grad log P0 comes from the
trained bulk.  The tower property covers the per-template scatter in v, since
only the expectation is needed.

Sanity check included: E[grad log P0] must be 0 for any smooth density.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear

D = "../bfd_cnf_imsims/data"
m_all, dm_dg, _ = shear.load(f"{D}/moments.fits")
N = 20000
m = jnp.asarray(m_all[:N]); dmdg = jnp.asarray(dm_dg[:N])

flow = bulk.build_flow(jr.key(0), np.asarray(m_all), shear=True)
flow = eqx.tree_deserialise_leaves("flows/ladder/bulk_60000.eqx",
                                   bulk.build_flow(jr.key(0), np.asarray(m_all)))
chart = flow.bijection.bijection.bijections[0]

def lp_z(z):
    """log density in CHART coordinates: log P_m(m) + log|dm/dz|."""
    mm, ld = chart.inverse_and_log_det(z)
    return flow.log_prob(mm) + ld

z = jax.vmap(chart.transform)(m)
score = jax.vmap(jax.grad(lp_z))(z)
print("sanity: E[grad log P0] should be ~0 relative to its own scatter")
sm, ss = np.asarray(score.mean(0)), np.asarray(score.std(0))
print("  mean/sd per slot:", np.array2string(sm / ss, precision=3))

# v = dz/dg : push bfd's dm/dg through the chart Jacobian
J = jax.vmap(jax.jacfwd(chart.transform))(m)            # (N,5,5) dz/dm
v = jnp.einsum("nij,nkj->nki", J, dmdg)                 # (N,2,5) dz/dg
div = -jnp.einsum("nki,ni->nk", v, score)               # per-target integrand
g = 0.02
print(f"\nE[div v] per shear component : "
      f"{np.array2string(np.asarray(div.mean(0)), precision=4)}")
print(f"  bootstrap sd               : "
      f"{np.array2string(np.asarray(div.std(0) / np.sqrt(N)), precision=4)}")
print(f"\nPHYSICAL mean log-det at g={g}: {float(div.mean(0)[0]) * g:+.6f}")
print(f"\nfor comparison, measured on trained layers:")
print(f"   6k steps   0.0019      96k steps   0.0214")
print(f"  whitened    0.43-0.70   400k        2.39")
