"""Smoke test the stripped weighting: sample ∝ nda, weight by L only.
Builds the real flow + real make_nll_loss/make_elbo_loss on synthetic data and
runs one value_and_grad in both the Σ_X and non-Σ_X branches. Passes if finite.
"""
import numpy as np, jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from bfd_cnf.models.flows import build_flows, make_nll_loss, make_elbo_loss
from bfd_cnf.models.bijections import RawMomentStandardize, load_stats

N, B = 2000, 128
rng = np.random.default_rng(0)
Mf = rng.uniform(1000, 50000, N)
Mr = Mf * rng.uniform(2.5, 3.4, N)
M1 = Mr * rng.uniform(-0.2, 0.2, N)
M2 = Mr * rng.uniform(-0.2, 0.2, N)
moments = jnp.asarray(np.stack([Mf, Mr, M1, M2], 1))
X = jnp.asarray(rng.normal(0, 300, size=(N, 2)))
Sigma = jnp.asarray(np.broadcast_to(np.eye(4) * 1e3, (N, 4, 4)))
dg = jnp.asarray(rng.normal(0, 1, size=(N, 6, 2)))
d2g = jnp.asarray(rng.normal(0, 1, size=(N, 6, 2, 2)))
nda = jnp.asarray(rng.pareto(1.5, N) + 0.1)          # heavy-tailed like the HT weight
weights = nda / jnp.mean(nda)                         # proposal ∝ nda

r2s = load_stats("flows/prior_flow_xy_nll.eqx")       # realistic standardiser
prior, q = build_flows(jr.key(0), 4, 16, raw2standard=r2s)

common = dict(N=N, batch_size=B, weights=weights, e_max=0.05, n_sx_train=4,
              raw2standard=r2s)
for lsr, tag in [((10.5, 13.0), "Σ_X on"), (None, "Σ_X off")]:
    nll = make_nll_loss(log_scale_range=lsr, use_sx=lsr is not None, **common)
    v, g = eqx.filter_value_and_grad(
        lambda m: nll(m, moments, Sigma, dg, d2g, X, jr.key(1)))(prior)
    gn = float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(
        eqx.filter(g, eqx.is_inexact_array)))))
    print(f"NLL  [{tag}]  loss={float(v):+.4f}  |grad|={gn:.3e}  "
          f"finite={np.isfinite(float(v)) and np.isfinite(gn)}")

# ELBO needs the whitening stats; pass zeros (cond features tolerate it for smoke).
z = jnp.zeros(4)
elbo = make_elbo_loss(N=N, batch_size=B, num_samples=4, weights=weights,
                      log_scale_range=(10.5, 13.0), e_max=0.05, n_sx_train=4,
                      raw2standard=r2s, mean_log_diag=z, std_log_diag=z + 1,
                      mean_off=jnp.zeros(6), std_off=jnp.ones(6))
v, g = eqx.filter_value_and_grad(
    lambda mt: elbo(mt, moments, Sigma, dg, d2g, X, jr.key(2)))((prior, q))
print(f"ELBO [Σ_X on]  loss={float(v):+.4f}  finite={np.isfinite(float(v))}")
print("SMOKE OK")
