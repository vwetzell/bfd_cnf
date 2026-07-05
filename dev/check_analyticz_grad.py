"""Authoritative no-op test: build the REAL make_nll_loss with analytic_z True vs False
and compare the gradient w.r.t. flow params on the SAME batch/flow/key.  If the analytic_z
branch changes the loss's global minimum, the gradients must differ somewhere.
"""
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx

from bfd_cnf.converge_train import load_training_dataset
from bfd_cnf.config import batch_size, log_scale_range, e_max, n_sx_train, use_nda_weight
from bfd_cnf.models.flows import build_flows, make_nll_loss
from bfd_cnf.training import load_models

data = load_training_dataset(key=jr.PRNGKey(0), subsample=4)
moments = data["moments_jnp"]; cov = data["cov_jnp"]
dm_dg = data["dm_dg_jnp"]; d2m = data["d2m_dg2_jnp"]; X = data["centroid_moments_jnp"]
nda = data["nda"]; weights = data["weights"]; r2s = data["raw2standard"]
N = moments.shape[0]
w_np = np.asarray(weights); w_np = w_np / w_np.sum()

prior, q = build_flows(jr.PRNGKey(0), latent_dim=4, cond_dim=16, raw2standard=r2s)
prior, q = load_models(prior, q, "flows/prior_flow_xy_nll_analyticZ.eqx", "flows/q_flow_xy.eqx")

common = dict(N=N, batch_size=batch_size, weights=jnp.asarray(w_np),
              nda=(jnp.asarray(nda) if use_nda_weight else None), nda_clip_percentile=None,
              log_scale_range=log_scale_range, e_max=e_max, n_sx_train=n_sx_train,
              use_sx=(log_scale_range is not None), raw2standard=r2s)

loss_soft = make_nll_loss(analytic_z=False, **common)
loss_az = make_nll_loss(analytic_z=True, z_ref_size=1_000_000, **common)

# identical args -> the same batch is sampled inside (same key), so any diff is the branch
key = jr.PRNGKey(123)
args = (prior, moments, cov, dm_dg, d2m, X, key)

v_soft, g_soft = eqx.filter_value_and_grad(lambda f: loss_soft(f, *args[1:]))(prior)
v_az,   g_az   = eqx.filter_value_and_grad(lambda f: loss_az(f, *args[1:]))(prior)

# flatten param grads
flat_soft = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(eqx.filter(g_soft, eqx.is_inexact_array))])
flat_az   = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(eqx.filter(g_az,   eqx.is_inexact_array))])

print(f"loss softmax  = {float(v_soft):.6f}")
print(f"loss analytic = {float(v_az):.6f}   (diff {float(v_az-v_soft):+.3e})")
print(f"grad dim = {flat_soft.size}")
print(f"||g_soft|| = {float(jnp.linalg.norm(flat_soft)):.5e}")
print(f"||g_az||   = {float(jnp.linalg.norm(flat_az)):.5e}")
print(f"||g_az - g_soft|| = {float(jnp.linalg.norm(flat_az-flat_soft)):.5e}")
print(f"rel diff = {float(jnp.linalg.norm(flat_az-flat_soft)/jnp.linalg.norm(flat_soft)):.5e}")
print(f"cosine(g_soft,g_az) = {float(jnp.dot(flat_soft,flat_az)/(jnp.linalg.norm(flat_soft)*jnp.linalg.norm(flat_az))):.8f}")
