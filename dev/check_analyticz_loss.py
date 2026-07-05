"""Diagnostic: does analytic_z change the per-g copy weights w_bg vs softmax, and
does the unbounded exp(log_w_bg - logZ_g) overflow/diverge for extreme sx_cond?

Replicates make_nll_loss's Σ_X branch on one batch and compares softmax vs analytic-Z
weights across the FULL training sx_cond range (log_scale, e), the weights_cdf
(importance) batch path actually used in training.  Reports max weight, any non-finite,
and the loss-driving difference per sx_cond.
"""
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr

from bfd_cnf.converge_train import load_training_dataset
from bfd_cnf.config import batch_size, log_scale_range, e_max, n_sx_train
from bfd_cnf.models.flows import build_flows, shear, _batch_log_L_X
from bfd_cnf.training import load_models

key = jr.PRNGKey(1)
data = load_training_dataset(key=key, subsample=4)
moments = data["moments_jnp"]; dm_dg = data["dm_dg_jnp"]; d2m = data["d2m_dg2_jnp"]
X_all = data["centroid_moments_jnp"]; nda = data["nda"]; r2s = data["raw2standard"]
weights = data["weights"]
N = moments.shape[0]
nda_arr = jnp.asarray(nda).astype(jnp.float32)
nda_arr = nda_arr / jnp.maximum(jnp.mean(nda_arr), jnp.finfo(jnp.float32).tiny)
w = jnp.asarray(weights); w = w / jnp.maximum(jnp.mean(w), jnp.finfo(jnp.float32).tiny)
w_cdf = jnp.cumsum(w / jnp.sum(w))

prior, q = build_flows(jr.PRNGKey(0), latent_dim=4, cond_dim=16, raw2standard=r2s)
prior, q = load_models(prior, q, "flows/prior_flow_xy_nll_analyticZ.eqx", "flows/q_flow_xy.eqx")

sqrt2 = 1.0 / np.sqrt(2.0)
g_grid = jnp.array([[0.,1.],[sqrt2,sqrt2],[1.,0.],[sqrt2,-sqrt2],
                    [0.,-1.],[-sqrt2,-sqrt2],[-1.,0.],[-sqrt2,sqrt2]])[None]
g = jnp.concatenate([jnp.array([[[0.,0.]]]), 0.01*g_grid, 0.02*g_grid], axis=1)
G = g.shape[1]; g2d = g.reshape(G, 2); B = batch_size; BG = B*G

# weights_cdf (importance) batch path, as in training
u = jr.uniform(jr.PRNGKey(7), shape=(B,))
idx = jnp.clip(jnp.searchsorted(w_cdf, u, side="right"), 0, N-1)
w_b = jnp.maximum(w[idx], jnp.finfo(jnp.float32).tiny)
is_corr = 1.0/w_b; is_corr = is_corr/jnp.maximum(jnp.mean(is_corr), jnp.finfo(jnp.float32).tiny)
log_is_corr = jnp.log(is_corr)

y_b, dg_b, d2g_b, X_b = moments[idx], dm_dg[idx], d2m[idx], X_all[idx]
nda_b = nda_arr[idx]
y_sheared = shear(y_b, g, dg_b[:, :4], d2g_b[:, :4]).reshape(BG, -1)
y_std = jax.vmap(r2s.transform_and_log_det)(y_sheared)[0]
g_flat = jnp.broadcast_to(g2d[None], (B, G, 2)).reshape(BG, 2)
X_bg = shear(X_b, g, dg_b[:, 4:6], d2g_b[:, 4:6]).reshape(BG, 2)
log_nda_b = jnp.log(jnp.maximum(nda_b, jnp.finfo(jnp.float32).tiny))

# reference 1M (cap at N), as in the run (--z-ref-size 1000000)
R = min(1_000_000, N)
ref_idx = jr.choice(jr.PRNGKey(0), N, shape=(R,), replace=False)
Xr, dgr, d2gr, ndar = X_all[ref_idx], dm_dg[ref_idx], d2m[ref_idx], nda_arr[ref_idx]
X_ref_g = shear(Xr, g, dgr[:, 4:6], d2gr[:, 4:6]).reshape(R*G, 2)
lognda_ref = jnp.log(jnp.maximum(ndar, jnp.finfo(jnp.float32).tiny))

# sweep the FULL training sx range: log_scale in range, e from 0 to e_max
lo, hi = log_scale_range
print(f"sx sweep over log_scale [{lo},{hi}], e up to {e_max}  (R={R}, B={B})")
print(f"{'log_scale':>9} {'e1':>6} {'logZg_min':>10} {'w_az_max':>10} {'nonfinite':>9} "
      f"{'loss_soft':>10} {'loss_az':>10} {'wlp_corr':>9}")
for ls in np.linspace(lo, hi, 6):
    for e1 in [0.0, e_max]:
        sx = jnp.array([ls, e1, 0.0])  # sx_cond = [log_scale, e1, e2] (flows.py:_sample_sx_conds)
        sx_tiled = jnp.broadcast_to(sx[None], (BG, 3))
        cond_p = jnp.concatenate([g_flat, sx_tiled], axis=-1)
        log_p = prior.log_prob(y_std, condition=cond_p).reshape(B, G)
        logL_bg = _batch_log_L_X(X_bg, sx).reshape(B, G)
        log_w_bg = logL_bg + (log_is_corr + log_nda_b)[:, None]
        w_soft = jax.nn.softmax(log_w_bg, axis=0)
        logZ_g = jax.scipy.special.logsumexp(
            lognda_ref[:, None] + _batch_log_L_X(X_ref_g, sx).reshape(R, G), axis=0)
        w_az = jnp.exp(log_w_bg - logZ_g[None, :])
        denom = jnp.mean(jnp.sum(w_az, axis=0))
        w_az = w_az / denom
        nonfin = int(jnp.sum(~jnp.isfinite(w_az)))
        ls_soft = float(-jnp.mean(jnp.sum(w_soft*log_p, 0)))
        ls_az = float(-jnp.mean(jnp.sum(jnp.nan_to_num(w_az)*log_p, 0)))
        wlp_s = jnp.sum(w_soft*log_p, 0); wlp_a = jnp.sum(jnp.nan_to_num(w_az)*log_p, 0)
        cc = float(jnp.corrcoef(wlp_s, wlp_a)[0,1])
        print(f"{ls:9.2f} {e1:6.3f} {float(logZ_g.min()):10.3f} {float(jnp.max(w_az)):10.3e} "
              f"{nonfin:9d} {ls_soft:10.4f} {ls_az:10.4f} {cc:9.4f}")
