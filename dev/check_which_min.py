"""Which flow is closer to the loss's global minimum: the +0.07 (prior_flow_xy_nll)
or the +0.67 (analyticZ)?  Evaluate the ACTUAL current-code nll loss for both on the
same batches/keys and compare.  Lower loss = closer to the global min.

If analyticZ has lower loss -> +0.67 IS the loss's optimum (per-g loss is structurally
biased; the +0.07 flow is the under-/differently-converged one).  If +0.07 has lower
loss -> +0.67 is a worse local basin (an optimization/LR trap, not the global min).
"""
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr

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

loss_fn = make_nll_loss(
    N=N, batch_size=batch_size, weights=jnp.asarray(w_np),
    nda=(jnp.asarray(nda) if use_nda_weight else None), nda_clip_percentile=None,
    log_scale_range=log_scale_range, e_max=e_max, n_sx_train=n_sx_train,
    use_sx=(log_scale_range is not None), raw2standard=r2s, analytic_z=False)

def avg_loss(prior_path, n_batches=20):
    prior, q = build_flows(jr.PRNGKey(0), latent_dim=4, cond_dim=16, raw2standard=r2s)
    prior, q = load_models(prior, q, prior_path, "flows/q_flow_xy.eqx")
    ls = []
    for i in range(n_batches):
        k = jr.PRNGKey(1000 + i)  # SAME keys for both flows -> same batches
        ls.append(float(loss_fn(prior, moments, cov, dm_dg, d2m, X, k)))
    return np.array(ls)

a = avg_loss("flows/prior_flow_xy_nll.eqx", n_batches=40)          # +0.07
b = avg_loss("flows/prior_flow_xy_nll_analyticZ.eqx", n_batches=40) # +0.67
d = b - a  # PAIRED per-batch differences (same batch+sx each i)
se = d.std(ddof=1) / np.sqrt(len(d))
print(f"+0.07 flow loss {a.mean():.5f} | +0.67 flow loss {b.mean():.5f}")
print(f"PAIRED delta (analyticZ - 0.07flow) = {d.mean():+.5f} +/- {se:.5f}  ({d.mean()/se:+.1f} sigma)")
sig = abs(d.mean()) > 3*se
if not sig:
    print("-> losses EQUAL within noise => m (+0.07 vs +0.67) is a near-FLAT direction of the loss: STRUCTURAL (loss doesn't constrain shear response)")
elif d.mean() < 0:
    print("-> analyticZ significantly LOWER => +0.67 is the loss optimum (loss is biased toward it)")
else:
    print("-> +0.07 flow significantly LOWER => +0.67 is a worse local basin (optimization/LR trap)")
