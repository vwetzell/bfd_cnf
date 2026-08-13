"""train_shear_bijection_bias_zero.py
=====================================
Combines the two things independently verified this session:
  1. A bijection-based shear layer (ShearPerturbative) chained after a FROZEN
     bulk flow automatically gives a valid, normalized P(m|g) for every g (no
     PSD or normalization penalties needed -- confirmed in
     dev/rederive_bijection_mechanics.py against a hand-computed closed form).
  2. Direct bias-zeroing as the training objective (not NLL, not Sobolev):
     shear real templates on the fly to a KNOWN g using their exact analytic
     dm_dg/d2m_dg2, hold the observed moment FIXED, and extract Q, R via
     autodiff of the flow's log_prob w.r.t. the CONDITIONING g at g=0 (the
     same pattern point_estimate_pqr_noiseless.py already uses for eval, and
     the same pattern verified against a closed form). Loss = the residual
     ||Q_tot - R_tot @ g_true||^2, which is zero iff the estimator
     g_hat=R_tot^{-1}Q_tot is unbiased for that batch -- no literal ratio or
     matrix inverse anywhere in the loss.

Only the shear layer trains; the bulk (early, unconditional) layers are
loaded from an already-trained checkpoint and frozen throughout.

Usage:
    cd bfd_cnf
    BFD_TRAIN_FITS=<fits> PYTHONPATH=. python dev/train_shear_bijection_bias_zero.py \\
        --bulk-prior flows/prior_flow_perturbative_v4_long.eqx \\
        --out flows/prior_flow_bias_zero_shear.eqx --steps 20000
"""
from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from bfd_cnf.config import TRAIN_FITS_PATH
from bfd_cnf.data import load_training_dataset
from bfd_cnf.models.bijections import load_stats
from bfd_cnf.models.flows import build_flows, shear, _find_shear_taylor_layer


def _build_trainable_mask(flow):
    """Boolean pytree, same structure as `flow`: True only for the shear
    layer's own INEXACT-ARRAY leaves (excluding e.g. the activation function
    stored alongside the Linear layers, which isn't an array and would break
    optax's zeros_like), False everywhere else (the frozen bulk)."""
    mask = jax.tree_util.tree_map(lambda x: False, flow)
    shear_layer = _find_shear_taylor_layer(flow)
    shear_mask = jax.tree_util.tree_map(eqx.is_inexact_array, shear_layer)
    return eqx.tree_at(lambda f: _find_shear_taylor_layer(f), mask, shear_mask)


def make_loss(N, batch_size, raw2standard, g_max):
    def logp_of_g(g, m_std_i, flow):
        cond = jnp.concatenate([g, jnp.zeros(3)])
        return flow.log_prob(m_std_i, condition=cond)

    def per_example_qr(m_std_i, flow):
        f = lambda g: logp_of_g(g, m_std_i, flow)
        Q = jax.grad(f)(jnp.zeros(2))
        R = -jax.hessian(f)(jnp.zeros(2))
        return Q, R

    def arm_residual(flow, m_raw, dg, d2g, g_true):
        g_arr = jnp.reshape(g_true, (1, 1, 2))
        m_obs = shear(m_raw, g_arr, dg, d2g)[:, 0, :]
        m_std = jax.vmap(lambda m: raw2standard.transform_and_log_det(m)[0])(m_obs)
        Q, R = jax.vmap(lambda m_i: per_example_qr(m_i, flow))(m_std)
        Q_tot, R_tot = jnp.mean(Q, axis=0), jnp.mean(R, axis=0)
        return Q_tot - R_tot @ g_true

    def loss_fn(flow, data_y, data_dg, data_d2g, key):
        k_idx, k_g, k_theta = jr.split(key, 3)
        idx = jr.choice(k_idx, N, shape=(batch_size,), replace=True)
        m_raw = data_y[idx]
        dg = data_dg[idx][:, :4]
        d2g = data_d2g[idx][:, :4]

        r = jr.uniform(k_g, (), minval=0.0, maxval=g_max)
        theta = jr.uniform(k_theta, ())
        g = r * jnp.array([jnp.cos(theta), jnp.sin(theta)])

        resid_p = arm_residual(flow, m_raw, dg, d2g, g)
        resid_m = arm_residual(flow, m_raw, dg, d2g, -g)
        return jnp.sum(resid_p ** 2) + jnp.sum(resid_m ** 2)

    return loss_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk-prior", required=True, help="Checkpoint providing the frozen bulk.")
    ap.add_argument("--bulk-sigmax-layer", default="none")
    ap.add_argument("--bulk-shear-layer-kind", default="perturbative",
                     help="MUST match whatever --bulk-prior was actually trained with "
                          "(only used to build a matching pytree for deserialisation; "
                          "its own shear layer is discarded right after).")
    ap.add_argument("--bulk-shear-nn-width", type=int, default=32)
    ap.add_argument("--bulk-shear-nn-depth", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shear-layer-kind", default="perturbative",
                     help="'perturbative' (ShearPerturbative), 'taylor' (ShearTaylorLast), "
                          "or 'poly' (ExplicitPolyLast, the pre-Taylor architecture) -- "
                          "the FRESH layer actually being trained.")
    ap.add_argument("--shear-nn-width", type=int, default=32)
    ap.add_argument("--shear-nn-depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096,
                     help="Small batches (e.g. 256) let the moment-conditioned "
                          "coefficient net find per-batch shortcuts that zero the "
                          "residual on that batch's noisy Q/R estimate without "
                          "learning the true physical response -- confirmed "
                          "2026-08-07 (loss ~1e-7 but real-target m=-0.48). Use a "
                          "large batch so the per-step Q_tot/R_tot estimate is low "
                          "enough variance that this shortcut isn't available.")
    ap.add_argument("--g-max", type=float, default=0.05)
    ap.add_argument("--mf-min", type=float, default=1500.0)
    ap.add_argument("--mf-max", type=float, default=90000.0)
    ap.add_argument("--mrmf-min", type=float, default=2.2)
    ap.add_argument("--mrmf-max", type=float, default=3.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Training set: {TRAIN_FITS_PATH}")
    raw2standard = load_stats(args.bulk_prior)
    data = load_training_dataset(key=jr.key(0), subsample=1)

    m_np = np.asarray(data["moments_jnp"])
    mf, mrmf = m_np[:, 0], m_np[:, 1] / m_np[:, 0]
    keep = ((mf > args.mf_min) & (mf < args.mf_max)
            & (mrmf > args.mrmf_min) & (mrmf < args.mrmf_max))
    keep_idx = np.flatnonzero(keep)
    print(f"Restricting to target window: {keep_idx.size:,}/{m_np.shape[0]:,} templates")
    data = dict(data)
    for k in ("moments_jnp", "dm_dg_jnp", "d2m_dg2_jnp"):
        data[k] = jnp.asarray(np.asarray(data[k])[keep_idx])
    N = data["moments_jnp"].shape[0]

    # Bulk source: the already-trained checkpoint, used ONLY for its early
    # (bulk) layers -- its own shear layer is discarded and replaced fresh.
    bulk_source, _q = build_flows(
        jr.key(1), latent_dim=4, cond_dim=16, raw2standard=raw2standard,
        sigmax_layer_kind=args.bulk_sigmax_layer, shear_layer_kind=args.bulk_shear_layer_kind,
        prior_last_nn_width=args.bulk_shear_nn_width, prior_last_nn_depth=args.bulk_shear_nn_depth,
    )
    bulk_source = eqx.tree_deserialise_leaves(args.bulk_prior, bulk_source)

    # Fresh flow: SAME early-layer architecture width/depth as bulk_source (so
    # the leaves being swapped in below match shape), fresh shear layer at
    # whatever width/depth this run wants.
    flow, _q2 = build_flows(
        jr.key(2), latent_dim=4, cond_dim=16, raw2standard=raw2standard,
        sigmax_layer_kind=args.bulk_sigmax_layer, shear_layer_kind=args.shear_layer_kind,
        prior_early_nn_width=32, prior_early_nn_depth=2,  # must match bulk_source's early arch
        prior_last_nn_width=args.shear_nn_width, prior_last_nn_depth=args.shear_nn_depth,
    )
    # Swap in the trained early (bulk) layers from bulk_source; keep flow's
    # own freshly-initialised shear layer untouched.
    flow = eqx.tree_at(
        lambda f: f.bijection.bijection.early, flow, bulk_source.bijection.bijection.early
    )

    # ShearPerturbative zero-initialises its last layer (correct, for identity
    # at g=0) -- but that makes the residual EXACTLY zero at init, and for a
    # squared-residual loss d(r^2)/dW = 2r*(dr/dW) is then EXACTLY zero
    # regardless of dr/dW, trapping gradient descent at the origin forever
    # (confirmed empirically: loss stayed at exactly 0.0 for 100 steps).
    # Perturb slightly off zero to break this saddle point.
    shear_layer0 = _find_shear_taylor_layer(flow)
    leaves, treedef = jax.tree_util.tree_flatten(shear_layer0)
    keys = jr.split(jr.key(args.seed + 12345), len(leaves))
    leaves = [
        leaf + 0.01 * jr.normal(k, leaf.shape) if eqx.is_inexact_array(leaf) else leaf
        for leaf, k in zip(leaves, keys)
    ]
    perturbed = jax.tree_util.tree_unflatten(treedef, leaves)
    flow = eqx.tree_at(lambda f: _find_shear_taylor_layer(f), flow, perturbed)

    trainable_mask = _build_trainable_mask(flow)
    n_trainable = sum(int(np.prod(np.shape(np.asarray(l))))
                       for l in jax.tree_util.tree_leaves(eqx.filter(flow, trainable_mask)))
    print(f"Trainable (shear layer) params: {n_trainable:,}")

    loss_fn = make_loss(N, args.batch_size, raw2standard, args.g_max)

    opt = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adam(args.lr))
    opt_state = opt.init(eqx.filter(flow, trainable_mask))

    @eqx.filter_jit
    def step_fn(flow, opt_state, key, data_y, data_dg, data_d2g):
        loss_val, grads = eqx.filter_value_and_grad(
            lambda m: loss_fn(m, data_y, data_dg, data_d2g, key)
        )(flow)
        # Filter grads down to the trainable (shear-layer-only) structure so
        # they match the filtered opt_state -- the standard equinox pattern
        # for frozen parameters (eqx.filter(..., mask) puts None in place of
        # every non-trainable leaf; apply_updates then leaves those alone).
        filtered_grads = eqx.filter(grads, trainable_mask)
        updates, opt_state = opt.update(filtered_grads, opt_state, eqx.filter(flow, trainable_mask))
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss_val

    key = jr.key(args.seed)
    for step in range(args.steps):
        key, subkey = jr.split(key)
        flow, opt_state, loss_val = step_fn(
            flow, opt_state, subkey, data["moments_jnp"], data["dm_dg_jnp"], data["d2m_dg2_jnp"]
        )
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:6d}  loss={float(loss_val):.6e}")

    eqx.tree_serialise_leaves(args.out, flow)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
