"""check_c_ge2_isolated_gd.py
=============================
Most isolated possible test: freeze EVERY ShearPerturbative coefficient
except c_ge2 (held fixed at zero), and run gradient descent on c_ge2 ALONE
(a single scalar) against the real make_nll_loss objective, starting from 0,
using the SAME optimizer chain (grad-clip + Adam) and non-finite-skip guard
as real training's _make_train_step -- so a rare catastrophic batch (see the
step-0 spike investigation) can't derail this any more than it derails real
training.

If the formula and the loss are both correct and c_ge2 is genuinely
identifiable from this data, gradient descent on this single scalar should
converge to something close to the true population value of c_ge2
(measurable independently from the real Pqr derivatives -- the aniso-part
magnitude gives a direct per-galaxy estimate of c_ge2's true value, which we
average here for a ground-truth comparison point).

This removes ALL possible confounds from a 51k-parameter coefficient net,
other coefficients moving simultaneously, or the coefficient-magnitude
regularizer coupling across parameters -- it's the cleanest possible signal
check.

Usage:
    cd bfd_cnf
    BFD_TRAIN_FITS=<fits> PYTHONPATH=. python dev/check_c_ge2_isolated_gd.py
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
from bfd_cnf.models.flows import build_flows, make_nll_loss, _find_shear_taylor_layer


def _std_jac_hess(raw2standard, m):
    f = lambda mm: raw2standard.transform_and_log_det(mm)[0]
    return jax.jacfwd(f)(m)


def _true_c_ge2_estimate(raw2standard, m_all, dm_dg, idx):
    """Per-galaxy estimate of the true c_ge2 from the real Pqr derivatives:
    c_ge2 = (aniso_x*p + aniso_y*q) / (p^2+q^2), aniso=(dM1/dg1-dM2/dg2, 2*dM1/dg2)."""
    m = jnp.asarray(m_all[idx])
    A_raw = jnp.asarray(dm_dg[idx])[:, :4]

    def per_example(m_i, A_raw_i):
        J = _std_jac_hess(raw2standard, m_i)
        A_std = J @ A_raw_i
        x_std, _ = raw2standard.transform_and_log_det(m_i)
        return A_std, x_std

    A_std, x_std = jax.vmap(per_example)(m, A_raw)
    A_std, x_std = np.asarray(A_std), np.asarray(x_std)
    e1, e2 = x_std[:, 2], x_std[:, 3]
    p, q = e1 ** 2 - e2 ** 2, 2 * e1 * e2
    aniso_x = A_std[:, 2, 0] - A_std[:, 3, 1]
    aniso_y = 2.0 * A_std[:, 2, 1]
    denom = p * p + q * q
    c_ge2_est = (aniso_x * p + aniso_y * q) / (denom + 1e-12)
    weight = denom
    return np.average(c_ge2_est, weights=weight), np.median(c_ge2_est)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--shear-g-train-max", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Training set: {TRAIN_FITS_PATH}")
    data = load_training_dataset(key=jr.key(0), subsample=1)
    raw2standard = data["raw2standard"]
    m_all = np.asarray(data["moments_jnp"])
    dm_dg = np.asarray(data["dm_dg_jnp"])
    weights_np = np.asarray(data["weights"])

    sel = (m_all[:, 0] > 1500.0) & (m_all[:, 0] < 90000.0)
    sel_idx = np.flatnonzero(sel)
    rng = np.random.default_rng(1)
    p_samp = weights_np[sel_idx] / weights_np[sel_idx].sum()
    probe_idx = rng.choice(sel_idx, size=50000, replace=True, p=p_samp)
    true_mean, true_median = _true_c_ge2_estimate(raw2standard, m_all, dm_dg, probe_idx)
    print(f"\nTrue c_ge2 estimate from real Pqr derivatives: "
          f"weighted mean={true_mean:+.4f}  median={true_median:+.4f}")

    N = m_all.shape[0]
    w = jnp.asarray(weights_np / weights_np.sum())

    prior_flow, _q = build_flows(
        jr.key(0), latent_dim=4, cond_dim=16, raw2standard=raw2standard,
        sigmax_layer_kind="none", shear_layer_kind="perturbative",
        prior_last_nn_width=32, prior_last_nn_depth=1,
    )
    shear_layer = _find_shear_taylor_layer(prior_flow)

    # Freeze everything: zero out net_coeffs entirely (weights AND bias).
    net = shear_layer.net_coeffs
    zeroed_layers = []
    for layer in net.layers:
        if hasattr(layer, "weight"):
            layer = eqx.tree_at(lambda l: l.weight, layer, jnp.zeros_like(layer.weight))
            layer = eqx.tree_at(lambda l: l.bias, layer, jnp.zeros_like(layer.bias))
        zeroed_layers.append(layer)
    net = eqx.tree_at(lambda n: n.layers, net, tuple(zeroed_layers))
    prior_flow = eqx.tree_at(
        lambda p: _find_shear_taylor_layer(p).net_coeffs, prior_flow, net
    )

    # c_ge2 lives at index 7 of the last linear layer's bias (see coeffs() order:
    # cA_f,c0_f,c1_f, cA_s,c0_s,c1_s, c_g,c_ge2,c_g2,c_g2b).
    def set_c_ge2(flow, value):
        shear = _find_shear_taylor_layer(flow)
        last_bias = shear.net_coeffs.layers[-1].bias
        new_bias = last_bias.at[7].set(value)
        new_net = eqx.tree_at(lambda n: n.layers[-1].bias, shear.net_coeffs, new_bias)
        return eqx.tree_at(lambda p: _find_shear_taylor_layer(p).net_coeffs, flow, new_net)

    nll_loss = make_nll_loss(
        N=N, batch_size=args.batch_size, weights=w,
        log_scale_range=None, e_max=0.0, use_sx=False,
        raw2standard=raw2standard,
        sobolev_g1_weight=0.0, sobolev_g2_weight=0.0,
        shear_g_train_max=args.shear_g_train_max,
        shear_coeff_reg_weight=0.0,
    )

    # Same optimizer chain as real training's _make_train_step (grad-clip + Adam),
    # and the same non-finite-skip guard, applied to the single scalar c_ge2.
    opt = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adam(args.lr))

    @jax.jit
    def step_fn(c_val, opt_state, key, data_y, data_Sigma, data_dg, data_d2g, data_X):
        loss_val, grad = jax.value_and_grad(lambda c: nll_loss(
            set_c_ge2(prior_flow, c), data_y, data_Sigma, data_dg, data_d2g, data_X, key
        ))(c_val)
        updates, new_opt_state = opt.update(grad, opt_state)
        new_c = optax.apply_updates(c_val, updates)
        is_finite = jnp.isfinite(loss_val) & jnp.isfinite(grad)
        c_val = jnp.where(is_finite, new_c, c_val)
        opt_state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(is_finite, a, b) if eqx.is_inexact_array(a) else a,
            new_opt_state, opt_state,
        )
        return c_val, opt_state, loss_val

    c_val = jnp.array(0.0)
    opt_state = opt.init(c_val)
    key = jr.key(args.seed)
    for step in range(args.steps):
        key, subkey = jr.split(key)
        c_val, opt_state, loss_val = step_fn(
            c_val, opt_state, subkey, data["moments_jnp"], data["cov_jnp"],
            data["dm_dg_jnp"], data["d2m_dg2_jnp"], data["centroid_moments_jnp"],
        )
        if step % 200 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  c_ge2={float(c_val):+.4f}  loss={float(loss_val):.5f}")

    print(f"\nFinal c_ge2 from isolated gradient descent: {float(c_val):+.4f}")
    print(f"True c_ge2 (from real Pqr derivatives, weighted mean): {true_mean:+.4f}")


if __name__ == "__main__":
    main()
