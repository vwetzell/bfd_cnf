"""check_shear_grad_snr.py
==========================
Direct diagnostic: is make_nll_loss's gradient w.r.t. ShearPerturbative's
net_coeffs a real, consistent signal, or is it dominated by batch/stencil
noise? Draws K independent gradients (fresh batch sample + fresh random g
stencil each draw, same as a real training step) at a FIXED set of flow
parameters, and reports the per-parameter signal-to-noise ratio:

    SNR = |mean_k(grad_k)| / std_k(grad_k)

SNR << 1 means the gradient direction is inconsistent draw-to-draw (pure
noise -- optimization has nothing real to climb). SNR >> 1 means there's a
real, resampling-stable direction (the earlier null results would then need
a different explanation -- e.g. optimizer/LR issues, not a signal-strength
problem).

Checked at TWO points: a fresh random init (coefficients ~0, matching the
very start of training) and the actual trained scaleup checkpoint (where
training plateaued) -- if the signal is real at init but vanishes at the
plateau point, that's consistent with the coefficients having already moved
to wherever the (real) signal points, just not to the true value. If the
signal is weak at BOTH points, that's consistent with the loss carrying
little real information about this parameter, at any point in the space
we've explored.

Usage:
    cd bfd_cnf
    BFD_TRAIN_FITS=<same fits used to train> PYTHONPATH=. python dev/check_shear_grad_snr.py \\
        --prior flows/prior_flow_perturbative_scaleup.eqx --shear-nn-width 128 --shear-nn-depth 4
"""
from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bfd_cnf.config import TRAIN_FITS_PATH
from bfd_cnf.data import load_training_dataset
from bfd_cnf.models.bijections import load_stats
from bfd_cnf.models.flows import build_flows, make_nll_loss, _find_shear_taylor_layer


def _shear_coeff_grad_vector(grads, prior_flow):
    """Flatten just the shear layer's net_coeffs gradient leaves into one vector."""
    shear_layer = _find_shear_taylor_layer(prior_flow)
    shear_grad_layer = _find_shear_taylor_layer(grads)
    leaves_ref = jax.tree_util.tree_leaves(shear_layer.net_coeffs)
    leaves_grad = jax.tree_util.tree_leaves(shear_grad_layer.net_coeffs)
    flat = [np.asarray(g).ravel() for g in leaves_grad if g is not None]
    return np.concatenate(flat) if flat else np.zeros(0)


def _snr_report(name: str, prior_flow, nll_loss, data, K: int, seed: int) -> None:
    grads_list = []
    for k in range(K):
        key = jr.key(seed + k)
        g = eqx.filter_grad(
            lambda p: nll_loss(p, data["moments_jnp"], data["cov_jnp"],
                               data["dm_dg_jnp"], data["d2m_dg2_jnp"], data["centroid_moments_jnp"], key)
        )(prior_flow)
        grads_list.append(_shear_coeff_grad_vector(g, prior_flow))
    G = np.stack(grads_list, axis=0)  # (K, n_params)
    mean = G.mean(0)
    std = G.std(0) + 1e-30
    snr = np.abs(mean) / std
    overall_snr = np.linalg.norm(mean) / np.sqrt(np.mean(std ** 2))
    print(f"\n-- {name} (K={K} independent batch+stencil draws, "
          f"{len(mean)} net_coeffs params) --")
    print(f"  |mean grad| = {np.linalg.norm(mean):.3e}   "
          f"rms(std across draws) = {np.sqrt(np.mean(std**2)):.3e}")
    print(f"  per-param SNR: median={np.median(snr):.3f}  90th pct={np.percentile(snr, 90):.3f}  "
          f"max={np.max(snr):.3f}")
    print(f"  aggregate SNR (|mean|/rms(std)) = {overall_snr:.3f}   "
          f"-> {'REAL SIGNAL' if overall_snr > 2 else 'NOISE-DOMINATED (<2)'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True, help="Trained checkpoint to test AT its plateau point.")
    ap.add_argument("--shear-nn-width", type=int, default=32)
    ap.add_argument("--shear-nn-depth", type=int, default=4)
    ap.add_argument("--shear-g-train-max", type=float, default=0.15)
    ap.add_argument("--shear-coeff-reg-weight", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=40, help="Number of independent gradient draws.")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    print(f"Training set: {TRAIN_FITS_PATH}")
    raw2standard = load_stats(args.prior)
    data = load_training_dataset(key=jr.key(0), subsample=1)
    N = data["moments_jnp"].shape[0]

    w_np = np.asarray(data["weights"])
    w_np = w_np / w_np.sum()
    weights = jnp.asarray(w_np)

    nll_loss = make_nll_loss(
        N=N, batch_size=args.batch_size, weights=weights,
        log_scale_range=None, e_max=0.0, use_sx=False,
        raw2standard=raw2standard,
        sobolev_g1_weight=0.0, sobolev_g2_weight=0.0,
        shear_g_train_max=args.shear_g_train_max,
        shear_coeff_reg_weight=args.shear_coeff_reg_weight,
    )

    # -- Point 1: fresh random init (coefficients ~0, start-of-training state) --
    fresh_flow, _q = build_flows(
        jr.key(1), latent_dim=4, cond_dim=16, raw2standard=raw2standard,
        sigmax_layer_kind="none", shear_layer_kind="perturbative",
        prior_last_nn_width=args.shear_nn_width, prior_last_nn_depth=args.shear_nn_depth,
    )
    _snr_report("fresh init (coeffs ~ 0)", fresh_flow, nll_loss, data, args.k, args.seed)

    # -- Point 2: the actual trained (plateaued) checkpoint --
    trained_flow, _q = build_flows(
        jr.key(2), latent_dim=4, cond_dim=16, raw2standard=raw2standard,
        sigmax_layer_kind="none", shear_layer_kind="perturbative",
        prior_last_nn_width=args.shear_nn_width, prior_last_nn_depth=args.shear_nn_depth,
    )
    trained_flow = eqx.tree_deserialise_leaves(args.prior, trained_flow)
    _snr_report("trained checkpoint (plateau point)", trained_flow, nll_loss, data, args.k, args.seed + 1000)


if __name__ == "__main__":
    main()
