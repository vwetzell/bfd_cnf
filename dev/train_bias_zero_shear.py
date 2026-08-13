"""train_bias_zero_shear.py
===========================
A from-scratch shear-response model, trained to directly zero the estimator's
bias rather than match derivatives or fit a conditional density.

Model (exactly the user's formula):
    P(m|g) = P(m) + A(m).g + 0.5 g^T B(m) g
  - P(m): the ALREADY-TRAINED bulk (unconditional) flow density, FROZEN. Since
    the shear bijection layer is identity at g=0 by construction, this is just
    --prior's log_prob evaluated at condition=[0,0,...] -- no retraining needed.
  - A(m) (2,), B(m) (2,2, symmetric): a small, plain MLP. No bijection, no
    log-det, no equivariance constraint -- just direct free outputs of the
    standardised moment.

Loss (mathematically equivalent to zero bias, without ever computing a ratio
or matrix inverse in the loss itself): the BFD ML shear estimator is
    g_hat = R_tot^{-1} Q_tot,   Q_tot = sum_i A(m_i)/P(m_i),
                                R_tot = sum_i [A(m_i)A(m_i)^T/P(m_i)^2 - B(m_i)/P(m_i)]
(eq. 16-17 of Bernstein et al. 2016, using P(m|g)'s own additive Taylor
coefficients as Q_i=A(m_i), R_i=B(m_i) -- NOT log-density derivatives, since
here P(m|g) is modelled directly, not as an exponential).
g_hat = g  <=>  Q_tot = R_tot . g. So each training step:
  1. draws a batch of REAL templates (sampled prop to nda*detj, the correct
     BFD population weight),
  2. shears them ON THE FLY to a randomly drawn g (and independently to -g,
     reusing the SAME galaxies -- shape-noise-cancelling, matches
     imsims.paired_bias's paired design) using each template's own EXACT
     analytic Pqr derivatives (dm_dg, d2m_dg2 -- exact for this Gaussian
     population, not an approximation we're fitting),
  3. forms Q_tot, R_tot for each arm from the CURRENT A, B,
  4. loss = ||Q_tot_plus - R_tot_plus @ g||^2 + ||Q_tot_minus - R_tot_minus @ (-g)||^2.

No NLL, no Sobolev, no log-determinant anywhere in this file.

Usage:
    cd bfd_cnf
    BFD_TRAIN_FITS=<fits> PYTHONPATH=. python dev/train_bias_zero_shear.py \\
        --bulk-prior flows/prior_flow_perturbative_v4_long.eqx \\
        --bulk-shear-nn-width 32 --bulk-shear-nn-depth 4 \\
        --out flows/shear_ab.eqx --steps 20000
"""
from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from bfd_cnf.config import TRAIN_FITS_PATH
from bfd_cnf.data import load_training_dataset
from bfd_cnf.models.bijections import CoeffNet, RawMomentStandardize, load_stats
from bfd_cnf.models.flows import build_flows, shear


class ShearAB(eqx.Module):
    """Free A(m) (2,), B(m) (2,2 symmetric) -- no bijection, no log-det."""

    net: CoeffNet

    def __init__(self, key, width=32, depth=3, activation=jnn.silu):
        self.net = CoeffNet(key, 4, 5, width, depth, activation)

    def __call__(self, m_std):
        raw = self.net(m_std)
        A = raw[:2]
        B = jnp.array([[raw[2], raw[3]], [raw[3], raw[4]]])
        return A, B


def _frozen_bulk_logp_fn(bulk_prior):
    """P(m) = exp(bulk_prior.log_prob(m_std, g=0)) -- always evaluated at g=0
    condition, regardless of what g the moment was actually sheared to; the
    shear response is now entirely A(m), B(m)'s job, not this flow's."""
    zero_cond = jnp.zeros(5)

    def logp(m_std):
        return bulk_prior.log_prob(m_std, condition=zero_cond)

    return logp


def make_bias_zero_loss(N, batch_size, raw2standard, bulk_prior, g_max, p_floor):
    """No nda*detj importance weighting here: this loss directly sums over
    discrete observed galaxies (BFD eq 16-17), which wants sky-density
    weighting (nda) alone -- detj was only ever a correction for the OLD
    NLL/density-fitting objective's own Jacobian bookkeeping, not applicable
    to this direct population sum. For no-shift templates nda=1 uniformly
    anyway, so plain uniform sampling is exactly right here. Uniform sampling
    also avoids nda*detj's correlation with the flow's worst-calibrated
    (lowest-P) tail, which made 1/P^2 blow up under importance sampling."""
    logp_fn = _frozen_bulk_logp_fn(bulk_prior)

    def arm_qr(shear_ab, m_raw, dg, d2g, g):
        """Q_tot, R_tot (batch MEAN, not sum -- scale-stable) for one arm."""
        g_arr = jnp.array([[[g[0], g[1]]]])  # (1,1,2), matches shear()'s shape
        m_sheared = shear(m_raw, g_arr, dg, d2g)[:, 0, :]  # (B, 4)
        m_std = jax.vmap(lambda m: raw2standard.transform_and_log_det(m)[0])(m_sheared)
        P = jnp.exp(jax.vmap(logp_fn)(m_std))
        P = jnp.maximum(P, p_floor)
        A, B = jax.vmap(shear_ab)(m_std)  # (B,2), (B,2,2)
        Q_i = A / P[:, None]
        R_i = jnp.einsum("bi,bj->bij", A, A) / (P ** 2)[:, None, None] - B / P[:, None, None]
        return jnp.mean(Q_i, axis=0), jnp.mean(R_i, axis=0)

    def psd_penalty(R):
        """R_tot = -E[hess_g log P] MUST be PSD for a sane Fisher-information
        estimator (g_hat = R_tot^{-1} Q_tot is meaningless otherwise -- a real
        problem found empirically: R_tot ended up with a NEGATIVE eigenvalue
        with no penalty at all, since nothing about the residual loss
        ||Q_tot-R_tot@g||^2 constrains A(m), B(m) to correspond to a locally
        CONCAVE log P(m|g)). Penalise negative eigenvalues softly."""
        eigvals = jnp.linalg.eigvalsh(R)
        return jnp.sum(jax.nn.relu(-eigvals) ** 2)

    def loss_fn(shear_ab, data_y, data_dg, data_d2g, key, psd_weight):
        k_idx, k_g, k_theta = jr.split(key, 3)
        idx = jr.choice(k_idx, N, shape=(batch_size,), replace=True)
        m_raw = data_y[idx]
        dg = data_dg[idx][:, :4]
        d2g = data_d2g[idx][:, :4]

        r = jr.uniform(k_g, (), minval=0.0, maxval=g_max)
        theta = jr.uniform(k_theta, ())
        g = r * jnp.array([jnp.cos(theta), jnp.sin(theta)])

        Qp, Rp = arm_qr(shear_ab, m_raw, dg, d2g, g)
        Qm, Rm = arm_qr(shear_ab, m_raw, dg, d2g, -g)
        resid_p = Qp - Rp @ g
        resid_m = Qm - Rm @ (-g)
        resid_loss = jnp.sum(resid_p ** 2) + jnp.sum(resid_m ** 2)
        psd_loss = psd_penalty(Rp) + psd_penalty(Rm)
        return resid_loss + psd_weight * psd_loss

    return loss_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk-prior", required=True, help="Checkpoint providing the frozen P(m).")
    ap.add_argument("--bulk-sigmax-layer", default="none")
    ap.add_argument("--bulk-shear-layer", default="perturbative")
    ap.add_argument("--bulk-shear-nn-width", type=int, default=32)
    ap.add_argument("--bulk-shear-nn-depth", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ab-width", type=int, default=32)
    ap.add_argument("--ab-depth", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--g-max", type=float, default=0.05, help="Physical shear range -- no need to widen, this loss has a direct regression signal.")
    ap.add_argument("--p-floor", type=float, default=1e-3,
                     help="Floor on P(m) before dividing. Even restricted to the target "
                          "window, P still spans ~6 orders of magnitude (a continuous "
                          "population density genuinely does this -- unlike real BFD's "
                          "per-template noise-likelihood P_i, which stays near its own "
                          "peak for a well-detected object), so 1/P^2 in R_tot is still "
                          "unstable without a floor. Since we only need the bias right in "
                          "the window (not the tails this floor most affects), the "
                          "resulting Fisher-weighting bias is an acceptable trade for "
                          "stability -- confirmed empirically: 1e-6 still gave loss curves "
                          "of 1e13->1e6 that never settled; 1e-3 settles to O(1) in <200 "
                          "steps.")
    ap.add_argument("--mf-min", type=float, default=1500.0,
                     help="Target selection window (only need the bias correct here, "
                          "plus a margin for the selection-term calculation).")
    ap.add_argument("--mf-max", type=float, default=90000.0)
    ap.add_argument("--mrmf-min", type=float, default=2.2)
    ap.add_argument("--mrmf-max", type=float, default=3.5)
    ap.add_argument("--psd-weight", type=float, default=1.0,
                     help="Soft penalty weight keeping R_tot's eigenvalues non-negative "
                          "(see psd_penalty docstring -- without this, R_tot can end up "
                          "indefinite, making g_hat=R_tot^{-1}Q_tot meaningless even "
                          "though the raw residual loss looks small and stable).")
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
    print(f"Restricting training to target window + margin "
          f"(Mf in [{args.mf_min:g},{args.mf_max:g}], Mr/Mf in [{args.mrmf_min:g},{args.mrmf_max:g}]): "
          f"{keep_idx.size:,}/{m_np.shape[0]:,} templates")
    data = dict(data)
    for k in ("moments_jnp", "dm_dg_jnp", "d2m_dg2_jnp"):
        data[k] = jnp.asarray(np.asarray(data[k])[keep_idx])
    N = data["moments_jnp"].shape[0]

    bulk_prior, _q = build_flows(
        jr.key(1), latent_dim=4, cond_dim=16, raw2standard=raw2standard,
        sigmax_layer_kind=args.bulk_sigmax_layer, shear_layer_kind=args.bulk_shear_layer,
        prior_last_nn_width=args.bulk_shear_nn_width, prior_last_nn_depth=args.bulk_shear_nn_depth,
    )
    bulk_prior = eqx.tree_deserialise_leaves(args.bulk_prior, bulk_prior)
    # bulk_prior is closed over inside loss_fn, not passed as the diff argument
    # to eqx.filter_value_and_grad below, so it never receives a gradient --
    # frozen by construction, no stop_gradient needed.

    shear_ab = ShearAB(jr.key(2), width=args.ab_width, depth=args.ab_depth)

    loss_fn = make_bias_zero_loss(
        N, args.batch_size, raw2standard, bulk_prior, args.g_max, args.p_floor
    )

    opt = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adam(args.lr))
    opt_state = opt.init(eqx.filter(shear_ab, eqx.is_inexact_array))

    @eqx.filter_jit
    def step_fn(shear_ab, opt_state, key, data_y, data_dg, data_d2g):
        loss_val, grads = eqx.filter_value_and_grad(
            lambda m: loss_fn(m, data_y, data_dg, data_d2g, key, args.psd_weight)
        )(shear_ab)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(shear_ab, eqx.is_inexact_array))
        shear_ab = eqx.apply_updates(shear_ab, updates)
        return shear_ab, opt_state, loss_val

    key = jr.key(args.seed)
    for step in range(args.steps):
        key, subkey = jr.split(key)
        shear_ab, opt_state, loss_val = step_fn(
            shear_ab, opt_state, subkey, data["moments_jnp"], data["dm_dg_jnp"], data["d2m_dg2_jnp"]
        )
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:6d}  loss={float(loss_val):.6e}")

    eqx.tree_serialise_leaves(args.out, shear_ab)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
