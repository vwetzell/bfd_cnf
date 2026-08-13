"""check_bias_zero_in_distribution.py
====================================
Cheapest possible diagnostic for a trained bias-zero shear layer: is it even
SELF-CONSISTENT on FRESH, unseen batches of its OWN training population
(window-restricted templates, on-the-fly resheared via their exact analytic
dm_dg/d2m_dg2 -- the exact same recipe train_shear_bijection_bias_zero.py
used), at g values spanning the training range?

If m is close to 0 here but badly biased on REAL targets
(point_estimate_pqr_noiseless.py), the problem is POPULATION MISMATCH between
the training templates and the real target population.

If m is ALREADY badly biased here (in-distribution, on data literally drawn
from the training recipe, at g's it was trained across), the bias-zero loss
itself is under-constrained: since it uses a BATCH MEAN Q_tot/R_tot at only a
single random g_true per step, it only forces the AGGREGATE (population-mean)
response to be correct at whatever g's were sampled -- individual per-galaxy
Q(x)/R(x) can be redistributed arbitrarily across moment-space as long as the
batch mean comes out right, and a single random batch's empirical mean isn't
the true population mean. This script uses MUCH bigger held-out batches
(default 200k per arm) than any single training step (4096) specifically to
distinguish "the population-level fit is right" from "one step's noisy batch
mean was zeroed."

Usage:
    cd bfd_cnf
    BFD_TRAIN_FITS=<fits> PYTHONPATH=. python dev/check_bias_zero_in_distribution.py \\
        --prior flows/prior_flow_bias_zero_shear_bs4096.eqx --shear-layer perturbative
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
from bfd_cnf.models.flows import build_flows, shear


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--sigmax-layer", default="none")
    ap.add_argument("--shear-layer", default="perturbative")
    ap.add_argument("--shear-nn-width", type=int, default=32)
    ap.add_argument("--shear-nn-depth", type=int, default=4)
    ap.add_argument("--mf-min", type=float, default=1500.0)
    ap.add_argument("--mf-max", type=float, default=90000.0)
    ap.add_argument("--mrmf-min", type=float, default=2.2)
    ap.add_argument("--mrmf-max", type=float, default=3.5)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--g1", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=999)  # different from training seed=0
    args = ap.parse_args()

    print(f"Training set: {TRAIN_FITS_PATH}")
    raw2standard = load_stats(args.prior)
    data = load_training_dataset(key=jr.key(0), subsample=1)

    m_np = np.asarray(data["moments_jnp"])
    mf, mrmf = m_np[:, 0], m_np[:, 1] / m_np[:, 0]
    keep = ((mf > args.mf_min) & (mf < args.mf_max)
            & (mrmf > args.mrmf_min) & (mrmf < args.mrmf_max))
    keep_idx = np.flatnonzero(keep)
    print(f"Window: {keep_idx.size:,}/{m_np.shape[0]:,} templates")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(keep_idx, size=min(args.n, keep_idx.size), replace=False)
    m_raw = jnp.asarray(m_np[idx])
    dg = jnp.asarray(np.asarray(data["dm_dg_jnp"])[idx])[:, :4]
    d2g = jnp.asarray(np.asarray(data["d2m_dg2_jnp"])[idx])[:, :4]
    print(f"Held-out (fresh seed, not necessarily disjoint from training draws) "
          f"in-distribution sample: {m_raw.shape[0]:,}")

    flow, _ = build_flows(
        jr.key(1), latent_dim=4, cond_dim=16, raw2standard=raw2standard,
        sigmax_layer_kind=args.sigmax_layer, shear_layer_kind=args.shear_layer,
        prior_early_nn_width=32, prior_early_nn_depth=2,
        prior_last_nn_width=args.shear_nn_width, prior_last_nn_depth=args.shear_nn_depth,
    )
    flow = eqx.tree_deserialise_leaves(args.prior, flow)

    def logp_of_g(g, m_std_i):
        cond = jnp.concatenate([g, jnp.zeros(3)])
        return flow.log_prob(m_std_i, condition=cond)

    def per_example_qr(m_std_i):
        f = lambda g: logp_of_g(g, m_std_i)
        Q = jax.grad(f)(jnp.zeros(2))
        R = -jax.hessian(f)(jnp.zeros(2))
        return Q, R

    def g_hat_for(g_signed):
        g_arr = jnp.reshape(g_signed, (1, 1, 2))
        m_obs = shear(m_raw, g_arr, dg, d2g)[:, 0, :]
        m_std = jax.vmap(lambda m: raw2standard.transform_and_log_det(m)[0])(m_obs)
        Q, R = jax.lax.map(per_example_qr, m_std, batch_size=8192)
        Q_tot, R_tot = jnp.mean(Q, axis=0), jnp.mean(R, axis=0)
        g_hat = jnp.linalg.solve(R_tot, Q_tot)
        return g_hat, R_tot

    for r in (0.005, 0.01, 0.02, 0.03, 0.05):
        g = jnp.array([args.g1, 0.0]) * (r / args.g1)
        gp, Rp = g_hat_for(g)
        gm, Rm = g_hat_for(-g)
        m_bias = (gp[0] - gm[0]) / (2 * r) - 1.0
        eig_p = np.linalg.eigvalsh(np.asarray(Rp))
        eig_m = np.linalg.eigvalsh(np.asarray(Rm))
        print(f"|g|={r:.3f}  g_hat(+)={np.asarray(gp)}  g_hat(-)={np.asarray(gm)}  "
              f"m={float(m_bias):+.4f}  eig(R+)={eig_p}  eig(R-)={eig_m}")


if __name__ == "__main__":
    main()
