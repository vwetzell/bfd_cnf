"""check_generative_formula.py
==============================
ShearPerturbative.shear_derivs_generative computes A_gen = -A (simple
negation of the encode-direction Jacobian, no further correction) and
B_gen = -B + correction. This is only exactly correct if d(shift)/dx = 0 at
the evaluation point (true at init, NOT generally true once coefficients
have moved away from zero, since shift depends on x through BOTH the fixed
tensors (e.g. e1,e2) AND the coefficient net's own dependence on x). The
implicit function theorem for m(g) solving m(g) + shift(m(g),g) = u (u
fixed) gives dm/dg = -(I + d(shift)/dx)^{-1} A, not just -A.

This script computes the generative response TWO ways and compares:
  1. shear_derivs_generative(x) -- the hand-derived formula in the class.
  2. Direct autodiff of the layer's own inverse_and_log_det (the actual code
     path used for sampling/generation) at a FIXED base point u, varying g --
     immune to any bug in the hand-derived formula, since it's just autodiff
     of the real inverse map.

If these two disagree, (1) has a bug. Then check which one (if either)
matches the true analytic Pqr derivative better.

Usage:
    cd bfd_cnf
    BFD_TRAIN_FITS=<fits> PYTHONPATH=. python dev/check_generative_formula.py \\
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
from bfd_cnf.models.flows import build_flows, _find_shear_taylor_layer


def _std_jac_hess(raw2standard, m):
    f = lambda mm: raw2standard.transform_and_log_det(mm)[0]
    J = jax.jacfwd(f)(m)
    H = jax.jacfwd(jax.jacfwd(f))(m)
    return J, H


def _raw_to_std_response(raw2standard, m, A_raw, B_raw):
    J, H = _std_jac_hess(raw2standard, m)
    A_std = J @ A_raw
    B_std = jnp.einsum("ij,jab->iab", J, B_raw) + jnp.einsum(
        "ijk,ja,kb->iab", H, A_raw, A_raw
    )
    return A_std, B_std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--shear-nn-width", type=int, default=32)
    ap.add_argument("--shear-nn-depth", type=int, default=4)
    ap.add_argument("--mf-min", type=float, default=1500.0)
    ap.add_argument("--mf-max", type=float, default=90000.0)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    print(f"Training set: {TRAIN_FITS_PATH}")
    raw2standard = load_stats(args.prior)
    data = load_training_dataset(key=jr.key(args.seed), subsample=1)
    m_all = np.asarray(data["moments_jnp"])
    dm_dg = np.asarray(data["dm_dg_jnp"])
    d2m_dg2 = np.asarray(data["d2m_dg2_jnp"])
    weights = np.asarray(data["weights"])

    sel = (m_all[:, 0] > args.mf_min) & (m_all[:, 0] < args.mf_max)
    sel_idx = np.flatnonzero(sel)
    rng = np.random.default_rng(args.seed + 1)
    p = weights[sel_idx] / weights[sel_idx].sum()
    idx = rng.choice(sel_idx, size=min(args.n, sel_idx.size), replace=True, p=p)

    m = jnp.asarray(m_all[idx])
    A_raw = jnp.asarray(dm_dg[idx])[:, :4]
    B_raw = jnp.asarray(d2m_dg2[idx])[:, :4]

    prior_flow, _q = build_flows(
        jr.key(0), latent_dim=4, cond_dim=16, raw2standard=raw2standard,
        sigmax_layer_kind="none", shear_layer_kind="perturbative",
        prior_last_nn_width=args.shear_nn_width, prior_last_nn_depth=args.shear_nn_depth,
    )
    prior_flow = eqx.tree_deserialise_leaves(args.prior, prior_flow)
    shear_layer = _find_shear_taylor_layer(prior_flow)

    def per_example(m_i, A_raw_i, B_raw_i):
        x_std0, _ = raw2standard.transform_and_log_det(m_i)
        A_true, B_true = _raw_to_std_response(raw2standard, m_i, A_raw_i, B_raw_i)

        # (1) hand-derived formula
        A_hand, B_hand = shear_layer.shear_derivs_generative(x_std0)

        # (2) direct autodiff of the ACTUAL inverse map at fixed base point
        def m_of_g(g):
            cond = jnp.array([g[0], g[1], 0.0, 0.0, 0.0])
            return shear_layer.inverse_and_log_det(x_std0, cond)[0]

        A_direct = jax.jacfwd(m_of_g)(jnp.zeros(2))
        B_direct = jax.jacfwd(jax.jacfwd(m_of_g))(jnp.zeros(2))

        return A_true, B_true, A_hand, B_hand, A_direct, B_direct

    A_true, B_true, A_hand, B_hand, A_direct, B_direct = jax.vmap(per_example)(m, A_raw, B_raw)
    A_true, A_hand, A_direct = map(np.asarray, (A_true, A_hand, A_direct))
    B_true, B_hand, B_direct = map(np.asarray, (B_true, B_hand, B_direct))

    print(f"\n-- Does shear_derivs_generative (hand formula) match direct autodiff "
          f"of the real inverse map? (n={len(idx):,}) --")
    diff_A = np.abs(A_hand - A_direct)
    diff_B = np.abs(B_hand - B_direct)
    print(f"  max |A_hand - A_direct| = {diff_A.max():.3e}   "
          f"(0 means the formula is exactly right, at least to 1st order)")
    print(f"  max |B_hand - B_direct| = {diff_B.max():.3e}")

    chans = ["Mf", "Mr", "M1", "M2"]
    print(f"\n-- Correlation of EACH candidate A with TRUTH, per channel/component --")
    for c in range(4):
        for k in range(2):
            t = A_true[:, c, k]
            corr_hand = np.corrcoef(t, A_hand[:, c, k])[0, 1]
            corr_direct = np.corrcoef(t, A_direct[:, c, k])[0, 1]
            print(f"  {chans[c]}, g{k+1}: corr(hand,true)={corr_hand:+.3f}   "
                  f"corr(direct,true)={corr_direct:+.3f}")


if __name__ == "__main__":
    main()
