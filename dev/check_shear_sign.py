"""check_shear_sign.py
======================
Quick diagnostic: is the trained ShearPerturbative response CORRELATED with
truth (right direction, wrong scale/noise) or ANTI-correlated (sign flip
somewhere in the encode/decode or shear() convention)? The fractional-error
metric in check_shear_response_accuracy.py can't distinguish these -- a pure
sign flip (A_flow = -A_true) gives ~200% fractional error, same ballpark as
noise, but a correlation coefficient tells them apart immediately.

Usage:
    cd bfd_cnf
    BFD_TRAIN_FITS=<fits> PYTHONPATH=. python dev/check_shear_sign.py \\
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
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    print(f"Training set: {TRAIN_FITS_PATH}")
    raw2standard = load_stats(args.prior)
    data = load_training_dataset(key=jr.key(args.seed), subsample=1)
    m_all = np.asarray(data["moments_jnp"])
    dm_dg = np.asarray(data["dm_dg_jnp"])
    d2m_dg2 = np.asarray(data["d2m_dg2_jnp"])
    weights = np.asarray(data["weights"])
    N = m_all.shape[0]

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
        x_std, _ = raw2standard.transform_and_log_det(m_i)
        A_true, B_true = _raw_to_std_response(raw2standard, m_i, A_raw_i, B_raw_i)
        A_flow, B_flow = shear_layer.shear_derivs_generative(x_std)
        return A_true, B_true, A_flow, B_flow

    A_true, B_true, A_flow, B_flow = jax.vmap(per_example)(m, A_raw, B_raw)
    A_true, B_true, A_flow, B_flow = map(np.asarray, (A_true, B_true, A_flow, B_flow))

    chans = ["Mf", "Mr", "M1", "M2"]
    print(f"\n-- A correlation/scale (n={len(idx):,}) --")
    for c in range(4):
        for k in range(2):
            t, f = A_true[:, c, k], A_flow[:, c, k]
            corr = np.corrcoef(t, f)[0, 1]
            slope = np.sum(t * f) / np.sum(t * t)  # best-fit f = slope * t
            print(f"  {chans[c]}, g{k+1}: corr={corr:+.3f}  best-fit slope (flow=slope*true)={slope:+.3f}")

    print(f"\n-- B correlation/scale (diag only, n={len(idx):,}) --")
    for c in range(4):
        for (a, b) in [(0, 0), (1, 1), (0, 1)]:
            t, f = B_true[:, c, a, b], B_flow[:, c, a, b]
            corr = np.corrcoef(t, f)[0, 1]
            slope = np.sum(t * f) / np.sum(t * t)
            print(f"  {chans[c]}, ({a},{b}): corr={corr:+.3f}  slope={slope:+.3f}")


if __name__ == "__main__":
    main()
