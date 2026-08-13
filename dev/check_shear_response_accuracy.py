"""check_shear_response_accuracy.py
====================================
Strict accuracy check for a ShearPerturbative-trained prior flow: compares the
flow's own per-galaxy shear response (dm_std/dg, d2m_std/dg2, from
shear_derivs_generative) against the EXACT analytic per-template response
(dm_dg/d2m_dg2 -- real bfd Pqr derivatives, exact for this elliptical-Gaussian
population), restricted to the flux/Mr-Mf SELECTION region actually used for
target detection. This is what the BFD estimator's Q=grad_g P(M|g) ultimately
depends on: getting the per-galaxy response wrong in the selection region
directly biases Q, R, and hence the inferred shear.

The generic "does g=0/g!=0 map to a unit normal" check (see
check_perturbative_calibration.py) is a much weaker, aggregate-population
test; the requirement here is per-channel fractional error < 1% in the
selection region, matching the precision BFD needs for grad_g P(M|g).

The exact per-template derivative is in RAW [Mf,Mr,M1,M2] units; the flow's
own response is in the STANDARDISED coordinate. We convert the raw truth into
standardised units via the exact chain rule through raw2standard's own
Jacobian/Hessian (autodiff), the same composition ShearTaylorLast's
shear_derivs_generative docstring describes for combining with a downstream
layer -- here there IS no downstream SigmaX layer (sigmax_layer_kind=none),
so the shear layer's own shear_derivs_generative already IS the full
data-space (standardised) response, nothing further to compose.

Usage:
    cd bfd_cnf
    BFD_TRAIN_FITS=<same fits used to train> PYTHONPATH=. python dev/check_shear_response_accuracy.py \\
        --prior flows/prior_flow_perturbative_noshift_noiseless.eqx
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
    """Jacobian (4,4) and Hessian (4,4,4) of x_std=f(m) at raw moment m (4,)."""
    f = lambda mm: raw2standard.transform_and_log_det(mm)[0]
    J = jax.jacfwd(f)(m)
    H = jax.jacfwd(jax.jacfwd(f))(m)
    return J, H


def _raw_to_std_response(raw2standard, m, A_raw, B_raw):
    """Chain-rule A_raw (4,2), B_raw (4,2,2) (raw units) -> standardised units."""
    J, H = _std_jac_hess(raw2standard, m)
    A_std = J @ A_raw
    B_std = jnp.einsum("ij,jab->iab", J, B_raw) + jnp.einsum(
        "ijk,ja,kb->iab", H, A_raw, A_raw
    )
    return A_std, B_std


def _frac_err(pred, true, eps):
    return np.abs(pred - true) / (np.abs(true) + eps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--mf-min", type=float, default=1500.0, help="Selection-region flux floor.")
    ap.add_argument("--mf-max", type=float, default=90000.0, help="Selection-region flux ceiling.")
    ap.add_argument("--n", type=int, default=20000, help="Number of held-out selection-region galaxies.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--shear-nn-width", type=int, default=32,
                     help="Must match the checkpoint's --shear-nn-width at train time.")
    ap.add_argument("--shear-nn-depth", type=int, default=4,
                     help="Must match the checkpoint's --shear-nn-depth at train time.")
    ap.add_argument("--eps-frac", type=float, default=1e-3,
                     help="Additive floor in the fractional-error denominator, in units of "
                          "the channel's own typical |A_true|/|B_true| scale -- avoids "
                          "division blowups at near-zero true response without hiding real error.")
    args = ap.parse_args()

    print(f"Training set (for a held-out-style sample): {TRAIN_FITS_PATH}")
    raw2standard = load_stats(args.prior)
    data = load_training_dataset(key=jr.key(args.seed), subsample=1)
    moments_jnp = data["moments_jnp"]
    dm_dg_jnp = data["dm_dg_jnp"]
    d2m_dg2_jnp = data["d2m_dg2_jnp"]
    weights = np.asarray(data["weights"])
    N = moments_jnp.shape[0]

    m_all = np.asarray(moments_jnp)
    sel = (m_all[:, 0] > args.mf_min) & (m_all[:, 0] < args.mf_max)
    print(f"Selection region ({args.mf_min:g} < Mf < {args.mf_max:g}): "
          f"{sel.sum():,}/{N:,} templates pass")

    rng = np.random.default_rng(args.seed + 1)
    p = weights[sel] / weights[sel].sum()
    sel_idx = np.flatnonzero(sel)
    idx = rng.choice(sel_idx, size=min(args.n, sel_idx.size), replace=True, p=p)

    m = jnp.asarray(m_all[idx])                                    # (n, 4) raw
    A_raw = jnp.asarray(np.asarray(dm_dg_jnp)[idx])[:, :4]          # (n, 4, 2)
    B_raw = jnp.asarray(np.asarray(d2m_dg2_jnp)[idx])[:, :4]        # (n, 4, 2, 2)

    prior_flow, _q = build_flows(
        jr.key(0), latent_dim=4, cond_dim=16, raw2standard=raw2standard,
        sigmax_layer_kind="none", shear_layer_kind="perturbative",
        prior_last_nn_width=args.shear_nn_width, prior_last_nn_depth=args.shear_nn_depth,
    )
    prior_flow = eqx.tree_deserialise_leaves(args.prior, prior_flow)
    shear_layer = _find_shear_taylor_layer(prior_flow)
    if shear_layer is None:
        raise RuntimeError("No shear layer found in the deserialised flow.")

    def per_example(m_i, A_raw_i, B_raw_i):
        x_std, _ = raw2standard.transform_and_log_det(m_i)
        A_true, B_true = _raw_to_std_response(raw2standard, m_i, A_raw_i, B_raw_i)
        A_flow, B_flow = shear_layer.shear_derivs_generative(x_std)
        return A_true, B_true, A_flow, B_flow

    A_true, B_true, A_flow, B_flow = jax.vmap(per_example)(m, A_raw, B_raw)
    A_true, B_true, A_flow, B_flow = map(np.asarray, (A_true, B_true, A_flow, B_flow))

    chans = ["Mf", "Mr", "M1", "M2"]
    print(f"\n-- A = dm_std/dg (n={len(idx):,}) --")
    for c in range(4):
        typical = np.median(np.abs(A_true[:, c, :]))
        eps = args.eps_frac * max(typical, 1e-12)
        fe = _frac_err(A_flow[:, c, :], A_true[:, c, :], eps)
        print(f"  {chans[c]:>3s}: median frac err = {np.median(fe)*100:6.2f}%   "
              f"90th pct = {np.percentile(fe, 90)*100:6.2f}%   "
              f"(typical |A_true| = {typical:.4g})")

    print(f"\n-- B = d2m_std/dg2 (n={len(idx):,}) --")
    for c in range(4):
        typical = np.median(np.abs(B_true[:, c, :, :]))
        eps = args.eps_frac * max(typical, 1e-12)
        fe = _frac_err(B_flow[:, c, :, :], B_true[:, c, :, :], eps)
        print(f"  {chans[c]:>3s}: median frac err = {np.median(fe)*100:6.2f}%   "
              f"90th pct = {np.percentile(fe, 90)*100:6.2f}%   "
              f"(typical |B_true| = {typical:.4g})")

    print("\nTarget: <1% median fractional error per channel (BFD's grad_g P(M|g) precision "
          "requirement in the selection region).")


if __name__ == "__main__":
    main()
