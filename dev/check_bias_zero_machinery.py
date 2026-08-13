"""check_bias_zero_machinery.py
===============================
Unit test for the bias-zeroing estimator, using the EXACT analytic
per-template shear response (dm_dg, d2m_dg2 -- real bfd Pqr derivatives) in
place of any trained net, to isolate whether the ESTIMATION MACHINERY is
correct from whether a trained ShearAB actually learned a good response.

Rather than hand-deriving the chain rule from (exact moment response) and
(bulk score) into a (2,)-shaped A(m) -- error-prone, and the earlier version
of this script had a shape bug doing exactly that -- this reuses the same
safe, already-validated pattern as point_estimate_pqr_noiseless.py: build an
explicit function of g by RESHEARING the raw moment with the template's own
EXACT dm_dg/d2m_dg2 (a known closed form), feed the result through the
FROZEN bulk's log_prob, and autodiff the whole composition w.r.t. g directly.
No hand-derived algebra, no shape ambiguity.

    log P_i(g) := bulk.log_prob(standardize(shear(m0_i, g, dm_dg_i, d2m_dg2_i)))
    Q_tot = mean_i grad_g log P_i(g)|0
    R_tot = -mean_i hess_g log P_i(g)|0
    g_hat = solve(R_tot, Q_tot)

If g_hat matches g_true here, the aggregation machinery is correct and any
bias with the trained model is in what the net learned, not in how the
estimate is computed.

Usage:
    cd bfd_cnf
    BFD_TRAIN_FITS=<fits> PYTHONPATH=. python dev/check_bias_zero_machinery.py \\
        --bulk-prior flows/prior_flow_perturbative_v4_long.eqx
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
    ap.add_argument("--bulk-prior", required=True)
    ap.add_argument("--bulk-sigmax-layer", default="none")
    ap.add_argument("--bulk-shear-layer", default="perturbative")
    ap.add_argument("--bulk-shear-nn-width", type=int, default=32)
    ap.add_argument("--bulk-shear-nn-depth", type=int, default=4)
    ap.add_argument("--g", type=float, nargs=2, default=[0.02, 0.01])
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--mf-min", type=float, default=1500.0)
    ap.add_argument("--mf-max", type=float, default=90000.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Training set: {TRAIN_FITS_PATH}")
    raw2standard = load_stats(args.bulk_prior)
    data = load_training_dataset(key=jr.key(0), subsample=1)
    m_all = np.asarray(data["moments_jnp"])
    dm_dg = np.asarray(data["dm_dg_jnp"])
    d2m_dg2 = np.asarray(data["d2m_dg2_jnp"])

    sel = (m_all[:, 0] > args.mf_min) & (m_all[:, 0] < args.mf_max)
    sel_idx = np.flatnonzero(sel)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(sel_idx, size=min(args.n, sel_idx.size), replace=False)

    m0 = jnp.asarray(m_all[idx])            # (n, 4)
    dg = jnp.asarray(dm_dg[idx])[:, :4]     # (n, 4, 2)
    d2g = jnp.asarray(d2m_dg2[idx])[:, :4]  # (n, 4, 2, 2)

    bulk_prior, _q = build_flows(
        jr.key(1), latent_dim=4, cond_dim=16, raw2standard=raw2standard,
        sigmax_layer_kind=args.bulk_sigmax_layer, shear_layer_kind=args.bulk_shear_layer,
        prior_last_nn_width=args.bulk_shear_nn_width, prior_last_nn_depth=args.bulk_shear_nn_depth,
    )
    bulk_prior = eqx.tree_deserialise_leaves(args.bulk_prior, bulk_prior)
    zero_cond = jnp.zeros(5)

    def logp_of_g(g, m0_i, dg_i, d2g_i):
        g_arr = jnp.reshape(g, (1, 1, 2))
        m_g = shear(m0_i[None, :], g_arr, dg_i[None, :], d2g_i[None, :])[0, 0, :]
        x_std, _ = raw2standard.transform_and_log_det(m_g)
        return bulk_prior.log_prob(x_std, condition=zero_cond)

    def per_example(m0_i, dg_i, d2g_i, g_eval):
        f = lambda g: logp_of_g(g, m0_i, dg_i, d2g_i)
        Q = jax.grad(f)(g_eval)
        R = -jax.hessian(f)(g_eval)
        return Q, R

    def g_hat_for(g_eval, sign):
        g_signed = sign * jnp.asarray(args.g)
        Q, R = jax.vmap(lambda a, b, c: per_example(a, b, c, g_signed))(m0, dg, d2g)
        Q_tot, R_tot = jnp.mean(Q, axis=0), jnp.mean(R, axis=0)
        return jnp.linalg.solve(R_tot, Q_tot)

    g_hat_p = g_hat_for(args.g, +1.0)
    g_hat_m = g_hat_for(args.g, -1.0)

    print(f"\ng_hat(+) = {np.asarray(g_hat_p)}   (true = {np.asarray(args.g)})")
    print(f"g_hat(-) = {np.asarray(g_hat_m)}   (true = {-np.asarray(args.g)})")
    m_bias = (g_hat_p[0] - g_hat_m[0]) / (2 * args.g[0]) - 1.0
    print(f"\nm (exact analytic response, autodiff through frozen bulk, no trained net) "
          f"= {float(m_bias):+.5f}")
    print("If this is not close to 0, the bug is in the estimation machinery "
          "(or the bulk itself), not in what a trained net learned.")


if __name__ == "__main__":
    main()
