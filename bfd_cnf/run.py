"""
run.py
======
End-to-end workflow script.  Calls functions from the other modules in the
correct order to reproduce the notebook analysis.

Usage
-----
    # from the repo root so the package import resolves:
    cd /home/vwetzell/gitrepos/bfd_cnf
    python -m bfd_cnf.run
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .config import (
    GRID_M_PATH,
    GRID_P_PATH,
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    Sigma0,
    e_max,
    log_scale_range,
    prior_flow_layers,
    q_flow_layers,
)

# ---------------------------------------------------------------------------
# Imports from this package
# ---------------------------------------------------------------------------
from .config import (
    key as _base_key,
)
from .data import load_training_dataset, transform_dataset_to_standard
from .inference import (
    _halton_sequence,
    integrate_catalog_pqr,
    make_flow_prob_and_derivs,
    prob_template,
    rqmc_integrate_pqr_jax,
    rqmc_pqr_grid,
    sample_trunc_mvn_noise_first_lower,
)
from .models.flows import batch_cholesky_of_sym, build_flows, cov2corr, cx_to_sx_cond
from .statistics import (
    bootstrap_independent_mult_bias,
    pqr2g,
)
from .training import (
    compute_std_stats,
    continue_training,
    load_or_train,
    save_models,
)
from .viz import (
    plot_flow_vs_obs_corner,
    plot_g_distributions,
    plot_Q_distributions,
    plot_R_distributions,
    plot_snr_histograms,
)


def main() -> None:
    """Run the full end-to-end BFD cNF workflow.

    Executes the following steps in order:

    1. Load and quality-filter the BFD FITS template table.
    2. Load or train the prior and q normalizing flows.
    3. Build flow-based PQR inference functions.
    4. Run a single-object RQMC PQR sanity check.
    5. Integrate the +/- shear grid catalogues independently (each selected on
       its own moments — see :func:`bfd_cnf.inference.integrate_catalog_pqr`).
    6. Estimate the per-object shear vector on each side.
    7. Bootstrap the multiplicative bias uncertainty (independent-ensemble).
    8. Produce diagnostic visualisations.
    """
    key = _base_key

    # -----------------------------------------------------------------------
    # 1. Load and pre-process data
    # -----------------------------------------------------------------------
    print("Loading data...")
    data = load_training_dataset(key=key)
    moments_jnp = data["moments_jnp"]
    centroid_moments_jnp = data["centroid_moments_jnp"]
    cov_jnp = data["cov_jnp"]
    dm_dg_jnp = data["dm_dg_jnp"]
    d2m_dg2_jnp = data["d2m_dg2_jnp"]
    weights = data["weights"]
    resampled = data["resampled"]
    data_mean = data["data_mean"]
    data_std = data["data_std"]
    raw2standard = data["raw2standard"]
    key = data["key"]

    # -----------------------------------------------------------------------
    # 2. Load or train flows
    # -----------------------------------------------------------------------
    import os as _os

    train_kwargs = dict(
        steps=20_000,
        learning_rate=1e-4,
        weight_decay=1e-5,
        grad_clip=0.5,
        log_scale_range=log_scale_range,
        e_max=e_max,
    )

    if _os.path.exists(PRIOR_FLOW_PATH) and _os.path.exists(Q_FLOW_PATH):
        print("Saved flow weights found — loading and continuing training...")
        prior_trained, q_trained, losses = load_or_train(
            key,
            moments_jnp, centroid_moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp,
            weights, raw2standard,
            prior_path=PRIOR_FLOW_PATH, q_path=Q_FLOW_PATH,
        )
        key, subkey = jr.split(key)
        prior_trained, q_trained, losses = continue_training(
            subkey,
            prior_trained, q_trained,
            moments_jnp, centroid_moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp,
            weights, raw2standard,
            **train_kwargs,
        )
    else:
        print("No saved weights found — training from scratch...")
        prior_trained, q_trained, losses = load_or_train(
            key,
            moments_jnp, centroid_moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp,
            weights, raw2standard,
            prior_path=PRIOR_FLOW_PATH, q_path=Q_FLOW_PATH,
            **train_kwargs,
        )

    # -----------------------------------------------------------------------
    # 3. Build inference functions (used for single-object sanity check)
    # -----------------------------------------------------------------------
    flow_prob_and_derivs = make_flow_prob_and_derivs(prior_trained)

    _log_scale_ref = 0.5 * (log_scale_range[0] + log_scale_range[1])

    def log_flow_fn(x):
        condition = jnp.array([0.0, 0.0, _log_scale_ref, 0.0, 0.0])
        return prior_trained.log_prob(x, condition=condition)

    # -----------------------------------------------------------------------
    # 4. Single-object RQMC PQR sanity check
    # -----------------------------------------------------------------------
    print("Running single-object RQMC PQR check...")
    x_near0 = jnp.array([2000.0, 3.3 * 2000.0, 0.1 * 3.3 * 2000.0, 0.0])
    x_near, Sigma_near = transform_dataset_to_standard(
        raw2standard, x_near0[None, ...], Sigma0[None, ...]
    )
    mu_near, sigma_near = x_near[0], Sigma_near[0]

    from .data import make_augmentation_noise_raw_jax

    key, subkey = jr.split(key)
    CA_raw = make_augmentation_noise_raw_jax(
        Sigma0, current_snr=x_near0[0] / np.sqrt(Sigma0[0, 0]), target_snr=20.0
    )
    (
        P_est_near,
        P_se_near,
        Q1_est_near,
        Q1_se_near,
        Q2_est_near,
        Q2_se_near,
        R11_est_near,
        R11_se_near,
        R22_est_near,
        R22_se_near,
        R12_est_near,
        R12_se_near,
        proposal_mu_near_aug,
        x_samples_near_aug,
    ) = rqmc_integrate_pqr_jax(
        log_flow_fn,
        flow_prob_and_derivs,
        mu_near,
        sigma_near,
        subkey,
        n_points=2**8,
        n_replicates=16,
        mu_raw=x_near0 if jnp.any(CA_raw != 0.0) else None,
        CM_raw=Sigma0 if jnp.any(CA_raw != 0.0) else None,
        CA_raw=CA_raw if jnp.any(CA_raw != 0.0) else None,
        std_scales=jnp.array(raw2standard.std) if jnp.any(CA_raw != 0.0) else None,
        hessian_scale=3.0,
    )

    print(f"P  (near): {P_est_near:.4e} +/- {P_se_near:.4e}")
    print(f"Q1 (near): {Q1_est_near:.4e} +/- {Q1_se_near:.4e}")
    print(f"Q2 (near): {Q2_est_near:.4e} +/- {Q2_se_near:.4e}")
    print(f"R11(near): {R11_est_near:.4e} +/- {R11_se_near:.4e}")
    print(f"R22(near): {R22_est_near:.4e} +/- {R22_se_near:.4e}")
    print(f"R12(near): {R12_est_near:.4e} +/- {R12_se_near:.4e}")

    # -----------------------------------------------------------------------
    # 5. Grid PQR — +/- catalogues integrated independently
    # -----------------------------------------------------------------------
    # The two catalogues are independent injection realisations over the same
    # footprint (only ~0.8% are genuine +/- ring pairs at the same sky position),
    # so each side is selected and integrated on its own moments rather than
    # matched into pairs — see integrate_catalog_pqr.
    print("Loading galaxy grid catalogues...")
    cat_p = np.load(GRID_P_PATH)
    cat_m = np.load(GRID_M_PATH)

    jax.config.update("jax_debug_nans", False)

    print("Running flow-based RQMC PQR on grid (+/- shear, independent ensembles)...")
    key, kp, km = jr.split(key, 3)
    common = dict(
        n_targets=100_000, n_points=2**10, n_replicates=16, batch_size=512,
    )
    res_p = integrate_catalog_pqr(cat_p, raw2standard, prior_trained, key=kp, **common)
    res_m = integrate_catalog_pqr(cat_m, raw2standard, prior_trained, key=km, **common)

    # -----------------------------------------------------------------------
    # 6. Per-object shear estimate on each side
    # -----------------------------------------------------------------------
    pqr_p, pqr_m = res_p["pqr"], res_m["pqr"]

    def _q_r_g(pqr):
        Q_vec = pqr[:, 1:3]
        R_mat = jnp.stack(
            [
                jnp.stack([pqr[:, 3], pqr[:, 5]], axis=-1),
                jnp.stack([pqr[:, 5], pqr[:, 4]], axis=-1),
            ],
            axis=-2,
        )
        Q_tot = Q_vec / pqr[:, 0:1]
        R_tot = (
            jnp.einsum("...i,...j->...ij", Q_vec, Q_vec) / pqr[:, 0, None, None] ** 2
            - R_mat / pqr[:, 0, None, None]
        )
        g_est = jnp.einsum("...ij,...j->...i", jnp.linalg.inv(R_tot), Q_tot)
        return Q_tot, R_tot, g_est

    Q_tot_p, R_tot_p, g_est_p = _q_r_g(pqr_p)
    Q_tot_m, R_tot_m, g_est_m = _q_r_g(pqr_m)

    g_p_pt, g_m_pt = pqr2g(pqr_p), pqr2g(pqr_m)
    print(f"Estimated g (flow +0.02): {g_p_pt}")
    print(f"Estimated g (flow -0.02): {g_m_pt}")
    m_flow = float((g_p_pt[0] - g_m_pt[0]) / 0.04 - 1.0)
    print(f"Multiplicative bias: {m_flow:.3f}")

    # -----------------------------------------------------------------------
    # 7. Bootstrap uncertainty (independent ensembles, unequal lengths allowed)
    # -----------------------------------------------------------------------
    print("Bootstrapping multiplicative bias uncertainty...")
    stats = bootstrap_independent_mult_bias(
        pqr_p, pqr_m, n_boot=5000, delta_g=0.04, key=jr.PRNGKey(42)
    )
    print(f"m (point): {float(stats['m_point']):0.3f}")
    print(f"m (bootstrap mean): {float(stats['m_mean']):0.3f}")
    print(f"m uncertainty (1sigma): {float(stats['m_std']):0.3f}")
    print(f"m 16-84%: {float(stats['m_p16']):0.3f}, {float(stats['m_p84']):0.3f}")

    # -----------------------------------------------------------------------
    # 8. Optional: visualisations
    # -----------------------------------------------------------------------
    plot_g_distributions(g_est_p, g_est_m)
    plot_Q_distributions(Q_tot_p, Q_tot_m)
    plot_R_distributions(R_tot_p, R_tot_m)


if __name__ == "__main__":
    main()
