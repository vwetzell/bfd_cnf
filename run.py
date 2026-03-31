"""
run.py
======
End-to-end workflow script.  Calls functions from the other modules in the
correct order to reproduce the notebook analysis.

Usage
-----
    python run.py
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

# ---------------------------------------------------------------------------
# Imports from this package
# ---------------------------------------------------------------------------
from .config import (
    key as _base_key,
    Sigma0,
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    GRID_P_PATH,
    GRID_M_PATH,
    prior_flow_layers,
    q_flow_layers,
    n_sx_train,
    log_scale_range,
    e_max,
)
from .data import load_data, transform_dataset_to_standard
from .models.flows import build_flows, batch_cholesky_of_sym, cov2corr
from .training import (
    load_or_train,
    compute_std_stats,
    continue_training,
    save_models,
)
from .inference import (
    make_flow_prob_and_derivs,
    sample_trunc_mvn_noise_first_lower,
    rqmc_pqr_grid,
    rqmc_integrate_pqr_jax,
    load_grid_data,
    assemble_pqr_from_flow,
    prob_template,
    _halton_sequence,
)
from .statistics import (
    pqr2g,
    pqr2multbias,
    clipR,
    bootstrap_total_mult_bias,
)
from .viz import (
    plot_snr_histograms,
    plot_flow_vs_obs_corner,
    plot_g_distributions,
    plot_Q_distributions,
    plot_R_distributions,
    plot_mult_bias_hexbin,
)


def main() -> None:
    """Run the full end-to-end BFD cNF workflow.

    Executes the following steps in order:

    1. Load and quality-filter the BFD FITS template table.
    2. Load or train the prior and q normalizing flows.
    3. Build flow-based PQR inference functions.
    4. Run a single-object RQMC PQR sanity check.
    5. Load galaxy grid data and compute flow-based RQMC PQR for ±shear grids.
    6. Assemble per-object PQR arrays and estimate the shear vector.
    7. Bootstrap the multiplicative bias uncertainty.
    8. Produce diagnostic visualisations.
    """
    key = _base_key

    # -----------------------------------------------------------------------
    # 1. Load and pre-process data
    # -----------------------------------------------------------------------
    print("Loading data...")
    data = load_data(key=key)
    moments_jnp = data["moments_jnp"]
    odd_moments_jnp = data["odd_moments_jnp"]
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
    print("Loading / training flows...")
    prior_trained, q_trained, losses = load_or_train(
        key,
        moments_jnp,
        odd_moments_jnp,
        cov_jnp,
        dm_dg_jnp,
        d2m_dg2_jnp,
        weights,
        raw2standard,
        prior_path=PRIOR_FLOW_PATH,
        q_path=Q_FLOW_PATH,
        steps=20_000,
        learning_rate=1e-4,
        weight_decay=1e-5,
        grad_clip=0.5,
        log_scale_range=log_scale_range,
        e_max=e_max,
        n_sx_train=n_sx_train,
    )

    # -----------------------------------------------------------------------
    # 3. Build inference functions
    # -----------------------------------------------------------------------
    flow_prob_and_derivs = make_flow_prob_and_derivs(prior_trained)

    def log_flow_fn(x):
        return prior_trained.log_prob(x, condition=jnp.array([0.0, 0.0]))

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
    # 5. Grid PQR on galaxy grid
    # -----------------------------------------------------------------------
    print("Loading galaxy grid data...")
    joined_grid = load_grid_data(GRID_P_PATH, GRID_M_PATH)

    import bfd

    pqr_sim_p = bfd.stripMuPqr(joined_grid["pqr_p"])
    pqr_sim_m = bfd.stripMuPqr(joined_grid["pqr_m"])

    target_p_moments = jnp.array(joined_grid["moments_p"][:, :4])
    target_m_moments = jnp.array(joined_grid["moments_m"][:, :4])

    selected_templates = target_p_moments[:, 0] > 1500.0
    selected_templates &= target_p_moments[:, 0] < 90000.0
    selected_templates &= (target_p_moments[:, 1] / target_p_moments[:, 0]) > 2.2
    selected_templates &= (target_p_moments[:, 1] / target_p_moments[:, 0]) < 3.5

    targets_p = target_p_moments[selected_templates]
    targets_m = target_m_moments[selected_templates]

    pqr_sim_p_sel = pqr_sim_p[selected_templates]
    pqr_sim_m_sel = pqr_sim_m[selected_templates]

    mask = (pqr_sim_p_sel[:, 0] >= 1e-10) & (pqr_sim_m_sel[:, 0] >= 1e-10)
    targets_p = targets_p[mask]
    targets_m = targets_m[mask]
    pqr_sim_p_sel = pqr_sim_p_sel[mask]
    pqr_sim_m_sel = pqr_sim_m_sel[mask]

    N_targets = targets_p.shape[0]
    key, subkey = jr.split(key)
    idx = jr.choice(subkey, N_targets, shape=(100_000,), replace=False)

    targets_p_10k = targets_p[idx]
    targets_m_10k = targets_m[idx]

    packed_p = joined_grid["covariance_p"][selected_templates]
    packed_p_10k = packed_p[idx]
    CM_raw_grid_p = np.zeros((packed_p_10k.shape[0], 5, 5), dtype=float)
    j = 0
    for i in range(5):
        nvals = 5 - i
        CM_raw_grid_p[:, i, i:] = packed_p_10k[:, j : j + nvals]
        CM_raw_grid_p[:, i:, i] = packed_p_10k[:, j : j + nvals]
        j += nvals
    CM_raw_grid_p = CM_raw_grid_p[:, :4, :4]

    mu_std_p, sigma_std_p = transform_dataset_to_standard(
        raw2standard, targets_p_10k, CM_raw_grid_p
    )

    jax.config.update("jax_debug_nans", False)

    print("Running flow-based RQMC PQR on grid (+shear)...")
    (
        P_p,
        P_se_p,
        Q1_p,
        Q1_se_p,
        Q2_p,
        Q2_se_p,
        R11_p,
        R11_se_p,
        R22_p,
        R22_se_p,
        R12_p,
        R12_se_p,
    ) = rqmc_pqr_grid(
        mu_std_p,
        sigma_std_p,
        targets_p_10k,
        CM_raw_grid_p,
        n_points=2**10,
        n_replicates=16,
        batch_size=512,
        raw2standard=raw2standard,
        flow_prob_and_derivs=flow_prob_and_derivs,
        log_flow_fn=log_flow_fn,
    )

    packed_m = joined_grid["covariance_m"][selected_templates]
    packed_m_10k = packed_m[idx]
    CM_raw_grid_m = np.zeros((packed_m_10k.shape[0], 5, 5), dtype=float)
    j = 0
    for i in range(5):
        nvals = 5 - i
        CM_raw_grid_m[:, i, i:] = packed_m_10k[:, j : j + nvals]
        CM_raw_grid_m[:, i:, i] = packed_m_10k[:, j : j + nvals]
        j += nvals
    CM_raw_grid_m = CM_raw_grid_m[:, :4, :4]

    mu_std_m, sigma_std_m = transform_dataset_to_standard(
        raw2standard, targets_m_10k, CM_raw_grid_m
    )

    print("Running flow-based RQMC PQR on grid (-shear)...")
    (
        P_m,
        P_se_m,
        Q1_m,
        Q1_se_m,
        Q2_m,
        Q2_se_m,
        R11_m,
        R11_se_m,
        R22_m,
        R22_se_m,
        R12_m,
        R12_se_m,
    ) = rqmc_pqr_grid(
        mu_std_m,
        sigma_std_m,
        targets_m_10k,
        CM_raw_grid_m,
        n_points=2**10,
        n_replicates=16,
        batch_size=512,
        raw2standard=raw2standard,
        flow_prob_and_derivs=flow_prob_and_derivs,
        log_flow_fn=log_flow_fn,
    )

    # -----------------------------------------------------------------------
    # 6. Assemble PQR and compute shear
    # -----------------------------------------------------------------------
    pqr_arr_p, pqr_arr_m = assemble_pqr_from_flow(
        P_p,
        Q1_p,
        Q2_p,
        R11_p,
        R22_p,
        R12_p,
        P_m,
        Q1_m,
        Q2_m,
        R11_m,
        R22_m,
        R12_m,
    )

    Q_p_vec = jnp.stack([Q1_p, Q2_p], axis=-1)
    R_p_mat = jnp.stack(
        [
            jnp.stack([R11_p, R12_p], axis=-1),
            jnp.stack([R12_p, R22_p], axis=-1),
        ],
        axis=-2,
    )
    Q_tot_p = Q_p_vec / P_p[:, None]
    R_tot_p = (
        jnp.einsum("...i,...j->...ij", Q_p_vec, Q_p_vec) / P_p[:, None, None] ** 2
        - R_p_mat / P_p[:, None, None]
    )
    g_est_p = jnp.einsum("...ij,...j->...i", jnp.linalg.inv(R_tot_p), Q_tot_p)

    Q_m_vec = jnp.stack([Q1_m, Q2_m], axis=-1)
    R_m_mat = jnp.stack(
        [
            jnp.stack([R11_m, R12_m], axis=-1),
            jnp.stack([R12_m, R22_m], axis=-1),
        ],
        axis=-2,
    )
    Q_tot_m = Q_m_vec / P_m[:, None]
    R_tot_m = (
        jnp.einsum("...i,...j->...ij", Q_m_vec, Q_m_vec) / P_m[:, None, None] ** 2
        - R_m_mat / P_m[:, None, None]
    )
    g_est_m = jnp.einsum("...ij,...j->...i", jnp.linalg.inv(R_tot_m), Q_tot_m)

    print(f"Estimated g (flow +0.02): {pqr2g(pqr_arr_p)}")
    print(f"Estimated g (flow -0.02): {pqr2g(pqr_arr_m)}")

    pqr_arr = jnp.concatenate([pqr_arr_p, pqr_arr_m], axis=-1)
    print(f"Multiplicative bias: {pqr2multbias(pqr_arr):.3f}")

    # -----------------------------------------------------------------------
    # 7. Bootstrap uncertainty
    # -----------------------------------------------------------------------
    print("Bootstrapping multiplicative bias uncertainty...")
    stats = bootstrap_total_mult_bias(
        pqr_arr_p, pqr_arr_m, n_boot=5000, delta_g=0.04, key=jr.PRNGKey(42)
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
    plot_mult_bias_hexbin(targets_p_10k, pqr_arr, pqr2multbias)


if __name__ == "__main__":
    main()
