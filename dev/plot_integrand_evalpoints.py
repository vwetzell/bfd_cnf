"""
Pick a few grid targets and scatter their RQMC evaluation points coloured by the
integrand value  prior(x|g=0) * N(x; M, Sigma)  at each point.

Reuses the production single-target integrator (rqmc_integrate_pqr_jax), which
returns the actual proposal draws it evaluated, then recolours those same points
by the integrand the integral sees (augmented Gaussian kernel, to match what the
integrator integrates).  Targets are chosen to span the flux range.

    python -m dev.plot_integrand_evalpoints            # 3 targets, faint/mid/bright
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

import bfd

from bfd_cnf.config import (
    GRID_P_PATH,
    PLOTS_DIR,
    key as base_key,
    target_flux_min,
)
from bfd_cnf.integrate_grid import load_raw2standard, load_prior_flow
from bfd_cnf.inference import (
    unpack_packed_cov,
    rqmc_integrate_pqr_jax,
    make_flow_prob_and_derivs,
)
from bfd_cnf.data import (
    transform_dataset_to_standard,
    make_augmentation_noise_raw_jax,
    augment_moments_raw_jax,
)
from bfd_cnf.models.bijections import propagate_cov_to_std_jax
from bfd_cnf.models.flows import cx_to_sx_cond, even_cov_to_CX

LABELS = [r"$\log_{10}M_f$", r"$M_r/M_f$", r"$M_1/M_r$", r"$M_2/M_r$"]
# Fixed physical axis ranges: 800<Mf<150000, 2.0<Mr/Mf<4.1, |M1/Mr|,|M2/Mr|<1.
AXLIM = [(np.log10(800), np.log10(150000)), (2.0, 4.1), (-1.0, 1.0), (-1.0, 1.0)]
N_POINTS = 2**11


def make_log_flow_fn(prior_flow, sx_cond):
    cond = jnp.concatenate([jnp.zeros(2), sx_cond])

    def log_flow_fn(x):  # x: (1, D) -> (1,)
        return prior_flow.log_prob(x[0], condition=cond)[None]

    return log_flow_fn


def integrand_at(prior_flow, sx_cond, mu_std, mu_raw, CM_raw, CA_raw, std_scales, x):
    """log[ prior(x|g=0) * N(x; M, Sigma_aug) ] at standardised points x (n, D)."""
    cond = jnp.concatenate([jnp.zeros(2), sx_cond])
    lp_flow = jax.vmap(lambda xi: prior_flow.log_prob(xi, condition=cond))(x)

    C_raw, _, _ = augment_moments_raw_jax(mu_raw, CM_raw, CA_raw)
    cov_aug = propagate_cov_to_std_jax(C_raw, mu_raw, std_scales)
    L = jnp.linalg.cholesky(cov_aug)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    const = -0.5 * (x.shape[1] * jnp.log(2 * jnp.pi) + logdet)
    diff = x - mu_std
    v = jax.scipy.linalg.solve_triangular(L, diff.T, lower=True)
    log_gauss = const - 0.5 * jnp.sum(v * v, axis=0)
    return lp_flow + log_gauss


def main():
    raw2standard = load_raw2standard()
    key, k_flow = jr.split(base_key)
    prior_flow = load_prior_flow(k_flow)
    std_scales = jnp.asarray(raw2standard.std)
    mean = np.asarray(raw2standard.mean)
    std = np.asarray(raw2standard.std)

    cat = np.load(GRID_P_PATH)
    mom = np.asarray(cat["moments"][:, :4])
    mf, mr_mf = mom[:, 0], mom[:, 1] / mom[:, 0]
    sel = (mf > target_flux_min) & (mf < 90000) & (mr_mf > 2.2) & (mr_mf < 3.5)
    sel &= np.asarray(bfd.stripMuPqr(cat["pqr"]))[:, 0] >= 1e-10
    sel_idx = np.where(sel)[0]

    # faint / mid / bright by flux among selected targets
    order = sel_idx[np.argsort(mf[sel_idx])]
    picks = order[[int(0.1 * len(order)), int(0.5 * len(order)), int(0.9 * len(order))]]
    pick_labels = ["faint (10th pct Mf)", "mid (median Mf)", "bright (90th pct Mf)"]

    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 4.6 * len(picks)))
    for r, (idx, plabel) in enumerate(zip(picks, pick_labels)):
        targ = jnp.asarray(mom[idx][None])  # (1,4)
        CM = jnp.asarray(unpack_packed_cov(cat["covariance"][idx][None]))  # (1,4,4)
        mu_std, cov_std = transform_dataset_to_standard(raw2standard, targ, CM)
        mu_std, cov_std = mu_std[0], cov_std[0]
        mu_raw, CM_raw = targ[0], CM[0]
        sx_cond = cx_to_sx_cond(even_cov_to_CX(CM_raw))

        # Noise augmentation OFF: zero CA_raw -> integrator & integrand both use
        # the raw measurement covariance N(x; M, Sigma).
        CA_raw = jnp.zeros_like(CM_raw)

        out = rqmc_integrate_pqr_jax(
            make_log_flow_fn(prior_flow, sx_cond),
            make_flow_prob_and_derivs(prior_flow, sx_cond),
            mu_std, cov_std, jr.fold_in(key, r),
            n_points=N_POINTS, n_replicates=2,
            mu_raw=mu_raw, CM_raw=CM_raw, CA_raw=CA_raw, std_scales=std_scales,
            hessian_scale=3.0,
        )
        proposal_mu, x = out[-2], out[-1]  # (D,), (n_points, D)

        log_I = np.asarray(
            integrand_at(prior_flow, sx_cond, mu_std, mu_raw, CM_raw, CA_raw, std_scales, x)
        )
        x = np.asarray(x)
        # Drop the points the main integrator's filter discards: their float32
        # integrand underflows to 0 (zero weight, contribute nothing).
        keep = np.isfinite(log_I) & (np.exp(log_I.astype(np.float32)) > 0)
        print(f"  {plabel}: kept {keep.sum()}/{len(keep)} points "
              f"({(~keep).sum()} dropped by integrand filter)")
        x, log_I = x[keep], log_I[keep]
        # physical coords for readability: z * std + mean
        xp = x * std + mean
        mu_p = np.asarray(mu_std) * std + mean
        prop_p = np.asarray(proposal_mu) * std + mean

        mf_t = mom[idx, 0]
        # Colour range: fixed floor at log integrand = -10 up to the peak.
        vmax = float(np.nanmax(log_I))
        vmin = -10.0
        for c, (i, j) in enumerate([(0, 1), (2, 3)]):
            ax = axes[r, c]
            sc = ax.scatter(xp[:, i], xp[:, j], c=log_I, s=4, cmap="viridis",
                            vmin=vmin, vmax=vmax)
            ax.scatter(*mu_p[[i, j]], marker="x", c="red", s=90, label="M (measured)")
            ax.scatter(*prop_p[[i, j]], marker="+", c="white", s=120,
                       linewidths=2, label="proposal mode")
            ax.set_xlim(*AXLIM[i]); ax.set_ylim(*AXLIM[j])
            ax.set_xlabel(LABELS[i]); ax.set_ylabel(LABELS[j])
            fig.colorbar(sc, ax=ax, label="log integrand")
            if c == 0:
                ax.set_title(f"{plabel}   Mf={mf_t:.0f}", loc="left")
            if r == 0 and c == 0:
                ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "integrand_evalpoints_noaug.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
