"""
Corner plot comparing observed templates vs. prior flow samples at g=0, log_scale=12, e=0.

Blue (data points + contours): deep-field summary-template moments
                               (summary_templates_new.fits — one row per galaxy,
                               no sub-pixel-shifted copies).
Orange (contours only):        samples from the trained prior flow.
Green lines / rectangles:      selection box bounds.
Red lines:                     stellar locus reference (Mr/Mf ≈ 3.976).
"""

import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import os

# Default condition is the median grid-target C_X scale (from the per-target PQR
# integration), so the corner plot shows the prior at the scale the shear integration
# actually evaluates.  Pass --log-scale to override (e.g. the physical template /
# corner reference ≈ 11.98).
_TARGET_MEDIAN_LOG_SCALE = 13.27
_ap = argparse.ArgumentParser(description="Corner plot of prior flow vs templates.")
_ap.add_argument("--log-scale", type=float, default=_TARGET_MEDIAN_LOG_SCALE,
                 help="Σ_X log_scale to condition the prior on (default: median "
                      f"target scale {_TARGET_MEDIAN_LOG_SCALE}).")
_args = _ap.parse_args()

import corner
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from bfd_cnf.config import (
    PLOTS_DIR,
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    prior_sigmax_log_scale_mean,
)
from bfd_cnf.data import load_data, load_summary_moments
from bfd_cnf.models.flows import build_flows
from bfd_cnf.training import load_models

# ── Load data and flow ───────────────────────────────────────────────────────
# load_data() reads the tmpl_t04_joined catalogue to rebuild the *exact*
# raw→standard transform the flow was trained with — the flow samples must be
# inverted through the same standardisation.  The observed-template *plot* data,
# however, comes from the deep-field summary table below, which has one row per
# galaxy (no sub-pixel-shifted copies inflating the density).
print("Loading data...")
data = load_data()
key = data["key"]
raw2standard = data["raw2standard"]

print("Loading flow...")
# Pure load — this script never trains.  Fail loudly if the weights are missing
# instead of silently kicking off a 20k-step retrain (as load_or_train would).
if not (os.path.exists(PRIOR_FLOW_PATH) and os.path.exists(Q_FLOW_PATH)):
    raise FileNotFoundError(
        "Trained flow weights not found:\n"
        f"  {PRIOR_FLOW_PATH}\n  {Q_FLOW_PATH}\n"
        "Train the flows first (e.g. `python -m bfd_cnf.train_from_scratch`). "
        "This script only loads, samples, and plots."
    )

# Build the architecture (random init), then overwrite every array leaf with the
# saved values.  The init key is irrelevant — tree_deserialise_leaves replaces
# all weights from disk, so no training happens here.
prior_flow, q_flow = build_flows(key, latent_dim=4, cond_dim=16)
prior_trained, _ = load_models(prior_flow, q_flow, PRIOR_FLOW_PATH, Q_FLOW_PATH)

# ── Observed templates in plot coordinates ───────────────────────────────────
N_PLOT = 1_000_000

print("Loading deep-field summary templates...")
summary_moments = load_summary_moments()
print(f"  {summary_moments.shape[0]} summary templates after quality cuts")

n_obs = min(N_PLOT, summary_moments.shape[0])
print(f"Building observed-template array (subsampled to {n_obs})...")
key, subkey = jr.split(key)
obs_idx = np.array(
    jr.choice(subkey, summary_moments.shape[0], shape=(n_obs,), replace=False)
)
m_sub = summary_moments[obs_idx]

obs_dist_np = np.column_stack(
    [
        np.log10(np.array(m_sub[:, 0])),
        np.array(m_sub[:, 1]) / np.array(m_sub[:, 0]),
        np.array(m_sub[:, 2]) / np.array(m_sub[:, 1]),
        np.array(m_sub[:, 3]) / np.array(m_sub[:, 1]),
    ]
)
mask_obs = np.all(np.isfinite(obs_dist_np), axis=1)
obs_dist_np = obs_dist_np[mask_obs]
print(f"  {mask_obs.sum()} finite observed templates")

# ── Sample from the flow ─────────────────────────────────────────────────────
N = N_PLOT
# Condition the prior on the requested C_X log_scale (default: the median grid-target
# scale 13.27 — the scale the PQR shear integration evaluates).  At the physical
# template scale (≈11.98) the prior reproduces the at-centre summary marginals; at
# larger target scales the L(X|C_X) weighting is broader, so the marginal is broader.
_log_scale_ref = float(_args.log_scale)
condition = jnp.array([0.0, 0.0, _log_scale_ref, 0.0, 0.0])

print(f"Sampling {N} points from prior flow at g=0, log_scale={_log_scale_ref:.2f}, e=0...")
key, subkey = jr.split(key)
samples_std = prior_trained.sample(subkey, sample_shape=(N,), condition=condition)

raw_samples, _ = jax.vmap(raw2standard.inverse_and_log_det)(samples_std)
raw_samples = jnp.clip(
    raw_samples,
    jnp.array([1.0, 1e-3, -jnp.inf, -jnp.inf]),
    jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf]),
)

flow_dist_np = np.column_stack(
    [
        np.log10(np.abs(np.array(raw_samples[:, 0]))),
        np.array(raw_samples[:, 1]) / np.array(raw_samples[:, 0]),
        np.array(raw_samples[:, 2]) / np.array(raw_samples[:, 1]),
        np.array(raw_samples[:, 3]) / np.array(raw_samples[:, 1]),
    ]
)
mask_flow = np.all(np.isfinite(flow_dist_np), axis=1)
flow_dist_np = flow_dist_np[mask_flow]
print(f"  {mask_flow.sum()} / {N} finite flow samples retained")

# ── Equalise sample counts ───────────────────────────────────────────────────
# Plot the same number of points for both distributions so the contour density
# estimates (and any datapoint scatter) are directly comparable.
n_common = min(len(obs_dist_np), len(flow_dist_np))
key, k_obs, k_flow = jr.split(key, 3)
obs_dist_np = obs_dist_np[
    np.array(jr.choice(k_obs, len(obs_dist_np), shape=(n_common,), replace=False))
]
flow_dist_np = flow_dist_np[
    np.array(jr.choice(k_flow, len(flow_dist_np), shape=(n_common,), replace=False))
]
print(f"  equalised both distributions to {n_common} samples each")

# ── Corner plot ──────────────────────────────────────────────────────────────
labels = [
    r"$\log_{10}M_f$",
    r"$M_r / M_f$",
    r"$M_1 / M_r$",
    r"$M_2 / M_r$",
]
plot_range = (
    (np.log10(500), np.log10(200000)),
    (1.2, 5.5),
    (-0.8, 0.8),
    (-0.8, 0.8),
)

fig = plt.figure(figsize=(16, 16))

corner.corner(
    obs_dist_np,
    labels=labels,
    bins=500,
    plot_datapoints=True,
    plot_density=False,
    plot_contours=True,
    fill_contours=False,
    hist_kwargs={"label": "Observed Templates"},
    smooth=5.0,
    smooth1d=1.0,
    color="tab:blue",
    range=plot_range,
    fig=fig,
)

corner.corner(
    flow_dist_np,
    labels=labels,
    bins=500,
    plot_datapoints=False,
    plot_density=False,
    plot_contours=True,
    fill_contours=False,
    hist_kwargs={"label": "Prior Flow"},
    smooth=5.0,
    color="tab:orange",
    range=plot_range,
    fig=fig,
)

# ── Selection bounds (green) and stellar reference (red) ────────────────────
axs = fig.axes

# Diagonal histograms: vlines at selection edges
axs[0].axvline(np.log10(1500), lw=1, c="tab:green")
axs[0].axvline(np.log10(90000), lw=1, c="tab:green")

axs[5].axvline(2.2, label="Target Bounds", lw=1, c="tab:green")
axs[5].axvline(3.5, lw=1, c="tab:green")

axs[10].axvline(-1.0, lw=1, c="tab:green")
axs[10].axvline(1.0, lw=1, c="tab:green")

axs[15].axvline(-1.0, lw=1, c="tab:green")
axs[15].axvline(1.0, lw=1, c="tab:green")

# Stellar locus reference
axs[5].axvline(3.976167, color="tab:red", lw=1, label="Stars")
axs[4].axhline(3.976167, lw=1, color="tab:red")
axs[8].axhline(0.0, lw=1, color="tab:red", zorder=500)
axs[9].axvline(3.976167, lw=1, color="tab:red")
axs[9].axhline(0.0, lw=1, color="tab:red", zorder=500)
axs[13].axhline(0.0, lw=1, color="tab:red", zorder=500)
axs[14].axhline(0.0, lw=1, color="tab:red", zorder=500)
axs[14].axvline(0.0, lw=1, color="tab:red", zorder=500)
axs[12].axhline(0.0, lw=1, color="tab:red", zorder=500)
axs[13].axvline(3.976167, lw=1, color="tab:red", zorder=500)
axs[10].axvline(0.0, lw=1, color="tab:red", zorder=500)
axs[15].axvline(0.0, lw=1, color="tab:red", zorder=500)


# Selection rectangles on 2-D scatter panels
def add_rect(ax, x0, y0, dx, dy):
    ax.add_patch(
        mpatches.Rectangle(
            (x0, y0),
            dx,
            dy,
            linewidth=1,
            edgecolor="tab:green",
            facecolor="none",
            zorder=100,
        )
    )


add_rect(axs[4], np.log10(1500), 2.2, np.log10(90000) - np.log10(1500), 3.5 - 2.2)
add_rect(axs[8], np.log10(1500), -1.0, np.log10(90000) - np.log10(1500), 2.0)
add_rect(axs[9], 2.2, -1.0, 3.5 - 2.2, 2.0)
add_rect(axs[12], np.log10(1500), -1.0, np.log10(90000) - np.log10(1500), 2.0)
add_rect(axs[13], 2.2, -1.0, 3.5 - 2.2, 2.0)
axs[14].add_patch(
    mpatches.Circle(
        (0.0, 0.0),
        1.0,
        linewidth=1,
        edgecolor="tab:green",
        facecolor="none",
        zorder=100,
    )
)

axs[5].legend(bbox_to_anchor=(0.0, 1.0), loc="lower left", fontsize=16)

out = os.path.join(PLOTS_DIR, f"corner_prior_g0_ls{_log_scale_ref:.1f}_e0.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
