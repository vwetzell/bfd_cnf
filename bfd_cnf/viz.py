"""
viz.py
======
Visualisation helpers for BFD cNF results.  All plotting code from the
notebook is wrapped into named functions rather than run inline.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
import corner

# ---------------------------------------------------------------------------
# Moment SNR histogram
# ---------------------------------------------------------------------------


def plot_snr_histograms(moments_jnp: jax.Array, cov_jnp: jax.Array) -> None:
    """Plot histograms of the fractional flux and size SNR for the template set.

    Parameters
    ----------
    moments_jnp : jax.Array, shape (N, 4)
        Raw template moments ``[Mf, Mr, M1, M2]``.
    cov_jnp : jax.Array, shape (N, 4, 4)
        Per-object moment covariance matrices.
    """
    plt.figure()
    plt.hist(
        jnp.sqrt(cov_jnp[:, 0, 0]) / moments_jnp[:, 0],
        bins=100,
        histtype="step",
        range=(0.0, 0.2),
        label=r"$\sigma_{M_f}/M_f$",
    )
    plt.hist(
        jnp.sqrt(cov_jnp[:, 1, 1]) / moments_jnp[:, 1],
        bins=100,
        histtype="step",
        range=(0.0, 0.2),
        label=r"$\sigma_{M_r}/M_r$",
    )
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Corner plots for template moments
# ---------------------------------------------------------------------------


def plot_moment_corner(
    moments_jnp: jax.Array, cov_jnp: jax.Array, key: jax.Array
) -> jax.Array:
    """Plot a corner plot of the template moments in (log10 Mf, Mr/Mf) space.

    Overlays the raw observed distribution and an approximate noise-corrected
    distribution using ``corner.corner``.

    Parameters
    ----------
    moments_jnp : jax.Array, shape (N, 4)
        Raw template moments.
    cov_jnp : jax.Array, shape (N, 4, 4)
        Per-object moment covariance matrices.
    key : jax.Array
        JAX PRNG key used for bootstrap resampling.

    Returns
    -------
    jax.Array
        Updated PRNG key.
    """
    import jax.random as jr

    key, subkey = jr.split(key)

    resampled_idx = jr.choice(
        subkey,
        np.arange(moments_jnp.shape[0]),
        shape=(moments_jnp.shape[0],),
        replace=True,
    )

    resampled = np.array(
        [
            jnp.log10(moments_jnp[:, 0]),
            (moments_jnp[:, 1] / moments_jnp[:, 0]),
            moments_jnp[:, 2] / moments_jnp[:, 1],
            moments_jnp[:, 3] / moments_jnp[:, 1],
        ]
    ).T[resampled_idx]

    resampled2 = np.array(
        [
            jnp.log10(moments_jnp[:, 0]),
            (moments_jnp[:, 1] / moments_jnp[:, 0])
            - cov_jnp[:, 0, 1] / moments_jnp[:, 0] ** 2
            + moments_jnp[:, 1] * cov_jnp[:, 0, 0] / moments_jnp[:, 0] ** 3,
            moments_jnp[:, 2] / moments_jnp[:, 1],
            moments_jnp[:, 3] / moments_jnp[:, 1],
        ]
    ).T[resampled_idx]

    fig = plt.figure(figsize=(10, 10))
    corner.corner(
        np.array([resampled[:, 0], resampled[:, 1]]).T,
        labels=[
            r"$\log_{10}M_f$",
            r"$M_r / M_f$",
        ],
        bins=500,
        plot_datapoints=True,
        plot_density=False,
        plot_contours=True,
        fill_contours=False,
        hist_kwargs={"label": "Observed Templates"},
        smooth=5.0,
        smooth1d=1.0,
        color="tab:blue",
        range=((np.log10(800), np.log10(200000)), (1.2, 5.5)),
        fig=fig,
    )
    corner.corner(
        np.array([resampled2[:, 0], resampled2[:, 1]]).T,
        labels=[
            r"$\log_{10}M_f$",
            r"$M_r / M_f$",
            r"$M_1 / M_r$",
            r"$M_2 / M_r$",
        ],
        bins=500,
        plot_datapoints=False,
        plot_density=False,
        plot_contours=True,
        fill_contours=False,
        hist_kwargs={"label": "Observed Templates"},
        smooth=5.0,
        smooth1d=1.0,
        color="tab:orange",
        range=((np.log10(800), np.log10(200000)), (1.2, 5.5)),
        fig=fig,
    )
    plt.show()
    return key


def plot_resampled_corner(
    resampled: np.ndarray,
    x_lo: float | None = None,
    x_hi: float | None = None,
    y_lo: float | None = None,
    y_hi: float | None = None,
) -> None:
    """Plot a corner plot of the importance-resampled template distribution.

    Optionally overlays the selection rectangle used for the galaxy grid.

    Parameters
    ----------
    resampled : np.ndarray, shape (N, 4)
        Resampled data in ``[log10 Mf, Mr/Mf, M1/Mr, M2/Mr]`` coordinates.
    x_lo, x_hi : float or None, optional
        Lower / upper log10 Mf bounds of the selection box.  If ``None``,
        no box is drawn.
    y_lo, y_hi : float or None, optional
        Lower / upper Mr/Mf bounds of the selection box.
    """
    fig = plt.figure(figsize=(10, 10))
    corner.corner(
        np.array(
            [
                resampled[:, 0],
                resampled[:, 1],
                np.hypot(resampled[:, 2], resampled[:, 3]),
            ]
        ).T,
        labels=[
            r"$\log_{10}M_f$",
            r"$M_r / M_f$",
            r"$\sqrt{(M_1 / M_r)^2 + (M_2 / M_r)^2}$",
        ],
        bins=500,
        plot_datapoints=True,
        plot_density=False,
        plot_contours=True,
        fill_contours=False,
        hist_kwargs={"label": "Observed Templates"},
        smooth=5.0,
        smooth1d=1.0,
        color="tab:blue",
        fig=fig,
    )

    if x_lo is not None and y_lo is not None:
        rect4 = Rectangle(
            (np.log10(1500), 2.2),
            np.log10(90000) - np.log10(1500),
            3.5 - 2.2,
            linewidth=1,
            edgecolor="tab:red",
            facecolor="none",
            zorder=100,
        )

        axs = fig.get_axes()
        axs[0].axvline(np.log10(1500), color="tab:red", linewidth=1, zorder=100)
        axs[0].axvline(np.log10(90000), color="tab:red", linewidth=1, zorder=100)
        axs[4].axvline(2.2, color="tab:red", linewidth=1, zorder=100)
        axs[4].axvline(3.5, color="tab:red", linewidth=1, zorder=100)
        axs[3].add_patch(rect4)

    plt.show()


# ---------------------------------------------------------------------------
# Flow vs. observed corner plot
# ---------------------------------------------------------------------------


def plot_flow_vs_obs_corner(
    obs_dist_np: np.ndarray,
    flow_dist_np_noise: np.ndarray,
    data_mean: jax.Array,
    data_std: jax.Array,
) -> None:
    """Compare the trained flow distribution against the observed template moments.

    Overlays ``corner.corner`` contours for both distributions and annotates
    the target selection region.

    Parameters
    ----------
    obs_dist_np : np.ndarray, shape (N, 4)
        Observed template moments in
        ``[log10 Mf, Mr/Mf, M1/Mr, M2/Mr]`` coordinates.
    flow_dist_np_noise : np.ndarray, shape (M, 4)
        Flow samples convolved with measurement noise in the same coordinates.
    data_mean : jax.Array, shape (4,)
        Standardisation mean (used for axis labelling / reference).
    data_std : jax.Array, shape (4,)
        Standardisation standard deviation.
    """
    fig = plt.figure(figsize=(16, 16))
    corner.corner(
        obs_dist_np,
        labels=[
            r"$\log_{10}M_f$",
            r"$M_r / M_f$",
            r"$M_1 / M_r$",
            r"$M_2 / M_r$",
        ],
        bins=500,
        plot_datapoints=True,
        plot_density=False,
        plot_contours=True,
        fill_contours=False,
        hist_kwargs={"label": "Observed Templates"},
        smooth=5.0,
        color="tab:blue",
        range=((np.log10(500), np.log10(200000)), (1.2, 5.5), (-0.8, 0.8), (-0.8, 0.8)),
        fig=fig,
        labels_kwargs={"fontsize": 20},
    )

    corner.corner(
        flow_dist_np_noise,
        labels=[
            r"$\log_{10}M_f$",
            r"$M_r / M_f$",
            r"$M_1 / M_r$",
            r"$M_2 / M_r$",
        ],
        bins=500,
        plot_datapoints=False,
        plot_density=False,
        plot_contours=True,
        fill_contours=False,
        hist_kwargs={"label": "Trained Distribution + Noise"},
        smooth=5.0,
        color="tab:orange",
        range=((np.log10(500), np.log10(200000)), (1.2, 5.5), (-0.8, 0.8), (-0.8, 0.8)),
        fig=fig,
        label_kwargs={"fontsize": 20},
    )

    axs = fig.axes

    axs[0].axvline(np.log10(1500), lw=1, c="tab:green")
    axs[0].axvline(np.log10(90000), lw=1, c="tab:green")

    axs[5].axvline(2.2, label="Target Bounds", lw=1, c="tab:green")
    axs[5].axvline(3.5, lw=1, c="tab:green")

    axs[10].axvline(-1.0, lw=1, c="tab:green")
    axs[10].axvline(1.0, lw=1, c="tab:green")

    axs[15].axvline(-1.0, lw=1, c="tab:green")
    axs[15].axvline(1.0, lw=1, c="tab:green")

    rect4 = Rectangle(
        (np.log10(1500), 2.2),
        np.log10(90000) - np.log10(1500),
        3.5 - 2.2,
        linewidth=1,
        edgecolor="tab:green",
        facecolor="none",
        zorder=100,
    )
    axs[4].add_patch(rect4)
    axs[4].scatter(np.log10(5000), 2.8, marker="+", s=100, color="r", zorder=200)

    rect8 = Rectangle(
        (np.log10(1500), -1.0),
        np.log10(90000) - np.log10(1500),
        1.0 + 1.0,
        linewidth=1,
        edgecolor="tab:green",
        facecolor="none",
        zorder=100,
    )
    axs[8].add_patch(rect8)

    rect9 = Rectangle(
        (2.2, -1.0),
        3.5 - 2.2,
        1.0 + 1.0,
        linewidth=1,
        edgecolor="tab:green",
        facecolor="none",
        zorder=100,
    )
    axs[9].add_patch(rect9)

    rect12 = Rectangle(
        (np.log10(1500), -1.0),
        np.log10(90000) - np.log10(1500),
        1.0 + 1.0,
        linewidth=1,
        edgecolor="tab:green",
        facecolor="none",
        zorder=100,
    )
    axs[12].add_patch(rect12)

    rect13 = Rectangle(
        (2.2, -1.0),
        3.5 - 2.2,
        1.0 + 1.0,
        linewidth=1,
        edgecolor="tab:green",
        facecolor="none",
        zorder=100,
    )
    axs[13].add_patch(rect13)

    rect14 = Circle(
        (0.0, 0.0),
        1.0,
        linewidth=1,
        edgecolor="tab:green",
        facecolor="none",
        zorder=100,
    )
    axs[14].add_patch(rect14)

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

    axs[5].legend(bbox_to_anchor=(0.0, 1.0), loc="lower left", fontsize=20)
    plt.show()


# ---------------------------------------------------------------------------
# P/dP/d2P grid plots
# ---------------------------------------------------------------------------


def plot_prob_derivs_m1mr_m2mr(
    m1mr: np.ndarray,
    m2mr: np.ndarray,
    probs: np.ndarray,
    dp_dg1: np.ndarray,
    dp_dg2: np.ndarray,
    d2p_dg1_dg1: np.ndarray,
    d2p_dg2_dg2: np.ndarray,
    d2p_dg1_dg2: np.ndarray,
    title: str | None = None,
) -> None:
    """Plot a 2×3 grid of P and its shear derivatives vs M1/Mr and M2/Mr.

    Parameters
    ----------
    m1mr : np.ndarray, shape (nx,)
        Grid values along the M1/Mr axis.
    m2mr : np.ndarray, shape (ny,)
        Grid values along the M2/Mr axis.
    probs : np.ndarray, shape (..., nx, ny)
        Probability P on the grid.
    dp_dg1 : np.ndarray, shape (..., nx, ny)
        First shear derivative dP/dg1.
    dp_dg2 : np.ndarray, shape (..., nx, ny)
        First shear derivative dP/dg2.
    d2p_dg1_dg1 : np.ndarray, shape (..., nx, ny)
        Second shear derivative d²P/dg1².
    d2p_dg2_dg2 : np.ndarray, shape (..., nx, ny)
        Second shear derivative d²P/dg2².
    d2p_dg1_dg2 : np.ndarray, shape (..., nx, ny)
        Mixed shear derivative d²P/dg1dg2.
    title : str or None, optional
        Figure title.  Default is ``None``.
    """
    p_m1mr_m2mr = probs[0, 0, :, :].T
    dp_dg1_m1mr_m2mr = dp_dg1[0, 0, :, :].T
    dp_dg2_m1mr_m2mr = dp_dg2[0, 0, :, :].T
    d2p_dg1_dg1_m1mr_m2mr = d2p_dg1_dg1[0, 0, :, :].T
    d2p_dg2_dg2_m1mr_m2mr = d2p_dg2_dg2[0, 0, :, :].T
    d2p_dg1_dg2_m1mr_m2mr = d2p_dg1_dg2[0, 0, :, :].T

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    p_plot = ax[0, 0].imshow(
        p_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(m1mr[0], m1mr[-1], m2mr[0], m2mr[-1]),
        cmap="Blues",
        interpolation="none",
    )
    ax[0, 0].set_xlabel(r"$M_1/M_r$", fontsize=20)
    ax[0, 0].set_ylabel(r"$M_2/M_r$", fontsize=20)
    plt.colorbar(p_plot, ax=ax[0, 0]).set_label(label=r"$P$", size=20)

    dp_dg_absmax = 0.5 * max(
        np.max(np.abs(dp_dg1_m1mr_m2mr)),
        np.max(np.abs(dp_dg2_m1mr_m2mr)),
    )
    dp_dg1_plot = ax[0, 1].imshow(
        dp_dg1_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(m1mr[0], m1mr[-1], m2mr[0], m2mr[-1]),
        cmap="RdBu",
        interpolation="none",
        vmin=-dp_dg_absmax,
        vmax=dp_dg_absmax,
    )
    ax[0, 1].set_xlabel(r"$M_1/M_r$", fontsize=20)
    ax[0, 1].set_ylabel(r"$M_2/M_r$", fontsize=20)
    plt.colorbar(dp_dg1_plot, ax=ax[0, 1]).set_label(
        label=r"$\frac{\partial P}{\partial g_1}$", size=20
    )
    ax[0, 1].axvline(0.0, color="grey")

    dp_dg2_plot = ax[0, 2].imshow(
        dp_dg2_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(m1mr[0], m1mr[-1], m2mr[0], m2mr[-1]),
        cmap="RdBu",
        interpolation="none",
        vmin=-dp_dg_absmax,
        vmax=dp_dg_absmax,
    )
    ax[0, 2].set_xlabel(r"$M_1/M_r$", fontsize=20)
    ax[0, 2].set_ylabel(r"$M_2/M_r$", fontsize=20)
    plt.colorbar(dp_dg2_plot, ax=ax[0, 2]).set_label(
        label=r"$\frac{\partial P}{ \partial g_2}$", size=20
    )
    ax[0, 2].axhline(0.0, color="grey")

    d2p_dg_dg_absmax = 0.5 * max(
        np.max(np.abs(d2p_dg1_dg1_m1mr_m2mr)),
        np.max(np.abs(d2p_dg2_dg2_m1mr_m2mr)),
    )
    d2p_dg1_dg1_plot = ax[1, 0].imshow(
        d2p_dg1_dg1_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(m1mr[0], m1mr[-1], m2mr[0], m2mr[-1]),
        cmap="RdBu",
        interpolation="none",
        vmin=-d2p_dg_dg_absmax,
        vmax=d2p_dg_dg_absmax,
    )
    ax[1, 0].set_xlabel(r"$M_1/M_r$", fontsize=20)
    ax[1, 0].set_ylabel(r"$M_2/M_r$", fontsize=20)
    plt.colorbar(d2p_dg1_dg1_plot, ax=ax[1, 0]).set_label(
        label=r"$\frac{\partial^2 P}{\partial g_1^2}$", size=20
    )
    ax[1, 0].axvline(0.0, color="grey")

    d2p_dg2_dg2_plot = ax[1, 1].imshow(
        d2p_dg2_dg2_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(m1mr[0], m1mr[-1], m2mr[0], m2mr[-1]),
        cmap="RdBu",
        interpolation="none",
        vmin=-d2p_dg_dg_absmax,
        vmax=d2p_dg_dg_absmax,
    )
    ax[1, 1].set_xlabel(r"$M_1/M_r$", fontsize=20)
    ax[1, 1].set_ylabel(r"$M_2/M_r$", fontsize=20)
    plt.colorbar(d2p_dg2_dg2_plot, ax=ax[1, 1]).set_label(
        label=r"$\frac{\partial^2 P}{\partial g_2^2}$", size=20
    )
    ax[1, 1].axhline(0.0, color="grey")

    d2p_dg1_dg2_plot = ax[1, 2].imshow(
        d2p_dg1_dg2_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(m1mr[0], m1mr[-1], m2mr[0], m2mr[-1]),
        cmap="RdBu",
        interpolation="none",
        vmin=0.5 * -np.max(np.abs(d2p_dg1_dg2_m1mr_m2mr)),
        vmax=0.5 * np.max(np.abs(d2p_dg1_dg2_m1mr_m2mr)),
    )
    ax[1, 2].set_xlabel(r"$M_1/M_r$", fontsize=20)
    ax[1, 2].set_ylabel(r"$M_2/M_r$", fontsize=20)
    plt.colorbar(d2p_dg1_dg2_plot, ax=ax[1, 2]).set_label(
        label=r"$\frac{\partial^2 P}{\partial g_1 \partial g_2}$", size=20
    )
    ax[1, 2].axhline(0.0, color="grey")
    ax[1, 2].axvline(0.0, color="grey")

    if title:
        plt.suptitle(title, fontsize=24)
    plt.tight_layout()
    plt.show()


def plot_prob_derivs_log10mf_mrmf(
    log10mf: np.ndarray,
    mrmf: np.ndarray,
    probs: np.ndarray,
    dp_dg1: np.ndarray,
    dp_dg2: np.ndarray,
    d2p_dg1_dg1: np.ndarray,
    d2p_dg2_dg2: np.ndarray,
    d2p_dg1_dg2: np.ndarray,
) -> None:
    """Plot a 2×3 grid of P and its shear derivatives vs log10(Mf) and Mr/Mf.

    Parameters
    ----------
    log10mf : np.ndarray, shape (nx,)
        Grid values along the log10(Mf) axis.
    mrmf : np.ndarray, shape (ny,)
        Grid values along the Mr/Mf axis.
    probs : np.ndarray, shape (nx, ny, ...)
        Probability P on the grid (summed over trailing axes for display).
    dp_dg1 : np.ndarray, shape (nx, ny, ...)
        First shear derivative dP/dg1.
    dp_dg2 : np.ndarray, shape (nx, ny, ...)
        First shear derivative dP/dg2.
    d2p_dg1_dg1 : np.ndarray, shape (nx, ny, ...)
        Second shear derivative d²P/dg1².
    d2p_dg2_dg2 : np.ndarray, shape (nx, ny, ...)
        Second shear derivative d²P/dg2².
    d2p_dg1_dg2 : np.ndarray, shape (nx, ny, ...)
        Mixed shear derivative d²P/dg1dg2.
    """
    p_m1mr_m2mr = np.sum(probs, axis=(2, 3))
    dp_dg1_m1mr_m2mr = np.sum(dp_dg1, axis=(2, 3))
    dp_dg2_m1mr_m2mr = np.sum(dp_dg2, axis=(2, 3))
    d2p_dg1_dg1_m1mr_m2mr = np.sum(d2p_dg1_dg1, axis=(2, 3))
    d2p_dg2_dg2_m1mr_m2mr = np.sum(d2p_dg2_dg2, axis=(2, 3))
    d2p_dg1_dg2_m1mr_m2mr = np.sum(d2p_dg1_dg2, axis=(2, 3))

    rect_ls = []
    for i in range(6):
        rect_ls.append(
            Rectangle(
                (np.log10(1500), 2.2),
                np.log10(90000) - np.log10(1500),
                3.5 - 2.2,
                linewidth=2,
                edgecolor="tab:green",
                facecolor="none",
                zorder=100,
            )
        )

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    p_plot = ax[0, 0].imshow(
        p_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(log10mf[0], log10mf[-1], mrmf[0], mrmf[-1]),
        cmap="Blues",
        interpolation="none",
    )
    ax[0, 0].set_xlabel(r"$\log_{10}M_f$", fontsize=20)
    ax[0, 0].set_ylabel(r"$M_r/M_f$", fontsize=20)
    plt.colorbar(p_plot, ax=ax[0, 0]).set_label(label=r"$P$", size=20)
    ax[0, 0].add_patch(rect_ls[0])

    d2p_dg_dg_absmax = 0.5 * max(
        np.max(np.abs(dp_dg1_m1mr_m2mr)),
        np.max(np.abs(dp_dg2_m1mr_m2mr)),
    )

    dp_dg1_plot = ax[0, 1].imshow(
        dp_dg1_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(log10mf[0], log10mf[-1], mrmf[0], mrmf[-1]),
        cmap="RdBu",
        interpolation="none",
        vmin=-d2p_dg_dg_absmax,
        vmax=d2p_dg_dg_absmax,
    )
    ax[0, 1].set_xlabel(r"$\log_{10}M_f$", fontsize=20)
    ax[0, 1].set_ylabel(r"$M_r/M_f$", fontsize=20)
    plt.colorbar(dp_dg1_plot, ax=ax[0, 1]).set_label(
        label=r"$\frac{\partial P}{\partial g_1}$", size=20
    )
    ax[0, 1].add_patch(rect_ls[1])

    dp_dg2_plot = ax[0, 2].imshow(
        dp_dg2_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(log10mf[0], log10mf[-1], mrmf[0], mrmf[-1]),
        cmap="RdBu",
        interpolation="none",
        vmin=-d2p_dg_dg_absmax,
        vmax=d2p_dg_dg_absmax,
    )
    ax[0, 2].set_xlabel(r"$\log_{10}M_f$", fontsize=20)
    ax[0, 2].set_ylabel(r"$M_r/M_f$", fontsize=20)
    plt.colorbar(dp_dg2_plot, ax=ax[0, 2]).set_label(
        label=r"$\frac{\partial P}{ \partial g_2}$", size=20
    )
    ax[0, 2].add_patch(rect_ls[2])

    d2p_dg_dg_absmax = 0.5 * max(
        np.max(np.abs(d2p_dg1_dg1_m1mr_m2mr)),
        np.max(np.abs(d2p_dg2_dg2_m1mr_m2mr)),
    )

    d2p_dg1_dg1_plot = ax[1, 0].imshow(
        d2p_dg1_dg1_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(log10mf[0], log10mf[-1], mrmf[0], mrmf[-1]),
        cmap="RdBu",
        interpolation="none",
        vmin=-d2p_dg_dg_absmax,
        vmax=d2p_dg_dg_absmax,
    )
    ax[1, 0].set_xlabel(r"$\log_{10}M_f$", fontsize=20)
    ax[1, 0].set_ylabel(r"$M_r/M_f$", fontsize=20)
    plt.colorbar(d2p_dg1_dg1_plot, ax=ax[1, 0]).set_label(
        label=r"$\frac{\partial^2 P}{\partial g_1^2}$", size=20
    )
    ax[1, 0].add_patch(rect_ls[3])

    d2p_dg2_dg2_plot = ax[1, 1].imshow(
        d2p_dg2_dg2_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(log10mf[0], log10mf[-1], mrmf[0], mrmf[-1]),
        cmap="RdBu",
        interpolation="none",
        vmin=-d2p_dg_dg_absmax,
        vmax=d2p_dg_dg_absmax,
    )
    ax[1, 1].set_xlabel(r"$\log_{10}M_f$", fontsize=20)
    ax[1, 1].set_ylabel(r"$M_r/M_f$", fontsize=20)
    plt.colorbar(d2p_dg2_dg2_plot, ax=ax[1, 1]).set_label(
        label=r"$\frac{\partial^2 P}{\partial g_2^2}$", size=20
    )
    ax[1, 1].add_patch(rect_ls[4])

    d2p_dg1_dg2_plot = ax[1, 2].imshow(
        d2p_dg1_dg2_m1mr_m2mr,
        aspect="auto",
        origin="lower",
        extent=(log10mf[0], log10mf[-1], mrmf[0], mrmf[-1]),
        cmap="RdBu",
        interpolation="none",
        vmin=0.5 * -np.max(np.abs(d2p_dg1_dg2_m1mr_m2mr)),
        vmax=0.5 * np.max(np.abs(d2p_dg1_dg2_m1mr_m2mr)),
    )
    ax[1, 2].set_xlabel(r"$\log_{10}M_f$", fontsize=20)
    ax[1, 2].set_ylabel(r"$M_r/M_f$", fontsize=20)
    plt.colorbar(d2p_dg1_dg2_plot, ax=ax[1, 2]).set_label(
        label=r"$\frac{\partial^2 P}{\partial g_1 \partial g_2}$", size=20
    )
    ax[1, 2].add_patch(rect_ls[5])
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Q and R distribution histograms
# ---------------------------------------------------------------------------


def plot_Q_distributions(
    Q_tot_p: jax.Array,
    Q_tot_m: jax.Array,
    Q_p_tmpl: jax.Array | None = None,
    Q_m_tmpl: jax.Array | None = None,
    Q_tot_p_sim: jax.Array | None = None,
    Q_tot_m_sim: jax.Array | None = None,
    lim: float = 15,
) -> None:
    """Plot histograms of the per-object Q (first shear derivative) distributions.

    Optionally overlays template and simulation Q distributions for comparison.

    Parameters
    ----------
    Q_tot_p : jax.Array, shape (N, 2)
        Per-object Q vectors for the +shear catalogue (flow).
    Q_tot_m : jax.Array, shape (N, 2)
        Per-object Q vectors for the -shear catalogue (flow).
    Q_p_tmpl : jax.Array or None, optional
        Template Q vectors for +shear.
    Q_m_tmpl : jax.Array or None, optional
        Template Q vectors for -shear.
    Q_tot_p_sim : jax.Array or None, optional
        Simulation Q vectors for +shear.
    Q_tot_m_sim : jax.Array or None, optional
        Simulation Q vectors for -shear.
    lim : float, optional
        Histogram range ``(-lim, lim)``.  Default is 15.
    """
    plt.figure()
    plt.hist(
        Q_tot_p[:, 0],
        bins=50,
        histtype="step",
        label="Q1 (shear +0.02)",
        range=(-lim, lim),
    )
    plt.hist(
        Q_tot_m[:, 0],
        bins=50,
        histtype="step",
        label="Q1 (shear -0.02)",
        range=(-lim, lim),
    )
    plt.xlabel("Q")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    if Q_p_tmpl is not None:
        plt.figure()
        plt.hist(
            Q_p_tmpl[:, 0],
            bins=50,
            histtype="step",
            label="Q1 (shear +0.02)",
            range=(-lim, lim),
        )
        plt.hist(
            Q_m_tmpl[:, 0],
            bins=50,
            histtype="step",
            label="Q1 (shear -0.02)",
            range=(-lim, lim),
        )
        plt.xlabel("Q")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    if Q_tot_p_sim is not None:
        plt.figure()
        plt.hist(
            Q_tot_p_sim[:, 0],
            bins=50,
            histtype="step",
            label="Q1 (shear +0.02)",
            range=(-lim, lim),
        )
        plt.hist(
            Q_tot_m_sim[:, 0],
            bins=50,
            histtype="step",
            label="Q1 (shear -0.02)",
            range=(-lim, lim),
        )
        plt.xlabel("Q")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


def plot_R_distributions(
    R_tot_p: jax.Array,
    R_tot_m: jax.Array,
    R_p_tmpl: jax.Array | None = None,
    R_m_tmpl: jax.Array | None = None,
    R_tot_p_sim: jax.Array | None = None,
    R_tot_m_sim: jax.Array | None = None,
) -> None:
    """Plot histograms of the per-object R (second shear derivative) distributions.

    Optionally overlays template and simulation R distributions for comparison.

    Parameters
    ----------
    R_tot_p : jax.Array, shape (N, 2, 2)
        Per-object R matrices for the +shear catalogue (flow).
    R_tot_m : jax.Array, shape (N, 2, 2)
        Per-object R matrices for the -shear catalogue (flow).
    R_p_tmpl : jax.Array or None, optional
        Template R matrices for +shear.
    R_m_tmpl : jax.Array or None, optional
        Template R matrices for -shear.
    R_tot_p_sim : jax.Array or None, optional
        Simulation R matrices for +shear.
    R_tot_m_sim : jax.Array or None, optional
        Simulation R matrices for -shear.
    """
    plt.figure()
    plt.title("R (Flow)")
    plt.hist(
        R_tot_p[:, 0, 0],
        bins=50,
        histtype="step",
        range=(-5, 20),
        label="R11 (shear +0.02)",
    )
    plt.hist(
        R_tot_m[:, 0, 0],
        bins=50,
        histtype="step",
        range=(-5, 20),
        label="R11 (shear -0.02)",
    )
    plt.hist(
        R_tot_p[:, 0, 1],
        bins=50,
        histtype="step",
        range=(-5, 20),
        label="R12 (shear +0.02)",
    )
    plt.hist(
        R_tot_m[:, 0, 1],
        bins=50,
        histtype="step",
        range=(-5, 20),
        label="R12 (shear -0.02)",
    )
    plt.xlabel("R")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    if R_p_tmpl is not None:
        plt.figure()
        plt.title("R (Templates)")
        plt.hist(
            R_p_tmpl[:, 0, 0],
            bins=50,
            histtype="step",
            range=(-5, 20),
            label="R11 (shear +0.02)",
        )
        plt.hist(
            R_m_tmpl[:, 0, 0],
            bins=50,
            histtype="step",
            range=(-5, 20),
            label="R11 (shear -0.02)",
        )
        plt.hist(
            R_p_tmpl[:, 0, 1],
            bins=50,
            histtype="step",
            range=(-5, 20),
            label="R12 (shear +0.02)",
        )
        plt.hist(
            R_m_tmpl[:, 0, 1],
            bins=50,
            histtype="step",
            range=(-5, 20),
            label="R12 (shear -0.02)",
        )
        plt.xlabel("R")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    if R_tot_p_sim is not None:
        plt.figure()
        plt.title("R (Real Pipeline)")
        plt.hist(
            R_tot_p_sim[:, 0, 0],
            bins=50,
            histtype="step",
            range=(-5, 20),
            label="R11 (shear +0.02)",
        )
        plt.hist(
            R_tot_m_sim[:, 0, 0],
            bins=50,
            histtype="step",
            range=(-5, 20),
            label="R11 (shear -0.02)",
        )
        plt.hist(
            R_tot_p_sim[:, 0, 1],
            bins=50,
            histtype="step",
            range=(-5, 20),
            label="R12 (shear +0.02)",
        )
        plt.hist(
            R_tot_m_sim[:, 0, 1],
            bins=50,
            histtype="step",
            range=(-5, 20),
            label="R12 (shear -0.02)",
        )
        plt.xlabel("R")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


# ---------------------------------------------------------------------------
# g estimate comparison scatter
# ---------------------------------------------------------------------------


def plot_g_comparison(
    g_est_p: jax.Array,
    g_est_p_tmpl: jax.Array,
    g_est_m: jax.Array,
    g_est_m_tmpl: jax.Array,
    n: int = 10000,
    lim: float = 1,
) -> None:
    """Compare per-object shear estimates from the flow against template-based estimates.

    Produces hexbin scatter plots of ``g1`` flow estimates vs template estimates
    for the + and - shear catalogues.

    Parameters
    ----------
    g_est_p : jax.Array, shape (N, 2)
        Flow per-object shear estimates for the +shear catalogue.
    g_est_p_tmpl : jax.Array, shape (N, 2)
        Template-based shear estimates for the +shear catalogue.
    g_est_m : jax.Array, shape (N, 2)
        Flow per-object shear estimates for the -shear catalogue.
    g_est_m_tmpl : jax.Array, shape (N, 2)
        Template-based shear estimates for the -shear catalogue.
    n : int, optional
        Number of objects to include in the scatter plot.  Default is 10 000.
    lim : float, optional
        Axis limits ``(-lim, lim)``.  Default is 1.
    """
    z = np.linspace(-lim, lim, 100)
    plt.figure(figsize=(8, 6))
    plt.hexbin(
        jnp.clip(g_est_p_tmpl[:n, 0], -lim, lim),
        jnp.clip(g_est_p[:n, 0], -lim, lim),
        bins="log",
        extent=(-lim, lim, -lim, lim),
        edgecolors="none",
    )
    plt.plot(z, z, color="r", alpha=0.5)
    plt.colorbar()
    plt.xlabel("g1 (Templates)")
    plt.ylabel("g1 (Flow)")
    plt.grid()
    plt.axis("equal")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hexbin(
        jnp.clip(g_est_m_tmpl[:n, 0], -lim, lim),
        jnp.clip(g_est_m[:n, 0], -lim, lim),
        bins="log",
        extent=(-lim, lim, -lim, lim),
        edgecolors="none",
    )
    plt.plot(z, z, color="r", alpha=0.5)
    plt.colorbar()
    plt.xlabel("g1 (Templates)")
    plt.ylabel("g1 (Flow)")
    plt.grid()
    plt.axis("equal")
    plt.show()


# ---------------------------------------------------------------------------
# g1/g2 distribution histograms
# ---------------------------------------------------------------------------


def plot_g_distributions(
    g_est_p: jax.Array, g_est_m: jax.Array, gmax: float = 1, bins: int = 101
) -> None:
    """Plot normalised histograms of the per-object g1 and g2 shear estimates.

    Produces two figures — one for each shear component — overlaying the +
    and - shear catalogues.

    Parameters
    ----------
    g_est_p : jax.Array, shape (N, 2)
        Per-object shear estimates for the +shear catalogue.
    g_est_m : jax.Array, shape (N, 2)
        Per-object shear estimates for the -shear catalogue.
    gmax : float, optional
        Histogram range ``(-gmax, gmax)``.  Default is 1.
    bins : int, optional
        Number of histogram bins.  Default is 101.
    """
    plt.figure()
    plt.hist(
        g_est_p[:, 0],
        bins=bins,
        histtype="step",
        label=r"Estimated $g_1$ (shear +0.02)",
        range=(-gmax, gmax),
        density=True,
    )
    plt.hist(
        g_est_m[:, 0],
        bins=bins,
        histtype="step",
        label=r"Estimated $g_1$ (shear -0.02)",
        range=(-gmax, gmax),
        density=True,
    )
    plt.xlabel(r"$g_1$")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(
        g_est_p[:, 1],
        bins=bins,
        histtype="step",
        label=r"Estimated $g_1$ (shear +0.02)",
        range=(-gmax, gmax),
        density=True,
    )
    plt.hist(
        g_est_m[:, 1],
        bins=bins,
        histtype="step",
        label=r"Estimated $g_1$ (shear -0.02)",
        range=(-gmax, gmax),
        density=True,
    )
    plt.xlabel(r"$g_2$")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Multiplicative bias map
# ---------------------------------------------------------------------------


def plot_mult_bias_hexbin(
    targets_10k: jax.Array,
    pqr_arr: jax.Array,
    pqr2multbias_fn: Callable,
    vmin: float = -1,
    vmax: float = 1,
) -> None:
    """Plot a hexbin map of multiplicative bias in (log10 Mf, Mr/Mf) space.

    Parameters
    ----------
    targets_10k : jax.Array, shape (N, 4)
        Raw template moments used to define the x / y axes.
    pqr_arr : jax.Array, shape (N, ...)
        PQR array passed to ``pqr2multbias_fn`` as the ``reduce_C_function``
        argument in ``plt.hexbin``.
    pqr2multbias_fn : callable
        Function that maps a sub-array of ``pqr_arr`` to a scalar
        multiplicative bias value (used as the hexbin reduction function).
    vmin : float, optional
        Minimum of the colour scale.  Default is -1.
    vmax : float, optional
        Maximum of the colour scale.  Default is 1.
    """
    plt.figure()
    plt.hexbin(
        jnp.log10(targets_10k[:, 0]),
        targets_10k[:, 1] / targets_10k[:, 0],
        gridsize=20,
        C=pqr_arr,
        reduce_C_function=pqr2multbias_fn,
        edgecolors="none",
        vmin=vmin,
        vmax=vmax,
        mincnt=5,
        cmap="managua",
        extent=(np.log10(1500), np.log10(90000), 2.2, 3.5),
    )
    plt.colorbar()
    plt.xlabel(r"$\log_{10}(M_f)$")
    plt.ylabel(r"$M_r/M_f$")
    plt.show()
