"""
plot_shear_diff_hexbin.py
==========================
2-D hexbin, flux--size plane (log10 Mf, Mr/Mf), of the **mean-shear difference**
between the analytic-sim PQR and the flow PQR:

    Δc_k = c_sim,k - c_flow,k,   c_k = (g(+)_k + g(-)_k) / 2   (k = g1, g2)

c_k is the same additive-bias combination used elsewhere in this repo (sums
the +/- arms, cancelling the injected-shear lever). 2x2 panels: top row = raw
Δc_k, bottom row = significance Δc_k / σ(Δc_k), one column per component.

σ is a **paired** bootstrap: sim and flow PQR come from the *same* targets
(same per-object noise draw, two different estimators), so they're strongly
correlated. Each bootstrap draw resamples target indices once per arm and
applies that same resampling to both the flow and sim PQR -- this correctly
propagates the correlation, unlike combining independently-bootstrapped
flow/sim errors in quadrature (which would overstate σ).

Run::

    python -m bfd_cnf.plot_shear_diff_hexbin --in data/pqr_indep_xy_elbo_1M.npz
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .config import PLOTS_DIR
from .plot_mbias_hexbin_independent import (
    BOX_MF, BOX_MRMF, _MF_LIM_PAD, _Y_LIM_PAD, CORNER_ASPECT, _g_from_pqr,
)


def _paired_arm(pqr: np.ndarray, pqr_sim: np.ndarray, targets: np.ndarray, yaxis: str):
    """(x=log10 Mf, y, pqr, pqr_sim), keeping targets valid in *both* fields so the
    two arrays stay index-aligned for paired resampling."""
    mf = np.abs(targets[:, 0])
    mr = np.abs(targets[:, 1])
    x = np.log10(mf)
    y = mr / mf if yaxis == "mr_mf" else np.log10(mr)
    valid = (
        np.all(np.isfinite(pqr), axis=1) & (pqr[:, 0] > 0)
        & np.all(np.isfinite(pqr_sim), axis=1) & (pqr_sim[:, 0] > 0)
        & np.isfinite(x) & np.isfinite(y)
    )
    return x[valid], y[valid], pqr[valid], pqr_sim[valid]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_indep_xy_elbo_1M.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gridsize", type=int, default=38)
    ap.add_argument("--mincnt-per-arm", type=int, default=80,
                    help="Min targets PER ARM in a cell for a stable estimate (default 80).")
    ap.add_argument("--siglim", type=float, default=3.0,
                    help="Symmetric colour limit for the significance panels (default 3.0).")
    ap.add_argument("--clim", type=float, default=None,
                    help="Symmetric colour limit for the raw Δg panels (default: robust "
                         "92nd pct of |Δg| across both components).")
    ap.add_argument("--yaxis", choices=["mr_mf", "logMr"], default="mr_mf")
    ap.add_argument("--nboot", type=int, default=50,
                    help="Paired bootstrap resamples per cell (default 50).")
    ap.add_argument("--mf-lim", type=float, nargs=2, default=_MF_LIM_PAD)
    ap.add_argument("--y-lim", type=float, nargs=2, default=_Y_LIM_PAD)
    args = ap.parse_args()

    d = np.load(args.inp)
    xp, yp, pqr_p, sim_p = _paired_arm(
        d["pqr_p"].astype(np.float64), d["pqr_sim_p"].astype(np.float64), d["targets_p"], args.yaxis)
    xm, ym, pqr_m, sim_m = _paired_arm(
        d["pqr_m"].astype(np.float64), d["pqr_sim_m"].astype(np.float64), d["targets_m"], args.yaxis)
    n_p, n_m = xp.shape[0], xm.shape[0]
    ylabel = r"$M_r / M_f$  (size)" if args.yaxis == "mr_mf" else r"$\log_{10} M_r$  (size)"
    print(f"+arm {n_p:,} paired targets, -arm {n_m:,} paired targets")

    x_all = np.concatenate([xp, xm])
    y_all = np.concatenate([yp, ym])
    C = np.arange(n_p + n_m)
    min_arm = args.mincnt_per_arm

    def _c(pqr_arr, idx_p, idx_m, k) -> float:
        gp = _g_from_pqr(pqr_arr[0][idx_p])
        gm = _g_from_pqr(pqr_arr[1][idx_m])
        return (gp[k] + gm[k]) / 2.0

    def make_reduce(k: int):
        def reduce_val(carried) -> float:
            idx = np.asarray(carried, dtype=np.int64)
            ip = idx[idx < n_p]
            im = idx[idx >= n_p] - n_p
            if ip.size < min_arm or im.size < min_arm:
                return np.nan
            return _c((sim_p, sim_m), ip, im, k) - _c((pqr_p, pqr_m), ip, im, k)
        return reduce_val

    _rng = np.random.default_rng(0)

    def make_reduce_sigma(k: int):
        def reduce_sig(carried) -> float:
            idx = np.asarray(carried, dtype=np.int64)
            ip = idx[idx < n_p]
            im = idx[idx >= n_p] - n_p
            if ip.size < min_arm or im.size < min_arm:
                return np.nan
            vals = np.empty(args.nboot)
            for b in range(args.nboot):
                sp = ip[_rng.integers(0, ip.size, ip.size)]
                sm = im[_rng.integers(0, im.size, im.size)]
                # same resampled indices applied to flow AND sim -> preserves correlation
                vals[b] = _c((sim_p, sim_m), sp, sm, k) - _c((pqr_p, pqr_m), sp, sm, k)
            return float(np.nanstd(vals))
        return reduce_sig

    if args.yaxis == "mr_mf":
        xlo, xhi = np.log10(args.mf_lim[0]), np.log10(args.mf_lim[1])
        ylo, yhi = args.y_lim
    else:
        xlo, xhi = x_all.min(), x_all.max()
        ylo, yhi = y_all.min(), y_all.max()
    extent = (xlo, xhi, ylo, yhi)
    gridsize = args.gridsize
    box_aspect = None
    if args.yaxis == "mr_mf":
        box_aspect = CORNER_ASPECT * (extent[3] - extent[2]) / (extent[1] - extent[0])
        gridsize = (args.gridsize, max(1, round(args.gridsize * box_aspect / math.sqrt(3))))
    hexkw = dict(gridsize=gridsize, extent=extent)

    panel_w = 9.0
    panel_h = panel_w * box_aspect if box_aspect is not None else panel_w * 6.6 / 7.7
    fig, ax2d = plt.subplots(2, 2, figsize=(2 * (panel_w + 1.5), 2 * (panel_h + 1.3)),
                             constrained_layout=True)
    ax = [ax2d[0, 0], ax2d[0, 1], ax2d[1, 0], ax2d[1, 1]]
    fig.suptitle("Mean-shear difference: Fiducial − Flow", fontsize=16)
    if box_aspect is not None:
        for a in ax:
            a.set_box_aspect(box_aspect)

    # pass 1: raw Δg + per-cell σ for both components (need clim across both before drawing)
    raw = {}
    for k, label in ((0, "g_1"), (1, "g_2")):
        hb = ax[k].hexbin(x_all, y_all, C=C, reduce_C_function=make_reduce(k),
                          mincnt=2 * min_arm, **hexkw)
        vals = np.ma.filled(hb.get_array().astype(float), np.nan)
        hb_s = ax[k].hexbin(x_all, y_all, C=C, reduce_C_function=make_reduce_sigma(k),
                            mincnt=2 * min_arm, **hexkw)
        s_vals = np.ma.filled(hb_s.get_array().astype(float), np.nan)
        hb_s.remove()
        raw[k] = (hb, vals, s_vals)

    clim = args.clim
    if clim is None:
        both = np.concatenate([raw[0][1], raw[1][1]])
        clim = float(np.nanpercentile(np.abs(both[np.isfinite(both)]), 92))
    clim = max(clim, 1e-6)

    for k, label in ((0, "g_1"), (1, "g_2")):
        hb, vals, s_vals = raw[k]
        hb.set_cmap("RdBu_r")
        hb.set_clim(-clim, clim)
        finite = vals[np.isfinite(vals)]
        fig.colorbar(hb, ax=ax[k]).set_label(rf"$\Delta {label}$  (Fiducial $-$ Flow)", size=12)
        ax[k].set_title(
            rf"$\Delta {label}$  median={np.nanmedian(finite):+.4f}, "
            rf"[p5,p95]=[{np.nanpercentile(finite,5):+.3f},{np.nanpercentile(finite,95):+.3f}]",
            fontsize=10)

        z = vals / np.where(s_vals > 0, s_vals, np.nan)
        panel = k + 2
        hbz = ax[panel].hexbin(x_all, y_all, C=C, reduce_C_function=make_reduce(k),
                               mincnt=2 * min_arm, **hexkw)
        hbz.set_array(np.ma.masked_invalid(z))
        hbz.set_cmap("RdBu_r")
        zfin = z[np.isfinite(z)]
        hbz.set_clim(-args.siglim, args.siglim)
        fig.colorbar(hbz, ax=ax[panel]).set_label(rf"significance  $\Delta {label}/\sigma$", size=12)
        ax[panel].set_title(
            rf"$\Delta {label}/\sigma$  (paired bootstrap $\sigma$, n={args.nboot}); "
            rf"$|\Delta {label}/\sigma|>2$ in {100*np.mean(np.abs(zfin) > 2):.0f}% of cells",
            fontsize=10)
        print(f"{label}: {zfin.size} cells; Δ median={np.nanmedian(finite):+.5f}; "
              f"σ median={np.nanmedian(s_vals):.5f}; "
              f"|Δ/σ|>2 in {100*np.mean(np.abs(zfin) > 2):.1f}% of cells")

    xline = np.array(extent[:2])
    bx0, bx1 = np.log10(BOX_MF[0]), np.log10(BOX_MF[1])
    for a in ax:
        a.set_xlabel(r"$\log_{10} M_f$  (flux)", fontsize=13)
        a.set_ylabel(ylabel, fontsize=13)
        if args.yaxis == "mr_mf":
            a.add_patch(mpatches.Rectangle(
                (bx0, BOX_MRMF[0]), bx1 - bx0, BOX_MRMF[1] - BOX_MRMF[0],
                lw=1.3, edgecolor="tab:green", facecolor="none", zorder=100))
        else:
            for r in BOX_MRMF:
                a.plot(xline, xline + np.log10(r), color="tab:green", ls="--", lw=1)
        a.set_xlim(extent[0], extent[1])
        a.set_ylim(extent[2], extent[3])

    tag = os.path.basename(args.inp).replace(".npz", "") + f"_{args.yaxis}"
    out = args.out or os.path.join(PLOTS_DIR, f"shear_diff_hexbin_{tag}.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
