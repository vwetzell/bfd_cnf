"""
plot_mbias_hexbin_independent.py
================================
2-D hexbin of the multiplicative bias ``m`` in the flux--size plane
``(log10 Mf, log10 Mr)``, from an **independent-ensemble** PQR npz (the +/- arms
are different objects of different length — produced by
``integrate_grid --independent``).

``m`` is a population quantity, not per-target:

    m_cell = (g1(+) - g1(-)) / Δg - 1,    Δg = 0.04

For each hexagonal cell the +shear and -shear targets are summed **separately**
(``g(±) = R_tot⁻¹ Q_tot`` accumulated over that arm's targets in the cell), and
m is formed from the two summed shears.  Because the arms aren't paired, both
are binned onto the *same* hex grid (one hexbin over the concatenated points)
and split back out by arm inside the reduce function; a cell is only coloured
when **both** arms have at least ``--mincnt-per-arm`` targets.

Run::

    python -m bfd_cnf.plot_mbias_hexbin_independent \
        --in data/pqr_grid_independent_FULL_mf3000_90000_newtmpl.npz
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
from .statistics import qr_log_totals

DELTA_G = 0.04
BOX_MF = (1500.0, 90000.0)  # green selection box flux bounds
BOX_MRMF = (2.2, 3.5)  # green selection box Mr/Mf bounds

# Default plot range: box bounds + 8% padding (in the plotted units -- log10 for Mf, linear
# for Mr/Mf) so the green box renders as a visible rectangle instead of hugging the panel edge.
_PAD = 0.08
_BOX_MF_LOG = (np.log10(BOX_MF[0]), np.log10(BOX_MF[1]))
_MF_LIM_PAD = tuple(10 ** np.array([
    _BOX_MF_LOG[0] - _PAD * (_BOX_MF_LOG[1] - _BOX_MF_LOG[0]),
    _BOX_MF_LOG[1] + _PAD * (_BOX_MF_LOG[1] - _BOX_MF_LOG[0]),
]))
_Y_LIM_PAD = (BOX_MRMF[0] - _PAD * (BOX_MRMF[1] - BOX_MRMF[0]),
              BOX_MRMF[1] + _PAD * (BOX_MRMF[1] - BOX_MRMF[0]))

# Physical (screen) aspect ratio of the flux-size panel in plot_corner.py: that panel is
# square in inches with PLOT_RANGE log10Mf=(log10 500, log10 200000), Mr/Mf=(1.2, 5.5), so
# matching this data-range ratio via ax.set_aspect renders our green box with the same
# on-screen shape as it has there, regardless of our own axis limits.
CORNER_ASPECT = (np.log10(200000.0) - np.log10(500.0)) / (5.5 - 1.2)


def _g_from_pqr(pqr: np.ndarray) -> np.ndarray:
    """Maximum-likelihood shear from a summed PQR block (pqr2g accumulation)."""
    keep = pqr[:, 0] >= 1e-10
    pqr = pqr[keep]
    if pqr.shape[0] < 3:
        return np.array([np.nan, np.nan])
    qt, Rtot = qr_log_totals(pqr)
    try:
        return np.linalg.solve(np.nansum(Rtot, axis=0), np.nansum(qt, axis=0))
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan])


def _arm_coords(pqr: np.ndarray, targets: np.ndarray, yaxis: str):
    """Return (x=log10 Mf, y, pqr) keeping only finite, P>0 targets.

    ``yaxis`` selects the size axis: ``"mr_mf"`` → Mr/Mf, ``"logMr"`` → log10 Mr.
    """
    mf = np.abs(targets[:, 0])
    mr = np.abs(targets[:, 1])
    x = np.log10(mf)
    y = mr / mf if yaxis == "mr_mf" else np.log10(mr)
    valid = (
        np.all(np.isfinite(pqr), axis=1) & (pqr[:, 0] > 0)
        & np.isfinite(x) & np.isfinite(y)
    )
    return x[valid], y[valid], pqr[valid]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp",
                    default="data/pqr_grid_independent_FULL_mf3000_90000_newtmpl.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gridsize", type=int, default=38)
    ap.add_argument("--mincnt-per-arm", type=int, default=80,
                    help="Min targets PER ARM in a cell for a stable m (default 80).")
    ap.add_argument("--clim", type=float, default=1.0,
                    help="Symmetric m colour limit (default 1.0).")
    ap.add_argument("--siglim", type=float, default=3.0,
                    help="Symmetric colour limit for the m/σ, c1/σ, c2/σ significance "
                         "panels (default 3.0).")
    ap.add_argument("--yaxis", choices=["mr_mf", "logMr"], default="mr_mf",
                    help="Size axis: Mr/Mf (default, filled band) or log10 Mr.")
    ap.add_argument("--nboot", type=int, default=50,
                    help="Bootstrap resamples per cell for the m error σ (default 50).")
    ap.add_argument("--mf-lim", type=float, nargs=2, default=_MF_LIM_PAD,
                    help="x-axis Mf range (linear; default = green box bounds + 8%% pad).")
    ap.add_argument("--y-lim", type=float, nargs=2, default=_Y_LIM_PAD,
                    help="y-axis Mr/Mf range (default = green box bounds + 8%% pad; mr_mf only).")
    ap.add_argument("--field", choices=["pqr", "pqr_sim"], default="pqr",
                    help="Which PQR to bin: flow ('pqr', default) or analytic sim ('pqr_sim').")
    args = ap.parse_args()

    d = np.load(args.inp)
    xp, yp, pqr_p = _arm_coords(d[f"{args.field}_p"].astype(np.float64), d["targets_p"], args.yaxis)
    xm, ym, pqr_m = _arm_coords(d[f"{args.field}_m"].astype(np.float64), d["targets_m"], args.yaxis)
    n_p, n_m = xp.shape[0], xm.shape[0]
    ylabel = r"$M_r / M_f$  (size)" if args.yaxis == "mr_mf" else r"$\log_{10} M_r$  (size)"
    print(f"+arm {n_p:,} targets, -arm {n_m:,} targets;  "
          f"log10Mf [{min(xp.min(), xm.min()):.2f},{max(xp.max(), xm.max()):.2f}]  "
          f"y({args.yaxis}) [{min(yp.min(), ym.min()):.2f},{max(yp.max(), ym.max()):.2f}]")

    # Concatenate both arms onto ONE hex grid; carried index < n_p => +arm.
    x_all = np.concatenate([xp, xm])
    y_all = np.concatenate([yp, ym])
    C = np.arange(n_p + n_m)
    min_arm = args.mincnt_per_arm

    def _formula(kind: str, gp: np.ndarray, gm: np.ndarray) -> float:
        if kind == "m":
            return (gp[0] - gm[0]) / DELTA_G - 1.0
        if kind == "c1":
            return (gp[0] + gm[0]) / 2.0
        return (gp[1] + gm[1]) / 2.0  # "c2"

    def make_reduce(kind: str):
        def reduce_val(carried) -> float:
            idx = np.asarray(carried, dtype=np.int64)
            ip = idx[idx < n_p]
            im = idx[idx >= n_p] - n_p
            if ip.size < min_arm or im.size < min_arm:
                return np.nan
            return _formula(kind, _g_from_pqr(pqr_p[ip]), _g_from_pqr(pqr_m[im]))
        return reduce_val

    _rng = np.random.default_rng(0)

    def make_reduce_sigma(kind: str):
        """Per-cell bootstrap error on ``kind``, resampling each arm independently."""
        def reduce_sig(carried) -> float:
            idx = np.asarray(carried, dtype=np.int64)
            ip = idx[idx < n_p]
            im = idx[idx >= n_p] - n_p
            if ip.size < min_arm or im.size < min_arm:
                return np.nan
            vals = np.empty(args.nboot)
            for b in range(args.nboot):
                sp = pqr_p[ip[_rng.integers(0, ip.size, ip.size)]]
                sm = pqr_m[im[_rng.integers(0, im.size, im.size)]]
                vals[b] = _formula(kind, _g_from_pqr(sp), _g_from_pqr(sm))
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
        # hexbin's default ny=nx/sqrt(3) only gives regular hexagons when the axes is
        # physically square; our panels are physically wide/flat (box_aspect = height/width),
        # so scale ny down to match, or the hexagons render squashed vertically.
        gridsize = (args.gridsize, max(1, round(args.gridsize * box_aspect / math.sqrt(3))))
    hexkw = dict(gridsize=gridsize, extent=extent)

    panel_w = 9.0
    panel_h = panel_w * box_aspect if box_aspect is not None else panel_w * 6.6 / 7.7
    fig, ax2d = plt.subplots(2, 2, figsize=(2 * (panel_w + 1.5), 2 * (panel_h + 1.3)),
                             constrained_layout=True)
    ax = [ax2d[0, 0], ax2d[0, 1], ax2d[1, 0], ax2d[1, 1]]
    fig.suptitle("Fiducial BFD" if args.field == "pqr_sim" else "Flow BFD", fontsize=16)
    if box_aspect is not None:
        # set_box_aspect fixes the panel's own physical shape (height/width) up front, so
        # layout + colorbar placement account for it -- unlike set_aspect(adjustable="box"),
        # which shrinks the axes only at draw time and leaves the colorbar stranded at the
        # pre-shrink position.
        for a in ax:
            a.set_box_aspect(box_aspect)

    # ---- m hexbin (sum +/- separately per cell, then m from the sums) ----
    hb = ax[0].hexbin(x_all, y_all, C=C, reduce_C_function=make_reduce("m"),
                      mincnt=2 * min_arm, cmap="RdBu_r", **hexkw)
    vals = np.ma.filled(hb.get_array().astype(float), np.nan)
    finite = vals[np.isfinite(vals)]
    clim = args.clim
    hb.set_clim(-clim, clim)
    print(f"{finite.size} cells with >= {min_arm}/arm;  m per-cell: "
          f"median={np.median(finite):+.4f}  "
          f"[p5,p95]=[{np.percentile(finite, 5):+.3f},{np.percentile(finite, 95):+.3f}]  "
          f"colour limit ±{clim:.3f}")
    fig.colorbar(hb, ax=ax[0]).set_label("multiplicative bias  m", size=12)
    ax[0].set_title(f"m in flux-size plane (sum ± separately)\n{os.path.basename(args.inp)}",
                    fontsize=10)

    # ---- significance hexbin: m / σ, σ = per-cell bootstrap error ----
    # Identical binning (same x_all/y_all/gridsize/extent) ⇒ cell order matches hb.
    hb_s = ax[1].hexbin(x_all, y_all, C=C, reduce_C_function=make_reduce_sigma("m"),
                        mincnt=2 * min_arm, **hexkw)
    s_vals = np.ma.filled(hb_s.get_array().astype(float), np.nan)
    z = vals / np.where(s_vals > 0, s_vals, np.nan)
    hb_s.set_array(np.ma.masked_invalid(z))
    hb_s.set_cmap("RdBu_r")
    zfin = z[np.isfinite(z)]
    hb_s.set_clim(-args.siglim, args.siglim)
    fig.colorbar(hb_s, ax=ax[1]).set_label(r"significance  $m/\sigma$", size=12)
    ax[1].set_title(rf"$m/\sigma$  (per-cell bootstrap $\sigma$, n={args.nboot}); "
                    rf"$|m/\sigma|>2$ in {100*np.mean(np.abs(zfin) > 2):.0f}% of cells",
                    fontsize=10)

    print(f"per-cell σ: median={np.nanmedian(s_vals):.4f}; "
          f"|m/σ|>2 in {100*np.mean(np.abs(zfin) > 2):.1f}% of cells")

    # ---- additive bias significance: c1/σ_c1 = (g1+ + g1-)/2 / bootstrap σ, c2/σ_c2 likewise ----
    for panel, kind, label in ((2, "c1", "c1"), (3, "c2", "c2")):
        hbc = ax[panel].hexbin(x_all, y_all, C=C, reduce_C_function=make_reduce(kind),
                               mincnt=2 * min_arm, cmap="RdBu_r", **hexkw)
        cvals = np.ma.filled(hbc.get_array().astype(float), np.nan)
        hbc_s = ax[panel].hexbin(x_all, y_all, C=C, reduce_C_function=make_reduce_sigma(kind),
                                 mincnt=2 * min_arm, **hexkw)
        c_svals = np.ma.filled(hbc_s.get_array().astype(float), np.nan)
        cz = cvals / np.where(c_svals > 0, c_svals, np.nan)
        hbc.set_array(np.ma.masked_invalid(cz))
        hbc_s.remove()
        czfin = cz[np.isfinite(cz)]
        hbc.set_clim(-args.siglim, args.siglim)
        fig.colorbar(hbc, ax=ax[panel]).set_label(rf"significance  ${label}/\sigma$", size=12)
        ax[panel].set_title(rf"${label}/\sigma$  (per-cell bootstrap $\sigma$, n={args.nboot}); "
                            rf"$|{label}/\sigma|>2$ in {100*np.mean(np.abs(czfin) > 2):.0f}% of cells",
                            fontsize=10)

    # Selection box (green) + stellar locus (red), matching plot_corner.py.
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

    field_tag = "" if args.field == "pqr" else f"_{args.field}"
    tag = (os.path.basename(args.inp).replace("pqr_grid_independent_", "").replace(".npz", "")
           + f"_{args.yaxis}{field_tag}")
    out = args.out or os.path.join(PLOTS_DIR, f"mbias_hexbin_indep_{tag}.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # ---- counts, separate figure (field-independent: same x_all/y_all for pqr/pqr_sim) ----
    fig_c, ax_c = plt.subplots(1, 1, figsize=(panel_w + 1.5, panel_h + 1.3),
                               constrained_layout=True)
    if args.yaxis == "mr_mf":
        ax_c.set_box_aspect(box_aspect)
    hc = ax_c.hexbin(x_all, y_all, bins="log", cmap="viridis", **hexkw)
    fig_c.colorbar(hc, ax=ax_c).set_label("targets per cell (log, both arms)", size=12)
    ax_c.set_title("target counts", fontsize=11)
    ax_c.set_xlabel(r"$\log_{10} M_f$  (flux)", fontsize=13)
    ax_c.set_ylabel(ylabel, fontsize=13)
    if args.yaxis == "mr_mf":
        ax_c.add_patch(mpatches.Rectangle(
            (bx0, BOX_MRMF[0]), bx1 - bx0, BOX_MRMF[1] - BOX_MRMF[0],
            lw=1.3, edgecolor="tab:green", facecolor="none", zorder=100))
    else:
        for r in BOX_MRMF:
            ax_c.plot(xline, xline + np.log10(r), color="tab:green", ls="--", lw=1)
    ax_c.set_xlim(extent[0], extent[1])
    ax_c.set_ylim(extent[2], extent[3])
    tag_counts = (os.path.basename(args.inp).replace("pqr_grid_independent_", "")
                  .replace(".npz", "") + f"_{args.yaxis}")
    out_c = os.path.join(PLOTS_DIR, f"mbias_hexbin_indep_{tag_counts}_counts.png")
    fig_c.savefig(out_c, dpi=140, bbox_inches="tight")
    plt.close(fig_c)
    print(f"saved {out_c}")


if __name__ == "__main__":
    main()
