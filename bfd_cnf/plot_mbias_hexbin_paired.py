"""
plot_mbias_hexbin_paired.py
============================
2-D hexbin of the multiplicative/additive shear bias in the flux--size plane
``(log10 Mf, Mr/Mf)``, from a **matched +/- ring-test pair** PQR npz (imsims
target output, integrated via ``imsims.integrate_targets`` -- e.g.
``pqr_targets_noiseless_1M_sub32.npz``). Unlike
``plot_mbias_hexbin_independent`` (independent grid ensembles, +/- arms are
different objects), every row here shares its ``id`` with the SAME intrinsic
galaxy + PSF in both arms (only the sign of the applied shear differs) -- see
``imsims/paired_bias.py``, whose ``match_by_id`` this mirrors.

For each hexagonal cell (binned on the pair's own mean flux/size), the +shear
and -shear PQR of the matched galaxies in that cell are summed separately and
``m``/``c1``/``c2`` formed from the two summed shears, exactly as in
``plot_mbias_hexbin_independent``. Per-cell bootstrap error resamples PAIR
indices -- one draw applied to BOTH arms -- so the shared intrinsic
ellipticity's positive p/m correlation is preserved (smaller sigma than
independent-arm resampling; see ``imsims.paired_bias`` docstring).

Run::

    python -m bfd_cnf.plot_mbias_hexbin_paired \
        --in ../bfd_cnf_imsims/data/pqr_targets_noiseless_1M_sub32.npz
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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_IN = os.path.join(
    os.path.dirname(_REPO_ROOT), "bfd_cnf_imsims", "data",
    "pqr_targets_noiseless_1M_sub32.npz",
)


def match_by_id(ids_p, pqr_p, ids_m, pqr_m):
    """Intersect ids (dropping non-finite / P<=0 rows from either arm first);
    return row-aligned matched (pqr_p, pqr_m) + each arm's row index into the
    original arrays. Mirrors imsims.paired_bias.match_by_id."""
    pqr_p, pqr_m = np.asarray(pqr_p, dtype=np.float64), np.asarray(pqr_m, dtype=np.float64)
    good_p = np.all(np.isfinite(pqr_p), axis=1) & (pqr_p[:, 0] > 1e-10)
    good_m = np.all(np.isfinite(pqr_m), axis=1) & (pqr_m[:, 0] > 1e-10)
    idx_p = {int(i): k for k, i in enumerate(np.asarray(ids_p)) if good_p[k]}
    idx_m = {int(i): k for k, i in enumerate(np.asarray(ids_m)) if good_m[k]}
    common = np.array(sorted(set(idx_p) & set(idx_m)))
    kp = np.array([idx_p[i] for i in common])
    km = np.array([idx_m[i] for i in common])
    return pqr_p[kp], pqr_m[km], kp, km


def _xy(targets: np.ndarray, yaxis: str):
    mf = np.abs(targets[:, 0])
    mr = np.abs(targets[:, 1])
    x = np.log10(mf)
    y = mr / mf if yaxis == "mr_mf" else np.log10(mr)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in", dest="inp", default=_DEFAULT_IN)
    ap.add_argument("--out", default=None)
    ap.add_argument("--input-g", type=float, default=0.02, help="applied |g1| (g_p=+g, g_m=-g).")
    ap.add_argument("--gridsize", type=int, default=38)
    ap.add_argument("--mincnt", type=int, default=80,
                    help="Min matched pairs in a cell for a stable estimate (default 80).")
    ap.add_argument("--clim", type=float, default=1.0,
                    help="Symmetric m colour limit (default 1.0).")
    ap.add_argument("--siglim", type=float, default=3.0,
                    help="Symmetric colour limit for the significance panels (default 3.0).")
    ap.add_argument("--yaxis", choices=["mr_mf", "logMr"], default="mr_mf")
    ap.add_argument("--nboot", type=int, default=50,
                    help="Paired bootstrap resamples per cell (default 50).")
    ap.add_argument("--mf-lim", type=float, nargs=2, default=_MF_LIM_PAD)
    ap.add_argument("--y-lim", type=float, nargs=2, default=_Y_LIM_PAD)
    ap.add_argument("--bin-arm", choices=["plus", "minus", "mean"], default="plus",
                    help="Which moment decides cell membership: a single arm's own noisy "
                         "(Mf, Mr) ('plus'/'minus', shear-DEPENDENT label), or the pair-"
                         "averaged 0.5*(plus+minus) ('mean', shear-INDEPENDENT: the +/-g "
                         "response cancels, so cells aren't defined by the sheared position "
                         "-- the correct label per the sub8 ring-test analysis).")
    args = ap.parse_args()

    d = np.load(args.inp)
    pqr_p, pqr_m, kp, km = match_by_id(d["ids_p"], d["pqr_p"], d["ids_m"], d["pqr_m"])
    n = len(kp)
    print(f"Matched {n:,} pairs by id (arms had {len(d['ids_p']):,} / {len(d['ids_m']):,} rows).")

    if args.bin_arm == "mean":
        bin_targets = 0.5 * (np.asarray(d["targets_p"], dtype=np.float64)[kp]
                             + np.asarray(d["targets_m"], dtype=np.float64)[km])
    else:
        bin_key, bin_idx = ("targets_p", kp) if args.bin_arm == "plus" else ("targets_m", km)
        bin_targets = np.asarray(d[bin_key], dtype=np.float64)[bin_idx]
    x, y = _xy(bin_targets, args.yaxis)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, pqr_p, pqr_m = x[valid], y[valid], pqr_p[valid], pqr_m[valid]
    n = len(x)
    ylabel = r"$M_r / M_f$  (size)" if args.yaxis == "mr_mf" else r"$\log_{10} M_r$  (size)"
    print(f"{n:,} matched pairs with finite flux/size; "
          f"log10Mf [{x.min():.2f},{x.max():.2f}]  y({args.yaxis}) [{y.min():.2f},{y.max():.2f}]")

    C = np.arange(n)

    def _formula(kind: str, gp: np.ndarray, gm: np.ndarray) -> float:
        if kind == "m":
            return (gp[0] - gm[0]) / (2 * args.input_g) - 1.0
        if kind == "c1":
            return (gp[0] + gm[0]) / 2.0
        return (gp[1] + gm[1]) / 2.0  # "c2"

    def make_reduce(kind: str):
        def reduce_val(carried) -> float:
            idx = np.asarray(carried, dtype=np.int64)
            if idx.size < args.mincnt:
                return np.nan
            return _formula(kind, _g_from_pqr(pqr_p[idx]), _g_from_pqr(pqr_m[idx]))
        return reduce_val

    _rng = np.random.default_rng(0)

    def make_reduce_sigma(kind: str):
        """Per-cell bootstrap error on `kind`, resampling PAIR indices (same draw
        applied to both arms) so the shared intrinsic ellipticity's p/m correlation
        is preserved rather than overstating sigma."""
        def reduce_sig(carried) -> float:
            idx = np.asarray(carried, dtype=np.int64)
            if idx.size < args.mincnt:
                return np.nan
            vals = np.empty(args.nboot)
            for b in range(args.nboot):
                draw = idx[_rng.integers(0, idx.size, idx.size)]
                vals[b] = _formula(kind, _g_from_pqr(pqr_p[draw]), _g_from_pqr(pqr_m[draw]))
            return float(np.nanstd(vals))
        return reduce_sig

    if args.yaxis == "mr_mf":
        xlo, xhi = np.log10(args.mf_lim[0]), np.log10(args.mf_lim[1])
        ylo, yhi = args.y_lim
    else:
        xlo, xhi = x.min(), x.max()
        ylo, yhi = y.min(), y.max()
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
    fig.suptitle("Paired ring-test bias (matched by id)", fontsize=16)
    if box_aspect is not None:
        for a in ax:
            a.set_box_aspect(box_aspect)

    # ---- m hexbin ----
    hb = ax[0].hexbin(x, y, C=C, reduce_C_function=make_reduce("m"),
                      mincnt=args.mincnt, cmap="RdBu_r", **hexkw)
    vals = np.ma.filled(hb.get_array().astype(float), np.nan)
    finite = vals[np.isfinite(vals)]
    clim = args.clim
    hb.set_clim(-clim, clim)
    print(f"{finite.size} cells with >= {args.mincnt} pairs;  m per-cell: "
          f"median={np.median(finite):+.4f}  "
          f"[p5,p95]=[{np.percentile(finite, 5):+.3f},{np.percentile(finite, 95):+.3f}]  "
          f"colour limit ±{clim:.3f}")
    fig.colorbar(hb, ax=ax[0]).set_label("multiplicative bias  m", size=12)
    ax[0].set_title(f"m in flux-size plane (paired)\n{os.path.basename(args.inp)}", fontsize=10)

    # ---- significance hexbin: m / sigma, sigma = per-cell paired bootstrap error ----
    hb_s = ax[1].hexbin(x, y, C=C, reduce_C_function=make_reduce_sigma("m"),
                        mincnt=args.mincnt, **hexkw)
    s_vals = np.ma.filled(hb_s.get_array().astype(float), np.nan)
    z = vals / np.where(s_vals > 0, s_vals, np.nan)
    hb_s.set_array(np.ma.masked_invalid(z))
    hb_s.set_cmap("RdBu_r")
    zfin = z[np.isfinite(z)]
    hb_s.set_clim(-args.siglim, args.siglim)
    fig.colorbar(hb_s, ax=ax[1]).set_label(r"significance  $m/\sigma$", size=12)
    ax[1].set_title(rf"$m/\sigma$  (paired bootstrap $\sigma$, n={args.nboot}); "
                    rf"$|m/\sigma|>2$ in {100*np.mean(np.abs(zfin) > 2):.0f}% of cells",
                    fontsize=10)
    print(f"per-cell sigma: median={np.nanmedian(s_vals):.4f}; "
          f"|m/sigma|>2 in {100*np.mean(np.abs(zfin) > 2):.1f}% of cells")

    # ---- additive bias significance ----
    for panel, kind, label in ((2, "c1", "c1"), (3, "c2", "c2")):
        hbc = ax[panel].hexbin(x, y, C=C, reduce_C_function=make_reduce(kind),
                               mincnt=args.mincnt, cmap="RdBu_r", **hexkw)
        cvals = np.ma.filled(hbc.get_array().astype(float), np.nan)
        hbc_s = ax[panel].hexbin(x, y, C=C, reduce_C_function=make_reduce_sigma(kind),
                                 mincnt=args.mincnt, **hexkw)
        c_svals = np.ma.filled(hbc_s.get_array().astype(float), np.nan)
        cz = cvals / np.where(c_svals > 0, c_svals, np.nan)
        hbc.set_array(np.ma.masked_invalid(cz))
        hbc_s.remove()
        czfin = cz[np.isfinite(cz)]
        hbc.set_clim(-args.siglim, args.siglim)
        fig.colorbar(hbc, ax=ax[panel]).set_label(rf"significance  ${label}/\sigma$", size=12)
        ax[panel].set_title(rf"${label}/\sigma$  (paired bootstrap $\sigma$, n={args.nboot}); "
                            rf"$|{label}/\sigma|>2$ in {100*np.mean(np.abs(czfin) > 2):.0f}% of cells",
                            fontsize=10)

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
    if args.bin_arm != "plus":
        tag += f"_bin{args.bin_arm}"
    out = args.out or os.path.join(PLOTS_DIR, f"mbias_hexbin_paired_{tag}.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # ---- counts, separate figure ----
    fig_c, ax_c = plt.subplots(1, 1, figsize=(panel_w + 1.5, panel_h + 1.3),
                               constrained_layout=True)
    if args.yaxis == "mr_mf":
        ax_c.set_box_aspect(box_aspect)
    hc = ax_c.hexbin(x, y, bins="log", cmap="viridis", **hexkw)
    fig_c.colorbar(hc, ax=ax_c).set_label("matched pairs per cell (log)", size=12)
    ax_c.set_title("pair counts", fontsize=11)
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
    out_c = os.path.join(PLOTS_DIR, f"mbias_hexbin_paired_{tag}_counts.png")
    fig_c.savefig(out_c, dpi=140, bbox_inches="tight")
    plt.close(fig_c)
    print(f"saved {out_c}")


if __name__ == "__main__":
    main()
