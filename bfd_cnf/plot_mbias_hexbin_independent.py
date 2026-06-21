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
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PLOTS_DIR

DELTA_G = 0.04


def _g_from_pqr(pqr: np.ndarray) -> np.ndarray:
    """Maximum-likelihood shear from a summed PQR block (numpy port of pqr2g)."""
    keep = pqr[:, 0] >= 1e-10
    pqr = pqr[keep]
    if pqr.shape[0] < 3:
        return np.array([np.nan, np.nan])
    P = pqr[:, 0]
    Q = pqr[:, 1:3]
    R = np.empty((pqr.shape[0], 2, 2), dtype=np.float64)
    R[:, 0, 0] = pqr[:, 3]
    R[:, 1, 1] = pqr[:, 4]
    R[:, 0, 1] = R[:, 1, 0] = pqr[:, 5]
    Q_tot = np.nansum(Q / P[:, None], axis=0)
    R_tot = np.nansum(
        np.einsum("ni,nj->nij", Q, Q) / P[:, None, None] ** 2 - R / P[:, None, None],
        axis=0,
    )
    try:
        return np.linalg.solve(R_tot, Q_tot)
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
    ap.add_argument("--clim", type=float, default=None,
                    help="Symmetric m colour limit (default: robust 92nd pct of |m|).")
    ap.add_argument("--yaxis", choices=["mr_mf", "logMr"], default="mr_mf",
                    help="Size axis: Mr/Mf (default, filled band) or log10 Mr.")
    ap.add_argument("--nboot", type=int, default=50,
                    help="Bootstrap resamples per cell for the m error σ (default 50).")
    args = ap.parse_args()

    d = np.load(args.inp)
    xp, yp, pqr_p = _arm_coords(d["pqr_p"].astype(np.float64), d["targets_p"], args.yaxis)
    xm, ym, pqr_m = _arm_coords(d["pqr_m"].astype(np.float64), d["targets_m"], args.yaxis)
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

    def reduce_m(carried) -> float:
        idx = np.asarray(carried, dtype=np.int64)
        ip = idx[idx < n_p]
        im = idx[idx >= n_p] - n_p
        if ip.size < min_arm or im.size < min_arm:
            return np.nan
        gp = _g_from_pqr(pqr_p[ip])
        gm = _g_from_pqr(pqr_m[im])
        return (gp[0] - gm[0]) / DELTA_G - 1.0

    _rng = np.random.default_rng(0)

    def reduce_sigma(carried) -> float:
        """Per-cell bootstrap error on m, resampling each arm independently."""
        idx = np.asarray(carried, dtype=np.int64)
        ip = idx[idx < n_p]
        im = idx[idx >= n_p] - n_p
        if ip.size < min_arm or im.size < min_arm:
            return np.nan
        ms = np.empty(args.nboot)
        for b in range(args.nboot):
            sp = pqr_p[ip[_rng.integers(0, ip.size, ip.size)]]
            sm = pqr_m[im[_rng.integers(0, im.size, im.size)]]
            gp = _g_from_pqr(sp)
            gm = _g_from_pqr(sm)
            ms[b] = (gp[0] - gm[0]) / DELTA_G - 1.0
        return float(np.nanstd(ms))

    if args.yaxis == "mr_mf":
        ylo, yhi = max(y_all.min(), 1.5), min(y_all.max(), 5.0)
    else:
        ylo, yhi = y_all.min(), y_all.max()
    extent = (x_all.min(), x_all.max(), ylo, yhi)
    hexkw = dict(gridsize=args.gridsize, extent=extent)

    fig, ax = plt.subplots(1, 3, figsize=(23, 6.6))

    # ---- m hexbin (sum +/- separately per cell, then m from the sums) ----
    hb = ax[0].hexbin(x_all, y_all, C=C, reduce_C_function=reduce_m,
                      mincnt=2 * min_arm, cmap="RdBu_r", **hexkw)
    vals = np.ma.filled(hb.get_array().astype(float), np.nan)
    finite = vals[np.isfinite(vals)]
    clim = args.clim if args.clim is not None else float(
        np.nanpercentile(np.abs(finite), 92))
    clim = max(clim, 1e-3)
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
    hb_s = ax[1].hexbin(x_all, y_all, C=C, reduce_C_function=reduce_sigma,
                        mincnt=2 * min_arm, **hexkw)
    s_vals = np.ma.filled(hb_s.get_array().astype(float), np.nan)
    z = vals / np.where(s_vals > 0, s_vals, np.nan)
    hb_s.set_array(np.ma.masked_invalid(z))
    hb_s.set_cmap("RdBu_r")
    zfin = z[np.isfinite(z)]
    zlim = float(min(np.nanpercentile(np.abs(zfin), 95), 8.0)) if zfin.size else 5.0
    hb_s.set_clim(-zlim, zlim)
    fig.colorbar(hb_s, ax=ax[1]).set_label(r"significance  $m/\sigma$", size=12)
    ax[1].set_title(rf"$m/\sigma$  (per-cell bootstrap $\sigma$, n={args.nboot}); "
                    rf"$|m/\sigma|>2$ in {100*np.mean(np.abs(zfin) > 2):.0f}% of cells",
                    fontsize=10)

    # ---- count hexbin (total targets per cell) ----
    hc = ax[2].hexbin(x_all, y_all, bins="log", cmap="viridis", **hexkw)
    fig.colorbar(hc, ax=ax[2]).set_label("targets per cell (log, both arms)", size=12)
    ax[2].set_title("target counts", fontsize=11)
    print(f"per-cell σ: median={np.nanmedian(s_vals):.4f}; "
          f"|m/σ|>2 in {100*np.mean(np.abs(zfin) > 2):.1f}% of cells")

    # Mr/Mf selection band (2.2-3.5): horizontal in the Mr/Mf plane, diagonal in
    # the log10 Mr plane.
    xline = np.array(extent[:2])
    for a in ax:
        a.set_xlabel(r"$\log_{10} M_f$  (flux)", fontsize=13)
        a.set_ylabel(ylabel, fontsize=13)
        for r in (2.2, 3.5):
            if args.yaxis == "mr_mf":
                a.axhline(r, color="green", ls="--", lw=1)
            else:
                a.plot(xline, xline + np.log10(r), color="green", ls="--", lw=1)
        a.set_ylim(extent[2], extent[3])

    fig.tight_layout()
    out = args.out or os.path.join(
        PLOTS_DIR, "mbias_hexbin_indep_"
        + os.path.basename(args.inp).replace("pqr_grid_independent_", "").replace(".npz", "")
        + f"_{args.yaxis}.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
