"""
plot_mbias_hexbin.py
====================
2-D hexbin of the multiplicative bias ``m`` in the (log10 Mf, Mr/Mf) plane,
from a saved PQR grid integration (``data/pqr_grid_*.npz``).

``m`` is not a per-target quantity — it is a ratio of the *summed* PQR over a
population:

    m_cell = (g1(+) - g1(-)) / Δg - 1,   Δg = 0.04

with ``g(±) = R_tot⁻¹ Q_tot`` accumulated over the targets in that cell.  We
therefore colour each hexagon by ``m`` recomputed from its targets' PQR, using
``hexbin``'s ``reduce_C_function`` with the target index as the carried value.

Two panels: ``m`` per cell (diverging, centred on 0) and the target count per
cell (so noisy low-count cells are visible).  The flux window, the Mr/Mf
selection band (2.2–3.5), and the stellar locus (Mr/Mf≈3.976) are overlaid.

Run::

    python -m bfd_cnf.plot_mbias_hexbin --in data/pqr_grid_FULL_mf3000_90000_adapt240k.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from .config import PLOTS_DIR

DELTA_G = 0.04


def _g_from_pqr(pqr: np.ndarray) -> np.ndarray:
    """Maximum-likelihood shear from a per-target PQR block (numpy port of pqr2g)."""
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
    QQ = np.einsum("ni,nj->nij", Q, Q) / P[:, None, None] ** 2
    R_tot = np.nansum(QQ - R / P[:, None, None], axis=0)
    try:
        return np.linalg.solve(R_tot, Q_tot)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp",
                    default="data/pqr_grid_FULL_mf3000_90000_adapt240k.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gridsize", type=int, default=38)
    ap.add_argument("--mincnt", type=int, default=150,
                    help="Min targets per hexagon for a stable m (default 150).")
    ap.add_argument("--clim", type=float, default=None,
                    help="Symmetric m colour limit (default: robust 92nd pct of |m|).")
    ap.add_argument("--uncertainty", action="store_true",
                    help="Encode per-cell uncertainty: panel A = m with opacity ∝ 1/σ, "
                         "panel B = significance m/σ.  σ is bootstrapped within each cell.")
    ap.add_argument("--n-boot", type=int, default=60,
                    help="Bootstrap resamples per cell for σ (--uncertainty).")
    args = ap.parse_args()

    d = np.load(args.inp)
    pqr_p = d["pqr_p"].astype(np.float64)
    pqr_m = d["pqr_m"].astype(np.float64)
    tp = d["targets_p"]
    mf = np.abs(tp[:, 0])
    x = np.log10(mf)                 # log10 Mf
    y = tp[:, 1] / mf                # Mr / Mf

    valid = (
        np.all(np.isfinite(pqr_p), axis=1) & np.all(np.isfinite(pqr_m), axis=1)
        & (pqr_p[:, 0] > 0) & (pqr_m[:, 0] > 0) & np.isfinite(x) & np.isfinite(y)
    )
    x, y = x[valid], y[valid]
    pqr_p, pqr_m = pqr_p[valid], pqr_m[valid]
    n = x.shape[0]
    print(f"{n} valid targets;  log10Mf [{x.min():.2f},{x.max():.2f}]  "
          f"Mr/Mf [{y.min():.2f},{y.max():.2f}]")

    def reduce_m(carried) -> float:
        idx = np.asarray(carried, dtype=np.int64)
        gp = _g_from_pqr(pqr_p[idx])
        gm = _g_from_pqr(pqr_m[idx])
        return (gp[0] - gm[0]) / DELTA_G - 1.0

    extent = (x.min(), x.max(), max(y.min(), 1.5), min(y.max(), 5.0))
    hexkw = dict(gridsize=args.gridsize, mincnt=args.mincnt, extent=extent)

    def _decorate(a):
        a.set_xlabel(r"$\log_{10} M_f$", fontsize=13)
        a.set_ylabel(r"$M_r / M_f$", fontsize=13)
        a.axhline(2.2, color="green", ls="--", lw=1)
        a.axhline(3.5, color="green", ls="--", lw=1, label="Mr/Mf selection")
        a.axhline(3.976, color="red", ls=":", lw=1, label="stellar locus")
        a.legend(fontsize=8, loc="upper right")

    # ------------------------------------------------------------------
    # Uncertainty-aware rendering: opacity ∝ 1/σ, plus a significance map.
    # ------------------------------------------------------------------
    if args.uncertainty:
        rng = np.random.default_rng(0)
        nb = args.n_boot

        def reduce_sigma(carried) -> float:
            idx = np.asarray(carried, dtype=np.int64)
            if idx.size < 3:
                return np.nan
            ms = np.empty(nb)
            for b in range(nb):
                s = idx[rng.integers(0, idx.size, idx.size)]
                gp = _g_from_pqr(pqr_p[s])
                gm = _g_from_pqr(pqr_m[s])
                ms[b] = (gp[0] - gm[0]) / DELTA_G - 1.0
            return float(np.nanstd(ms))

        fig, ax = plt.subplots(1, 2, figsize=(17, 6.6))
        for a in ax:
            a.set_facecolor("0.82")  # so faded (uncertain) cells read as grey

        # geometry + per-cell m on the keeper axis
        hb_m = ax[0].hexbin(x, y, C=np.arange(n), reduce_C_function=reduce_m,
                            cmap="RdBu_r", **hexkw)
        m_vals = np.ma.filled(hb_m.get_array().astype(float), np.nan)

        # per-cell bootstrap σ on a throwaway axis (identical binning ⇒ aligned)
        figtmp, axtmp = plt.subplots()
        hb_s = axtmp.hexbin(x, y, C=np.arange(n), reduce_C_function=reduce_sigma, **hexkw)
        s_vals = np.ma.filled(hb_s.get_array().astype(float), np.nan)
        plt.close(figtmp)

        clim = args.clim if args.clim is not None else float(
            np.nanpercentile(np.abs(m_vals[np.isfinite(m_vals)]), 92))
        clim = max(clim, 1e-3)
        norm = Normalize(-clim, clim)
        cmap = plt.get_cmap("RdBu_r")

        cert = np.where(s_vals > 0, 1.0 / s_vals, np.nan)        # certainty = 1/σ
        a_ref = np.nanpercentile(cert, 85)                       # ~most-certain cells -> opaque
        alpha = np.clip(cert / a_ref, 0.05, 1.0)
        alpha[~np.isfinite(alpha)] = 0.0
        rgba = cmap(norm(m_vals))
        rgba[:, 3] = alpha
        hb_m.set_color(rgba)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=ax[0]).set_label("multiplicative bias m", size=12)
        ax[0].set_title(r"m — opacity $\propto 1/\sigma$ (uncertain cells faded)", fontsize=11)

        # significance panel m/σ (reuse aligned m_vals / s_vals)
        hb_z = ax[1].hexbin(x, y, C=np.arange(n), reduce_C_function=reduce_m,
                            cmap="RdBu_r", **hexkw)
        z = m_vals / np.where(s_vals > 0, s_vals, np.nan)
        hb_z.set_array(np.ma.masked_invalid(z))
        zlim = float(min(np.nanpercentile(np.abs(z[np.isfinite(z)]), 95), 5.0))
        hb_z.set_clim(-zlim, zlim)
        fig.colorbar(hb_z, ax=ax[1]).set_label(r"significance  $m/\sigma$", size=12)
        ax[1].set_title(r"$m/\sigma$  (per-cell bootstrap $\sigma$, n=%d)" % nb, fontsize=11)

        zf = z[np.isfinite(z)]
        print(f"{np.isfinite(m_vals).sum()} cells; per-cell σ median={np.nanmedian(s_vals):.3f}; "
              f"|m/σ|>2 in {100*np.mean(np.abs(zf)>2):.1f}% of cells; clim ±{clim:.3f}, zlim ±{zlim:.2f}")
        for a in ax:
            _decorate(a)
        fig.tight_layout()
        out = args.out or os.path.join(
            PLOTS_DIR,
            "mbias_hexbin_uncert_"
            + os.path.basename(args.inp).replace("pqr_grid_", "").replace(".npz", "") + ".png")
        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"saved {out}")
        return

    fig, ax = plt.subplots(1, 2, figsize=(17, 6.6))

    # ---- m hexbin ----
    hb = ax[0].hexbin(
        x, y, C=np.arange(n), reduce_C_function=reduce_m,
        gridsize=args.gridsize, mincnt=args.mincnt, cmap="RdBu_r", extent=extent,
    )
    vals = hb.get_array()
    finite = vals[np.isfinite(vals)]
    clim = args.clim if args.clim is not None else float(np.nanpercentile(np.abs(finite), 92))
    clim = max(clim, 1e-3)
    hb.set_clim(-clim, clim)
    print(f"{finite.size} hexbins with >= {args.mincnt} targets;  "
          f"m per-cell: median={np.median(finite):+.4f}  "
          f"[p5,p95]=[{np.percentile(finite,5):+.3f},{np.percentile(finite,95):+.3f}]  "
          f"colour limit ±{clim:.3f}")
    cb = fig.colorbar(hb, ax=ax[0]); cb.set_label("multiplicative bias  m", size=12)
    ax[0].set_title(f"m in (log10 Mf, Mr/Mf)  —  {os.path.basename(args.inp)}", fontsize=11)

    # ---- count hexbin ----
    hc = ax[1].hexbin(x, y, gridsize=args.gridsize, bins="log", cmap="viridis", extent=extent)
    fig.colorbar(hc, ax=ax[1]).set_label("targets per cell (log)", size=12)
    ax[1].set_title("target counts", fontsize=11)

    for a in ax:
        a.set_xlabel(r"$\log_{10} M_f$", fontsize=13)
        a.set_ylabel(r"$M_r / M_f$", fontsize=13)
        a.axhline(2.2, color="green", ls="--", lw=1)
        a.axhline(3.5, color="green", ls="--", lw=1, label="Mr/Mf selection")
        a.axhline(3.976, color="red", ls=":", lw=1, label="stellar locus")
        a.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = args.out or os.path.join(
        PLOTS_DIR,
        "mbias_hexbin_" + os.path.basename(args.inp).replace("pqr_grid_", "").replace(".npz", "") + ".png",
    )
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
