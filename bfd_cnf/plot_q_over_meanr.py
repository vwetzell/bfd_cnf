"""
plot_q_over_meanr.py
====================
Per-target distribution of the shear contribution ``q_i / mean(r)`` from a saved
PQR npz, using the **pqr2g accumulation terms**

    q_i = Q_i / P_i,    r_i = Q_i^2 / P_i^2 - R_ii / P_i

so that the population shear is ``g = mean(q) / mean(r)`` (diagonal approx).  We
plot ``x_i = q_i / mean(r_i)`` (units of shear) histogrammed **weighted by r_i**,
the per-target curvature.  The unweighted mean of ``x_i`` is exactly ``g``; the
r-weighted histogram shows which targets carry the response.

One panel per shear component (g1, g2), +shear arm by default.

Run::

    python -m bfd_cnf.plot_q_over_meanr --in data/pqr_grid_indep_1500_90000.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PLOTS_DIR


def _accum(pqr: np.ndarray):
    """pqr2g per-target accumulation terms q (N,2) and r (N,2) (diagonal R)."""
    P = pqr[:, 0]
    Q = pqr[:, 1:3]
    Rdiag = pqr[:, 3:5]  # R11, R22
    q = Q / P[:, None]
    r = Q ** 2 / P[:, None] ** 2 - Rdiag / P[:, None]
    return q, r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_grid_indep_1500_90000.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", choices=["p", "m"], default="p", help="shear arm (default +).")
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--clip", type=float, default=8.0,
                    help="x-axis half-range as a multiple of the robust scale (default 8).")
    args = ap.parse_args()

    d = np.load(args.inp)
    pqr = d[f"pqr_{args.arm}"].astype(np.float64)
    valid = np.all(np.isfinite(pqr), axis=1) & (pqr[:, 0] > 1e-10)
    pqr = pqr[valid]
    print(f"{valid.sum():,}/{valid.size:,} valid targets ({args.arm}-arm)")

    q, r = _accum(pqr)
    rbar = r.mean(axis=0)  # mean(r) per component
    x = q / rbar[None, :]  # q_i / mean(r)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for k, comp in enumerate(("g_1", "g_2")):
        xi, wi = x[:, k], r[:, k]
        g = float(xi.mean())  # unweighted mean == mean(q)/mean(r) == g
        wmean = float(np.sum(wi * xi) / np.sum(wi))
        # robust symmetric x-range about g (weights can be negative; clip view only)
        scale = np.nanpercentile(np.abs(xi - g), 84)
        lo, hi = g - args.clip * scale, g + args.clip * scale
        bins = np.linspace(lo, hi, args.bins)
        ax[k].hist(xi, bins=bins, weights=wi, color="tab:blue", alpha=0.75,
                   label="r-weighted")
        ax[k].hist(xi, bins=bins, histtype="step", color="0.3", lw=1,
                   weights=np.full(xi.shape, np.sign(rbar[k]) / xi.size),
                   density=False, label="unweighted (scaled)")
        ax[k].axvline(g, color="tab:red", lw=1.4,
                      label=f"unweighted mean = {comp} = {g:+.4f}")
        ax[k].axvline(wmean, color="tab:green", lw=1.2, ls="--",
                      label=f"r-weighted mean = {wmean:+.4f}")
        ax[k].axvline(0.0, color="0.6", lw=0.8)
        ax[k].set_xlabel(rf"$q_i / \langle r\rangle$  (${comp}$ units)", fontsize=12)
        ax[k].set_ylabel("r-weighted target count", fontsize=12)
        ax[k].set_title(rf"${comp}$  contribution spread", fontsize=12)
        ax[k].legend(fontsize=9)
        print(f"{comp}: unweighted mean (=g) {g:+.5f}   r-weighted mean {wmean:+.5f}   "
              f"mean(r)={rbar[k]:.3e}")

    fig.suptitle(f"Q/mean(R) weighted by R — {os.path.basename(args.inp)} ({args.arm}-arm)",
                 fontsize=13)
    fig.tight_layout()
    out = args.out or os.path.join(
        PLOTS_DIR, "q_over_meanr_"
        + os.path.basename(args.inp).replace("pqr_grid_", "").replace(".npz", "")
        + f"_{args.arm}.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
