"""
sweep_per_galaxy_shear_mf.py
============================
Sweep the lower Mf cut and track how the +/- shear groups' analytic per-galaxy
shear distributions converge in shape (spread + peakedness), to see at what flux
the selection-boundary asymmetry washes out.

For each threshold the +/- groups are cut on the shared +shear Mf (= targets_p
col 0); g1 and g2 are pooled per group. Reports MAD and frac|g|<0.1 per group.

Run::

    python -m bfd_cnf.sweep_per_galaxy_shear_mf \
        --in data/pqr_grid_100k_mf3000_90000_nda510k.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PLOTS_DIR
from .plot_per_galaxy_shear import per_object_g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp",
                    default="data/pqr_grid_100k_mf3000_90000_nda510k.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mf-his", type=float, default=90000.0,
                    help="Upper Mf bound held fixed for every threshold.")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[3000, 5000, 8000, 10000, 15000, 20000, 30000])
    args = ap.parse_args()

    d = np.load(args.inp)
    mf = d["targets_p"][:, 0]

    rows = []
    print(f"{'Mf>':>7s} {'n':>7s}  {'MAD+':>6s} {'MAD-':>6s} {'MAD-/+':>6s}  "
          f"{'core+':>6s} {'core-':>6s}")
    for lo in args.thresholds:
        cut = (mf > lo) & (mf < args.mf_his)
        stats = {}
        for grp in ("p", "m"):
            a1, a2 = per_object_g(d[f"pqr_sim_{grp}"][cut].astype(np.float64))
            g = np.concatenate([a1, a2])
            g = g[np.isfinite(g)]
            stats[grp] = (float(np.median(np.abs(g - np.median(g)))),
                          float(np.mean(np.abs(g) < 0.1)))
        madp, corep = stats["p"]
        madm, corem = stats["m"]
        rows.append((lo, int(cut.sum()), madp, madm, corep, corem))
        print(f"{lo:7.0f} {int(cut.sum()):7d}  {madp:6.3f} {madm:6.3f} "
              f"{madm/madp:6.3f}  {corep:6.3f} {corem:6.3f}")

    arr = np.array(rows)
    los, ns = arr[:, 0], arr[:, 1]
    madp, madm, corep, corem = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(los, madp, "o-", color="red", label="+ shear")
    ax[0].plot(los, madm, "s-", color="blue", label=r"$-$ shear")
    ax[0].set_ylabel("MAD of per-galaxy g (g1,g2 pooled)")
    ax[1].plot(los, corep, "o-", color="red", label="+ shear")
    ax[1].plot(los, corem, "s-", color="blue", label=r"$-$ shear")
    ax[1].set_ylabel(r"frac $|g|<0.1$ (core fraction)")
    for a in ax:
        a.set_xlabel("lower Mf cut")
        a.set_xscale("log")
        a.legend(fontsize=9)
        a.grid(alpha=0.3)
    fig.suptitle("Per-galaxy analytic shear: +/- convergence vs lower Mf cut",
                 fontsize=12)
    fig.tight_layout()

    out = args.out or os.path.join(PLOTS_DIR, "per_galaxy_shear_mf_sweep.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
