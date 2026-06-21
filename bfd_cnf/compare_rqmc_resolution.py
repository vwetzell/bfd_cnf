"""
compare_rqmc_resolution.py
==========================
Compare two flow PQR grid integrations of the **same targets** that differ only
in RQMC resolution (``n_points`` / ``n_replicates``).  Aligns the two runs by
``id`` and asks whether the per-galaxy shear distribution is *stable* under more
quadrature points — i.e. whether the width of the flow per-galaxy distribution
is RQMC integration noise (would shrink at higher resolution) or an intrinsic
property of the flow likelihood (would not).

Run::

    python -m bfd_cnf.compare_rqmc_resolution \
        --baseline data/pqr_grid_100k_mf3000_90000_plateau310k.npz \
        --hires    data/pqr_grid_100k_mf3000_90000_plateau310k_hires.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .config import PLOTS_DIR
from .plot_per_galaxy_shear import per_object_g


def _align(a: np.ndarray, b: np.ndarray, ids_a: np.ndarray, ids_b: np.ndarray):
    """Return (a, b) restricted to common ids, in matching row order."""
    common = np.intersect1d(ids_a, ids_b)
    ia = {int(i): k for k, i in enumerate(ids_a)}
    ib = {int(i): k for k, i in enumerate(ids_b)}
    sel_a = np.array([ia[int(i)] for i in common])
    sel_b = np.array([ib[int(i)] for i in common])
    return a[sel_a], b[sel_b], common


def _dist_stats(g1: np.ndarray) -> dict:
    g1 = g1[np.isfinite(g1)]
    med = np.median(g1)
    return dict(
        n=g1.size,
        median=med,
        IQR=float(np.subtract(*np.percentile(g1, [75, 25]))),
        MAD=float(np.median(np.abs(g1 - med))),
        core=float(np.mean(np.abs(g1) < 0.1)),
        tail1=float(np.mean(np.abs(g1) > 1.0)),
        tail10=float(np.mean(np.abs(g1) > 10.0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--hires", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--range", type=float, default=1.0)
    ap.add_argument("--bins", type=int, default=200)
    args = ap.parse_args()

    db = np.load(args.baseline)
    dh = np.load(args.hires)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    edges = np.linspace(-args.range, args.range, args.bins + 1)

    print(f"{'comp':4s} {'res':9s} {'median':>9s} {'IQR':>7s} {'MAD':>7s} "
          f"{'core<.1':>8s} {'|g|>1':>7s} {'|g|>10':>7s}")
    for col, (axi, name) in enumerate([(ax[0], "g_1"), (ax[1], "g_2")]):
        for grp, color in [("p", "red"), ("m", "blue")]:
            pqr_b, pqr_h, common = _align(
                db[f"pqr_{grp}"], dh[f"pqr_{grp}"], db["ids"], dh["ids"])
            gb = per_object_g(pqr_b.astype(np.float64))[col]
            gh = per_object_g(pqr_h.astype(np.float64))[col]
            sb, sh = _dist_stats(gb), _dist_stats(gh)

            axi.hist(gb[np.isfinite(gb)], bins=edges, density=True, histtype="step",
                     color=color, lw=1.5, label=f"{grp}: baseline (n_pts low)")
            axi.hist(gh[np.isfinite(gh)], bins=edges, density=True, histtype="step",
                     color=color, ls="--", lw=1.3, alpha=0.7,
                     label=f"{grp}: hi-res (n_pts high)")

            if col == 0:
                print(f"{name:4s} base/{grp:3s} {sb['median']:+9.4f} {sb['IQR']:7.3f} "
                      f"{sb['MAD']:7.3f} {sb['core']:8.3f} {sb['tail1']:7.4f} {sb['tail10']:7.4f}")
                print(f"{name:4s} hire/{grp:3s} {sh['median']:+9.4f} {sh['IQR']:7.3f} "
                      f"{sh['MAD']:7.3f} {sh['core']:8.3f} {sh['tail1']:7.4f} {sh['tail10']:7.4f}")

        axi.set_title(rf"${name}$  (common n={common.size:,})", fontsize=12)
        axi.set_xlabel(rf"per-galaxy ${name}$ estimate"); axi.set_ylabel("density")
        axi.set_xlim(-args.range, args.range)
        handles = [
            Line2D([], [], color="red", lw=1.5, label="+ group"),
            Line2D([], [], color="blue", lw=1.5, label=r"$-$ group"),
            Line2D([], [], color="0.3", ls="-", lw=1.5, label="baseline RQMC"),
            Line2D([], [], color="0.3", ls="--", lw=1.3, label="hi-res RQMC"),
        ]
        axi.legend(handles=handles, fontsize=9, loc="upper right")

    fig.suptitle("Per-galaxy shear: baseline vs hi-res RQMC (same flow, same targets)",
                 fontsize=13)
    fig.tight_layout()
    out = args.out or os.path.join(PLOTS_DIR, "rqmc_resolution_compare.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
