"""
plot_trR_frac_hexbin.py
=======================
Standalone (numpy + matplotlib + BFD only): hexbin of the per-cell FRACTION of
objects with tr(R_tot)/2 < THRESH, in the usual flux-size plane (log10 Mf, Mr/Mf).

tr(R_tot)/2 = (R_tot11 + R_tot22)/2 (mean diagonal curvature), R_tot = QQ^T/P^2
- R/P (the shear-estimator curvature).  bfd.logPqr stores the raw
d2logP = R/P - QQ/P^2 = -R_tot in its R block, so tr(R_tot)/2 = -(col3 + col5)/2.

Both arms are pooled (the cut is a property of the target + noise, not the applied
shear).  Cell colour = mean of the boolean over that cell's objects.

Run:
    python plot_trR_frac_hexbin.py --in data/pqr_indep_xy_elbo_1M.npz --field pqr --thresh 10
    # the earlier sim tr(R_tot)<0 map is just:
    python plot_trR_frac_hexbin.py --in data/pqr_indep_xy_elbo_1M.npz --field pqr_sim --thresh 0
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
import bfd

# flow [P,Q1,Q2,R11,R22,R12] -> bfd [P,Q1,Q2,R11,R12,R22] (its own inverse)
FLOW_BFD = [0, 1, 2, 3, 5, 4]

BOX_MF = (1500.0, 90000.0)   # green selection box flux bounds
BOX_MRMF = (2.2, 3.5)        # green selection box Mr/Mf bounds
_PAD = 0.08
_BOX_MF_LOG = (math.log10(BOX_MF[0]), math.log10(BOX_MF[1]))
# physical aspect matching the corner-plot flux-size panel (see plot_mbias_hexbin_independent)
CORNER_ASPECT = (math.log10(200000.0) - math.log10(500.0)) / (5.5 - 1.2)


def _iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.subtract(*np.percentile(x, [75, 25])))


def _logP(pqr_p: np.ndarray, pqr_m: np.ndarray) -> np.ndarray:
    """Pooled log(P) over both arms (finite, P>0)."""
    out = []
    for pqr in (pqr_p, pqr_m):
        pqr = np.asarray(pqr, dtype=np.float64)
        ok = np.all(np.isfinite(pqr), axis=1) & (pqr[:, 0] > 1e-10)
        out.append(np.log(pqr[ok, 0]))
    return np.concatenate(out)


def _arm(pqr: np.ndarray, targets: np.ndarray):
    """(x=log10 Mf, y=Mr/Mf, half_tr) for finite, P>0 targets; half_tr = tr(R_tot)/2."""
    pqr = np.asarray(pqr, dtype=np.float64)
    mf = np.abs(targets[:, 0])
    x = np.log10(mf)
    y = np.abs(targets[:, 1]) / mf
    ok = (np.all(np.isfinite(pqr), axis=1) & (pqr[:, 0] > 1e-10)
          & np.isfinite(x) & np.isfinite(y))
    lp = np.asarray(bfd.logPqr(pqr[ok][:, FLOW_BFD]))
    half_tr = -0.5 * (lp[:, 3] + lp[:, 5])  # tr(R_tot)/2  (bfd R block = -R_tot)
    return x[ok], y[ok], half_tr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_indep_xy_elbo_1M.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--field", choices=["pqr", "pqr_sim"], default="pqr",
                    help="flow ('pqr', default) or analytic sim ('pqr_sim').")
    ap.add_argument("--thresh", type=float, default=10.0,
                    help="fraction with alpha*tr(R_tot)/2 < THRESH (default 10).")
    ap.add_argument("--alpha", type=float, default=None,
                    help="temperature scale on tr(R_tot); default = IQR(logP_sim)/IQR(logP_flow) "
                         "for --field pqr, else 1.")
    ap.add_argument("--gridsize", type=int, default=38)
    ap.add_argument("--mincnt", type=int, default=80,
                    help="min objects per cell for a stable fraction (default 80).")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    args = ap.parse_args()
    field_name = "Flow" if args.field == "pqr" else "Sim"

    d = np.load(args.inp)
    if args.alpha is not None:
        alpha = args.alpha
    elif args.field == "pqr":  # temperature that maps the flow onto the sim (from logP)
        alpha = _iqr(_logP(d["pqr_sim_p"], d["pqr_sim_m"])) / _iqr(_logP(d["pqr_p"], d["pqr_m"]))
    else:
        alpha = 1.0
    print(f"alpha = {alpha:.4g}")

    xp, yp, hp = _arm(d[f"{args.field}_p"], d["targets_p"])
    xm, ym, hm = _arm(d[f"{args.field}_m"], d["targets_m"])
    x = np.concatenate([xp, xm])
    y = np.concatenate([yp, ym])
    c = (alpha * np.concatenate([hp, hm]) < args.thresh).astype(float)
    print(f"{field_name}: {c.size:,} targets pooled;  "
          f"global alpha*tr(R_tot)/2 < {args.thresh:g} fraction = {c.mean():.4f}")

    w = _BOX_MF_LOG[1] - _BOX_MF_LOG[0]
    h = BOX_MRMF[1] - BOX_MRMF[0]
    extent = (_BOX_MF_LOG[0] - _PAD * w, _BOX_MF_LOG[1] + _PAD * w,
              BOX_MRMF[0] - _PAD * h, BOX_MRMF[1] + _PAD * h)
    box_aspect = CORNER_ASPECT * (extent[3] - extent[2]) / (extent[1] - extent[0])
    gridsize = (args.gridsize, max(1, round(args.gridsize * box_aspect / math.sqrt(3))))

    fig, ax = plt.subplots(figsize=(9, 9 * box_aspect + 1.3), constrained_layout=True)
    ax.set_box_aspect(box_aspect)
    hb = ax.hexbin(x, y, C=c, reduce_C_function=np.mean, gridsize=gridsize,
                   extent=extent, mincnt=args.mincnt, cmap="magma",
                   vmin=args.vmin, vmax=args.vmax)
    fig.colorbar(hb, ax=ax).set_label(
        rf"fraction with $\alpha\,$tr$(R_{{\rm tot}})/2 < {args.thresh:g}$", size=12)

    ax.add_patch(mpatches.Rectangle(
        (_BOX_MF_LOG[0], BOX_MRMF[0]), w, h,
        lw=1.3, edgecolor="tab:green", facecolor="none", zorder=100))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(r"$\log_{10} M_f$  (flux)", fontsize=13)
    ax.set_ylabel(r"$M_r / M_f$  (size)", fontsize=13)
    ax.set_title(rf"{field_name}: fraction with $\alpha\,$tr$(R_{{\rm tot}})/2 < {args.thresh:g}$"
                 rf"  ($\alpha$={alpha:.3g}) — {os.path.basename(args.inp)}", fontsize=12)

    tag = f"{args.field}_a{alpha:.3g}_lt{args.thresh:g}".replace(".", "p").replace("-", "m")
    out = args.out or os.path.join(
        "plots", f"trR_frac_hexbin_{tag}_"
        + os.path.basename(args.inp).replace(".npz", "") + ".png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
