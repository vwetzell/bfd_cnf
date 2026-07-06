"""
plot_per_galaxy_shear.py
========================
Distribution of *individual-galaxy* shear estimates from a saved independent-
ensemble PQR grid integration, with the per-galaxy **median** and the
**sum-PQR** estimate overlaid.

For galaxy i the single-object shear is the per-object version of ``pqr2g``:

    Q_tot,i = Q_i / P_i,
    R_tot,i = Q_i Q_iᵀ / P_i² − R_i / P_i,
    g_i     = R_tot,i⁻¹ Q_tot,i,

which is wildly noisy (each galaxy barely constrains shear, and R_tot,i is
often not positive-definite → heavy tails).  The **sum-PQR** estimate
``g = (Σ R_tot,i)⁻¹ (Σ Q_tot,i)`` is the inverse-variance-weighted combination
actually used for the shear measurement.  Comparing the per-galaxy median to the
sum-PQR value shows whether the optimal weighting pulls the estimate away from
the robust unweighted centre.

The +shear and -shear arms are independent injection realisations (different
objects, generally different lengths) — everything here is computed per arm
with no cross-arm indexing, so this works directly on that schema.

Panels: 2×2 = {+shear, −shear} × {g1, g2}.

Run::

    python -m bfd_cnf.plot_per_galaxy_shear --in data/pqr_grid.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .config import PLOTS_DIR
from .statistics import pqr2g


def per_object_g(pqr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-galaxy shear g_i = R_tot,i⁻¹ Q_tot,i (analytic 2×2 inverse)."""
    keep = pqr[:, 0] >= 1e-10
    pqr = pqr[keep]
    P = pqr[:, 0]
    Q1, Q2 = pqr[:, 1], pqr[:, 2]
    R11, R22, R12 = pqr[:, 3], pqr[:, 4], pqr[:, 5]
    qt1, qt2 = Q1 / P, Q2 / P
    a = Q1 * Q1 / P**2 - R11 / P          # R_tot[0,0]
    c = Q2 * Q2 / P**2 - R22 / P          # R_tot[1,1]
    b = Q1 * Q2 / P**2 - R12 / P          # R_tot[0,1]
    det = a * c - b * b
    bad = np.abs(det) < 1e-30
    det = np.where(bad, np.nan, det)
    g1 = (c * qt1 - b * qt2) / det
    g2 = (-b * qt1 + a * qt2) / det
    return g1, g2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_grid.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--range", type=float, default=0.5,
                    help="Half-width of the g axis for the histograms (default 0.5).")
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--input-g", type=float, default=0.02,
                    help="Input shear magnitude marked by the dashed lines (default 0.02).")
    args = ap.parse_args()

    d = np.load(args.inp)
    # Flow PQR ("pqr_*") and analytic-BFD PQR ("pqr_sim_*") on the same targets.
    gp1, gp2 = per_object_g(d["pqr_p"].astype(np.float64))     # flow,  + shear group
    gm1, gm2 = per_object_g(d["pqr_m"].astype(np.float64))     # flow,  - shear group
    ap1, ap2 = per_object_g(d["pqr_sim_p"].astype(np.float64))  # analytic, + group
    am1, am2 = per_object_g(d["pqr_sim_m"].astype(np.float64))  # analytic, - group
    g_sum_p = np.asarray(pqr2g(jnp.asarray(d["pqr_p"])))       # flow sum-PQR
    g_sum_m = np.asarray(pqr2g(jnp.asarray(d["pqr_m"])))
    gs_sum_p = np.asarray(pqr2g(jnp.asarray(d["pqr_sim_p"])))  # analytic sum-PQR
    gs_sum_m = np.asarray(pqr2g(jnp.asarray(d["pqr_sim_m"])))
    n_tot = int(np.isfinite(gp1).sum() + np.isfinite(gm1).sum())
    gi = args.input_g

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    edges = np.linspace(-args.range, args.range, args.bins + 1)

    # Applied (input) shear per component: g1 = +/-gi, g2 = 0 for both groups.
    applied = {"g_1": (+gi, -gi), "g_2": (0.0, 0.0)}

    hdr = (f"{'comp':4s} {'src':8s} {'+grp med':>9s} {'-grp med':>9s}   "
           f"{'+ sum-PQR':>9s} {'- sum-PQR':>9s}")
    print(hdr)
    for a, name, gpos, gneg, apos, aneg, gsp, gsm, gssp, gssm in [
        (ax[0], "g_1", gp1, gm1, ap1, am1,
         g_sum_p[0], g_sum_m[0], gs_sum_p[0], gs_sum_m[0]),
        (ax[1], "g_2", gp2, gm2, ap2, am2,
         g_sum_p[1], g_sum_m[1], gs_sum_p[1], gs_sum_m[1]),
    ]:
        gpos = gpos[np.isfinite(gpos)]; gneg = gneg[np.isfinite(gneg)]
        apos = apos[np.isfinite(apos)]; aneg = aneg[np.isfinite(aneg)]
        mp, mm = float(np.median(gpos)), float(np.median(gneg))
        mpa, mma = float(np.median(apos)), float(np.median(aneg))
        ap_p, ap_m = applied[name]

        # Distributions: flow = solid step, analytic-BFD = dashed step (same colors).
        a.hist(gpos, bins=edges, density=True, histtype="step", color="red", lw=1.4)
        a.hist(gneg, bins=edges, density=True, histtype="step", color="blue", lw=1.4)
        a.hist(apos, bins=edges, density=True, histtype="step", color="red",
               ls="--", lw=1.2, alpha=0.6)
        a.hist(aneg, bins=edges, density=True, histtype="step", color="blue",
               ls="--", lw=1.2, alpha=0.6)

        # Applied shear (black dashed): one line per group (coincide at 0 for g2).
        a.axvline(ap_p, color="k", ls="--", lw=1.2)
        a.axvline(ap_m, color="k", ls="--", lw=1.2)
        # Flow per-galaxy median (solid) and sum-PQR mean shear (dash-dot), per group.
        a.axvline(mp, color="red", ls="-", lw=1.8)
        a.axvline(mm, color="blue", ls="-", lw=1.8)
        a.axvline(gsp, color="red", ls="-.", lw=1.8)
        a.axvline(gsm, color="blue", ls="-.", lw=1.8)

        a.set_title(rf"${name}$  (Mf>3000, n={n_tot:,})", fontsize=12)
        a.set_xlabel(rf"per-galaxy ${name}$ estimate"); a.set_ylabel("density")
        a.set_xlim(-args.range, args.range)

        group_handles = [
            Line2D([], [], color="red", lw=1.4,
                   label=f"+ flow: med {mp:+.4f}, sum-PQR {gsp:+.4f} (in {ap_p:+.2f})"),
            Line2D([], [], color="blue", lw=1.4,
                   label=rf"$-$ flow: med {mm:+.4f}, sum-PQR {gsm:+.4f} (in {ap_m:+.2f})"),
            Line2D([], [], color="red", ls="--", lw=1.2, alpha=0.6,
                   label=f"+ analytic: med {mpa:+.4f}, sum-PQR {gssp:+.4f}"),
            Line2D([], [], color="blue", ls="--", lw=1.2, alpha=0.6,
                   label=rf"$-$ analytic: med {mma:+.4f}, sum-PQR {gssm:+.4f}"),
        ]
        style_handles = [
            Line2D([], [], color="k", ls="--", lw=1.2, label="applied shear"),
            Line2D([], [], color="0.3", ls="-", lw=1.8, label="flow median"),
            Line2D([], [], color="0.3", ls="-.", lw=1.8, label="flow sum-PQR"),
        ]
        leg1 = a.legend(handles=group_handles, fontsize=8, loc="upper right")
        a.add_artist(leg1)
        a.legend(handles=style_handles, fontsize=8, loc="upper left")
        print(f"{name:4s} {'flow':8s} {mp:+9.4f} {mm:+9.4f}   {gsp:+9.4f} {gsm:+9.4f}")
        print(f"{name:4s} {'analytic':8s} {mpa:+9.4f} {mma:+9.4f}   {gssp:+9.4f} {gssm:+9.4f}")

    fig.suptitle("Per-galaxy shear estimates: + vs - shear groups "
                 "(solid=flow, dashed=analytic-BFD; vlines: dashed=applied, "
                 "solid=flow median, dash-dot=flow sum-PQR)", fontsize=12)
    fig.tight_layout()
    out = args.out or os.path.join(
        PLOTS_DIR, "per_galaxy_shear_overlay_"
        + os.path.basename(args.inp).replace("pqr_grid_", "").replace(".npz", "") + ".png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
