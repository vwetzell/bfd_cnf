"""
plot_sigma_f_selection.py
=========================
Overlay the histogram of the targets' flux-moment noise ``σ_f,i = sqrt(Σ_i[0,0])``
with the per-bin selection PQR log-totals (``inference.selection_pqr_binned``): the
same σ_f bins that drive the flux-limit selection correction, so you can see which
part of the σ_f distribution carries the correction.

Left axis: σ_f histogram (target counts per bin).  Right axis: the per-bin selection
term ``R^tot_11`` at each bin's representative σ_f, swept over several lower flux cuts
``f_min`` (upper edge fixed).  For a pure flux cut Q_sel≈0 (flux moment is spin-0) and
R is isotropic (R11≈R22), so R11 alone carries the correction.  The per-bin R does not
depend on the target counts, so one σ_f grid (from the loosest cut) serves every f_min.

Reuses the +arm alone by default (the σ_f histogram is essentially identical per
arm).  σ_f comes from the SOURCE grid catalogue matched by ``id`` (not stored in the
integrated npz), exactly as ``apply_selection_correction.py`` does.

Run:
    python plot_sigma_f_selection.py --in data/pqr_indep_xy_w128_1M_bruteforce.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apply_selection_correction import _bins_from_sigma_f, _sigma_f_for_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_indep_xy_w128_1M_bruteforce.npz")
    ap.add_argument("--source", default=None, help="source grid npy (default GRID_P_PATH).")
    ap.add_argument("--arm", choices=["p", "m"], default="p")
    ap.add_argument("--f-mins", type=float, nargs="+", default=[1000, 1500, 3000, 5000],
                    help="lower flux cuts to sweep (one R11 curve each).")
    ap.add_argument("--f-max", type=float, default=90000.0)
    ap.add_argument("--n-bins", type=int, default=16)
    ap.add_argument("--central", type=float, default=0.98,
                    help="bin over this central fraction of the σ_f distribution.")
    ap.add_argument("--n-samples", type=int, default=2**16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from bfd_cnf.config import GRID_M_PATH, GRID_P_PATH, PRIOR_FLOW_PATH, Q_FLOW_PATH
    from bfd_cnf.config import key as base_key
    from bfd_cnf.inference import selection_pqr_binned
    from bfd_cnf.integrate_grid import load_prior_flow
    from bfd_cnf.models.bijections import load_stats

    f_mins = sorted(args.f_mins)
    f_max = args.f_max

    d = np.load(args.inp)
    ids = np.asarray(d[f"ids_{args.arm}"])
    mf = np.asarray(d[f"targets_{args.arm}"])[:, 0]
    # σ_f grid from the loosest cut (most inclusive), central `--central` fraction.
    keep = (mf > f_mins[0]) & (mf < f_max)
    src = np.load(args.source or (GRID_P_PATH if args.arm == "p" else GRID_M_PATH))
    sigma_f = _sigma_f_for_ids(src, ids[keep])
    lo, hi = np.percentile(sigma_f, [50 * (1 - args.central), 50 * (1 + args.central)])
    sigma_f = sigma_f[(sigma_f >= lo) & (sigma_f <= hi)]
    rep, counts, _ = _bins_from_sigma_f(sigma_f, args.n_bins)

    r2s = load_stats(PRIOR_FLOW_PATH)
    flow = load_prior_flow(base_key, PRIOR_FLOW_PATH, Q_FLOW_PATH)
    # R11 per bin for each flux floor (counts don't enter the per-bin R; shared key
    # → common base draws so the curves differ only by f_min).
    r11 = {
        fm: np.asarray(selection_pqr_binned(
            flow, r2s, fm, f_max, rep, counts, n_samples=args.n_samples, key=base_key
        )["r_tot_b"])[:, 0, 0]
        for fm in f_mins
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    edges = np.histogram_bin_edges(sigma_f, bins=args.n_bins)
    ax.hist(sigma_f, bins=edges, color="0.85", edgecolor="0.6",
            label=rf"$\sigma_f$ targets ($M_f>{f_mins[0]:g}$)")
    ax.set_xlabel(r"$\sigma_f = \sqrt{\Sigma[0,0]}$")
    ax.set_ylabel(f"target count  (arm {args.arm}, n={sigma_f.size}, central {args.central:.0%})")

    ax2 = ax.twinx()
    colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(f_mins)))
    for fm, c in zip(f_mins, colors):
        ax2.plot(rep, r11[fm], color=c, marker="o", ms=4, lw=1.6,
                 label=rf"$R^{{\rm tot}}_{{11}}\,|\,M_f>{fm:g}$")
    ax2.set_ylabel(r"per-bin selection $R^{\rm tot}_{11}$")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    ax.set_title(
        f"σ_f histogram vs selection $R_{{11}}$ per flux floor — {os.path.basename(args.inp)}\n"
        f"$M_f<{f_max:g}$,  {rep.size} σ_f bins"
    )
    fig.tight_layout()

    out_path = args.out or os.path.join(
        "plots", "sigma_f_selection_" + os.path.basename(args.inp).replace(".npz", "") + ".png"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
