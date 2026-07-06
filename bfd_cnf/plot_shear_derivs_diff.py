"""
plot_shear_derivs_diff.py
=========================
Plot the **difference** of the shear-derivative panels between two centering-bias
ellipticity slices, e.g. ``e2=0.2`` minus ``e2=0.0``:

    Δ(field) = field(e2=b) − field(e2=a)   for field in
    {P, ∂P/∂g1, ∂P/∂g2, ∂²P/∂g1², ∂²P/∂g2², ∂²P/∂g1∂g2}

The top-left panel ΔP = P(e2=b) − P(e2=a) is the centering-ellipticity response
(after the SigmaX dipole + permute fix, this is the +M2/−M2 dipole).

Reuses :func:`bfd_cnf.plot_shear_derivs_compare.compute_planes` (the prior-flow
evaluation + autodiff shear derivatives), so it only loads the trained flow and
the cached standardiser — the FITS table is not needed.

Run from the repo root::

    python -m bfd_cnf.plot_shear_derivs_diff --log-scale 13.3 --e2a 0.0 --e2b 0.2
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bfd_cnf.config import PRIOR_FLOW_PATH, Q_FLOW_PATH, PLOTS_DIR, key as base_key
from bfd_cnf.models.flows import build_flows
from bfd_cnf.training import load_models
from bfd_cnf.plot_shear_derivs_compare import load_standardiser, compute_planes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Difference of shear-derivative panels between two e2 slices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log10mf", type=float, default=math.log10(2000.0))
    p.add_argument("--mrmf", type=float, default=3.3)
    p.add_argument("--log-scale", type=float, default=13.3)
    p.add_argument("--e1", type=float, default=0.0)
    p.add_argument("--e2a", type=float, default=0.0, help="Reference e2 (subtracted).")
    p.add_argument("--e2b", type=float, default=0.2, help="Comparison e2.")
    p.add_argument("--g1", type=float, default=0.0)
    p.add_argument("--g2", type=float, default=0.0)
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--m-range", type=float, default=0.5)
    p.add_argument("--prior", default=None,
                   help="Prior-flow .eqx to evaluate (default: config canonical).")
    p.add_argument("--q", default=None,
                   help="Q-flow .eqx (default: config canonical).")
    p.add_argument("--stats", default=None,
                   help="raw2standard_stats npz (mean,std) the flow was trained with "
                        "(default: legacy data/raw2standard_stats.npz).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log10mf = args.log10mf
    mf_raw = 10.0**log10mf
    prior_path = args.prior or PRIOR_FLOW_PATH
    q_path = args.q or Q_FLOW_PATH
    mean, std = load_standardiser(prior_path, args.stats)
    print("Loading flow...")
    if not (os.path.exists(prior_path) and os.path.exists(q_path)):
        raise FileNotFoundError(f"Trained flow weights not found:\n  {prior_path}\n  {q_path}")
    prior_flow, q_flow = build_flows(base_key, latent_dim=4, cond_dim=16, prior_size_loc_c1=float(mean[1] / std[1]))
    prior_trained, _ = load_models(prior_flow, q_flow, prior_path, q_path)

    common = dict(log10mf=log10mf, mrmf=args.mrmf, log_scale=args.log_scale,
                  e1=args.e1, g1=args.g1, g2=args.g2, n=args.n, r=args.m_range)
    print(f"Evaluating e2={args.e2a} ...")
    planes_a, (m1mr, m2mr) = compute_planes(prior_trained, mean, std, e2=args.e2a, **common)
    print(f"Evaluating e2={args.e2b} ...")
    planes_b, _ = compute_planes(prior_trained, mean, std, e2=args.e2b, **common)

    keys = ["p", "dg1", "dg2", "dg1dg1", "dg2dg2", "dg1dg2"]
    labels = [
        r"$\Delta P$",
        r"$\Delta\,\partial P/\partial g_1$",
        r"$\Delta\,\partial P/\partial g_2$",
        r"$\Delta\,\partial^2 P/\partial g_1^2$",
        r"$\Delta\,\partial^2 P/\partial g_2^2$",
        r"$\Delta\,\partial^2 P/\partial g_1\partial g_2$",
    ]
    diff = {k: planes_b[k] - planes_a[k] for k in keys}

    extent = (m1mr[0], m1mr[-1], m2mr[0], m2mr[-1])
    imshow_kw = dict(aspect="auto", origin="lower", extent=extent, interpolation="none", cmap="RdBu")

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    for a, k, lab in zip(ax.ravel(), keys, labels):
        v = np.abs(diff[k]).max() + 1e-30
        im = a.imshow(diff[k], vmin=-v, vmax=v, **imshow_kw)
        a.set_xlabel(r"$M_1/M_r$", fontsize=18)
        a.set_ylabel(r"$M_2/M_r$", fontsize=18)
        a.axvline(0.0, color="grey", lw=1)
        a.axhline(0.0, color="grey", lw=1)
        plt.colorbar(im, ax=a).set_label(label=lab, size=18)

    fig.suptitle(
        rf"Shear-derivative difference  $e_2={args.e2b:.3g}$ minus $e_2={args.e2a:.3g}$  "
        rf"($\log_{{10}}M_f = {log10mf:.3g}$, $M_r/M_f = {args.mrmf:.3g}$, "
        rf"$g = ({args.g1:.3g}, {args.g2:.3g})$, $\log\,\mathrm{{scale}} = {args.log_scale:.3g}$, "
        rf"$e_1 = {args.e1:.3g}$)",
        fontsize=20,
    )
    plt.tight_layout()
    out = os.path.join(
        PLOTS_DIR,
        f"shear_derivs_DIFF_lgmf{log10mf:.2f}_mrmf{args.mrmf:.2f}_ls{args.log_scale:.1f}"
        f"_e1{args.e1:.2f}_e2{args.e2a:.2f}_vs_e2{args.e2b:.2f}.png",
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
