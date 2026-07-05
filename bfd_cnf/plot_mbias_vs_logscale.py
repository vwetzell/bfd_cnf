"""
plot_mbias_vs_logscale.py
==========================
Multiplicative bias ``m`` binned by an ``sx_conds`` component (``log_scale``,
``e1``, or ``e2`` -- the PSF/centroid noise condition ``[log_scale, e1, e2]``),
Flow vs Fiducial, from an independent-ensemble PQR npz (e.g.
``data/pqr_indep_xy_elbo_1M.npz``). The +/- arms are different targets, so
each arm is binned by its own ``sx_conds`` value and ``m`` is formed from the
per-bin summed +/- shears (as in :mod:`bfd_cnf.plot_bias_table`).

Run::

    python -m bfd_cnf.plot_mbias_vs_logscale --in data/pqr_indep_xy_elbo_1M.npz --xvar log_scale
    python -m bfd_cnf.plot_mbias_vs_logscale --in data/pqr_indep_xy_elbo_1M.npz --xvar e1
    python -m bfd_cnf.plot_mbias_vs_logscale --in data/pqr_indep_xy_elbo_1M.npz --xvar e2
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PLOTS_DIR, log_scale_range, e_max
from .plot_per_galaxy_shear_meanr import _qr_terms, _sum_pqr_g, _boot_mc

DELTA_G = 0.04

# column index into sx_conds = [log_scale, e1, e2], plot label, vlines, equation text
_XVAR = {
    "log_scale": dict(
        col=0, xlabel="log_scale  (½ log det C_X)", vlines=log_scale_range,
        eqn=r"$C_X = \sigma_{xy}^2 I,\ \ \mathrm{log\_scale} = \frac{1}{2}\log\det C_X = \log\sigma_{xy}^2$",
    ),
    "e1": dict(
        col=1, xlabel=r"$e_1$  (ellipticity of $C_X$)", vlines=(-e_max, e_max),
        eqn=r"$e_1 = (C_{X,xx} - C_{X,yy}) / \mathrm{tr}\,C_X$",
    ),
    "e2": dict(
        col=2, xlabel=r"$e_2$  (ellipticity of $C_X$)", vlines=(-e_max, e_max),
        eqn=r"$e_2 = 2\,C_{X,xy} / \mathrm{tr}\,C_X$",
    ),
}


def _arm_terms(pqr, x):
    valid = np.all(np.isfinite(pqr), axis=1) & (pqr[:, 0] > 1e-10)
    Qtot, Rtot = _qr_terms(pqr[valid].astype(np.float64))
    return Qtot, Rtot, x[valid]


def _binned_m(Qp, Rp, lsp, Qm, Rm, lsm, edges, input_g, nboot, rng, min_n):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mp = (lsp >= lo) & (lsp < hi)
        mm = (lsm >= lo) & (lsm < hi)
        if mp.sum() < min_n or mm.sum() < min_n:
            rows.append(dict(mid=0.5 * (lo + hi), m=np.nan, err=np.nan, n=mp.sum() + mm.sum()))
            continue
        gp = _sum_pqr_g(Qp[mp], Rp[mp])
        gm = _sum_pqr_g(Qm[mm], Rm[mm])
        m = (gp[0] - gm[0]) / (2 * input_g) - 1
        m_err, _, _ = _boot_mc(Qp[mp], Rp[mp], Qm[mm], Rm[mm], input_g, nboot, rng)
        rows.append(dict(mid=0.5 * (lo + hi), m=m, err=m_err, n=int(mp.sum() + mm.sum())))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_indep_xy_elbo_1M.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nbins", type=int, default=12)
    ap.add_argument("--min-n", type=int, default=200, help="Min targets per arm per bin.")
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--input-g", type=float, default=DELTA_G / 2)
    ap.add_argument("--xvar", choices=list(_XVAR), default="log_scale",
                    help="sx_conds component to bin by (default log_scale).")
    args = ap.parse_args()

    spec = _XVAR[args.xvar]
    d = np.load(args.inp)
    rng = np.random.default_rng(0)
    all_x = np.concatenate([d["sx_conds_p"][:, spec["col"]], d["sx_conds_m"][:, spec["col"]]])
    lo, hi = spec["vlines"]
    edges = np.linspace(lo, hi, args.nbins + 1)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    hax = ax.twinx()
    hax.hist(all_x, bins=60, range=(lo, hi), color="0.7", alpha=0.5, zorder=0)
    hax.set_ylabel("target count", color="0.5")
    hax.tick_params(axis="y", colors="0.5")
    hax.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    for field, label, color in (("pqr", "Flow", "C0"), ("pqr_sim", "Fiducial", "C1")):
        Qp, Rp, xp = _arm_terms(d[f"{field}_p"], d["sx_conds_p"][:, spec["col"]])
        Qm, Rm, xm = _arm_terms(d[f"{field}_m"], d["sx_conds_m"][:, spec["col"]])
        rows = _binned_m(Qp, Rp, xp, Qm, Rm, xm, edges, args.input_g, args.nboot, rng, args.min_n)
        x = [r["mid"] for r in rows if np.isfinite(r["m"])]
        y = [r["m"] for r in rows if np.isfinite(r["m"])]
        e = [r["err"] for r in rows if np.isfinite(r["m"])]
        ax.errorbar(x, y, yerr=e, fmt="o-", capsize=3, color=color, label=label)

    ax.axhline(0, color="k", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel("m")
    ax.set_title(f"Multiplicative bias vs {args.xvar} (trained range) — {os.path.basename(args.inp)}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.98, 0.03, spec["eqn"], transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    tag = os.path.basename(args.inp).replace(".npz", "")
    out = args.out or os.path.join(PLOTS_DIR, f"mbias_vs_{args.xvar}_{tag}.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
