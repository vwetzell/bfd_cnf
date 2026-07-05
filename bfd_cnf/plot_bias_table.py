"""
plot_bias_table.py
===================
Table figures of the population-level shear bias, comparing the flow PQR
("pqr") against the analytic-BFD PQR ("pqr_sim"), from an independent-ensemble
PQR npz (e.g. ``data/pqr_indep_xy_elbo_1M.npz``).

Two tables, both with bootstrap ± errors:

  * **per-arm sums** -- g1, g2 from summing PQR within each arm separately:
    Flow +, Flow -, Fiducial +, Fiducial -.
  * **total sums** -- m, c1, c2 formed by combining the +/- arm sums (the
    usual multiplicative/additive bias): Flow, Fiducial.

Run::

    python -m bfd_cnf.plot_bias_table --in data/pqr_indep_xy_elbo_1M.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PLOTS_DIR
from .plot_per_galaxy_shear_meanr import _qr_terms, _sum_pqr_g, _boot, _boot_mc

DELTA_G = 0.04


def _render_table(rows, col_labels, title, out):
    fig, ax = plt.subplots(figsize=(1.7 * len(col_labels) + 1.6, 0.7 + 0.45 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=[r[1:] for r in rows], rowLabels=[r[0] for r in rows],
                    colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.0)
    ax.set_title(title, fontsize=14, pad=16)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_indep_xy_elbo_1M.npz")
    ap.add_argument("--nboot", type=int, default=200,
                    help="Bootstrap resamples per population (default 200).")
    ap.add_argument("--input-g", type=float, default=DELTA_G / 2,
                    help="Per-arm applied shear (default 0.02, i.e. Δg=0.04).")
    args = ap.parse_args()

    d = np.load(args.inp)
    rng = np.random.default_rng(0)

    arms = {}
    for field, name in (("pqr", "Flow"), ("pqr_sim", "Fiducial")):
        for sign, label in (("p", "+"), ("m", "-")):
            pqr = d[f"{field}_{sign}"].astype(np.float64)
            keep = np.all(np.isfinite(pqr), axis=1) & (pqr[:, 0] > 1e-10)
            Qtot, Rtot = _qr_terms(pqr[keep])
            arms[(name, label)] = (Qtot, Rtot, int(keep.sum()))

    tag = os.path.basename(args.inp).replace(".npz", "")

    # ---- per-arm table: g1, g2 ± bootstrap sigma ----
    rows = []
    for name in ("Flow", "Fiducial"):
        for label in ("+", "-"):
            Qtot, Rtot, n = arms[(name, label)]
            g = _sum_pqr_g(Qtot, Rtot)
            sig = _boot(Qtot, Rtot, args.nboot, rng)
            rows.append((f"{name} {label}", f"{n:,}",
                         f"{g[0]:+.4f} ± {sig[0]:.4f}",
                         f"{g[1]:+.4f} ± {sig[1]:.4f}"))
    _render_table(rows, ["N targets", "g1", "g2"],
                  "Per-arm sum-PQR shear  (Flow vs Fiducial)",
                  os.path.join(PLOTS_DIR, f"bias_table_per_arm_{tag}.png"))

    # ---- total table: m, c1, c2 ± bootstrap sigma ----
    rows = []
    for name in ("Flow", "Fiducial"):
        Qp, Rp, _ = arms[(name, "+")]
        Qm, Rm, _ = arms[(name, "-")]
        gp = _sum_pqr_g(Qp, Rp)
        gm = _sum_pqr_g(Qm, Rm)
        m = (gp[0] - gm[0]) / (2 * args.input_g) - 1
        c1 = (gp[0] + gm[0]) / 2
        c2 = (gp[1] + gm[1]) / 2
        m_err, c1_err, c2_err = _boot_mc(Qp, Rp, Qm, Rm, args.input_g, args.nboot, rng)
        rows.append((name,
                     f"{m:+.4f} ± {m_err:.4f}",
                     f"{c1:+.4f} ± {c1_err:.4f}",
                     f"{c2:+.4f} ± {c2_err:.4f}"))
    _render_table(rows, ["m", "c1", "c2"],
                  "Total (combined-arm) bias  (Flow vs Fiducial)",
                  os.path.join(PLOTS_DIR, f"bias_table_total_{tag}.png"))


if __name__ == "__main__":
    main()
