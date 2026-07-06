"""
plot_per_galaxy_shear_analytic.py
=================================
Bare-bones histogram of the **analytic-BFD** per-galaxy shear estimates from a
saved independent-ensemble PQR grid (``pqr_sim_{p,m}``), restricted to
``2000 < Mf < 4000``.

Every selected measurement is plotted with equal weight (plain counts, no
inverse-variance / nda weighting). The +shear and -shear catalogues are
independent injection realisations rather than ring pairs, so each arm's flux
cut is applied using that arm's *own* ``targets_{p,m}`` — not a single cut
built from one arm and reused on the other, which would misalign the two
(generally different-length) arrays.

Run::

    python -m bfd_cnf.plot_per_galaxy_shear_analytic --in data/pqr_grid.npz
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
    ap.add_argument("--in", dest="inp", default="data/pqr_grid.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mf-lo", type=float, default=2000.0)
    ap.add_argument("--mf-hi", type=float, default=4000.0)
    ap.add_argument("--bins", type=int, default=200)
    args = ap.parse_args()

    d = np.load(args.inp)

    # Each arm's flux cut uses that arm's OWN targets — the +/- catalogues are
    # independent injection realisations of different (and differently-ordered)
    # galaxies, not row-aligned ring pairs, so a cut built from one arm cannot be
    # reused on the other.
    g = {}
    for grp in ("p", "m"):
        mf = d[f"targets_{grp}"][:, 0]
        cut = (mf > args.mf_lo) & (mf < args.mf_hi)
        a1, a2 = per_object_g(d[f"pqr_sim_{grp}"][cut].astype(np.float64))
        g[grp] = (a1[np.isfinite(a1)], a2[np.isfinite(a2)])

    edges = np.linspace(-1.0, 1.0, args.bins + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, comp, name in [(ax[0], 0, "g_1"), (ax[1], 1, "g_2")]:
        for grp, color, lab in [("p", "red", "+ shear"), ("m", "blue", r"$-$ shear")]:
            gi = g[grp][comp]
            n_in = int(np.sum((gi >= -1.0) & (gi <= 1.0)))
            a.hist(gi, bins=edges, histtype="step", color=color, lw=1.2,
                   label=f"{lab} (n={n_in:,})")
        a.set_xlim(-1.0, 1.0)
        a.set_xlabel(rf"per-galaxy ${name}$")
        a.set_ylabel("count")
        a.set_title(rf"${name}$", fontsize=11)
        a.legend(fontsize=9, loc="upper right")

    fig.suptitle(f"Analytic-BFD per-galaxy shear, {args.mf_lo:.0f} < Mf < "
                 f"{args.mf_hi:.0f} (unweighted, {os.path.basename(args.inp)})",
                 fontsize=11)
    fig.tight_layout()

    out = args.out or os.path.join(
        PLOTS_DIR, f"per_galaxy_shear_analytic_mf{args.mf_lo:.0f}_{args.mf_hi:.0f}.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}  (+ n={g['p'][0].size:,}, - n={g['m'][0].size:,})")


if __name__ == "__main__":
    main()
