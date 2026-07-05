"""
plot_copy_centroid_weights.py
=============================
Pick one galaxy's set of shifted template copies and scatter their centroid
offsets (MX, MY), color-coded by the four per-copy ELBO weight factors:
detj, nda, L(X|C_X) (e=0 isotropic), and their product.

Two rows: log_scale = min and max of config.log_scale_range, so the
truncation/clipping of the centroid Gaussian by the finite copy grid is visible
(at max log_scale the Gaussian is wider than the grid → L is still large at the
outermost copies → the X-integral is clipped).

    python -m bfd_cnf.plot_copy_centroid_weights [--id 6010200003]

L(X|C_X) matches the training loss kernel exactly (flows.py:441-449):
  C_X = exp(log_scale)·I (e=0),  L = exp(-0.5·(MX²+MY²)/exp(log_scale)).
"""
from __future__ import annotations

import argparse
import os

import fitsio
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bfd_cnf.config import TRAIN_FITS_PATH, PLOTS_DIR, log_scale_range


def _L_iso(X: np.ndarray, log_scale: float) -> np.ndarray:
    """Centroid weight, e=0: C_X = exp(log_scale)·I  →  L = exp(-r²/(2 exp(ls)))."""
    r2 = X[:, 0] ** 2 + X[:, 1] ** 2
    return np.exp(-0.5 * r2 / np.exp(log_scale))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", type=int, default=None, help="galaxy id (default: most copies in scan)")
    ap.add_argument("--scan", type=int, default=20000, help="rows to scan to collect copies")
    ap.add_argument("--log-scales", type=float, nargs=2, default=list(log_scale_range),
                    metavar=("MIN", "MAX"), help="log_scale rows (default: config range)")
    args = ap.parse_args()

    h = fitsio.FITS(TRAIN_FITS_PATH)[1]
    blk = h.read(rows=np.arange(min(args.scan, h.get_nrows())),
                 columns=["id", "moments", "centroid", "nda"])
    if args.id is not None:
        gid = args.id
    else:  # well-populated parent with the SMALLEST fan radius → most clipped at max ls
        ids = blk["id"]
        r = np.hypot(blk["centroid"][:, 0], blk["centroid"][:, 1])
        uids = np.unique(ids)
        cand = [(uid, (ids == uid).sum(), r[ids == uid].max()) for uid in uids]
        cand = [c for c in cand if c[1] >= 40]          # need a visible fan
        gid = int(min(cand, key=lambda c: c[2])[0])     # smallest r_max
    sel = blk["id"] == gid
    assert sel.sum() > 1, f"id {gid} not found / single copy in first {args.scan} rows"

    m = blk["moments"][sel].astype(np.float64)
    X = blk["centroid"][sel, :2].astype(np.float64)        # [MX, MY] offsets
    nda = blk["nda"][sel].astype(np.float64)
    detj = 0.25 * (m[:, 1] ** 2 - m[:, 2] ** 2 - m[:, 3] ** 2)
    r_max = np.hypot(X[:, 0], X[:, 1]).max()

    fig, axes = plt.subplots(2, 4, figsize=(22, 11), constrained_layout=True)
    for row, ls in enumerate(args.log_scales):
        L = _L_iso(X, ls)
        prod = detj * nda * L
        sig = np.exp(ls / 2.0)                       # σ_X in MX,MY units
        # clipped-weight fraction: L mass sitting in the outer 10% radial shell
        outer = np.hypot(X[:, 0], X[:, 1]) > 0.9 * r_max
        clip_frac = L[outer].sum() / L.sum()
        panels = [("detj", detj, False), ("nda", nda, False),
                  ("L(X|C_X), e=0", L, True), ("detj × nda × L", prod, True)]
        for col, (name, c, show_sig) in enumerate(panels):
            ax = axes[row, col]
            sc = ax.scatter(X[:, 0], X[:, 1], c=c, cmap="viridis", s=45,
                            edgecolors="k", linewidths=0.3)
            fig.colorbar(sc, ax=ax)
            ax.set_title(f"{name}   [log_scale={ls:g}]", fontsize=11)
            ax.set_xlabel("MX"); ax.set_ylabel("MY")
            ax.axhline(0, color="grey", lw=0.7); ax.axvline(0, color="grey", lw=0.7)
            ax.set_aspect("equal")
            if show_sig:
                for k, st in ((1, "-"), (2, "--")):
                    ax.add_patch(plt.Circle((0, 0), k * sig, fill=False,
                                            color="red", ls=st, lw=1.2))
                ax.set_title(f"{name}   [ls={ls:g}, σ_X={sig:.0f}, "
                             f"clip={clip_frac:.0%}]", fontsize=11)
    fig.suptitle(f"Template copy centroid weights — id {gid} ({sel.sum()} copies, "
                 f"grid r_max={r_max:.0f})   red: 1σ_X / 2σ_X", fontsize=14)

    lo, hi = args.log_scales
    out = os.path.join(PLOTS_DIR, f"copy_centroid_weights_id{gid}_ls{lo:g}-{hi:g}.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"  copies={sel.sum()}  grid r_max={r_max:.1f}")
    for ls in args.log_scales:
        sig = np.exp(ls / 2.0)
        print(f"  log_scale={ls:g}: σ_X={sig:.1f}  σ_X/r_max={sig/r_max:.2f}")


if __name__ == "__main__":
    main()
