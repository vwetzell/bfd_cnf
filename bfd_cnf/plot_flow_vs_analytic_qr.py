"""
plot_flow_vs_analytic_qr.py
===========================
Per-cell comparison of the **flow** PQR vs the **analytic BFD** PQR
(``pqr_sim_*``) from a saved grid integration, in the (log10 Mf, Mr/Mf) plane.

The flow and the analytic estimator run on the *same* targets with the *same*
noise — only the prior differs — so any difference in the recovered shear (and
hence the multiplicative bias ``m``) comes entirely from their summed
first-derivative ``Q`` (the shear signal) and second-derivative ``R`` (the
curvature / normalisation), with ``g = R_tot⁻¹ Q_tot``.

To first order with a diagonal response,
``g1(±) ≈ Q1_tot(±) / R11_tot(±)`` and, for symmetric ±shear,
``m ≈ 2·Q1_tot(+)/(R11_tot(+)·Δg) − 1`` — so a per-cell map of the fractional
flow−analytic difference in ``Q1`` and ``R11`` (on the +shear arm) attributes
the residual ``m`` to a **signal (Q)** vs a **normalisation (R)** error.

Panels: m(flow), m(analytic), Δm=flow−analytic; ΔQ1/Q1, ΔR11/R11, counts.

Run::

    python -m bfd_cnf.plot_flow_vs_analytic_qr --in data/pqr_grid_FULL_mf3000_90000_adapt240k.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PLOTS_DIR

DELTA_G = 0.04
_TINY = 1e-30


def _qr_tot(pqr: np.ndarray):
    """Summed Q_tot (2,) and R_tot (2,2) for a per-target PQR block (pqr2g accumulation)."""
    keep = pqr[:, 0] >= 1e-10
    pqr = pqr[keep]
    if pqr.shape[0] < 3:
        return np.full(2, np.nan), np.full((2, 2), np.nan)
    P = pqr[:, 0]
    Q = pqr[:, 1:3]
    R = np.empty((pqr.shape[0], 2, 2))
    R[:, 0, 0] = pqr[:, 3]; R[:, 1, 1] = pqr[:, 4]; R[:, 0, 1] = R[:, 1, 0] = pqr[:, 5]
    Q_tot = np.nansum(Q / P[:, None], axis=0)
    R_tot = np.nansum(np.einsum("ni,nj->nij", Q, Q) / P[:, None, None] ** 2
                      - R / P[:, None, None], axis=0)
    return Q_tot, R_tot


def _g1(pqr):
    Q, R = _qr_tot(pqr)
    try:
        return float(np.linalg.solve(R, Q)[0])
    except np.linalg.LinAlgError:
        return np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp",
                    default="data/pqr_grid_FULL_mf3000_90000_adapt240k.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gridsize", type=int, default=12)
    ap.add_argument("--mincnt", type=int, default=3000)
    ap.add_argument("--mclim", type=float, default=0.3, help="m / Δm colour limit.")
    ap.add_argument("--qrclim", type=float, default=0.3, help="ΔQ1/Q1, ΔR11/R11 colour limit.")
    args = ap.parse_args()

    d = np.load(args.inp)
    pqr_p = d["pqr_p"].astype(np.float64); pqr_m = d["pqr_m"].astype(np.float64)
    sim_p = d["pqr_sim_p"].astype(np.float64); sim_m = d["pqr_sim_m"].astype(np.float64)
    tp = d["targets_p"]; mf = np.abs(tp[:, 0])
    x = np.log10(mf); y = tp[:, 1] / mf

    valid = (
        np.all(np.isfinite(pqr_p), axis=1) & np.all(np.isfinite(pqr_m), axis=1)
        & np.all(np.isfinite(sim_p), axis=1) & np.all(np.isfinite(sim_m), axis=1)
        & (pqr_p[:, 0] > 0) & (pqr_m[:, 0] > 0) & (sim_p[:, 0] > 0) & (sim_m[:, 0] > 0)
        & np.isfinite(x) & np.isfinite(y)
    )
    x, y = x[valid], y[valid]
    pqr_p, pqr_m, sim_p, sim_m = pqr_p[valid], pqr_m[valid], sim_p[valid], sim_m[valid]
    n = x.shape[0]
    print(f"{n} valid targets")

    def reduce_m_flow(c):
        i = np.asarray(c, np.int64); return (_g1(pqr_p[i]) - _g1(pqr_m[i])) / DELTA_G - 1.0

    def reduce_m_sim(c):
        i = np.asarray(c, np.int64); return (_g1(sim_p[i]) - _g1(sim_m[i])) / DELTA_G - 1.0

    def reduce_dm(c):
        return reduce_m_flow(c) - reduce_m_sim(c)

    def reduce_dQ1(c):
        i = np.asarray(c, np.int64)
        qf, _ = _qr_tot(pqr_p[i]); qs, _ = _qr_tot(sim_p[i])
        return (qf[0] - qs[0]) / (np.abs(qs[0]) + _TINY)

    def reduce_dR11(c):
        i = np.asarray(c, np.int64)
        _, rf = _qr_tot(pqr_p[i]); _, rs = _qr_tot(sim_p[i])
        return (rf[0, 0] - rs[0, 0]) / (np.abs(rs[0, 0]) + _TINY)

    def reduce_imbalance(c):
        # d(ln Q1) - d(ln R11) ≈ d(ln g1):  >0 => flow over-responds => m up
        return reduce_dQ1(c) - reduce_dR11(c)

    extent = (x.min(), x.max(), max(y.min(), 1.5), min(y.max(), 5.0))
    hexkw = dict(gridsize=args.gridsize, mincnt=args.mincnt, extent=extent, C=np.arange(n))

    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    mc = args.mclim
    # (axis, reduce, cmap, vmin, vmax, title)
    panels = [
        (ax[0, 0], reduce_m_flow, "RdBu_r", -mc, mc, "m  (flow)"),
        (ax[0, 1], reduce_m_sim,  "RdBu_r", -mc, mc, "m  (analytic BFD)"),
        (ax[0, 2], reduce_dm,     "RdBu_r", -mc, mc, r"$\Delta m$ = flow $-$ analytic"),
        # Q,R suppression: sequential, centred on the ~-0.5 level so structure shows
        (ax[1, 0], reduce_dQ1,    "magma", -0.70, -0.30, r"$\Delta Q_1/Q_1$  (+shear)  [flow$\approx$0.5$\times$analytic]"),
        (ax[1, 1], reduce_dR11,   "magma", -0.70, -0.30, r"$\Delta R_{11}/R_{11}$  (+shear)"),
        # the m driver: Q preserved more than R (>0) -> g=Q/R inflated -> m up
        (ax[1, 2], reduce_imbalance, "RdBu_r", -0.15, 0.15,
         r"imbalance $\Delta\!\ln Q_1-\Delta\!\ln R_{11}$  ($>0\Rightarrow m\!\uparrow$)"),
    ]
    arrays = {}
    for a, fn, cmap, vmin, vmax, title in panels:
        hb = a.hexbin(x, y, reduce_C_function=fn, cmap=cmap, **hexkw)
        hb.set_clim(vmin, vmax)
        arrays[title] = np.ma.filled(hb.get_array().astype(float), np.nan)
        fig.colorbar(hb, ax=a)
        a.set_title(title, fontsize=10)

    for a in ax.ravel():
        a.set_xlabel(r"$\log_{10} M_f$", fontsize=12); a.set_ylabel(r"$M_r/M_f$", fontsize=12)
        a.axhline(2.2, color="green", ls="--", lw=0.8); a.axhline(3.5, color="green", ls="--", lw=0.8)

    # quantitative attribution: does Δm track the Q/R imbalance across cells?
    dm = arrays[r"$\Delta m$ = flow $-$ analytic"]
    dq = arrays[r"$\Delta Q_1/Q_1$  (+shear)  [flow$\approx$0.5$\times$analytic]"]
    dr = arrays[r"$\Delta R_{11}/R_{11}$  (+shear)"]
    imb = arrays[r"imbalance $\Delta\!\ln Q_1-\Delta\!\ln R_{11}$  ($>0\Rightarrow m\!\uparrow$)"]
    ok = np.isfinite(dm) & np.isfinite(dq) & np.isfinite(dr) & np.isfinite(imb)
    def _corr(a, b):
        a, b = a[ok], b[ok]
        return float(np.corrcoef(a, b)[0, 1]) if a.size > 3 else np.nan
    print(f"per-cell medians:  ΔQ1/Q1 = {np.nanmedian(dq):+.3f}   ΔR11/R11 = {np.nanmedian(dr):+.3f}   "
          f"imbalance = {np.nanmedian(imb):+.3f}")
    print(f"corr(Δm, ΔQ1/Q1)   = {_corr(dm, dq):+.3f}")
    print(f"corr(Δm, ΔR11/R11) = {_corr(dm, dr):+.3f}")
    print(f"corr(Δm, imbalance)= {_corr(dm, imb):+.3f}  (Q preserved more than R -> m up: expect positive)")

    fig.suptitle(f"Flow vs analytic Q/R per cell — {os.path.basename(args.inp)}", fontsize=13)
    fig.tight_layout()
    out = args.out or os.path.join(
        PLOTS_DIR, "flow_vs_analytic_qr_"
        + os.path.basename(args.inp).replace("pqr_grid_", "").replace(".npz", "") + ".png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
