"""
plot_flow_vs_analytic_qr.py
===========================
Per-cell comparison of the **flow** PQR vs the **analytic BFD** PQR
(``pqr_sim_*``) from a saved independent-ensemble grid integration, in the
(log10 Mf, Mr/Mf) plane.

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

The +shear and -shear catalogues are independent injection realisations, not
ring pairs, so the ``m``/``Δm`` panels bin both arms onto one shared hex grid
and split each cell's carried indices back out by arm (mirroring
``plot_mbias_hexbin_independent.py``); a cell needs at least
``--mincnt-per-arm`` targets from *each* arm to be coloured. The Q/R
attribution panels are a +shear-arm-only diagnostic and only need that arm's
own per-cell count.

Panels: m(flow), m(analytic), Δm=flow−analytic; ΔQ1/Q1, ΔR11/R11 (+arm only), counts.

Run::

    python -m bfd_cnf.plot_flow_vs_analytic_qr --in data/pqr_grid.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PLOTS_DIR
from .statistics import qr_log_totals

DELTA_G = 0.04
_TINY = 1e-30


def _qr_tot(pqr: np.ndarray):
    """Summed Q_tot (2,) and R_tot (2,2) for a per-target PQR block (pqr2g accumulation)."""
    keep = pqr[:, 0] >= 1e-10
    pqr = pqr[keep]
    if pqr.shape[0] < 3:
        return np.full(2, np.nan), np.full((2, 2), np.nan)
    qt, Rtot = qr_log_totals(pqr)
    return np.nansum(qt, axis=0), np.nansum(Rtot, axis=0)


def _g1(pqr):
    Q, R = _qr_tot(pqr)
    try:
        return float(np.linalg.solve(R, Q)[0])
    except np.linalg.LinAlgError:
        return np.nan


def _arm_coords(pqr: np.ndarray, sim: np.ndarray, targets: np.ndarray):
    """Return (x=log10 Mf, y=Mr/Mf, pqr, sim) keeping only finite, P>0 targets in both."""
    mf = np.abs(targets[:, 0])
    x = np.log10(mf)
    y = targets[:, 1] / mf
    valid = (
        np.all(np.isfinite(pqr), axis=1) & np.all(np.isfinite(sim), axis=1)
        & (pqr[:, 0] > 0) & (sim[:, 0] > 0) & np.isfinite(x) & np.isfinite(y)
    )
    return x[valid], y[valid], pqr[valid], sim[valid]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_grid.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gridsize", type=int, default=12)
    ap.add_argument("--mincnt-per-arm", type=int, default=1500,
                     help="Min targets PER ARM in a cell for a stable m (default 1500).")
    ap.add_argument("--mclim", type=float, default=0.3, help="m / Δm colour limit.")
    ap.add_argument("--qrclim", type=float, default=0.3, help="ΔQ1/Q1, ΔR11/R11 colour limit.")
    args = ap.parse_args()

    d = np.load(args.inp)
    xp, yp, pqr_p, sim_p = _arm_coords(
        d["pqr_p"].astype(np.float64), d["pqr_sim_p"].astype(np.float64), d["targets_p"]
    )
    xm, ym, pqr_m, sim_m = _arm_coords(
        d["pqr_m"].astype(np.float64), d["pqr_sim_m"].astype(np.float64), d["targets_m"]
    )
    n_p, n_m = xp.shape[0], xm.shape[0]
    print(f"+arm {n_p:,} valid targets, -arm {n_m:,} valid targets")

    # Concatenate both arms onto ONE hex grid; carried index < n_p => +arm.
    x_all = np.concatenate([xp, xm])
    y_all = np.concatenate([yp, ym])
    min_arm = args.mincnt_per_arm

    def _split(c):
        """Split a cell's carried indices by arm; None if either arm is under-populated.

        matplotlib's hexbin *drops* a cell outright (from get_array() and
        get_offsets() alike) whenever reduce_C_function returns NaN, rather than
        masking it in place — so every reduce function below must return NaN on
        exactly the same condition, or the panels' returned arrays end up
        different lengths and silently misaligned with each other. All six
        panels therefore share this one both-arms-populated gate, even the
        +arm-only Q/R attribution panels.
        """
        idx = np.asarray(c, np.int64)
        ip, im = idx[idx < n_p], idx[idx >= n_p] - n_p
        if ip.size < min_arm or im.size < min_arm:
            return None, None
        return ip, im

    def reduce_m_flow(c):
        ip, im = _split(c)
        if ip is None:
            return np.nan
        return (_g1(pqr_p[ip]) - _g1(pqr_m[im])) / DELTA_G - 1.0

    def reduce_m_sim(c):
        ip, im = _split(c)
        if ip is None:
            return np.nan
        return (_g1(sim_p[ip]) - _g1(sim_m[im])) / DELTA_G - 1.0

    def reduce_dm(c):
        ip, im = _split(c)
        if ip is None:
            return np.nan
        m_flow = (_g1(pqr_p[ip]) - _g1(pqr_m[im])) / DELTA_G - 1.0
        m_sim = (_g1(sim_p[ip]) - _g1(sim_m[im])) / DELTA_G - 1.0
        return m_flow - m_sim

    def reduce_dQ1(c):
        ip, _ = _split(c)
        if ip is None:
            return np.nan
        qf, _r = _qr_tot(pqr_p[ip]); qs, _rs = _qr_tot(sim_p[ip])
        return (qf[0] - qs[0]) / (np.abs(qs[0]) + _TINY)

    def reduce_dR11(c):
        ip, _ = _split(c)
        if ip is None:
            return np.nan
        _q, rf = _qr_tot(pqr_p[ip]); _qs, rs = _qr_tot(sim_p[ip])
        return (rf[0, 0] - rs[0, 0]) / (np.abs(rs[0, 0]) + _TINY)

    def reduce_imbalance(c):
        # d(ln Q1) - d(ln R11) ≈ d(ln g1):  >0 => flow over-responds => m up
        ip, _ = _split(c)
        if ip is None:
            return np.nan
        qf, rf = _qr_tot(pqr_p[ip]); qs, rs = _qr_tot(sim_p[ip])
        dq = (qf[0] - qs[0]) / (np.abs(qs[0]) + _TINY)
        dr = (rf[0, 0] - rs[0, 0]) / (np.abs(rs[0, 0]) + _TINY)
        return dq - dr

    extent = (x_all.min(), x_all.max(), max(y_all.min(), 1.5), min(y_all.max(), 5.0))
    hexkw = dict(gridsize=args.gridsize, mincnt=2 * min_arm, extent=extent,
                 C=np.arange(n_p + n_m))

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
        hb = a.hexbin(x_all, y_all, reduce_C_function=fn, cmap=cmap, **hexkw)
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
