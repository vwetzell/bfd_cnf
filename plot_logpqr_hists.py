"""
plot_logpqr_hists.py
====================
Standalone (numpy + matplotlib + BFD only, no bfd_cnf code): a 1x6 overlay of
the per-object log-PQR component distributions, sim vs a temperature-rescaled
flow.

log-PQR = ``bfd.logPqr(pqr)``: components [logP, Q1/P, Q2/P, then the R block].
bfd.logPqr's R block is the raw d2logP = R/P - QQ/P2; we negate it to the R_tot
= QQ/P2 - R/P convention (as statistics.qr_log_totals does), so the R panels
match the shear-estimator sign.  The saved flow catalogs pack R as [R11,R22,R12];
BFD uses [R11,R12,R22], so columns are reordered (FLOW_BFD) before the BFD call.

Principled rescale from logP alone: if the flow log-likelihood surface is just a
temperature-scaled sim, logP_sim = alpha*logP_flow + const, then EVERY shear
derivative scales by the same alpha (the const drops): Q_sim = alpha*Q_flow,
R_sim = alpha*R_flow.  We fix alpha = IQR(logP_sim)/IQR(logP_flow) from the logP
component only, then apply that single alpha to all six components.  logP also
gets its (physically irrelevant) additive offset removed; the derivative
components are scaled with NO shift, so any residual mismatch is a real failure
of the single-temperature model.

Binning: Freedman-Diaconis width (same scheme as plot_per_galaxy_shear_meanr),
over fixed per-component display ranges (RANGES).

Run:
    python plot_logpqr_hists.py --in data/pqr_indep_xy_elbo_1M.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bfd

# flow [P,Q1,Q2,R11,R22,R12] -> bfd [P,Q1,Q2,R11,R12,R22] (its own inverse)
FLOW_BFD = [0, 1, 2, 3, 5, 4]

LABELS = [
    r"$\log P$",
    r"$Q_1/P$",
    r"$Q_2/P$",
    r"$R^{\rm tot}_{11}$",
    r"$R^{\rm tot}_{12}$",
    r"$R^{\rm tot}_{22}$",
]
LABELS_TXT = ["logP", "Q1/P", "Q2/P", "Rtot11", "Rtot12", "Rtot22"]

# Fixed display range per component. R uses the R_tot = QQ/P2 - R/P sign, so the
# R11/R22 ranges are the mirror of the earlier d2logP ranges (-50,25) -> (-25,50).
RANGES = [
    (10.0, 26.0),    # logP
    (-12.0, 12.0),   # Q1/P
    (-12.0, 12.0),   # Q2/P
    (-25.0, 50.0),   # Rtot11
    (-10.0, 10.0),   # Rtot12
    (-25.0, 50.0),   # Rtot22
]


def _logpqr_paired(flow_p, sim_p, flow_m, sim_m):
    """Pooled flow & sim log-PQR (N,6), row-ALIGNED on a common per-arm validity
    mask (finite & P>0 in BOTH), so a flow histogram can be selected by a sim
    condition.  bfd.logPqr stores logQ = Q/P and raw d2logP = R/P - QQ/P2; we
    negate the R block to the R_tot = QQ/P2 - R/P convention (as
    statistics.qr_log_totals) so all six components match the shear-estimator sign."""
    fl, sl = [], []
    for f, s in ((flow_p, sim_p), (flow_m, sim_m)):
        f = np.asarray(f, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        ok = (np.all(np.isfinite(f), axis=1) & (f[:, 0] > 1e-10)
              & np.all(np.isfinite(s), axis=1) & (s[:, 0] > 1e-10))
        lf = np.asarray(bfd.logPqr(f[ok][:, FLOW_BFD])); lf[:, 3:6] *= -1.0
        ls = np.asarray(bfd.logPqr(s[ok][:, FLOW_BFD])); ls[:, 3:6] *= -1.0
        fl.append(lf)
        sl.append(ls)
    return np.concatenate(fl), np.concatenate(sl)


def _iqr(x: np.ndarray) -> float:
    """Interquartile range over the finite values."""
    x = x[np.isfinite(x)]
    return float(np.subtract(*np.percentile(x, [75, 25])))


def _median(x: np.ndarray) -> float:
    return float(np.median(x[np.isfinite(x)]))


def _edges(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Freedman-Diaconis bin width (from x's IQR) over the fixed range [lo, hi]."""
    x = x[np.isfinite(x)]
    q1, q3 = np.percentile(x, [25, 75])
    h = 2 * (q3 - q1) / len(x) ** (1 / 3)
    n = max(1, int(np.ceil((hi - lo) / h))) if h > 0 else 50
    return np.linspace(lo, hi, n + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_indep_xy_elbo_1M.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck", action="store_true",
                    help="verify the single-alpha scaling identity and exit")
    args = ap.parse_args()

    if args.selfcheck:
        demo()
        return

    d = np.load(args.inp)
    flow, sim = _logpqr_paired(d["pqr_p"], d["pqr_sim_p"], d["pqr_m"], d["pqr_sim_m"])

    # Single temperature scale from the logP component alone.
    alpha = _iqr(sim[:, 0]) / _iqr(flow[:, 0])
    print(f"principled temperature scale from logP:  alpha = {alpha:.4g}")
    print("per-component native IQR-match scale (for contrast; alpha would match all if flow=sim^(1/alpha)):")
    for i in range(6):
        print(f"  {LABELS_TXT[i]:14s} native = {_iqr(sim[:, i]) / _iqr(flow[:, i]):.4g}")

    med_f0, med_s0 = _median(flow[:, 0]), _median(sim[:, 0])

    # Split the sim's Q panels by the sign of tr(R_tot) = Rtot11 + Rtot22 (cols 3+5).
    # tr(R_tot)>0 is the normal peaked-likelihood bulk; tr(R_tot)<0 is anti-peaked.
    trR_pos = (sim[:, 3] + sim[:, 5]) > 0
    frac_pos = float(np.mean(trR_pos))
    print(f"sim tr(R_tot)>0 fraction: {frac_pos:.4f}")

    fig, ax = plt.subplots(1, 6, figsize=(28, 4.8))
    for i in range(6):
        if i == 0:  # logP: strip the (irrelevant) additive offset, then temperature scale
            f_res = alpha * (flow[:, i] - med_f0) + med_s0
        else:  # derivatives: pure temperature scale, NO shift (const drops under d/dg)
            f_res = alpha * flow[:, i]
        edges = _edges(sim[:, i], *RANGES[i])  # FD width over the fixed range
        if i in (1, 2):  # Q panels: split the sim by sign of tr(R_tot)
            ax[i].hist(sim[trR_pos, i], bins=edges, histtype="step", color="tab:orange",
                       lw=1.4, density=True, label=rf"Sim tr$(R_{{\rm tot}})>0$ ({frac_pos:.0%})")
            ax[i].hist(sim[~trR_pos, i], bins=edges, histtype="step", color="tab:green",
                       lw=1.4, density=True, label=rf"Sim tr$(R_{{\rm tot}})<0$ ({1 - frac_pos:.0%})")
        else:
            ax[i].hist(sim[:, i], bins=edges, histtype="step", color="tab:orange",
                       lw=1.4, density=True, label="Sim")
        ax[i].hist(f_res, bins=edges, histtype="step", color="tab:blue",
                   lw=1.4, density=True, label=rf"Flow $\times\alpha$")
        if i in (3, 5):  # R11, R22: flow (x alpha) for objects the SIM flags tr(R)<0
            ax[i].hist(alpha * flow[~trR_pos, i], bins=edges, histtype="step",
                       color="tab:green", lw=1.4, density=True,
                       label=rf"Flow$\times\alpha$ | sim tr$(R)<0$")
        ax[i].set_xlim(*RANGES[i])
        ax[i].set_title(LABELS[i], fontsize=13)
        ax[i].legend(fontsize=9)
    ax[0].set_ylabel("density", fontsize=13)

    fig.suptitle(rf"log-PQR: sim vs flow$\times\alpha$ (single temperature $\alpha$={alpha:.3g} "
                 rf"from logP) — {os.path.basename(args.inp)}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = args.out or os.path.join(
        "plots", "logpqr_hists_"
        + os.path.basename(args.inp).replace(".npz", "") + ".png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def demo() -> None:
    """Prove the single-alpha scaling is proper: the temperature scaling
    P -> P^alpha maps the WHOLE log-PQR (logP, logQ=Q/P, logR) by the same
    factor alpha, so main()'s `alpha * flow[:, i]` on every derivative component
    is exact (alpha, not alpha^2 -- a vertical scale of logP, not a rescale of g).
    Checks logPqr(P^alpha) == alpha * logPqr(P) on random per-object PQR."""
    rng = np.random.default_rng(0)
    alpha = 1.795
    P = rng.uniform(0.5, 2.0, 6)
    Q = rng.normal(size=(6, 2))
    A = rng.normal(size=(6, 2, 2))
    R = A + A.transpose(0, 2, 1)  # symmetric raw R
    # probability-PQR of P, bfd order [P, Q1, Q2, R11, R12, R22]
    pqr = np.column_stack([P, Q[:, 0], Q[:, 1], R[:, 0, 0], R[:, 0, 1], R[:, 1, 1]])
    # f = P^alpha and its shear derivatives (chain rule)
    f = P ** alpha
    fg = alpha * P[:, None] ** (alpha - 1) * Q
    fgg = (alpha * (alpha - 1) * P[:, None, None] ** (alpha - 2) * Q[:, :, None] * Q[:, None, :]
           + alpha * P[:, None, None] ** (alpha - 1) * R)
    pqr_a = np.column_stack([f, fg[:, 0], fg[:, 1], fgg[:, 0, 0], fgg[:, 0, 1], fgg[:, 1, 1]])

    lp = np.asarray(bfd.logPqr(pqr))
    lp_a = np.asarray(bfd.logPqr(pqr_a))
    err = np.abs(lp_a - alpha * lp).max()
    print("component-wise  logPqr(P^alpha) / (alpha * logPqr(P))  (should all be 1):")
    ratio = lp_a / (alpha * lp)
    for name, r in zip(LABELS_TXT, ratio.mean(axis=0)):
        print(f"  {name:8s} {r:.6f}")
    print(f"max|logPqr(P^alpha) - alpha*logPqr(P)| = {err:.2e}")
    assert err < 1e-8, err
    print("OK: one alpha scales logP, logQ and logR identically (temperature scaling)")


if __name__ == "__main__":
    main()
