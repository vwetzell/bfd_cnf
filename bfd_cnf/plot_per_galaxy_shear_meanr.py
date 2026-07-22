"""
plot_per_galaxy_shear_meanr.py
==============================
Per-galaxy shear distribution using the **mean-R linearised** estimator
(reproduces ``pergal_shear_*_meanR_invvar.png``), over a selectable flux range.

Per-object terms (pqr2g accumulation, target i):

    Q_tot,i = Q_i / P_i,
    R_tot,i = Q_i Q_iᵀ / P_i² − R_i / P_i,
    R̄       = ⟨R_tot,i⟩            (population mean, shared 2×2).

Histogram values (mean-R linearised, avoids per-object R inversion noise):

    g_i = R̄⁻¹ Q_tot,i,   histogrammed weighted by inverse variance
    w_i = (R_tot,i)_kk   for component g_k.

With ``--batch-size N``, galaxies are grouped into batches of ~N via a kd-tree
split on (log10 Mf, Mr/Mf) so each batch is compact in flux and size, and the
per-object mean-R/inv-var scheme above is replaced by the plain sum-PQR
estimate per batch: ĝ_batch = (Σ_batch R_tot,i)⁻¹ Σ_batch Q_tot,i, histogrammed
unweighted. This trades resolution for a per-estimate noise reduction of
~sqrt(N).

Annotated mean shear (proper, sum the PQRs over the *entire* population,
independent of --batch-size):
    ĝ = (Σ R_tot,i)⁻¹ Σ Q_tot,i,   ± bootstrap error.

The four groups (flow+, flow-, sim+, sim-) are each built from a saved
independent-ensemble PQR grid: the +shear and -shear catalogues are
independent injection realisations (different objects, generally different
lengths), so every quantity here — selection, sum-PQR, and the bootstrap
error — is computed per arm with no cross-arm indexing, and ``_boot_mc``
resamples each arm's indices independently.

Two panels (g1, g2); flow = solid, analytic-BFD (sim) = dotted;
+shear arm = blue, −shear arm = red.

Run::

    python -m bfd_cnf.plot_per_galaxy_shear_meanr \
        --in data/pqr_grid.npz --mf-range 1500 90000
    python -m bfd_cnf.plot_per_galaxy_shear_meanr \
        --in data/pqr_grid.npz --batch-size 100 --range 0.2
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .config import PLOTS_DIR
from .statistics import qr_log_totals


def _qr_terms(pqr: np.ndarray):
    """Per-object Q_tot,i (N,2) and R_tot,i (N,2,2) accumulation terms."""
    return qr_log_totals(pqr)


def _maybe_zero_r(pqr: np.ndarray, zero_r: bool) -> np.ndarray:
    """Diagnostic: zero the R block (cols 3:6 = R11,R22,R12) before any Q/R math,
    so R_tot collapses to Q Qᵀ/P². No-op when ``zero_r`` is False."""
    if not zero_r:
        return pqr
    pqr = np.array(pqr, dtype=np.float64)  # copy: don't mutate the loaded npz
    pqr[:, 3:6] = 0.0
    return pqr


def _sum_pqr_g(Qtot, Rtot):
    """Proper sum-PQR shear ĝ = (ΣR)⁻¹ ΣQ."""
    return np.linalg.solve(Rtot.sum(0), Qtot.sum(0))


def _arm(pqr, targets, mf_lo, mf_hi):
    mf = np.abs(targets[:, 0])
    m = (
        np.all(np.isfinite(pqr), axis=1)
        & (pqr[:, 0] > 1e-10)
        & (mf >= mf_lo)
        & (mf <= mf_hi)
    )
    Qtot, Rtot = _qr_terms(pqr[m].astype(np.float64))
    return Qtot, Rtot, targets[m], m


def _meanr_linear(Qtot, Rtot):
    """Per-object mean-R linearised estimate: g_i = R̄⁻¹ Q_tot,i, inv-var weight w_i."""
    Rbar = Rtot.mean(0)
    g_lin = (np.linalg.inv(Rbar) @ Qtot.T).T
    w = np.stack([Rtot[:, 0, 0], Rtot[:, 1, 1]], axis=1)
    return g_lin, w


def _kdtree_batches(X, batch_size):
    """Recursive kd-tree split of X (N,d), features rescaled to unit variance so
    neither axis dominates, into index groups of ~batch_size — each group compact
    in every feature, not just the one with the largest raw variance."""
    Xz = (X - X.mean(0)) / X.std(0)

    def recurse(idx):
        k = max(1, round(len(idx) / batch_size))
        if k <= 1:
            return [idx]
        axis = np.argmax(Xz[idx].var(axis=0))
        order = idx[np.argsort(Xz[idx, axis], kind="stable")]
        split = min(max(1, round(len(idx) * (k // 2) / k)), len(idx) - 1)
        return recurse(order[:split]) + recurse(order[split:])

    return recurse(np.arange(len(X)))


def _batch_g(Qtot, Rtot, targets, batch_size):
    """Per-batch sum-PQR shear; batches from a kd-tree on (log10 Mf, Mr/Mf)."""
    feats = np.column_stack(
        [np.log10(np.abs(targets[:, 0])), targets[:, 1] / targets[:, 0]]
    )
    groups = _kdtree_batches(feats, batch_size)
    vals = np.array([_sum_pqr_g(Qtot[ix], Rtot[ix]) for ix in groups])
    return vals, len(groups)


def _auto_edges(x, n_bins=None, half_range=None):
    """Bin edges: width by Freedman-Diaconis, range +/- (P99-P1)/2 (either overridable)."""
    x = x[np.isfinite(x)]
    if half_range is None:
        p1, p99 = np.percentile(x, [1, 99])
        half_range = (p99 - p1) / 2
    if n_bins is None:
        q1, q3 = np.percentile(x, [25, 75])
        h = 2 * (q3 - q1) / len(x) ** (1 / 3)
        n_bins = max(1, int(np.ceil(2 * half_range / h))) if h > 0 else 50
    return np.linspace(-half_range, half_range, n_bins + 1)


def _boot(Qtot, Rtot, nboot, rng):
    n = Qtot.shape[0]
    gs = np.empty((nboot, 2))
    for b in range(nboot):
        i = rng.integers(0, n, n)
        gs[b] = _sum_pqr_g(Qtot[i], Rtot[i])
    return gs.std(0)


def _boot_mc(Qp, Rp, Qm, Rm, gi, nboot, rng):
    """Bootstrap errors on m1, c1 (from the +/- g1 lever) and c2 (g2 input is 0)."""
    np_, nm_ = Qp.shape[0], Qm.shape[0]
    m1s = np.empty(nboot)
    c1s = np.empty(nboot)
    c2s = np.empty(nboot)
    for b in range(nboot):
        ip, im = rng.integers(0, np_, np_), rng.integers(0, nm_, nm_)
        gp = _sum_pqr_g(Qp[ip], Rp[ip])
        gm = _sum_pqr_g(Qm[im], Rm[im])
        m1s[b] = (gp[0] - gm[0]) / (2 * gi) - 1
        c1s[b] = (gp[0] + gm[0]) / 2
        c2s[b] = (gp[1] + gm[1]) / 2
    return m1s.std(), c1s.std(), c2s.std()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_grid.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--mf-range",
        type=float,
        nargs=2,
        default=(1500.0, 90000.0),
        help="flux selection Mf_lo Mf_hi (default full 1500 90000).",
    )
    ap.add_argument(
        "--range",
        type=float,
        default=None,
        help="g-axis half-width (default: auto, +/- (P99-P1)/2 of the pooled data).",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=None,
        help="number of bins (default: auto via the Freedman-Diaconis rule).",
    )
    ap.add_argument("--input-g", type=float, default=0.02)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument(
        "--zero-r",
        action="store_true",
        help="Diagnostic: set R11=R22=R12=0 in every PQR (flow and sim) before any "
        "Q/R math, so R_tot = Q Qᵀ/P². Output is tagged '_zeroR'.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="group galaxies into batches of ~this size (kd-tree split on "
        "log10 Mf, Mr/Mf) and histogram the sum-PQR estimate per batch, "
        "instead of the per-galaxy mean-R/inv-var estimate.",
    )
    ap.add_argument(
        "--selection",
        action="store_true",
        help="also compute the flux-limit selection-corrected flow (subtract each "
        "object's σ_f-bin selection log-totals over the --mf-range band), and list "
        "its bias alongside the uncorrected flow and the fiducial.",
    )
    ap.add_argument(
        "--n-samples", type=int, default=2**16,
        help="flow draws for the selection term (default 65536).",
    )
    ap.add_argument(
        "--n-bins", type=int, default=16, help="σ_f histogram bins for selection (default 16).",
    )
    args = ap.parse_args()

    lo, hi = args.mf_range
    d = np.load(args.inp)
    if args.zero_r:
        print("DIAGNOSTIC --zero-r: R11=R22=R12=0 in every PQR (R_tot = Q Qᵀ/P²)")
    # (Qtot, Rtot, targets) per arm × {flow, sim}
    arms = {
        "flow+": _arm(_maybe_zero_r(d["pqr_p"], args.zero_r), d["targets_p"], lo, hi),
        "flow-": _arm(_maybe_zero_r(d["pqr_m"], args.zero_r), d["targets_m"], lo, hi),
        "sim+": _arm(_maybe_zero_r(d["pqr_sim_p"], args.zero_r), d["targets_p"], lo, hi),
        "sim-": _arm(_maybe_zero_r(d["pqr_sim_m"], args.zero_r), d["targets_m"], lo, hi),
    }
    # Selection-corrected flow: subtract each object's σ_f-bin selection log-totals
    # (net_qr_totals) over the same [lo, hi] band, matched to the flow arms' masked
    # rows by id -> source-catalogue σ_f (as apply_selection_correction does).
    if args.selection:
        import jax.random as jr

        from apply_selection_correction import _selection_terms, _sigma_f_for_ids
        from .config import GRID_M_PATH, GRID_P_PATH, PRIOR_FLOW_PATH, Q_FLOW_PATH
        from .config import key as base_key
        from .integrate_grid import load_prior_flow
        from .models.bijections import load_stats

        r2s = load_stats(PRIOR_FLOW_PATH)
        flow = load_prior_flow(base_key, PRIOR_FLOW_PATH, Q_FLOW_PATH)
        kp, km = jr.split(base_key)
        src_p, src_m = np.load(GRID_P_PATH), np.load(GRID_M_PATH)
        for tag, base_tag, ids_all, src, ky in (
            ("fcorr+", "flow+", d["ids_p"], src_p, kp),
            ("fcorr-", "flow-", d["ids_m"], src_m, km),
        ):
            Qtot, Rtot, tg, mask = arms[base_tag]
            sig = _sigma_f_for_ids(src, np.asarray(ids_all)[mask])
            selq, selr, _, _ = _selection_terms(
                flow, r2s, lo, hi, sig, args.n_bins, args.n_samples, ky, None
            )
            arms[tag] = (Qtot - selq, Rtot - selr, tg, mask)

    fp, fm, sp, sm = arms["flow+"], arms["flow-"], arms["sim+"], arms["sim-"]
    n_tot = fp[0].shape[0] + fm[0].shape[0]
    rng = np.random.default_rng(0)

    # proper sum-PQR means + bootstrap errors -- always over the full population,
    # regardless of --batch-size.
    g = {k: _sum_pqr_g(v[0], v[1]) for k, v in arms.items()}
    e = {k: _boot(v[0], v[1], args.nboot, rng) for k, v in arms.items()}
    print(f"{n_tot:,} targets in {lo:g}<Mf<{hi:g}")
    for k in ("flow+", "flow-", "sim+", "sim-"):
        print(
            f"  {k:6s} sum-PQR g = [{g[k][0]:+.4f}±{e[k][0]:.4f}, {g[k][1]:+.4f}±{e[k][1]:.4f}]"
        )

    # m1,c1 from the g1 = +/-gi lever; c2 only (g2's true value is 0 for both arms, no
    # lever to fit a multiplicative term from).
    gi = args.input_g
    m1, c1, c2, m1_err, c1_err, c2_err = {}, {}, {}, {}, {}, {}
    bias_sets = [("flow", fp, fm)]
    if args.selection:
        bias_sets.append(("fcorr", arms["fcorr+"], arms["fcorr-"]))
    bias_sets.append(("sim", sp, sm))
    for name, arm_p, arm_m in bias_sets:
        gp_, gm_ = g[f"{name}+"], g[f"{name}-"]
        m1[name] = (gp_[0] - gm_[0]) / (2 * gi) - 1
        c1[name] = (gp_[0] + gm_[0]) / 2
        c2[name] = (gp_[1] + gm_[1]) / 2
        m1_err[name], c1_err[name], c2_err[name] = _boot_mc(
            arm_p[0], arm_p[1], arm_m[0], arm_m[1], gi, args.nboot, rng
        )
        print(
            f"  {name:4s} m1={m1[name]:+.4f}±{m1_err[name]:.4f}  "
            f"c1={c1[name]:+.4f}±{c1_err[name]:.4f}  c2={c2[name]:+.4f}±{c2_err[name]:.4f}"
        )

    # Histogram source: per-batch sum-PQR (--batch-size) or per-galaxy mean-R/inv-var.
    if args.batch_size:
        hist = {}
        for name, v in arms.items():
            vals, n_batches = _batch_g(v[0], v[1], v[2], args.batch_size)
            hist[name] = (vals, None)
            print(f"  {name:6s} {n_batches:,} batches (~{args.batch_size}/batch)")
    else:
        hist = {name: _meanr_linear(v[0], v[1]) for name, v in arms.items()}

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))

    # flow uncorr = solid, flow+sel = dashed, fiducial = dotted; +arm blue, -arm red.
    curve_specs = [("flow+", "blue", "-", 1.6), ("flow-", "red", "-", 1.6)]
    if args.selection:
        curve_specs += [("fcorr+", "blue", "--", 1.4), ("fcorr-", "red", "--", 1.4)]
    curve_specs += [("sim+", "blue", ":", 1.3), ("sim-", "red", ":", 1.3)]

    for k, comp in enumerate(("g_1", "g_2")):
        a = ax[k]
        pooled = np.concatenate([hist[name][0][:, k] for name, *_ in curve_specs])
        edges = _auto_edges(pooled, n_bins=args.bins, half_range=args.range)
        half = float(edges[-1])
        print(f"  {comp}: range=±{half:.4f}, bins={len(edges) - 1}")

        for name, color, ls, lw in curve_specs:
            vals, w = hist[name]
            a.hist(
                vals[:, k],
                bins=edges,
                weights=None if w is None else w[:, k],
                density=True,
                histtype="step",
                color=color,
                ls=ls,
                lw=lw,
            )

        # input shear (g1 = ±gi, g2 = 0) and proper sum-PQR means
        gin = (gi, -gi) if k == 0 else (0.0, 0.0)
        for v in set(gin):
            a.axvline(v, color="green", ls=":", lw=1.2)
        a.axvline(g["flow+"][k], color="blue", lw=1.5)
        a.axvline(g["flow-"][k], color="red", lw=1.5)
        a.axvline(g["sim+"][k], color="blue", lw=1.2, ls="-.")
        a.axvline(g["sim-"][k], color="red", lw=1.2, ls="-.")

        arm_disp = [("flow+", "flow+"), ("flow-", "flow$-$")]
        if args.selection:
            arm_disp += [("fcorr+", "flow+sel+"), ("fcorr-", "flow+sel$-$")]
        arm_disp += [("sim+", "fiduc+"), ("sim-", "fiduc$-$")]
        box_lines = [rf"sum-PQR mean ${comp}$:"]
        box_lines += [
            rf"{lab} = {g[key][k]:+.4f} $\pm$ {e[key][k]:.4f}" for key, lab in arm_disp
        ]
        box_lines.append("")

        bias_disp = [("flow", "flow")]
        if args.selection:
            bias_disp.append(("fcorr", "flow+sel"))
        bias_disp.append(("sim", "fiduc"))
        if k == 0:
            box_lines += [
                rf"$m_1$ {lab} = {m1[key]:+.4f} $\pm$ {m1_err[key]:.4f}"
                for key, lab in bias_disp
            ]
            box_lines += [
                rf"$c_1$ {lab} = {c1[key]:+.4f} $\pm$ {c1_err[key]:.4f}"
                for key, lab in bias_disp
            ]
        else:
            box_lines += [
                rf"$c_2$ {lab} = {c2[key]:+.4f} $\pm$ {c2_err[key]:.4f}"
                for key, lab in bias_disp
            ]
            box_lines.append(r"($m_2$ n/a: injected $g_2=0$)")
        box = "\n".join(box_lines)
        a.text(
            0.02,
            0.98,
            box,
            transform=a.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7"),
        )
        if args.batch_size:
            a.set_xlabel(
                rf"batch (n$\approx${args.batch_size}) ${comp}$  (sum-PQR per batch)"
            )
            a.set_ylabel("density")
        else:
            a.set_xlabel(
                rf"per-galaxy ${comp}$  ($\bar R^{{-1}} Q_{{tot,i}}$, inv-var weighted)"
            )
            a.set_ylabel("weighted density")
        a.set_xlim(-half, half)
        a.set_title(rf"${comp}$", fontsize=12)

    handles = [
        Line2D([], [], color="blue", lw=1.6, label="+shear arm"),
        Line2D([], [], color="red", lw=1.6, label="$-$shear arm"),
        Line2D([], [], color="0.3", lw=1.6, label="flow (solid)"),
    ]
    if args.selection:
        handles.append(
            Line2D([], [], color="0.3", lw=1.4, ls="--", label="flow+sel (dashed)")
        )
    handles += [
        Line2D([], [], color="0.3", lw=1.3, ls=":", label="fiducial (dotted)"),
        Line2D([], [], color="green", lw=1.2, ls=":", label=rf"input $g_1=\pm{gi:g}$"),
    ]
    ax[1].legend(handles=handles, fontsize=9, loc="upper right")

    fig.suptitle(
        f"Per-galaxy shear, {lo:g} < $M_f$ < {hi:g} ({os.path.basename(args.inp)})",
        fontsize=12,
    )
    if args.batch_size:
        hist_desc = (
            rf"Histogram (batched sum-PQR, n$\approx${args.batch_size}/batch, "
            r"batches from a kd-tree on $\log_{10}M_f, M_r/M_f$): "
            r"$\hat g_{batch}=(\sum_{batch}R_{tot,i})^{-1}\sum_{batch}Q_{tot,i}$."
        )
    else:
        hist_desc = (
            r"Histogram (mean-R linearised): $g_i=\bar R^{-1}Q_{tot,i}$, weighted by "
            r"inv-var $w_i=(R_{tot,i})_{kk}$."
        )
    cap = (
        r"Per-object: $Q_{tot,i}=Q_i/P_i$, $R_{tot,i}=Q_iQ_i^T/P_i^2-R_i/P_i$, "
        r"$\bar R=\langle R_{tot,i}\rangle$.  "
        + hist_desc
        + "\n"
        + r"Annotated (proper, full population): $\hat g=(\sum_i R_{tot,i})^{-1}\sum_i Q_{tot,i}$.  "
        r"$m_1,c_1$ from the $\pm g_1$ lever: $\hat g_1=(1+m_1)g_{1,true}+c_1$; "
        r"$c_2=\hat g_2$ (injected $g_2=0$)."
    )
    fig.text(
        0.5,
        0.08,
        cap,
        ha="center",
        va="bottom",
        fontsize=8.5,
        linespacing=1.8,
        bbox=dict(boxstyle="round,pad=1.5", fc="0.95", ec="0.8"),
    )
    fig.tight_layout(rect=(0, 0.22, 1, 1))

    tag = f"mf{lo:g}_{hi:g}".replace(".", "p")
    method_tag = f"batch{args.batch_size}_sumPQR" if args.batch_size else "meanR_invvar"
    if args.selection:
        method_tag += "_sel"
    out = args.out or os.path.join(PLOTS_DIR, f"pergal_shear_{tag}_{method_tag}.png")
    if args.zero_r:  # record the diagnostic in the filename, even for an explicit --out
        base, ext = os.path.splitext(out)
        out = f"{base}_zeroR{ext}"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
