"""
check_mean_shear_bias.py
========================
Standalone diagnostic (imports only numpy + BFD, no bfd_cnf code): mean shear
per arm and for the concatenated catalog, for three R treatments — full R,
eigenvalue-clipped R, and zeroed R — for both the flow and the analytic (sim) PQR.

Mean shear is BFD's own maximiser: ĝ, cov = ``bfd.meanShear(Σ bfd.logPqr(pqr))``.
The saved flow catalogs pack R as [R11,R22,R12]; BFD's pqr uses [R11,R12,R22],
so we reorder columns with the involution FLOW_BFD before any BFD call.

Per-arm bias from the +/- lever (injected g1 = +/- gi, g2 = 0):
    m1 = (g1+ - g1-) / (2 gi) - 1        (multiplicative)
    c1 = (g1+ + g1-) / 2                 (additive, g1)
    c2 = (g2+ + g2-) / 2                 (additive, g2)

Combined check: the concatenated-catalog mean shear is a curvature-weighted
average of the two arms, so with equal-weight arms the +/- multiplicative signal
cancels and ĝ should equal the additive bias (c1, c2).  We print ĝ_comb and the
residual ĝ_comb - (c1, c2); a non-zero residual flags a +/- weight imbalance.

Run:
    python check_mean_shear_bias.py --in data/pqr_grid.npz
    python check_mean_shear_bias.py --selfcheck
"""

from __future__ import annotations

import argparse

import numpy as np
import bfd

# flow [P,Q1,Q2,R11,R22,R12] <-> bfd [P,Q1,Q2,R11,R12,R22] (its own inverse)
FLOW_BFD = [0, 1, 2, 3, 5, 4]


def _clean(pqr: np.ndarray) -> np.ndarray:
    """Keep finite rows with P > 1e-10 (float64)."""
    pqr = np.asarray(pqr, dtype=np.float64)
    return pqr[np.all(np.isfinite(pqr), axis=1) & (pqr[:, 0] > 1e-10)]


def _load_arm(d, pqr_key: str, targets_key: str, mf_range) -> np.ndarray:
    """Load one arm's PQR, optionally cut to LO < Mf < HI, then clean."""
    pqr = np.asarray(d[pqr_key], dtype=np.float64)
    if mf_range is not None:
        mf = np.abs(d[targets_key][:, 0])  # targets row-aligned with the PQR array
        pqr = pqr[(mf > mf_range[0]) & (mf < mf_range[1])]
    return _clean(pqr)


def _zero_r(pqr: np.ndarray) -> np.ndarray:
    """Copy with the R block (cols 3:6 = R11,R22,R12) zeroed -> R_tot = Q Qᵀ/P²."""
    pqr = pqr.copy()
    pqr[:, 3:6] = 0.0
    return pqr


def _clip_r(pqr: np.ndarray, lo: float = -100.0, hi: float = 100.0) -> np.ndarray:
    """Copy with per-object R eigenvalues clipped to [lo,hi] and Q rescaled to
    keep the probability mode fixed (via bfd.splitPqr/packPqr)."""
    p, q, r = bfd.splitPqr(pqr[:, FLOW_BFD])
    evals, evecs = np.linalg.eigh(r)
    ev = np.clip(evals, lo, hi)
    ratio = np.where(evals != 0.0, ev / np.where(evals != 0.0, evals, 1.0), 1.0)
    r = np.einsum("nij,nj,nkj->nik", evecs, ev, evecs)
    q = np.einsum("nij,nj,nkj,nk->ni", evecs, ratio, evecs, q)
    return bfd.packPqr(p, q, r)[:, FLOW_BFD]  # back to flow order


def _logpqr(pqr: np.ndarray) -> np.ndarray:
    """Per-object log-PQR (N,6) via bfd.logPqr (flow-order in)."""
    return np.asarray(bfd.logPqr(np.asarray(pqr, dtype=np.float64)[:, FLOW_BFD]))


def _g(lp_sum: np.ndarray) -> np.ndarray:
    """ML shear from a SUMMED log-PQR vector (6,) via bfd.meanShear."""
    return np.asarray(bfd.meanShear(lp_sum)[0])  # [0]=shear, [1]=covariance


def _g_cov(lp_sum: np.ndarray):
    """Shear and its analytic (Fisher) covariance (Sum R_tot)^-1 via bfd.meanShear."""
    g, cov = bfd.meanShear(lp_sum)
    return np.asarray(g), np.asarray(cov)


def _bias(g_p: np.ndarray, g_m: np.ndarray, gi: float) -> tuple[float, float, float]:
    """(m1, c1, c2) from the +/- shear lever."""
    m1 = (g_p[0] - g_m[0]) / (2 * gi) - 1.0
    c1 = (g_p[0] + g_m[0]) / 2.0
    c2 = (g_p[1] + g_m[1]) / 2.0
    return m1, c1, c2


def _report(label, pqr_p, pqr_m, gi, nboot, rng) -> None:
    """Point estimates with BOTH analytic (Fisher, (Sum R_tot)^-1) and
    independent-arm bootstrap errors.  The ensemble sum is linear in the
    per-object log-PQR, so a bootstrap resample is (bincount @ LP)."""
    lp_p, lp_m = _logpqr(pqr_p), _logpqr(pqr_m)
    n_p, n_m = lp_p.shape[0], lp_m.shape[0]
    Sp, Sm = lp_p.sum(0), lp_m.sum(0)
    g_p, cov_p = _g_cov(Sp)
    g_m, cov_m = _g_cov(Sm)
    g_c, cov_c = _g_cov(Sp + Sm)
    m1, c1, c2 = _bias(g_p, g_m, gi)

    # Analytic 1-sigma from the Fisher covariances (arms independent).
    a_gp, a_gm, a_gc = np.sqrt(np.diag(cov_p)), np.sqrt(np.diag(cov_m)), np.sqrt(np.diag(cov_c))
    a_g11 = np.sqrt(cov_p[0, 0] + cov_m[0, 0])
    a_m1 = a_g11 / (2 * gi)
    a_c1 = a_g11 / 2
    a_c2 = np.sqrt(cov_p[1, 1] + cov_m[1, 1]) / 2

    # Flip-combine proxy: reflect the -arm (Q1 -> -Q1) and concatenate, so the
    # multiplicative signal adds and the additive c1 cancels; then m = g1/gi - 1.
    Smf = Sm.copy(); Smf[1] = -Smf[1]  # column 1 = DG1 = Q1
    g_flip, cov_flip = _g_cov(Sp + Smf)
    m_flip = g_flip[0] / gi - 1.0
    a_mflip = np.sqrt(cov_flip[0, 0]) / gi

    # Independent-arm bootstrap.
    gp = np.empty((nboot, 2)); gm = np.empty((nboot, 2)); gc = np.empty((nboot, 2)); gf = np.empty(nboot)
    for b in range(nboot):
        sp = np.bincount(rng.integers(0, n_p, n_p), minlength=n_p) @ lp_p
        sm = np.bincount(rng.integers(0, n_m, n_m), minlength=n_m) @ lp_m
        gp[b], gm[b], gc[b] = _g(sp), _g(sm), _g(sp + sm)
        smf = sm.copy(); smf[1] = -smf[1]
        gf[b] = _g(sp + smf)[0]
    b_m1 = ((gp[:, 0] - gm[:, 0]) / (2 * gi) - 1.0).std()
    b_c1 = ((gp[:, 0] + gm[:, 0]) / 2.0).std()
    b_c2 = ((gp[:, 1] + gm[:, 1]) / 2.0).std()
    b_mflip = (gf / gi - 1.0).std()

    def f(v, b, a):  # value (bootstrap, analytic)
        return f"{v:+.5f} (b{b:.5f} a{a:.5f})"

    print(f"  {label}")
    print(f"    g(+)  g1={f(g_p[0], gp[:, 0].std(), a_gp[0])}  g2={f(g_p[1], gp[:, 1].std(), a_gp[1])}   n={n_p:,}")
    print(f"    g(-)  g1={f(g_m[0], gm[:, 0].std(), a_gm[0])}  g2={f(g_m[1], gm[:, 1].std(), a_gm[1])}   n={n_m:,}")
    print(f"    m1(lever)      ={f(m1, b_m1, a_m1)}   c1={f(c1, b_c1, a_c1)}   c2={f(c2, b_c2, a_c2)}")
    print(f"    m1(flip-comb)  ={f(m_flip, b_mflip, a_mflip)}   [g2_flip={g_flip[1]:+.5f} ~ c2]")
    print(f"    g(comb) g1={f(g_c[0], gc[:, 0].std(), a_gc[0])}  g2={f(g_c[1], gc[:, 1].std(), a_gc[1])}   n={n_p + n_m:,}")
    print(f"    g(comb)-(c1,c2) = [{g_c[0] - c1:+.5f}, {g_c[1] - c2:+.5f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_grid.npz")
    ap.add_argument("--input-g", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=100.0,
                    help="symmetric eigenvalue bound for the clip-R row (default 100).")
    ap.add_argument("--no-clip", action="store_true", help="skip the clip-R row.")
    ap.add_argument("--mf-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="keep only targets with LO < Mf < HI (default: all).")
    ap.add_argument("--nboot", type=int, default=200,
                    help="independent-arm bootstrap resamples for the errors (default 200).")
    ap.add_argument("--selfcheck", action="store_true", help="run the assert-based self-check and exit")
    args = ap.parse_args()

    if args.selfcheck:
        demo()
        return

    d = np.load(args.inp)
    gi = args.input_g
    c = args.clip
    rng = np.random.default_rng(0)
    if args.mf_range:
        print(f"flux selection: {args.mf_range[0]:g} < Mf < {args.mf_range[1]:g}")
    for field, (kp, km) in {
        "flow": ("pqr_p", "pqr_m"),
        "sim ": ("pqr_sim_p", "pqr_sim_m"),
    }.items():
        if kp not in d or km not in d:
            continue  # e.g. imsims targets: no analytic ground truth to compare against
        p = _load_arm(d, kp, "targets_p", args.mf_range)
        m = _load_arm(d, km, "targets_m", args.mf_range)
        print(f"{field.strip()}:")
        _report("full R", p, m, gi, args.nboot, rng)
        if not args.no_clip:
            _report(f"clip R [±{c:g}]", _clip_r(p, -c, c), _clip_r(m, -c, c), gi, args.nboot, rng)
        _report("zero R", _zero_r(p), _zero_r(m), gi, args.nboot, rng)


def demo() -> None:
    """Self-check: _bias inverts the linear model g = (1+m) g_true + c."""
    gi, m_true, c_true = 0.02, 0.03, (0.001, -0.002)
    g_p = np.array([(1 + m_true) * gi + c_true[0], c_true[1]])
    g_m = np.array([(1 + m_true) * (-gi) + c_true[0], c_true[1]])
    m1, c1, c2 = _bias(g_p, g_m, gi)
    assert np.isclose(m1, m_true), m1
    assert np.isclose(c1, c_true[0]) and np.isclose(c2, c_true[1]), (c1, c2)
    print("OK: bias formulas recover (m, c1, c2)")


if __name__ == "__main__":
    main()
