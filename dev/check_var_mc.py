"""Measure Var_MC(qhat) directly, from two runs that differ ONLY in S.

jhat = -(C/A - (B/A)^2) contains the square of a Monte-Carlo estimate, so

    E[jhat] = j - Var_MC(qhat)      =>   sum jhat is too SMALL

and since m1 = sum qhat / (g sum jhat) - 1, the estimator picks up

    dm1 = sum_i Var_MC(qhat_i) / sum_i j_i          (positive, flat in g)

Two saved runs over the same targets at S and S' give Var_MC without any new
compute: with q(S) = q + d(S) and Var(d) = V/S,

    Var[q(S) - q(S')] = V/S + V/S' - 2 Cov  =  (V/S)(1 - S/S')   if nested
                                             (V/S)(1 + S/S')     if independent

so V/S is bracketed within a factor <2 either way.  Both brackets are printed.

ANSWER (2026-08-17): the term is REAL and it is NOT SMALL -- it contributes
+0.0018 to +0.0030 to m1 at S = 32768 -- but it is the SAME in both paths
(moment-space +0.00176..+0.00294, centroid +0.00178..+0.00301), so it is not
the centroid path's differential +0.006.

Two consequences that outlive that null:

1. "Converged in S" is a CANCELLATION, not a convergence.  This one term is
   +0.0025 at S = 32768 and would be +0.010 at 8192, yet the measured net
   moves the other way (moment-space control -0.0048 -> -0.0002).  So the
   competing O(1/S) terms are cancelling at the ~0.003 level and every
   "converged" number in HANDOFF.md is converged only to about that.
2. The obvious next measurement is the same quantity for the CENTROID CONTROL
   specifically, which needs two runs differing only in the DRAW seed --
   `check_selfconsistency_noisy.py` has no such flag today (`--seed` moves the
   targets too).  That run also delivers the draw-realisation error bar that no
   number in this investigation currently has.
"""
import numpy as np


def load(path):
    d = np.load(path)
    keep = d["keep"] if "keep" in d.files else np.ones(len(d["plus_q"]), bool)
    return (d["plus_q"][keep], d["plus_r"][keep],
            d["minus_q"][keep], d["minus_r"][keep])


def compare(lo_path, hi_path, S_lo, S_hi, label, g=0.02):
    qlo, rlo, qmlo, rmlo = load(lo_path)
    qhi, rhi, qmhi, rmhi = load(hi_path)
    n = min(len(qlo), len(qhi))
    ok = np.ones(n, bool)
    for a in (qlo[:n], qhi[:n], qmlo[:n], qmhi[:n]):
        ok &= np.isfinite(a).all(1)
    for a in (rlo[:n], rhi[:n], rmlo[:n], rmhi[:n]):
        ok &= np.isfinite(a).reshape(n, -1).all(1)

    print(f"--- {label}   S: {S_lo} -> {S_hi}   (N = {int(ok.sum())})")
    for arm, (a, b, rr) in (("+g", (qlo, qhi, rhi)), ("-g", (qmlo, qmhi, rmhi))):
        d = (a[:n][ok] - b[:n][ok])[:, 0]
        j = -rr[:n][ok][:, 0, 0]
        vS = d.var()                       # = V/S_lo * (1 -/+ S_lo/S_hi)
        lo = vS / (1.0 + S_lo / S_hi)      # independent-draws reading
        hi = vS / (1.0 - S_lo / S_hi)      # nested-draws reading
        # translate to the R deficit AT THE HIGHER S
        f = S_lo / S_hi
        print(f"    {arm}: Var[q(S)-q(S')] = {vS:.4g} ->  V/S_lo in "
              f"[{lo:.4g}, {hi:.4g}]")
        print(f"        predicted dm1 at S={S_hi}:  "
              f"[{f*lo/j.mean():+.5f}, {f*hi/j.mean():+.5f}]"
              f"     (mean j = {j.mean():.4g})")
    print()


if __name__ == "__main__":
    compare("dev/pqr_depth_2.73.npz", "dev/pqr_depth_2.73_S32k.npz",
            8192, 32768, "moment-space (CLEAN path), deep")
    compare("dev/pqr_deep_centroid_20k.npz", "dev/pqr_deep_centroid_S32k.npz",
            8192, 32768, "image-noise + CENTROID path, deep")
