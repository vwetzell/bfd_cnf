"""Are the estimator's Qhat and Rhat mutually consistent?

Any correctly normalised density obeys INT P(M|g) dM = 1 for every g, so
differentiating twice under the integral gives the information identity

    E_0[ s s^T ] = -E_0[ h ],     s = dlogP/dg|_0,  h = d2logP/dg2|_0

i.e. exactly the two quantities BFD calls Q/P and R/P.  This is a SECOND moment
condition on the same saved arrays, independent of the one m1 tests:

    m1        = sum q_1 / (g sum j_11) - 1
    identity  = sum q_1^2 / sum j_11 - 1                     (j = -h)

They share the denominator.  If sum j is X% too small both read +X%; if sum q
is too large only m1 moves.  That separates an R-scale error from a Q-scale
error, which is the whole question for a bias that is flat in g.

Targets are drawn at +/-g while s,h are evaluated at g=0, so each arm's
identity is violated at O(g); averaging the two arms cancels that, leaving
O(g^2) ~ 4e-4.  For the self-consistency CONTROLS the flow is truth by
construction, so any residual is the estimator's.  For the real runs the
identity also picks up genuine model error -- that is the point of listing both.

ANSWER (2026-08-17): the violation is NOT a Q-vs-R inconsistency, and reading
it as one is the trap.  Arm-averaging leaves

    sum q^2 / sum j - 1  =  (g^2/2) Var_0(s_1^2 + h_11) / F  +  2 Var_MC(qhat)/sum j

i.e. a FOURTH-moment property of the per-target log-likelihood plus the finite-S
term.  Measured (11 component, g = 0.02):

    centroid control, deep     +0.0566        centroid control, shallow  +0.0514
    moment-space CONTROL       +0.0372        moment-space REAL          +0.0067
    centroid control, jk on    +0.0544        centroid real, S 32768     +0.0264

CORRECTION (same day): the first version of this block put the moment-space REAL
run on the "moment-space control" row and read off a 8.5x ratio.  Control vs
real is the wrong comparison; against the actual moment-space CONTROL the ratio
is 1.4x.

The 22 component is ~0 everywhere, as it must be -- with shear applied in g1
only, the g2 version is a cross-covariance, not a variance.  That the 11/22
split comes out right is what identifies the term.

So the centroid path's per-target log P is ~1.4x more non-Gaussian in g than
the moment-space path's.  This is exactly the regime the BFD paper's eq. (61)
warns about ("alpha is expected to be of order unity UNLESS d log P/dg becomes
large for some targets"), and it is a genuine, previously unmeasured difference
between the two paths.  It is NOT the +0.006 itself: alpha g^2 is excluded at
~10 sigma by the g-scan (HANDOFF 2026-08-14).
"""
import sys

import numpy as np


def report(path, g=0.02):
    d = np.load(path)
    if "plus_q" not in d.files:
        return
    keep = d["keep"] if "keep" in d.files else np.ones(len(d["plus_q"]), bool)
    qp, rp = d["plus_q"][keep], d["plus_r"][keep]
    qm, rm = d["minus_q"][keep], d["minus_r"][keep]
    fin = (np.isfinite(qp).all(1) & np.isfinite(qm).all(1)
           & np.isfinite(rp).reshape(len(rp), -1).all(1)
           & np.isfinite(rm).reshape(len(rm), -1).all(1))
    qp, rp, qm, rm = qp[fin], rp[fin], qm[fin], rm[fin]
    n = len(qp)

    gp = -np.linalg.solve(rp.sum(0), qp.sum(0))
    gm = -np.linalg.solve(rm.sum(0), qm.sum(0))
    m1 = (gp[0] - gm[0]) / (2 * g) - 1
    m2 = (gp[1] - gm[1]) / (2 * g) - 1

    # arm-averaged score outer product vs arm-averaged observed information
    S = 0.5 * (np.einsum("ni,nj->ij", qp, qp) + np.einsum("ni,nj->ij", qm, qm))
    J = 0.5 * (-rp.sum(0) - rm.sum(0))
    print(f"--- {path}   (N = {n})")
    print(f"    m1 = {m1:+.5f}    m2 = {m2:+.5f}")
    for i, lab in ((0, "11"), (1, "22")):
        print(f"    identity {lab}: sum q^2 / sum j - 1 = {S[i,i]/J[i,i] - 1:+.5f}"
              f"      (sum q^2 = {S[i,i]:.6g}, sum j = {J[i,i]:.6g})")
    print(f"    off-diagonal   : sum q1q2 / sum j12    = "
          f"{S[0,1]:.6g} / {J[0,1]:.6g}")
    # how much of each sum lives in the top 0.1% of targets
    for name, v in (("|q|", np.linalg.norm(qp, axis=1)),
                    ("j11", -rp[:, 0, 0])):
        srt = np.sort(v)[::-1]
        tot = srt.sum()
        k = max(1, n // 1000)
        print(f"    {name}: top 0.1% carry {srt[:k].sum()/tot:6.2%} of the sum,"
              f"  max/median = {srt[0]/np.median(v):.3g}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
