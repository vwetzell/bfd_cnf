"""Does the finite-S Monte-Carlo integral account for the centroid control's +0.006?

Consumes two `check_selfconsistency_noisy.py --save-pqr` runs that differ ONLY
in `--draw-seed`: same targets, same noise realisation, same flow, same S.

Two things come out that nothing else in this investigation can give.

1. **A draw-realisation error bar.**  Every quoted uncertainty in HANDOFF.md is
   a bootstrap over TARGETS, which resamples galaxies and leaves each galaxy's
   draws untouched -- it is structurally blind to the Monte-Carlo error of the
   integral.  |m1_A - m1_B| / sqrt(2) is that missing number.

2. **A bias-corrected m1.**  `pqr_streamed` forms

       qhat = B/A          jhat = -(C/A - (B/A)^2)

   and (B/A)^2 is the square of a Monte-Carlo estimate, so

       E[jhat] = j + Cov_MC(qhat)

   i.e. the observed information comes out systematically HIGH, ghat too small,
   and since m1 = sum qhat / (g sum jhat) - 1 the estimator reads systematically
   LOW by

       dm1 = - sum_i Cov_MC(qhat_i)_11 / sum_i j_i,11

   exactly flat in g, and a pure response-scale error.  (Settled in closed form
   on a Gaussian where P(M|g) is exact: <jhat> tracks j + Var_MC at every M and
   every S.  See `dev/toy_is_bias.py` for the mechanism against exact
   quadrature, `dev/check_var_mc.py` for the term on the real runs.)

   Two independent draw realisations estimate that covariance directly,
   Cov_MC = (1/2) (qhat_A - qhat_B)(qhat_A - qhat_B)^T, per target, unbiased.
   SUBTRACTING it from R gives a corrected m1, which moves m1 UP.

    python dev/check_draw_seed.py dev/pqr_control_deep_dsA.npz \\
                                  dev/pqr_control_deep_dsB.npz

ANSWER (2026-08-17), deep centroid control, S = 32768, alpha 0.5, 20000 targets,
draw seeds 107 and 500, everything else identical:

    m1, seed A                      +0.00522
    m1, seed B                      +0.00450
    DRAW-realisation sigma           0.00051     (1 dof)
    sum Cov_MC(qhat)_11 / sum j_11  +0.00263
    m1, averaged, uncorrected       +0.00486
    m1, Cov_MC SUBTRACTED from R    +0.00751 +/- 0.00158

So the artifact does **NOT** explain the centroid control's bias -- removing it
DEEPENS the bias, from +0.0049 to +0.0075.  The artifact is real and worth
removing on its own account (`bias.pqr_streamed(crossfit=True)`), but the
+0.006 thread stays open.

Three things to keep straight before quoting that.

1. The runs REPRODUCE.  +0.0052 and +0.0045 land inside the historical cluster
   at this operating point (+0.0053, +0.0055, +0.0058, +0.0060, +0.0070), and
   the HUMP matches too (-0.0363, -0.0420 against -0.0364..-0.0407).  The
   "+0.0070" quoted throughout HANDOFF.md as THE anchor is the high member of
   that cluster; the mean of all seven is ~+0.0056.  Quote that instead.
2. The draw-realisation sigma is SMALL, 0.0005 against the target bootstrap's
   0.0017.  That worry is closed: the bootstrap does dominate after all, even
   though it could never have shown so by itself.
3. The correction is NOT a universal fix.  It removes only the `(B/A)^2` term.
   Applied to the moment-space control it gives +0.0057 at S = 8192 and +0.0024
   at 32768 -- away from that path's known zero, and not constant.  So other
   O(1/S) terms of comparable size and opposite sign are present; the honest
   budget is several terms of order 0.003.

A tension the sign fix mostly resolves: the information identity moved only
0.0566 -> 0.0524 between S = 8192 and 32768, which looked incompatible with a
Var_MC changing 4x.  With the right sign it is not -- the identity is
`sum qhat^2 / sum jhat - 1`, and Var_MC inflates numerator and denominator
alike (`sum qhat^2 = sum q^2 + V`, `sum jhat = sum j + V`), so with
`sum q^2 ~ sum j` at small g the two largely cancel and the identity is nearly
blind to this term.  A ~7% residual drop against a predicted ~0.8% remains.
"""
import sys

import numpy as np

NBOOT = 300


def load(path):
    d = np.load(path)
    keep = d["keep"] if "keep" in d.files else np.ones(len(d["plus_q"]), bool)
    return {k: d[k][keep] for k in ("plus_q", "plus_r", "minus_q", "minus_r")}


def m1_of(qp, jp, qm, jm, sel=None):
    if sel is not None:
        qp, jp, qm, jm = qp[sel], jp[sel], qm[sel], jm[sel]
    gp = np.linalg.solve(jp.sum(0), qp.sum(0))
    gm = np.linalg.solve(jm.sum(0), qm.sum(0))
    return gp, gm


def main(pa, pb, g=0.02):
    A, B = load(pa), load(pb)
    n = min(len(A["plus_q"]), len(B["plus_q"]))
    ok = np.ones(n, bool)
    for d in (A, B):
        for k in ("plus_q", "minus_q"):
            ok &= np.isfinite(d[k][:n]).all(1)
        for k in ("plus_r", "minus_r"):
            ok &= np.isfinite(d[k][:n]).reshape(n, -1).all(1)
    print(f"two draw realisations, {int(ok.sum())} of {n} targets finite in both\n")

    # j = -r, the per-target observed information
    jA = {"p": -A["plus_r"][:n][ok], "m": -A["minus_r"][:n][ok]}
    jB = {"p": -B["plus_r"][:n][ok], "m": -B["minus_r"][:n][ok]}
    qA = {"p": A["plus_q"][:n][ok], "m": A["minus_q"][:n][ok]}
    qB = {"p": B["plus_q"][:n][ok], "m": B["minus_q"][:n][ok]}

    def m1(q, j):
        gp = np.linalg.solve(j["p"].sum(0), q["p"].sum(0))
        gm = np.linalg.solve(j["m"].sum(0), q["m"].sum(0))
        return (gp[0] - gm[0]) / (2 * g) - 1

    mA, mB = m1(qA, jA), m1(qB, jB)
    print(f"  m1, draw seed A          = {mA:+.5f}")
    print(f"  m1, draw seed B          = {mB:+.5f}")
    print(f"  DRAW-realisation sigma   = {abs(mA - mB) / np.sqrt(2):.5f}"
          f"   (1 dof; the target bootstrap does NOT contain this)")

    # Cov_MC(qhat) per target, then the predicted upward bias in m1
    dm1 = {}
    cov = {}
    for arm in ("p", "m"):
        d = qA[arm] - qB[arm]
        cov[arm] = 0.5 * np.einsum("ni,nj->nij", d, d)
        dm1[arm] = cov[arm][:, 0, 0].sum() / jA[arm][:, 0, 0].sum()
    pred = 0.5 * (dm1["p"] + dm1["m"])
    print(f"\n  sum Cov_MC(qhat)_11 / sum j_11 = {pred:+.5f}"
          f"   (+g {dm1['p']:+.5f}, -g {dm1['m']:+.5f})")
    print(f"  => predicted finite-S excess in m1 = {pred:+.5f}")

    # Bias-corrected.  j = qhat qhat^T - C/A, and E[qhat qhat^T] carries an
    # extra Cov_MC(qhat), so jhat is too LARGE and must have it SUBTRACTED --
    # verified in closed form (a Gaussian where P(M|g) is known exactly, so
    # <jhat> can be compared with j directly: it tracks j + Var_MC at every S).
    # Too-large R means ghat too small, so the uncorrected m1 is too NEGATIVE
    # and correcting it moves m1 UP.
    qm_ = {a: 0.5 * (qA[a] + qB[a]) for a in ("p", "m")}
    junc = {a: 0.5 * (jA[a] + jB[a]) for a in ("p", "m")}
    jc = {a: junc[a] - cov[a] for a in ("p", "m")}
    mc = m1(qm_, jc)
    print(f"\n  m1, both realisations averaged, UNcorrected = {m1(qm_, junc):+.5f}")
    print(f"  m1, Cov_MC SUBTRACTED from R  (CORRECTED)   = {mc:+.5f}")

    rng = np.random.default_rng(0)
    idx = np.arange(len(qA["p"]))
    boot = np.std([m1({a: qm_[a][s] for a in ("p", "m")},
                      {a: jc[a][s] for a in ("p", "m")})
                   for s in (rng.choice(idx, len(idx)) for _ in range(NBOOT))])
    print(f"  target bootstrap on the corrected value     = +/- {boot:.5f}")
    print("\n  reading: corrected ~ 0 means the +0.006 IS the finite-S "
          "(B/A)^2 term;\n           corrected still ~ +0.006 means it is not, "
          "and the draw sigma\n           above says how seriously to take any "
          "single-run number.")


if __name__ == "__main__":
    main(*sys.argv[1:3])
