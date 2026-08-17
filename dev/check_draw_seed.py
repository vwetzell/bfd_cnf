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

       E[jhat] = j - Cov_MC(qhat)

   i.e. the observed information comes out systematically LOW, and since
   m1 = sum qhat / (g sum jhat) - 1 the estimator reads systematically HIGH by

       dm1 = sum_i Cov_MC(qhat_i)_11 / sum_i j_i,11

   which is positive, exactly flat in g, and a pure response-scale error --
   the fingerprint of the unexplained +0.006 (see `dev/toy_is_bias.py` for the
   mechanism validated against exact quadrature, and `dev/check_var_mc.py` for
   the same term measured on the real runs from two S values).

   Two independent draw realisations estimate that covariance directly,
   Cov_MC = (1/2) (qhat_A - qhat_B)(qhat_A - qhat_B)^T, per target, unbiased.
   Adding it back gives a corrected m1.  If the mechanism is the answer the
   corrected value lands on zero; if it does not, this is the number that says
   how much of the +0.006 is still unexplained.

    python dev/check_draw_seed.py dev/pqr_control_deep_dsA.npz \\
                                  dev/pqr_control_deep_dsB.npz
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

    # bias-corrected: add the missing information back, averaging the two runs'
    # qhat so the corrected estimate also uses 2S worth of draws in the numerator
    qm_ = {a: 0.5 * (qA[a] + qB[a]) for a in ("p", "m")}
    jc = {a: 0.5 * (jA[a] + jB[a]) + cov[a] for a in ("p", "m")}
    mc = m1(qm_, jc)
    print(f"\n  m1, both realisations averaged, UNcorrected = "
          f"{m1(qm_, {a: 0.5*(jA[a]+jB[a]) for a in ('p','m')}):+.5f}")
    print(f"  m1, Cov_MC added back to R  (CORRECTED)     = {mc:+.5f}")

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
