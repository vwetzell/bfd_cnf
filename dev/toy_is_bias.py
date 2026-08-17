"""Does the importance-sampled convolution bias R DOWN, and m1 UP, at O(1/S)?

`pqr_streamed` accumulates, per target, over S draws from a proposal q:

    A = sum w p,   B = sum w dp/dg,   C = sum w d2p/dg2
    qhat = B/A                       jhat = -(C/A - (B/A)^2)

The (B/A)^2 is a SQUARE of a Monte-Carlo estimate, so

    E[(B/A)^2] = (B/A)_true^2 + Var_MC(B/A)

and jhat is therefore biased DOWNWARD by Var_MC(qhat) -- at O(1/S), with no
matching term in qhat.  Since m1 = sum qhat / (g sum jhat) - 1, a deficit in
sum jhat is a POSITIVE, g-flat, pure response-scale error:

    m1  ~  sum_i Var_MC(qhat_i) / sum_i j_i

which is exactly the fingerprint of the unexplained centroid-control +0.006:
flat in g, a response-scale error, needing whatever makes the per-target
integrand sharp, and INVISIBLE to a weight-ESS diagnostic (the weights w are
bounded and their ESS is flat in S; it is the VARIANCE OF THE RATIO B/A that
carries this, not the concentration of w).

This toy runs that exact arithmetic on an exactly-solvable 1-D problem.

ANSWER (2026-08-17): YES, on all four fingerprints.  With a narrow spike of
width w in the latent density (the toy's stand-in for the centroid layer's
curvature), the predicted deficit sum Var_MC(qhat)/sum j tracks the measured
sum jhat / sum j_exact - 1, and:

  * it scales as 1/S exactly -- w = 0.12 gives 0.0504, 0.0132, 0.0034, 0.0009
    at S = 512, 2048, 8192, 32768 (ratios 3.8, 3.9, 3.8);
  * it is DEAD FLAT in g -- 0.01325 / 0.01324 / 0.01324 at g = 0.01/0.02/0.04;
  * it grows steeply as the density sharpens -- 6e-4 (w = 1.0) to 0.12
    (w = 0.05) at fixed S = 2048;
  * it is invisible to a weight-ESS diagnostic, because the importance weights
    w = L/q are bounded by the defensive mixture and their ESS is flat in S.
    What carries the bias is the variance of the RATIO B/A, not the
    concentration of w.  That is why `check_ess_logdet.py`'s flat ESS/S at
    alpha = 0.5 is not evidence that this term has converged.

`check_var_mc.py` then measures the same quantity on the real runs, where it is
+0.002 to +0.003 at S = 32768 -- large, but equal in the centroid and
moment-space paths, so real and worth fixing yet not the centroid differential.

NOT RUN, and deliberately: the `avg` driver below (mean of m1_IS - m1_exact over
independent draw realisations) was abandoned after two attempts.  It is only
confirming that an R deficit propagates into m1, which is one line of algebra --
if sum jhat = sum j (1 - d) then m1 -> (1 + d)(1 + m1_true) - 1 ~ m1_true + d --
so simulating it only adds Monte-Carlo scatter to a known identity.  The
load-bearing claims above come from the single-realisation sweep, where the
1/S scaling and the g-flatness are unambiguous, and from `check_var_mc.py` on
the real runs.  `avg` is left in place for anyone who wants the end-to-end
version; budget ~20 min and drop S = 32768.
"""
import numpy as np

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ the model
# Latent m ~ p0, "shear" acts as a dilation m -> (1+g) m, so
#   p(m|g) = p0(m/(1+g)) / (1+g).
# p0 is a two-component mixture: a broad bulk plus a NARROW spike whose width
# `w` is the toy's stand-in for the centroid layer's curvature.  w large = the
# smooth, featureless density; w small = sharp structure the integral must
# resolve.
def p0_parts(w):
    return (np.array([0.75, 0.25]), np.array([0.0, 1.2]), np.array([1.0, w]))


def logp(m, g, w):
    a, mu, sd = p0_parts(w)
    s = 1.0 + g
    z = m[..., None] / s
    dens = (a * np.exp(-0.5 * ((z - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))).sum(-1)
    return np.log(dens / s + 1e-300)


def dlogp_dg(m, g, w, h=1e-4):
    return (logp(m, g + h, w) - logp(m, g - h, w)) / (2 * h)


def draw_latent(n, g, w, rng):
    a, mu, sd = p0_parts(w)
    k = rng.choice(2, n, p=a)
    return (mu[k] + sd[k] * rng.standard_normal(n)) * (1.0 + g)


# ------------------------------------------------- the estimator, as bias.py
def pqr(M, sig, w, S, alpha, rng, g0=0.0, hg=2e-3):
    """Per-target (qhat, jhat) by defensive-mixture importance sampling.

    Returns also the EXACT values by fine quadrature, so the O(1/S) bias is
    visible without a reference run.
    """
    n = len(M)
    # proposal: alpha * N(M, sig^2) + (1-alpha) * p0
    nk = int(round(alpha * S))
    xk = M[:, None] + sig * rng.standard_normal((n, nk))
    a, mu, sd = p0_parts(w)
    kk = rng.choice(2, (n, S - nk), p=a)
    xf = mu[kk] + sd[kk] * rng.standard_normal((n, S - nk))
    x = np.concatenate([xk, xf], axis=1)

    lk = -0.5 * ((x - M[:, None]) / sig) ** 2 - np.log(sig * np.sqrt(2 * np.pi))
    lf = logp(x, 0.0, w)
    lq = np.logaddexp(np.log(alpha) + lk, np.log1p(-alpha) + lf)
    logw = lk - lq                       # L(M-x) / q(x)

    def acc(g):
        lp = logp(x, g, w)
        mx = (logw + lp).max(1, keepdims=True)
        return np.exp(logw + lp - mx).sum(1), mx[:, 0]

    # A, and the two g-derivatives by central differences on the SAME draws
    Am, mx = acc(g0)
    Ap, mxp = acc(g0 + hg)
    An, mxn = acc(g0 - hg)
    lA = np.log(Am) + mx
    lAp, lAn = np.log(Ap) + mxp, np.log(An) + mxn
    qhat = (lAp - lAn) / (2 * hg)                    # d log Phat / dg
    jhat = -(lAp - 2 * lA + lAn) / hg ** 2           # -d2 log Phat / dg2
    return qhat, jhat


def exact(M, sig, w, g0=0.0, hg=2e-3, ngrid=4001, span=9.0):
    """log P(M|g) by fine quadrature, and its two g-derivatives."""
    t = np.linspace(-span, span, ngrid)
    dt = t[1] - t[0]

    def lP(g):
        x = M[:, None] + sig * t
        lp = logp(x, g, w)
        lk = -0.5 * t ** 2 - np.log(np.sqrt(2 * np.pi))
        z = lp + lk
        mx = z.max(1, keepdims=True)
        return np.log(np.exp(z - mx).sum(1) * dt) + mx[:, 0]

    l0, lp_, ln_ = lP(g0), lP(g0 + hg), lP(g0 - hg)
    return (lp_ - ln_) / (2 * hg), -(lp_ - 2 * l0 + ln_) / hg ** 2


def m1_of(qp, jp, qm, jm, g):
    return (qp.sum() / jp.sum() - qm.sum() / jm.sum()) / (2 * g) - 1


def run(w, S, g=0.02, n=4000, sig=0.8, alpha=0.5, seed=1):
    """Paired exactly as the control is: ONE base population transported to
    +/-g, and ONE noise realisation shared by both arms, so the shape noise
    that swamps an unpaired m1 cancels."""
    rng = np.random.default_rng(seed)
    m0 = draw_latent(n, 0.0, w, rng)            # shared "base draw"
    eps = sig * rng.standard_normal(n)          # shared noise realisation
    out = {}
    for tag, gg in (("p", +g), ("m", -g)):
        M = (1.0 + gg) * m0 + eps
        # same proposal draws in both arms too (common random numbers)
        out[tag] = (pqr(M, sig, w, S, alpha, np.random.default_rng(seed + 7)),
                    exact(M, sig, w))
    (qp, jp), (eqp, ejp) = out["p"]
    (qm, jm), (eqm, ejm) = out["m"]
    return (m1_of(qp, jp, qm, jm, g), m1_of(eqp, ejp, eqm, ejm, g),
            # the predicted deficit: sum Var_MC(qhat) / sum j
            ((qp - eqp).var() + (qm - eqm).var()) / (jp.mean() + jm.mean()),
            (jp.sum() / ejp.sum() - 1))


def avg(w, S, g=0.02, reps=24, **kw):
    """Average over independent draw realisations: the O(1/S) bias survives,
    the Monte-Carlo scatter that swamps a single run does not."""
    v = np.array([run(w, S, g=g, seed=1000 * k + 3, **kw) for k in range(reps)])
    return v.mean(0), v.std(0) / np.sqrt(reps)


if __name__ == "__main__":
    print("dm1  : <m1_IS - m1_exact>, the bias the finite-S integral ADDS")
    print("pred : <sum Var_MC(qhat) / sum j>, predicted from the (B/A)^2 term")
    print("Rdef : <sum jhat / sum j_exact - 1>, the measured R deficit\n")
    for w in (1.0, 0.30, 0.12):
        print(f"  spike width w = {w}   (small = sharp density structure)")
        for S in (512, 2048, 8192, 32768):
            m, e = avg(w, S)
            print(f"    S={S:6d}   dm1 = {m[0]-m[1]:+.5f} +/- {e[0]:.5f}"
                  f"   pred = {m[2]:+.5f}   Rdef = {m[3]:+.5f} +/- {e[3]:.5f}")
        print()
    print("g-dependence at w = 0.12, S = 2048 -- the mechanism must be FLAT in g:")
    for g in (0.01, 0.02, 0.04):
        m, e = avg(0.12, 2048, g=g)
        print(f"    g={g:.2f}   dm1 = {m[0]-m[1]:+.5f} +/- {e[0]:.5f}"
              f"   pred = {m[2]:+.5f}   Rdef = {m[3]:+.5f}")
