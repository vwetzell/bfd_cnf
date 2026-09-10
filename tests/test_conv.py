"""The noise integral, against a case where it can be done in closed form.
`python -m tests.test_conv`.

`bias.log_conv` estimates the paper's eq. (38) convolution

    P(M|g) = INT dm P(m|g) L(M - m),    L = N(., C_M)

by Monte Carlo over L.  Substitute a Gaussian prior for the flow and the
convolution is another Gaussian, with C_M simply added to its covariance -- so
P, Q = dP/dg and R = d2P/dg2 all have exact values to check the estimator, and
its derivatives, against.  Anything that gets the kernel width, the sign of the
offset or the normalisation wrong fails here.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import bias  # noqa: E402
import bulk  # noqa: E402
import jax.random as jr  # noqa: E402
from bias import (  # noqa: E402  (after the x64 flag)
    ess, kernel_draws, log_conv, log_conv_is, mixture_draws, pqr_streamed)
from models.bijections import in_domain  # noqa: E402

# A prior N(mu(g), S0) in raw moment space, with mu linear in g.  mu0 sits far
# from EVERY edge of `in_domain` -- Mf, Mr > 0, Mr/Mf < POINT_SOURCE and
# Mc/Mr < POINT_SOURCE_MC -- so log_conv's domain mask never fires and the
# comparison is against the unmasked integral.  Mr/Mf = 3.0 and Mc/Mr = 5.0
# here, against ceilings of 3.69 and 6.66; the earlier values (Mr/Mf = 3.667,
# Mc/Mr = 6.818) predate the slot-2 ceiling and sat outside it, which silently
# turned every "unmasked" comparison below into a masked one.
MU0 = np.array([6.0e3, 1.8e4, -1.5e2, 1.2e2, 9.0e4])
DMU = np.array([[30.0, 400.0, 4.4e3, 10.0, 2.0e3],       # d mu / d g1
                [-20.0, 250.0, 15.0, 4.1e3, -1.5e3]])    # d mu / d g2
_A = np.diag([80.0, 340.0, 240.0, 240.0, 2900.0])
_A[1, 0], _A[4, 1], _A[3, 2] = 90.0, 700.0, -40.0
S0 = _A @ _A.T                       # prior covariance
# The noise kernel, narrow next to the prior as a real C_M is (the imsims
# targets sit at S/N ~ 50).  Wider kernels give the estimator a heavy-tailed
# weight distribution and the tolerances below would have to grow like 1/ESS.
COV = 0.02 * S0 + np.diag(np.diag(S0)) * 0.01


class GaussPrior:
    """Stand-in for the flow: P(m|g) = N(m; MU0 + DMU.g, S0)."""

    def log_prob(self, x, condition):
        d = x - (MU0 + condition @ DMU)
        return (-0.5 * jnp.einsum("...i,ij,...j->...", d, jnp.linalg.inv(S0), d)
                - 0.5 * jnp.linalg.slogdet(2 * jnp.pi * S0)[1])

    def sample(self, key, sample_shape=(), condition=None):
        """N(MU0 + condition @ DMU, S0) -- flowjax's `sample` convention, so this
        stands in for the flow on the sampling side too, for `mixture_draws`."""
        mu = MU0 + condition @ DMU
        z = jax.random.normal(key, sample_shape + (5,))
        return mu + z @ np.linalg.cholesky(S0).T


class MismatchedPrior:
    """A stand-in PROPOSAL that is genuinely different from `GaussPrior`: its
    mean is shifted by 300 in every raw-moment component and its covariance is
    1.5x wider.  Used to check that `mixture_draws`'s first argument only has
    to be A proposal, not the density being integrated -- see
    `test_mismatched_proposal_is_still_unbiased`."""

    MU = MU0 + 300.0
    S = 1.5 * S0

    def log_prob(self, x, condition):
        d = x - (self.MU + condition @ DMU)
        return (-0.5 * jnp.einsum("...i,ij,...j->...", d, jnp.linalg.inv(self.S), d)
                - 0.5 * jnp.linalg.slogdet(2 * jnp.pi * self.S)[1])

    def sample(self, key, sample_shape=(), condition=None):
        mu = self.MU + condition @ DMU
        z = jax.random.normal(key, sample_shape + (5,))
        return mu + z @ np.linalg.cholesky(self.S).T


class _StubBijection:
    """Self-referencing stand-in for flowjax's `bijection.bijection.bijections`
    chain -- just enough of it for `bias.split_centroid` to find no
    `CentroidMarginalize` at `bijections[0]` and return `(flow, None)`."""

    def __init__(self):
        self.bijection = self
        self.bijections = [object()]


class GaussPriorEval(GaussPrior):
    """`GaussPrior` plus the `.bijection` attribute `split_centroid` needs, so
    it can stand in as the EVAL flow passed to `pqr_streamed` -- see
    `test_pqr_streamed_proposal_draws_from_proposal_evaluates_with_flow`."""

    def __init__(self):
        self.bijection = _StubBijection()


def exact(M, g):
    """log N(M; MU0 + DMU.g, S0 + COV) -- the convolution, done analytically."""
    d = M - (MU0 + g @ DMU)
    S = S0 + COV
    return (-0.5 * d @ np.linalg.solve(S, d)
            - 0.5 * np.linalg.slogdet(2 * np.pi * S)[1])


def _assert_matches_exact_conv(f, M, tol_logp, tol_q=3e-3, tol_r=4e-3):
    """Shared check for `test_convolution_and_its_shear_derivatives`,
    `test_mixture_is_unbiased_against_the_exact_convolution` and
    `test_mismatched_proposal_is_still_unbiased`: logP, Q and R at g = 0 must
    match the closed-form Gaussian convolution, and logP must still match away
    from zero shear, where mu(g) has moved."""
    A = np.linalg.inv(S0 + COV)
    q_true = DMU @ A @ (M - MU0)
    r_true = -DMU @ A @ DMU.T

    zero = jnp.zeros(2)
    logP = float(f(zero))
    q = np.asarray(jax.grad(f)(zero))
    r = np.asarray(jax.hessian(f)(zero))

    assert abs(logP - exact(M, np.zeros(2))) < tol_logp, (logP, exact(M, np.zeros(2)))
    assert np.max(np.abs(q - q_true) / np.abs(q_true)) < tol_q, (q, q_true)
    assert np.max(np.abs(r - r_true) / np.abs(r_true)) < tol_r, (r, r_true)

    g = jnp.array([0.05, -0.03])
    assert abs(float(f(g)) - exact(M, np.asarray(g))) < tol_logp


def test_convolution_and_its_shear_derivatives():
    M = MU0 + np.array([2.0e2, -8.0e2, 3.0e2, -2.0e2, 5.0e3])   # a target
    eps = kernel_draws(COV, 1, 400_000, seed=7)[0]
    f = lambda g: log_conv(GaussPrior(), jnp.asarray(M), jnp.asarray(eps), g)

    # Tolerance is ~2x the Monte Carlo scatter measured over seeds at this S;
    # every structural way of getting the convolution wrong misses by far more.
    _assert_matches_exact_conv(f, M, tol_logp=6e-3)


def test_zero_kernel_is_the_point_evaluation():
    """C_M -> 0 must reduce to evaluating the prior at the target itself."""
    M = MU0 + np.array([1.0e2, 5.0e2, -2.0e2, 1.0e2, -3.0e3])
    eps = kernel_draws(1e-12 * COV, 1, 64, seed=3)[0]
    prior = GaussPrior()
    got = log_conv(prior, jnp.asarray(M), jnp.asarray(eps), jnp.zeros(2))
    want = prior.log_prob(jnp.asarray(M), jnp.zeros(2))
    assert abs(float(got) - float(want)) < 1e-8


def test_out_of_domain_draws_get_zero_weight_not_nan():
    """Draws at Mf <= 0 or Mr <= 0 are off the prior's chart; they must drop out
    of the sum without poisoning the value or the gradient."""
    M = np.array([50.0, 40.0, 5.0, -5.0, 1.0e3])          # right at the edge
    eps = kernel_draws(COV, 1, 4096, seed=11)[0]
    x = M + np.asarray(eps)
    assert ((x[:, 0] <= 0) | (x[:, 1] <= 0)).mean() > 0.2, "test point too safe"

    f = lambda g: log_conv(GaussPrior(), jnp.asarray(M), jnp.asarray(eps), g)
    assert np.isfinite(float(f(jnp.zeros(2))))
    assert np.all(np.isfinite(np.asarray(jax.grad(f)(jnp.zeros(2)))))
    assert np.all(np.isfinite(np.asarray(jax.hessian(f)(jnp.zeros(2)))))

    # And the value is the masked sum: the same draws, restricted by the SAME
    # predicate the estimator uses.  Spelling the conditions out by hand here
    # silently drifted once the chart grew a second ceiling.
    ok = np.asarray(in_domain(jnp.asarray(x)))
    lp = np.asarray(GaussPrior().log_prob(jnp.asarray(x[ok]), jnp.zeros(2)))
    want = jax.scipy.special.logsumexp(lp) - np.log(len(x))
    assert abs(float(f(jnp.zeros(2))) - float(want)) < 1e-9


def test_mixture_is_unbiased_against_the_exact_convolution():
    """`mixture_draws` + `log_conv_is` at alpha = 0.5 must reproduce the SAME
    exact convolution and its shear derivatives as the pure-kernel estimator
    does above.  A wrong mixture normalisation, an alpha bookkeeping slip, or
    an inverted L/q ratio all show up here as a biased (not just noisier)
    logP, Q or R."""
    M = MU0 + np.array([2.0e2, -8.0e2, 3.0e2, -2.0e2, 5.0e3])   # same target
    draws, log_wt = mixture_draws(GaussPrior(), jnp.asarray([M]), COV,
                                  400_000, 0.5, seed=7)
    f = lambda g: log_conv_is(GaussPrior(), jnp.asarray(M), jnp.asarray(draws[0]),
                              jnp.asarray(log_wt[0]), g)

    # Tolerance ~2x the scatter measured over 20 seeds at this S (half the
    # draws going to the flow component costs some precision relative to the
    # all-kernel test above): logP std 1.7e-3 (max 4.6e-3), q rel std 4e-4
    # (max 1.5e-3), r rel std 4e-4 (max 1.6e-3); away from zero, std 1.5e-3,
    # max 3.8e-3.
    _assert_matches_exact_conv(f, M, tol_logp=8e-3)


def test_mismatched_proposal_is_still_unbiased():
    """The mathematical invariant `mixture_draws`'s docstring now states: its
    first argument is the PROPOSAL, which only has to COVER the density being
    integrated, not match it.  Draw from `MismatchedPrior` (a different mean
    and a wider covariance than `GaussPrior`) at alpha = 0.5, then evaluate
    with the ORIGINAL `GaussPrior` via `log_conv_is` -- the result must still
    be the SAME exact convolution as the matched-proposal test above, not
    merely close to it (a mismatched proposal costs ESS, and a systematic
    offset would be a real bias, not noise)."""
    M = MU0 + np.array([2.0e2, -8.0e2, 3.0e2, -2.0e2, 5.0e3])   # same target
    draws, log_wt = mixture_draws(MismatchedPrior(), jnp.asarray([M]), COV,
                                  400_000, 0.5, seed=7)
    f = lambda g: log_conv_is(GaussPrior(), jnp.asarray(M), jnp.asarray(draws[0]),
                              jnp.asarray(log_wt[0]), g)

    # Tolerance ~2x the scatter measured over 20 seeds at this S: logP std
    # 1.7e-3 (max 4.7e-3), q rel std 3.7e-4 (max 1.5e-3), r rel std 4.3e-4
    # (max 1.7e-3) -- close to the matched-proposal test's own numbers, because
    # MismatchedPrior's offset (300) and width (1.5x) are both mild next to S0;
    # away from zero, std 1.5e-3, max 4.0e-3.
    _assert_matches_exact_conv(f, M, tol_logp=1e-2)


def test_pqr_streamed_proposal_draws_from_proposal_evaluates_with_flow():
    """`pqr_streamed(..., proposal=...)` must feed `proposal` to the internal
    `mixture_draws` call for the DRAWS, while everything downstream --
    `split_centroid(flow)` and `log_conv_is` -- keeps evaluating `flow`, the
    density actually being integrated.  `GaussPriorEval` stands in for a flow
    with no centroid layer (`split_centroid` sees no `CentroidMarginalize` and
    returns `(flow, None)`); `MismatchedPrior` supplies the draws.  A single
    chunk (`chunk = samples`) sidesteps the merge entirely, so this is a
    one-target plumbing check, not a merge-accuracy one."""
    M = MU0 + np.array([2.0e2, -8.0e2, 3.0e2, -2.0e2, 5.0e3])
    m = jnp.asarray([M])
    S = 200_000

    A = np.linalg.inv(S0 + COV)
    q_true = DMU @ A @ (M - MU0)
    r_true = -DMU @ A @ DMU.T

    q, r = pqr_streamed(GaussPriorEval(), m, COV, S, 0.5, seed=13, chunk=S,
                        proposal=MismatchedPrior())
    # Tolerances ~2x the scatter measured over 15 seeds at this S: q rel std
    # 4.4e-4 (max 2.0e-3), r rel std 5.8e-4 (max 1.9e-3).
    assert np.max(np.abs(q[0] - q_true) / np.abs(q_true)) < 4e-3, (q[0], q_true)
    assert np.max(np.abs(r[0] - r_true) / np.abs(r_true)) < 4e-3, (r[0], r_true)

    # `proposal=None` must be unchanged: the SAME flow drawn from directly, to
    # machine precision (same seed, same draws, no mismatch to cost ESS).
    q0, r0 = pqr_streamed(GaussPriorEval(), m, COV, S, 0.5, seed=13, chunk=S)
    q1, r1 = pqr_streamed(GaussPriorEval(), m, COV, S, 0.5, seed=13, chunk=S,
                          proposal=None)
    assert np.array_equal(q0, q1) and np.array_equal(r0, r1)


def test_alpha_one_is_the_kernel_estimator():
    """alpha = 1.0 draws nothing from the flow (guard 4 of `mixture_draws`) and
    log_wt = 0 everywhere, so log_conv_is on its output must reproduce
    `log_conv` on the SAME kernel draws to machine precision -- not just to
    Monte Carlo tolerance."""
    M = MU0 + np.array([1.0e2, 5.0e2, -2.0e2, 1.0e2, -3.0e3])
    seed, S = 9, 4096
    eps = kernel_draws(COV, 1, S, seed=seed)[0]

    draws, log_wt = mixture_draws(GaussPrior(), jnp.asarray([M]), COV, S, 1.0, seed)
    assert np.all(log_wt == 0.0), "alpha = 1 must give log_wt identically zero"
    # The draws must be the kernel ones, up to the float32 `mixture_draws`
    # stores them in (its docstring: float64 would be 26 GB per catalog at 32k
    # draws/target).  Moments are ~1e5, so f32 costs ~8e-3 in moment space.
    assert np.abs(np.asarray(draws[0], np.float64)
                  - (np.asarray(M) + eps)).max() < 1e-2

    # Compare like with like.  The claim is that alpha = 1 IS the kernel
    # estimator, not that f32 and f64 round identically: fed the same f32
    # numbers the two agree to 2e-12, while against the f64 eps the shear layer
    # amplifies the storage difference to 5e-9 at g != 0.
    eps32 = np.asarray(draws[0], np.float32) - np.asarray(M, np.float32)
    want = lambda g: log_conv(GaussPrior(), jnp.asarray(M), jnp.asarray(eps32), g)
    got = lambda g: log_conv_is(GaussPrior(), jnp.asarray(M), jnp.asarray(draws[0]),
                                jnp.asarray(log_wt[0]), g)

    for g in (jnp.zeros(2), jnp.array([0.03, -0.01])):
        assert abs(float(want(g)) - float(got(g))) < 1e-10


def test_mixture_rescues_the_starved_kernel():
    """Widen the kernel far past the prior's own width (100x S0, against a
    prior-scale kernel of ~0.02x S0 elsewhere in this file) and the pure-
    kernel estimator starves: draws from N(M_i, C_M) almost never land where
    the prior has mass, so its ESS collapses to a handful even at S = 20000.
    The mixture's flow component covers that mass directly, so its ESS should
    recover by orders of magnitude -- AND the estimate must still be right;
    a high ESS on a biased answer would be worse than no fix at all."""
    M = MU0 + np.array([2.0e2, -8.0e2, 3.0e2, -2.0e2, 5.0e3])
    cov = 100.0 * S0
    seed, S = 5, 20_000

    eps = kernel_draws(cov, 1, S, seed=seed)[0]
    e_kernel = float(ess(GaussPrior(), jnp.asarray([M]), jnp.asarray([M + eps]),
                         jnp.zeros((1, S)))[0])

    draws, log_wt = mixture_draws(GaussPrior(), jnp.asarray([M]), cov, S, 0.5, seed)
    e_mix = float(ess(GaussPrior(), jnp.asarray([M]), jnp.asarray(draws),
                      jnp.asarray(log_wt))[0])

    assert e_kernel < 10, f"test kernel not actually starved: ESS = {e_kernel}"
    assert e_mix > 5 * e_kernel, (e_mix, e_kernel)

    def exact_wide(M, g):
        d = M - (MU0 + g @ DMU)
        S_ = S0 + cov
        return (-0.5 * d @ np.linalg.solve(S_, d)
                - 0.5 * np.linalg.slogdet(2 * np.pi * S_)[1])

    logP = float(log_conv_is(GaussPrior(), jnp.asarray(M), jnp.asarray(draws[0]),
                             jnp.asarray(log_wt[0]), jnp.zeros(2)))
    # Scatter measured over 15 seeds at this (factor, S): std 5e-4, max 1.1e-3.
    assert abs(logP - exact_wide(M, np.zeros(2))) < 3e-3, (logP, exact_wide(M, np.zeros(2)))


def test_gauge_k_is_the_same_convolution():
    """`log_conv_is_kernel` is a change of variables inside ONE integral, not a
    different estimator of a different thing.

    `GaussPrior` is a pure translation, `P(m|g) = N(m; MU0 + g.DMU, S0)`, so its
    pushforward map is exactly `Psi_g(u) = u + g.DMU` and every step of the gauge
    algebra is checkable against a closed form.  Two claims are pinned here:

    1. At g = 0 the two gauges are BIT-IDENTICAL on the same draws -- the
       substitution is the identity there, so any sign slip, missing Jacobian or
       confusion about which coordinate `c` lives in shows up as a nonzero diff.
    2. With a kernel WIDE next to the prior -- the faint-end regime this exists
       for, where `d_g log p` is large and `d_g log L` is not -- gauge K is the
       better-conditioned one, by ~3x on Q and ~9x on R.

    Note the module's own narrow `COV` is the OPPOSITE regime (S/N ~ 50), where
    gauge P wins by 30x.  Neither gauge is universally better conditioned; what
    makes gauge K necessary on real faint targets is that gauge P's `Var[d_g
    log p]` does not exist there at all (Hill index 1.20), which no toy with a
    Gaussian prior can exhibit.
    """
    M = MU0 + np.array([2.0e2, -8.0e2, 3.0e2, -2.0e2, 5.0e3])   # same target
    psi = lambda u, g, sigma_x: u + g @ jnp.asarray(DMU)
    assert np.allclose(np.asarray(psi(jnp.asarray(MU0), jnp.zeros(2), None)),
                       MU0), "Psi must be the identity at g = 0"

    def gauges(cov, seed=7):
        draws, log_wt = mixture_draws(GaussPrior(), jnp.asarray([M]), cov,
                                      400_000, 0.5, seed=seed)
        x, lw = jnp.asarray(draws[0]), jnp.asarray(log_wt[0])
        cinv = jnp.asarray(np.linalg.inv(cov))
        ok = jnp.ones(x.shape[0], dtype=bool)
        c = bias._gauge_k_c(GaussPrior(), x, lw, jnp.asarray(M), None, ok,
                            peeled=False)
        r0 = jnp.asarray(M) - x
        return (lambda g: bias.log_conv_is_kernel(psi, r0, x, x, c, g, None,
                                                  cinv, ok),
                lambda g: log_conv_is(GaussPrior(), jnp.asarray(M), x, lw, g))

    zero = jnp.zeros(2)
    err = lambda got, want: float(np.max(np.abs(np.asarray(got) - want)
                                         / np.abs(want)))

    # (1) identical at g = 0, on this module's own narrow kernel.
    fk, fp = gauges(COV)
    assert float(fk(zero)) == float(fp(zero)), (float(fk(zero)), float(fp(zero)))
    assert abs(float(fk(zero)) - exact(M, np.zeros(2))) < 8e-3

    # (2) the wide-kernel regime, against the closed form.  Tolerances are 2x
    # the max seen over five seeds (K: Q 1.0e-2, R 8.1e-3).
    cov = 3.0 * S0
    fk, fp = gauges(cov)
    A = np.linalg.inv(S0 + cov)
    q_true, r_true = DMU @ A @ (M - MU0), -DMU @ A @ DMU.T
    qk, rk = jax.grad(fk)(zero), jax.hessian(fk)(zero)
    assert err(qk, q_true) < 2.0e-2, (qk, q_true)
    assert err(rk, r_true) < 1.6e-2, (rk, r_true)
    # ...and better conditioned than gauge P on the same draws.
    assert err(rk, r_true) < 0.5 * err(jax.hessian(fp)(zero), r_true)

    # (3) still the same integral away from g = 0.
    g = jnp.array([0.05, -0.03])
    d = M - (MU0 + np.asarray(g) @ DMU)
    S = S0 + cov
    want = -0.5 * d @ np.linalg.solve(S, d) - 0.5 * np.linalg.slogdet(2 * np.pi * S)[1]
    assert abs(float(fk(g)) - want) < 8e-3, (float(fk(g)), want)


def test_psi_is_the_identity_at_zero_shear():
    """`make_psi` builds `Psi_g = Phi^-1 . shear(., g) . Phi`, so at g = 0 it
    must return its input EXACTLY -- it is a round trip through the chart and
    the centroid layer with an identity in between.

    Written after applying `cen.inverse` on the way out without `cen.transform`
    on the way in, which left `chart^-1 . cen^-1 . chart`: still smooth, still
    finite, still the identity in shape, and wrong by a median 0.76% of the
    moment.  Nothing else in the estimator would have caught that -- log P and
    the g = 0 value are unaffected, only Q and R move.

    An untrained flow is enough: the identity is structural, not learned.
    """
    key = jr.key(0)
    m_train = np.asarray(
        jr.uniform(key, (256, 5), minval=jnp.asarray([2e3, 6e3, -5e2, -5e2, 3e4]),
                   maxval=jnp.asarray([9e3, 2.4e4, 5e2, 5e2, 1.2e5])), np.float64)
    # in_domain also bounds Mr/Mf and Mc/Mr; keep only rows the chart accepts.
    m_train = m_train[np.asarray(in_domain(jnp.asarray(m_train)))]
    assert len(m_train) > 32, len(m_train)

    for centroid in (False, True):
        flow = bulk.build_flow(key, m_train, shear=True, centroid=centroid)
        psi = bias.make_psi(flow, peeled=False)
        x = jnp.asarray(m_train[:16], jnp.float32)
        # The catalogs' own Sigma_X (`cov_odd` ~ 1.15e4), not a token value:
        # the centroid layer's shift scales with it, so at a small Sigma_X the
        # layer is near-identity and the round trip passes either way -- the
        # bug this pins was invisible at 4e-3 and is a clean 10x at the real one.
        sigma_x = (jnp.tile(jnp.asarray([1.155e4, 0.0, 1.155e4]), (16, 1))
                   if centroid else None)
        got = jax.vmap(psi, in_axes=(0, None, 0 if centroid else None))(
            x, jnp.zeros(2), sigma_x)
        rel = np.max(np.abs(np.asarray(got) - np.asarray(x)) / np.abs(np.asarray(x)))
        # float32 round trip through chart (+ centroid) leaves 3.3e-4; the
        # dropped `cen.transform` leaves 3.5e-3.
        assert rel < 1e-3, (f"centroid={centroid}", rel)


def test_blend_is_exact_at_every_lambda():
    """`log_conv_is_blend` is a rigid translation `m = u + t(g)` inside eq. (38),
    so it is exact for ANY lambda -- that is what makes lambda free to be chosen
    for variance alone, and it is the one property that must not be traded away.

    Exercised in the WIDE-kernel regime, where every lambda is well conditioned,
    so one tolerance covers all three.  (In the module's own narrow `COV` the
    errors just track the regime: lambda = 0 lands at 1.5e-3 and lambda = 1 at
    4.8e-2 on Q, with no lambda showing a bias that fails to shrink.)  A wrong
    Jacobian, a mis-signed kernel term, or a lambda that leaked into the
    estimand rather than only into its variance are all O(1) here.
    """
    M = MU0 + np.array([2.0e2, -8.0e2, 3.0e2, -2.0e2, 5.0e3])
    # `log_conv_is_blend` takes a `make_psi_ld`-style map: (point, log-det).
    # A rigid translation has unit Jacobian, hence log-det 0.
    psi = lambda u, g, sigma_x: (u + g @ jnp.asarray(DMU), jnp.zeros(()))
    zero = jnp.zeros(2)

    def blend(cov, lam):
        draws, log_wt = mixture_draws(GaussPrior(), jnp.asarray([M]), cov,
                                      400_000, 0.5, seed=7)
        x, lw = jnp.asarray(draws[0]), jnp.asarray(log_wt[0])
        ok = jnp.ones(x.shape[0], dtype=bool)
        return lambda g: bias.log_conv_is_blend(
            GaussPrior(), psi, jnp.asarray(M), x, jnp.asarray(M) - x, lw, g,
            None, jnp.asarray(np.linalg.inv(cov)), lam, ok)

    cov = 3.0 * S0
    A = np.linalg.inv(S0 + cov)
    q_true, r_true = DMU @ A @ (M - MU0), -DMU @ A @ DMU.T
    S = S0 + cov
    # `exact` is hardcoded to the module's own COV, so the wide-kernel closed
    # form has to be written out here.
    logN = lambda g: (lambda d: -0.5 * d @ np.linalg.solve(S, d)
                      - 0.5 * np.linalg.slogdet(2 * np.pi * S)[1])(
                          M - (MU0 + np.asarray(g) @ DMU))
    at_zero, away = logN(np.zeros(2)), logN(np.array([0.05, -0.03]))

    for lam in (0.0, 0.5, 1.0):
        f = blend(cov, lam)
        # 2x the max over five seeds (worst lambda: Q 3.0e-2, R 7.3e-2).
        assert abs(float(f(zero)) - at_zero) < 8e-3, lam
        q, r = jax.grad(f)(zero), jax.hessian(f)(zero)
        assert np.max(np.abs(np.asarray(q) - q_true) / np.abs(q_true)) < 6e-2, (lam, q)
        assert np.max(np.abs(np.asarray(r) - r_true) / np.abs(r_true)) < 1.6e-1, (lam, r)
        # away from zero too, where t(g) has actually moved the draws
        assert abs(float(f(jnp.array([0.05, -0.03]))) - away) < 8e-3, lam

    # lambda = 0 is `log_conv_is` to the bit, on this module's own kernel.
    draws, log_wt = mixture_draws(GaussPrior(), jnp.asarray([M]), COV,
                                  400_000, 0.5, seed=7)
    x, lw = jnp.asarray(draws[0]), jnp.asarray(log_wt[0])
    got = float(bias.log_conv_is_blend(
        GaussPrior(), psi, jnp.asarray(M), x, jnp.asarray(M) - x, lw, zero, None,
        jnp.asarray(np.linalg.inv(COV)), 0.0, jnp.ones(x.shape[0], dtype=bool)))
    assert got == float(log_conv_is(GaussPrior(), jnp.asarray(M), x, lw, zero))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
