"""
truth.py
========
Exact P(m | g), Q and R for the gauss2 analytic population -- ground truth to
difference a trained flow against.

`imsims.sim.sample_population_gauss2_fwd` FORWARD-samples the galaxy
parameters theta = [log F, log sigma, logit rho, e1, e2] from a chosen box
(`truth.log_p_theta` is its closed form) and renders them.  A galaxy observed
under shear g has that SAME theta, and moments m = analytic.moments(theta, g)
that are a deterministic, invertible function of theta -- so P(m | g) is a
plain change of variables, with NO marginalisation:

    log P(m | g) = log P_theta(theta) - logdet Jg(theta)

where theta solves analytic.moments(theta, g) == m and Jg = d/dtheta
analytic.moments(theta, g).  At g = 0 this is exactly log P0(m) --
`test_log_prob_at_zero_shear_is_log_p0`.

This replaced a P0 that was Gaussian in a moment-space chart, drawn there and
inverted to a galaxy (2026-09-05).  That direction rejects ~80% of draws
because the reachable set is curved, so the realised population was a
TRUNCATED Gaussian -- 0.41 sd on the size axis against the 0.84 it was fitted
to -- and no P0 tuning fixed it.  Forward sampling has no rejection at all.

Solving theta without a g-dependent Newton solve
-------------------------------------------------
`analytic.theta_of_m` only solves at g = 0.  But the co-elliptical family is
CLOSED under shear (`analytic.sheared_theta`), and the reduced-shear matrix
obeys S(-g) = S(g)^-1 EXACTLY -- S(-g) S(g) = I falls out of the 2x2 algebra
in two lines, not an approximation (checked numerically to 1e-17 during
development).  So

    theta' = theta_of_m(m)                 # solves moments(theta', 0) == m
    theta  = sheared_theta(theta', -g)      # solves moments(theta, g)  == m

is CLOSED FORM, no iteration in g at all: theta' is "the galaxy whose own
zero-shear moments are m", and un-shearing it by -g gives the galaxy whose
moments become m only once g has acted on it, which is exactly the theta the
formula above needs.  That has a second payoff besides correctness: because
theta' does not depend on g, differentiating `log_prob` with respect to g
(`pqr`) never has to backprop through `theta_of_m`'s 30-step Newton scan --
with m closed over as a plain (non-g-traced) value, JAX's symbolic-zero
tangent propagation means that scan is evaluated once as an ordinary forward
pass and contributes nothing to the g-VJP.  `jax.lax.custom_root` was the
fallback the spec anticipated for a g-DEPENDENT solve; it turns out not to be
needed, because there is no g-dependent solve left to differentiate through.

A caveat: the theta -> m map folds
-----------------------------------
`theta_of_m` returns SOME preimage of m, and the map is not globally injective
(det dm/dtheta takes both signs across the parameter box).  Measured on a
forward-sampled population, 96.8% of galaxies solve back to the theta they were
generated from and ~1-3% land on a genuine second root.  The moment-space
density P0 is unaffected -- it is chosen, not pushed forward, so `log_p0` and
`log_prob(m, 0)` are exact for every m.  What a branch mismatch perturbs is the
JACOBIAN RATIO at g != 0, which is evaluated at whichever galaxy the solve
finds rather than the one actually observed, and hence `pqr` for those few
targets.  Detect them where it matters: at generation time theta is known, so
`allclose(theta_of_m(moments(theta, g))[0], sheared_theta(theta, g))` flags the
affected galaxies and they can be dropped from a comparison.

No additive constant, and that is not just tidiness
----------------------------------------------------
Every block of `log_p_theta` is normalised, so these are absolute log
densities.  The old rejection-sampled P0 carried an unknown -log Z and argued
it was harmless because Q and R are g-derivatives of a quantity offset by a
g-INDEPENDENT constant.  The premise was wrong in one respect worth recording:
the accepted set was a set of GALAXIES, so its image in moment space moves
with g, i.e. the truncation's support boundary was g-dependent and not
something a constant can absorb.  It was never measured how much that mattered
-- forward sampling removes the question instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
import numpy as np

jax.config.update("jax_enable_x64", True)   # exactness is the whole point

# Mirror the import pattern already used in centroid.py:68-76 verbatim
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bfd_cnf_imsims"))
try:
    from imsims import analytic, sim
except ImportError as exc:                                  # pragma: no cover
    raise SystemExit("truth.py needs ../bfd_cnf_imsims on the path: "
                     f"{exc}") from None

def folded(thetas, g=(0.0, 0.0), atol=1e-6):
    """Mask of galaxies whose moments `theta_of_m` inverts to the WRONG root.

    `theta_of_m` returns SOME preimage and the two-Gaussian moment map is not
    injective (see the module docstring).  Under the old moment-space P0 that
    was nearly harmless -- the density was chosen in m, so any preimage gave
    the same number, and only the g != 0 Jacobian ratio was perturbed.  Under
    the pushforward it is NOT harmless: the density is evaluated AT the galaxy
    the solve finds, and the second root usually lies outside the box, where
    `log_p_theta` is correctly -inf.  Such a target then contributes nothing
    (`jnp.where`'s gradient is zero on the dead branch), which biases any sum
    over the population.

    Measured on `gauss2_fwd`: 2.8% at g = 0, ~4% at |g| = 0.02.  Dropping them
    takes the shear-recovery check from ghat = 0.01959 to 0.020039 against a
    true 0.02, i.e. the density is right and this is the whole residual.

    Needs the GENERATING theta, so it is only available where the population is
    known -- which is every use truth.py has, since it exists to score a
    catalog we rendered.  `sheared_theta(theta, g)` is what the solve should
    return; anything else is a different root.
    """
    g = jnp.asarray(g, dtype=jnp.float64)
    thetas = jnp.asarray(thetas, dtype=jnp.float64)
    want = jax.vmap(lambda t: analytic.sheared_theta(t, g))(thetas)
    ms = jax.vmap(lambda t: analytic.moments(t, g))(thetas)
    got, _residual = analytic.theta_of_m_batch(ms)
    return ~np.isclose(np.asarray(got), np.asarray(want), atol=atol).all(axis=1)


def log_p_theta(theta):
    """log P(theta) for the `gauss2_fwd` box, EXACTLY.

    `theta` is `analytic`'s galaxy vector [log F, log sigma, logit rho, e1, e2]
    and this is the closed form of `sim.sample_population_gauss2_fwd`, block by
    block.  No normalising constant is dropped -- every block is normalised --
    though `log_prob` still carries no -log Z of its own because there is now
    no rejection to renormalise (see the module docstring).

      * (log F, log sigma): a Gaussian copula of correlation
        `SIZE_FLUX_RHO` over a bounded power-law flux and a log-normal size.
        The copula density is evaluated at x = Phi^-1(F's CDF) and
        y = (log sigma - median) / spread, which IS Phi^-1 of sigma's CDF.
      * logit rho: rho ~ U(a, b), so the logit carries a rho(1-rho) Jacobian.
      * (e1, e2): P(e) ~ e exp(-e^2/2 sigma_e^2) on [0, 1) with a uniform
        orientation, so the JOINT density in the plane drops the radial `e`
        (it is the polar Jacobian) and is flat in angle.

    Outside any block's support the density is zero, returned as -inf rather
    than clipped: a target that could not have been drawn must not be given a
    finite prior, or the estimator silently invents support.
    """
    log_f, log_sigma, logit_rho, e1, e2 = (theta[..., 0], theta[..., 1],
                                           theta[..., 2], theta[..., 3],
                                           theta[..., 4])

    # --- flux x size, coupled by a Gaussian copula -------------------------
    lo, hi = sim.FLUX_RANGE
    b = 1.0 - sim.FLUX_ALPHA
    f = jnp.exp(log_f)
    # p(F) ~ F^-alpha normalised on [lo, hi]; in log F that is F^b / Zf.
    z_f = (hi**b - lo**b) / b
    log_p_logf = b * log_f - jnp.log(z_f)
    u = (f**b - lo**b) / (hi**b - lo**b)
    x = ndtri(jnp.clip(u, 1e-12, 1.0 - 1e-12))

    s = sim.SIZE_LOGSTD
    y = (log_sigma - sim.GAUSS2_FWD_SIZE_LOGMEDIAN) / s
    log_p_logsigma = -0.5 * y * y - jnp.log(s) - 0.5 * jnp.log(2.0 * jnp.pi)

    r = sim.SIZE_FLUX_RHO
    one_m = 1.0 - r * r
    log_c = -(r * r * (x * x + y * y) - 2.0 * r * x * y) / (2.0 * one_m) \
        - 0.5 * jnp.log(one_m)

    # --- bulge/disc size ratio ---------------------------------------------
    a_rho, b_rho = sim.GAUSS2_FWD_RHO_RANGE
    rho = jax.nn.sigmoid(logit_rho)
    log_p_rho = jnp.log(rho) + jnp.log1p(-rho) - jnp.log(b_rho - a_rho)

    # --- ellipticity --------------------------------------------------------
    se = sim.GAUSS2_FWD_ELLIP_SIGMA
    e_sq = e1 * e1 + e2 * e2
    # Z_e = int_0^1 e exp(-e^2/2 se^2) de, so the plane density is
    # exp(-e^2/2 se^2) / (2 pi Z_e).
    z_e = se * se * (1.0 - jnp.exp(-0.5 / (se * se)))
    log_p_e = -0.5 * e_sq / (se * se) - jnp.log(2.0 * jnp.pi * z_e)

    total = log_p_logf + log_p_logsigma + log_c + log_p_rho + log_p_e
    in_support = ((f > lo) & (f < hi) & (rho > a_rho) & (rho < b_rho)
                  & (e_sq < 1.0))
    return jnp.where(in_support, total, -jnp.inf)


def log_p0(m):
    """log P0(m), the LATENT (unsheared) moment density.  `m` is a (5,) target.

    The population is FORWARD-sampled in galaxy parameters, so the moment
    density is a pushforward: solve for the galaxy, evaluate the box density
    there, and divide by |det dm/dtheta|.

    This replaced a Gaussian-in-chart-coordinates P0 on 2026-09-05.  That P0
    was drawn in MOMENT space and inverted, which rejects ~80% of draws and
    leaves the realised population a truncated Gaussian rather than the fitted
    one -- 0.41 sd on the size axis against 0.84 (see
    `sim.sample_population_gauss2_fwd`).  The pushforward also removes the
    unknown `Z` and the g-DEPENDENT support boundary that truncation carried:
    the accepted set was a set of galaxies, so its image in moment space moved
    with g, which is not something a g-independent constant can absorb.
    """
    theta, _residual = analytic.theta_of_m(m)
    j0 = jax.jacfwd(lambda th: analytic.moments(th, jnp.zeros(2)))(theta)
    _, ld0 = jnp.linalg.slogdet(j0)
    return log_p_theta(theta) - ld0


def log_prob(m, g):
    """log P(m | g) for one target, up to -log Z (see the module docstring).

    `m` is (5,), `g` a 2-vector.  One Newton solve, at g = 0 -- see the module
    docstring for why the g-dependent theta does not need its own solve.
    """
    m = jnp.asarray(m, dtype=jnp.float64)
    g = jnp.asarray(g, dtype=jnp.float64)
    zero = jnp.zeros(2)

    theta_g0, _residual = analytic.theta_of_m(m)   # solves moments(., 0) == m
    theta = analytic.sheared_theta(theta_g0, -g)    # the g-shear solution

    jg = jax.jacfwd(lambda th: analytic.moments(th, g))(theta)
    _, ldg = jnp.linalg.slogdet(jg)
    # log_p0(m0) + ld0 - ldg, with log_p0's own solve and its -ld0 cancelled
    # against the +ld0: theta_of_m(m0) IS this theta, so the pushforward
    # collapses to one term and saves the second 30-step Newton solve.
    return log_p_theta(theta) - ldg


def log_prob_batch(ms, gs, batch_size=512):
    """`log_prob` over a leading axis, at flat memory cost.

    `jax.lax.map`, not `vmap` -- mirrors `analytic.theta_of_m_batch`'s reason:
    each call runs a `jacfwd(moments)` inside a 30-step Newton scan, so a plain
    `vmap` over a large catalog holds the whole (n, 5, n_masked)-scale
    intermediate at once.  `gs` broadcasts against `ms`'s leading shape, so a
    single shared g or one g per target both work.
    """
    ms = jnp.asarray(ms, dtype=jnp.float64)
    gs = jnp.broadcast_to(jnp.asarray(gs, dtype=jnp.float64),
                          ms.shape[:-1] + (2,))
    return jax.lax.map(lambda args: log_prob(*args), (ms, gs),
                       batch_size=batch_size)


def pqr(m):
    """Exact (Q, R) at g = 0 for one target: shapes (2,) and (2,2).

    Q = d/dg log P(m|g), R = d2/dg2 log P(m|g) -- what `bias.pqr` estimates by
    Monte Carlo (paper eq. 12-13).  Cheap despite the Hessian: see the module
    docstring on why differentiating through `theta_of_m`'s Newton scan never
    happens here in the first place, so there is no tape to worry about.
    """
    m = jnp.asarray(m, dtype=jnp.float64)
    zero = jnp.zeros(2)
    f = lambda g: log_prob(m, g)
    return jax.grad(f)(zero), jax.hessian(f)(zero)


def pqr_batch(ms, batch_size=512):
    """`pqr` over a leading axis, at flat memory cost (see `log_prob_batch`)."""
    ms = jnp.asarray(ms, dtype=jnp.float64)
    return jax.lax.map(pqr, ms, batch_size=batch_size)
