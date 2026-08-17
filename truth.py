"""
truth.py
========
Exact P(m | g), Q and R for the gauss2 analytic population -- ground truth to
difference a trained flow against.

`imsims.sim.sample_population_gauss2` draws the LATENT (unsheared) moments
from a chosen density P0 (`sim.GAUSS2_MU`, `sim.GAUSS2_COV`, in the chart
`bulk.to_coords` builds -- see `to_coords` below) and solves for the galaxy
realising them (`analytic.theta_of_m`).  A galaxy observed under shear g has
that SAME theta, and moments m = analytic.moments(theta, g) that are a
deterministic, invertible function of theta -- so P(m | g) is a plain change
of variables, with NO marginalisation:

    log P(m | g) = log P0(m0) + logdet J0(theta) - logdet Jg(theta)

where theta solves analytic.moments(theta, g) == m, m0 = analytic.moments
(theta, 0) is the latent (unsheared) moments of that same galaxy, J0 = d/dtheta
analytic.moments(theta, 0) and Jg = d/dtheta analytic.moments(theta, g).  At
g = 0 the two Jacobians are the same function evaluated at the same point, so
this collapses to exactly log P0(m) -- `test_log_prob_at_zero_shear_is_log_p0`.

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

The additive constant
----------------------
`sample_population_gauss2` keeps a drawn galaxy only if the Newton solve
converges and lands in a physical parameter range (about 30% of P0's mass;
see that function's docstring), so the population actually realised is P0
RESTRICTED to that reachable set and renormalised by an unknown constant Z.
Nothing here computes Z.  `log_p0`, `log_prob`, `pqr` all return or
differentiate the density up to an additive -log Z.  That is harmless: Q =
d/dg log P and R = d2/dg2 log P (Bernstein & Armstrong 2014 eq. 12-13) are
g-derivatives of a quantity offset by a g-INDEPENDENT constant, so -log Z
drops out of both exactly, and with it out of any bias built from them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)   # exactness is the whole point

# Mirror the import pattern already used in centroid.py:68-76 verbatim
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bfd_cnf_imsims"))
try:
    from imsims import analytic, sim
except ImportError as exc:                                  # pragma: no cover
    raise SystemExit("truth.py needs ../bfd_cnf_imsims on the path: "
                     f"{exc}") from None

# sim.py only duplicates the slot-1 ceiling (POINT_SOURCE) as its own literal;
# slot 2's has no imsims-side twin to match, so pull it straight from the
# source of truth rather than add one.
from models.bijections import POINT_SOURCE_MC


def to_coords(m):
    """Raw moments -> chart coordinates t = [log10 Mf, logit(Mr/(POINT_SOURCE
    Mf)), logit(Mc/(POINT_SOURCE_MC Mr)), M1/Mr, M2/Mr].

    Duplicates `bulk.to_coords` line for line (ellipsis indexing in place of a
    fixed leading batch axis, so this same function autodiffs a single (5,)
    target under `jacfwd` and still runs on a (n, 5) batch) -- the two must
    NOT drift apart; `tests/test_truth.py::test_to_coords_matches_bulk` is the
    guard.
    """
    u = m[..., 1] / (sim.POINT_SOURCE * m[..., 0])
    v = m[..., 4] / (POINT_SOURCE_MC * m[..., 1])
    return jnp.stack([jnp.log10(m[..., 0]), jnp.log(u) - jnp.log1p(-u),
                      jnp.log(v) - jnp.log1p(-v), m[..., 2] / m[..., 1],
                      m[..., 3] / m[..., 1]], axis=-1)


def log_p0(m):
    """log P0(m), the LATENT (unsheared) moment density, up to -log Z (see the
    module docstring).  `m` is a single (5,) target.

    P0 is Gaussian in the chart `to_coords` builds, so this is that Gaussian's
    log density plus the change-of-variables log|det dt/dm| -- a Cholesky
    solve rather than an explicit inverse of `sim.GAUSS2_COV`, the usual
    numerically stable way to get a quadratic form and a log-det together.
    """
    t = to_coords(m)
    mu = jnp.asarray(sim.GAUSS2_MU)
    cov = jnp.asarray(sim.GAUSS2_COV)
    L = jnp.linalg.cholesky(cov)
    y = jax.scipy.linalg.solve_triangular(L, t - mu, lower=True)
    quad = jnp.dot(y, y)
    log_det_cov = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    log_norm = -0.5 * (quad + log_det_cov + 5.0 * jnp.log(2.0 * jnp.pi))

    jac = jax.jacfwd(to_coords)(m)
    _, logdet_j = jnp.linalg.slogdet(jac)
    return log_norm + logdet_j


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

    m0 = analytic.moments(theta, zero)
    j0 = jax.jacfwd(lambda th: analytic.moments(th, zero))(theta)
    jg = jax.jacfwd(lambda th: analytic.moments(th, g))(theta)
    _, ld0 = jnp.linalg.slogdet(j0)
    _, ldg = jnp.linalg.slogdet(jg)
    return log_p0(m0) + ld0 - ldg


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
