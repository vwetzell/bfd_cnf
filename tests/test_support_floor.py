"""The physical support indicator and the defensive density floor.

Both are evaluation-time properties of the PRIOR, not of the trained weights,
so they are testable on a freshly built flow.
"""
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

sys.path.insert(0, ".")

import bulk                                                     # noqa: E402
from models.bijections import (POINT_SOURCE, POINT_SOURCE_MC,   # noqa: E402
                               in_domain, in_support, safe_point)
from tests.test_bounded_chart import sample_moments             # noqa: E402


def a_flow(n=256, seed=0):
    return bulk.build_flow(jr.key(seed), np.asarray(sample_moments(n), np.float64))


# --------------------------------------------------------------------------
# 1. the analytic support
# --------------------------------------------------------------------------

def test_in_support_is_the_point_source_region():
    m = sample_moments(128)
    assert np.all(np.asarray(in_support(m)))          # the sampler stays inside

    Mf, Mr, Mc = 1e4, 3.0e4, 1.2e5
    ok = jnp.array([Mf, Mr, 0.1 * Mr, -0.1 * Mr, Mc])
    assert bool(in_support(ok))

    # each constraint, violated one at a time
    past_size = ok.at[1].set(1.001 * POINT_SOURCE * Mf)
    past_conc = ok.at[4].set(1.001 * POINT_SOURCE_MC * Mr)
    assert not bool(in_support(past_size))
    assert not bool(in_support(past_conc))
    assert not bool(in_support(ok.at[0].set(-1.0)))
    assert not bool(in_support(ok.at[1].set(0.0)))


def test_in_support_is_stricter_than_in_domain():
    """The chart must still REPRESENT moments past the ceiling -- that is what
    noisy templates will need -- while the prior is zero there."""
    m = jnp.array([1e4, 1.05 * POINT_SOURCE * 1e4, 1e3, -1e3, 1e5])
    assert bool(in_domain(m)) and not bool(in_support(m))


def test_in_support_never_nans_on_junk():
    junk = jnp.array([[0.0, 1.0, 0.0, 0.0, 1.0],
                      [-1e4, -3e4, 0.0, 0.0, -1e5],
                      [np.inf, 1.0, 0.0, 0.0, 1.0]])
    v = np.asarray(in_support(junk))
    assert v.dtype == bool and not v[0] and not v[1]


def test_safe_point_lands_inside_the_support():
    """`0 * -inf` is NaN, so a masked-out dummy must not be out of support."""
    bad = jnp.array([[0.0, 1e9, 0.0, 0.0, 1e12],
                     [-5.0, -5.0, 1.0, 1.0, -5.0],
                     [1e4, 10.0 * POINT_SOURCE * 1e4, 0.0, 0.0, 1e12]])
    s = safe_point(bad)
    assert np.all(np.asarray(in_support(s))), np.asarray(s)
    assert np.all(np.isfinite(np.asarray(s)))


# --------------------------------------------------------------------------
# 2. the floor
# --------------------------------------------------------------------------

def test_floor_is_off_by_default_construction_and_reproduces_the_flow():
    f = a_flow()
    w = bulk.SupportedFlow(f, eps=0.0, support=False)
    m = jnp.asarray(np.asarray(sample_moments(32), np.float32))
    a = np.asarray(jax.vmap(f.log_prob)(m))
    b = np.asarray(jax.vmap(w.log_prob)(m))
    assert np.allclose(a, b, atol=0, rtol=0), np.abs(a - b).max()


def test_floor_barely_moves_the_density_where_the_flow_is_confident():
    """In-distribution the mixture must sit within eps of the flow.

    Where P_broad << P_flow the mixture is (1-eps) P_flow, i.e. log p drops by
    |log1p(-eps)| ~ eps -- a uniform rescaling, which is exactly the O(eps)
    shift of the estimand the floor is documented to cost.  That is a floor on
    how much the mixture can TAKE AWAY, and it holds for any flow; there is no
    matching upper bound, because adding mass wherever the flow is
    under-confident is the entire point.
    """
    eps = 1e-3
    f = a_flow()
    m = jnp.asarray(np.asarray(sample_moments(64), np.float32))
    lp = np.asarray(jax.vmap(f.log_prob)(m))
    w = bulk.SupportedFlow(f, eps=eps, broad_std=4.0)
    lw = np.asarray(jax.vmap(w.log_prob)(m))
    assert np.all(lw >= lp + np.log1p(-eps) - 1e-4)


def test_floor_is_exactly_the_documented_mixture():
    eps = 1e-2
    f = a_flow()
    w = bulk.SupportedFlow(f, eps=eps, broad_std=4.0)
    m = jnp.asarray(np.asarray(sample_moments(16), np.float32))
    lp = np.asarray(jax.vmap(f.log_prob)(m))
    lb = np.asarray(jax.vmap(w._log_broad)(m))
    want = np.logaddexp(np.log1p(-eps) + lp, np.log(eps) + lb)
    got = np.asarray(jax.vmap(w.log_prob)(m))
    assert np.allclose(got, want, rtol=1e-5, atol=1e-4)


def test_floor_replaces_garbage_with_a_finite_smooth_value():
    """A point the flow answers nonsense for must come back at the floor."""
    f = a_flow()
    chart = bulk.chart_of(f)
    m = np.asarray(sample_moments(1), np.float64)
    z = chart.transform(jnp.asarray(m[0], jnp.float32))
    # Walk out until the raw flow is nonsense but the point is still in support.
    wild = None
    for slot in (0, 3, 4):
        for d in (5.0, 20.0, 50.0, 100.0, 300.0, 1000.0):
            cand = chart.inverse(z.at[slot].add(d))
            if bool(in_support(cand)) and float(f.log_prob(cand)) < -1e4:
                wild = cand
                break
        if wild is not None:
            break
    assert wild is not None, "no in-support point where the raw flow is garbage"

    w = bulk.SupportedFlow(f, eps=1e-3)
    lw = float(w.log_prob(wild))
    lb = float(w._log_broad(wild))
    assert np.isfinite(lw)
    # Where the flow is garbage the mixture IS the floor.
    assert abs(lw - (np.log(1e-3) + lb)) < 1e-3, (lw, lb)


def test_floor_kills_the_gradient_of_a_garbage_score():
    """The point of the floor for Q and R: a nonsense score must contribute
    ~nothing, not nonsense.  d(mixed)/d(lp) = (1-eps) e^lp / mixture -> 0."""
    f = a_flow()
    chart = bulk.chart_of(f)
    m = np.asarray(sample_moments(1), np.float64)
    z = chart.transform(jnp.asarray(m[0], jnp.float32))
    good = chart.inverse(z)
    w = bulk.SupportedFlow(f, eps=1e-3)

    g_raw = jax.grad(lambda x: f.log_prob(x))(good)
    g_flr = jax.grad(lambda x: w.log_prob(x))(good)
    assert np.all(np.isfinite(np.asarray(g_flr)))
    # In-distribution the floor must not disturb the score.
    assert np.allclose(np.asarray(g_raw), np.asarray(g_flr), rtol=1e-3,
                       atol=1e-3 * (1 + np.abs(np.asarray(g_raw)).max()))


def test_support_indicator_zeroes_the_prior_outside():
    f = a_flow()
    w = bulk.SupportedFlow(f, eps=1e-3, support=True)
    out = jnp.array([1e4, 1.05 * POINT_SOURCE * 1e4, 1e3, -1e3, 1e5])
    assert bool(in_domain(out)) and not bool(in_support(out))
    assert float(w.log_prob(out)) == -np.inf
    # ... and with the indicator off it is merely a number again
    w2 = bulk.SupportedFlow(f, eps=1e-3, support=False)
    assert np.isfinite(float(w2.log_prob(out)))


def test_wrapper_forwards_the_rest_of_the_flow():
    f = a_flow()
    w = bulk.SupportedFlow(f, eps=1e-3)
    assert w.bijection is f.bijection
    s = w.sample(jr.key(0), (4,))
    assert np.asarray(s).shape == (4, 5)


def test_wrapper_survives_jit_and_vmap():
    f = a_flow()
    w = bulk.SupportedFlow(f, eps=1e-3)
    m = jnp.asarray(np.asarray(sample_moments(16), np.float32))
    v = jax.jit(jax.vmap(lambda x: w.log_prob(x)))(m)
    assert np.all(np.isfinite(np.asarray(v)))
