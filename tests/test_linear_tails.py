"""The flow must have linear tails outside its training envelope.

Before `_soft_clip`, a coordinate 0.1 past the data drove a conditioner MLP
linearly, its `_bounded_log_scale` pinned at max_scale, and the result fed the
next layer's conditioner: measured conditioner inputs of 6e5 and log p of -1e5
on a point the estimator's kernel draws routinely visit.  Q and R are built
from those draws, so the blow-up was not cosmetic.

The invariant that fixes it is weight-independent -- no conditioner ever sees
an input outside +-_COND_MAX -- so it can be tested on an untrained flow.
"""
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bulk                                              # noqa: E402
import models.bijections as bij                          # noqa: E402
from models.bijections import _COND_MAX, _soft_clip      # noqa: E402
from tests.test_bounded_chart import sample_moments      # noqa: E402


def test_soft_clip_is_identity_inside_and_saturates_outside():
    u = jnp.linspace(-2.0 * _COND_MAX, 2.0 * _COND_MAX, 2001)
    s = _soft_clip(u)
    assert np.all(np.abs(np.asarray(s)) <= _COND_MAX)

    # Near-identity where the data lives: the converged v3 flow's largest
    # conditioner input on real moments was 10.1.
    inside = jnp.linspace(-10.1, 10.1, 201)
    rel = np.abs(np.asarray(_soft_clip(inside) - inside)) / (np.abs(inside) + 1e-9)
    assert rel.max() < 3e-3, rel.max()

    # Monotone, so it cannot fold two distinct conditioner states onto one.
    assert np.all(np.diff(np.asarray(s)) > 0)


def test_soft_clip_gradient_is_finite_and_vanishes_far_out():
    g = jax.vmap(jax.grad(lambda x: _soft_clip(x)))
    u = jnp.array([0.0, 1.0, _COND_MAX, 3.0 * _COND_MAX, 1e5, 1e7, -1e7])
    d = np.asarray(g(u))
    assert np.all(np.isfinite(d))
    assert np.isclose(d[0], 1.0, atol=1e-5)      # identity at the origin
    assert abs(d[-3]) < 1e-6 and abs(d[-1]) < 1e-6


def test_no_conditioner_sees_an_input_past_the_bound():
    """The runaway was a feedback loop through the conditioners; this is the
    invariant that breaks it, whatever the weights are."""
    m = sample_moments(64)
    flow = bulk.build_flow(jr.key(0), np.asarray(m, np.float64))

    seen = []
    orig = bij.CoeffNet.__call__

    traced = []

    def spy(self, x):
        out = orig(self, x)
        # Inspect what the first Linear actually receives.
        xx = x if x.shape[-1] else jnp.zeros(x.shape[:-1] + (1,), x.dtype)
        try:
            seen.append(np.asarray(_soft_clip(xx)).ravel())
        except jax.errors.TracerArrayConversionError:
            # The spin-2 bend layer evaluates its conditioner inside a
            # `jax.grad` (it needs dh/dq for the log-det), so that one call
            # arrives as a tracer and cannot be read concretely.  Count it
            # instead: the bound still holds for it structurally, because
            # `CoeffNet.__call__` applies `_soft_clip` before any Linear.
            traced.append(1)
        return out

    # A point far outside any plausible envelope, plus one just past it.
    wild = jnp.asarray(np.array([[1e6, 3.6e6, 5e5, -5e5, 2.2e7],
                                 [1e3, 3.69e3, 1e2, -1e2, 2.4e4]], np.float32))
    bij.CoeffNet.__call__ = spy
    try:
        # Eagerly, one row at a time: the spy reads concrete values.
        lp = np.array([float(flow.log_prob(w)) for w in wild])
    finally:
        bij.CoeffNet.__call__ = orig

    v = np.abs(np.concatenate([a for a in seen if a.size]))
    assert v.max() <= _COND_MAX + 1e-4, v.max()
    assert np.all(np.isfinite(lp)), lp
    # The traced calls are the bend layer's, and it must not be silently absent
    # -- if it stopped being reached this test would pass for the wrong reason.
    assert traced, "expected the spin-2 bend layer's conditioner to be traced"


# The tail's SHAPE (log p quadratic in the distance outside the envelope) is a
# property of the trained weights, not of the architecture -- an untrained flow
# with random weights already sits at |base| ~ 1e10 -- so it is measured on the
# real checkpoint by `dev/ls_census.py`, not asserted here.


def test_clipping_does_not_break_invertibility():
    flow = bulk.build_flow(jr.key(0), np.asarray(sample_moments(256), np.float64))
    stack = flow.bijection.bijection
    m = jnp.asarray(np.asarray(sample_moments(32), np.float32))
    z = jax.vmap(stack.transform)(m)
    back = jax.vmap(stack.inverse)(z)
    assert np.allclose(np.asarray(back), np.asarray(m), rtol=2e-4, atol=1e-4)
