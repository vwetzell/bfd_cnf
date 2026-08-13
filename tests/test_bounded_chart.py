"""The point-source ceiling is a hard support boundary of the flow's chart.

One check per property that could silently break: the chart round-trips, its
analytic log-det matches autodiff, its inverse cannot produce a point at or
above the ceiling however extreme the latent coordinate, and a training set
that violates the bound fails loudly instead of returning NaNs.
"""
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

sys.path.insert(0, ".")

import bulk                                          # noqa: E402
from models.bijections import POINT_SOURCE, RawMomentStandardize, in_domain  # noqa: E402


def sample_moments(n=64, seed=0):
    """Plausible in-domain raw moments, spanning most of the Mr/Mf range."""
    rng = np.random.default_rng(seed)
    Mf = 10 ** rng.uniform(3.0, 4.6, n)
    Mr = Mf * rng.uniform(1.8, 0.995 * POINT_SOURCE, n)
    Mc = Mr * rng.uniform(0.3, 0.9, n)
    e = rng.normal(0, 0.15, (n, 2))
    return jnp.asarray(np.stack([Mf, Mr, e[:, 0] * Mr, e[:, 1] * Mr, Mc], -1))


def test_round_trip_and_log_det():
    b = RawMomentStandardize(mean=jnp.zeros(5), std=jnp.ones(5) * 0.7)
    m = sample_moments()
    z, lad = jax.vmap(b.transform_and_log_det)(m)
    back, lad_inv = jax.vmap(b.inverse_and_log_det)(z)
    # Tolerances are float32's, which is what the flow actually runs in.
    assert np.allclose(np.asarray(back), np.asarray(m), rtol=1e-5)
    assert np.allclose(np.asarray(lad_inv), -np.asarray(lad), atol=1e-4)

    # The analytic log-det is the whole point of the chart being hand-written.
    auto = jax.vmap(lambda x: jnp.linalg.slogdet(
        jax.jacfwd(lambda y: b.transform_and_log_det(y)[0])(x))[1])(m)
    assert np.allclose(np.asarray(lad), np.asarray(auto), atol=1e-3)


def test_support_closes_at_the_ceiling():
    """No latent coordinate, however large, maps to Mr/Mf >= POINT_SOURCE."""
    b = RawMomentStandardize(mean=jnp.zeros(5), std=jnp.ones(5))
    z = jnp.zeros((5, 5)).at[:, 1].set(jnp.array([0.0, 5.0, 10.0, 50.0, 1e4]))
    x, _ = jax.vmap(b.inverse_and_log_det)(z)
    r = np.asarray(x[:, 1] / x[:, 0])
    # Containment is unconditional: never ABOVE the ceiling, at worst exactly on
    # it once float32's sigmoid saturates (|z| ~ 17; float64 would be ~37).  The
    # population's largest galaxy sits at z ~ 2.4, so saturation is ~7 sigma out.
    assert (r <= POINT_SOURCE).all(), r
    assert in_domain(x[:3]).all()


def test_in_domain_rejects_the_ceiling():
    m = sample_moments(8)
    over = m.at[0, 1].set(POINT_SOURCE * m[0, 0] * 1.001)
    at = m.at[1, 1].set(POINT_SOURCE * m[1, 0])
    ok = np.asarray(in_domain(over.at[1].set(at[1])))
    assert not ok[0] and not ok[1] and ok[2:].all()


def test_build_flow_rejects_out_of_domain_training_data():
    m = np.array(sample_moments(32), dtype=np.float64)
    m[3, 1] = POINT_SOURCE * m[3, 0] * 1.01
    with pytest.raises(ValueError, match="point-source ceiling"):
        bulk.build_flow(jr.key(0), m)
