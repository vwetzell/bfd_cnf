"""`concentration_phi` replaces the Gaussian-weight shortcut `Mc/Mr = 2 r` with
the exact real-weight-kernel prediction.  Checks: the fixed 64-point GL
quadrature it uses agrees with an independent, much finer quadrature; the
tau -> 0 limit reproduces `POINT_SOURCE_MC`; the chart still round-trips; and
`bulk.to_coords` (used for plotting) agrees with `RawMomentStandardize`
(used by the flow) on the same inputs.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, ".")

jax.config.update("jax_enable_x64", True)

import bulk                                          # noqa: E402
from models.bijections import (POINT_SOURCE, POINT_SOURCE_MC,  # noqa: E402
                               RawMomentStandardize, concentration_phi)


def sample_moments(n=64, seed=0):
    """Plausible in-domain raw moments, spanning most of the Mr/Mf and
    Mc/Mr ranges (copied from `tests/test_bounded_chart.py`'s helper)."""
    rng = np.random.default_rng(seed)
    Mf = 10 ** rng.uniform(3.0, 4.6, n)
    Mr = Mf * rng.uniform(1.8, 0.995 * POINT_SOURCE, n)
    Mc = Mr * rng.uniform(0.1, 0.995 * POINT_SOURCE_MC, n)
    e = rng.normal(0, 0.15, (n, 2))
    return jnp.asarray(np.stack([Mf, Mr, e[:, 0] * Mr, e[:, 1] * Mr, Mc], -1))


def _phi_reference(r):
    """Independent ground truth: a much finer, DIFFERENT quadrature (plain
    trapezoid on a fixed fine grid), not reusing the 64-node GL constants."""
    c = np.array([0.349792, 0.487396, 0.150208, 0.012604])
    sigma_w = 0.65
    kmax = 1.07635 * np.pi / sigma_w
    kr = np.linspace(0.0, kmax, 20001)
    u = kr * np.pi / kmax
    w = c[0] + c[1] * np.cos(u) + c[2] * np.cos(2 * u) + c[3] * np.cos(3 * u)
    coef = 2 * np.pi * kr * w  # trapz handles its own weights

    def log_A(tau):
        return np.log(np.trapezoid(coef * np.exp(-0.5 * kr ** 2 * tau), kr))

    def Lp(tau, h=1e-6):
        return (log_A(tau + h) - log_A(tau - h)) / (2 * h)

    def Lpp(tau, h=1e-4):
        return (log_A(tau + h) - 2 * log_A(tau) + log_A(tau - h)) / h ** 2

    def tau_of_r(r, iters=60):
        tau = 2.0 / r
        for _ in range(iters):
            f = -2.0 * Lp(tau) - r
            fp = -2.0 * Lpp(tau)
            tau = max(tau - f / fp, 1e-8)
        return tau

    tau = tau_of_r(r)
    return r ** 2 + 4.0 * Lpp(tau)


@pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 2.75, 3.3, 3.6])
def test_quadrature_matches_independent_reference(r):
    got = float(concentration_phi(jnp.asarray(r, dtype=jnp.float64)))
    ref = _phi_reference(r)
    assert abs(got / ref - 1.0) < 1e-6, (got, ref)


def test_point_source_limit_matches_POINT_SOURCE_MC():
    got = float(concentration_phi(jnp.asarray(POINT_SOURCE, dtype=jnp.float64))) / POINT_SOURCE
    assert abs(got / POINT_SOURCE_MC - 1.0) < 1e-3, got


def test_round_trip():
    b = RawMomentStandardize(mean=jnp.zeros(5), std=jnp.ones(5) * 0.7)
    m = sample_moments(1000)
    z, _ = jax.vmap(b.transform_and_log_det)(m)
    back, _ = jax.vmap(b.inverse_and_log_det)(z)
    assert np.allclose(np.asarray(back), np.asarray(m), rtol=1e-6)


def test_to_coords_matches_forward_transform():
    b = RawMomentStandardize(mean=jnp.zeros(5), std=jnp.ones(5))
    m = sample_moments(1000)
    z_flow = np.asarray(jax.vmap(lambda x: b._forward_transform(x)[1])(m))
    z_bulk = bulk.to_coords(np.asarray(m, dtype=np.float64))
    assert np.allclose(z_flow, z_bulk, rtol=1e-5, atol=1e-5)
