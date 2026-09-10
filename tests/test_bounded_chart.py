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
from models.bijections import (POINT_SOURCE, POINT_SOURCE_MC,  # noqa: E402
                               RawMomentStandardize, in_domain)


def sample_moments(n=64, seed=0):
    """Plausible in-domain raw moments, spanning most of the Mr/Mf and
    Mc/Mr ranges."""
    rng = np.random.default_rng(seed)
    Mf = 10 ** rng.uniform(3.0, 4.6, n)
    Mr = Mf * rng.uniform(1.8, 0.995 * POINT_SOURCE, n)
    Mc = Mr * rng.uniform(0.1, 0.995 * POINT_SOURCE_MC, n)
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


def test_chart_admits_moments_past_the_point_source_limits():
    """The chart must represent Mr/Mf and Mc/Mr ABOVE the point-source values.

    The bound is a property of the LATENT moment, and every measurement of it
    is noisy: target moments already exceed it (4.65% of the v3 g=0 arm in
    Mr/Mf, 4.33% in Mc/Mr), and templates will too once they carry image
    noise.  A chart that sends those to +/-inf cannot be trained or evaluated
    on them, so slots 1 and 2 are now the bare ratios.
    """
    rng = np.random.default_rng(1)
    n = 200
    Mf = 10 ** rng.uniform(3.0, 4.6, n)
    Mr = Mf * rng.uniform(1.5, 2.0 * POINT_SOURCE, n)        # spans past it
    Mc = Mr * rng.uniform(0.1, 2.0 * POINT_SOURCE_MC, n)     # and past this
    e = rng.normal(0, 0.15, (n, 2))
    m = jnp.asarray(np.stack([Mf, Mr, e[:, 0] * Mr, e[:, 1] * Mr, Mc], -1))
    assert (np.asarray(Mr / Mf) > POINT_SOURCE).mean() > 0.3
    assert (np.asarray(Mc / Mr) > POINT_SOURCE_MC).mean() > 0.3

    b = RawMomentStandardize(mean=jnp.zeros(5), std=jnp.ones(5) * 0.7)
    z, lad = jax.vmap(b.transform_and_log_det)(m)
    assert bool(jnp.isfinite(z).all()), "chart is finite past the old ceilings"
    assert bool(jnp.isfinite(lad).all())
    back = jax.vmap(b.inverse)(z)
    assert np.allclose(np.asarray(back), np.asarray(m), rtol=1e-5)
    assert bool(in_domain(m).all()), "in_domain must not reimpose the ceiling"


def test_in_domain_is_only_positivity():
    """Mf > 0 and Mr > 0 are all the chart needs: Mr/Mf and Mc/Mr are plain
    ratios, and Mr is the only denominator. Mc may be any sign."""
    m = sample_moments(16)
    assert bool(in_domain(m).all())
    assert not bool(in_domain(m.at[3, 0].set(-1.0))[3])
    assert not bool(in_domain(m.at[4, 1].set(0.0))[4])
    # a NEGATIVE Mc is representable -- noise can produce one
    neg = m.at[5, 4].set(-abs(float(m[5, 4])))
    assert bool(in_domain(neg)[5])
    assert bool(jnp.isfinite(
        RawMomentStandardize(mean=jnp.zeros(5),
                             std=jnp.ones(5)).transform(neg[5])).all())


def test_build_flow_accepts_training_data_past_the_ceiling():
    """Training data above the point-source value must be trainable, because
    noisy templates will land there."""
    m = np.array(sample_moments(32), dtype=np.float64)
    m[3, 1] = POINT_SOURCE * m[3, 0] * 1.05
    flow = bulk.build_flow(jr.key(0), m)
    lp = flow.log_prob(jnp.asarray(m[3], jnp.float32))
    assert bool(jnp.isfinite(lp)), lp


def test_point_source_constants_match_the_weight_function():
    """POINT_SOURCE and POINT_SOURCE_MC recomputed from bfd's actual weight.

    Both are properties of the weight function -- a point source has
    Itilde/T = 1, so its moments ARE the pure weight moments and the ceilings
    are sum(W k^2)/sum(W) and sum(W k^4)/sum(W k^2).  Their own comments say
    they must be recomputed whenever the weight changes, and record that
    POINT_SOURCE was once silently stale by 1.7% -- which put the real support
    boundary at a finite logit and handed the flow an interior cliff to learn.

    `set_k` demands both axes reach kmax, so this needs a genuine 2-D grid.
    """
    from bfd.weightfunction import KBlackmanHarris

    # imsims is a sibling checkout and is not installed; the suite only found it
    # before because tests/test_truth.py imports `truth`, which puts it on the
    # path as a side effect.  Do it here so this test stands on its own.
    sys.path.insert(0, "../bfd_cnf_imsims")
    from imsims import sim

    w = KBlackmanHarris(weightSigma=sim.WEIGHT_SIGMA)
    ax = np.linspace(-1.01 * w.kmax, 1.01 * w.kmax, 1501)
    kx, ky = np.meshgrid(ax, ax, indexing="ij")
    w.set_k(kx, ky)
    kr2 = (kx ** 2 + ky ** 2).ravel()
    W = np.zeros(kr2.shape)
    W[np.asarray(w.mask_flat)] = np.asarray(w.w_mask)

    ps = (W * kr2).sum() / W.sum()
    ps_mc = (W * kr2 ** 2).sum() / (W * kr2).sum()
    assert abs(ps / POINT_SOURCE - 1) < 1e-4, (ps, POINT_SOURCE)
    assert abs(ps_mc / POINT_SOURCE_MC - 1) < 1e-4, (ps_mc, POINT_SOURCE_MC)
