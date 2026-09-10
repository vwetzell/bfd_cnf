"""The score-function form of BFD's selection terms (`selection_terms_score`).

`F` is a cut on the MEASURED moment and shear acts on the prior, so `F` carries
no `g`:

    P_s(g) = INT P(m|g) F(m) dm
    Q_s = E[F Q],   R_s = E[F (R + Q Q^T)],   m ~ P(.|0)

Differentiating the density rather than the sample is what removes the
`1/sigma^2` from the integrand that gives the pathwise estimator a Hill tail
index of ~0.75 (no finite mean).  These are exactness checks on that identity,
not fit-quality checks, so they use a small untrained flow and run in float64.
"""
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx          # noqa: E402
import jax.numpy as jnp        # noqa: E402
import jax.random as jr        # noqa: E402
import numpy as np             # noqa: E402

import bulk                    # noqa: E402
from bias import selection_terms, selection_terms_score   # noqa: E402

COV = np.diag([50.0 ** 2, 30.0 ** 2, 100.0, 100.0, 100.0])
COV[0, 1] = COV[1, 0] = 0.6 * 50.0 * 30.0

# Everything: F == 1 identically.  `window_prob`'s flux-floor guard only fires
# for a FINITE size window, so an infinite one is allowed an infinite flux cut.
NO_WINDOW = ((-np.inf, np.inf), (-np.inf, np.inf))


def catalog(n=4000, seed=0):
    """A synthetic moment catalog inside the chart's physical support."""
    rng = np.random.default_rng(seed)
    Mf = 10.0 ** rng.normal(3.4, 0.25, n)
    Mr = Mf * rng.uniform(2.0, 3.4, n)
    e1, e2 = rng.normal(0, 0.12, n), rng.normal(0, 0.12, n)
    Mc = Mr * rng.uniform(3.0, 5.5, n)
    return np.stack([Mf, Mr, Mr * e1, Mr * e2, Mc], -1)


def flow_and_draws(n_draw=20000, seed=0):
    m = catalog()
    flow = bulk.build_flow(jr.key(seed), m, shear=True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    z = flow.base_dist.sample(jr.key(seed + 31), (n_draw,)).astype(jnp.float64)
    draw = lambda g, zz: jax.vmap(
        lambda z1: flow.bijection.transform(z1, g))(zz)
    return flow, draw, z


def test_no_window_gives_exactly_zero_derivatives():
    """With F == 1, P_s(g) = 1 for every g, so Q_s and R_s vanish identically.

    That is the normalisation identity E[Q] = 0, E[R + Q Q^T] = 0 -- the same
    Fisher check used elsewhere in this project -- and it is the strongest
    available test of the estimator, because the answer is known exactly rather
    than by comparison with another estimator.
    """
    flow, draw, z = flow_and_draws()
    m = draw(jnp.zeros(2), z)
    ps, qs, rs, _ = selection_terms_score(flow, m, COV, *NO_WINDOW)
    assert np.isclose(ps, 1.0, atol=1e-12), ps
    # Monte Carlo over 20k draws, so these are zero to their sampling error,
    # not to machine precision.  Scale-free tolerance: compare against the
    # per-draw RMS the mean is formed from.
    assert np.abs(qs).max() < 0.05, qs
    assert np.abs(rs).max() < 0.5, rs


def test_agrees_with_the_pathwise_estimator_where_that_one_converges():
    """Same estimand, differentiated on the other side of the integral.

    A flux FLOOR is the regime where the pathwise estimator is well behaved
    (Hill index 2.5-2.9, finite variance), so the two must agree there -- and
    that is the only place a comparison is meaningful.  Against a size cut the
    pathwise value has no finite mean to agree with.
    """
    flow, draw, z = flow_and_draws()
    size, flux = (-np.inf, np.inf), (2000.0, np.inf)
    p_a, q_a, r_a, _ = selection_terms(draw, z, COV, size, flux)
    p_b, q_b, r_b, _ = selection_terms_score(
        flow, draw(jnp.zeros(2), z), COV, size, flux)
    assert np.isclose(p_a, p_b, rtol=1e-10), (p_a, p_b)      # same F, same m
    assert np.allclose(q_a, q_b, atol=0.05), (q_a, q_b)
    # R_s is the noisier of the two; compare against the scale of the terms.
    assert np.allclose(r_a, r_b, atol=0.5 * max(1.0, np.abs(r_a).max())), \
        (r_a, r_b)


def test_partial_window_lies_between_zero_and_the_full_integral():
    """A sanity bound rather than an identity: F in [0, 1] means P_s is a
    probability, and the derivative terms must stay finite for a size cut --
    the case the pathwise estimator cannot do at all."""
    flow, draw, z = flow_and_draws()
    ps, qs, rs, qs_err = selection_terms_score(
        flow, draw(jnp.zeros(2), z), COV, (2.2, 3.0), (2000.0, np.inf))
    assert 0.0 < ps < 1.0, ps
    assert np.all(np.isfinite(qs)) and np.all(np.isfinite(rs))
    assert np.all(np.isfinite(qs_err))


if __name__ == "__main__":
    test_no_window_gives_exactly_zero_derivatives()
    test_agrees_with_the_pathwise_estimator_where_that_one_converges()
    test_partial_window_lies_between_zero_and_the_full_integral()
    print("ok")
