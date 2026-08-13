"""Self-checks for bulk.py's Hyvarinen score-matching term.

Run: python -m pytest tests/test_bulk_score_match.py

Covers:
  1. `_hyvarinen` is the right quantity: on an isotropic Gaussian model scored
     against a mismatched isotropic Gaussian, the objective has a closed form,
     and it is minimised at the true sigma.
  2. `z_of` peels exactly the chart: `flow.log_prob(m)` decomposes into
     `z_of(flow).log_prob(z) + logdet` with `(z, logdet) =
     chart.transform_and_log_det(m)`.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flowjax.distributions import MultivariateNormal

import bulk
from bulk import _hyvarinen, z_of


def test_hyvarinen_matches_closed_form_and_is_minimised_at_truth():
    """N(0, sigma^2 I) in d=5, scored against data drawn from N(0, I).

    The closed form d/(2 sigma^4) - d/sigma^2 follows from grad log p(z) =
    -z/sigma^2 (so 0.5|grad|^2 has expectation 0.5 d/sigma^4 under E[|z|^2]=d)
    and tr grad^2 log p = -d/sigma^2, constant.
    """
    d = 5
    key = jr.key(0)
    z = jr.normal(key, (20000, d))

    vals = {}
    for sigma in (0.8, 1.0, 1.2):
        model = MultivariateNormal(jnp.zeros(d), sigma**2 * jnp.eye(d))
        got = float(_hyvarinen(model.log_prob, z))
        want = d / (2 * sigma**4) - d / sigma**2
        assert abs(got - want) < 0.05, (sigma, got, want)
        vals[sigma] = got

    assert vals[1.0] < vals[0.8] and vals[1.0] < vals[1.2], vals

    # A SUBSET of the coordinates is the objective for those partials alone,
    # which is how `bulk.SCORE_COORDS` is allowed to keep just the resolution
    # direction: k coordinates of the isotropic model give k times one term.
    model = MultivariateNormal(jnp.zeros(d), 1.2**2 * jnp.eye(d))
    for coords in ((1,), (0, 1)):
        got = float(_hyvarinen(model.log_prob, z, coords))
        want = len(coords) * (0.5 / 1.2**4 - 1 / 1.2**2)
        assert abs(got - want) < 0.05, (coords, got, want)


def _synthetic_moments(n, seed):
    """Physical raw moments [Mf, Mr, M1, M2, Mc], clear of both chart edges."""
    rng = np.random.default_rng(seed)
    Mf = 10 ** rng.uniform(3.0, 4.0, n)
    Mr = Mf * rng.uniform(2.0, 3.5, n)
    Mc = Mr * rng.uniform(0.8, 1.2, n)
    e = rng.normal(0, 0.05, (n, 2))
    M1, M2 = e[:, 0] * Mr, e[:, 1] * Mr
    return np.stack([Mf, Mr, M1, M2, Mc], axis=-1)


def test_z_of_peels_exactly_the_chart():
    m_all = _synthetic_moments(2100, seed=1)
    m_train, m_val = m_all[:2000], m_all[2000:]

    flow = bulk.build_flow(jr.key(0), m_train, layers=2)
    chart = flow.bijection.bijection.bijections[0]

    m_val = jnp.asarray(m_val)
    z, logdet = jax.vmap(chart.transform_and_log_det)(m_val)

    full = flow.log_prob(m_val)
    peeled = z_of(flow).log_prob(z) + logdet
    np.testing.assert_allclose(np.asarray(full), np.asarray(peeled), rtol=1e-4)


if __name__ == "__main__":
    test_hyvarinen_matches_closed_form_and_is_minimised_at_truth()
    test_z_of_peels_exactly_the_chart()
    print("OK: bulk score-match self-checks passed")
