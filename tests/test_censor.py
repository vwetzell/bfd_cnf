"""bulk.log_p_unselected (soft selection, pathwise gradient): value and gradient against a closed form."""
import jax, jax.numpy as jnp, jax.random as jr, numpy as np
from flowjax.distributions import Normal
from scipy.stats import norm
import bulk


def test_log_p_unselected_matches_closed_form():
    # 5-d Normal(loc, 1).  Size cut made vacuous: k = 0, Mr ~ N(100, 1), s_u = 1 -> its Phi = 1.
    # Then S(x) = Phi((x0 - c)/s) and E[S] = Phi((mu0 - c)/sqrt(1 + s^2)) exactly.
    c, s = 0.3, 0.5
    cov = jnp.zeros((3, 15)).at[:, 0].set(s * s).at[:, 5].set(1.0)
    f = lambda loc: bulk.log_p_unselected(Normal(loc, jnp.ones(5)), jr.key(0), 400_000, (c, 0.0), jnp.ones(3), cov[0])
    loc = jnp.zeros(5).at[1].set(100.0)
    v = f(loc)
    g = jax.grad(lambda l: f(l)[0])(loc)
    a = (0.0 - c) / np.sqrt(1 + s * s)
    assert np.allclose(np.asarray(v), np.log(norm.sf(a)), atol=3e-3)
    # d/dmu0 log(1 - Phi(a)) = -phi(a) / sqrt(1 + s^2) / (1 - Phi(a))
    assert abs(float(g[0]) - (-norm.pdf(a) / np.sqrt(1 + s * s) / norm.sf(a))) < 1e-2
    assert np.all(np.abs(np.asarray(g[2:])) < 1e-2)


def test_sel_prob_hard_limit():
    m = np.array([[1300.0, 2000.0, 0, 0, 0], [1100.0, 2000.0, 0, 0, 0], [1300.0, 1900.0, 0, 0, 0]])
    cov = np.zeros((3, 15)); cov[:, 0] = 1e-6; cov[:, 5] = 1e-6
    assert np.allclose(np.asarray(bulk.sel_prob(jnp.asarray(m), jnp.asarray(cov), (1200.0, 1.5))),
                       bulk.selected(m, (1200.0, 1.5)).astype(float))
