"""Self-check for SigmaXCouplingLayer / SigmaXBlockLayer (bijectivity + log-det
consistency).

Run: python tests/test_sigmax_layer.py
Checks (assert-based, no framework):
  1. zero-init layer is the identity.
  2. forward∘inverse = identity (closed-form inverse is correct).
  3. forward log_det = −(inverse log_det) (consistent Jacobian).

SigmaXBlockLayer (flux shift aware of the galaxy's own final ellipticity,
closed-form throughout -- an earlier unconstrained-joint-net + Newton-solve
version caused training loss to explode, see its docstring) additionally
checks:
  4. the exact autodiff (jacfwd+slogdet) log-det matches an independent
     finite-difference Jacobian determinant.
  5. forward-then-inverse round-trips EXACTLY (closed-form algebra, no
     iterative solver -- unlike the Newton version this replaced, there is no
     convergence tolerance to tune).
  6. gradients w.r.t. the coeff nets' weights are finite through the inverse
     (needed since training backprops through log_prob's inverse pass).
  7. round-trip and a positive log-det survive LARGE de-zeroed weights and an
     off-manifold input -- stress-testing exactly the regime (weights moved
     far from zero during training) that broke the unconstrained joint design.
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from bfd_cnf.models.bijections import SigmaXCouplingLayer, SigmaXBlockLayer
import equinox as eqx

jax.config.update("jax_enable_x64", True)

MEAN, STD, EMAX = 12.0, 2.0, 0.05


def _layer(key, zero_init=True):
    lay = SigmaXCouplingLayer(
        key, nn_width=16, nn_depth=2, log_scale_mean=MEAN, log_scale_std=STD, e_max=EMAX
    )
    if zero_init:
        return lay
    # de-zero the last layers so coeffs are non-trivial (random small weights)
    nets = []
    for i, n in enumerate(("net_flux", "net_size", "net_dipquad")):
        sub = getattr(lay, n)
        last = sub.layers[-1]
        k = jr.fold_in(key, i)
        w = 0.3 * jr.normal(k, last.weight.shape)
        sub = eqx.tree_at(lambda m: m.layers[-1].weight, sub, w)
        nets.append(sub)
    lay = eqx.tree_at(
        lambda m: (m.net_flux, m.net_size, m.net_dipquad), lay, tuple(nets)
    )
    return lay


def _cond(log_scale, e1=0.03, e2=0.0):
    return jnp.array([0.0, 0.0, log_scale, e1, e2])  # [g1,g2,log_scale,e1,e2]


def test_roundtrip_and_logdet():
    lay = _layer(jr.PRNGKey(0), zero_init=False)
    x = jnp.array([0.4, -0.7, 0.2, -0.1])
    cond = _cond(12.6)
    y, lad_f = lay.transform_and_log_det(x, cond)
    xr, lad_i = lay.inverse_and_log_det(y, cond)
    assert jnp.allclose(x, xr, atol=1e-9), f"roundtrip off: {x} vs {xr}"
    assert jnp.allclose(lad_f, -lad_i, atol=1e-9), f"logdet {lad_f} vs {-lad_i}"


def test_identity_at_zero_init():
    lay = _layer(jr.PRNGKey(2), zero_init=True)
    x = jnp.array([0.4, -0.7, 0.2, -0.1])
    y, lad = lay.transform_and_log_det(x, _cond(12.6))
    assert jnp.allclose(x, y, atol=1e-12), "zero-init layer must be identity"
    assert jnp.allclose(lad, 0.0, atol=1e-12)


def _block_layer(key, zero_init=True, scale=0.3):
    lay = SigmaXBlockLayer(key, nn_width=16, nn_depth=2, log_scale_mean=MEAN, e_max=EMAX)
    if zero_init:
        return lay
    nets = []
    for i, n in enumerate(("net_flux", "net_size", "net_dipquad", "net_flux_e")):
        sub = getattr(lay, n)
        last = sub.layers[-1]
        k = jr.fold_in(key, i)
        w = scale * jr.normal(k, last.weight.shape)
        sub = eqx.tree_at(lambda m: m.layers[-1].weight, sub, w)
        nets.append(sub)
    lay = eqx.tree_at(
        lambda m: (m.net_flux, m.net_size, m.net_dipquad, m.net_flux_e), lay, tuple(nets)
    )
    return lay


def test_block_identity_at_zero_init():
    lay = _block_layer(jr.PRNGKey(2), zero_init=True)
    x = jnp.array([0.4, -0.7, 0.2, -0.1])
    y, lad = lay.transform_and_log_det(x, _cond(12.6))
    assert jnp.allclose(x, y, atol=1e-12), "zero-init block layer must be identity"
    assert jnp.allclose(lad, 0.0, atol=1e-12)


def test_block_roundtrip_and_logdet():
    lay = _block_layer(jr.PRNGKey(0), zero_init=False)
    x = jnp.array([0.4, -0.7, 0.2, -0.1])
    cond = _cond(12.6)
    y, lad_f = lay.transform_and_log_det(x, cond)
    xr, lad_i = lay.inverse_and_log_det(y, cond)
    assert jnp.allclose(x, xr, atol=1e-9), f"block roundtrip off: {x} vs {xr}"
    assert jnp.allclose(lad_f, -lad_i, atol=1e-9), f"block logdet {lad_f} vs {-lad_i}"


def test_block_logdet_matches_finite_difference():
    lay = _block_layer(jr.PRNGKey(1), zero_init=False)
    x = jnp.array([0.4, -0.7, 0.2, -0.1])
    cond = _cond(12.6, e1=0.03, e2=-0.02)
    _, lad = lay.transform_and_log_det(x, cond)

    eps = 1e-5
    jac_fd = jnp.zeros((4, 4))
    for i in range(4):
        dx = jnp.zeros(4).at[i].set(eps)
        yp = lay._raw_transform(x + dx, cond)
        ym = lay._raw_transform(x - dx, cond)
        jac_fd = jac_fd.at[:, i].set((yp - ym) / (2 * eps))
    _, lad_fd = jnp.linalg.slogdet(jac_fd)
    assert jnp.allclose(lad, lad_fd, atol=1e-5), f"autodiff logdet {lad} vs FD {lad_fd}"


def test_block_gradient_through_inverse():
    lay = _block_layer(jr.PRNGKey(3), zero_init=False)
    cond = _cond(12.6)
    y = jnp.array([0.5, -0.6, 0.15, -0.05])

    def loss(lay):
        x, lad = lay.inverse_and_log_det(y, cond)
        return jnp.sum(x**2) + lad

    grad = eqx.filter_grad(loss)(lay)
    g_leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_inexact_array))
    assert g_leaves, "expected at least one gradient leaf"
    assert all(jnp.all(jnp.isfinite(g)) for g in g_leaves), "non-finite gradient through inverse"


def test_block_survives_large_weights_and_outliers():
    """Stress test for exactly the regime that broke the earlier Newton design:
    weights moved far from zero (a real training trajectory, not just a small
    de-zeroing) AND an off-manifold input (a many-sigma outlier moment, e.g. an
    untrained q's garbage draw during ELBO fine-tuning). Round-trip must still
    be exact (it's closed-form algebra, not an iterative solve with a
    convergence radius) and the log-det must stay finite with a sane sign."""
    lay = _block_layer(jr.PRNGKey(4), zero_init=False, scale=3.0)
    xs = [
        jnp.array([0.4, -0.7, 0.2, -0.1]),
        jnp.array([15.0, -12.0, 8.0, -9.0]),   # off-manifold outlier
        jnp.array([-20.0, 20.0, -5.0, 5.0]),
    ]
    conds = [_cond(12.6), _cond(11.9, e1=0.05, e2=-0.04), _cond(13.0, e1=-0.05, e2=0.05)]
    for x in xs:
        for cond in conds:
            y, lad_f = lay.transform_and_log_det(x, cond)
            assert jnp.all(jnp.isfinite(y)) and jnp.isfinite(lad_f), f"non-finite forward at x={x}, cond={cond}"
            xr, lad_i = lay.inverse_and_log_det(y, cond)
            assert jnp.allclose(x, xr, atol=1e-6), f"stress roundtrip off: {x} vs {xr}"
            assert jnp.allclose(lad_f, -lad_i, atol=1e-6), f"stress logdet {lad_f} vs {-lad_i}"


if __name__ == "__main__":
    test_identity_at_zero_init()
    test_roundtrip_and_logdet()
    test_block_identity_at_zero_init()
    test_block_roundtrip_and_logdet()
    test_block_logdet_matches_finite_difference()
    test_block_gradient_through_inverse()
    test_block_survives_large_weights_and_outliers()
    print("OK: identity@init, roundtrip, logdet, FD-logdet, gradient, and large-weight/outlier stress test all pass")
