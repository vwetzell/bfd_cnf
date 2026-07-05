"""Self-check for SigmaXCouplingLayer (bijectivity + log-det consistency).

Run: python tests/test_sigmax_layer.py
Checks (assert-based, no framework):
  1. zero-init layer is the identity.
  2. forward∘inverse = identity (closed-form inverse is correct).
  3. forward log_det = −(inverse log_det) (consistent Jacobian).
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from bfd_cnf.models.bijections import SigmaXCouplingLayer
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
    for i, n in enumerate(("net_flux", "net_size", "net_dip", "net_quad")):
        sub = getattr(lay, n)
        last = sub.layers[-1]
        k = jr.fold_in(key, i)
        w = 0.3 * jr.normal(k, last.weight.shape)
        sub = eqx.tree_at(lambda m: m.layers[-1].weight, sub, w)
        nets.append(sub)
    lay = eqx.tree_at(
        lambda m: (m.net_flux, m.net_size, m.net_dip, m.net_quad), lay, tuple(nets)
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


if __name__ == "__main__":
    test_identity_at_zero_init()
    test_roundtrip_and_logdet()
    print("OK: identity@init, roundtrip, logdet all pass")
