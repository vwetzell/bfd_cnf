"""Symmetry self-checks for the shear layer.  `python -m tests.test_shear`.

These assert the three properties the layer's functional form is built on.  They
are properties of the *architecture*, so they hold at any parameter values --
an untrained layer passes.  Whether the learned coefficients match reality is a
separate question, answered by `python shear.py validate` against bfd.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from models.shear import ShearResponse, dm_dg

jax.config.update("jax_enable_x64", True)

LAYER = ShearResponse(jr.key(0))
M = jnp.array([5.0e3, 1.8e4, 900.0, -400.0])
G = jnp.array([0.04, -0.025])


def _rot(m, g, phi):
    """Rotate the frame by phi: the spin-2 pairs (M1,M2) and g turn by 2*phi."""
    z = jax.lax.complex(m[2], m[3]) * jnp.exp(2j * phi)
    w = jax.lax.complex(g[0], g[1]) * jnp.exp(2j * phi)
    return jnp.stack([m[0], m[1], z.real, z.imag]), jnp.stack([w.real, w.imag])


def test_rotation_equivariance():
    for phi in np.linspace(0.0, np.pi, 7):
        mr, gr = _rot(M, G, phi)
        lhs = LAYER.unshear(mr, gr)
        rhs = _rot(LAYER.unshear(M, G), G, phi)[0]
        assert jnp.max(jnp.abs(lhs - rhs) / jnp.abs(rhs)) < 1e-12, phi


def test_parity_equivariance():
    flip = lambda v: v.at[-1].set(-v[-1])       # M2 -> -M2, g2 -> -g2
    lhs = LAYER.unshear(flip(M), flip(G))
    rhs = flip(LAYER.unshear(M, G))
    assert jnp.max(jnp.abs(lhs - rhs) / jnp.abs(rhs)) < 1e-12


def test_flux_homogeneity():
    """Moments are linear in the image, so the whole map must be degree-1
    homogeneous and the response coefficients flux-blind."""
    for lam in (1e-3, 1.0, 3e2):
        lhs = LAYER.unshear(lam * M, G)
        rhs = lam * LAYER.unshear(M, G)
        assert jnp.max(jnp.abs(lhs - rhs) / jnp.abs(rhs)) < 1e-12, lam


def test_bijection_round_trip():
    """Newton must invert the map exactly, and the two log-dets must cancel."""
    for gmag in (0.0, 0.01, 0.05, 0.1, 0.2):
        g = jnp.array([gmag, -0.6 * gmag])
        y, ld_i = LAYER.inverse_and_log_det(M, g)
        back, ld_t = LAYER.transform_and_log_det(y, g)
        assert jnp.max(jnp.abs(back / M - 1.0)) < 1e-12, gmag
        assert abs(float(ld_i + ld_t)) < 1e-10, gmag


def test_identity_at_zero_shear():
    y, ld = LAYER.inverse_and_log_det(M, jnp.zeros(2))
    assert jnp.max(jnp.abs(y - M)) == 0.0 and float(ld) == 0.0


def test_derivatives_match_finite_differences():
    """dm_dg reports what the map actually does, in bfd's column layout."""
    q, r = dm_dg(LAYER, M)
    h = 1e-4
    f = lambda g: LAYER.shear(M, jnp.array(g, dtype=float))
    fd1 = [(f([h, 0]) - f([-h, 0])) / (2 * h), (f([0, h]) - f([0, -h])) / (2 * h)]
    assert jnp.max(jnp.abs(jnp.stack(fd1) - q) / jnp.abs(q)) < 1e-6
    fd11 = (f([h, 0]) - 2 * f([0, 0]) + f([-h, 0])) / h**2
    assert jnp.max(jnp.abs(fd11 - r[0]) / jnp.abs(r[0])) < 1e-4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
