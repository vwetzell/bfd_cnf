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
M = jnp.array([5.0e3, 1.8e4, 900.0, -400.0, 1.1e5])
G = jnp.array([0.04, -0.025])


def _rot(m, g, phi):
    """Rotate the frame by phi: the spin-2 pairs (M1,M2) and g turn by 2*phi."""
    z = jax.lax.complex(m[2], m[3]) * jnp.exp(2j * phi)
    w = jax.lax.complex(g[0], g[1]) * jnp.exp(2j * phi)
    return (jnp.stack([m[0], m[1], z.real, z.imag, m[4]]),
            jnp.stack([w.real, w.imag]))


def test_rotation_equivariance():
    for phi in np.linspace(0.0, np.pi, 7):
        mr, gr = _rot(M, G, phi)
        lhs = LAYER.unshear(mr, gr)
        rhs = _rot(LAYER.unshear(M, G), G, phi)[0]
        assert jnp.max(jnp.abs(lhs - rhs) / jnp.abs(rhs)) < 1e-12, phi


def test_parity_equivariance():
    # y -> -y flips M2 and g2; Mf, Mr, M1 and Mc are all parity even.
    flip_m = lambda v: v.at[3].set(-v[3])
    flip_g = lambda v: v.at[1].set(-v[1])
    lhs = LAYER.unshear(flip_m(M), flip_g(G))
    rhs = flip_m(LAYER.unshear(M, G))
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


def test_flow_isotropy():
    """The WHOLE stack must be isotropic, not just the shear layer.

    The unlensed population has no preferred direction on the sky, so rotating
    a galaxy's shape and the shear together must leave the density alone.  The
    training sample here is deliberately lopsided in M1 vs M2 -- exactly the
    shape-noise fluctuation a real catalog has -- because the one place that
    can absorb it is `RawMomentStandardize`, whose per-coordinate mean and std
    are fitted to it.  A prior with a preferred direction reads out as additive
    shear bias, so this is a bias test wearing a symmetry test's clothes.
    """
    import bulk

    n = 4000
    Mf = 10 ** jr.uniform(jr.key(1), (n,), minval=3.0, maxval=4.5)
    Mr = Mf * jr.uniform(jr.key(2), (n,), minval=1.0, maxval=3.5)
    Mc = Mr * jr.uniform(jr.key(3), (n,), minval=1.8, maxval=2.4)
    # Lopsided on purpose: different width AND a nonzero mean on M1/Mr.
    e1 = 0.05 * jr.normal(jr.key(4), (n,)) + 0.01
    e2 = 0.04 * jr.normal(jr.key(5), (n,))
    m_train = jnp.stack([Mf, Mr, e1 * Mr, e2 * Mr, Mc], axis=-1)
    flow = bulk.build_flow(jr.key(0), np.asarray(m_train), shear=True)

    # The test point must be IN this population.  The module-level M is built
    # for the layer tests and sits far outside it, where `Spin2CouplingLayer`'s
    # scale saturates at its floor, crushes the spin-2 coordinates to nothing
    # and makes log_prob trivially shape-independent -- an asymmetric flow
    # passes that.  In distribution, this test resolves the asymmetry at ~7e-2
    # nats, six orders of magnitude above the tolerance.
    Mr0 = 2.0e4
    m = jnp.array([1.0e4, Mr0, 0.05 * Mr0, -0.0222 * Mr0, 2.1 * Mr0])
    ref = flow.log_prob(m, condition=G)
    for phi in np.linspace(0.0, np.pi, 5):
        mr_, gr_ = _rot(m, G, phi)
        assert abs(float(flow.log_prob(mr_, condition=gr_) - ref)) < 1e-8, phi


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
