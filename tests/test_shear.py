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

# The layer now acts on STANDARDISED coordinates, downstream of
# `RawMomentStandardize` -- so the spin-0 coordinates are slots 0, 1, 2 and the
# spin-2 pair is 3, 4, NOT the raw layout's (0, 1, 4) / (2, 3).  `e_scale` is the
# standardiser's shared spin-2 std, which `build_flow` measures; 0.04 is what it
# comes out at on the bulgedisc training set.
E_SCALE = 0.04
LAYER = ShearResponse(jr.key(0), e_scale=E_SCALE)
Z = jnp.array([0.7, -0.4, 1.1, 1.3, -0.9])
G = jnp.array([0.04, -0.025])


def _rot(z, g, phi):
    """Rotate the frame by phi: the spin-2 pair (z3, z4) and g turn by 2*phi."""
    w = jax.lax.complex(z[3], z[4]) * jnp.exp(2j * phi)
    v = jax.lax.complex(g[0], g[1]) * jnp.exp(2j * phi)
    return (jnp.stack([z[0], z[1], z[2], w.real, w.imag]),
            jnp.stack([v.real, v.imag]))


def _rot_raw(m, g, phi):
    """The same rotation on RAW moments, where the spin-2 pair is (M1, M2) =
    slots 2, 3 and Mc is slot 4 -- the layout `test_flow_isotropy` needs,
    because it feeds the whole flow rather than the layer."""
    w = jax.lax.complex(m[2], m[3]) * jnp.exp(2j * phi)
    v = jax.lax.complex(g[0], g[1]) * jnp.exp(2j * phi)
    return (jnp.stack([m[0], m[1], w.real, w.imag, m[4]]),
            jnp.stack([v.real, v.imag]))


def test_rotation_equivariance():
    for phi in np.linspace(0.0, np.pi, 7):
        zr, gr = _rot(Z, G, phi)
        lhs = LAYER.unshear(zr, gr)
        rhs = _rot(LAYER.unshear(Z, G), G, phi)[0]
        assert jnp.max(jnp.abs(lhs - rhs)) < 1e-12, phi


def test_parity_equivariance():
    # y -> -y flips the second spin-2 component and g2; slots 0, 1, 2 are even.
    flip_z = lambda v: v.at[4].set(-v[4])
    flip_g = lambda v: v.at[1].set(-v[1])
    lhs = LAYER.unshear(flip_z(Z), flip_g(G))
    rhs = flip_z(LAYER.unshear(Z, G))
    assert jnp.max(jnp.abs(lhs - rhs)) < 1e-12


def test_bijection_round_trip():
    """`shear` inverts `unshear` through second order in g -- every order the
    layer models -- so the round trip is EXACT at g = 0 and drifts as |g|^3.

    This is weaker than the Newton solve it replaced, which round-tripped to
    machine precision at any |g|.  What is asserted here is the real guarantee
    of the closed form: exactness where the map is the identity, and a cubic
    residual beyond it.  The cubic scaling is the sharp part of the test -- a
    first- or second-order error in the inverted series would still look small
    at |g| = 0.01 but would break the x8-per-doubling ratio immediately.

    The log-dets cancel exactly at every g regardless, because both directions
    evaluate the same 5x5 Jacobian at the same point.
    """
    assert jnp.max(jnp.abs(LAYER.shear(Z, jnp.zeros(2)) - Z)) == 0.0

    res = {}
    for gmag in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
        g = jnp.array([gmag, -0.6 * gmag])
        y, ld_i = LAYER.inverse_and_log_det(Z, g)
        back, ld_t = LAYER.transform_and_log_det(y, g)
        # z is standardised and passes through zero, so an ABSOLUTE residual is
        # the meaningful one here; the raw version could divide by M.
        res[gmag] = float(jnp.max(jnp.abs(back - Z)))
        assert abs(float(ld_i + ld_t)) < 1e-10, gmag

    assert res[0.0] == 0.0
    # Measured 1.2e-7 / 9.9e-7 / 1.6e-5 / 1.3e-4 / 1.1e-3; bound at ~3x.
    assert res[0.01] < 4e-7, res
    assert res[0.02] < 3e-6, res
    assert res[0.2] < 3.5e-3, res
    for lo, hi in ((0.01, 0.02), (0.05, 0.1)):
        assert 6.0 < res[hi] / res[lo] < 11.0, (lo, hi, res)   # cubic: x8


def test_identity_at_zero_shear():
    y, ld = LAYER.inverse_and_log_det(Z, jnp.zeros(2))
    assert jnp.max(jnp.abs(y - Z)) == 0.0 and float(ld) == 0.0


def test_derivatives_match_finite_differences():
    """dm_dg returns dm/dg -- RAW moments -- via the chart's change of variables.

    The layer responds in standardised coordinates, so its bare g-derivative is
    dz/dg.  `dm_dg` composes the chart on both sides to recover the physical
    dm/dg that bfd tabulates, and this pins that composition: autodiff of the
    composite against finite differences of the same composite, to second order.
    Getting only the first order right would still pass a Jacobian-factor
    conversion, which is why the second-order check is here.

    That conversion is not cosmetic -- `shear.train` regresses against `dm_dg`
    with deriv_weight = 1e4, so it is the dominant term in the loss.
    """
    from models.bijections import RawMomentStandardize

    chart = RawMomentStandardize(mean=jnp.array([4.35, 0.95, 1.05, 0.0, 0.0]),
                                 std=jnp.array([0.35, 0.75, 0.60, 0.04, 0.04]))
    m = jnp.array([1.0e4, 2.0e4, 1.0e3, -444.0, 4.2e4])
    q, r = dm_dg(LAYER, m, chart)

    f = lambda g: chart.inverse(
        LAYER.shear(chart.transform(m), jnp.array(g, dtype=float)))
    h = 1e-5
    fd1 = [(f([h, 0]) - f([-h, 0])) / (2 * h), (f([0, h]) - f([0, -h])) / (2 * h)]
    assert jnp.max(jnp.abs(jnp.stack(fd1) - q) / jnp.abs(q)) < 1e-5
    fd11 = (f([h, 0]) - 2 * f([0, 0]) + f([-h, 0])) / h**2
    assert jnp.max(jnp.abs(fd11 - r[0]) / jnp.abs(r[0])) < 1e-3


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
        mr_, gr_ = _rot_raw(m, G, phi)
        assert abs(float(flow.log_prob(mr_, condition=gr_) - ref)) < 1e-8, phi


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
