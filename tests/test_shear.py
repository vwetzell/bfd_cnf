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

    That conversion is not cosmetic -- `shear.check` compares against `dm_dg`
    as an independent held-out diagnostic of the trained layer's response.
    """
    from models.bijections import RawMomentStandardize

    mean = jnp.array([4.35, 0.95, 1.05, 0.0, 0.0])
    std = jnp.array([0.35, 0.75, 0.60, 0.04, 0.04])
    chart = RawMomentStandardize(mean=mean, std=std)
    # The layer reads the chart's spin-0 mean and std for its own Jacobian
    # factor, so hand it the SAME chart -- a mismatched pair is still a valid
    # bijection and would still pass, but it would not be the configuration
    # `build_flow` produces.
    layer = ShearResponse(jr.key(0), e_scale=E_SCALE,
                          chart_loc=mean[:3], chart_scale=std[:3])
    m = jnp.array([1.0e4, 2.0e4, 1.0e3, -444.0, 4.2e4])
    q, r = dm_dg(layer, m, chart)

    f = lambda g: chart.inverse(
        layer.shear(chart.transform(m), jnp.array(g, dtype=float)))
    h = 1e-5
    fd1 = [(f([h, 0]) - f([-h, 0])) / (2 * h), (f([0, h]) - f([0, -h])) / (2 * h)]
    assert jnp.max(jnp.abs(jnp.stack(fd1) - q) / jnp.abs(q)) < 1e-5
    # A second difference of moments ~1e4 cancels to ~1e-16 * 1e4 / h^2, so h
    # cannot be as small as the first-order check's.
    h = 1e-3
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


def test_antithetic_pairing():
    """`_antithetic` pairs the SAME template at +g and -g, at the full batch.

    Both halves of this failed silently for months: the shears were drawn at
    half length, so the duplicated template list was truncated back down and
    the +g / -g arms landed on different templates.  No scatter cancelled and
    `--batch` delivered half its rows.  Neither shows up in a loss curve --
    `val nll` is fine either way -- so only an explicit check catches it.
    """
    from shear import _antithetic

    idx = jnp.arange(8)
    g, take = _antithetic(idx, jr.key(0), 0.02)
    n = idx.shape[0]

    # Full batch: 2n rows out for n templates in, not n.
    assert take.shape[0] == 2 * n, take.shape
    assert g.shape[0] == 2 * n, g.shape

    # The pairing itself: row k and row k + n are one template at +/- one shear.
    assert jnp.array_equal(take[:n], take[n:]), (take[:n], take[n:])
    assert jnp.allclose(g[:n], -g[n:]), (g[:n], g[n:])

    # And the pair really does cancel: the batch mean shear is exactly zero.
    assert jnp.max(jnp.abs(g.mean(0))) < 1e-7, g.mean(0)


def test_scan_preserves_rng_stream():
    """Fusing the training loop into `lax.scan` must not change what is trained on.

    `train` runs its steps inside one `lax.scan` per REPORT steps rather than one
    jit dispatch per step (7x faster: the flow is ~1e8 FLOPs per step against
    ~8ms of dispatch latency).  The risk in that change is the RNG: keys used to
    be split in Python between dispatches and are now split inside the scan
    carry.  This pins that the two orderings draw the identical templates and
    shears, so the speedup cannot silently become a different experiment.
    """
    from shear import _antithetic

    batch, n, steps = 1024, 90000, 40

    key = jr.key(1)
    loop_idx, loop_g = [], []
    for _ in range(steps):
        key, sk, gk = jr.split(key, 3)
        idx = jr.randint(sk, (batch // 2,), 0, n)
        g, _ = _antithetic(idx, gk, 0.02)
        loop_idx.append(idx)
        loop_g.append(g)

    def one(k, _):
        k, sk, gk = jr.split(k, 3)
        idx = jr.randint(sk, (batch // 2,), 0, n)
        g, _ = _antithetic(idx, gk, 0.02)
        return k, (idx, g)

    _, (scan_idx, scan_g) = jax.lax.scan(one, jr.key(1), None, length=steps)

    assert jnp.array_equal(jnp.stack(loop_idx), scan_idx)
    assert jnp.array_equal(jnp.stack(loop_g), scan_g)


def test_coefficient_bound_does_not_kill_gradients():
    """The response-coefficient bound must saturate WITHOUT severing the gradient.

    It used to be `C*tanh(net/C)`, whose derivative `sech^2(net/C)` dies
    exponentially -- a saturated coefficient got no gradient and could never
    recover, so training was a one-way ratchet that progressively killed
    coefficients (median attenuation 3.5e-9 and 63% of coefficients dead by
    768k steps).  The rational form bounds identically but decays as (C/x)^3.

    Pins all three properties that matter: the bound still holds, the small-
    signal behaviour is unchanged (unit slope at 0, so it is a drop-in), and a
    deeply saturated coefficient still receives usable gradient.
    """
    from models.shear import _COEFF_MAX, _Coeffs

    C = _COEFF_MAX
    f = lambda x: x * jax.lax.rsqrt(1.0 + (x / C) ** 2)
    g = jax.grad(f)

    assert abs(float(g(0.0)) - 1.0) < 1e-6          # drop-in near the origin
    for k in (1, 5, 10, 50):                        # bound holds everywhere
        assert abs(float(f(k * C))) < C
    # the point of the change: still alive far into saturation, where tanh is not
    assert float(g(10.0 * C)) > 1e-4
    assert float(g(10.0 * C)) > 1e4 * float(1 / jnp.cosh(jnp.array(10.0)) ** 2)

    # and the real module agrees with the closed form it is meant to implement
    c = _Coeffs(jr.key(0), 32, 2, jax.nn.silu)
    out = c(3.8, 0.5, 1.9, 2.0)
    assert out.shape == (14,)
    assert jnp.all(jnp.abs(out) < C)


def test_coeff_input_whitening():
    """`bulk.build_flow` must hand the coefficient net WHITENED inputs.

    The four inputs are badly conditioned as they stand: on the bulgedisc
    catalog `a` (size) and `b` (concentration) correlate at 0.997 and the
    covariance condition number is ~2600, while `(q-2)/2` has std 2.76 rather
    than the 1.0 its chi^2(2) derivation assumes.  Adam cannot fix input
    correlation, so this is a plausible source of the spin-0 seed spread.

    Pins that the stored statistics genuinely whiten what the net sees, and
    that the identity default is a true no-op for callers that supply none.
    """
    import bulk
    from paramax import unwrap
    from models.shear import _Coeffs, _invariants, _Q_LOC, _Q_SCALE

    rng = np.random.default_rng(0)
    n = 4000
    mf = 10 ** rng.uniform(3.2, 4.2, n)
    mr = mf * rng.uniform(2.0, 3.5, n)
    mc = mr * rng.uniform(2.0, 6.0, n)
    e = rng.normal(0, 0.05, (n, 2))
    m = np.stack([mf, mr, e[:, 0] * mr, e[:, 1] * mr, mc], axis=-1)

    flow = bulk.build_flow(jr.key(0), m, layers=2, shear=True)
    layer = [b for b in flow.bijection.bijection.bijections
             if type(b).__name__ == "ShearResponse"][0]
    chart = flow.bijection.bijection.bijections[0]

    z = jax.vmap(chart.transform)(jnp.asarray(m))
    mu, W = unwrap(layer.coeffs.u_mean), unwrap(layer.coeffs.u_white)
    u = jax.vmap(lambda zi: W @ (jnp.stack(
        [_invariants(zi)[0], _invariants(zi)[1], _invariants(zi)[2],
         (_invariants(zi)[3] - _Q_LOC) / _Q_SCALE]) - mu))(z)

    cov = jnp.cov(u.T)
    assert float(jnp.linalg.cond(cov)) < 3.0, float(jnp.linalg.cond(cov))
    assert jnp.max(jnp.abs(u.mean(0))) < 0.05, u.mean(0)

    # identity default really is a no-op
    plain = _Coeffs(jr.key(0), 16, 1, jax.nn.silu)
    assert jnp.array_equal(unwrap(plain.u_mean), jnp.zeros(4))
    assert jnp.array_equal(unwrap(plain.u_white), jnp.eye(4))


def test_chart_spin0_jacobian():
    """`_chart_spin0_jac` IS the chart's Jacobian along the pure spin-0 directions.

    That is the whole content of the reparameterisation: the network emits raw
    response coefficients `c_X` in `dX/dg = X c_X Re(ebar.g)`, and this matrix
    carries them into z.  If it drifts from the chart -- a changed logit, a
    changed ceiling -- the layer's coefficients silently stop meaning what
    `models/shear.py` says they mean, so pin it against autodiff of the chart
    itself rather than against the algebra it was derived from.
    """
    import numpy as np
    import bulk
    from models.shear import _chart_spin0_jac
    from paramax import unwrap

    rng = np.random.default_rng(1)
    n = 2000
    mf = 10 ** rng.uniform(3.2, 4.2, n)
    mr = mf * rng.uniform(2.0, 3.5, n)
    mc = mr * rng.uniform(2.0, 6.0, n)
    e = rng.normal(0, 0.05, (n, 2))
    m = np.stack([mf, mr, e[:, 0] * mr, e[:, 1] * mr, mc], axis=-1)

    flow = bulk.build_flow(jr.key(0), m, layers=2, shear=True)
    layer = [b for b in flow.bijection.bijection.bijections
             if type(b).__name__ == "ShearResponse"][0]
    chart = flow.bijection.bijection.bijections[0]
    loc, scale = unwrap(layer.chart_loc), unwrap(layer.chart_scale)

    def dz_dlogX(mi):
        """dz_{0,1,2} / d(log Mf, log Mr, log Mc), at fixed M1/Mr and M2/Mr."""
        def f(t):
            s = jnp.exp(t)
            return chart.transform(jnp.stack([mi[0] * s[0], mi[1] * s[1],
                                              mi[2] * s[1], mi[3] * s[1],
                                              mi[4] * s[2]]))[:3]
        return jax.jacfwd(f)(jnp.zeros(3))

    mj = jnp.asarray(m[:50])
    exact = jax.vmap(dz_dlogX)(mj)
    ours = jax.vmap(lambda mi: _chart_spin0_jac(chart.transform(mi), loc, scale))(mj)
    # chart_loc/chart_scale are held in float32, like every other frozen chart
    # statistic in the layer, so this is float32 round-off and not a mismatch.
    rel = jnp.max(jnp.abs(exact - ours) / jnp.abs(exact).max())
    assert rel < 1e-6, rel


def test_e_scale_is_the_charts_effective_spin2_std():
    """`e_scale` must be the std z3 and z4 are ACTUALLY divided by.

    `RawMomentStandardize._effective` symmetrises the spin-2 pair -- that is
    what keeps the chart isotropic -- so slot 3's own std is not it.  Passing
    the wrong one scales every physical-unit quantity in `response` by the
    ratio, which was 0.67% on the bulgedisc training set: small, absorbable by
    training, and silently wrong in every comparison against bfd.
    """
    import numpy as np
    import bulk
    from paramax import unwrap

    rng = np.random.default_rng(2)
    n = 3000
    mf = 10 ** rng.uniform(3.2, 4.2, n)
    mr = mf * rng.uniform(2.0, 3.5, n)
    # Deliberately anisotropic sample: the two spin-2 stds differ, so a layer
    # that took slot 3's own std would fail here and pass on a symmetric one.
    m = np.stack([mf, mr, mr * rng.normal(0, 0.06, n), mr * rng.normal(0, 0.03, n),
                  mr * rng.uniform(2.0, 6.0, n)], axis=-1)

    flow = bulk.build_flow(jr.key(0), m, layers=2, shear=True)
    layer = [b for b in flow.bijection.bijection.bijections
             if type(b).__name__ == "ShearResponse"][0]
    chart = flow.bijection.bijection.bijections[0]
    want = float(chart._effective()[1][3])
    assert abs(float(unwrap(layer.e_scale)) - want) < 1e-9 * want
