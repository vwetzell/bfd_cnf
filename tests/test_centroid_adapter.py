"""`CentroidShearAdapter` and `SigmaXBlockLayer.g_blind` (models/bijections.py).

Run: `python -m tests.test_centroid_adapter` or pytest.

`bulk.build_flow(..., adapter=True)` moves the centroid stage's shear
dependence off `SigmaXBlockLayer` (which becomes `g_blind`, i.e. ignores g
entirely) and onto this new cheap near-identity layer, so the (now
g-independent) centroid layer can be peeled off by `bias.split_centroid` and
evaluated once per draw outside the g-autodiff.  Checks, in order:

  1. zero-init / g=0 identity (transform, inverse, log-det);
  2. joint rotation covariance (galaxy e, g and Sigma_X's E all spin-2;
     spin-0 outputs invariant);
  3. parity;
  4. inverse round trip at |g| = 0.02, 0.05 with de-zeroed weights, and
     log-det(transform) = -log-det(inverse);
  5. `SigmaXBlockLayer(g_blind=True)` is literally g-independent;
  6. PEEL EQUIVALENCE -- the one that matters for `bias.py`: with an adapter
     flow built via `bulk.build_flow`, `bias.split_centroid`'s peeled
     evaluation of `log_conv_is` and its g=0 value/gradient/Hessian must
     match the unpeeled one exactly (up to float32 roundoff inside
     `bias.centroid_transform`, see that test's own tolerance note).
"""
import jax

# Exact-symmetry / round-trip checks, so run in float64.
jax.config.update("jax_enable_x64", True)

import equinox as eqx            # noqa: E402
import jax.numpy as jnp          # noqa: E402
import jax.random as jr          # noqa: E402
import numpy as np               # noqa: E402

import bias                      # noqa: E402
import bulk                      # noqa: E402
from models.bijections import (  # noqa: E402
    CentroidShearAdapter, SigmaXBlockLayer, in_domain, spin2_bound, spin2_unbound)

MEAN, EMAX = 12.0, 0.1


def _de_zero(net, key, scale=0.05):
    """Replace `net`'s zero-initialised last layer with small random weights."""
    last = net.layers[-1]
    w = scale * jr.normal(key, last.weight.shape)
    b = scale * jr.normal(jr.fold_in(key, 1), last.bias.shape)
    return eqx.tree_at(lambda n: (n.layers[-1].weight, n.layers[-1].bias), net, (w, b))


def _adapter(key, zero_init=True, scale=0.05):
    lay = CentroidShearAdapter(key, nn_width=16, nn_depth=2, log_scale_mean=MEAN, e_max=EMAX)
    if zero_init:
        return lay
    return eqx.tree_at(lambda l: l.net, lay, _de_zero(lay.net, jr.fold_in(key, 99), scale))


def _cx_cond(g1, g2, C00, C01, C11):
    return jnp.array([g1, g2, C00, C01, C11])


def _rot2(v, ang):
    c, s = jnp.cos(ang), jnp.sin(ang)
    return jnp.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


Z = jnp.array([0.1, -0.2, 0.05, 0.3, -0.1])
COND0 = _cx_cond(0.0, 0.0, 1.0, 0.1, 1.2)
COND = _cx_cond(0.015, -0.008, 1.0, 0.1, 1.2)


def test_adapter_identity_at_zero_init():
    lay = _adapter(jr.key(0), zero_init=True)
    y, ld = lay.transform_and_log_det(Z, COND)
    assert jnp.allclose(y, Z, atol=1e-12), (y, Z)
    assert jnp.allclose(ld, 0.0, atol=1e-12)
    x, ld_i = lay.inverse_and_log_det(y, COND)
    assert jnp.allclose(x, Z, atol=1e-12)
    assert jnp.allclose(ld_i, 0.0, atol=1e-12)


def test_adapter_identity_at_g_zero_even_with_de_zeroed_weights():
    """Every term in `_w` carries an explicit factor of g or conj(g) -- so
    the map is the identity at g=0 REGARDLESS of the net's weights, not only
    at init."""
    lay = _adapter(jr.key(1), zero_init=False)
    y, ld = lay.transform_and_log_det(Z, COND0)
    assert jnp.allclose(y, Z, atol=1e-10), (y, Z)
    assert jnp.allclose(ld, 0.0, atol=1e-10)
    x, ld_i = lay.inverse_and_log_det(Z, COND0)
    assert jnp.allclose(x, Z, atol=1e-10)
    assert jnp.allclose(ld_i, 0.0, atol=1e-10)


def test_adapter_joint_rotation_covariance():
    """Rotate the frame by phi: (z3,z4) [galaxy e], g and Sigma_X's E all
    carry spin 2 (rotate by 2*phi in this chart, since `spin2_bound` is a
    radial -- same-angle -- reparametrisation of the physical ellipticity);
    the three spin-0 outputs (z0,z1,z2) must stay exactly unchanged and
    (z3,z4) must rotate the SAME way the input galaxy ellipticity did."""
    lay = _adapter(jr.key(2), zero_init=False)
    phi = 0.37

    e1, e2 = spin2_unbound(Z[3], Z[4])
    e_rot = _rot2(jnp.array([e1, e2]), 2 * phi)
    w1r, w2r = spin2_bound(e_rot[0], e_rot[1])
    z_rot = Z.at[3].set(w1r).at[4].set(w2r)

    g_rot = _rot2(COND[:2], 2 * phi)
    C = jnp.array([[COND[2], COND[3]], [COND[3], COND[4]]])
    R = jnp.array([[jnp.cos(phi), -jnp.sin(phi)], [jnp.sin(phi), jnp.cos(phi)]])
    Cr = R @ C @ R.T
    cond_rot = jnp.concatenate([g_rot, jnp.array([Cr[0, 0], Cr[0, 1], Cr[1, 1]])])

    y, _ = lay.transform_and_log_det(Z, COND)
    y_rot, _ = lay.transform_and_log_det(z_rot, cond_rot)

    assert jnp.allclose(y_rot[:3], y[:3], atol=1e-10), \
        f"spin-0 outputs must be rotation-invariant: {y_rot[:3]} vs {y[:3]}"
    e1y, e2y = spin2_unbound(y[3], y[4])
    ey_rot = _rot2(jnp.array([e1y, e2y]), 2 * phi)
    w1e, w2e = spin2_bound(ey_rot[0], ey_rot[1])
    assert jnp.allclose(jnp.array([y_rot[3], y_rot[4]]), jnp.array([w1e, w2e]), atol=1e-10), \
        "spin-2 output must rotate by 2*phi, same as the input ellipticity"


def test_adapter_parity():
    """(z4, g2, C01) -> (-z4, -g2, -C01) must map output z4 -> -z4 and leave
    every other output slot unchanged -- the standard parity transform for
    spin-2 fields in this chart (g, e, E all conjugate, see
    `models/shear.py`'s own docstring)."""
    lay = _adapter(jr.key(3), zero_init=False)
    y, _ = lay.transform_and_log_det(Z, COND)
    zp = Z.at[4].set(-Z[4])
    condp = COND.at[1].set(-COND[1]).at[3].set(-COND[3])
    yp, _ = lay.transform_and_log_det(zp, condp)
    assert jnp.allclose(yp[4], -y[4], atol=1e-10), (yp[4], y[4])
    for i in (0, 1, 2, 3):
        assert jnp.allclose(yp[i], y[i], atol=1e-10), f"slot {i}: {yp[i]} vs {y[i]}"


def test_adapter_roundtrip_and_logdet():
    lay = _adapter(jr.key(4), zero_init=False, scale=0.05)
    for h in (0.02, 0.05):
        cond = _cx_cond(0.9 * h, -0.4 * h, 1.0, 0.1, 1.2)
        y, lad_f = lay.transform_and_log_det(Z, cond)
        x, lad_i = lay.inverse_and_log_det(y, cond)
        assert jnp.abs(x - Z).max() < 1e-9, (h, x, Z)
        assert jnp.allclose(lad_f, -lad_i, atol=1e-9), (h, lad_f, lad_i)


def test_adapter_finite_at_e_zero_float32():
    """Value, log-det and g-gradient/Hessian finite at e = 0, in FLOAT32.

    `log_conv_is` stands masked draws in at z = 0 on the peeled path, so one
    NaN here poisons a whole chunk: cv6 sn8r lost 8.4% of its targets to
    `spin2_unbound`'s safety rescale (1e-30 denominator, whose tangent
    underflows to inf only in float32 -- this file is float64 otherwise)."""
    f32 = lambda t: jax.tree.map(
        lambda a: a.astype(jnp.float32) if eqx.is_inexact_array(a) else a, t)
    J = jax.jacfwd(lambda w: jnp.stack(spin2_unbound(w[0], w[1])))(jnp.zeros(2, jnp.float32))
    assert jnp.isfinite(J).all(), J
    lay = f32(_adapter(jr.key(7), zero_init=False, scale=0.05))
    sx = jnp.array([1.0, 0.1, 1.2], jnp.float32)
    for z in (jnp.zeros(5, jnp.float32), jnp.array([0.3, 0.1, -0.2, 0.0, 0.0], jnp.float32)):
        y, ld = lay.transform_and_log_det(z, jnp.concatenate([jnp.array([0.01, 0.0], jnp.float32), sx]))
        assert jnp.isfinite(y).all() and jnp.isfinite(ld), (z, y, ld)
        v, g, h = bias._val_grad_hess(
            lambda gg: lay.transform_and_log_det(z, jnp.concatenate([gg, sx]))[1])
        assert jnp.isfinite(v) and jnp.isfinite(g).all() and jnp.isfinite(h).all(), (z, v, g, h)


def test_sigmax_g_blind_ignores_g():
    """A g_blind SigmaXBlockLayer must give identical output AND log-det for
    any g, de-zeroed weights included -- `g_blind` bypasses `_A`/`net_K`
    (`_sigma_eff`'s `_kernel_fixed` branch) entirely."""
    lay = SigmaXBlockLayer(jr.key(5), nn_width=16, nn_depth=2,
                           full_cond_dim=5, log_scale_mean=MEAN, e_max=EMAX,
                           g_blind=True)
    names = ("net_flux", "net_size", "net_dip", "net_quad", "net_flux_e",
            "net_mc", "net_eown", "net_K")
    nets = tuple(_de_zero(getattr(lay, n), jr.fold_in(jr.key(5), i), 0.3)
                for i, n in enumerate(names))
    lay = eqx.tree_at(lambda l: tuple(getattr(l, n) for n in names), lay, nets)

    x = jnp.array([0.4, -0.7, 0.2, -0.1, 0.55])
    c0 = _cx_cond(0.0, 0.0, jnp.exp(2 * MEAN), 0.0, jnp.exp(2 * MEAN))
    c1 = _cx_cond(0.03, -0.015, jnp.exp(2 * MEAN), 0.0, jnp.exp(2 * MEAN))
    y0, ld0 = lay.transform_and_log_det(x, c0)
    y1, ld1 = lay.transform_and_log_det(x, c1)
    assert jnp.allclose(y0, y1, atol=1e-12), (y0, y1)
    assert jnp.allclose(ld0, ld1, atol=1e-12), (ld0, ld1)
    xr0, ldi0 = lay.inverse_and_log_det(y0, c0)
    xr1, ldi1 = lay.inverse_and_log_det(y1, c1)
    assert jnp.allclose(xr0, xr1, atol=1e-12), (xr0, xr1)
    assert jnp.allclose(ldi0, ldi1, atol=1e-12), (ldi0, ldi1)


# ---------------------------------------------------------------------------
# PEEL EQUIVALENCE
# ---------------------------------------------------------------------------

#: Same convention as tests/test_psi_logdet.py: [C00, C01, C11].
SIGMA_X = jnp.asarray([1.155e4, 0.0, 1.155e4])


def _synthetic_flow():
    key = jr.key(0)
    m = np.asarray(
        jr.uniform(key, (512, 5), minval=jnp.asarray([2e3, 6e3, -5e2, -5e2, 3e4]),
                   maxval=jnp.asarray([9e3, 2.4e4, 5e2, 5e2, 1.2e5])), np.float64)
    m = m[np.asarray(in_domain(jnp.asarray(m)))]
    assert len(m) > 64, len(m)
    flow = bulk.build_flow(key, m, shear=True, centroid=True, adapter=True)

    # De-zero the SigmaXBlockLayer's, the adapter's and the shear coefficient
    # net's last layers with SMALL random weights -- a trained-but-untested
    # flow, not a zero-init one, is what the peel has to survive.
    bij = flow.bijection.bijection.bijections
    sx_idx = next(i for i, b in enumerate(bij) if isinstance(b, SigmaXBlockLayer))
    ad_idx = next(i for i, b in enumerate(bij) if isinstance(b, CentroidShearAdapter))
    from models.shear import ShearResponse
    sh_idx = next(i for i, b in enumerate(bij) if isinstance(b, ShearResponse))

    sx_names = ("net_flux", "net_size", "net_dip", "net_quad", "net_flux_e",
               "net_mc", "net_eown")
    sx = bij[sx_idx]
    sx_nets = tuple(_de_zero(getattr(sx, n), jr.fold_in(jr.key(10), i), 0.03)
                    for i, n in enumerate(sx_names))
    sx = eqx.tree_at(lambda l: tuple(getattr(l, n) for n in sx_names), sx, sx_nets)

    ad = bij[ad_idx]
    ad = eqx.tree_at(lambda l: l.net, ad, _de_zero(ad.net, jr.key(11), 0.03))

    sh = bij[sh_idx]
    sh = eqx.tree_at(lambda l: l.coeffs.net, sh,
                     _de_zero(sh.coeffs.net, jr.key(12), 0.03))

    flow = eqx.tree_at(lambda f: (f.bijection.bijection.bijections[sx_idx],
                                  f.bijection.bijection.bijections[ad_idx],
                                  f.bijection.bijection.bijections[sh_idx]),
                       flow, (sx, ad, sh))
    return flow, m


def test_peel_equivalence_value_grad_hess():
    flow, m = _synthetic_flow()
    rng = np.random.default_rng(0)
    m_is = jnp.asarray(m[rng.choice(len(m), 3, replace=False)])
    draws = jnp.asarray(m[rng.choice(len(m), 64, replace=False)])
    log_wt = jnp.zeros(draws.shape[0])

    flow_g, peel = bias.split_centroid(flow)
    assert peel is not None, "a g_blind SigmaXBlockLayer must be peelable"
    z, ld = bias.centroid_transform(peel, draws, SIGMA_X, to_host=False)
    ok = bias.in_domain(draws)

    for m_i in m_is:
        f_full = lambda g: bias.log_conv_is(
            flow, m_i, draws, log_wt, bias.condition(g, SIGMA_X))
        f_peel = lambda g: bias.log_conv_is(
            flow_g, m_i, z, log_wt + ld, bias.condition(g, SIGMA_X), ok=ok)

        v0, g0, h0 = bias._val_grad_hess(f_full)
        v1, g1, h1 = bias._val_grad_hess(f_peel)

        # `bias.centroid_transform` casts draws to float32 internally (see
        # its own docstring/`_centroid_apply`), so the two paths agree only
        # to float32 roundoff, not float64 -- tolerances combine a tight
        # relative term with a small absolute floor for the near-zero
        # gradient/Hessian entries, rather than a bare rtol.
        assert jnp.allclose(v0, v1, rtol=1e-6, atol=1e-5), (v0, v1)
        assert jnp.allclose(g0, g1, rtol=1e-6, atol=1e-5), (g0, g1)
        assert jnp.allclose(h0, h1, rtol=1e-6, atol=1e-4), (h0, h1)


if __name__ == "__main__":
    test_adapter_identity_at_zero_init()
    test_adapter_identity_at_g_zero_even_with_de_zeroed_weights()
    test_adapter_joint_rotation_covariance()
    test_adapter_parity()
    test_adapter_roundtrip_and_logdet()
    test_adapter_finite_at_e_zero_float32()
    test_sigmax_g_blind_ignores_g()
    test_peel_equivalence_value_grad_hess()
    print("ok")
