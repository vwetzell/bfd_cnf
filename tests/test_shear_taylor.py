"""Self-checks for the structural shear layer (ShearTaylorLast) and the Sobolev
shear-derivative training term.

Run: python tests/test_shear_taylor.py   (or via pytest)

Covers:
  1. ShearTaylorLast is a valid bijection: forward∘inverse == identity, log|det|==0,
     and it is the identity at g=0.
  2. shear_derivs(x) equals the autodiff g-Jacobian / Hessian of the forward map.
  3. The Sobolev quadratic-fit target recovers known A, B from a synthetic
     quadratic-in-g trajectory.
  4. Both loss paths (make_nll_loss, make_elbo_loss) run with a taylor-layer flow and
     Sobolev weights > 0, and return a finite gradient.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)

from bfd_cnf.models.bijections import RawMomentStandardize, ShearTaylorLast
from bfd_cnf.models.flows import (
    build_flows,
    make_elbo_loss,
    make_nll_loss,
    _sobolev_pinv,
    _sobolev_target,
    _sobolev_row_weights,
    _batch_log_L_X,
)


def _layer():
    return ShearTaylorLast(
        jr.key(0), dim=4, raw_cond_dim=5, last_width=8, last_depth=1,
        activation=jax.nn.silu,
    )


def test_bijection_roundtrip_and_identity_at_g0():
    lay = _layer()
    # perturb coefficients off zero so the layer is non-trivial
    lay = jax.tree_util.tree_map(
        lambda a: a + 0.1 * jr.normal(jr.key(1), a.shape) if eqx_is_arr(a) else a,
        lay,
    )
    x = jnp.array([0.3, -0.7, 0.2, -0.4])
    cond = jnp.array([0.03, -0.02, 12.0, 0.0, 0.0])  # [g1,g2,log_scale,e1,e2]

    y, lad = lay.transform_and_log_det(x, cond)
    assert jnp.allclose(lad, 0.0), "shift layer must have log|det J| == 0"
    x_rt, lad_inv = lay.inverse_and_log_det(y, cond)
    assert jnp.allclose(x, x_rt, atol=1e-10), "forward∘inverse must be identity"
    assert jnp.allclose(lad_inv, 0.0)

    # identity at g=0
    cond0 = cond.at[:2].set(0.0)
    y0, _ = lay.transform_and_log_det(x, cond0)
    assert jnp.allclose(y0, x, atol=1e-12), "layer must be identity at g=0"


def test_shear_derivs_match_autodiff():
    lay = _layer()
    lay = jax.tree_util.tree_map(
        lambda a: a + 0.1 * jr.normal(jr.key(2), a.shape) if eqx_is_arr(a) else a,
        lay,
    )
    x = jnp.array([0.5, 0.1, -0.3, 0.25])
    log_scale_e = jnp.array([12.0, 0.0, 0.0])

    def decode(g):  # forward at fixed x, condition = [g1, g2, log_scale, e1, e2]
        return lay.transform_and_log_det(x, jnp.concatenate([g, log_scale_e]))[0]

    g0 = jnp.zeros(2)
    A_ad = jax.jacfwd(decode)(g0)              # (4, 2)
    B_ad = jax.jacfwd(jax.jacfwd(decode))(g0)  # (4, 2, 2)

    A, B = lay.shear_derivs(x)
    assert jnp.allclose(A, A_ad, atol=1e-9), "shear_derivs A must match autodiff dy/dg"
    assert jnp.allclose(B, B_ad, atol=1e-9), "shear_derivs B must match autodiff d2y/dg2"
    # spin-0 (flux, size) carry no first-order response
    assert jnp.allclose(A[:2], 0.0)


def test_sobolev_target_recovers_known_quadratic():
    # Build the g-grid used by the losses (centre + two 8-dir rings).
    sqrt2 = 1.0 / np.sqrt(2.0)
    ring = jnp.array([[0, 1], [sqrt2, sqrt2], [1, 0], [sqrt2, -sqrt2],
                      [0, -1], [-sqrt2, -sqrt2], [-1, 0], [-sqrt2, sqrt2]], dtype=float)
    g2d = jnp.concatenate([jnp.zeros((1, 2)), 0.01 * ring, 0.02 * ring], axis=0)  # (17,2)
    pinv, scale = _sobolev_pinv(g2d)

    # Known per-component A (4,2) and symmetric B (4,2,2); synth trajectory over grid.
    key = jr.key(3)
    A_true = jr.normal(key, (4, 2))
    Bsym = jr.normal(jr.key(4), (4, 2, 2))
    B_true = 0.5 * (Bsym + jnp.swapaxes(Bsym, -1, -2))
    g1, g2 = g2d[:, 0], g2d[:, 1]

    def traj(d):  # moment component d over the grid
        lin = A_true[d, 0] * g1 + A_true[d, 1] * g2
        quad = 0.5 * (B_true[d, 0, 0] * g1**2 + B_true[d, 1, 1] * g2**2) + B_true[d, 0, 1] * g1 * g2
        return 1.0 + lin + quad  # arbitrary constant offset

    y_grid = jnp.stack([traj(d) for d in range(4)], axis=-1)[None]  # (1, 17, 4)
    A_fit, B_fit = _sobolev_target(y_grid, pinv, scale)
    # Tolerances are the float32 floor (config.py forces x64 off): the g^2 curvature is
    # ~1e-4, so the 2nd-order B target carries ~few % noise.  A wrong sign/scale/index
    # would miss by O(1), which these still catch.
    assert jnp.allclose(A_fit[0], A_true, atol=2e-3), "Sobolev fit must recover A"
    assert jnp.allclose(B_fit[0], B_true, atol=1e-1), "Sobolev fit must recover B"


def _synthetic_dataset(n=48, seed=5):
    rng = np.random.default_rng(seed)
    Mf = rng.uniform(1000.0, 5000.0, n)
    Mr = Mf * rng.uniform(0.35, 0.8, n)
    M1 = Mr * rng.uniform(-0.2, 0.2, n)
    M2 = Mr * rng.uniform(-0.2, 0.2, n)
    y = np.stack([Mf, Mr, M1, M2], axis=-1)
    # simple positive-definite diagonal-ish covariance in raw units
    cov = np.broadcast_to(np.diag([1e4, 1e3, 5e2, 5e2]), (n, 4, 4)).copy()
    dg = rng.normal(0, 50.0, (n, 6, 2))
    d2g = rng.normal(0, 50.0, (n, 6, 2, 2))
    X = rng.normal(0, 100.0, (n, 2))
    # standardiser stats from the transformed coords
    t = np.stack([np.log10(Mf), Mr / Mf, M1 / Mr, M2 / Mr], axis=-1)
    r2s = RawMomentStandardize(mean=jnp.asarray(t.mean(0)), std=jnp.asarray(t.std(0) + 1e-6))
    return (jnp.asarray(y), jnp.asarray(cov), jnp.asarray(dg), jnp.asarray(d2g),
            jnp.asarray(X), r2s)


def test_losses_run_with_taylor_and_sobolev():
    y, cov, dg, d2g, X, r2s = _synthetic_dataset()
    N = y.shape[0]
    prior, q = build_flows(jr.key(6), latent_dim=4, cond_dim=16,
                           raw2standard=r2s, shear_layer_kind="taylor")

    # NLL path (prior only), Sobolev on, Σ_X off for a compact check.
    nll = make_nll_loss(N, batch_size=16, use_sx=False, raw2standard=r2s,
                        sobolev_g1_weight=1.0, sobolev_g2_weight=0.5)
    grads = eqx.filter_grad(lambda p: nll(p, y, cov, dg, d2g, X, jr.key(7)))(prior)
    leaves = [a for a in jax.tree_util.tree_leaves(grads) if eqx.is_inexact_array(a)]
    assert leaves and all(jnp.all(jnp.isfinite(a)) for a in leaves), "NLL+Sobolev grad must be finite"

    # ELBO path, Sobolev on, Σ_X off.
    elbo = make_elbo_loss(N, batch_size=16, num_samples=2, use_sx=False, raw2standard=r2s,
                          sobolev_g1_weight=1.0, sobolev_g2_weight=0.5)
    grads2 = eqx.filter_grad(lambda mt: elbo(mt, y, cov, dg, d2g, X, jr.key(8)))((prior, q))
    leaves2 = [a for a in jax.tree_util.tree_leaves(grads2) if eqx.is_inexact_array(a)]
    assert leaves2 and all(jnp.all(jnp.isfinite(a)) for a in leaves2), "ELBO+Sobolev grad must be finite"

    # Σ_X on: exercises the new _sobolev_row_weights(X_b, sx_conds) path (row_w != None).
    nll_sx = make_nll_loss(N, batch_size=16, use_sx=True, log_scale_range=(-1.0, 1.0),
                           raw2standard=r2s, sobolev_g1_weight=1.0, sobolev_g2_weight=0.5)
    grads3 = eqx.filter_grad(lambda p: nll_sx(p, y, cov, dg, d2g, X, jr.key(9)))(prior)
    leaves3 = [a for a in jax.tree_util.tree_leaves(grads3) if eqx.is_inexact_array(a)]
    assert leaves3 and all(jnp.all(jnp.isfinite(a)) for a in leaves3), \
        "NLL+Sobolev+use_sx grad must be finite"

    elbo_sx = make_elbo_loss(N, batch_size=16, num_samples=2, use_sx=True,
                             log_scale_range=(-1.0, 1.0), raw2standard=r2s,
                             sobolev_g1_weight=1.0, sobolev_g2_weight=0.5)
    grads4 = eqx.filter_grad(lambda mt: elbo_sx(mt, y, cov, dg, d2g, X, jr.key(10)))((prior, q))
    leaves4 = [a for a in jax.tree_util.tree_leaves(grads4) if eqx.is_inexact_array(a)]
    assert leaves4 and all(jnp.all(jnp.isfinite(a)) for a in leaves4), \
        "ELBO+Sobolev+use_sx grad must be finite"


def _fsowne_layer():
    lay = ShearTaylorLast(
        jr.key(20), dim=4, raw_cond_dim=5, last_width=8, last_depth=1,
        activation=jax.nn.silu, split_ab=True, spin2_owne=True, flux_size_owne=True,
    )
    return jax.tree_util.tree_map(
        lambda a: a + 0.3 * jr.normal(jr.key(21), a.shape) if eqx_is_arr(a) else a,
        lay,
    )


def test_flux_size_owne_bijection_roundtrip():
    """flux_size_owne (combined with spin2_owne, the trained-flow config) must
    still be a valid bijection and an identity at g=0. Unlike the plain (non-
    spin2_owne) layer, log|det| here is the real spin-2 value (not 0), and the
    spin-2 inverse is only exact to O(g^2) by construction (see
    inverse_and_log_det's spin2_owne branch docstring) -- flux_size_owne's own
    stage is exact (block-triangular, see __init__ note), so the roundtrip
    residual should stay at that same O(g^2) floor, not blow up."""
    lay = _fsowne_layer()
    x = jnp.array([0.3, -0.7, 0.2, -0.4])
    cond = jnp.array([0.03, -0.02, 12.0, 0.0, 0.0])

    y, lad = lay.transform_and_log_det(x, cond)
    assert jnp.all(jnp.isfinite(lad))
    x_rt, lad_inv = lay.inverse_and_log_det(y, cond)
    assert jnp.allclose(x, x_rt, atol=2e-3), "flux_size_owne forward∘inverse must match to O(g^2)"
    assert jnp.allclose(lad + lad_inv, 0.0, atol=2e-3), "fwd/inv log|det| must cancel to O(g^2)"

    cond0 = cond.at[:2].set(0.0)
    y0, _ = lay.transform_and_log_det(x, cond0)
    assert jnp.allclose(y0, x, atol=1e-9), "flux_size_owne layer must be identity at g=0"


def test_flux_size_owne_B12_nonzero_and_equivariant():
    """The bug this fixes: net_flux_e/net_size_e used to emit (B11,B22,B12) directly
    from an orientation-blind input, which can only ever represent B11==B22, B12==0
    (see memory shear-taylor-flux-size-blind-to-orientation). The fix builds B from
    two invariant scalars (c0,c1) combined with the equivariant tensor
    (e1^2,e2^2,e1e2) — this checks (1) B12 is now generically NONZERO, and (2) the
    resulting shift is exactly invariant under a rotation applied jointly to
    (e1,e2) and (g1,g2), the defining property a spin-0 (flux/size) shift must have."""
    lay = _fsowne_layer()
    e1, e2 = 0.18, -0.09
    g1, g2 = 0.02, -0.015

    B_flux = lay._flux_owne_B(e1, e2)
    B_size = lay._size_owne_B(0.4, e1, e2)
    assert abs(float(B_flux[2])) > 1e-8, "flux B12 must be able to be nonzero"
    assert abs(float(B_size[2])) > 1e-8, "size B12 must be able to be nonzero"

    def shift_scalar(e1v, e2v, g1v, g2v):
        Bf = lay._flux_owne_B(e1v, e2v)
        Bs = lay._size_owne_B(0.4, e1v, e2v)
        zeroA = jnp.zeros(2)
        return lay._shift(zeroA, Bf, g1v, g2v), lay._shift(zeroA, Bs, g1v, g2v)

    theta = 0.7
    c, s = jnp.cos(theta), jnp.sin(theta)
    e1r, e2r = c * e1 - s * e2, s * e1 + c * e2
    g1r, g2r = c * g1 - s * g2, s * g1 + c * g2

    sf0, ss0 = shift_scalar(e1, e2, g1, g2)
    sf1, ss1 = shift_scalar(e1r, e2r, g1r, g2r)
    assert jnp.allclose(sf0, sf1, atol=1e-9), "flux shift must be rotation-invariant"
    assert jnp.allclose(ss0, ss1, atol=1e-9), "size shift must be rotation-invariant"


def test_sobolev_row_weights_match_density_softmax_mean():
    """_sobolev_row_weights caps each stencil point's softmax at max_weight_mult/B
    before renormalising (see docstring: an uncapped self-normalised softmax can
    collapse onto one outlier row and caused a real training NaN divergence,
    2026-07-30). This checks the capped-then-renormalised result matches a manual
    reimplementation, and that a wide-open cap recovers the uncapped
    mean-of-softmax identity."""
    rng = np.random.default_rng(3)
    X = jnp.asarray(rng.normal(0, 50.0, (12, 2)))
    sx_conds = jnp.asarray(
        np.stack([rng.uniform(-1, 1, 5), rng.uniform(-0.3, 0.3, 5),
                  rng.uniform(-0.3, 0.3, 5)], axis=-1)
    )
    B = X.shape[0]

    def _manual(cap_mult):
        w_c = jnp.stack([jax.nn.softmax(_batch_log_L_X(X, c)) for c in sx_conds], axis=0)
        w_c = jnp.minimum(w_c, cap_mult / B)
        w_c = w_c / jnp.sum(w_c, axis=1, keepdims=True)
        return jnp.mean(w_c, axis=0)

    w = _sobolev_row_weights(X, sx_conds)  # default cap = 20
    assert jnp.allclose(jnp.sum(w), 1.0, atol=1e-6), "row weights must sum to 1"
    assert jnp.allclose(w, _manual(20.0), atol=1e-10), "must match manual capped mean-of-softmax"
    assert jnp.max(w) <= 20.0 / B + 1e-9, "no row may exceed the cap"

    # A wide-open cap (never binds) must recover the uncapped identity.
    w_uncapped = _sobolev_row_weights(X, sx_conds, max_weight_mult=1e6)
    expected_uncapped = jnp.mean(
        jnp.stack([jax.nn.softmax(_batch_log_L_X(X, c)) for c in sx_conds], axis=0), axis=0
    )
    assert jnp.allclose(w_uncapped, expected_uncapped, atol=1e-9), \
        "an effectively-infinite cap must recover the uncapped mean-of-softmax"


def eqx_is_arr(a):
    return isinstance(a, (jnp.ndarray, np.ndarray)) and jnp.issubdtype(
        jnp.asarray(a).dtype, jnp.floating
    )


if __name__ == "__main__":
    test_bijection_roundtrip_and_identity_at_g0()
    test_shear_derivs_match_autodiff()
    test_sobolev_target_recovers_known_quadratic()
    test_flux_size_owne_bijection_roundtrip()
    test_flux_size_owne_B12_nonzero_and_equivariant()
    test_sobolev_row_weights_match_density_softmax_mean()
    test_losses_run_with_taylor_and_sobolev()
    print("OK: ShearTaylorLast + Sobolev term self-checks passed")
