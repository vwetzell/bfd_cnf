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


def eqx_is_arr(a):
    return isinstance(a, (jnp.ndarray, np.ndarray)) and jnp.issubdtype(
        jnp.asarray(a).dtype, jnp.floating
    )


if __name__ == "__main__":
    test_bijection_roundtrip_and_identity_at_g0()
    test_shear_derivs_match_autodiff()
    test_sobolev_target_recovers_known_quadratic()
    test_losses_run_with_taylor_and_sobolev()
    print("OK: ShearTaylorLast + Sobolev term self-checks passed")
