"""Self-checks for ShearPerturbative, the Sobolev-free shear-conditioning layer.

Run: python tests/test_shear_perturbative.py   (or via pytest)

Covers:
  1. ShearPerturbative is a valid bijection: forward∘inverse == identity (to the
     Picard-iteration inverse's residual floor), and it is the identity at g=0.
  2. shear_derivs(x) matches the autodiff g-Jacobian/Hessian of the forward map.
  3. The shift is generically nonzero and dimensionally sane. NOTE: this design
     (coefficients = free neural-net outputs of the full moment vector,
     multiplied by plain g-monomials -- see the class docstring for why) does
     NOT build rotation-equivariance into the architecture the way the earlier
     invariant-tensor design did, so there is no structural equivariance
     assertion to test here for a random/untrained net; equivariance is
     expected to be approximately LEARNED from the (symmetric) training data,
     checked empirically against real Pqr derivatives in
     dev/check_functional_form.py, not asserted exactly in this unit test.
  4. new_masked_autoregressive_flow / build_flows can select shear_layer_kind
     ="perturbative" and produce a working flow.
  5. make_nll_loss's shear_g_train_max (randomised, widened-g training) and
     shear_coeff_reg_weight (soft coefficient penalty) both run and produce a
     finite gradient -- WITHOUT any Sobolev term.
  6. shear_g_train_max raises if combined with Sobolev weights (the two are
     meant to be alternatives, not combined).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)

from bfd_cnf.models.bijections import (
    RawMomentStandardize,
    ShearPerturbative,
    new_masked_autoregressive_flow,
)
from bfd_cnf.models.flows import build_flows, make_nll_loss
from flowjax.distributions import MultivariateNormal
from paramax import non_trainable


def _layer():
    lay = ShearPerturbative(
        jr.key(20), dim=4, raw_cond_dim=5, last_width=8, last_depth=1,
        activation=jax.nn.silu,
    )
    # perturb coefficients off zero so the layer is non-trivial
    return jax.tree_util.tree_map(
        lambda a: a + 0.3 * jr.normal(jr.key(21), a.shape) if _is_arr(a) else a,
        lay,
    )


def _is_arr(a):
    return isinstance(a, (jnp.ndarray, np.ndarray)) and jnp.issubdtype(
        jnp.asarray(a).dtype, jnp.floating
    )


def test_bijection_roundtrip_and_identity_at_g0():
    lay = _layer()
    x = jnp.array([0.3, -0.7, 0.2, -0.4])
    cond = jnp.array([0.03, -0.02, 12.0, 0.0, 0.0])  # [g1,g2,log_scale,e1,e2]

    y, lad = lay.transform_and_log_det(x, cond)
    assert jnp.all(jnp.isfinite(lad))
    x_rt, lad_inv = lay.inverse_and_log_det(y, cond)
    assert jnp.allclose(x, x_rt, atol=2e-3), "forward∘inverse must match to O(g^2)"
    assert jnp.allclose(lad + lad_inv, 0.0, atol=2e-3), "fwd/inv log|det| must cancel to O(g^2)"

    cond0 = cond.at[:2].set(0.0)
    y0, _ = lay.transform_and_log_det(x, cond0)
    assert jnp.allclose(y0, x, atol=1e-9), "layer must be identity at g=0"


def test_shear_derivs_match_autodiff():
    lay = _layer()
    x = jnp.array([0.5, 0.1, -0.3, 0.25])
    log_scale_e = jnp.array([12.0, 0.0, 0.0])

    def decode(g):
        return lay.transform_and_log_det(x, jnp.concatenate([g, log_scale_e]))[0]

    g0 = jnp.zeros(2)
    A_ad = jax.jacfwd(decode)(g0)
    B_ad = jax.jacfwd(jax.jacfwd(decode))(g0)

    A, B = lay.shear_derivs(x)
    assert jnp.allclose(A, A_ad, atol=1e-9), "shear_derivs A must match autodiff dy/dg"
    assert jnp.allclose(B, B_ad, atol=1e-9), "shear_derivs B must match autodiff d2y/dg2"
    assert jnp.any(jnp.abs(A[:2]) > 1e-6), "flux/size A should be nonzero for this probe"


def test_shift_nonzero_and_coeffs_shape():
    """No structural equivariance assertion here -- see module docstring.
    Just checks the shift/coefficients are generically nonzero and correctly
    shaped for a perturbed (non-init) layer."""
    lay = _layer()
    x0, x1 = 0.4, -0.2
    e1, e2 = 0.18, -0.09
    g1, g2 = 0.02, -0.015
    x = jnp.array([x0, x1, e1, e2])

    A, B = lay.shear_derivs(x)
    assert A.shape == (4, 2) and B.shape == (4, 2, 2)
    assert jnp.any(jnp.abs(A) > 1e-8), "A must be able to be nonzero"
    assert jnp.any(jnp.abs(B) > 1e-8), "B must be able to be nonzero"
    assert jnp.allclose(B, jnp.swapaxes(B, -1, -2), atol=1e-9), \
        "B must be symmetric in its two g-indices by construction (0.5*g^T B g)"

    coeffs = lay.coeffs(x0, x1, e1, e2)
    assert coeffs.shape == (20,)

    shift = lay._shift(x, g1, g2)
    assert shift.shape == (4,)
    assert jnp.any(jnp.abs(shift) > 1e-8), "shift must be able to be nonzero away from g=0"


def test_flow_builds_with_perturbative_layer():
    prior = new_masked_autoregressive_flow(
        jr.key(1),
        base_dist=non_trainable(MultivariateNormal(jnp.zeros(4), jnp.eye(4))),
        flow_layers=3, nn_width=8, nn_depth=1, nn_activation=jax.nn.silu,
        last_layer_cond_dim=5, last_layer_nn_width=8, last_layer_nn_depth=1,
        sigmax_cond_dim=None,
        shear_layer_kind="perturbative",
    )
    x = jnp.array([0.1, -0.2, 0.05, -0.03])
    cond = jnp.array([0.01, 0.0, 0.0, 0.0, 0.0])
    lp = prior.log_prob(x, condition=cond)
    assert jnp.isfinite(lp)


def _synthetic_dataset(n=48, seed=5):
    rng = np.random.default_rng(seed)
    Mf = rng.uniform(1000.0, 5000.0, n)
    Mr = Mf * rng.uniform(0.35, 0.8, n)
    M1 = Mr * rng.uniform(-0.2, 0.2, n)
    M2 = Mr * rng.uniform(-0.2, 0.2, n)
    y = np.stack([Mf, Mr, M1, M2], axis=-1)
    cov = np.broadcast_to(np.diag([1e4, 1e3, 5e2, 5e2]), (n, 4, 4)).copy()
    dg = rng.normal(0, 50.0, (n, 6, 2))
    d2g = rng.normal(0, 50.0, (n, 6, 2, 2))
    X = rng.normal(0, 100.0, (n, 2))
    t = np.stack([np.log10(Mf), Mr / Mf, M1 / Mr, M2 / Mr], axis=-1)
    r2s = RawMomentStandardize(mean=jnp.asarray(t.mean(0)), std=jnp.asarray(t.std(0) + 1e-6))
    return (jnp.asarray(y), jnp.asarray(cov), jnp.asarray(dg), jnp.asarray(d2g),
            jnp.asarray(X), r2s)


def test_widened_g_and_coeff_reg_loss_runs_no_sobolev():
    y, cov, dg, d2g, X, r2s = _synthetic_dataset()
    N = y.shape[0]
    prior, _q = build_flows(
        jr.key(6), latent_dim=4, cond_dim=16, raw2standard=r2s,
        shear_layer_kind="perturbative",
    )

    nll = make_nll_loss(
        N, batch_size=16, use_sx=False, raw2standard=r2s,
        sobolev_g1_weight=0.0, sobolev_g2_weight=0.0,
        shear_g_train_max=0.15, shear_coeff_reg_weight=1e-3,
    )
    loss_val = nll(prior, y, cov, dg, d2g, X, jr.key(7))
    assert jnp.isfinite(loss_val)

    grads = eqx.filter_grad(lambda p: nll(p, y, cov, dg, d2g, X, jr.key(8)))(prior)
    leaves = [a for a in jax.tree_util.tree_leaves(grads) if eqx.is_inexact_array(a)]
    assert leaves and all(jnp.all(jnp.isfinite(a)) for a in leaves), \
        "widened-g + coeff-reg grad must be finite, with no Sobolev term involved"

    # Two calls with different keys must in general use different (random) g
    # stencils, not one fixed at factory-construction time.
    loss_a = nll(prior, y, cov, dg, d2g, X, jr.key(100))
    loss_b = nll(prior, y, cov, dg, d2g, X, jr.key(101))
    assert loss_a != loss_b, "the g stencil should be redrawn per call, not fixed"


def test_shear_g_train_max_rejects_sobolev():
    try:
        make_nll_loss(64, shear_g_train_max=0.15, sobolev_g1_weight=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("shear_g_train_max + Sobolev should raise ValueError")


if __name__ == "__main__":
    test_bijection_roundtrip_and_identity_at_g0()
    test_shear_derivs_match_autodiff()
    test_shift_nonzero_and_coeffs_shape()
    test_flow_builds_with_perturbative_layer()
    test_widened_g_and_coeff_reg_loss_runs_no_sobolev()
    test_shear_g_train_max_rejects_sobolev()
    print("OK: ShearPerturbative self-checks passed")
