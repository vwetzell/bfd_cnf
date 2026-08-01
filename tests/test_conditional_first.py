"""EarlyChain.conditional_first: both orders must be exact bijections.

The flip only reorders composition, so encode(decode(z)) == z and the change-of-
variables log_prob must hold in either order. This fails if the transform/inverse
loops in EarlyChain.conditional_first=True get out of sync.
"""
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from bfd_cnf.models.flows import build_flows, _flow_shear_derivs
from bfd_cnf.models.bijections import ShearTaylorLast


def _round_trip(conditional_first):
    prior, _ = build_flows(
        jax.random.PRNGKey(0), latent_dim=4, cond_dim=16,
        prior_size_loc_c1=0.0, prior_conditional_first=conditional_first,
    )
    bij = prior.bijection
    cond = jnp.array([0.02, -0.01, 12.0, 0.05, -0.03])  # [g1,g2,log_scale,e1,e2]
    z = jax.random.normal(jax.random.PRNGKey(1), (4,))
    x, ld_fwd = bij.transform_and_log_det(z, cond)          # base -> data
    z2, ld_inv = bij.inverse_and_log_det(x, cond)           # data -> base
    return z, z2, ld_fwd, ld_inv, prior, cond


def test_both_orders_round_trip():
    for cf in (False, True):
        z, z2, ld_fwd, ld_inv, prior, cond = _round_trip(cf)
        assert np.allclose(z, z2, atol=1e-4), f"cf={cf}: encode(decode(z)) != z"
        assert np.allclose(ld_fwd, -ld_inv, atol=1e-4), f"cf={cf}: log-dets disagree"
        lp = prior.log_prob(prior.bijection.transform(z, cond), cond)
        assert np.isfinite(lp), f"cf={cf}: non-finite log_prob"


def _perturbed(cf):
    """A taylor+conditional_first flow pushed off the g=0 identity so A,B != 0."""
    prior, _ = build_flows(
        jax.random.PRNGKey(0), latent_dim=4, cond_dim=16, prior_size_loc_c1=0.0,
        shear_layer_kind="taylor", prior_conditional_first=cf,
    )
    params, static = eqx.partition(prior, eqx.is_inexact_array)
    leaves, tdef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(3), len(leaves))
    leaves = [l + 0.1 * jax.random.normal(k, l.shape) for l, k in zip(leaves, keys)]
    return eqx.combine(jax.tree_util.tree_unflatten(tdef, leaves), static)


def test_analytic_generative_derivs_match_autodiff_when_data_adjacent():
    # The analytic layer response equals the whole-flow autodiff response EXACTLY
    # only when the shear layer is data-adjacent (conditional_first=True).
    p = _perturbed(cf=True)
    layer = next(
        n for n in jax.tree_util.tree_leaves(
            p, is_leaf=lambda x: isinstance(x, ShearTaylorLast))
        if isinstance(n, ShearTaylorLast)
    )
    m0 = jax.random.normal(jax.random.PRNGKey(7), (5, 4))
    A_an, B_an = jax.vmap(layer.shear_derivs_generative)(m0)
    A_fl, B_fl = _flow_shear_derivs(p, m0, jnp.array([12.0, 0.03, -0.04]))
    assert np.allclose(A_an, A_fl, atol=1e-4), "1st-order analytic != autodiff"
    assert np.allclose(B_an, B_fl, atol=1e-3), "2nd-order analytic != autodiff"


def test_own_e_is_an_exact_bijection():
    # own_e adds a 2nd-stage M1 shift conditioned on the final M2; the spin-2 Jacobian
    # block still has det 1, so the layer must stay an exact, log-det-0 bijection.
    prior, _ = build_flows(
        jax.random.PRNGKey(0), latent_dim=4, cond_dim=16, prior_size_loc_c1=0.0,
        shear_layer_kind="taylor", prior_conditional_first=True, prior_shear_own_e=True,
    )
    # perturb off the identity so the own-|e| net is actually active
    params, static = eqx.partition(prior, eqx.is_inexact_array)
    leaves, tdef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(5), len(leaves))
    prior = eqx.combine(
        jax.tree_util.tree_unflatten(
            tdef, [l + 0.1 * jax.random.normal(k, l.shape) for l, k in zip(leaves, keys)]
        ),
        static,
    )
    bij = prior.bijection
    cond = jnp.array([0.02, -0.01, 12.0, 0.05, -0.03])
    z = jax.random.normal(jax.random.PRNGKey(1), (4,))
    x, ld_fwd = bij.transform_and_log_det(z, cond)
    z2, ld_inv = bij.inverse_and_log_det(x, cond)
    assert np.allclose(z, z2, atol=1e-4), "own_e: encode(decode(z)) != z (not invertible)"
    assert np.allclose(ld_fwd, -ld_inv, atol=1e-4), "own_e: log-dets disagree"
    assert np.isfinite(prior.log_prob(bij.transform(z, cond), cond)), "own_e: bad log_prob"


if __name__ == "__main__":
    test_both_orders_round_trip()
    test_analytic_generative_derivs_match_autodiff_when_data_adjacent()
    test_own_e_is_an_exact_bijection()
    print("ok: round-trip + analytic==autodiff + own_e invertibility verified")
