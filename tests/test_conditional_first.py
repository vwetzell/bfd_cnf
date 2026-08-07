"""EarlyChain (always data-adjacent): must be an exact bijection, and the
analytic (shear + Σ_X) generative response must match whole-flow autodiff.

This fails if the transform/inverse loops in EarlyChain get out of sync, or if
ShearTaylorLast.shear_derivs_generative / SigmaXCouplingLayer's closed-form
composition in ``_shear_response`` drifts from what a whole-flow autodiff
round-trip actually computes. Since the shear/Σ_X layer swap, Σ_X (not
ShearTaylorLast) is the data-adjacent layer, so the analytic shortcut is
``_shear_response`` (which chains the shear layer's local response through Σ_X's
own local Jacobian/Hessian) rather than ``ShearTaylorLast.shear_derivs_generative``
alone.
"""
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from bfd_cnf.models.flows import (
    build_flows,
    _flow_shear_derivs,
    _shear_response,
    _find_shear_taylor_layer,
    _find_sigmax_layer,
)


def _round_trip():
    prior, _ = build_flows(
        jax.random.PRNGKey(0), latent_dim=4, cond_dim=16, prior_size_loc_c1=0.0,
    )
    bij = prior.bijection
    cond = jnp.array([0.02, -0.01, 12.0, 0.05, -0.03])  # [g1,g2,log_scale,e1,e2]
    z = jax.random.normal(jax.random.PRNGKey(1), (4,))
    x, ld_fwd = bij.transform_and_log_det(z, cond)          # base -> data
    z2, ld_inv = bij.inverse_and_log_det(x, cond)           # data -> base
    return z, z2, ld_fwd, ld_inv, prior, cond


def test_round_trip():
    z, z2, ld_fwd, ld_inv, prior, cond = _round_trip()
    assert np.allclose(z, z2, atol=1e-4), "encode(decode(z)) != z"
    assert np.allclose(ld_fwd, -ld_inv, atol=1e-4), "log-dets disagree"
    lp = prior.log_prob(prior.bijection.transform(z, cond), cond)
    assert np.isfinite(lp), "non-finite log_prob"


def _perturbed():
    """A taylor flow pushed off the g=0 identity so A,B != 0."""
    prior, _ = build_flows(
        jax.random.PRNGKey(0), latent_dim=4, cond_dim=16, prior_size_loc_c1=0.0,
    )
    params, static = eqx.partition(prior, eqx.is_inexact_array)
    leaves, tdef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(3), len(leaves))
    leaves = [l + 0.1 * jax.random.normal(k, l.shape) for l, k in zip(leaves, keys)]
    return eqx.combine(jax.tree_util.tree_unflatten(tdef, leaves), static)


def test_analytic_generative_derivs_match_autodiff():
    # _shear_response (shear layer's local response chained through Σ_X's own
    # local Jacobian/Hessian) must equal the whole-flow autodiff response EXACTLY
    # at 1st order. No 2nd-order (B) check here: ShearTaylorLast.inverse_and_log_det's
    # spin-2 block uses a closed-form SUBSTITUTION inverse (evaluate the shift at y
    # instead of solving for the unknown x -- see its own docstring), which matches
    # the true implicit inverse's g-Taylor series only through 1st order -- its OWN
    # 2nd derivative at g=0 already differs from shear_derivs_generative's exact
    # implicit-function-theorem formula in complete isolation, with no Σ_X layer
    # involved at all, by an amount that depends on the coeff-net weights and even
    # on ambient float32/float64 config (pre-existing, predates the shear/Σ_X swap;
    # not a fixed constant, so no atol here is both tight and robust). See
    # test_sigmax_chain_rule_composition for the floor-free B check that actually
    # isolates _shear_response's (new) composition logic.
    p = _perturbed()
    sx_ref = jnp.array([12.0, 0.03, -0.04])
    m0 = jax.random.normal(jax.random.PRNGKey(7), (5, 4))
    A_an, _ = _shear_response(p, m0, sx_ref)
    A_fl, _ = _flow_shear_derivs(p, m0, sx_ref)
    assert np.allclose(A_an, A_fl, atol=1e-4), "1st-order analytic != autodiff"


def test_sigmax_chain_rule_composition():
    """_shear_response's 2nd-order chain-rule composition through Σ_X's own local
    Jacobian/Hessian is correct, decoupled from ShearTaylorLast's own known
    approximate-inverse floor (see test_analytic_generative_derivs_match_autodiff).

    Reconstructs the shear layer's local decode-direction displacement as an
    EXACT quadratic Taylor model in g (using shear_derivs_generative's own A, B
    as its coefficients by construction, bypassing ShearTaylorLast.inverse_and_
    log_det's substitution-inverse entirely) and composes it through Σ_X's
    EXACT closed-form inverse (SigmaXCouplingLayer has no such approximation --
    see test_sigmax_layer.py's roundtrip test). Plain autodiff of that
    reconstruction must match _shear_response's analytic composition tightly.
    """
    p = _perturbed()
    shear = _find_shear_taylor_layer(p)
    sigmax = _find_sigmax_layer(p)
    sx_ref = jnp.array([12.0, 0.03, -0.04])
    sx_cond_full = jnp.concatenate([jnp.zeros(2), sx_ref])
    m0 = jax.random.normal(jax.random.PRNGKey(7), (5, 4))

    def _one(m0_i):
        v0, _ = sigmax.transform_and_log_det(m0_i, sx_cond_full)
        A_shear, B_shear = shear.shear_derivs_generative(v0)

        def v_of_g(g):
            return (
                v0
                + jnp.einsum("ia,a->i", A_shear, g)
                + 0.5 * jnp.einsum("iab,a,b->i", B_shear, g, g)
            )

        def m_of_g(g):
            cond = jnp.concatenate([g, sx_ref])
            return sigmax.inverse_and_log_det(v_of_g(g), cond)[0]

        A_ref = jax.jacfwd(m_of_g)(jnp.zeros(2))
        B_ref = jax.jacfwd(jax.jacfwd(m_of_g))(jnp.zeros(2))
        return A_ref, B_ref

    A_ref, B_ref = jax.vmap(_one)(m0)
    A_an, B_an = _shear_response(p, m0, sx_ref)
    # float32 + nested jacfwd/vmap reduction-order noise sits close to 1e-4; a real
    # composition bug (verified during development) shows up at the 0.01-0.1 scale.
    assert np.allclose(A_an, A_ref, atol=5e-4), "1st-order composition != reconstruction"
    assert np.allclose(B_an, B_ref, atol=5e-3), "2nd-order composition != reconstruction"


if __name__ == "__main__":
    test_round_trip()
    test_analytic_generative_derivs_match_autodiff()
    test_sigmax_chain_rule_composition()
    print("ok: round-trip + analytic==autodiff + Σ_X composition verified")
