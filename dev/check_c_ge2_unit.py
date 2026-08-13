"""check_c_ge2_unit.py
======================
Isolated unit test: force ShearPerturbative's coefficient net to output a
CONSTANT vector with ONLY c_ge2 nonzero (all others zero), then check that
shear_derivs_generative's implied (dM1,dM2)/(dg1,dg2) Jacobian exactly
matches the hand-derived closed form for that term alone:

    J = c_ge2 * [[p, q], [q, -p]],  p = e1^2-e2^2, q = 2*e1*e2

If this fails, there's a genuine formula/sign bug in the c_ge2 term itself,
independent of anything training does. If it passes, the c_ge2 FORMULA is
correct and the training-time failure is something else (optimization
landscape / identifiability specific to that term, or an interaction with
other parts of the loss).

Usage:
    cd bfd_cnf
    PYTHONPATH=. python dev/check_c_ge2_unit.py
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bfd_cnf.models.bijections import ShearPerturbative


def _const_coeffs_layer(coeff_values):
    """Build a ShearPerturbative whose net_coeffs outputs a FIXED constant
    vector `coeff_values` (10,) regardless of input, by zeroing all weights
    and setting the last layer's bias directly."""
    lay = ShearPerturbative(jr.key(0), dim=4, raw_cond_dim=5, last_width=8, last_depth=2,
                             activation=jax.nn.silu)
    net = lay.net_coeffs
    # zero every weight (so output doesn't depend on input at all)
    new_layers = []
    for layer in net.layers:
        if hasattr(layer, "weight"):
            layer = eqx.tree_at(lambda l: l.weight, layer, jnp.zeros_like(layer.weight))
            layer = eqx.tree_at(lambda l: l.bias, layer, jnp.zeros_like(layer.bias))
        new_layers.append(layer)
    net = eqx.tree_at(lambda n: n.layers, net, tuple(new_layers))
    # set the FINAL layer's bias to the desired constant output
    last_linear = net.layers[-1]
    net = eqx.tree_at(
        lambda n: n.layers[-1].bias, net,
        jnp.asarray(coeff_values, dtype=last_linear.bias.dtype),
    )
    return eqx.tree_at(lambda l: l.net_coeffs, lay, net)


def main() -> None:
    # coeffs order: cA_f,c0_f,c1_f, cA_s,c0_s,c1_s, c_g,c_ge2,c_g2,c_g2b
    c_ge2_val = 0.37
    coeffs = jnp.array([0., 0., 0., 0., 0., 0., 0., c_ge2_val, 0., 0.])
    lay = _const_coeffs_layer(coeffs)

    # sanity: does the net really output the constant we set, at a nontrivial point?
    x0, x1, e1, e2 = 0.3, -0.5, 0.21, -0.14
    out = lay.coeffs(x0, x1, e1, e2)
    print("coeffs() output:", [float(v) for v in out])
    print(f"expected c_ge2 (raw, before /s^2 rescale) = {c_ge2_val}")

    x = jnp.array([x0, x1, e1, e2])
    A, B = lay.shear_derivs(x)          # encode-direction (x + shift(x,g))
    A_gen, B_gen = lay.shear_derivs_generative(x)

    p, q = e1 * e1 - e2 * e2, 2.0 * e1 * e2
    J_expected_encode = c_ge2_val * jnp.array([[p, q], [q, -p]])
    J_expected_gen = -J_expected_encode  # A_gen = -A_encode (shift vanishes at g=0 identically)

    print(f"\nEncode-direction (M1,M2) Jacobian (rows=M1,M2; cols=g1,g2):")
    print("  A[2:4,:] =", np.asarray(A[2:4, :]))
    print("  expected =", np.asarray(J_expected_encode))

    print(f"\nGenerative-direction (M1,M2) Jacobian:")
    print("  A_gen[2:4,:] =", np.asarray(A_gen[2:4, :]))
    print("  expected     =", np.asarray(J_expected_gen))

    match_encode = np.allclose(np.asarray(A[2:4, :]), np.asarray(J_expected_encode), atol=1e-5)
    match_gen = np.allclose(np.asarray(A_gen[2:4, :]), np.asarray(J_expected_gen), atol=1e-5)
    print(f"\nEncode formula matches hand-derivation: {match_encode}")
    print(f"Generative formula matches hand-derivation: {match_gen}")


if __name__ == "__main__":
    main()
