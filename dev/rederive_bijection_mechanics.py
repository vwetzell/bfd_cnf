"""rederive_bijection_mechanics.py
==================================
Independent, from-scratch check of the EarlyChain + Invert + ShearPerturbative
composition, trusting NOTHING from earlier claims in this investigation --
only the literal code and a hand-computable closed-form case.

Setup: base_dist = N(0,I) EXACTLY (no early/bulk layers at all -- early=[]),
last = ShearPerturbative with coefficients HAND-SET (not trained) so the
shift only moves channel 0, by a CONSTANT (position-independent: shift(x,g) =
[c*g1, 0, 0, 0]). This makes the Jacobian d(shift)/dx = 0 EXACTLY, so
log|det(I+J)| = 0 identically -- removing the log-det machinery as a variable
and leaving a fully hand-computable closed form:

    bijection = Invert(EarlyChain([], shear))
    Transformed(N(0,I), bijection).log_prob(m, g)
      = EarlyChain([], shear).transform_and_log_det(m, g) fed to base log_prob
      = shear.transform_and_log_det(m, g) = (m + shift(m,g), 0)   [J=0 here]
      = N(0,I).log_prob(m + [c*g1,0,0,0])
      = -0.5*(m0+c*g1)^2 - 0.5*(m1^2+m2^2+m3^2) + const

By hand: this says, as a function of m0 at FIXED g, the density is a unit
Gaussian centered at m0 = -c*g1 (mode where the exponent vanishes) -- i.e.
shearing by g MOVES THE OBSERVED DISTRIBUTION's mean to -c*g, the OPPOSITE
sign of the encode-direction shift +c*g. If real code disagrees with this by
hand-computation, the composition has a bug; if it agrees, this specific
composition is correct (independent of any earlier claim).

Then: build "observed" data as m0 ~ N(-c*g_true, 1) (matching the model's own
implied shift, so this checks INTERNAL CONSISTENCY of the recover-g pipeline,
not real template physics), extract Q,R via autodiff of log_prob w.r.t. the
CONDITION at g=0 (matching point_estimate_pqr_noiseless.py's pattern, but
independently reimplemented here, not imported), and check g_hat matches
g_true.

Usage:
    cd bfd_cnf
    PYTHONPATH=. python dev/rederive_bijection_mechanics.py
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flowjax.bijections import Invert
from flowjax.distributions import MultivariateNormal, Transformed
from paramax import non_trainable

from bfd_cnf.models.bijections import EarlyChain, ShearPerturbative


def _hand_set_constant_shift_layer(c1: float, c2: float) -> ShearPerturbative:
    """Build a ShearPerturbative whose shift is EXACTLY [c1*g1, c2*g2, 0, 0],
    independent of x (so J=0 identically), by zeroing the coefficient net and
    manually wiring two outputs to constants. Channel 0 responds to g1 only,
    channel 1 responds to g2 only -- g1 AND g2 are both independently
    identifiable (unlike a single-channel version, which leaves g2 with
    exactly zero Fisher information and a singular R_tot -- a degenerate toy
    setup, not a machinery bug, confirmed by hand)."""
    lay = ShearPerturbative(jr.key(0), dim=4, raw_cond_dim=5, last_width=4, last_depth=1,
                             activation=jax.nn.relu)
    net = lay.net_coeffs
    new_layers = []
    for layer in net.layers:
        if hasattr(layer, "weight"):
            layer = eqx.tree_at(lambda l: l.weight, layer, jnp.zeros_like(layer.weight))
            layer = eqx.tree_at(lambda l: l.bias, layer, jnp.zeros_like(layer.bias))
        new_layers.append(layer)
    net = eqx.tree_at(lambda n: n.layers, net, tuple(new_layers))
    # coeffs() output order (see _shift): 5 outputs per channel [A1,A2,B11,B12,B22].
    # index 0 = channel-0's A1 (dchannel0/dg1); index 6 = channel-1's A2 (dchannel1/dg2).
    last_bias = net.layers[-1].bias
    new_bias = last_bias.at[0].set(c1).at[6].set(c2)
    net = eqx.tree_at(lambda n: n.layers[-1].bias, net, new_bias)
    return eqx.tree_at(lambda l: l.net_coeffs, lay, net)


def main() -> None:
    c1, c2 = 0.8, -0.5
    shear = _hand_set_constant_shift_layer(c1, c2)

    # sanity: shift really is [c1*g1, c2*g2, 0, 0], independent of x, J=0
    x_probe = jnp.array([0.3, -0.7, 0.2, -0.4])
    g_probe = jnp.array([0.05, -0.02])
    shift_val = shear._shift(x_probe, g_probe[0], g_probe[1])
    print(f"shift at x={np.asarray(x_probe)}, g={np.asarray(g_probe)}: {np.asarray(shift_val)}")
    print(f"expected [c1*g1,c2*g2,0,0] = [{c1*float(g_probe[0])}, {c2*float(g_probe[1])}, 0, 0]")
    print(f"dshift/dx (should be all zero): max|.| = {float(jnp.max(jnp.abs(jax.jacfwd(lambda xx: shear._shift(xx, g_probe[0], g_probe[1]))(x_probe))))}")

    base = non_trainable(MultivariateNormal(jnp.zeros(4), jnp.eye(4)))
    bijection = Invert(EarlyChain([], shear))
    prior = Transformed(base, bijection)

    # hand-computed closed form:
    # log_prob(m,g) = -0.5*(m0+c1*g1)^2 -0.5*(m1+c2*g2)^2 -0.5*(m2^2+m3^2) + const
    def hand_log_prob(m, g):
        d = 4
        norm = -0.5 * d * jnp.log(2 * jnp.pi)
        return (norm - 0.5 * (m[0] + c1 * g[0]) ** 2 - 0.5 * (m[1] + c2 * g[1]) ** 2
                - 0.5 * (m[2] ** 2 + m[3] ** 2))

    m_test = jnp.array([0.4, 0.1, -0.2, 0.3])
    g_test = jnp.array([0.03, -0.01])
    cond = jnp.concatenate([g_test, jnp.zeros(3)])
    code_lp = float(prior.log_prob(m_test, condition=cond))
    hand_lp = float(hand_log_prob(m_test, g_test))
    print(f"\nlog_prob from CODE: {code_lp:.6f}")
    print(f"log_prob HAND-COMPUTED: {hand_lp:.6f}")
    print(f"match: {abs(code_lp-hand_lp) < 1e-4}")

    # Now the full g-recovery pipeline, fully independent implementation here.
    g_true = jnp.array([0.02, 0.01])
    rng = np.random.default_rng(0)
    n = 200_000
    m0 = rng.normal(-c1 * float(g_true[0]), 1.0, size=n)  # matches the model's own implied shift
    m1 = rng.normal(-c2 * float(g_true[1]), 1.0, size=n)
    m2 = rng.normal(0, 1.0, size=n)
    m3 = rng.normal(0, 1.0, size=n)
    m_obs = jnp.asarray(np.stack([m0, m1, m2, m3], axis=-1))

    def logp_of_g(g, m_i):
        cond = jnp.concatenate([g, jnp.zeros(3)])
        return prior.log_prob(m_i, condition=cond)

    def per_example(m_i):
        f = lambda g: logp_of_g(g, m_i)
        Q = jax.grad(f)(jnp.zeros(2))
        R = -jax.hessian(f)(jnp.zeros(2))
        return Q, R

    Q, R = jax.vmap(per_example)(m_obs)
    Q_tot, R_tot = jnp.mean(Q, axis=0), jnp.mean(R, axis=0)
    g_hat = jnp.linalg.solve(R_tot, Q_tot)
    print(f"\ng_true = {np.asarray(g_true)}")
    print(f"g_hat  = {np.asarray(g_hat)}")
    print(f"R_tot eigenvalues (should be positive): {np.linalg.eigvalsh(np.asarray(R_tot))}")


if __name__ == "__main__":
    main()
