"""
models/shear.py
===============
The shear-conditioning layer of the BFD prior P(m | g, Sigma_X), for the four
even moments m = [Mf, Mr, M1, M2] (Bernstein et al. 2016, MNRAS 459, 4467).

Design
------
Shear does not act on the density; it acts on each *galaxy*, deterministically.
So the layer is the map that carries an unlensed moment vector to its lensed
one, and P(m|g) is the pushforward of the unlensed p(m) through it.  The layer
lives data-adjacent, in raw moment space, so its coefficients ARE dm/dg -- the
quantity `bfd.MomentCalculator.getTemplate` computes exactly (appendix C).

The paper truncates at second order in g (eq. 10, and sec. 5.5: "the lensing is
weak, so that a second-order Taylor expansion about g = 0 fully describes
P(M|g)"), so the layer is a second-order polynomial in g.  Everything else is
fixed by three symmetries of the moments, which hold exactly and are verified
against `bfd` in `tests/test_shear.py`:

**Spin.**  In complex form the moments carry definite spin under a rotation of
the frame by phi: Mf and Mr are spin 0, and e = (M1 + i M2)/Mr is spin 2, as is
g = g1 + i g2 (appendix C, eq. C6-C9).  The unlensed population is isotropic, so
the response must be spin-covariant.  That leaves, through second order, exactly

    spin-0 (each of Mf, Mr):  Re(e* g),  |g|^2,  Re(e*^2 g^2)
    spin-2 (e):               g,  e^2 g*,  e* g^2,  e |g|^2,  e^3 g*^2

**Parity.**  Under y -> -y both e and g conjugate, and the population is
parity-symmetric, so every coefficient above is REAL.  (Equivalently: the
parity-odd partners Im(e*g) etc. are forbidden.)

**Flux.**  Moments are linear in the image, so scaling a galaxy's flux scales
both its moments and their shear derivatives by the same factor.  The eleven
coefficients therefore depend on the two dimensionless invariants

    r = Mr/Mf   (size)        q = |e|^2   (shape)

and on NOTHING else -- verified exact to six digits over a 1000x flux range.
That is why `_Coeffs` is a 2-input network: not a modelling choice, a symmetry.

What the layer is NOT
---------------------
It is not a fit to bfd's dm/dg.  For a real population the shear response is not
a function of m at all: two galaxies with identical moments have different
higher-order structure and respond differently, so regressing a map onto dm/dg
learns only the conditional mean and cannot generalize.  What BFD actually needs
is the *density* P(m|g), and a transport layer trained by likelihood learns
exactly that -- the transport form is what guarantees P stays normalised at every
g (it is the continuity equation), while the fit itself is free to be whatever
matches the sheared population.  `shear.py` trains it that way; `dm_dg` below is
a diagnostic that can be compared to bfd, not the training target.

Direction and invertibility
---------------------------
`transform` (data -> base, "unshear") is the closed-form direction, because it
is the one likelihood training calls.  `inverse` (sampling) solves it by Newton;
the map is O(g) from the identity, so from a cold start the error goes as
|g|^(2^n) and `_NEWTON_STEPS = 4` is exact at g = 0 and at machine precision by
|g| = 0.1.  Log-dets are taken by autodiff of the 4x4 Jacobian.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
from flowjax.bijections import AbstractBijection

from .bijections import CoeffNet

# Newton steps for the data -> base direction.  Error goes as |g|^(2^n) from a
# cold start, so 4 is machine precision over any shear this model is valid for.
_NEWTON_STEPS = 4

# Coefficients are bounded so the map stays a diffeomorphism for any g the
# network might see.  The true values span roughly [-2, 7] (see the ranges
# printed by tests/test_shear.py), so 12 constrains nothing real.
_COEFF_MAX = 12.0

# The two invariants fed to the network, shifted/scaled to O(1).  r = Mr/Mf runs
# over ~[0.7, 4.0] for this weight function and q = |e|^2 over ~[0, 0.3].
_R_LOC, _R_SCALE, _Q_SCALE = 2.5, 1.0, 0.1

N_COEFFS = 11


def _invariants(m):
    """(r, q) and the complex shape e, from raw moments m = [Mf, Mr, M1, M2]."""
    Mf, Mr, M1, M2 = m[0], m[1], m[2], m[3]
    e = jax.lax.complex(M1 / Mr, M2 / Mr)
    return Mr / Mf, (e * jnp.conj(e)).real, e


class _Coeffs(eqx.Module):
    """(r, q) -> the eleven real response coefficients, bounded by _COEFF_MAX."""

    net: CoeffNet

    def __init__(self, key, nn_width, nn_depth, activation):
        self.net = CoeffNet(key, 2, N_COEFFS, nn_width, nn_depth, activation)

    def __call__(self, r, q):
        u = jnp.stack([(r - _R_LOC) / _R_SCALE, q / _Q_SCALE])
        return _COEFF_MAX * jnp.tanh(self.net(u) / _COEFF_MAX)


def response(coeffs, m, g):
    """The equivariant second-order response of `m` to `g`; the layer's core map.

    `coeffs` is the 11-vector [a_f, b_f, c_f, a_r, b_r, c_r, A, B, mu, nu, rho]
    evaluated at `m`'s invariants.  Spin-0 responses are exponentiated so Mf and
    Mr stay positive; to second order in g that only relabels b and c.
    """
    a_f, b_f, c_f, a_r, b_r, c_r, A, B, mu, nu, rho = coeffs
    Mf, Mr = m[0], m[1]
    e = jax.lax.complex(m[2] / Mr, m[3] / Mr)
    gc = jax.lax.complex(g[0], g[1])
    ec, gcc = jnp.conj(e), jnp.conj(gc)

    # The three spin-0 invariants of (e, g), through second order.
    p1 = (ec * gc).real
    p2 = (gc * gcc).real
    p3 = (ec * ec * gc * gc).real

    Mf_s = Mf * jnp.exp(a_f * p1 + b_f * p2 + c_f * p3)
    Mr_s = Mr * jnp.exp(a_r * p1 + b_r * p2 + c_r * p3)
    # Spin-2: an additive response scaled by the UNLENSED Mr, which is how bfd's
    # d(M1 + i M2)/dg is normalised, so A and B are directly comparable to it.
    de = Mr * (A * gc + B * e * e * gcc + mu * ec * gc * gc
               + nu * e * p2 + rho * e * e * e * gcc * gcc)
    return jnp.stack([Mf_s, Mr_s, m[2] + de.real, m[3] + de.imag])


class ShearResponse(AbstractBijection):
    """Second-order shear response of the four even moments, conditioned on g.

    ``transform`` maps data -> base (unshear), ``inverse`` base -> data (shear),
    matching flowjax's convention that a flow wraps its bijection in ``Invert``.
    """

    shape: tuple = (4,)
    cond_shape: tuple = (2,)
    coeffs: _Coeffs

    def __init__(self, key, nn_width=64, nn_depth=2, activation=jnn.silu):
        self.coeffs = _Coeffs(key, nn_width, nn_depth, activation)

    def unshear(self, x, g):
        """Closed form; `transform`'s map, from observed moments back to base."""
        r, q, _ = _invariants(x)
        return response(self.coeffs(r, q), x, g)

    def shear(self, y, g):
        """Solve unshear(x, g) = y for x by Newton, cold-started at the identity."""
        def step(x, _):
            J = jax.jacfwd(self.unshear)(x, g)
            return x - jnp.linalg.solve(J, self.unshear(x, g) - y), None

        x, _ = jax.lax.scan(step, y, None, length=_NEWTON_STEPS)
        return x

    def transform_and_log_det(self, x, condition=None):
        return (self.unshear(x, condition),
                jnp.linalg.slogdet(jax.jacfwd(self.unshear)(x, condition))[1])

    def inverse_and_log_det(self, y, condition=None):
        x = self.shear(y, condition)
        return x, -jnp.linalg.slogdet(jax.jacfwd(self.unshear)(x, condition))[1]


def dm_dg(layer, m):
    """The layer's generative (dm/dg, d2m/dg2) at g = 0, in bfd's column layout.

    Returns ``(2, 4)`` ordered [g1, g2] and ``(3, 4)`` ordered
    [g1g1, g1g2, g2g2] -- directly comparable to the ``dm_dg`` / ``d2m_dg2``
    columns an imsims catalog carries.  This is a *diagnostic*: the layer is
    trained by likelihood on the sheared population, not fitted to these.
    """
    f = lambda g: layer.shear(m, g)
    g0 = jnp.zeros(2)
    first = jax.jacfwd(f)(g0).T                      # (2, 4)
    H = jax.jacfwd(jax.jacfwd(f))(g0)                # (4, 2, 2)
    second = jnp.stack([H[:, 0, 0], H[:, 0, 1], H[:, 1, 1]])
    return first, second
