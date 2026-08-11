"""
models/shear.py
===============
The shear-conditioning layer of the BFD prior P(m | g, Sigma_X), for the five
even moments m = [Mf, Mr, M1, M2, Mc] (Bernstein et al. 2016, MNRAS 459, 4467).

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
the frame by phi: Mf, Mr and Mc are spin 0, and e = (M1 + i M2)/Mr is spin 2, as is
g = g1 + i g2 (appendix C, eq. C6-C9).  The unlensed population is isotropic, so
the response must be spin-covariant.  That leaves, through second order, exactly

    spin-0 (each of Mf, Mr, Mc):  Re(e* g),  |g|^2,  Re(e*^2 g^2)
    spin-2 (e):                   g,  e^2 g*,  e* g^2,  e |g|^2,  e^3 g*^2

Mc, the k^4 concentration moment, is spin 0 and so takes the same three
structures as Mf and Mr -- 14 coefficients in total, no new symmetry analysis.
It is in the vector because the shear response of a non-Gaussian galaxy is not
determined by the other four: on a bulge+disc population the residual scatter of
d(M1,M2)/dg at fixed [Mr/Mf, |e|^2] is 8.1%, and adding Mc drops it to 2.0%
(paper sec. 6.4 suggests exactly this).

**Parity.**  Under y -> -y both e and g conjugate, and the population is
parity-symmetric, so every coefficient above is REAL.  (Equivalently: the
parity-odd partners Im(e*g) etc. are forbidden.)

**Flux.**  Moments are linear in the image, so scaling a galaxy's flux scales
both its moments and their shear derivatives by the same factor.  The fourteen
coefficients therefore depend on the three dimensionless invariants

    r = Mr/Mf   (size)     k = Mc Mf/Mr^2  (concentration)     q = |e|^2  (shape)

The concentration enters as the dimensionless k = Mc Mf/Mr^2 rather than the
flow coordinate Mc/Mr, purely for conditioning: Mc/Mr correlates with Mr/Mf at
0.999, so the two would hand the network a nearly degenerate pair and bury the
information Mc actually adds.  k is the part of Mc that Mr/Mf does not predict,
and it is exactly 1/2 for a point source under any weight function.

and on NOTHING else -- verified exact to six digits over a 1000x flux range.
That is why `_Coeffs` is a 3-input network: the inputs are the complete set of
flux-blind spin-0 invariants, not a modelling choice.

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
is the one likelihood training calls.  `inverse` (sampling) is closed form too:
the map is the identity at g = 0, so its g-series inverts term by term and
`shear` evaluates that inverted series directly.  Nothing here iterates.

The price is that the two directions are exact inverses only through O(g^2) --
which is every order this layer claims to model, `response` being a second-order
shear response by construction.  Beyond that the round trip drifts; see
`tests/test_shear.py::test_bijection_round_trip` for the measured residual
versus |g|.  Log-dets are taken by autodiff of the 5x5 Jacobian.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
from flowjax.bijections import AbstractBijection

from .bijections import CoeffNet

# Coefficients are bounded so the map stays a diffeomorphism for any g the
# network might see.  The true values span roughly [-2, 7] (see the ranges
# printed by tests/test_shear.py), so 12 constrains nothing real.
_COEFF_MAX = 12.0

# The invariants fed to the network, shifted/scaled to O(1).  For this weight
# function r = Mr/Mf runs over ~[0.7, 4.0], the concentration k = Mc Mf/Mr^2
# over ~[1.85, 2.4], and q = |e|^2 over ~[0, 0.3].
_R_LOC, _R_SCALE, _Q_SCALE = 2.5, 1.0, 0.1
_K_LOC, _K_SCALE = 2.0, 0.2

N_COEFFS = 14


def _invariants(m):
    """(r, k, q) and the complex shape e, from m = [Mf, Mr, M1, M2, Mc]."""
    Mf, Mr, M1, M2, Mc = m[0], m[1], m[2], m[3], m[4]
    e = jax.lax.complex(M1 / Mr, M2 / Mr)
    return Mr / Mf, Mc * Mf / (Mr * Mr), (e * jnp.conj(e)).real, e


class _Coeffs(eqx.Module):
    """(r, k, q) -> the fourteen real response coefficients, bounded by _COEFF_MAX."""

    net: CoeffNet

    def __init__(self, key, nn_width, nn_depth, activation):
        self.net = CoeffNet(key, 3, N_COEFFS, nn_width, nn_depth, activation)

    def __call__(self, r, k, q):
        u = jnp.stack([(r - _R_LOC) / _R_SCALE, (k - _K_LOC) / _K_SCALE,
                       q / _Q_SCALE])
        return _COEFF_MAX * jnp.tanh(self.net(u) / _COEFF_MAX)


def response(coeffs, m, g):
    """The equivariant second-order response of `m` to `g`; the layer's core map.

    `coeffs` is the 14-vector [a, b, c] x [Mf, Mr, Mc] followed by
    [A, B, mu, nu, rho], evaluated at `m`'s invariants.  Spin-0 responses are
    exponentiated so Mf, Mr and Mc stay positive; to second order in g that only
    relabels b and c.
    """
    s0, A, B, mu, nu, rho = (coeffs[:9].reshape(3, 3), coeffs[9], coeffs[10],
                             coeffs[11], coeffs[12], coeffs[13])
    Mf, Mr, Mc = m[0], m[1], m[4]
    e = jax.lax.complex(m[2] / Mr, m[3] / Mr)
    gc = jax.lax.complex(g[0], g[1])
    ec, gcc = jnp.conj(e), jnp.conj(gc)

    # The three spin-0 invariants of (e, g), through second order.
    p1 = (ec * gc).real
    p2 = (gc * gcc).real
    p3 = (ec * ec * gc * gc).real

    spin0 = jnp.stack([Mf, Mr, Mc]) * jnp.exp(s0 @ jnp.stack([p1, p2, p3]))
    # Spin-2: an additive response scaled by the UNLENSED Mr, which is how bfd's
    # d(M1 + i M2)/dg is normalised, so A and B are directly comparable to it.
    de = Mr * (A * gc + B * e * e * gcc + mu * ec * gc * gc
               + nu * e * p2 + rho * e * e * e * gcc * gcc)
    return jnp.stack([spin0[0], spin0[1], m[2] + de.real, m[3] + de.imag,
                      spin0[2]])


class ShearResponse(AbstractBijection):
    """Second-order shear response of the five even moments, conditioned on g.

    ``transform`` maps data -> base (unshear), ``inverse`` base -> data (shear),
    matching flowjax's convention that a flow wraps its bijection in ``Invert``.
    """

    shape: tuple = (5,)
    # Static: it is metadata, not a parameter.  Assigning a plain field in
    # __init__ would make its contents pytree LEAVES, which changes the
    # serialised leaf count and breaks every existing checkpoint.
    cond_shape: tuple = eqx.field(static=True, default=(2,))
    coeffs: _Coeffs

    def __init__(self, key, nn_width=128, nn_depth=3, activation=jnn.silu,
                 cond_dim=2):
        self.coeffs = _Coeffs(key, nn_width, nn_depth, activation)
        # (2,) alone, or (5,) = [g1, g2, C00, C01, C11] when chained with the
        # centroid layer, which reads the other three.
        self.cond_shape = (cond_dim,)

    def unshear(self, x, condition):
        """Closed form; `transform`'s map, from observed moments back to base.

        Reads g as the FIRST two entries of `condition` -- see `cond_dim`.
        """
        g = condition[:2]
        r, k, q, _ = _invariants(x)
        return response(self.coeffs(r, k, q), x, g)

    def shear(self, y, condition):
        """Invert `unshear` in closed form, by inverting its g-series.

        `unshear` is the identity at g = 0, so write it as
        U(x, g) = x + A(x).g + (1/2) g.B(x).g + O(g^3) and solve
        U(X(y, g), g) = y order by order:

            X = y - A.g + (1/2) g.[ (C + C^T) - B ].g,   C_iab = d_j A_ia A_jb

        with A, B and dA/dx all evaluated at y.  No iteration, no solve, no
        convergence criterion -- three nested jacfwd calls of fixed cost.

        The residual is O(g^3), which is the order at which `response` stops
        modelling anything: the layer IS a second-order shear response, so
        inverting it past second order would be inverting its truncation error.
        Exact at g = 0, where the map is the identity in both directions and
        where every proposal this flow samples for is drawn.  What it costs is
        exact invertibility at g != 0 -- see tests/test_shear.py for the
        measured round-trip residual, and note that a proposal drawn through
        this path is importance-weighted by a log_prob that uses the exact
        `unshear`, so the residual biases nothing, it only costs ESS.
        """
        g = condition[:2]
        zero = jnp.zeros(2)
        A = lambda x: jax.jacfwd(self.unshear, argnums=1)(x, zero)
        a = A(y)                                                # (5, 2)
        b = jax.jacfwd(A)(y)                                    # (5, 2, 5)
        B = jax.jacfwd(jax.jacfwd(self.unshear, argnums=1), argnums=1)(y, zero)
        C = jnp.einsum("iaj,jb->iab", b, a)                     # (5, 2, 2)
        quad = C + C.transpose(0, 2, 1) - B
        return y - a @ g + 0.5 * jnp.einsum("iab,a,b->i", quad, g, g)

    def transform_and_log_det(self, x, condition=None):
        return (self.unshear(x, condition),
                jnp.linalg.slogdet(jax.jacfwd(self.unshear)(x, condition))[1])

    def inverse_and_log_det(self, y, condition=None):
        x = self.shear(y, condition)
        return x, -jnp.linalg.slogdet(jax.jacfwd(self.unshear)(x, condition))[1]


def dm_dg(layer, m):
    """The layer's generative (dm/dg, d2m/dg2) at g = 0, in bfd's column layout.

    Returns ``(2, 5)`` ordered [g1, g2] and ``(3, 5)`` ordered
    [g1g1, g1g2, g2g2] -- directly comparable to the ``dm_dg`` / ``d2m_dg2``
    columns an imsims catalog carries.  This is a *diagnostic*: the layer is
    trained by likelihood on the sheared population, not fitted to these.
    """
    f = lambda g: layer.shear(m, g)
    g0 = jnp.zeros(2)
    first = jax.jacfwd(f)(g0).T                      # (2, 5)
    H = jax.jacfwd(jax.jacfwd(f))(g0)                # (5, 2, 2)
    second = jnp.stack([H[:, 0, 0], H[:, 0, 1], H[:, 1, 1]])
    return first, second
