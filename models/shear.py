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

**Flux.**  Moments are linear in the image, so scaling ONE galaxy's flux scales
both its moments and their shear derivatives by the same factor.  That is exact
and is not in question.

What used to be inferred from it -- that the coefficients must therefore be
flux-blind, and that `_Coeffs` taking three scale-free inputs was "not a
modelling choice" -- does NOT follow, and the flux input was restored on
2026-08-17.  This layer models `E[dm/dg | m]`, a conditional mean over the
population at fixed m.  Two galaxies sharing (r, k, q) at different fluxes are
not rescalings of one another; they are different galaxies that happen to share
three invariants.  Their AVERAGE response is flux-independent only if the
morphology mix at fixed (r, k, q) does not vary with flux -- a property of the
population, not a theorem, and false on real sky, where brighter galaxies are
systematically nearer and of a different type mix.  The old "verified exact to
six digits over a 1000x flux range" check passes trivially on a simulation that
draws flux independently of morphology, so it never tested the assumption it
was cited for.

The coefficients now depend on the flux coordinate z0 as well as the three
scale-free invariants -- in standardised coordinates,

    z0 (flux)   z1 (size)   z2 (concentration)   q = |e|^2 (shape)

which is what `_invariants` returns and what makes `_Coeffs` a 4-input network.
Nothing forces the network to USE z0; if the response really is flux-blind on a
given population it can learn to ignore it, which is the weaker and honest
version of the old claim.

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

from paramax import non_trainable, unwrap

from .bijections import CoeffNet

# Coefficients are bounded so the map stays a diffeomorphism for any g the
# network might see.  The true values span roughly [-2, 7] (see the ranges
# printed by tests/test_shear.py), so 12 constrains nothing real.
_COEFF_MAX = 12.0

# This layer now acts on the STANDARDISED coordinates
#     z = [log10 Mf, logit(Mr/(r* Mf)), logit(Mc/(rc* Mr)), M1/Mr, M2/Mr]
# minus mean over std, i.e. downstream of `RawMomentStandardize` rather than on
# raw moments.  Three things get simpler and one gets harder.
#
#   * z3, z4 ARE M1/Mr and M2/Mr up to the shared spin-2 std, so the complex
#     shape e needs no division by Mr.
#   * z0, z1, z2 are unbounded, so the spin-0 response is ADDITIVE.  The raw
#     version had to exponentiate to keep Mf, Mr, Mc positive; here there is
#     nothing to keep positive and nothing to overflow.
#   * the chart's point-source ceilings live in `RawMomentStandardize`, which is
#     now applied FIRST, so no map downstream of it can push a point off-chart.
#     That removes a g-dependent seam: `bias.log_conv_is` masks draws with
#     `in_domain` before the flow runs, and under the old ordering the chart saw
#     the POST-shear moment while the mask saw the pre-shear one.
#   * what is lost: the coefficients are no longer directly comparable to bfd's
#     dm/dg, because they are responses of z rather than of m.  `dm_dg` below
#     returns dz/dg now; converting needs the standardiser's Jacobian.
#
# The spin structure survives the change untouched, which is what makes it
# cheap: `RawMomentStandardize` keeps the three spin-0 coordinates in slots
# 0, 1, 2 and the spin-2 pair in 3, 4, exactly as the raw vector did.

# z1 and z2 are already standardised, so they need no shift.  q = |e|^2 is a
# chi^2 with two degrees of freedom on standardised spin-2 slots, hence mean 2.
_Q_LOC, _Q_SCALE = 2.0, 2.0

N_COEFFS = 14


def _invariants(z):
    """(f, a, b, q) and the complex shape e, from the STANDARDISED z.

    `a = z1` and `b = z2` are monotone functions of Mr/Mf and Mc/Mr, carrying
    the size and concentration information the raw (r, k) pair did; `q = |e|^2`
    is the same shape invariant.  `f = z0` is the FLUX, and it is now an input.

    It used to be excluded on the grounds that moments are linear in the image,
    so a rescaled galaxy has a rescaled response and the coefficients must be
    flux-blind.  That argument is exact FOR ONE GALAXY RESCALED -- but this
    layer models `E[dm/dg | m]`, a conditional mean over the population at fixed
    m, and two galaxies sharing (a, b, q) at different fluxes are not rescalings
    of one another.  Their average response is flux-independent only if the
    morphology mix at fixed (a, b, q) does not vary with flux, which is a
    property of the population, not a theorem -- and false on real sky, where
    brighter galaxies are systematically nearer and of different type.  The
    "verified exact over a 1000x flux range" check passes trivially on a sim
    that draws flux independently of morphology.
    """
    e = jax.lax.complex(z[3], z[4])
    return z[0], z[1], z[2], (e * jnp.conj(e)).real, e


class _Coeffs(eqx.Module):
    """(r, k, q) -> the fourteen real response coefficients, bounded by _COEFF_MAX."""

    net: CoeffNet

    def __init__(self, key, nn_width, nn_depth, activation):
        self.net = CoeffNet(key, 4, N_COEFFS, nn_width, nn_depth, activation)

    def __call__(self, f, a, b, q):
        u = jnp.stack([f, a, b, (q - _Q_LOC) / _Q_SCALE])
        return _COEFF_MAX * jnp.tanh(self.net(u) / _COEFF_MAX)


def response(coeffs, z, g, e_scale):
    """The equivariant second-order response of `z` to `g`; the layer's core map.

    `coeffs` is the 14-vector [a, b, c] x [z0, z1, z2] followed by
    [A, B, mu, nu, rho], evaluated at `z`'s invariants.

    Both blocks are ADDITIVE.  In raw moment space the spin-0 block had to be
    multiplicative-with-an-exponential to keep Mf, Mr and Mc positive; z is
    unbounded, so the exponential buys nothing and costs conditioning.  The
    spin-2 block loses its factor of Mr for the same kind of reason: z3, z4 are
    already M1/Mr, M2/Mr over the shared spin-2 std, i.e. dimensionless, so the
    response is a pure shift of the shape vector.
    """
    s0, A, B, mu, nu, rho = (coeffs[:9].reshape(3, 3), coeffs[9], coeffs[10],
                             coeffs[11], coeffs[12], coeffs[13])
    # `e_scale` puts the shape back in PHYSICAL units before the polynomial
    # runs.  z3, z4 are M1/Mr, M2/Mr divided by the shared spin-2 std (~0.04),
    # so a standardised |e| runs to ~5 where the physical one stops near 0.5 --
    # and `response` is cubic in e, so using the standardised value would inflate
    # the top term by ~(1/0.04)^3 and destroy the meaning of the coefficient
    # bound.  The response is formed in physical units and divided back out.
    e = e_scale * jax.lax.complex(z[3], z[4])
    gc = jax.lax.complex(g[0], g[1])
    ec, gcc = jnp.conj(e), jnp.conj(gc)

    # The three spin-0 invariants of (e, g), through second order.
    p1 = (ec * gc).real
    p2 = (gc * gcc).real
    p3 = (ec * ec * gc * gc).real

    spin0 = z[:3] + s0 @ jnp.stack([p1, p2, p3])
    de = (A * gc + B * e * e * gcc + mu * ec * gc * gc
          + nu * e * p2 + rho * e * e * e * gcc * gcc) / e_scale
    # SLOT LAYOUT: `RawMomentStandardize` groups the three spin-0 coordinates in
    # slots 0, 1, 2 and the spin-2 pair in 3, 4.  That is NOT the raw layout,
    # where spin-0 is (Mf, Mr, Mc) = slots 0, 1, 4 and spin-2 is (M1, M2) = 2, 3.
    return jnp.stack([spin0[0], spin0[1], spin0[2],
                      z[3] + de.real, z[4] + de.imag])


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
    # The standardiser's spin-2 std, FROZEN -- see `response`.
    e_scale: jax.Array = eqx.field(default=None)

    def __init__(self, key, nn_width=128, nn_depth=3, activation=jnn.silu,
                 cond_dim=2, e_scale=1.0):
        self.coeffs = _Coeffs(key, nn_width, nn_depth, activation)
        self.e_scale = non_trainable(jnp.asarray(e_scale))
        # (2,) alone, or (5,) = [g1, g2, C00, C01, C11] when chained with the
        # centroid layer, which reads the other three.
        self.cond_shape = (cond_dim,)

    def unshear(self, x, condition):
        """Closed form; `transform`'s map, from observed moments back to base.

        Reads g as the FIRST two entries of `condition` -- see `cond_dim`.
        """
        g = condition[:2]
        f, a, b, q, _ = _invariants(x)
        return response(self.coeffs(f, a, b, q), x, g, unwrap(self.e_scale))

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


def dm_dg(layer, m, chart):
    """The layer's generative (dm/dg, d2m/dg2) at g = 0, in bfd's column layout.

    Returns ``(2, 5)`` ordered [g1, g2] and ``(3, 5)`` ordered
    [g1g1, g1g2, g2g2] -- directly comparable to the ``dm_dg`` / ``d2m_dg2``
    columns an imsims catalog carries.

    `chart` is the `RawMomentStandardize` the layer now sits behind, and it is
    NOT optional.  The layer responds in standardised coordinates, so its raw
    g-derivative is dz/dg; what bfd tabulates is dm/dg.  Composing the chart on
    both sides -- raw in, shear in z, raw out -- converts it, and taking the
    g-derivatives of that composite gets the second order right too (a bare
    Jacobian factor would miss the chart's own curvature term).

    This is not only a diagnostic: `shear.train` regresses against it with
    `deriv_weight = 1e4`, so it is the dominant term in the loss and the model's
    one piece of external ground truth.  Feeding the layer raw moments here
    after the reordering would compare z-derivatives against raw truth and train
    the layer to fit noise.
    """
    f = lambda g: chart.inverse(layer.shear(chart.transform(m), g))
    g0 = jnp.zeros(2)
    first = jax.jacfwd(f)(g0).T                      # (2, 5)
    H = jax.jacfwd(jax.jacfwd(f))(g0)                # (5, 2, 2)
    second = jnp.stack([H[:, 0, 0], H[:, 0, 1], H[:, 1, 1]])
    return first, second
