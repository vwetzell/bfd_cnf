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
# network might see.
#
# THE BOUND IS MIS-SIZED FOR THIS CHART, measured 2026-08-20 by
# `dev/flexibility_audit.py` and the tables in its docstring.  12 was sized on
# the pre-16ece5c RAW parameterisation, where the true spin-0 coefficients
# really do span [-2, 7] -- inverting bfd's dm/dg as dX/dg = X c_X Re(ebar.g)
# gives c_Mf 0.98 -> 2.11, c_Mr 0.84 -> 3.63, c_Mc 0.16 -> 4.74 across the whole
# population.  But 16ece5c moved the layer BEHIND the chart, and z1 =
# standardised logit(Mr/(r* Mf)) has a Jacobian that diverges at the
# point-source ceiling.  The identical physics, expressed in z, needs
#
#     |a_Mr| median 32, p99 253, above 12 for 76.5% of templates
#     |a_Mc| median 28, p99 257, above 12 for 70.5%
#
# so the bound forbids the true first-order response over three quarters of the
# resolution axis, and the trained network duly sits pinned at it: 61% of
# templates for a_Mr, 53% for a_Mc, 50-74% for p2_Mc, mu and nu.
#
# The claim this comment used to make -- "|a1| p99 = 6.4, comfortably inside 12,
# clipping destroys 0.94% of the response" -- came from
# `dev/physical_steepness.py`, whose weighted polynomial smoothing underfits (it
# also reports the physical steepness as 0.00, against ~15 from a binned
# estimate of the same quantity).  Ratio-of-sums conditional means, which cannot
# underfit, are what the numbers above use.
#
# Raising the bound is NOT the fix on its own: `dev/combined_test.py` paired 5
# seeds at 12 against 100 and spin-2 alpha fell 0.95 -> 0.44 with triple the
# seed spread, because a larger bound also unleashes the small-|e| direction
# where a_i = (dz_i/dg . e)/(e_scale |e|^2) is pure Var[Q|m] noise.  Size the
# bound from the SMOOTHED coefficient, never per-template.
#
# FIXED by not asking the network for a divergent function at all: its spin-0
# outputs are the RAW coefficients now and `_chart_spin0_jac` supplies the
# chart's Jacobian analytically, so 12 is back to bounding a quantity whose
# physical range is [-2, 7].  The paragraphs above are kept because the trap is
# easy to walk back into -- read them before touching this constant.
_COEFF_MAX = 12.0

# Radius of the training shear disc, and the operating point of the bias
# measurement.  It lives here rather than in shear.py because
# models/centroid.py needs it too, to normalise its g invariants, and a
# layer must not import the training script.
G_MAX = 0.02

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
#   * what was lost and then recovered: for a while the coefficients were
#     responses of z rather than of m, so they stopped being comparable to
#     bfd's dm/dg -- and worse, z1's diverging logit Jacobian made them
#     divergent too.  The spin-0 nine are raw coefficients again, with that
#     Jacobian applied analytically by `_chart_spin0_jac`.  `dm_dg` below still
#     returns the layer's dm/dg by composing the chart on both sides.
#
# The spin structure survives the change untouched, which is what makes it
# cheap: `RawMomentStandardize` keeps the three spin-0 coordinates in slots
# 0, 1, 2 and the spin-2 pair in 3, 4, exactly as the raw vector did.

# z1 and z2 are already standardised, so they need no shift.  q = |e|^2 is a
# chi^2 with two degrees of freedom on standardised spin-2 slots, hence mean 2.
_Q_LOC, _Q_SCALE = 2.0, 2.0

N_COEFFS = 14

# The chart's spin-0 Jacobian is applied ANALYTICALLY, so the network's spin-0
# outputs are raw-moment response coefficients again.
#
# Since 16ece5c the layer acts on z, and z1 = (logit(Mr/(r* Mf)) - mu)/sd has a
# Jacobian that diverges at the point-source ceiling.  Writing the raw response
# as dX/dg = X c_X Re(ebar.g) for X in (Mf, Mr, Mc) and pushing it through the
# chart gives
#
#     dz0/dg = c_Mf / (sd0 ln10)               . p1
#     dz1/dg = (c_Mr - c_Mf) / (sd1 (1 - u))   . p1,   u = Mr/(r* Mf)
#     dz2/dg = (c_Mc - c_Mr) / (sd2 (1 - v))   . p1,   v = Mc/(rc* Mr)
#
# i.e. M = D L with L the difference matrix and D the diagonal above.  Asking
# the network for the left-hand side is what broke the layer: measured on the
# bulgedisc catalog the physical c stay inside [-2, 7] (c_Mf 0.98 -> 2.11, c_Mr
# 0.84 -> 3.63, c_Mc 0.16 -> 4.74) while the same physics in z needs |a_Mr|
# median 32 and p99 253, above `_COEFF_MAX` for 76.5% of templates -- so the
# bound forbade the true response over three quarters of the resolution axis,
# and the fitted net sat pinned at it for 61% of them.  It also made the target
# STEEP: a_Mr climbs 0.84 -> 61 along z1 where c_Mr climbs 0.84 -> 3.63, and
# that steepness is what buys the unphysical log-det
# (`dev/flexibility_audit.py`: even part 0.509 nats against a physical 0.00033).
#
# `M` is lower triangular with nonzero diagonal, so this is a change of
# coordinates on coefficient space -- nothing representable is lost, the target
# function is just O(1) and flat now.  Its (-1, +1) rows also decorrelate z1 and
# z2, which the whitening in `_Coeffs` could only partly fix: those two are
# correlated at 0.997, and it was their DIFFERENCE that carried the physics.
#
# u -> 1 is a real divergence of the coordinate, not an artifact, but a draw can
# land arbitrarily close to the ceiling where the population never goes, so the
# logit is clipped.  6.9 is u = 0.999; the bulgedisc catalog tops out at 0.975.
_LOGIT_MAX = 6.9


def _chart_spin0_jac(z, loc, scale):
    """dz_i/d(raw log X) for the three spin-0 slots: the M above, shape (3, 3)."""
    # 1/(1 - u) = 1 + exp(logit u), and slots 1 and 2 carry logit u, logit v.
    logit = jnp.clip(scale[1:3] * z[1:3] + loc[1:3], -_LOGIT_MAX, _LOGIT_MAX)
    d1, d2 = (1.0 + jnp.exp(logit)) / scale[1:3]
    d0 = 1.0 / (scale[0] * jnp.log(10.0))
    zero = jnp.zeros_like(d0)
    return jnp.stack([jnp.stack([d0, zero, zero]),
                      jnp.stack([-d1, d1, zero]),
                      jnp.stack([zero, -d2, d2])])


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
    """(r, k, q) -> the fourteen real response coefficients, bounded by _COEFF_MAX.

    The bound is RATIONAL, not `tanh`, and that is the whole point.  Both squash
    to +/-_COEFF_MAX and both have unit slope at the origin, so they are
    interchangeable for any coefficient in the normal range -- they differ only
    once the pre-activation runs deep into saturation, and there `tanh` is a
    gradient trap.  Its derivative `sech^2(net/C)` decays EXPONENTIALLY, so a
    saturated coefficient receives no gradient and can never come back: training
    is a one-way ratchet that progressively kills coefficients.

    Measured on 2026-08-20 (pure-NLL training, `dev/spin0_gradient_snr.py` and
    the attenuation scan beside it): the median coefficient's gradient
    attenuation `sech^2(net/C)` fell 0.996 -> 0.837 -> 3.2e-4 -> 3.5e-9 at
    6k -> 96k -> 400k -> 768k steps, with 63% of coefficients below 0.01 by the
    end.  That is the "runaway": the surviving coefficients compensate while the
    dead ones are frozen at the bound.  It also explains the batch dependence
    (more progress per step saturates sooner) and why the SUPERVISED flow never
    dies (`frac < 1e-4` is 0.000 -- the regression term keeps pulling
    coefficients back into the live range).  The signal itself was never
    missing: the NLL pins the spin-0 scale to sigma ~0.03 with a gradient SNR of
    ~36 at the point training converged to.

    `x / sqrt(1 + (x/C)^2)` decays only as `(C/x)^3`, so at ten times into
    saturation a coefficient still gets ~1e-3 of its gradient rather than 8e-9
    and can recover.  Same form as `models/centroid.py`'s `_EXP_MAX`
    saturation, and analytic everywhere -- `C*x/(C+|x|)` would bound just as
    well but has a discontinuous second derivative at the origin.
    """

    net: CoeffNet
    u_mean: jax.Array
    u_white: jax.Array

    def __init__(self, key, nn_width, nn_depth, activation,
                 u_mean=None, u_white=None):
        self.net = CoeffNet(key, 4, N_COEFFS, nn_width, nn_depth, activation)
        # Identity default keeps this a no-op for callers that do not supply
        # statistics (tests, ad-hoc layers); `bulk.build_flow` always does.
        self.u_mean = non_trainable(
            jnp.zeros(4) if u_mean is None else jnp.asarray(u_mean, jnp.float32))
        self.u_white = non_trainable(
            jnp.eye(4) if u_white is None else jnp.asarray(u_white, jnp.float32))

    def __call__(self, f, a, b, q):
        # WHITENED, not merely per-slot standardised.  Measured on the bulgedisc
        # training catalog: `a` (size) and `b` (concentration) correlate at
        # 0.997, and the covariance eigenvalues span 3.0e-3 to 8.25 -- condition
        # number 2625.  `(q - 2)/2` is not standardised either: the chi^2(2)
        # argument behind `_Q_SCALE` assumes Gaussian spin-2 slots, but real
        # ellipticities are heavy-tailed, giving std 2.761, skew 8.4 and a max
        # of 85.6.  Adam's diagonal preconditioner fixes per-parameter SCALE but
        # not input CORRELATION, which lands in the weight-space Hessian, so the
        # near-degenerate direction leaves the fit poorly determined along it --
        # the suspected cause of the seed-to-seed spread in the spin-0 response
        # (sd 0.54 in Mf at fixed steps, against 7e-4 in val nll).
        # The transform is the training set's own mean and inverse Cholesky
        # factor, held non-trainable: a data-determined constant of exactly the
        # same kind as `RawMomentStandardize`'s mean/std, not a hyperparameter.
        u = jnp.stack([f, a, b, (q - _Q_LOC) / _Q_SCALE])
        u = unwrap(self.u_white) @ (u - unwrap(self.u_mean))
        x = self.net(u)
        return x * jax.lax.rsqrt(1.0 + (x / _COEFF_MAX) ** 2)


def project_to_physics(coeffs, z, g, e_scale, chart_loc, chart_scale):
    """
    Project 14 basis coefficients into the physical Q (5,2) and R (5,2,2) tensors,
    based on the shift defined in response().
    """
    # The shift function that defines the physics
    def shift_func(g_vec):
        s0, A, B, mu, nu, rho = (coeffs[:9].reshape(3, 3), coeffs[9], coeffs[10],
                                 coeffs[11], coeffs[12], coeffs[13])
        e = e_scale * jax.lax.complex(z[3], z[4])
        gc = jax.lax.complex(g_vec[0], g_vec[1])
        ec, gcc = jnp.conj(e), jnp.conj(gc)
        p1 = (ec * gc).real
        p2 = (gc * gcc).real
        p3 = (ec * ec * gc * gc).real
        spin0 = (_chart_spin0_jac(z, chart_loc, chart_scale)
                         @ (s0 @ jnp.stack([p1, p2, p3])))
        de = (A * gc + B * e * e * gcc + mu * ec * gc * gc
              + nu * e * p2 + rho * e * e * e * gcc * gcc) / e_scale
        return jnp.concatenate([spin0, jnp.array([de.real, de.imag])])

    # Q (5, 2)
    Q = jax.jacfwd(shift_func)(jnp.zeros(2))
    # R (5, 2, 2)
    R = jax.hessian(shift_func)(jnp.zeros(2))
    return Q, R


def response(coeffs, z, g, e_scale, chart_loc, chart_scale):
    """The equivariant second-order response of `z` to `g`."""
    Q, R = project_to_physics(coeffs, z, g, e_scale, chart_loc, chart_scale)
    return z + Q @ g + 0.5 * jnp.einsum("ijk,j,k->i", R, g, g)




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
    # Its mean and std for the three spin-0 slots, FROZEN for the same reason
    # and used the same way: `_chart_spin0_jac` needs them to recover u and v.
    chart_loc: jax.Array = eqx.field(default=None)
    chart_scale: jax.Array = eqx.field(default=None)

    def __init__(self, key, nn_width=128, nn_depth=3, activation=jnn.silu,
                 cond_dim=2, e_scale=1.0, u_mean=None, u_white=None,
                 chart_loc=None, chart_scale=None):
        self.coeffs = _Coeffs(key, nn_width, nn_depth, activation,
                              u_mean=u_mean, u_white=u_white)
        self.e_scale = non_trainable(jnp.asarray(e_scale))
        # Defaults match `RawMomentStandardize`'s own (mean 0, std 1), so an
        # ad-hoc layer stays consistent with an ad-hoc chart.
        self.chart_loc = non_trainable(
            jnp.zeros(3) if chart_loc is None else jnp.asarray(chart_loc, jnp.float32))
        self.chart_scale = non_trainable(
            jnp.ones(3) if chart_scale is None else jnp.asarray(chart_scale, jnp.float32))
        # (2,) alone, or (5,) = [g1, g2, C00, C01, C11] when chained with the
        # centroid layer, which reads the other three.
        self.cond_shape = (cond_dim,)

    def unshear(self, x, condition):
        """Closed form; `transform`'s map, from observed moments back to base.

        Reads g as the FIRST two entries of `condition` -- see `cond_dim`.
        """
        g = condition[:2]
        f, a, b, q, _ = _invariants(x)
        return response(self.coeffs(f, a, b, q), x, g, unwrap(self.e_scale),
                        unwrap(self.chart_loc), unwrap(self.chart_scale))

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

    `shear.check` compares against it as the model's one piece of external
    ground truth -- an independent held-out test, since training is now pure
    NLL.  Feeding the layer raw moments here after the reordering would
    compare z-derivatives against raw truth, silently.
    """
    f = lambda g: chart.inverse(layer.shear(chart.transform(m), g))
    g0 = jnp.zeros(2)
    first = jax.jacfwd(f)(g0).T                      # (2, 5)
    H = jax.jacfwd(jax.jacfwd(f))(g0)                # (5, 2, 2)
    second = jnp.stack([H[:, 0, 0], H[:, 0, 1], H[:, 1, 1]])
    return first, second
