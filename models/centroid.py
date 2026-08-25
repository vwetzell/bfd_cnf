"""
models/centroid.py
==================
The centroid-marginalisation layer of the BFD prior P(m | g, Sigma_X), for the
five even moments m = [Mf, Mr, M1, M2, Mc] (Bernstein et al. 2016, MNRAS 459,
4467).

What it models
--------------
A target's centroid is not known.  It is found, by solving X = 0 on a noisy
image, so the moments a target reports are those of the galaxy seen from a
slightly wrong origin u.  BFD never assumes the template's origin either: each
template is replicated over a grid of origins and the prior is the weighted sum
over that grid (eq. 36),

    P(M, s | G, g) = J(M) SUM_u d2u L[X^G(u); 0, Sigma_X] L[M - M^G(u)]

The displacement is distributed by the target's own first-moment covariance:
recentring solves X^G(u) + X^n = 0 with X^n ~ N(0, Sigma_X), so

    u ~ |J(u)| N(X^G(u); 0, Sigma_X),      J = dX/du

This layer is that marginalisation, as a map on the moments.  It is the last
thing before the data --- base -> bulk -> shear(g) -> centroid(Sigma_X) -> data
--- because it is the last thing that happens to a galaxy: it is lensed on the
sky, and only then measured about a centroid somebody had to guess.

Why it matters
--------------
Displacing the origin costs flux and beats down the high-k moments hardest, so a
copy is dimmer, larger and less concentrated than its parent (Mf, Mr/Mf and
Mc/Mr all fall; mind that Mr/Mf and Mc/Mr are INVERSE size and concentration).
The part that bites is spin-2: the marginalisation coherently AMPLIFIES the
apparent ellipticity, measured on the imsims bulge+disc population as

    <e_w e0*> / <|e0|^2> - 1  =  +3.5e-4 (S/N > 40), +2.1e-3 (20-40), +1.4e-2 (<20)

scaling as 1/(S/N)^2.  Left unmodelled that is a multiplicative shear bias of
the same size, i.e. 3x to 15x an LSST error budget, which is what this layer is
for.

The form is fixed by symmetry, not chosen
-----------------------------------------
Sigma_X is a symmetric 2-tensor, so it decomposes into a spin-0 trace and a
spin-2 traceless part, exactly as (|g|^2, g) does for shear --- and the same
three symmetries as `models/shear.py` then fix the map.  The natural variable is
not Sigma_X itself but the displacement covariance it induces,

    Sigma_u = J^-1 Sigma_X J^-T,    J = -(1/2) [[Mr + M1, M2], [M2, Mr - M1]]

(bfd's `MomentCalculator.xyJacobian`), made dimensionless with the galaxy's own
scale as T = (Mr/Mf) Sigma_u.  Using the exact J rather than J ~ -(Mr/2) I costs
nothing and carries the leading shape dependence for free.  Splitting T,

    t0 = (T00 + T11)/2   (spin 0)      t2 = [(T00 - T11) + 2i T01]/2   (spin 2)

the marginalisation is second order in u and hence FIRST order in T, leaving
exactly two spin-0 structures and three spin-2 ones:

    spin-0 (each of Mf, Mr, Mc):   t0,  Re(e* t2)
    spin-2 (e):                    t2,  e t0,  e^2 t2*

Parity makes all nine coefficients real, and flux-linearity makes them functions
of the same three flux-blind invariants the shear layer uses --- r = Mr/Mf,
k = Mc Mf/Mr^2, q = |e|^2.

With Sigma_X isotropic, t2 does NOT vanish: Sigma_u = sigma^2 J^-1 J^-T and J is
elliptical, so an elliptical galaxy's centroid error is anisotropic --- larger
along the major axis, where the flux gradient is shallower.  What does hold is
that J depends only on the galaxy's own e, so t2 comes out PARALLEL to e and so
does every spin-2 structure built from it.  The marginalisation then rescales a
galaxy's ellipticity without rotating it and picks out no axis on the sky, which
is the isotropy `RawMomentStandardize` was symmetrised to protect.  Only a
genuinely anisotropic Sigma_X --- an elliptical PSF --- can turn a galaxy, and
then it should.  `tests/test_centroid.py` pins both halves.

Invertibility
-------------
`transform` (data -> base, "un-marginalise") is closed form; it is the direction
likelihood training calls.  `inverse` is a fixed-point iteration, which is
honest here rather than lazy: T is the squared centroid error in units of the
galaxy size and runs ~5e-4 at S/N 70 to ~9e-3 at S/N 13, so three passes reach
machine precision.  There is no small-g-style truncation --- unlike shear,
Sigma_X is never zero for a real target, so a series inversion about T = 0 would
be approximating in exactly the regime that matters.  See
`tests/test_centroid.py::test_round_trip`.

Cost, and why the log-det is not analytic
-----------------------------------------
The two apparent expenses are not what they look like.

The fixed-point passes are off the hot path entirely: `log_prob` runs data ->
base, which is `transform_and_log_det`, the closed-form direction.  The
iterations only run in `inverse`, i.e. in `sample`.

The `jacfwd` 5x5 log-det is a real cost, but it does not have to be paid inside
a shear derivative.  This layer is data-adjacent, so in a full flow it is
applied FIRST, to the raw moments, conditioned only on Sigma_X --- both its
input and its condition are independent of g.  Its log-det is therefore exactly
g-independent and contributes NOTHING to Q or R, so a caller can evaluate it
once, outside the autodiff, and fold it into an importance weight.  That is what
`bias.split_centroid` / `bias.centroid_transform` do, and it is exact.

Making the log-det analytic instead would need a triangular Jacobian, and that
costs physics rather than code: a triangular spin-0 block requires each
coordinate's response to be independent of that coordinate, but the natural
variable T = (Mr/Mf) Sigma_u needs Mf in order to transform Mf; and a
closed-determinant spin-2 block of the `s I + c E` form requires dropping
q = |e|^2 from the coefficient inputs.  Both are real restrictions on the
response, for a speedup the hoist above already provides.

What it cannot carry
--------------------
The same `Var[.|m]` floor as the shear layer.  Two galaxies with identical
moments have different structure and so marginalise differently; a deterministic
transport can only carry the conditional mean of that.  It is the same argument
as `models/shear.py`'s docstring and it wants the same eventual fix (a
stochastic layer), not a wider network.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from paramax import non_trainable, unwrap

from .bijections import POINT_SOURCE, CoeffNet
from .shear import G_MAX, _invariants

# As in models/shear.py: bound the coefficients so the map stays a
# diffeomorphism for any Sigma_X the network might see.  T itself is ~1e-3, so
# nothing real comes close to this.
_COEFF_MAX = 12.0

# Invariant normalisation, shared with the shear layer (same population, same
# weight function): r = Mr/Mf over ~[0.7, 4.0], k = Mc Mf/Mr^2 over ~[1.85, 2.4],
# q = |e|^2 over ~[0, 0.3].
# As in models/shear.py, this layer now acts on the STANDARDISED z rather than
# on raw moments, so z1 and z2 need no shift and q = |e|^2 is a chi^2_2.
_Q_LOC, _Q_SCALE = 2.0, 2.0

# T is fed in RAW, not rescaled to O(1).  Its median over the imsims bulge+disc
# population is 1.4e-3, and the fractional moment shift it has to produce is the
# same 1e-3 -- so a coefficient of order one is already the right answer, and
# normalising T to O(1) would instead demand coefficients of order 1e-3 and start
# the layer a factor of 1000 wrong.
#
# The tail is the reason for _EXP_MAX.  t0 runs 7.9e-6 (1st pct) to 1.4e-3
# (median) to 0.10 (max): a 74x spread over the median, all of it the barely
# resolved galaxies (Mr/Mf -> 3.88 against the 3.976 point-source ceiling) whose
# centroid error reaches 0.29 arcsec, comparable to the PSF.  First order in T is
# meaningless there -- so the exponent saturates rather than overflowing, which
# keeps the map a diffeomorphism at every T instead of returning inf.
_EXP_MAX = 0.5

N_COEFFS = 9

# p1 = Re(ebar g)/G_MAX carries a 1/G_MAX = 50x chain-rule factor into
# d(coeff)/dg, and `response`'s de_z = de_raw/std[3] divides by the spin-2 std
# (~0.04-0.16) for ANOTHER 6-25x -- so an entirely ordinary coefficient
# sensitivity to p1 becomes an order-1 sensitivity to g once those two
# divisions compound (measured 320x on the gauss2_deep flow).  Bounding
# d(coeff)/dp1 by _SLOPE_MAX, rather than the coefficient itself, is what
# stops that compounding from blowing up -- see `_Coeffs`.
#
# 3.0 (the first value tried) was not tight enough to act as a real
# constraint: nothing stopped training from driving MOST galaxies' slopes
# toward it rather than just a spiky few, so d(shift)/dg1 got WORSE across
# the population (median 0.0012 -> 0.0039, frac > 1e-2 16.5% -> 33%) even
# though it fixed the mechanism the tail came from.  Swept post-hoc on the
# trained net (re-bounding its raw output costs nothing -- no retrain per
# point) on 2000 in-domain `copies_gauss2_deep` targets:
#
#   _SLOPE_MAX   max|d(shift)/dg1|   frac > 1e-2   frac > 1e-1
#        3.0           0.63             0.331         0.042
#        0.3           0.11             0.103         0.002
#        0.1           0.038            0.041         0.000
#        0.03          0.012            0.002         0.000
#
# 0.1 cuts the worst case 16x against the original UNCONSTRAINED nonlinear
# net's 0.60 and clears the >1e-1 tail entirely, while its median (2.9e-4)
# stays within an order of magnitude of a well-behaved target's typical
# sensitivity (~1e-4) rather than being forced to exactly zero.
_SLOPE_MAX = 0.1


def _rational_bound(x, c):
    """`x / sqrt(1 + (x/c)^2)`: |result| < c, unit slope at 0, gradient never
    dies in saturation -- see `_Coeffs`."""
    return x * jax.lax.rsqrt(1.0 + (x / c) ** 2)


def displacement_covariance(m, sigma_x):
    """Sigma_u = J^-1 Sigma_X J^-T, the covariance of the centroid error.

    `J = dX/du = -(1/2) [[Mr + M1, M2], [M2, Mr - M1]]` is bfd's
    `MomentCalculator.xyJacobian`; a target recentres to X^G(u) + X^n = 0, so
    linearising gives u = -J^-1 X^n and hence this.

    Parameters
    ----------
    m : jax.Array, shape (5,)
        Raw moments [Mf, Mr, M1, M2, Mc].
    sigma_x : jax.Array, shape (3,)
        Sigma_X as its three unique values [C00, C01, C11].
    """
    Mr, M1, M2 = m[1], m[2], m[3]
    j = -0.5 * jnp.array([[Mr + M1, M2], [M2, Mr - M1]])
    sx = jnp.array([[sigma_x[0], sigma_x[1]], [sigma_x[1], sigma_x[2]]])
    jinv = jnp.linalg.inv(j)
    return jinv @ sx @ jinv.T


def split_condition(condition):
    """(g, Sigma_X) from a condition vector of either length.

    The chained flow passes ``(5,) = [g1, g2, C00, C01, C11]``; a standalone
    centroid layer, and `dm_dsigma`, pass a bare ``(3,)`` Sigma_X.  Sigma_X is
    always the LAST three entries, so only g has to be decided -- and it must be
    decided on the SHAPE, not by slicing blindly: `condition[:2]` on a (3,)
    vector would silently hand C00, C01 to the layer as a shear.

    Shapes are static under jit, so this branch costs nothing at run time.
    """
    c = jnp.asarray(condition)
    g = c[..., :2] if c.shape[-1] >= 5 else jnp.zeros_like(c[..., :2])
    return g, c[..., -3:]


def g_invariants(z, g, mean, std):
    """The two spin-0 invariants of (e, g) the coefficients may depend on.

    The coefficients are SCALARS, so equivariance lets them depend on any spin-0
    invariant -- and the ones g brings are exactly `p1 = Re(ebar g)` and
    `p2 = |g|^2`, the same pair `models/shear.py` builds its response from.
    Letting the coefficients carry them is what makes the marginalisation
    shear-dependent at fixed moments: expanding e.g. `A(p1, p2) t2` generates
    `Re(ebar g) t2` and the rest of the g-bearing spin-2 structures with the
    right symmetry automatically, rather than by hand.

    `e` is the PHYSICAL shape, for the same reason as in `response`: z3, z4 are
    divided by the spin-2 std (~0.04), so a standardised e would inflate p1 by
    ~25x and make the two inputs incommensurate.

    Both are normalised by the training shear disc `G_MAX` so they arrive at the
    network as O(|e|) and O(1) rather than as 1e-3 and 4e-4, which the net could
    not resolve against its four O(1) inputs.
    """
    m = raw_from_standard(z, mean, std)
    e = jax.lax.complex(m[2] / m[1], m[3] / m[1])
    gc = jax.lax.complex(g[0], g[1])
    return ((jnp.conj(e) * gc).real / G_MAX,
            (gc * jnp.conj(gc)).real / G_MAX**2)


def raw_from_standard(z, mean, std):
    """Undo `RawMomentStandardize` far enough to rebuild [Mf, Mr, M1, M2].

    This layer acts on z, but T is a PHYSICAL object: Sigma_X is an area and has
    to be made dimensionless with the galaxy's own scale, which only exists in
    raw moment space.  So reconstruct just the four moments the Jacobian needs.
    Mc is not among them and is not rebuilt.

    The reconstruction does NOT have to track a trained standardiser exactly.
    `mean`/`std` are frozen at the values `build_flow` measured, and the only
    thing they set is the overall scale of T -- which the response coefficients
    absorb, since they are free functions of the invariants.  What must be right
    is how T SCALES with Sigma_X and with the galaxy's size and shape, and that
    is exact regardless of where the standardiser drifts to.
    """
    z0 = std[0] * z[0] + mean[0]
    z1 = std[1] * z[1] + mean[1]
    Mf = jnp.power(10.0, z0)
    Mr = POINT_SOURCE * jnn.sigmoid(z1) * Mf
    M1 = (std[3] * z[3] + mean[3]) * Mr
    M2 = (std[4] * z[4] + mean[4]) * Mr
    return jnp.stack([Mf, Mr, M1, M2, Mr])      # slot 4 unused by the Jacobian


def _tensor(z, sigma_x, mean, std):
    """The dimensionless (t0, t2) that drive the response.

    T = (Mr/Mf) Sigma_u is dimensionless because Sigma_u is an area and Mr/Mf is
    an inverse area under this weight function.  `z` is standardised, so the
    physical moments are rebuilt first -- see `raw_from_standard`.
    """
    m = raw_from_standard(z, mean, std)
    su = displacement_covariance(m, sigma_x)
    t = (m[1] / m[0]) * su
    t0 = 0.5 * (t[0, 0] + t[1, 1])
    # ponytail: t2 is ~1% of t0 for a typical galaxy, so this difference of two
    # nearly-equal entries loses most of its float32 significance -- the spin-2
    # response picks up a roundoff component perpendicular to e of up to 1.5%,
    # against 5e-11 in float64.  It is structured by the coordinate axes (period
    # pi/2, alternating sign) and averages to zero over an isotropic population,
    # so it does not read out as additive shear; tests/test_centroid.py checks
    # the symmetry in float64.  If it ever needs to be exact in float32, form the
    # traceless part analytically -- for isotropic Sigma_X it is
    # -sigma^2 (traceless part of J^2) / det(J^2) -- instead of subtracting here.
    t2 = jax.lax.complex(0.5 * (t[0, 0] - t[1, 1]), t[0, 1])
    return t0, t2


class _Coeffs(eqx.Module):
    """(f, a, b, q) -> nine response coefficients, AFFINE in the shear
    invariants (p1, p2): `coeff = base(f,a,b,q) + slope1(f,a,b,q)*p1 +
    slope2(f,a,b,q)*p2`, base bounded by _COEFF_MAX, the slopes by _SLOPE_MAX.

    Not merely a simplification.  A first version let a plain 6-input net
    (f, a, b, q, p1, p2) learn an arbitrary NONLINEAR dependence on p1, p2.
    Nothing constrained that dependence to be smooth, and the antithetic
    +/-g training (`centroid.py train`) sees only two p1 values per galaxy
    per step -- nowhere near enough to pin down curvature -- so the net was
    free to fit a much steeper LOCAL slope in some (f,a,b,q) pockets than
    its typical one, concentrated on faint targets (where |e|, and so
    |dp1/dg1| = |Re(ebar)|/G_MAX, runs largest): traced on
    `copies_gauss2_deep.fits`, one galaxy's d(shift)/dg1 measured -0.60
    against a typical well-behaved target's 1e-4, a ~6000x spread with no
    physical justification.  It produced a deep `bias.py` run with
    windowed-corrected m1 = -0.163 (quintile q1 = -1.20), dominated by a
    handful of such galaxies (see HANDOFF.md, 2026-08-24).

    Root cause: p1 = Re(ebar g)/G_MAX divides by the training disc radius
    (1/G_MAX = 50x), and `response`'s de_z = de_raw/std[3] divides by the
    spin-2 std (~0.04-0.16, another 6-25x) -- so an entirely ordinary net
    sensitivity to p1 (order 1e-2) becomes an order-1-10 sensitivity to g once
    those two divisions compound (measured 320x here).  Rescaling p1's INPUT
    scale cannot fix this: it only moves the amplification between "input
    scale" and "net's required slope" without changing their product.  What
    has to be bounded is the SLOPE itself, `d(coeff)/dp1`, which the affine
    form makes an explicit, capped, (f,a,b,q)-only quantity -- CONSTANT in p1,
    so no curvature is representable at all -- rather than an emergent,
    uncontrolled net derivative.

    Rational bound, not `tanh`, for the same reason `models/shear.py`'s
    `_Coeffs` uses it (2026-08-20, `dev/spin0_gradient_snr.py`): `x / sqrt(1 +
    (x/C)^2)` decays as `(C/x)^3` in saturation rather than `sech^2`'s
    exponential, so a saturated output still gets gradient and can recover.

    Six of the nine coefficients are further SIGN-CONSTRAINED, not just
    magnitude-bounded: see the `t0_mask`/`e_mask` step in `__call__`.  NLL
    training of this layer is a density-match between a deterministic
    bijective transport and a genuinely stochastic marginalisation (a mixture
    over centroid offsets, not a point map) -- and that mismatch lets NLL
    prefer the WRONG SIGN, verified by scanning NLL directly against the
    trained Mf/t0 coefficient: flipping it to the physically-required sign
    makes NLL monotonically worse.  Displacing the origin costs Mf, Mr/Mf and
    Mc/Mr for every copy of every galaxy regardless of shear (a deterministic
    fact, not a population tendency), so those coefficients don't need NLL to
    discover their sign -- it's known, and is built in rather than hoped for.
    Both s0 columns get this treatment, and both signs are the OPPOSITE of
    the naive first guess -- see `__call__` for why `marginalize` being the
    inverse of `unmarginalize` flips them: t0's coefficient >= 0 (t0 >= 0
    always), and (ec*t2).real's coefficient <= 0 (that invariant is <= 0
    always for isotropic Sigma_X).
    """

    net: CoeffNet

    def __init__(self, key, nn_width, nn_depth, activation):
        net = CoeffNet(key, 4, 3 * N_COEFFS, nn_width, nn_depth, activation)
        # ZERO-INIT the slope heads (output rows N_COEFFS: onward), so a
        # freshly initialised layer starts exactly g-independent, same as the
        # old zero-init did for the (now removed) g input columns -- enforced
        # on the output side since p1, p2 no longer reach the net as inputs.
        w, b = net.layers[-1].weight, net.layers[-1].bias
        net = eqx.tree_at(
            lambda n: (n.layers[-1].weight, n.layers[-1].bias), net,
            (w.at[N_COEFFS:].set(0.0), b.at[N_COEFFS:].set(0.0)))
        self.net = net

    def __call__(self, f, a, b, q, p1=0.0, p2=0.0):
        # p1, p2 default to the unsheared values so a standalone layer and
        # every g = 0 diagnostic keep working unchanged.
        u = jnp.stack([f, a, b, (q - _Q_LOC) / _Q_SCALE])
        x = self.net(u)
        base, s1, s2 = x[:N_COEFFS], x[N_COEFFS:2 * N_COEFFS], x[2 * N_COEFFS:]
        base = _rational_bound(base, _COEFF_MAX)
        s1 = _rational_bound(s1, _SLOPE_MAX)
        s2 = _rational_bound(s2, _SLOPE_MAX)
        coeff = base + s1 * jnp.asarray(p1, f.dtype) + s2 * jnp.asarray(p2, f.dtype)
        # Re-bound the SUM: the affine combination can exceed _COEFF_MAX near
        # the disc edge (large p1, p2), and the Newton fixed point + jacfwd
        # log-det this feeds want every coefficient bounded, not just `base`.
        bounded = _rational_bound(coeff, _COEFF_MAX)
        # The three T0-driven spin-0 coefficients (s0's t0-column: flat
        # indices 0, 2, 4 of the (3,2) reshape, i.e. z0=Mf, z1=logit(Mr/Mf),
        # z2=logit(Mc/Mr)) are SIGN-CONSTRAINED to >= 0.  Displacing the
        # origin costs Mf, Mr/Mf and Mc/Mr for every copy of every galaxy --
        # a deterministic geometric fact (mind the INVERSE size/concentration
        # convention -- see this module's header), true regardless of shear --
        # but that constrains `marginalize` (base -> data), not the quantity
        # THIS function computes, which feeds `unmarginalize` (data -> base,
        # `response`'s `shift`).  `marginalize = unmarginalize^-1`, and to
        # leading order `unmarginalize(x) ~= x + shift(x)` inverts to
        # `marginalize(y) ~= y - shift(y)` -- ONE SIGN FLIP relative to the
        # physical requirement.  So `marginalize(x) - x <= 0` (the physical
        # fact) requires `shift(x) >= 0`, i.e. a POSITIVE t0-coefficient here,
        # not negative -- easy to get backwards (this module did, once,
        # 2026-08-24, and shipped a t0<=0 constraint that left the layer's
        # check() shift positive-signed, unchanged from unconstrained).
        #
        # Plain NLL does not know this constraint and gets it wrong on its
        # own: verified 2026-08-24 by scanning NLL directly against the
        # trained (unconstrained) Mf/t0 coefficient, which converged to
        # NEGATIVE -- the sign that makes NLL LOWEST, and monotonically WORSE
        # moving toward the physically-required positive value.  Not a
        # training failure more steps would fix: the objective (density-
        # matching a deterministic transport against a genuinely stochastic
        # marginalisation) simply does not encode the constraint, so it has
        # to be structural.
        #
        # The other column of s0 -- indices 1, 3, 5, multiplying (ec*t2).real
        # -- needs the OPPOSITE sign, <= 0, and for the same reason once the
        # same inversion is tracked through.  For ISOTROPIC Sigma_X (every
        # population on disk today: "Sigma_X is the same for every galaxy in
        # the current sims, fixed noise level, circular PSF" -- this module's
        # Staging section), J shares its principal axes with the galaxy's own
        # ellipticity e, so t2's phase locks to e's and (ec*t2).real reduces
        # to a PHASE-INDEPENDENT function of |e| alone -- verified numerically
        # (any Mr, |e|, phase): it is <= 0 always.  A regression of the
        # catalog's true `marginalize(x)-x` against both invariants gives a
        # POSITIVE coefficient on this always-negative term (+5.6 to +14.4 for
        # Mf/Mr/Mc); after the same sign flip through the inverse relationship
        # as t0, that means `shift`'s own e-coefficient must be <= 0 -- the
        # negative of what a first pass at this comment claimed.  (Both
        # proofs are isotropic-Sigma_X-specific; an elliptical-PSF population
        # would need rederiving before trusting either sign.)
        neg = -_rational_bound(jnn.softplus(coeff), _COEFF_MAX)
        pos = _rational_bound(jnn.softplus(coeff), _COEFF_MAX)
        idx = jnp.arange(N_COEFFS)
        t0_mask = (idx == 0) | (idx == 2) | (idx == 4)
        e_mask = (idx == 1) | (idx == 3) | (idx == 5)
        out = jnp.where(t0_mask, pos, bounded)
        out = jnp.where(e_mask, neg, out)
        return out


def response(coeffs, z, sigma_x, mean, std):
    """The equivariant first-order-in-Sigma_X response of `z`; the layer's map.

    `coeffs` is [a, b] x [z0, z1, z2] followed by [A, B, D].  Both blocks are
    ADDITIVE, for the reason given in models/shear.py's `response`: z is
    unbounded, so nothing needs exponentiating to stay positive, and z3, z4 are
    already dimensionless so the spin-2 term needs no factor of Mr.
    """
    s0, A, B, D = coeffs[:6].reshape(3, 2), coeffs[6], coeffs[7], coeffs[8]
    # Physical shape, for the same reason as models/shear.py's `response`.
    m = raw_from_standard(z, mean, std)
    e = jax.lax.complex(m[2] / m[1], m[3] / m[1])
    t0, t2 = _tensor(z, sigma_x, mean, std)
    ec, t2c = jnp.conj(e), jnp.conj(t2)

    # The two spin-0 invariants of (e, T) at first order in T, saturated so the
    # unresolved tail cannot blow the map up (see _EXP_MAX).
    shift = s0 @ jnp.stack([t0, (ec * t2).real])
    spin0 = z[:3] + _EXP_MAX * jnp.tanh(shift / _EXP_MAX)
    # Spin-2: additive in the standardised shape slots, bounded the same way, so
    # |de| < _EXP_MAX and the shape cannot run away.
    de_raw = A * t2 + B * e * t0 + D * e * e * t2c
    # Saturate as a scalar function of |de|^2, which keeps the term spin-2
    # equivariant AND analytic at de = 0.  `jnp.abs` would not be: with an
    # isotropic Sigma_X and a round galaxy de_raw is exactly zero, and the
    # gradient of a complex modulus there is NaN -- which is precisely the batch
    # this layer sees most often.
    # Convert to standardised units BEFORE saturating, so the cap means the same
    # thing as the spin-0 one: at most `_EXP_MAX` of a standard deviation.
    # Saturating in physical units and then dividing by the spin-2 std (~0.04)
    # would leave the shape free to move ~12 sigma, which is no cap at all.
    de_z = de_raw / std[3]
    q = (de_z * jnp.conj(de_z)).real
    de = de_z / jnp.sqrt(1.0 + q / _EXP_MAX**2)
    # SLOT LAYOUT: spin-0 is 0, 1, 2 and spin-2 is 3, 4 in standardised
    # coordinates -- not the raw (0, 1, 4) / (2, 3) layout.
    return jnp.stack([spin0[0], spin0[1], spin0[2],
                      z[3] + de.real, z[4] + de.imag])


class CentroidMarginalize(AbstractBijection):
    """Centroid marginalisation of the five even moments, conditioned on Sigma_X.

    ``transform`` maps data -> base (un-marginalise), ``inverse`` base -> data,
    matching flowjax's convention that a flow wraps its bijection in ``Invert``.
    The condition is Sigma_X as its three unique values ``[C00, C01, C11]``.
    """

    shape: tuple = (5,)
    # Static, for the reason given in models/shear.py: a plain field assigned in
    # __init__ turns its contents into pytree leaves and breaks checkpoints.
    cond_shape: tuple = eqx.field(static=True, default=(3,))
    coeffs: _Coeffs
    n_iter: int = eqx.field(static=True, default=3)
    # The standardiser's constants, FROZEN: the layer needs them only to rebuild
    # a physical scale for T, and `raw_from_standard` explains why a frozen copy
    # is enough even though `RawMomentStandardize` trains its own.
    mean: jax.Array = eqx.field(default=None)
    std: jax.Array = eqx.field(default=None)

    def __init__(self, key, nn_width=128, nn_depth=3, activation=jnn.silu,
                 n_iter=3, cond_dim=3, mean=None, std=None):
        self.coeffs = _Coeffs(key, nn_width, nn_depth, activation)
        self.n_iter = n_iter
        self.mean = non_trainable(jnp.zeros(5) if mean is None
                                  else jnp.asarray(mean))
        self.std = non_trainable(jnp.ones(5) if std is None
                                 else jnp.asarray(std))
        # (3,) alone, or (5,) = [g1, g2, C00, C01, C11] when chained after shear.
        self.cond_shape = (cond_dim,)

    def unmarginalize(self, x, condition):
        """Closed form; `transform`'s map, from observed moments back to base.

        Reads Sigma_X as the LAST three entries of `condition` and g as the
        first two, so the same layer serves a standalone ``(3,)`` condition and
        the combined ``(5,)`` one ``[g1, g2, C00, C01, C11]`` that a
        shear-plus-centroid flow passes down the chain -- see `split_condition`.

        The marginalisation is a property of the LENSED galaxy: it is sheared on
        the sky and only then measured about a guessed origin.  Most of that
        dependence already arrives through `x`, since the layer sits downstream
        of `ShearResponse` generatively and so is evaluated on lensed moments.
        What `x` cannot carry is the dependence at FIXED m: this layer models
        E[marginalisation | m], a conditional mean over profiles, and shear
        changes which profiles map to a given m.  That is what conditioning the
        coefficients on g adds.
        """
        mean, std = unwrap(self.mean), unwrap(self.std)
        g, sigma_x = split_condition(condition)
        f, a, b, q, _ = _invariants(x)
        p1, p2 = g_invariants(x, g, mean, std)
        return response(self.coeffs(f, a, b, q, p1, p2), x, sigma_x, mean, std)

    def marginalize(self, y, condition):
        """Invert `unmarginalize` by fixed point: x = y - (U(x) - x).

        U is the identity at Sigma_X = 0 and departs from it by O(T) with
        T ~ 1e-3, so the iteration contracts at that rate and three passes are
        already at machine precision.  Unrolled rather than looped so it stays
        one jittable expression.
        """
        x = y
        for _ in range(self.n_iter):
            x = y - (self.unmarginalize(x, condition) - x)
        return x

    def transform_and_log_det(self, x, condition=None):
        return (self.unmarginalize(x, condition),
                jnp.linalg.slogdet(jax.jacfwd(self.unmarginalize)(x, condition))[1])

    def inverse_and_log_det(self, y, condition=None):
        x = self.marginalize(y, condition)
        return x, -jnp.linalg.slogdet(
            jax.jacfwd(self.unmarginalize)(x, condition))[1]


def dm_dsigma(layer, m, sigma_x, chart):
    """The layer's generative shift of `m` at this Sigma_X, in RAW moments.

    Returns the raw-moment displacement the layer thinks centroid
    marginalisation produces for a galaxy sitting at `m`, to compare against the
    weighted copy mean from an `imsims.copies` catalog -- which is a raw-moment
    quantity.

    `chart` is NOT optional, for the same reason as `models/shear.dm_dg`: the
    layer marginalises in STANDARDISED coordinates now, so its bare shift is in
    z.  Composing the chart on both sides converts it.  `centroid.check`
    compares against it as a held-out diagnostic, since training is now pure
    NLL; feeding raw moments to a z-space layer would compare incommensurate
    quantities, silently.
    """
    return chart.inverse(layer.marginalize(chart.transform(m), sigma_x)) - m
