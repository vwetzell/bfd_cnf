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
from .shear import _invariants

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
    """(f, a, b, q) -> the nine response coefficients, bounded by _COEFF_MAX.

    Rational bound, not `tanh`: `x / sqrt(1 + (x/C)^2)` has the same +/-C
    asymptote and unit slope at the origin, but decays as `(C/x)^3` instead of
    `sech^2` -- exponentially -- so a coefficient driven deep into saturation
    still gets gradient and can recover, rather than being frozen there for the
    rest of training.  `models/shear.py`'s `_Coeffs` made the same swap for the
    same reason (2026-08-20, `dev/spin0_gradient_snr.py`); this layer still had
    the old tanh form, and a 16k-step retrace on `copies_gauss2_deep.fits`
    reproduced the identical one-way ratchet -- 0% of D coefficients pinned at
    step 0, 92% by step 16000 -- which is the trained centroid layer's low-flux
    ellipticity-response blowup on gauss2 (see HANDOFF.md).
    """

    net: CoeffNet

    def __init__(self, key, nn_width, nn_depth, activation):
        self.net = CoeffNet(key, 4, N_COEFFS, nn_width, nn_depth, activation)

    def __call__(self, f, a, b, q):
        u = jnp.stack([f, a, b, (q - _Q_LOC) / _Q_SCALE])
        x = self.net(u)
        return x * jax.lax.rsqrt(1.0 + (x / _COEFF_MAX) ** 2)


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

        Reads Sigma_X as the LAST three entries of `condition`, so the same layer
        serves a standalone ``(3,)`` condition and the combined ``(5,)`` one
        ``[g1, g2, C00, C01, C11]`` that a shear-plus-centroid flow passes down
        the chain.
        """
        sigma_x = condition[-3:]
        f, a, b, q, _ = _invariants(x)
        return response(self.coeffs(f, a, b, q), x, sigma_x,
                        unwrap(self.mean), unwrap(self.std))

    def marginalize(self, y, sigma_x):
        """Invert `unmarginalize` by fixed point: x = y - (U(x) - x).

        U is the identity at Sigma_X = 0 and departs from it by O(T) with
        T ~ 1e-3, so the iteration contracts at that rate and three passes are
        already at machine precision.  Unrolled rather than looped so it stays
        one jittable expression.
        """
        x = y
        for _ in range(self.n_iter):
            x = y - (self.unmarginalize(x, sigma_x) - x)
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
