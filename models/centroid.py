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

An exact, zero-parameter marginalisation integral (2026-08-26)
----------------------------------------------------------------
A per-galaxy coefficient net fitted to `E[marginalisation | m]` (this module's
previous architecture) is a first-order Taylor expansion in the dimensionless
perturbation T = (Mr/Mf) Sigma_u, valid only near T = 0.  On the imsims
population T runs from ~1e-3 (bright, resolved) to 0.5+ (faint, barely
resolved), and the fitted LINEAR coefficient, extrapolated past where the
linear approximation holds, overshot the true (self-averaging) marginalisation
response by up to 5.9x in exactly that tail -- a structural defect in the
functional form, not a fitting one (HANDOFF.md, 2026-08-26).

The fix replaces the Taylor term with the (approximately) EXACT integral.  bfd
moments are `M_a(u) = Re[Sum_k W(k) kernel_a(k) I(k) e^{ik.u}]`
(`bfd/momentcalc.py`), so averaging over the target's own centroid-error
distribution `u ~ N(0, Sigma_u)` is EXACTLY equivalent to damping the k-space
integrand by an extra Gaussian factor `exp(-1/2 k^T Sigma_u k)` -- this step
needs no assumption about the galaxy's profile shape, only the u ~ N(0,
Sigma_u) linearisation already used throughout this module.  Modelling the
product `W(k) I(k)` itself as Gaussian in k (exact for a single Gaussian
profile times a Gaussian-shaped weight; `gauss2` is literally a Gaussian
mixture) then gives every one of the five moments a CLOSED FORM in terms of a
single real-space "width" matrix R, solved EXACTLY from the galaxy's own
measured (Mf, Mr, M1, M2) -- no fitting, no free parameters:

    R = (1 / (2 Mf)) [[Mr + M1, M2], [M2, Mr - M1]]      (== -J / Mf, bfd's
                                                            own xyJacobian,
                                                            `displacement_
                                                            covariance`'s J)

    P = R @ Sigma_u
    R~ = (I + P)^-1 @ R                    (NOT R @ (I+P)^-1 -- R and Sigma_u
                                             do not commute in general, only
                                             when Sigma_X is isotropic)
    Mf' = Mf / sqrt(det(I + P))
    Mr' = Mf' * trace(R~)          M1' = Mf' * (R~_xx - R~_yy)
    M2' = Mf' * 2 R~_xy
    Mc' = Mc + [(2 Mr'^2+M1'^2+M2'^2)/Mf' - (2 Mr^2+M1^2+M2^2)/Mf]

`Mf', Mr', M1', M2'` reduce EXACTLY to the measured input at Sigma_X = 0 (no
approximation there).  Mc does not, because a real galaxy is not exactly
Gaussian -- so Mc is corrected DIFFERENTIALLY (the ansatz's own S=0 prediction
subtracted off), which keeps the Sigma_X = 0 identity exact regardless of how
good the Gaussian approximation is for Mc specifically.  Validated on
`copies_gauss2_deep` (100k galaxies, all five quantities, including the
faintest-and-most-compact quintile where the old model's overshoot was worst):
Mf/Mr/Mc match the true copy-weighted mean to a few percent with NO trained
parameters, and the ellipticity response error flips from a 4.5x-5.9x
OVERSHOOT to a roughly flat 25-30% UNDERSHOOT across the whole population --
see HANDOFF.md, 2026-08-26/27, for the full derivation and validation tables.

This closed form is its own inverse.  `A := R^-1` is the k-space precision
matrix the ansatz implicitly assigns the galaxy; damping by Sigma_u maps
A -> A + Sigma_u, and by the SAME self-consistency (the Gaussian family is
closed under composition of dampings), solving R from a DATA point's own
measured moments and damping by -Sigma_u exactly undoes it.  So `marginalize`
(base -> data) and `unmarginalize` (data -> base) are the SAME function with
the sign of `P` flipped and R solved from whichever point is the input --
see `_transport`.  Both are closed-form; there is no fixed-point iteration
left in this layer.

The z-space OUTPUT of both directions is still capped at `_EXP_MAX`
(`unmarginalize`/`marginalize`, `tanh`-saturated exactly as the retired
architecture's output was) -- not because `_transport` needs it to stay
finite (it does not: no free parameters, no cap of its own, exact at any T
within the ansatz), but because this layer's map is summed over a batch of
targets downstream, and an exact-but-large shift for one barely-resolved
galaxy can land the frozen bulk+shear flow -- never retrained against this
layer's OWN output distribution -- in a steep-curvature region of ITS
density and dominate that sum on its own, past what a batch-level outlier
guard catches.  Restoring this cap took `bias.py`'s `gauss2_deep` m1 from
~-0.7 (uncapped) back to something governed by the actual physics (HANDOFF.md,
2026-08-27).  Same value as the retired architecture's `_EXP_MAX`, reused
rather than re-derived.

No shear-conditioning.  The old net modelled `E[marginalisation | m]`, a
population-averaged conditional mean, which is why it needed to know that
shear changes which profiles land at a given observed m
(`shear_delta_invariants`, retired).  `_transport` is not a population
average -- it solves the Gaussian ansatz EXACTLY for the one galaxy in front
of it, at whatever point (base or data) it is handed, so there is no mixture
whose composition shear could change.

No population-tuned constants anywhere in this formula.  `R` is solved from
each galaxy's own measured moments; the moment kernels (1, k^2, kx^2-ky^2,
2 kx ky, k^4) are bfd's own fixed definitions.  The one place a smooth
NUMERICAL floor appears (`_transport`'s `sign < 0` branch) is a
diffeomorphism-safety bound in the same spirit as the old `_EXP_MAX`/
`_COEFF_MAX`, not a physics constant -- see its docstring.

What it cannot carry
---------------------
The Gaussian-profile-in-k approximation itself: real galaxies are not exactly
Gaussian, which is exactly what the Mc self-consistency ratio and the
remaining ellipticity-response undershoot measure (see HANDOFF.md).  And, as
before, the same `Var[.|m]` floor as the shear layer -- two galaxies with
identical moments marginalise differently, and a deterministic transport can
only carry the conditional mean of that.

Cost, and why the log-det is not analytic
-----------------------------------------
`log_prob` runs data -> base, which is `transform_and_log_det`; both
directions are now closed-form matrix algebra on 2x2 matrices (no iteration),
so this layer's own forward cost is small either way.  The `jacfwd` 5x5
log-det is the real cost, and it does not have to be paid inside a shear
derivative: this layer is data-adjacent, so in a full flow it is applied
FIRST, to the raw moments, conditioned only on Sigma_X -- both its input and
its condition are independent of g, so its log-det is exactly g-independent
and a caller can evaluate it once, outside the autodiff, and fold it into an
importance weight.  That is what `bias.split_centroid` / `bias.
centroid_transform` do, and it remains exact.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from paramax import non_trainable, unwrap

from .bijections import POINT_SOURCE, POINT_SOURCE_MC

# The tail's OUTPUT magnitude, in standardised-z units -- not a physics bound,
# a numerical/estimator-stability one.  `_transport` is exact within the
# Gaussian-in-k ansatz at ANY T with no cap of its own, but a downstream
# consumer (`bias.py`) sums this layer's map over a batch of targets, and the
# rare barely-resolved galaxy whose EXACT shift is large enough to land the
# frozen bulk+shear flow in a steep-curvature part of its own (unrelated,
# never retrained against this layer) density can dominate that sum even
# after the batch-level "|Q| or |R| > 1000x median" guard -- measured as
# catastrophic m1 (~-0.7) on `targets_gauss2_deep_g0_200k.fits` with no cap,
# clean once this one is restored (HANDOFF.md, 2026-08-27).  Reusing the old
# architecture's own `_EXP_MAX` value, not inventing a new number: this is
# the same "must stay a diffeomorphism / bounded per-galaxy contribution"
# argument that constant was already established under.
_EXP_MAX = 0.5


def _rational_bound(x, c):
    """`x / sqrt(1 + (x/c)^2)`: `|result| < c` always, unit slope at 0, and
    the derivative `1/(1+(x/c)^2)^1.5` is NEVER exactly zero for finite `x`
    -- unlike `jnp.clip`, whose zero gradient at its boundary turns a
    `jacfwd` log-det into `-inf` exactly there (see `_safe_logit`)."""
    return x * jax.lax.rsqrt(1.0 + (x / c) ** 2)


def _rational_scale(x, c):
    """Multiplicative safety factor `1 / sqrt(1 + (x/c)^2)`.

    `x * _rational_scale(x, c)` saturates strictly below `c` for any `x >= 0`
    (the multiplicative form of `_rational_bound` used elsewhere in this
    codebase), is exactly 1 at `x = 0`, and has no 0/0 at `x = 0` the way
    `_rational_bound(x, c) / x` would.  See `_transport`.
    """
    return jax.lax.rsqrt(1.0 + (x / c) ** 2)


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
    `g` is unused by this layer (see module docstring, "No shear-
    conditioning") but is still split out so callers built for the chained
    (5,) condition do not have to special-case this layer.
    """
    c = jnp.asarray(condition)
    g = c[..., :2] if c.shape[-1] >= 5 else jnp.zeros_like(c[..., :2])
    return g, c[..., -3:]


def raw_from_standard(z, mean, std):
    """Undo `RawMomentStandardize`: the inverse chart, [Mf, Mr, M1, M2, Mc].

    Exactly `RawMomentStandardize._inverse_transform` (`models/bijections.py`)
    -- duplicated rather than imported because this layer only ever needs a
    FROZEN copy of the chart's constants, never the trainable chart itself
    (see `CentroidMarginalize.mean/std`).
    """
    z0 = std[0] * z[0] + mean[0]
    z1 = std[1] * z[1] + mean[1]
    z2 = std[2] * z[2] + mean[2]
    Mf = jnp.power(10.0, z0)
    Mr = POINT_SOURCE * jnn.sigmoid(z1) * Mf
    Mc = POINT_SOURCE_MC * jnn.sigmoid(z2) * Mr
    M1 = (std[3] * z[3] + mean[3]) * Mr
    M2 = (std[4] * z[4] + mean[4]) * Mr
    return jnp.stack([Mf, Mr, M1, M2, Mc])


def _safe_logit(v):
    """`logit(v)`, with `v` clipped to `(eps, 1-eps)` first.

    `_transport`'s data <-> base direction is only exact WITHIN the Gaussian-
    in-k ansatz (module docstring); on a real (not exactly Gaussian) galaxy,
    its output ratio can land at or past the chart's own point-source
    ceiling -- `v >= 1` -- where the true `log(v) - log1p(-v)` is a log of a
    non-positive number, i.e. NaN, not -inf, and nothing downstream survives
    that (same failure mode `models/bijections.py`'s `safe_point` exists
    for).

    A smooth rational-bound rescale of `v` itself (an earlier version of
    this function) is the WRONG tool: real galaxies span most of `(0, 1)`,
    not just a neighbourhood of 0.5, so any smooth global rescale with a
    bound near 1 measurably distorts ordinary, safely-in-range targets, not
    just the rare boundary case -- caught as a median z-space shift of
    ~0.5-1.2 (should be ~1e-3, matching T's own scale) on
    `targets_gauss2_deep_g0_200k.fits` (HANDOFF.md, 2026-08-27).  A hard
    `jnp.clip` is exact (a true no-op) for any `v` not already at the
    boundary, unlike a smooth rescale over the whole domain; it does zero
    the gradient exactly at the clip, so this layer's `jacfwd` log-det comes
    back `-inf` for a row that hits it, same as `-inf`/NaN Q or R elsewhere
    in this codebase -- `bias.py` already drops those (`dropping N targets
    with a non-finite Q or R`), which is the right outcome for a target
    this ansatz genuinely cannot represent, not a bug to paper over.
    """
    v = jnp.clip(v, 1e-6, 1.0 - 1e-6)
    return jnp.log(v) - jnp.log1p(-v)


def standard_from_raw(m, mean, std):
    """The chart, forward: [Mf, Mr, M1, M2, Mc] -> standardised z.

    Matches `RawMomentStandardize._forward_transform`, exactly inverting
    `raw_from_standard` EXCEPT where `_safe_logit`'s clip is active -- see its
    docstring.
    """
    Mf, Mr, M1, M2, Mc = m[0], m[1], m[2], m[3], m[4]
    u = Mr / (POINT_SOURCE * Mf)
    v = Mc / (POINT_SOURCE_MC * Mr)
    z0 = jnp.log10(Mf)
    z1 = _safe_logit(u)
    z2 = _safe_logit(v)
    z3 = M1 / Mr
    z4 = M2 / Mr
    z = jnp.stack([z0, z1, z2, z3, z4])
    return (z - mean) / std


def _transport(m, sigma_x, sign, gain=1.0):
    """The centroid-marginalisation map on raw moments, both directions.

    `sign = +1.0`: base -> data (forward marginalisation, physically what
    happens to a galaxy).  `sign = -1.0`: data -> base, its EXACT inverse
    within the Gaussian-in-k ansatz -- see module docstring.  `R` (the
    ansatz's effective real-space width matrix) is solved from `m`'s OWN
    measured moments regardless of which direction is being evaluated; only
    the sign of the k-space damping `P` differs.

    `gain` scales the SPIN-2 displacement only, leaving Mf and Mr exactly as
    the ansatz computes them.  At `gain = 1.0` (the default) nothing changes.
    It exists because the Gaussian-in-k ansatz undershoots the ellipticity
    response by ~30% on realistic galaxies -- a two-Gaussian galaxy with W(k)
    carried exactly recovers only a third of that, the rest being bulge/disc
    misalignment no co-elliptical model can represent, while a single scalar
    calibrated against the population's own copy catalog closes it to ~0.4% on
    held-out galaxies.  The spin-0 channels are left alone because they are
    already right to 2-9%; rescaling Sigma_u instead would have moved them by
    the full gain.  `centroid.py calibrate` measures it; it does NOT transfer
    between populations, so it must be recalibrated for each.
    """
    Mf, Mr, M1, M2, Mc = m[0], m[1], m[2], m[3], m[4]
    R = (0.5 / Mf) * jnp.array([[Mr + M1, M2], [M2, Mr - M1]])
    sigma_u = displacement_covariance(m, sigma_x)
    P = R @ sigma_u
    # Only the DATA -> BASE direction can hit a singularity: no valid base
    # galaxy exists, under this ansatz, once Sigma_u "exceeds" the data's own
    # effective width.  Both eigenvalues of P are <= trace(P) (P is similar
    # to a PSD matrix, `Rh @ sigma_u @ Rh` with Rh = R^1/2, so its eigenvalues
    # are real and >= 0), so capping the TRACE strictly below 1 keeps I - P
    # invertible for ANY input -- a numerical safety net (the role _EXP_MAX
    # played in the old architecture), not a physics bound: it is exactly 1
    # at Sigma_X = 0 (trace(P) = 0) and has unit slope there, so it never
    # touches the identity case, only the pathological tail.
    if sign < 0:
        P = P * _rational_scale(jnp.trace(P), 1.0 - 1e-3)
    ip = jnp.eye(2) + sign * P
    det = jnp.linalg.det(ip)
    r_tilde = jnp.linalg.solve(ip, R)
    Mf_out = Mf / jnp.sqrt(det)
    Mr_out = Mf_out * (r_tilde[0, 0] + r_tilde[1, 1])
    M1_out = Mf_out * (r_tilde[0, 0] - r_tilde[1, 1])
    M2_out = Mf_out * 2.0 * r_tilde[0, 1]
    # Amplify the ELLIPTICITY shift, not the moments: e = (M1 + i M2) / Mr is
    # what the response is measured in, and scaling M1/M2 directly would drag
    # the gain through Mr's own (already accurate) change.
    M1_out = Mr_out * (M1 / Mr + gain * (M1_out / Mr_out - M1 / Mr))
    M2_out = Mr_out * (M2 / Mr + gain * (M2_out / Mr_out - M2 / Mr))
    mc_ansatz_in = (2.0 * Mr ** 2 + M1 ** 2 + M2 ** 2) / Mf
    mc_ansatz_out = (2.0 * Mr_out ** 2 + M1_out ** 2 + M2_out ** 2) / Mf_out
    Mc_out = Mc + (mc_ansatz_out - mc_ansatz_in)
    return jnp.stack([Mf_out, Mr_out, M1_out, M2_out, Mc_out])


class CentroidMarginalize(AbstractBijection):
    """Centroid marginalisation of the five even moments, conditioned on Sigma_X.

    ``transform`` maps data -> base (un-marginalise), ``inverse`` base -> data,
    matching flowjax's convention that a flow wraps its bijection in ``Invert``.
    The condition is Sigma_X as its three unique values ``[C00, C01, C11]``
    (or the chained ``(5,)`` ``[g1, g2, C00, C01, C11]`` -- see
    `split_condition`).

    No trainable parameters: see the module docstring.  `mean`/`std` are a
    frozen copy of the chart's constants, needed only to convert between the
    standardised z this bijection operates on and the raw moments `_transport`
    needs a physical scale for -- see `raw_from_standard`.
    """

    shape: tuple = (5,)
    # Static, for the reason given in models/shear.py: a plain field assigned in
    # __init__ turns its contents into pytree leaves and breaks checkpoints.
    cond_shape: tuple = eqx.field(static=True, default=(3,))
    # Static for a second reason too: a calibration constant, like the chart's
    # POINT_SOURCE, and keeping it off the leaf list means every checkpoint
    # written before it existed still deserialises.
    gain: float = eqx.field(static=True, default=1.0)
    mean: jax.Array = eqx.field(default=None)
    std: jax.Array = eqx.field(default=None)

    def __init__(self, mean=None, std=None, cond_dim=3, gain=1.0):
        self.gain = float(gain)
        self.mean = non_trainable(jnp.zeros(5) if mean is None
                                  else jnp.asarray(mean))
        self.std = non_trainable(jnp.ones(5) if std is None
                                 else jnp.asarray(std))
        # (3,) alone, or (5,) = [g1, g2, C00, C01, C11] when chained after shear.
        self.cond_shape = (cond_dim,)

    def unmarginalize(self, x, condition):
        """Closed form; `transform`'s map, from observed moments back to base."""
        mean, std = unwrap(self.mean), unwrap(self.std)
        _, sigma_x = split_condition(condition)
        m = raw_from_standard(x, mean, std)
        y = standard_from_raw(_transport(m, sigma_x, -1.0, self.gain), mean, std)
        return x + _EXP_MAX * jnp.tanh((y - x) / _EXP_MAX)

    def marginalize(self, y, condition):
        """Closed form; the forward physical map, base -> data."""
        mean, std = unwrap(self.mean), unwrap(self.std)
        _, sigma_x = split_condition(condition)
        m = raw_from_standard(y, mean, std)
        x = standard_from_raw(_transport(m, sigma_x, 1.0, self.gain), mean, std)
        return y + _EXP_MAX * jnp.tanh((x - y) / _EXP_MAX)

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
    compares against it as a held-out diagnostic.
    """
    return chart.inverse(layer.marginalize(chart.transform(m), sigma_x)) - m
