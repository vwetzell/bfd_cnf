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
import jax.random as jr
from flowjax.bijections import AbstractBijection
from paramax import non_trainable, unwrap

from .bijections import sas, sas_inv, sas_log_deriv

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

# How far the k^4 spin-2 coefficient may depart from its Gaussian value of 1.
# Generous on purpose: a bound that bites is the `_COEFF_MAX` trap -- it
# underfits AND severs the gradient at saturation.  The population-average
# value the two response channels ask for is ~0.87 (ellipticity) to ~-0.3
# (size), so the useful range straddles zero and 3.0 covers both with room.
_C_MAX = 3.0


def _zero_head(mlp):
    """`mlp` with its output layer zeroed, so it returns exactly 0 at init."""
    mlp = eqx.tree_at(lambda m: m.layers[-1].weight, mlp,
                      jnp.zeros_like(mlp.layers[-1].weight))
    return eqx.tree_at(lambda m: m.layers[-1].bias, mlp,
                       jnp.zeros_like(mlp.layers[-1].bias))


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


def raw_from_standard(z, mean, std, flux_sas=None):
    """Undo `RawMomentStandardize`: the inverse chart, [Mf, Mr, M1, M2, Mc].

    Exactly `RawMomentStandardize._inverse_transform` (`models/bijections.py`)
    -- duplicated rather than imported because this layer only ever needs a
    FROZEN copy of the chart's constants, never the trainable chart itself
    (see `CentroidMarginalize.mean/std`).
    """
    z0 = std[0] * z[0] + mean[0]
    z1 = std[1] * z[1] + mean[1]
    z2 = std[2] * z[2] + mean[2]
    Mf = jnp.power(10.0, z0 if flux_sas is None else sas_inv(z0, flux_sas))
    Mr = z1 * Mf
    Mc = z2 * Mr
    M1 = (std[3] * z[3] + mean[3]) * Mr
    M2 = (std[4] * z[4] + mean[4]) * Mr
    return jnp.stack([Mf, Mr, M1, M2, Mc])


def standard_from_raw(m, mean, std, flux_sas=None):
    """The chart, forward: [Mf, Mr, M1, M2, Mc] -> standardised z.

    Matches `RawMomentStandardize._forward_transform`, exactly inverting
    `raw_from_standard` EXCEPT where `_safe_logit`'s clip is active -- see its
    docstring.
    """
    Mf, Mr, M1, M2, Mc = m[0], m[1], m[2], m[3], m[4]
    z0 = jnp.log10(Mf)
    if flux_sas is not None:
        z0 = sas(z0, flux_sas)
    z1 = Mr / Mf
    z2 = Mc / Mr
    z3 = M1 / Mr
    z4 = M2 / Mr
    z = jnp.stack([z0, z1, z2, z3, z4])
    return (z - mean) / std


def _transport(m, sigma_x, sign, c_spin2=1.0, c_spin4=1.0):
    """The centroid-marginalisation map on raw moments, both directions.

    `sign = +1.0`: base -> data (forward marginalisation, physically what
    happens to a galaxy).  `sign = -1.0`: data -> base, its EXACT inverse
    within the Gaussian-in-k ansatz -- see module docstring.  `R` (the
    ansatz's effective real-space width matrix) is solved from `m`'s OWN
    measured moments regardless of which direction is being evaluated; only
    the sign of the k-space damping `P` differs.

    The two corrections after the backbone are the measured k^4 brackets.
    Every first-order response is the contraction
    `dM_a = -1/2 Sigma_u^{ij} <k_i k_j kernel_a>`, and writing each symmetric
    bracket as trace + traceless gives, with `s0 = tr(Sigma_u)`,
    `s1 = Su_xx - Su_yy`, `s2 = 2 Su_xy`:

        dMf = -1/4 (s0 Mr + s1 M1 + s2 M2)
        dMr = -1/4 (s0 Mc + s1 N1 + s2 N2)
        dM1 = -1/4 (s0 N1 + s1 (Mc + K1)/2 + s2 K2/2)
        dM2 = -1/4 (s0 N2 + s1 K2/2   + s2 (Mc - K1)/2)

    `Mf, Mr, M1, M2, Mc` are all measured; `N = N1 + i N2` (the k^4 SPIN-2
    moment `int k^2 (kx^2-ky^2) G + i int k^2 2 kx ky G`) and `K = K1 + i K2`
    (k^4 spin-4) are not.  Wick on the Gaussian ansatz gives
    `N = 3 Mr (M1 + i M2) / Mf` and `K = 3 (M1 + i M2)^2 / Mf`, and those are
    what the closed-form backbone above implicitly uses -- verified against
    its own expansion to 8 digits on the v3 prior.

    `c_spin2` and `c_spin4` are the trained coefficients:
    `N = c_spin2 * 3 Mr (M1+iM2)/Mf` and `K = c_spin4 * 3 (M1+iM2)^2/Mf`.
    Both are REAL, not complex.  An imaginary part would rotate each k^4 moment
    away from the direction the measured spin-2 moment sets -- real per galaxy
    (isophote twist, bulge/disc misalignment) but zero in conditional mean given
    parity-even conditioning.  Every invariant these coefficients can see
    (Mr/Mf, Mc/Mr, |e|) is parity even, so a free imaginary part could only
    learn a parity-odd artifact of the training sample.  Same failure class as
    the spin-2 standardisation anisotropy.

    That is also the precise sense in which `K` is "zero for a co-elliptical
    galaxy": not that the spin-4 MOMENT vanishes (it does not -- an ellipse has
    one), but that its phase is locked to twice the spin-2 moment's, leaving
    only a real magnitude for `c_spin4` to carry.  Breaking the lock needs a
    second spin-2 direction, which on this population does not exist and which
    an elliptical PSF would supply.

    All three corrections are added at first order rather than resummed: once
    the ansatz's own `Mc`, `N` and `K` are overridden there is no closed form
    left to resum, and each stays a few percent of its channel even in the
    barely-resolved tail.
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
    mc_ansatz_in = (2.0 * Mr ** 2 + M1 ** 2 + M2 ** 2) / Mf
    s0 = jnp.trace(sigma_u)
    s1 = sigma_u[0, 0] - sigma_u[1, 1]
    s2 = 2.0 * sigma_u[0, 1]
    # The k^4 TRACE bracket, measured instead of assumed: the Wick trace is
    # `mc_ansatz_in`, and Mc/Mc_ansatz has p50 0.928 on the v3 prior (sd
    # 0.003-0.006 within an Mr/Mf bin, so it is nearly a function of size).
    Mr_out = Mr_out - sign * 0.25 * s0 * (Mc - mc_ansatz_in)
    # The k^4 SPIN-2 bracket.  It enters dMr contracted with Sigma_u's own
    # traceless part (|s|/s0 = 0.178 on this population, ~2|e|, and ANTI-
    # aligned with the galaxy) and dM1/dM2 contracted with its trace -- so one
    # coefficient moves the size response and the ellipticity response
    # together, with ~35x more leverage on the latter.
    dN1 = (c_spin2 - 1.0) * 3.0 * Mr * M1 / Mf
    dN2 = (c_spin2 - 1.0) * 3.0 * Mr * M2 / Mf
    Mr_out = Mr_out - sign * 0.25 * (s1 * dN1 + s2 * dN2)
    M1_out = M1_out - sign * 0.25 * s0 * dN1
    M2_out = M2_out - sign * 0.25 * s0 * dN2
    # The k^4 SPIN-4 bracket.  It appears ONLY in dM1/dM2, and only against
    # Sigma_u's traceless part -- there is no spin-4 piece of dMf or dMr,
    # because a trace cannot see one.  This is the slot an elliptical PSF
    # needs: it is the only bracket that can carry a spin-2 direction the
    # galaxy's own ellipticity does not set.
    dK1 = (c_spin4 - 1.0) * 3.0 * (M1 ** 2 - M2 ** 2) / Mf
    dK2 = (c_spin4 - 1.0) * 6.0 * M1 * M2 / Mf
    M1_out = M1_out - sign * 0.125 * (s1 * dK1 + s2 * dK2)
    M2_out = M2_out - sign * 0.125 * (s1 * dK2 - s2 * dK1)
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

    One trainable object: `coeff`, the net behind `_transport`'s `c_spin2`
    (the k^4 spin-2 bracket -- see `_transport`).  `mean`/`std` are a frozen
    copy of the chart's constants, needed only to convert between the
    standardised z this bijection operates on and the raw moments `_transport`
    needs a physical scale for -- see `raw_from_standard`.
    """

    shape: tuple = (5,)
    # Static, for the reason given in models/shear.py: a plain field assigned in
    # __init__ turns its contents into pytree leaves and breaks checkpoints.
    cond_shape: tuple = eqx.field(static=True, default=(3,))
    # Must match the chart this layer sits behind -- see `mean`/`std`.
    flux_sas: tuple | None = eqx.field(static=True, default=None)
    mean: jax.Array = eqx.field(default=None)
    std: jax.Array = eqx.field(default=None)
    coeff: eqx.nn.MLP = eqx.field(default=None)

    def __init__(self, mean=None, std=None, cond_dim=3, flux_sas=None,
                 key=None):
        self.flux_sas = None if flux_sas is None else tuple(
            float(v) for v in flux_sas)
        self.mean = non_trainable(jnp.zeros(5) if mean is None
                                  else jnp.asarray(mean))
        self.std = non_trainable(jnp.ones(5) if std is None
                                 else jnp.asarray(std))
        # Zero final layer => both coefficients == 1 EXACTLY at init, i.e. the untrained
        # layer is the bare Gaussian-in-k ansatz (plus the measured k^4 trace),
        # so a --steps 0 run reproduces the zero-parameter transport bit for
        # bit and the hidden layer's own init never shows up in a baseline.
        self.coeff = _zero_head(eqx.nn.MLP(
            3, 2, 32, 1, activation=jnn.tanh,
            key=jr.key(0) if key is None else key))
        # (3,) alone, or (5,) = [g1, g2, C00, C01, C11] when chained after shear.
        self.cond_shape = (cond_dim,)

    def chart(self):
        """The frozen chart's constants, spin-2 pair forced isotropic.

        EXACTLY `RawMomentStandardize._effective`, and for the same reason --
        this layer must read the chart the chart actually applies.  It was
        reading `self.mean`/`self.std` RAW, which `bulk.build_flow` and
        `bulk.sync_chart_constants` both copy straight off `chart.mean`/
        `chart.std` while the chart itself only ever uses `_effective()`.  So
        the layer decoded M1 and M2 with two DIFFERENT scales and a nonzero
        mean, and its map was not rotation-equivariant: on `centroid_v10.eqx`
        `std[3]/std[4] = 1.0146` and `mean[3:] = (1.1e-4, -5.1e-4)`, worth a
        45-deg equivariance residual of 4.3e-5 (p50) against the 8.6e-7 float32
        floor -- 50x, and 850x at p99.9 (`dev/centroid_equivariance.py`).  The
        `_EXP_MAX` clamp, the other suspect, is measurably inert: removing it
        changes the residual by nothing (`dev/clamp_census.py` says why -- the
        spin-2 |dz| p99 is 0.098 against `_EXP_MAX = 0.5`).
        Same failure `e_scale` had in the shear layer
        ([[e-scale-was-unsymmetrised]]); this layer never got the fix.

        Applied HERE rather than at construction for the reason `_effective`
        gives, plus one this layer has of its own: the raw values are baked
        into every existing checkpoint, and `eqx.tree_deserialise_leaves`
        overwrites whatever `build_flow` put there.  A construction-site fix
        would be inert for exactly the flows that need it.
        """
        mean, std = unwrap(self.mean), unwrap(self.std)
        spin2_scale = jnp.sqrt(0.5 * (std[3] ** 2 + std[4] ** 2))
        return mean.at[3:].set(0.0), std.at[3:].set(spin2_scale)

    def bracket_coeffs(self, m, mean, std):
        """`(c_spin2, c_spin4)` for one galaxy, from parity-even invariants.

        The chart's own standardised size and concentration logits (`z[1]`,
        `z[2]`) plus `|e|^2` in units of the chart's symmetrised spin-2 std.
        Nothing here is a new population constant -- all three come from the
        frozen chart copy this layer already carries, which is what keeps this
        net out of the stale-constants failure mode the shear layer's
        coefficient whitening fell into.  Flux is deliberately NOT an input:
        it is dimensionful, and the ansatz's error is a property of the
        galaxy's shape, not of how bright it is.

        `|e|^2`, not `|e|`: the two carry the same information (the net can
        compose any monotone function of one from the other) but `|e|` is
        `sqrt(M1^2 + M2^2)`, whose gradient is infinite at a perfectly round
        galaxy -- and a round galaxy is exactly the case
        `test_gradients_are_finite_for_a_round_galaxy` exists to catch.

        One trunk, two heads -- both coefficients are functions of the same
        three invariants, and a shared trunk is the arrangement the shear
        layer's five coefficients already use.
        """
        z = standard_from_raw(m, mean, std, self.flux_sas)
        u = jnp.stack([z[1], z[2],
                       (m[2] ** 2 + m[3] ** 2) / (m[1] * std[3]) ** 2])
        return 1.0 + _rational_bound(self.coeff(u), _C_MAX)

    def unmarginalize(self, x, condition):
        """Closed form; `transform`'s map, from observed moments back to base."""
        mean, std = self.chart()
        _, sigma_x = split_condition(condition)
        m = raw_from_standard(x, mean, std, self.flux_sas)
        y = standard_from_raw(
            _transport(m, sigma_x, -1.0,
                       *self.bracket_coeffs(m, mean, std)),
            mean, std, self.flux_sas)
        return x + _EXP_MAX * jnp.tanh((y - x) / _EXP_MAX)

    def marginalize(self, y, condition):
        """Closed form; the forward physical map, base -> data."""
        mean, std = self.chart()
        _, sigma_x = split_condition(condition)
        m = raw_from_standard(y, mean, std, self.flux_sas)
        x = standard_from_raw(
            _transport(m, sigma_x, 1.0,
                       *self.bracket_coeffs(m, mean, std)),
            mean, std, self.flux_sas)
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
