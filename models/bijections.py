"""
models/bijections.py
====================
Custom flowjax bijection classes used in the bfd_cnf normalizing-flow model:

  - RawMomentStandardize   — transforms raw moments to standardized coordinates
  - BoundedAffine          — affine bijection with bounded scale parameter
  - Spin0AutoregressiveLayer / Spin2CouplingLayer / EquivariantAutoregressiveLayer
  - ExplicitPolyLast / SigmaXCouplingLayer / EarlyChain
  - new_masked_autoregressive_flow

Helper functions: _bounded_log_scale, _bounded_scale, _inv_bounded_scale,
make_lower_tri_index, lower_tri_flat, _raw2std_jacobian_single_jax,
propagate_cov_to_std_jax.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import equinox as eqx
from flowjax.bijections import AbstractBijection, Chain, Invert, Permute
from flowjax.distributions import AbstractDistribution, Transformed
from flowjax.utils import arraylike_to_array
from jaxtyping import Array, ArrayLike, Shaped
from paramax import Parameterize, AbstractUnwrappable

# Mr/Mf ceiling: a point source, i.e. the PSF itself.  Anything at or above it
# is unresolved and carries no shape information.  It lives here rather than in
# `bulk` because `RawMomentStandardize` makes it a hard support boundary and
# `bulk` imports this module, not the other way round; `bulk.POINT_SOURCE`
# re-exports it.
#
# THIS IS A PROPERTY OF THE WEIGHT FUNCTION and must be recomputed whenever the
# weight changes -- it is sum(W k^2)/sum(W) over the k grid.  It was 3.976167
# for Harris's coefficients and stayed at that value through the switch to
# Nuttall's, whose true ceiling is 3.909604: stale by 1.7%.  That is not
# cosmetic.  The whole point of slot 1's logit is to put the support boundary
# at infinity so the density vanishes there on its own; with the ceiling set too
# high the real boundary lands at a FINITE logit (4.07 for the Nuttall
# catalogs, whose moments top out at Mr/Mf = 3.8925), i.e. an interior cliff the
# flow has to learn instead of one the chart handles for free.  It also made the
# "nothing lies above the ceiling" check vacuous.  3.692575 is the value for the
# second-order-safe window now in bfd/weightfunction.py.
POINT_SOURCE = 3.692575

# Mc/Mr ceiling, by the identical argument: a point source has Itilde/T = 1
# everywhere, so its moments are the pure weight moments and
# sum(W k^4)/sum(W k^2) is the point-source value of Mc/Mr, exactly as
# sum(W k^2)/sum(W) is the point-source value of Mr/Mf above.  Any extended
# galaxy has a positive, decreasing Itilde/T that downweights high k more than
# low k, so Mc/Mr sits strictly below this the same way Mr/Mf sits strictly
# below POINT_SOURCE -- verified on both catalogs (bulgedisc tops out at
# 6.6365, gauss2 at 6.5110).
#
# THIS IS A PROPERTY OF THE WEIGHT FUNCTION and must be recomputed whenever
# the weight changes, same as POINT_SOURCE above -- see that comment for what
# happens when it goes stale.
POINT_SOURCE_MC = 6.662089


def in_domain(m):
    """Rows whose raw moments the flow's chart can EVALUATE.

    With the bare-ratio chart this is only positivity of Mf and Mr: slots 1 and
    2 are `Mr/Mf` and `Mc/Mr`, which are finite for any positive denominator,
    and the log-det needs `log Mf` and `log Mr`.  There is deliberately no
    point-source ceiling here -- the chart must be able to represent moments
    past it, because noisy templates will land there.

    This is a NUMERIC domain, not a physical one.  For "where is the prior
    density nonzero", which is a different and stricter question, see
    :func:`in_support`.
    """
    return (m[..., 0] > 0) & (m[..., 1] > 0)


def in_support(m):
    """Rows inside the PHYSICAL support of the noiseless moment population.

    The integration variable in BFD's `int P(M|m) P(m|g) dm` is a NOISELESS
    template moment vector, so `P(m|g)` is identically zero wherever no
    noiseless galaxy can live.  That region is known analytically and is
    model-blind -- it follows from the positivity of the surface brightness
    against a known weight function, not from any galaxy model:

      * `Mf > 0`, `Mr > 0`;
      * `Mr/Mf < POINT_SOURCE` -- a point source is the SMALLEST an object can
        be, so it maximises the weighted k^2 moment per unit flux;
      * `Mc/Mr < POINT_SOURCE_MC` -- the same statement one k^2 higher.

    Both constants are the pure-weight-moment ratios of a delta convolved with
    the PSF, verified against `KBlackmanHarris`, and both hold on the v3
    noiseless catalog with margin (max 3.6835 vs 3.692575; 6.6488 vs 6.6621).

    This is what the flow's learned tails were approximating badly.  Imposing
    it explicitly costs nothing in accuracy -- 99.9% of out-of-support kernel
    draws were already being discarded as poisoned -- but it makes the boundary
    the PHYSICAL surface rather than wherever the flow's extrapolation happened
    to fall off a cliff, which is what the boundary term in `dP/dg` needs.

    Written without division so a row with `Mf <= 0` cannot produce a NaN that
    a downstream mask fails to stop.
    """
    Mf, Mr, Mc = m[..., 0], m[..., 1], m[..., 4]
    return ((Mf > 0) & (Mr > 0)
            & (Mr < POINT_SOURCE * Mf)
            & (Mc < POINT_SOURCE_MC * Mr))


def safe_point(m):
    """An in-domain stand-in for rows a mask will discard anyway.

    The chart is still EVALUATED on masked rows -- `jnp.where` evaluates both
    branches, and a NaN or an infinity in the discarded one still poisons the
    gradient -- so they need coordinates it can actually evaluate, at half of
    each point-source ceiling rather than merely below it.

    BOTH ceilings, and Mc last.  Clamping Mr DOWN raises Mc/Mr, so a row that
    only violated slot 1 comes out of the slot-1 clamp violating slot 2 --
    which is silent: the dummy's own derivative overflows, `jnp.where`
    multiplies it by zero, 0 * inf = NaN, and the caller discards the whole
    row.  That cost 18.9% of the noisy targets when `POINT_SOURCE_MC` was added
    without updating this function.

    The stand-in must satisfy :func:`in_support`, not merely :func:`in_domain`:
    once the prior carries an explicit support indicator, a dummy outside it
    gets `log p = -inf`, and `0 * -inf` is the same NaN this function exists to
    prevent.  Hence the ceilings are clamped here even though the chart itself
    no longer has them.
    """
    mf = jnp.maximum(m[..., 0], 1e-6)
    # Mr before Mc: clamping Mr DOWN raises Mc/Mr, so the concentration clamp
    # has to see the value Mr actually ends up with.
    mr = jnp.clip(m[..., 1], 1e-6, 0.5 * POINT_SOURCE * mf)
    mc = jnp.minimum(m[..., 4], 0.5 * POINT_SOURCE_MC * mr)
    return m.at[..., 0].set(mf).at[..., 1].set(mr).at[..., 4].set(mc)


# ---------------------------------------------------------------------------
# Bounded-scale helpers
# ---------------------------------------------------------------------------


#: Conditioner inputs are soft-clipped to +-this before reaching an MLP.  The
#: converged v3 bulk flow's largest conditioner input on real data is 10.1
#: (p99 = 4.4), so 20 is a no-op in-distribution: the p=6 clip below is within
#: 0.3% of the identity out to |u| = 10.
_COND_MAX = 20.0


def _soft_clip(u: jax.Array, bound: float = _COND_MAX) -> jax.Array:
    """Smoothly saturate ``u`` to ``(-bound, bound)``, near-identity inside.

    ``u / (1 + (u/bound)**6)**(1/6)``: within 0.3% of ``u`` for
    ``|u| < bound/2``, and flat to within 0.1% of ``bound`` by ``|u| = 2 bound``.

    This is what gives the flow LINEAR TAILS.  Without it, a coordinate a little
    outside the training envelope drives a conditioner MLP linearly, its
    ``_bounded_log_scale`` pins at ``max_scale``, and the resulting coordinate
    feeds the next layer's conditioner -- a runaway that took conditioner inputs
    to 6e5 and ``log p`` to -1e5 only 0.1 past the data.  With the inputs
    clipped, every ``loc`` and ``log_scale`` goes CONSTANT out there, so each
    layer degenerates to a plain affine map and the density's tail is Gaussian.

    Smooth rather than a hard ``clip`` because the score (and hence Q and R) is
    evaluated at targets that sit near the envelope, and a kink there would show
    up as a discontinuity in the shear response.
    """
    t = u / bound
    # Past |t| = 4 the closed form is within 1e-4 of the asymptote while t**6
    # overflows float32 for the 1e5-sized inputs this exists to tame, so branch.
    big = jnp.abs(t) > 4.0
    safe = jnp.where(big, 0.0, t)
    smooth = safe / (1.0 + safe ** 6) ** (1.0 / 6.0)
    return bound * jnp.where(big, jnp.sign(t) * (1.0 - 1e-4), smooth)


def _bounded_scale(u: jax.Array, min_scale: float, max_scale: float) -> jax.Array:
    """Map an unconstrained parameter to a bounded scale via sigmoid.

    Parameters
    ----------
    u : jax.Array
        Unconstrained parameter.
    min_scale : float
        Lower bound of the output range.
    max_scale : float
        Upper bound of the output range.

    Returns
    -------
    jax.Array
        Constrained scale in ``(min_scale, max_scale)``.
    """
    return min_scale + (max_scale - min_scale) * jnn.sigmoid(u)


def _inv_bounded_scale(s: jax.Array, min_scale: float, max_scale: float) -> jax.Array:
    """Inverse of :func:`_bounded_scale`; maps a constrained scale to unconstrained space.

    Parameters
    ----------
    s : jax.Array
        Constrained scale value in ``(min_scale, max_scale)``.
    min_scale : float
        Lower bound of the constrained range.
    max_scale : float
        Upper bound of the constrained range.

    Returns
    -------
    jax.Array
        Unconstrained parameter ``u`` such that ``_bounded_scale(u) ≈ s``.
    """
    s = (s - min_scale) / (max_scale - min_scale)
    s = jnp.clip(s, 1e-6, 1 - 1e-6)
    return jnp.log(s) - jnp.log1p(-s)


def _bounded_log_scale(
    u: jax.Array, min_scale: float = 1e-2, max_scale: float = 100.0
) -> jax.Array:
    """Map an unconstrained parameter to a log-scale bounded between log(min) and log(max).

    Parameters
    ----------
    u : jax.Array
        Unconstrained parameter.
    min_scale : float, optional
        Minimum allowed scale value.  Default is ``1e-2``.
    max_scale : float, optional
        Maximum allowed scale value.  Default is 100.

    Returns
    -------
    jax.Array
        Log-scale value in ``(log(min_scale), log(max_scale))``.
    """
    log_min = jnp.log(min_scale)
    log_max = jnp.log(max_scale)
    return log_min + (log_max - log_min) * jax.nn.sigmoid(u)


# ---------------------------------------------------------------------------
# Lower-triangular index helpers
# ---------------------------------------------------------------------------


def make_lower_tri_index(D: int) -> tuple[jax.Array, jax.Array]:
    """Return row and column indices of all lower-triangular elements of a D×D matrix.

    Parameters
    ----------
    D : int
        Matrix dimension.

    Returns
    -------
    tuple of jax.Array
        ``(row_indices, col_indices)`` of length ``D*(D+1)//2``.
    """
    mask = jnp.tril(jnp.ones((D, D), dtype=bool))
    return jnp.where(mask)


def lower_tri_flat(L: jax.Array, r_idx: jax.Array, c_idx: jax.Array) -> jax.Array:
    """Extract lower-triangular elements from a batch of matrices.

    Parameters
    ----------
    L : jax.Array, shape (..., D, D)
        Batch of matrices.
    r_idx : jax.Array, shape (K,)
        Row indices of the lower-triangular elements.
    c_idx : jax.Array, shape (K,)
        Column indices of the lower-triangular elements.

    Returns
    -------
    jax.Array, shape (..., K)
        Flattened lower-triangular values.
    """
    return L[..., r_idx, c_idx]


# ---------------------------------------------------------------------------
# Jacobian helpers for raw → standardized coordinate change
# ---------------------------------------------------------------------------


def _raw2std_jacobian_single_jax(
    raw_moments: jax.Array, std_scales: jax.Array
) -> jax.Array:
    """Compute the Jacobian matrix of the raw-to-standardised coordinate transform.

    The transform is
    ``z = [(log10 Mf - mean[0])/s[0],  (Mr/Mf - mean[1])/s[1],
           (M1/Mr - mean[2])/s[2],      (M2/Mr - mean[3])/s[3]]``
    and this function returns ``J = diag(1/s) @ dz0/dx``.

    Parameters
    ----------
    raw_moments : jax.Array, shape (4,)
        Raw moments ``[Mf, Mr, M1, M2]``.
    std_scales : jax.Array, shape (4,)
        Standardisation scales (``raw2standard.std``).

    Returns
    -------
    jax.Array, shape (4, 4)
        Jacobian ``dz/dx`` incorporating the standardisation scaling.
    """
    Mf = raw_moments[0]
    Mr = raw_moments[1]
    M1 = raw_moments[2]
    M2 = raw_moments[3]
    log10 = jnp.log(10.0)
    z = 0.0

    J = jnp.array(
        [
            [1.0 / (Mf * log10), z, z, z],
            [-Mr / Mf**2, 1.0 / Mf, z, z],
            [z, -M1 / Mr**2, 1.0 / Mr, z],
            [z, -M2 / Mr**2, z, 1.0 / Mr],
        ]
    )
    S = jnp.diag(1.0 / std_scales)
    return S @ J


def propagate_cov_to_std_jax(
    C_raw: jax.Array,
    raw_moments: jax.Array,
    std_scales: jax.Array,
) -> jax.Array:
    """Propagate a raw-moment covariance to standardised coordinates.

    Applies a first- plus second-order Jacobian correction, following the
    same formula as :func:`~bfd_cnf.data.transform_dataset_to_standard`.

    Parameters
    ----------
    C_raw : jax.Array, shape (4, 4)
        Covariance matrix in raw moment space.
    raw_moments : jax.Array, shape (4,)
        Raw moments ``[Mf, Mr, M1, M2]`` at which to evaluate the Jacobian.
    std_scales : jax.Array, shape (4,)
        Standardisation scales (``raw2standard.std``).

    Returns
    -------
    jax.Array, shape (4, 4)
        Covariance matrix in standardised coordinates (first + second order).
    """
    Mf = raw_moments[0]
    Mr = raw_moments[1]
    M1 = raw_moments[2]
    M2 = raw_moments[3]
    log10 = jnp.log(10.0)
    z = 0.0

    A = _raw2std_jacobian_single_jax(raw_moments, std_scales)
    Sigma1 = A @ C_raw @ A.T

    def row4(a, b, c, d):
        return jnp.array([a, b, c, d])

    H0 = jnp.stack(
        [
            row4(-1.0 / (Mf**2 * log10), z, z, z),
            row4(z, z, z, z),
            row4(z, z, z, z),
            row4(z, z, z, z),
        ]
    )
    H1 = jnp.stack(
        [
            row4(2.0 * Mr / Mf**3, -1.0 / Mf**2, z, z),
            row4(-1.0 / Mf**2, z, z, z),
            row4(z, z, z, z),
            row4(z, z, z, z),
        ]
    )
    H2 = jnp.stack(
        [
            row4(z, z, z, z),
            row4(z, 2.0 * M1 / Mr**3, -1.0 / Mr**2, z),
            row4(z, -1.0 / Mr**2, z, z),
            row4(z, z, z, z),
        ]
    )
    H3 = jnp.stack(
        [
            row4(z, z, z, z),
            row4(z, 2.0 * M2 / Mr**3, z, -1.0 / Mr**2),
            row4(z, z, z, z),
            row4(z, -1.0 / Mr**2, z, z),
        ]
    )
    H = jnp.stack([H0, H1, H2, H3], axis=0)

    def sigma2_element(i, j):
        return 0.5 * jnp.trace(H[i] @ C_raw @ H[j] @ C_raw)

    # Compute full Sigma2 matrix
    idx = jnp.arange(4)
    ii, jj = jnp.meshgrid(idx, idx, indexing="ij")
    Sigma2 = jax.vmap(jax.vmap(sigma2_element))(ii, jj)

    S = jnp.diag(1.0 / std_scales)
    Sigma2_std = S @ Sigma2 @ S
    return Sigma1 + Sigma2_std


# ---------------------------------------------------------------------------
# RawMomentStandardize
# ---------------------------------------------------------------------------


def sas(x, p):
    """sinh-arcsinh warp of the flux coordinate.  `p = (mu, sig, a, b)`.

    `log10 Mf` is the one badly non-Gaussian chart axis on a realistic
    population -- skew 1.775 and excess kurtosis 3.807 on bulgedisc_v2, against
    0.01 and 0.01 for gauss2, which the same stack fits cleanly.  The cause is a
    hard detection cut (Mf >= 300 exactly) with a power-law tail above, and the
    strain shows up as a bulk score that rings at 0.65 noise sigma where the
    true prior's turns over every 2.4 (HANDOFF.md, 2026-08-29).

    This is the two-parameter Jones-Pewsey family on the centred coordinate
    `w = (x - mu)/sig`.  It zeroes skew AND kurtosis exactly, is C-infinity,
    monotone for `b > 0`, inverts in closed form, and -- unlike a Box-Cox power
    with lambda < 0 -- stays UNBOUNDED, so it invents no support edge at the
    bright end where there is no physics.  It does not reach a quantile
    transform (Anderson-Darling 144 against 919 now and 0.15 for the rank map):
    no smooth two-parameter family Gaussianises a truncation.  It is the cheap
    test of whether the chart is what makes the score ring.

    `p = None` is the identity, which is the default everywhere, so every
    checkpoint written before this existed still means what it used to.
    """
    mu, sig, a, b = p
    return jnp.sinh((jnp.arcsinh((x - mu) / sig) - a) / b)


def sas_inv(y, p):
    """Inverse of `sas`, in closed form."""
    mu, sig, a, b = p
    return mu + sig * jnp.sinh(b * jnp.arcsinh(y) + a)


def sas_log_deriv(x, p):
    """log dS/dx, for the chart's log-det."""
    mu, sig, a, b = p
    w = (x - mu) / sig
    return (jnp.log(jnp.cosh((jnp.arcsinh(w) - a) / b))
            - jnp.log(b) - 0.5 * jnp.log1p(w ** 2) - jnp.log(sig))


class RawMomentStandardize(AbstractBijection):
    """
    Raw x = [Mf, Mr, M1, M2, Mc]   (bfd's even-moment order)
    Transformed z0 = [log10(Mf), logit(Mr/Mf/r*), logit(Mc/Mr/rc*), M1/Mr, M2/Mr]
    Then standardized: z = (z0 - mean) / std

    Slot 1 carries the point-source ceiling `r* = POINT_SOURCE` as a HARD
    boundary: `Mr/Mf` is the galaxy's size in units of the PSF's, and `r*` is
    the PSF itself, so the true density is exactly zero at and above it.  With
    the bare ratio the flow had no way to know that and duly leaked mass past
    the edge, which is a wrong SCORE in the region every noisy target's
    convolution integral reaches through.  The logit closes the support by
    construction, at the cost of stretching the top of the population -- if
    that stretch turns out to cost more resolution than the closure buys,
    the map to try next is gentler (a power, or `-log(1 - r/r*)`), not absent.

    Slot 2 carries the identical treatment for `Mc/Mr` against its own
    point-source ceiling `rc* = POINT_SOURCE_MC`: the same argument that makes
    `r*` a hard boundary on `Mr/Mf` makes `rc*` one on `Mc/Mr`, since both are
    the pure-weight-moment ratio a point source (Itilde/T = 1) would produce.

    NOTE this makes the chart undefined for `Mr/Mf >= r*` or `Mc/Mr >= rc*`,
    which real NOISY moments do reach (~1.8% of the deep targets for slot 1).
    That is legal only because the flow is the density of the LATENT moment: a
    noisy M never enters `log_prob` directly, only through
    `P(M) = INT dm P(m) L(M-m)`, whose draws are latent points.  Anything
    evaluating the flow at measured moments must mask on `in_domain` first --
    see `in_domain` above.

    The transformed order groups the three SPIN-0 coordinates first (0, 1, 2)
    and the spin-2 pair last (3, 4); the equivariant layers and the
    permutations between them rely on that split.  Mc/Mr carries the same units
    as Mr/Mf (both are k^2), so the two size-like coordinates share a scale.

    The spin-2 standardisation is SYMMETRISED (`_effective`), which is what
    keeps the whole stack isotropic.  Every other layer already is -- `Permute`
    never touches slots 3 and 4, `Spin0AutoregressiveLayer` never reads them,
    `Spin2CouplingLayer` applies one shared scale to both, the base is
    isotropic, and `ShearResponse` is spin-covariant by construction -- so a
    per-coordinate mean and std here is the only thing in the flow that can
    pick out a direction on the sky.  It is not a harmless initialisation
    either: `Spin2CouplingLayer` has no location parameter, so the flow's spin-2
    mean is ENTIRELY this `mean[3:]`, and `bulk.train` hands both slots to Adam
    independently.  Left free, they fit the training sample's shape-noise mean
    (~1e-4 in M1/Mr at 90k galaxies) and a spurious axis ratio (~5e-3), and a
    prior with a preferred direction reads out as additive shear bias.
    """

    mean: jax.Array
    std: jax.Array
    # Static: a fixed reparameterisation of the flux axis, not a fitted leaf.
    # `None` (the default) is the plain log10 this chart has always used.
    flux_sas: tuple | None = eqx.field(static=True, default=None)

    def __init__(self, mean=None, std=None, flux_sas=None):
        self.flux_sas = None if flux_sas is None else tuple(
            float(v) for v in flux_sas)
        """Initialise with optional mean and std for the standardisation step.

        Parameters
        ----------
        mean : array-like or None, optional
            Mean of the transformed coordinates.  Defaults to zeros.
        std : array-like or None, optional
            Standard deviation of the transformed coordinates.  Defaults to
            ones.
        """
        if mean is None:
            self.mean = jnp.zeros(5)
        else:
            self.mean = jnp.asarray(mean)
        if std is None:
            self.std = jnp.ones(5)
        else:
            self.std = jnp.asarray(std)

    def _effective(self):
        """(mean, std) with the spin-2 pair forced isotropic.

        Isotropy fixes both: the mean of (M1/Mr, M2/Mr) is exactly zero, and the
        two share one scale.  Applying it HERE rather than at construction means
        the optimiser cannot undo it -- the raw `mean[3:]` is simply never read,
        and `std[3:]` is read only through the rotation-invariant combination
        sqrt((s3^2 + s4^2)/2), i.e. half the trace of the spin-2 covariance.
        """
        spin2_scale = jnp.sqrt(0.5 * (self.std[3] ** 2 + self.std[4] ** 2))
        return (self.mean.at[3:].set(0.0), self.std.at[3:].set(spin2_scale))

    @property
    def shape(self):  # type: ignore[override]
        """Shape of the bijection input/output: ``(5,)``."""
        return (5,)

    @property
    def cond_shape(self):  # type: ignore[override]
        """Conditioning shape: ``None`` (unconditional)."""
        return None

    def _forward_transform(self, x):
        """Apply the raw-to-standardised coordinate change.

        Parameters
        ----------
        x : jax.Array, shape (..., 5)
            Raw moments ``[Mf, Mr, M1, M2, Mc]``.

        Returns
        -------
        z : jax.Array, shape (..., 4)
            Standardised coordinates.
        z0 : jax.Array, shape (..., 4)
            Intermediate un-standardised transformed coordinates.
        """
        Mf, Mr, M1, M2, Mc = (x[..., 0], x[..., 1], x[..., 2],
                              x[..., 3], x[..., 4])

        z0_0 = jnp.log10(Mf)
        if self.flux_sas is not None:
            z0_0 = sas(z0_0, self.flux_sas)
        z0_1 = Mr / Mf
        z0_2 = Mc / Mr
        z0_3 = M1 / Mr
        z0_4 = M2 / Mr

        z0 = jnp.stack([z0_0, z0_1, z0_2, z0_3, z0_4], axis=-1)
        mean, std = self._effective()
        z = (z0 - mean) / std
        return z, z0

    def _inverse_transform(self, z):
        """Invert the standardised-to-raw coordinate change.

        Parameters
        ----------
        z : jax.Array, shape (..., 4)
            Standardised coordinates.

        Returns
        -------
        x : jax.Array, shape (..., 5)
            Raw moments ``[Mf, Mr, M1, M2, Mc]``.
        z0 : jax.Array, shape (..., 4)
            Intermediate un-standardised coordinates.
        """
        mean, std = self._effective()
        z0 = z * std + mean
        log10_const = jnp.log(jnp.array(10.0, dtype=z0.dtype))
        l10 = (z0[..., 0] if self.flux_sas is None
               else sas_inv(z0[..., 0], self.flux_sas))
        Mf = jnp.exp(l10 * log10_const)
        Mr = z0[..., 1] * Mf
        Mc = z0[..., 2] * Mr
        M1 = z0[..., 3] * Mr
        M2 = z0[..., 4] * Mr
        x = jnp.stack([Mf, Mr, M1, M2, Mc], axis=-1).astype(z0.dtype)
        return x, z0

    def transform_and_log_det(self, x, condition=None):
        """Transform raw moments to standardised coordinates and compute the log-abs-det.

        Parameters
        ----------
        x : jax.Array, shape (..., 4)
            Raw moments.
        condition : ignored

        Returns
        -------
        z : jax.Array, shape (..., 4)
            Standardised output.
        log_abs_det : jax.Array, scalar
            Log absolute determinant of the Jacobian.
        """
        z, _ = self._forward_transform(x)
        Mf = x[..., 0]
        Mr = x[..., 1]
        Mc = x[..., 4]
        ln10 = jnp.log(jnp.array(10.0, dtype=Mf.dtype))
        # Triangular in the order (Mf, Mr, M1, M2, Mc) for the bare-ratio
        # chart [log10 Mf, Mr/Mf, Mc/Mr, M1/Mr, M2/Mr]: the only nonzero
        # permutation gives 1/(ln10 Mf) * 1/Mf * 1/Mr * 1/Mr * 1/Mr.
        lad_geom = -(2.0 * jnp.log(Mf) + 3.0 * jnp.log(Mr) + jnp.log(ln10))
        if self.flux_sas is not None:
            # Slot 0's own factor is now d sas(log10 Mf)/dMf, i.e. the plain
            # 1/(Mf ln10) already counted above TIMES dS/d(log10 Mf).
            lad_geom = lad_geom + sas_log_deriv(jnp.log10(Mf), self.flux_sas)
        lad_std = -jnp.sum(jnp.log(self._effective()[1]))
        return z, lad_geom + lad_std

    def inverse_and_log_det(self, y, condition=None):
        """Invert the standardisation and return the log-abs-det of the inverse.

        Parameters
        ----------
        y : jax.Array, shape (..., 4)
            Standardised coordinates.
        condition : ignored

        Returns
        -------
        x : jax.Array, shape (..., 4)
            Raw moments.
        log_abs_det : jax.Array, scalar
            Log absolute determinant of the inverse Jacobian.
        """
        x, _ = self._inverse_transform(y)
        lad_fwd = self.transform_and_log_det(x)[1]
        return x, -lad_fwd


# ---------------------------------------------------------------------------
# BoundedAffine
# ---------------------------------------------------------------------------


class BoundedAffine(AbstractBijection):
    """Affine bijection ``y = scale * x + loc`` with a bounded learnable scale.

    The scale parameter is stored in unconstrained space and mapped to
    ``(min_scale, max_scale)`` via :func:`_bounded_scale` so that it remains
    strictly positive and bounded during training.

    Parameters
    ----------
    loc : ArrayLike, optional
        Location parameter.  Default is 0.
    scale : ArrayLike, optional
        Initial scale value (will be converted to the unconstrained
        parametrisation).  Default is 1.
    min_scale : float, optional
        Lower bound for the scale.  Default is ``1e-3``.
    max_scale : float, optional
        Upper bound for the scale.  Default is 10.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    scale: Array | AbstractUnwrappable[Array]
    min_scale: float = eqx.field(static=True)
    max_scale: float = eqx.field(static=True)

    def __init__(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
        *,
        min_scale: float = 1e-3,
        max_scale: float = 10.0,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.loc, scale = jnp.broadcast_arrays(
            *(arraylike_to_array(a, dtype=float) for a in (loc, scale)),
        )
        self.shape = scale.shape

        # `scale` is the constrained init; convert to unconstrained param
        unconstrained_scale = _inv_bounded_scale(scale, min_scale, max_scale)

        self.scale = Parameterize(
            lambda u: _bounded_scale(u, min_scale, max_scale),
            unconstrained_scale,
        )

    def transform_and_log_det(self, x, condition=None):
        """Apply the affine map ``y = scale * x + loc``.

        Parameters
        ----------
        x : jax.Array
            Input array.
        condition : ignored

        Returns
        -------
        y : jax.Array
            Affine-transformed output.
        log_abs_det : jax.Array, scalar
            Log absolute determinant of the Jacobian.
        """
        s = self.scale
        return x * s + self.loc, jnp.log(jnp.abs(s)).sum()

    def inverse_and_log_det(self, y, condition=None):
        """Invert the affine map ``x = (y - loc) / scale``.

        Parameters
        ----------
        y : jax.Array
            Output-space array.
        condition : ignored

        Returns
        -------
        x : jax.Array
            Input-space array.
        log_abs_det : jax.Array, scalar
            Log absolute determinant of the inverse Jacobian.
        """
        s = self.scale
        return (y - self.loc) / s, -jnp.log(jnp.abs(s)).sum()


# ---------------------------------------------------------------------------
# CoeffNet
# ---------------------------------------------------------------------------


class CoeffNet(eqx.Module):
    """Simple MLP used to generate flow coefficients from conditioner inputs.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key for weight initialisation.
    in_dim : int
        Input dimensionality.  If 0, a single constant input is used.
    out_dim : int
        Output dimensionality.
    width : int
        Hidden-layer width.
    depth : int
        Number of hidden layers.
    activation : callable, optional
        Activation function applied between layers.  Default is
        ``jax.nn.relu``.
    """

    layers: tuple

    def __init__(self, key, in_dim, out_dim, width, depth, activation=jnn.relu):
        ks = jr.split(key, depth + 1)
        h = max(in_dim, 1)
        layers = []
        for i in range(depth):
            layers.append(eqx.nn.Linear(h, width, key=ks[i]))
            layers.append(activation)
            h = width
        layers.append(eqx.nn.Linear(h, out_dim, key=ks[-1]))
        self.layers = tuple(layers)

    def __call__(self, x):
        if x.shape[-1] == 0:
            x = jnp.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
        # Every conditioner in the stack goes through here, so this one call is
        # what stops a coordinate outside the training envelope from driving the
        # loc/scale nets linearly.  See `_soft_clip`.
        x = _soft_clip(x)
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Spin0AutoregressiveLayer
# ---------------------------------------------------------------------------


class Spin0AutoregressiveLayer(AbstractBijection):
    """Autoregressive bijection on the three spin-0 components
    (flux, size, concentration).

    Transforms components 0, 1, 2 of the input vector with an affine
    autoregressive map:

    * ``z[0] = (x[0] - loc0) * exp(-s0)`` — unconditional affine.
    * ``z[1] = (x[1] - loc1(z[0])) * exp(-s1(z[0]))`` — conditioned on z[0].
    * ``z[2] = (x[2] - loc2(z[0], z[1])) * exp(-s2(z[0], z[1]))``.

    Leaves the spin-2 components 3 and 4 unchanged.  Which physical quantity
    sits in which slot is set by the permutation between layers, so each spin-0
    coordinate takes a turn as the unconditional one.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    nn_width : int
        Hidden width of the scale network.
    nn_depth : int
        Number of hidden layers in the scale network.
    activation : callable
        Activation function for the scale network.
    """

    loc0: jax.Array
    log_scale0: jax.Array
    net_ls1: CoeffNet
    net_ls2: CoeffNet

    def __init__(self, key, nn_width, nn_depth, activation):
        k1, k2 = jr.split(key, 2)
        # The head of the autoregression gets a free LOCATION as well as a
        # scale, which is what makes this an affine MAF step rather than a
        # scale-only one.  Without it the head coordinate could be stretched but
        # never re-centred -- and which coordinate is the head rotates with the
        # permutation, so the deficit followed whichever one was in slot 0.
        self.loc0 = jnp.zeros(())
        self.log_scale0 = jnp.zeros(())
        self.net_ls1 = CoeffNet(k1, 1, 2, nn_width, nn_depth, activation)
        self.net_ls2 = CoeffNet(k2, 2, 2, nn_width, nn_depth, activation)

    @property
    def shape(self):
        return (5,)

    @property
    def cond_shape(self):
        return None

    def _loc_scale(self, net, inputs):
        out = net(jnp.array(inputs))
        return out[0], _bounded_log_scale(out[1])

    def transform_and_log_det(self, x, condition=None):
        ls0 = _bounded_log_scale(self.log_scale0)
        z0 = (x[0] - self.loc0) * jnp.exp(-ls0)
        loc1, ls1 = self._loc_scale(self.net_ls1, [z0])
        z1 = (x[1] - loc1) * jnp.exp(-ls1)
        loc2, ls2 = self._loc_scale(self.net_ls2, [z0, z1])
        z2 = (x[2] - loc2) * jnp.exp(-ls2)
        y = x.at[0].set(z0).at[1].set(z1).at[2].set(z2)
        return y, -(ls0 + ls1 + ls2)

    def inverse_and_log_det(self, y, condition=None):
        ls0 = _bounded_log_scale(self.log_scale0)
        x0 = y[0] * jnp.exp(ls0) + self.loc0
        loc1, ls1 = self._loc_scale(self.net_ls1, [y[0]])
        x1 = y[1] * jnp.exp(ls1) + loc1
        loc2, ls2 = self._loc_scale(self.net_ls2, [y[0], y[1]])
        x2 = y[2] * jnp.exp(ls2) + loc2
        x = y.at[0].set(x0).at[1].set(x1).at[2].set(x2)
        return x, ls0 + ls1 + ls2


# ---------------------------------------------------------------------------
# Spin2CouplingLayer
# ---------------------------------------------------------------------------


#: Floor on |e|^2 for the bisection's log, and the radius below which the
#: inverse falls back on the q -> 0 stretch.
_Q_EPS = 1e-12

#: log1p(q) is centred on the population's mean |e|^2, which is 2 because the
#: chart standardises each spin-2 slot.
_W0 = 1.0986122886681098             # log(3) = log1p(2)


class Spin2CouplingLayer(AbstractBijection):
    """Radial bijection on the spin-2 (shape) components conditioned on spin-0.

    Acts on q = |e|^2 = x3^2 + x4^2 through a monotone map T, and carries the
    pair along it: ``y[3:] = x[3:] * sqrt(T(q)/q)``.  Only |e| moves and its
    direction never does, which is what preserves spin-2 equivariance --
    anything treating the two components differently would pick out a direction
    on the sky.

    THE POINT OF T IS THAT IT IS NOT LINEAR.  A pure shared scale (T(q) = gamma
    q, what this layer used to be) makes the model's conditional shape density
    an isotropic GAUSSIAN, exactly, for every spin-0 point -- because the spin-0
    map never reads slots 3, 4, the whole stack collapses to

        b[0:3] = F(z[0:3]),   b[3:] = z[3:] * S(z[0:3])

    and the base is N(0, I).  Then |e|^2 at fixed (Mf, Mr/Mf, Mc/Mr) is
    exponential, i.e. var/mean^2 = 1 identically, and the spin-2 score is
    forced to be exactly linear in e.  The templates are not like that.
    Measured on bulgedisc_v3 by 600-nearest-neighbour spin-0 windows,
    var(|e|^2)/mean(|e|^2)^2 is 0.61 in the data against 1.45 for
    `flows/bulk_v3.eqx`: real |e| is a narrow shell, and the old architecture
    could only err on the over-dispersed side.  That is the one direction the
    spin-2 shear signal lives in.

    The map is a pure STRETCH of the spin-2 pair -- a scale, no shift --
    whose factor is a free function of |e| as well as of the spin-0 slots:

        y[3:] = x[3:] * s,     s = exp(h(z0, z1, z2, log1p(q))),   q = |e|^2

    `h` is just the conditioner MLP with `q` added as a fourth input.  Nothing
    about the profile in |e| is assumed: no analytic family, no basis, no
    spline.  Earlier versions of this layer parameterised it (a power tail,
    then an asymptotically linear rational) and both were the wrong kind of
    choice -- the power pinned its exponent at a stability ceiling, and the
    rational's learnable bend location was a flat direction that drifted.

    The Jacobian is still the radial one: with T(q) = q s(q)^2,

        T'(q) = s^2 (1 + 2 q dh/dq),   log-det = 2h + log1p(2 q dh/dq)

    since for a radial map in d = 2 det = (R/r) R'(r), and R = sqrt(T(r^2))
    collapses that to T'.  `dh/dq` is one scalar autodiff of the conditioner.

    **A free stretch is monotone only where 2 q dh/dq > -1, and that is not
    enforced -- deliberately.**  Ellipticity does not physically fold, so the
    job is to keep the flow out of that region, not to make it unreachable at
    the cost of contorting the map everywhere else.  Two things make it safe
    to leave unconstrained:

    * **It is loud, not silent.**  A fold makes `log1p` see an argument at or
      below -1, so the log-det is NaN at the step it happens rather than a
      quietly wrong density with two preimages.  (That silent-failure mode is
      real, but only if one takes log|det|, which this does not.)
    * **`_soft_clip` already handles the far field.**  Outside the training
      envelope the conditioner's inputs saturate, so `h` goes CONSTANT in q,
      `dh/dq -> 0`, and `T' -> s^2 > 0` on its own.  Folding is therefore
      confined to the region the data actually occupies, which is the region
      training can see and a diagnostic can check.

    `check_monotone` below reports the violated fraction; if a fit ever shows
    one, the fix is a hinge penalty on `1 + 2 q dh/dq` in `bulk.train`, not a
    change of parameterisation.

    The inverse has no closed form -- T(q) = q exp(2h(q)) is transcendental --
    so `shear` does a bisection on log q.  T is increasing wherever the map is
    valid, so that is unconditionally convergent, and it costs only `sample`:
    `log_prob`, which is every training step and every `bias.py` draw, uses
    the forward direction and is untouched.

    **Exactly one layer in the stack may set `bend`, and `bulk.build_flow`
    gives it to the FIRST one.**  First, because the spline's `interval` is a
    fixed range in |e|^2 and only the data-adjacent layer sees a coordinate
    whose scale is known -- there |e|^2 has mean 2 (the chart standardises the
    spin-2 slots) and tops out near 54.  Deeper in the stack the preceding
    scales have moved it somewhere the interval no longer matches.  One layer,
    because the bend COMPOSES MULTIPLICATIVELY: N stacked layers multiply their
    slope ratios, so bounding each one bounds the stack only by the Nth power.  Left free, all eight bent and pinned at the
    ceiling -- measured on the first 150k-step bulk fit, when this was a power
    family, per-layer median `a` of 2.97, 3.85, 2.04, 3.83, 3.65, 1.31, 0.31,
    0.33, product 43 and up to 2746, i.e. the stack mapped |e|^2 to |e|^86 in
    the tail.  That is finite on the training set it was fit to and a cliff
    just outside it: the next stage (shear) nudges a point outward, the
    coordinate overflows, the frozen bulk answers -inf, and the NLL is `inf`
    from step 21.  One bend is also all the data asks for -- going from the old
    architecture's forced var/mean^2 = 1 to the measured 0.57 needs a ~ 2, well
    inside a single layer's range.  The other layers keep the pure scale, which
    is what they always were.

    No change to the layer ORDERING is needed, and that is the reason for this
    particular design.  `Spin0AutoregressiveLayer` still reads only slots 0-2
    and `Permute` still never touches 3, 4, so the composite Jacobian is block
    lower triangular exactly as before and the two log-dets still add.  Nor
    does the spin-0 conditioner need |e|^2: the stack already factorises as
    p(spin-0) p(e | spin-0) with a fully general 3-D MAF for the first factor,
    so ALL the missing generality was in the second, and it is here.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    nn_width : int
        Hidden width of the coupling network.
    nn_depth : int
        Number of hidden layers.
    activation : callable
        Activation function.
    """

    net: CoeffNet
    bend: bool = eqx.field(static=True)

    def __init__(self, key, nn_width, nn_depth, activation, bend=False):
        # ONLY ONE LAYER IN THE STACK MAY BEND -- see the class docstring.
        # Bending adds log1p(q) as a fourth conditioner input; without it this
        # is the plain spin-0-conditioned scale the layer has always been.
        self.bend = bool(bend)
        self.net = CoeffNet(key, 4 if bend else 3, 1, nn_width, nn_depth,
                            activation)

    @property
    def shape(self):
        return (5,)

    @property
    def cond_shape(self):
        return None

    def _h(self, x, q):
        """log s, the log stretch, from the spin-0 slots and (if bending) q."""
        u = x[:3] if not self.bend else jnp.concatenate(
            [x[:3], (jnp.log1p(q) - _W0)[None]])
        # (1e-4, 1e4) on the SQUARED stretch, i.e. s in (0.01, 100).  Wider
        # than `_bounded_log_scale`'s default, which left 1.59% of templates
        # sitting on the ceiling -- and it is a sigmoid, so a coefficient
        # that reaches it cannot come back.  The layer's original
        # log(1-c) - log_scale form had a comparable range.
        return _bounded_log_scale(self.net(u)[0], 1e-4, 1e4) * 0.5

    def _stretch(self, x, q):
        """(s, log T'(q)) at |e|^2 = q.  `log T'` IS the 5x5 log-det."""
        if not self.bend:
            h = self._h(x, q)
            return jnp.exp(h), 2.0 * h
        h, dh = jax.value_and_grad(lambda t: self._h(x, t))(q)
        return jnp.exp(h), 2.0 * h + jnp.log1p(2.0 * q * dh)

    def transform_and_log_det(self, x, condition=None):
        q = x[3] ** 2 + x[4] ** 2
        s, lad = self._stretch(x, q)
        return x.at[3].set(x[3] * s).at[4].set(x[4] * s), lad

    def inverse_and_log_det(self, y, condition=None):
        w = y[3] ** 2 + y[4] ** 2
        if not self.bend:
            s, lad = self._stretch(y, w)          # q-independent, so w serves
            inv = 1.0 / s
            return y.at[3].set(y[3] * inv).at[4].set(y[4] * inv), -lad
        # T(q) = q exp(2 h(q)) = w has no closed form; bisect on log q.  T is
        # increasing wherever the map is valid, so this cannot fail to bracket.
        # 48 halvings of a 32-decade bracket is far below float32 resolution.
        logw = jnp.log(jnp.maximum(w, _Q_EPS))

        def step(bounds, _):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            q = jnp.exp(mid)
            below = 2.0 * self._h(y, q) + mid < logw     # log T(q) < log w
            return (jnp.where(below, mid, lo), jnp.where(below, hi, mid)), None

        (lo, hi), _ = jax.lax.scan(
            step, (jnp.float32(-37.0) + 0.0 * logw, jnp.float32(37.0) + 0.0 * logw),
            None, length=48)
        q = jnp.exp(0.5 * (lo + hi))
        s, lad = self._stretch(y, q)
        inv = jnp.where(w > _Q_EPS, jnp.sqrt(q / jnp.maximum(w, _Q_EPS)), 1.0 / s)
        return y.at[3].set(y[3] * inv).at[4].set(y[4] * inv), -lad


def check_monotone(layer, x):
    """Fraction of rows where the stretch folds, i.e. 1 + 2 q dh/dq <= 0.

    The one thing the free parameterisation does not guarantee.  Physically
    ellipticity never folds, so this should be exactly 0; if a fit ever shows
    otherwise, add a hinge penalty on the same quantity in `bulk.train` rather
    than constraining the map.  `x` is a batch of the layer's own input
    coordinates.
    """
    def one(v):
        q = v[3] ** 2 + v[4] ** 2
        dh = jax.grad(lambda t: layer._h(v, t))(q)
        return 1.0 + 2.0 * q * dh

    return float(jnp.mean(jax.vmap(one)(x) <= 0.0))


# ---------------------------------------------------------------------------
# EquivariantAutoregressiveLayer
# ---------------------------------------------------------------------------


class EquivariantAutoregressiveLayer(AbstractBijection):
    """Composed bijection combining :class:`Spin0AutoregressiveLayer` and :class:`Spin2CouplingLayer`.

    The spin-0 layer transforms components 0–2 autoregressively and the
    spin-2 layer then transforms components 3–4 conditioned on the spin-0
    output.  Together they implement a fully equivariant normalizing-flow
    layer for galaxy moments.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key (split internally for the two sublayers).
    nn_width : int
        Hidden width for both sub-networks.
    nn_depth : int
        Number of hidden layers for both sub-networks.
    activation : callable
        Activation function for both sub-networks.
    """

    spin0: Spin0AutoregressiveLayer
    spin2: Spin2CouplingLayer

    def __init__(self, key, nn_width, nn_depth, activation, bend=False):
        k0, k2 = jr.split(key)
        self.spin0 = Spin0AutoregressiveLayer(k0, nn_width, nn_depth, activation)
        self.spin2 = Spin2CouplingLayer(k2, nn_width, nn_depth, activation,
                                        bend=bend)

    @property
    def shape(self):
        return (5,)

    @property
    def cond_shape(self):
        return None

    def transform_and_log_det(self, x, condition=None):
        x, lad0 = self.spin0.transform_and_log_det(x)
        x, lad2 = self.spin2.transform_and_log_det(x)
        return x, lad0 + lad2

    def inverse_and_log_det(self, y, condition=None):
        y, lad2 = self.spin2.inverse_and_log_det(y)
        y, lad0 = self.spin0.inverse_and_log_det(y)
        return y, lad0 + lad2


# ---------------------------------------------------------------------------
# ExplicitPolyLast
# ---------------------------------------------------------------------------


class ExplicitPolyLast(AbstractBijection):
    """Conditional last-layer bijection using explicit polynomial shear features.

    Implements an affine coupling layer whose scale and shift coefficients
    are polynomials in the shear components ``(g1, g2)`` with network-
    generated coefficients conditioned on the transformed spin-0 inputs.
    Optionally includes quadratic (``|g|²``) and cross-term (``g1·g2``)
    features.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    dim : int
        Input dimensionality (should be 4 for galaxy moments).
    raw_cond_dim : int
        Conditioning dimensionality (must be ≥ 2 for ``[g1, g2]``; extra
        dimensions such as Σ_X parameters are silently ignored).
    use_quadratic : bool
        Whether to include ``|g|²`` as an even feature.
    use_cross : bool
        Whether to include the ``g1 * g2`` cross term.
    last_width : int
        Hidden width for the coefficient networks.
    last_depth : int
        Number of hidden layers.
    activation : callable
        Activation function.
    """

    dim: int = eqx.field(static=True)
    raw_cond_dim: int = eqx.field(static=True)
    nets: tuple
    n_even: int = eqx.field(static=True)
    n_odd: int = eqx.field(static=True)
    use_quadratic: bool = eqx.field(static=True)
    use_cross: bool = eqx.field(static=True)

    def __init__(
        self,
        key,
        dim,
        raw_cond_dim,
        use_quadratic,
        use_cross,
        last_width,
        last_depth,
        activation,
    ):
        self.dim = dim
        self.raw_cond_dim = raw_cond_dim
        self.use_quadratic = use_quadratic
        self.use_cross = use_cross
        self.n_even = 1 + (1 if use_quadratic else 0)
        self.n_odd = 2
        k_list = jr.split(key, 3)
        self.nets = tuple(
            [
                CoeffNet(k_list[0], 0, self.n_even, last_width, last_depth, activation),
                CoeffNet(k_list[1], 1, self.n_even, last_width, last_depth, activation),
                CoeffNet(
                    k_list[2], 2, 1 + self.n_even, last_width, last_depth, activation
                ),
            ]
        )

    def _features(self, g):
        # reads only condition[:2] — extra dims from Σ_X are silently ignored
        g1 = g[..., 0:1]
        g2 = g[..., 1:2]
        even_feats = [jnp.ones(g.shape[:-1] + (1,), dtype=g.dtype)]
        if self.use_quadratic:
            even_feats.append(g1**2 + g2**2)
        even = jnp.concatenate(even_feats, axis=-1)
        odd = jnp.concatenate([g1, g2], axis=-1)
        return even, odd

    @property
    def shape(self):
        return (self.dim,)

    @property
    def cond_shape(self):
        return (self.raw_cond_dim,)

    def transform_and_log_det(self, x, condition=None):
        even_feats, odd_feats = self._features(condition)
        y_parts = []
        log_scales_parts = []
        for i in range(2):
            ls = _bounded_log_scale(
                jnp.sum(self.nets[i](x[..., :i]) * even_feats, axis=-1)
            )
            y_parts.append(x[..., i] * jnp.exp(-ls))
            log_scales_parts.append(ls)
        x0, x1 = x[..., 0], x[..., 1]
        coeff_vec2 = self.nets[2](jnp.stack([x0 + x1, x0 * x1], axis=-1))
        ls2 = _bounded_log_scale(jnp.sum(coeff_vec2[..., 1:] * even_feats, axis=-1))
        shared_c = coeff_vec2[..., 0]
        y_parts.append((x[..., 2] - shared_c * odd_feats[..., 0]) * jnp.exp(-ls2))
        y_parts.append((x[..., 3] - shared_c * odd_feats[..., 1]) * jnp.exp(-ls2))
        log_scales_parts.extend([ls2, ls2])
        return jnp.stack(y_parts, axis=-1), -jnp.sum(
            jnp.stack(log_scales_parts, axis=-1), axis=-1
        )

    def inverse_and_log_det(self, y, condition=None):
        even_feats, odd_feats = self._features(condition)
        x = jnp.zeros_like(y)
        lad = jnp.zeros(y.shape[:-1])
        for i in range(2):
            ls = _bounded_log_scale(
                jnp.sum(self.nets[i](x[..., :i]) * even_feats, axis=-1)
            )
            x = x.at[..., i].set(y[..., i] * jnp.exp(ls))
            lad = lad + ls
        x0, x1 = x[..., 0], x[..., 1]
        coeff_vec2 = self.nets[2](jnp.stack([x0 + x1, x0 * x1], axis=-1))
        ls2 = _bounded_log_scale(jnp.sum(coeff_vec2[..., 1:] * even_feats, axis=-1))
        shared_c = coeff_vec2[..., 0]
        x = x.at[..., 2].set(y[..., 2] * jnp.exp(ls2) + shared_c * odd_feats[..., 0])
        x = x.at[..., 3].set(y[..., 3] * jnp.exp(ls2) + shared_c * odd_feats[..., 1])
        lad = lad + ls2 + ls2
        return x, lad


def sigmax_log_scale_stats(sigma_x):
    """Mean/std of `log_scale = 0.5 * log det(Sigma_X)`, for `SigmaXCouplingLayer`.

    MUST be computed from the TARGET catalog's own Sigma_X (e.g. bias.py's
    `cov_odd` column) -- never from the training templates behind
    `bulk.build_flow`, which are noiseless and so carry no Sigma_X at all. A
    guessed placeholder here is exactly the kind of stale hardcoded constant
    `POINT_SOURCE` already burned once.
    """
    log_scale = 0.5 * jnp.linalg.slogdet(jnp.asarray(sigma_x))[1]
    return float(log_scale.mean()), float(log_scale.std())


# SigmaXCouplingLayer
# ---------------------------------------------------------------------------


def _zero_last_layer(net: CoeffNet) -> CoeffNet:
    """Return ``net`` with its final Linear layer zeroed (so it outputs 0 at init)."""
    last = net.layers[-1]
    return eqx.tree_at(
        lambda n: (n.layers[-1].weight, n.layers[-1].bias),
        net,
        (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)),
    )


class SigmaXCouplingLayer(AbstractBijection):
    """Centroid-covariance (C_X) conditioned coupling bijection.

    The final, conditional layer of the prior flow.  It carries the entire
    dependence of the moment prior on the target's **centroid uncertainty** C_X
    (the integral over the unknown centroid), conditioned on
    ``[g1, g2, log_scale, e1, e2]`` with ``log_scale = ½ log det C_X`` and
    ``(e1, e2)`` the ellipticity of C_X.  ``g1, g2`` are ignored (shear is handled
    by :class:`ShearTaylorLast`).

    The transform is a *physically constrained* map — it allows only the changes
    the centroid marginalisation produces to leading order (analogous to
    :class:`ShearTaylorLast` for shear, but a different effect):

    * **Flux** ``m0`` → translation ``m0 + s0`` (the C_X-dependent flux-mean tilt).
    * **Size** ``m1`` → *locked* scaling ``κ·m1 + c1·(κ−1)``, ``κ = exp(g_s)``,
      ``c1 = μ1/σ1``.  This equals ``κ × (Mr/Mf)`` on the un-centred ratio, so the
      multiplicative knob produces the physical size *mean* shift.
    * **Ellipticity** ``(m2, m3)`` → ``(1/κ)·[(I + c·E)·(m2, m3) + D·(e1, e2)]``:
      the **same** ``κ`` (reciprocal lock — size inflation shrinks the ellipticity
      ratio), an additive **dipole** ``D·(e1, e2)`` (leading O(e) centring bias),
      and an anisotropic-broadening **quadrupole** ``c·E``,
      ``E = [[e1, e2], [e2, −e1]]`` (O(C_X²)).

    Conditioning: the locked scale ``g_s`` depends on flux + C_X only (preserving a
    closed-form inverse); the directional terms ``D, c`` additionally depend on
    size ``m1`` (where the small size-dependent corrections live).  Nets are
    zero-initialised, so the layer starts at the identity (a small perturbation).

    Analytic C_X scaling: the leading marginalisation response is known in closed
    form — the mean shifts ``s0, g_s, D`` are ``O(tr C_X)`` and the broadening ``c``
    is ``O(tr C_X²)``.  We factor ``T_n ∝ exp(log_scale)`` (and ``T_n²``) out of the
    coeff nets, so each net only learns the slowly-varying ``(flux, size)`` form
    factor.  NOTE: factoring T out spreads the (already weak, ``|e|≤e_max``) dipole
    gradient across ``log_scale`` and dropped its SNR — empirically it killed the
    dipole at 550k steps unless paired with the fixed e-ring stencil below.

    ``log|det J| = −g_s + log(1 − c²|e|²)``.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    nn_width, nn_depth : int, optional
        Coupling-network hidden width / depth.  Defaults 32, 2.
    activation : callable, optional
        Activation.  Default ``jax.nn.silu``.
    full_cond_dim : int, optional
        Conditioning dimension (5 for ``[g1, g2, log_scale, e1, e2]``).
    log_scale_mean, log_scale_std : float, optional
        Normalisation of ``log_scale`` for the network inputs.
    e_max : float, optional
        Reference ellipticity magnitude; ``|e|²`` is normalised by ``e_max²``.
        Default 0.2.
    size_loc : float, optional
        The constant ``c1 = μ1/σ1`` for the locked size transform.  Default 0.
    g_s_max : float, optional
        Bound ``|g_s| ≤ g_s_max`` keeping ``κ`` in a sane range.  Default 1.
    """

    net_flux: CoeffNet  # (log_scale_n, ehat2)          → (1,)  s0  flux shift
    net_size: CoeffNet  # (m0, log_scale_n, ehat2)      → (1,)  g_s size log-scale
    net_dipquad: (
        CoeffNet  # (m0, m1, log_scale_n, ehat2)  → (2,)  D, c  dipole + quadrupole (pre-tanh), shared trunk
    )
    _cond_dim: int = eqx.field(static=True)
    _log_scale_mean: float = eqx.field(static=True)
    _log_scale_std: float = eqx.field(static=True)
    _e_mag_sq_scale: float = eqx.field(static=True)
    _size_loc: float = eqx.field(static=True)
    _g_s_max: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        nn_width=32,
        nn_depth=2,
        activation=jnn.silu,
        full_cond_dim=5,
        log_scale_mean=12.0,
        log_scale_std=3.0,
        e_max=0.2,
        size_loc=0.0,
        g_s_max=1.0,
    ):
        self._cond_dim = full_cond_dim
        self._log_scale_mean = float(log_scale_mean)
        self._log_scale_std = float(log_scale_std)
        self._e_mag_sq_scale = float(e_max**2)
        self._size_loc = float(size_loc)
        self._g_s_max = float(g_s_max)
        k0, k1, k2 = jr.split(key, 3)
        nets = [
            CoeffNet(k0, 2, 1, nn_width, nn_depth, activation),  # net_flux
            CoeffNet(k1, 3, 1, nn_width, nn_depth, activation),  # net_size
            CoeffNet(k2, 4, 2, nn_width, nn_depth, activation),  # net_dipquad: (D, c)
        ]
        # Zero each net's final layer → all outputs start at 0, so the layer is the
        # identity at init (κ=1, D=0, c=0, s0=0): training starts from a small
        # perturbation of the C_X-independent base shape.
        nets = [_zero_last_layer(n) for n in nets]
        self.net_flux, self.net_size, self.net_dipquad = nets

    @property
    def shape(self):
        return (4,)

    @property
    def cond_shape(self):
        return (self._cond_dim,)

    def _unpack(self, condition):
        log_scale = condition[2]
        e1 = condition[3]
        e2 = condition[4]
        e_mag_sq = e1**2 + e2**2
        log_scale_n = log_scale - self._log_scale_mean
        e_mag_sq_n = e_mag_sq / (self._e_mag_sq_scale + 1e-8)
        # T_n ∝ tr(C_X) = exp(log_scale)/exp(mean) (=1 at the reference log_scale_n=0).
        # Leading C_X response is analytic: mean shifts (s0, g_s, D) are O(T), the
        # anisotropic broadening (c) is O(T²).  Factoring these powers of T out of the
        # coeff nets leaves them learning only the (flux,size) form factor.
        # ponytail: dropped the 1/√(1−|e|²) factor in T (≤1.001 at e_max=0.05).
        T_n = jnp.exp(log_scale_n)
        return log_scale_n, e1, e2, e_mag_sq, e_mag_sq_n, T_n

    def _s0(self, log_scale_n, e_mag_sq_n, T_n):
        # flux shift s0 (C_X only). Analytic scaling: s0 ∝ T.
        return T_n * self.net_flux(jnp.array([log_scale_n, e_mag_sq_n]))[0]

    def _g_s(self, x0, log_scale_n, e_mag_sq_n, T_n):
        # size log-scale g_s (flux + C_X, bounded). Analytic scaling: g_s ∝ T.
        return self._g_s_max * jnn.tanh(
            T_n * self.net_size(jnp.array([x0, log_scale_n, e_mag_sq_n]))[0]
        )

    def _dip_quad(self, x0, x1, log_scale_n, e_mag_sq_n, T_n):
        # dipole D and quadrupole c (flux + size + C_X), shared-trunk net.
        # Analytic scaling: D ∝ T, c ∝ T². |c| < 1 ⇒ det(I+cE)=1-c²|e|² > 0
        dq_in = jnp.array([x0, x1, log_scale_n, e_mag_sq_n])
        D_raw, c_raw = self.net_dipquad(dq_in)
        return T_n * D_raw, jnn.tanh(T_n**2 * c_raw)

    def _coeffs(self, x0, x1, log_scale_n, e_mag_sq_n, T_n):
        # Analytic C_X scaling factored out (s0,g_s,D ∝ T; c ∝ T²); nets learn the
        # (flux,size) form factor only.
        s0 = self._s0(log_scale_n, e_mag_sq_n, T_n)
        g_s = self._g_s(x0, log_scale_n, e_mag_sq_n, T_n)
        D, c = self._dip_quad(x0, x1, log_scale_n, e_mag_sq_n, T_n)
        return s0, g_s, D, c

    def transform_and_log_det(self, x, condition=None):
        log_scale_n, e1, e2, e_mag_sq, e_mag_sq_n, T_n = self._unpack(condition)
        s0, g_s, D, c = self._coeffs(x[0], x[1], log_scale_n, e_mag_sq_n, T_n)
        kappa = jnp.exp(g_s)
        c1 = self._size_loc

        # flux: translation;  size: locked scale + loc-shift (= κ × Mr/Mf)
        y0 = x[0] + s0
        y1 = kappa * x[1] + c1 * (kappa - 1.0)
        # ellipticity: (1/κ)[(I + c·E)(x2,x3) + D·(e1,e2)],  E=[[e1,e2],[e2,-e1]]
        m2 = (1.0 + c * e1) * x[2] + c * e2 * x[3] + D * e1
        m3 = c * e2 * x[2] + (1.0 - c * e1) * x[3] + D * e2
        y2 = m2 / kappa
        y3 = m3 / kappa

        # block-triangular Jacobian: 0 (flux) + g_s (size) − 2g_s + log det(I+cE)
        log_det = -g_s + jnp.log(1.0 - c**2 * e_mag_sq)
        return jnp.stack([y0, y1, y2, y3]), log_det

    def inverse_and_log_det(self, y, condition=None):
        log_scale_n, e1, e2, e_mag_sq, e_mag_sq_n, T_n = self._unpack(condition)

        # s0, g_s depend only on the recoverable x0 (and x1 for D, c) ⇒ closed form.
        s0 = self._s0(log_scale_n, e_mag_sq_n, T_n)
        x0 = y[0] - s0
        g_s = self._g_s(x0, log_scale_n, e_mag_sq_n, T_n)
        kappa = jnp.exp(g_s)
        c1 = self._size_loc
        x1 = (y[1] - c1 * (kappa - 1.0)) / kappa

        D, c = self._dip_quad(x0, x1, log_scale_n, e_mag_sq_n, T_n)

        # invert (y2,y3) = (1/κ)[(I+cE)(x2,x3) + D·e]  ⇒
        #   (x2,x3) = (I+cE)^{-1}[κ(y2,y3) − D·e],  (I+cE)^{-1} = (I−cE)/(1−c²|e|²)
        r2 = kappa * y[2] - D * e1
        r3 = kappa * y[3] - D * e2
        det_e = 1.0 - c**2 * e_mag_sq
        x2 = ((1.0 - c * e1) * r2 - c * e2 * r3) / det_e
        x3 = (-c * e2 * r2 + (1.0 + c * e1) * r3) / det_e

        log_det = g_s - jnp.log(1.0 - c**2 * e_mag_sq)
        return jnp.stack([x0, x1, x2, x3]), log_det


# ---------------------------------------------------------------------------
# cx_to_sx_cond / _sx_cond_to_CX
# ---------------------------------------------------------------------------
#
# Ported from `models/flows.py` on the `working` branch (verbatim), where they
# convert between a 2x2 centroid noise covariance C_X and the
# [log_scale, e1, e2] condition vector SigmaXCouplingLayer/SigmaXBlockLayer
# read. `SigmaXBlockLayer._unpack` in THIS repo builds CX itself from the
# [g1, g2, C00, C01, C11] condition `bulk.build_flow` actually supplies (see
# `bias.condition`) and calls `cx_to_sx_cond` to get [log_scale, e1, e2].


def cx_to_sx_cond(CX: jax.Array) -> jax.Array:
    """Convert a 2x2 centroid noise covariance C_X to [log_scale, e1, e2].

    Inverse of :func:`_sx_cond_to_CX`.  The parameterisation is::

        C_X = (T/2) * [[1+e1, e2], [e2, 1-e1]]

    so that::

        log_scale = 0.5 * log det(C_X)
        e1 = (C_X[0,0] - C_X[1,1]) / (C_X[0,0] + C_X[1,1])
        e2 = 2 * C_X[0,1]           / (C_X[0,0] + C_X[1,1])

    Parameters
    ----------
    CX : jax.Array, shape (2, 2)
        Centroid noise covariance (positive definite, symmetric).

    Returns
    -------
    jax.Array, shape (3,)
        ``[log_scale, e1, e2]`` condition vector.
    """
    _, logdet = jnp.linalg.slogdet(CX)
    log_scale = 0.5 * logdet
    trace = CX[0, 0] + CX[1, 1]
    e1 = (CX[0, 0] - CX[1, 1]) / jnp.maximum(trace, 1e-30)
    e2 = 2.0 * CX[0, 1]        / jnp.maximum(trace, 1e-30)
    return jnp.array([log_scale, e1, e2])


def _sx_cond_to_CX(sx_cond: jax.Array) -> jax.Array:
    """Reconstruct the 2x2 noise covariance C_X from [log_scale, e1, e2].

    Inverse of :func:`cx_to_sx_cond`.  C_X is reconstructed as

        C_X = (T/2) * [[1+e1, e2], [e2, 1-e1]]

    with  ``T/2 = exp(log_scale) / sqrt(1 - |e|**2)``.

    Parameters
    ----------
    sx_cond : jax.Array, shape (3,)
        Condition vector ``[log_scale, e1, e2]``.

    Returns
    -------
    jax.Array, shape (2, 2)
        Positive-definite noise covariance matrix.
    """
    log_scale, e1, e2 = sx_cond[0], sx_cond[1], sx_cond[2]
    e_mag_sq = e1**2 + e2**2
    half_T = jnp.exp(log_scale) / jnp.sqrt(jnp.maximum(1.0 - e_mag_sq, 1e-8))
    return half_T * jnp.array([[1.0 + e1, e2], [e2, 1.0 - e1]])


# ---------------------------------------------------------------------------
# SigmaXBlockLayer's coefficient-net input clip
# ---------------------------------------------------------------------------
#
# Ported verbatim from `working` (there it lives near `ShearTaylorLast`, whose
# coefficient nets have the identical off-manifold extrapolation problem).
# `CoeffNet.__call__` in THIS repo already soft-clips its full packed input
# vector at `_COND_MAX = 20` before the MLP; `_bound_coeff_input` additionally
# hard-clips the individual (x0, x1) scalars at 8 before they are packed. The
# two are redundant in the sense that either alone stops a runaway, but not
# equivalent: the hard clip here fires first (bound=8 < 20) and is what the
# original SigmaXBlockLayer algebra was written and tested against, so it is
# kept rather than dropped as "already covered".
_SIGMAX_COEFF_INPUT_CLIP = 8.0


def _bound_coeff_input(x):
    return jnp.clip(x, -_SIGMAX_COEFF_INPUT_CLIP, _SIGMAX_COEFF_INPUT_CLIP)


class SigmaXBlockLayer(AbstractBijection):
    """Centroid-covariance (C_X) conditioned bijection -- flux shift is aware
    of the galaxy's own (final) ellipticity, closed-form throughout.

    HISTORY: an earlier version let a single unconstrained net see the FULL
    moment vector for all four coefficients (s0,g_s,D,c), with the log-det via
    ``jax.jacfwd`` and the INVERSE via an unrolled Newton solve. Training loss
    exploded. ``SigmaXCouplingLayer``'s invertibility is an ALGEBRAIC GUARANTEE
    -- ``kappa=exp(g_s)`` is tanh-bounded away from 0/inf and ``|c|<1`` via
    tanh, so ``1-c^2|e|^2>0`` always, no matter how training moves the weights.
    Nothing analogous bounded the unconstrained net's CROSS-derivatives
    (``d s0/d x1``, ``d g_s/d x2``, ...), so ``det(J)`` could drift toward 0 or
    negative with nothing to stop it, breaking Newton's linear solve. Replaced
    with this fully closed-form design (project feedback: prefer analytic
    transforms/Jacobians over iterative solvers).

    Mirrors :class:`ShearTaylorLast`'s OWN fix for the identical problem
    (``net_flux_e``/``net_size_e``, "flux/size blind to own ellipticity"): the
    ellipticity sub-transform (dipole ``D``, quadrupole ``c``) is UNCHANGED
    from ``SigmaXCouplingLayer`` -- it only reads the RAW ``(x0,x1)``, always
    available immediately in both directions -- so it is computed FIRST. Its
    output ``(y3,y4)`` (the ellipticity pair in THIS chart -- see the 5-D
    EXTENSION note below for the exact slot assignment) then feeds, as the
    bounded invariant ``L=log1p(y3^2+y4^2)`` (same saturating form
    ShearTaylorLast uses -- its gradient w.r.t. ``y3,y4`` VANISHES as ``|y|``
    grows, so this input can't blow up the flux Jacobian row even for outlier
    ellipticities), a correction to the flux shift ``s0``. Safe with NO
    circularity in either direction:

    * forward: ``(y3,y4)`` come from ``x0,x1`` (given) alone, computable
      before ``y0`` needs them.
    * inverse: ``(y3,y4)`` are literally part of the given ``y`` -- no
      recovery needed at all -- so the correction is immediate; ``x0`` comes
      out via one subtraction, then ``x1``, then finally ``(x3,x4)`` via the
      SAME closed-form ellipticity inverse ``SigmaXCouplingLayer`` uses.

    ``g_s`` (size log-scale) is deliberately left UNCHANGED (depends on ``x0``
    only): letting it read the final ellipticity would need
    ``kappa=exp(g_s)`` to determine that same ellipticity's value
    (``y3,y4 = .../kappa``) -- a genuine circular dependency, unlike flux's
    case.

    The transform and inverse are both fully explicit algebra (no iterative
    solver anywhere). The log-det is obtained via ``jax.jacfwd`` + ``slogdet``
    on the resulting Jacobian -- EXACT autodiff of a CLOSED-FORM expression
    (not an approximation, and not iterative like the Newton solve it
    replaces). The flux row is no longer block-sparse (``s0`` now depends on
    all of ``x`` through the ellipticity chain), so a fully hand-derived
    analytic determinant is possible but was not attempted here, to avoid
    rushing another subtle algebra bug under the same time pressure that
    produced the Newton version -- happy to do that pass separately if
    wanted. ``s0``'s ellipticity correction is itself tanh-bounded (see
    :meth:`_s0`) so its contribution to the Jacobian stays controlled the same
    way ``kappa``/``c`` are, rather than being an unconstrained linear term.

    5-D EXTENSION -- and the slot assignment this class actually operates on.
    This layer sits behind ``RawMomentStandardize`` in ``build_flow``'s chain
    (``Chain([raw2standard, *head, *bulk])``), so ``x``/``y`` here are that
    chart's OUTPUT z-coordinates, not raw moments and NOT the naive
    ``(Mf,Mr,e1,e2,Mc)`` ordering an earlier version of this docstring wrongly
    assumed. Straight from ``RawMomentStandardize._forward_transform``:
    ``z0=log10(Mf)``, ``z1=Mr/Mf`` (logit), ``z2=Mc/Mr`` (logit), ``z3=M1/Mr``,
    ``z4=M2/Mr``. So ``x2``/``y2`` is concentration (``Mc/Mr``), and the
    ellipticity pair is ``(x3,x4)``/``(y3,y4)`` -- ONE SLOT LATER than the
    ``working``-branch 4-D chart (``e1,e2`` at 2,3) this class was originally
    written against, and Mc is at index 2, not 4. Both `_ellipticity` (dipole
    `D`, quadrupole `c`) and `_mc_shift` are wired to these corrected indices
    below; a version of this class briefly had them swapped (`_ellipticity`
    reading/writing `(x2,x3)` -- i.e. mixing concentration with one
    ellipticity component -- and `_mc_shift` writing `x4`, the OTHER
    ellipticity component, not `Mc` at all), fixed once the mismatch with the
    real chart order was found.

    Index 2 (``Mc/Mr``) gets its own additive shift, ``y[2] = x[2] +
    mc_shift``, structured exactly like the flux shift's base term ``s0`` --
    ``O(tr C_X)`` leading order (``mc_shift = T_n * net_mc(...)``), zero-init
    last layer (identity at init, same convention as every other coefficient
    here). This is deliberately NOT the Gaussian-ansatz-derived correction
    `models/centroid.py`'s `CentroidMarginalize` uses (`g(Mf,Mr,M1,M2) =
    (2*Mr^2+M1^2+M2^2)/Mf`, a closed-form identity that assumes a Gaussian
    profile) -- that formula bakes in exactly the profile-shape assumption
    this layer's whole design (a learned coefficient, no assumed profile)
    exists to avoid. `net_mc` reads ONLY the galaxy's own flux `x0` (plus the
    population-level noise trace/ellipticity, like `net_flux`) and NOT `x2`
    itself, specifically so the inverse has no circularity: `x0` is already
    recovered by the time `x2` needs to be un-shifted, so `x[2] = y[2] -
    mc_shift(x0, ...)` is a plain subtraction, not an implicit equation.
    `mc_shift` depends on `x0` (an output-independent input, already used by
    every other coefficient here) so it adds one more off-diagonal Jacobian
    entry (`d(y2)/d(x0) != 0`); no special-casing is needed because the
    log-det is already the full ``jax.jacfwd`` + ``slogdet`` of
    ``_raw_transform``, not a hand-derived block form.

    ``condition`` here is ``[g1, g2, C00, C01, C11]`` (the raw Sigma_X
    covariance components -- see ``bias.condition``), not the
    ``[g1, g2, log_scale, e1, e2]`` this class's ``working``-branch ancestor
    (and ``SigmaXCouplingLayer`` in this file) expects.  ``_unpack`` does the
    ``CX -> [log_scale, e1, e2]`` conversion via :func:`cx_to_sx_cond`
    locally, so every caller keeps passing the raw covariance.
    """

    net_flux: CoeffNet     # (x0, log_scale_n, ehat2)      -> (1,)  s0 base (now reads the galaxy's own flux, see _s0/_solve_x0)
    net_size: CoeffNet     # (x0, log_scale_n, ehat2)      -> (1,)  g_s (UNCHANGED, no owne correction)
    net_dipquad: CoeffNet  # (x0, x1, log_scale_n, ehat2)  -> (2,)  D, c (UNCHANGED)
    net_flux_e: CoeffNet   # (log1p(|e_final|^2),)         -> (1,)  s0 "own final ellipticity" correction (NEW)
    net_mc: CoeffNet       # (x0, log_scale_n, ehat2)      -> (1,)  Mc additive shift (NEW, 5-D extension)
    _cond_dim: int = eqx.field(static=True)
    _log_scale_mean: float = eqx.field(static=True)
    _e_mag_sq_scale: float = eqx.field(static=True)
    _size_loc: float = eqx.field(static=True)
    _g_s_max: float = eqx.field(static=True)
    _s0_e_max: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        nn_width=32,
        nn_depth=2,
        activation=jnn.silu,
        full_cond_dim=5,
        log_scale_mean=12.0,
        e_max=0.2,
        size_loc=0.0,
        g_s_max=1.0,
        s0_e_max=1.0,
    ):
        self._cond_dim = full_cond_dim
        self._log_scale_mean = float(log_scale_mean)
        self._e_mag_sq_scale = float(e_max**2)
        self._size_loc = float(size_loc)
        self._g_s_max = float(g_s_max)
        self._s0_e_max = float(s0_e_max)
        k0, k1, k2, k3, k4 = jr.split(key, 5)
        nets = [
            CoeffNet(k0, 3, 1, nn_width, nn_depth, activation),  # net_flux (x0, log_scale_n, ehat2)
            CoeffNet(k1, 3, 1, nn_width, nn_depth, activation),  # net_size
            CoeffNet(k2, 4, 2, nn_width, nn_depth, activation),  # net_dipquad
            CoeffNet(k3, 1, 1, nn_width, nn_depth, activation),  # net_flux_e
            CoeffNet(k4, 3, 1, nn_width, nn_depth, activation),  # net_mc (5-D extension)
        ]
        # Zero each net's final layer -> all coefficients start at 0, so the
        # layer is the identity at init (same convention as SigmaXCouplingLayer).
        nets = [_zero_last_layer(n) for n in nets]
        (self.net_flux, self.net_size, self.net_dipquad, self.net_flux_e,
         self.net_mc) = nets

    @property
    def shape(self):
        return (5,)

    @property
    def cond_shape(self):
        return (self._cond_dim,)

    def _unpack(self, condition):
        # `condition` is [g1, g2, C00, C01, C11] when chained with shear, or
        # bare [C00, C01, C11] when standalone (self._cond_dim is 5 or 3
        # respectively -- see `bulk.build_flow`) -- not [g1, g2, log_scale,
        # e1, e2] -- convert locally so every caller keeps passing the raw
        # covariance. The last three entries are always C00, C01, C11.
        off = self._cond_dim - 3
        CX = jnp.array([[condition[off], condition[off + 1]],
                         [condition[off + 1], condition[off + 2]]])
        log_scale, e1, e2 = cx_to_sx_cond(CX)
        log_scale_n = log_scale - self._log_scale_mean
        e_mag_sq = e1**2 + e2**2
        e_mag_sq_n = e_mag_sq / (self._e_mag_sq_scale + 1e-8)
        T_n = jnp.exp(log_scale_n)  # analytic C_X scaling factored out, as in SigmaXCouplingLayer
        return log_scale_n, e1, e2, e_mag_sq, e_mag_sq_n, T_n

    def _ellipticity(self, x0, x1, x3, x4, e1, e2, log_scale_n, e_mag_sq_n, T_n):
        """UNCHANGED from SigmaXCouplingLayer: reads only the RAW (x0,x1), so it
        is safe to compute before the flux transform in BOTH directions
        (forward: x0,x1 given; inverse: recovered first, see below).

        Operates on the ellipticity pair, which is ``(x3,x4)`` in THIS chart
        (``z3=M1/Mr``, ``z4=M2/Mr`` -- see the class docstring), not
        ``(x2,x3)``.
        """
        g_s = self._g_s_max * jnn.tanh(
            T_n * self.net_size(jnp.array([_bound_coeff_input(x0), log_scale_n, e_mag_sq_n]))[0]
        )
        kappa = jnp.exp(g_s)
        dq_in = jnp.array([_bound_coeff_input(x0), _bound_coeff_input(x1), log_scale_n, e_mag_sq_n])
        D_raw, c_raw = self.net_dipquad(dq_in)
        D = T_n * D_raw
        c = jnn.tanh(T_n**2 * c_raw)
        m3 = (1.0 + c * e1) * x3 + c * e2 * x4 + D * e1
        m4 = c * e2 * x3 + (1.0 - c * e1) * x4 + D * e2
        return m3 / kappa, m4 / kappa, kappa, D, c

    def _s0(self, x0, y3, y4, log_scale_n, e_mag_sq_n, T_n):
        """Flux shift: base (C_X + the galaxy's OWN flux `x0`, unlike
        SigmaXCouplingLayer -- see the class docstring's "5-D EXTENSION"
        note and `net_flux`'s FIX ONE) + a bounded correction reading the
        galaxy's OWN final ellipticity (y3,y4) -- always available (given
        directly on inverse, computed first on forward).

        Reading `x0` here makes ``y0 = x0 + s0(x0, ...)`` IMPLICIT in `x0`,
        so `inverse_and_log_det` can no longer recover `x0` by a plain
        subtraction -- see :meth:`_solve_x0`, which this class's own inverse
        now calls instead."""
        s0_base = T_n * self.net_flux(
            jnp.array([_bound_coeff_input(x0), log_scale_n, e_mag_sq_n]))[0]
        L = jnp.log1p(y3 * y3 + y4 * y4)
        s0_owne = self._s0_e_max * jnn.tanh(T_n * self.net_flux_e(jnp.array([L]))[0])
        return s0_base + s0_owne

    #: Bisection bracket for :meth:`_solve_x0`, in ABSOLUTE `x0` (z-space)
    #: units -- NOT an offset from `y0`.  `s0`'s magnitude is driven by
    #: `T_n = exp(log_scale_n)`, which is unbounded (a large-|C_X| condition
    #: or large de-zeroed weights can make it many orders of magnitude
    #: bigger than 1), so `s0(x0)` itself can be enormous even at a
    #: perfectly ordinary `x0` -- i.e. `|y0 - x0|` is NOT a bound on `x0`'s
    #: own location, and bracketing around `y0` (an earlier version of this
    #: constant did that) misses the root entirely whenever `s0` is large.
    #: `_bound_coeff_input` clips `x0` at +-8 before `net_flux` ever sees
    #: it, so `s0(x0)` is CONSTANT for `|x0| > 8` and `f(x0) = x0 + s0(x0)`
    #: has slope exactly 1 there -- i.e. `f` is monotone outside a bounded
    #: band regardless of how large `s0` gets, and a fixed bracket wide
    #: enough to contain any physically-reachable `x0` (the off-manifold
    #: stress inputs in `tests/test_sigmax_block_layer.py` reach |x0|=20)
    #: is guaranteed to bracket the root as long as `f` does not fold
    #: *inside* [-8, 8] -- the one thing this bisection does not itself
    #: certify, mirroring `Spin2CouplingLayer`'s identical caveat.
    _S0_BRACKET_LO = -100.0
    _S0_BRACKET_HI = 100.0
    #: Bisection halvings for :meth:`_solve_x0`.  48 (`Spin2CouplingLayer`'s
    #: own count) resolves a bracket half this wide to float64 eps; 60 is
    #: used here for margin since `s0` is not analytically bounded away from
    #: chatter the way `Spin2CouplingLayer`'s `log1p` guard is.
    _S0_BISECT_STEPS = 60

    def _solve_x0(self, y0, y3, y4, log_scale_n, e_mag_sq_n, T_n):
        """Invert ``y0 = x0 + s0(x0, y3, y4, ...)`` for `x0` by bisection.

        `s0` now reads the galaxy's own flux (FIX ONE, see the class
        docstring), so unlike every other coefficient here its forward map
        is not a plain shift of an independently-known quantity -- `x0` is
        both the argument and (via `y0`) the target.  `Spin2CouplingLayer`
        already carries an identical no-closed-form-inverse case in this
        file (its radial stretch `T(q) = q*exp(2h(q))`), solved the same way:
        forward stays a direct evaluation (this method is never called from
        `transform_and_log_det`/`_raw_transform`, only from
        `inverse_and_log_det`), and only `sample`'s inverse pays the
        bisection.  Monotone by construction outside `_bound_coeff_input`'s
        +-8 clip (see `_S0_BRACKET`'s comment); nothing here enforces it
        inside that band, mirroring `Spin2CouplingLayer`'s own
        `check_monotone`-flagged, not analytically excluded, fold risk.

        The bisection ITSELF is differentiated via `_solve_x0_ift`'s
        implicit-function-theorem `custom_jvp`, not by autodiff-through-the-
        loop: `jax.lax.scan`'s `jnp.where(below, ...)` branches on a boolean
        comparison, and JAX's `<` carries no gradient, so a plain `jacfwd`
        of the bisection above returns EXACTLY ZERO for `dx0/dy0` -- caught
        by `tests/test_psi_logdet.py` as a singular (`-inf` log-det) `Psi`
        Jacobian at `g=0`, where the true Jacobian is the identity.

        `custom_jvp` needs its differentiable argument to be a pytree of
        arrays, and `self` (an `eqx.Module`) also carries non-array leaves
        (each `CoeffNet`'s activation function) -- so it is split into
        `(params, static)` here and only `params` crosses the `custom_jvp`
        boundary, exactly as `eqx.filter_jit` does internally.
        """
        params, static = eqx.partition(self, eqx.is_inexact_array)
        return _solve_x0_ift(static, params, y0, y3, y4, log_scale_n,
                              e_mag_sq_n, T_n)

    def _solve_x0_bisect(self, y0, y3, y4, log_scale_n, e_mag_sq_n, T_n):
        """Numeric root only, no gradient guarantees -- see `_solve_x0_ift`."""
        def f(x0):
            return x0 + self._s0(x0, y3, y4, log_scale_n, e_mag_sq_n, T_n)

        def step(bounds, _):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            below = f(mid) < y0
            return (jnp.where(below, mid, lo), jnp.where(below, hi, mid)), None

        lo0 = self._S0_BRACKET_LO + 0.0 * y0
        hi0 = self._S0_BRACKET_HI + 0.0 * y0
        (lo, hi), _ = jax.lax.scan(step, (lo0, hi0), None,
                                   length=self._S0_BISECT_STEPS)
        return 0.5 * (lo + hi)

    def _mc_shift(self, x0, log_scale_n, e_mag_sq_n, T_n):
        """Mc/Mr's own additive shift (5-D extension, see the class
        docstring): flux + C_X only, deliberately NOT `x2`/`y2` (the slot it
        shifts) -- so the inverse below is a plain subtraction, not an
        implicit equation."""
        return T_n * self.net_mc(
            jnp.array([_bound_coeff_input(x0), log_scale_n, e_mag_sq_n]))[0]

    def _raw_transform(self, x, condition):
        log_scale_n, e1, e2, _, e_mag_sq_n, T_n = self._unpack(condition)
        x0, x1, x2, x3, x4 = x
        y3, y4, kappa, _, _ = self._ellipticity(x0, x1, x3, x4, e1, e2, log_scale_n, e_mag_sq_n, T_n)
        s0 = self._s0(x0, y3, y4, log_scale_n, e_mag_sq_n, T_n)
        c1 = self._size_loc
        y0 = x0 + s0
        y1 = kappa * x1 + c1 * (kappa - 1.0)
        y2 = x2 + self._mc_shift(x0, log_scale_n, e_mag_sq_n, T_n)
        return jnp.stack([y0, y1, y2, y3, y4])

    def transform_and_log_det(self, x, condition=None):
        y = self._raw_transform(x, condition)
        jac = jax.jacfwd(self._raw_transform, argnums=0)(x, condition)
        _, log_det = jnp.linalg.slogdet(jac)
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        # Fully closed-form (no iteration anywhere): y3,y4 (ellipticity) are
        # GIVEN, so s0 (and hence x0) is immediately computable; x1 follows
        # from x0; x3,x4 invert the SAME closed-form ellipticity block
        # SigmaXCouplingLayer uses; x2 (Mc/Mr) is a plain subtraction.
        log_scale_n, e1, e2, e_mag_sq, e_mag_sq_n, T_n = self._unpack(condition)
        y0, y1, y2, y3, y4 = y

        # `s0` now reads the galaxy's OWN flux `x0` (FIX ONE, see the class
        # docstring / `_s0`), so `x0` can no longer be recovered by a plain
        # subtraction -- `_solve_x0` bisects `y0 = x0 + s0(x0, ...)` instead.
        x0 = self._solve_x0(y0, y3, y4, log_scale_n, e_mag_sq_n, T_n)

        g_s = self._g_s_max * jnn.tanh(
            T_n * self.net_size(jnp.array([_bound_coeff_input(x0), log_scale_n, e_mag_sq_n]))[0]
        )
        kappa = jnp.exp(g_s)
        c1 = self._size_loc
        x1 = (y1 - c1 * (kappa - 1.0)) / kappa

        dq_in = jnp.array([_bound_coeff_input(x0), _bound_coeff_input(x1), log_scale_n, e_mag_sq_n])
        D_raw, c_raw = self.net_dipquad(dq_in)
        D = T_n * D_raw
        c = jnn.tanh(T_n**2 * c_raw)
        r3 = kappa * y3 - D * e1
        r4 = kappa * y4 - D * e2
        det_e = 1.0 - c**2 * e_mag_sq
        x3 = ((1.0 - c * e1) * r3 - c * e2 * r4) / det_e
        x4 = (-c * e2 * r3 + (1.0 + c * e1) * r4) / det_e
        # x0 is already recovered above, so this is a plain subtraction (no
        # circularity -- see `_mc_shift`'s docstring).
        x2 = y2 - self._mc_shift(x0, log_scale_n, e_mag_sq_n, T_n)

        x = jnp.stack([x0, x1, x2, x3, x4])
        jac = jax.jacfwd(self._raw_transform, argnums=0)(x, condition)
        _, log_det_fwd = jnp.linalg.slogdet(jac)
        return x, -log_det_fwd


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _solve_x0_ift(static, params, y0, y3, y4, log_scale_n, e_mag_sq_n, T_n):
    """`SigmaXBlockLayer._solve_x0`'s numeric root, `custom_jvp`-wrapped.

    `static` is the non-array half of an `eqx.partition` of the layer
    (`_solve_x0` does the split); `params` is the array half and the only
    argument this function differentiates. The primal is exactly
    `eqx.combine(params, static)._solve_x0_bisect(...)`; the point of this
    wrapper is the JVP rule below, which replaces "differentiate through the
    bisection's loop" (whose gradient is identically zero, see
    `_solve_x0`'s docstring) with the implicit-function-theorem derivative of
    `f(x0) = x0 + layer._s0(x0, ...) = y0`.
    """
    layer = eqx.combine(params, static)
    return layer._solve_x0_bisect(y0, y3, y4, log_scale_n, e_mag_sq_n, T_n)


@_solve_x0_ift.defjvp
def _solve_x0_ift_jvp(static, primals, tangents):
    params, y0, y3, y4, log_scale_n, e_mag_sq_n, T_n = primals
    params_dot, y0_dot, y3_dot, y4_dot, ls_dot, ems_dot, tn_dot = tangents
    layer = eqx.combine(params, static)
    x0 = layer._solve_x0_bisect(y0, y3, y4, log_scale_n, e_mag_sq_n, T_n)

    def f(params_, x0_, y3_, y4_, ls_, ems_, tn_):
        layer_ = eqx.combine(params_, static)
        return x0_ + layer_._s0(x0_, y3_, y4_, ls_, ems_, tn_)

    # f_x0 = d(x0 + s0)/dx0 at the root -- IFT's denominator.
    f_x0 = jax.grad(f, argnums=1)(params, x0, y3, y4, log_scale_n, e_mag_sq_n, T_n)
    # f's total derivative w.r.t. everything EXCEPT x0 (held fixed at the
    # root), i.e. the -f_theta numerator via a single jvp.
    _, f_dot_rest = jax.jvp(
        lambda p, a, b, c, d, e: f(p, x0, a, b, c, d, e),
        (params, y3, y4, log_scale_n, e_mag_sq_n, T_n),
        (params_dot, y3_dot, y4_dot, ls_dot, ems_dot, tn_dot))
    x0_dot = (y0_dot - f_dot_rest) / f_x0
    return x0, x0_dot


# ---------------------------------------------------------------------------
# EarlyChain
# ---------------------------------------------------------------------------


class EarlyChain(AbstractBijection):
    """Bijection chain where early layers are unconditional and the last is conditional.

    Applies a sequence of unconditional bijections followed by a single
    conditional bijection.  The log-absolute-determinant is the sum of
    all constituent determinants.

    Parameters
    ----------
    early : iterable of AbstractBijection
        Sequence of unconditional bijections applied first.
    last : AbstractBijection
        Final conditional bijection.  Its ``cond_shape`` determines the
        ``cond_shape`` of the whole chain.
    """

    early: tuple
    last: AbstractBijection

    def __init__(self, early, last):
        object.__setattr__(self, "early", tuple(early))
        object.__setattr__(self, "last", last)

    @property
    def shape(self):
        return self.last.shape

    @property
    def cond_shape(self):
        return self.last.cond_shape

    def transform_and_log_det(self, x, condition=None):
        lad_total = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        for b in self.early:
            x, lad = b.transform_and_log_det(x)
            lad_total = lad_total + lad
        x, lad = self.last.transform_and_log_det(x, condition)
        return x, lad_total + lad

    def inverse_and_log_det(self, y, condition=None):
        lad_total = jnp.zeros(y.shape[:-1], dtype=y.dtype)
        y, lad = self.last.inverse_and_log_det(y, condition)
        lad_total = lad_total + lad
        for b in reversed(self.early):
            y, lad = b.inverse_and_log_det(y)
            lad_total = lad_total + lad
        return y, lad_total


# ---------------------------------------------------------------------------
# Helper: add equivariant permute
# ---------------------------------------------------------------------------


def _add_equivariant_permute(
    bijection: AbstractBijection, key: jax.Array
) -> AbstractBijection:
    """Append a random spin-equivariant permutation after a bijection.

    The permutation randomly swaps components 0↔1 and/or 2↔3 so that
    the autoregressive order is shuffled between layers while preserving
    the spin-0 / spin-2 block structure.

    Parameters
    ----------
    bijection : AbstractBijection
        The bijection to wrap.
    key : jax.Array
        JAX PRNG key used to draw the random permutation.

    Returns
    -------
    AbstractBijection
        A merged :class:`~flowjax.bijections.Chain` of the input bijection
        followed by the random permutation.
    """
    k0, k1 = jr.split(key)
    perm = jnp.where(
        jr.randint(k0, (), 0, 2),
        jnp.array([1, 0, 2, 3]),
        jnp.array([0, 1, 2, 3]),
    )
    perm = jnp.where(
        jr.randint(k1, (), 0, 2),
        perm.at[2].set(perm[3]).at[3].set(perm[2]),
        perm,
    )
    return Chain([bijection, Permute(perm)]).merge_chains()


# ---------------------------------------------------------------------------
# new_masked_autoregressive_flow
# ---------------------------------------------------------------------------


def new_masked_autoregressive_flow(
    key: jax.Array,
    *,
    base_dist: AbstractDistribution,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    invert: bool = True,
    last_layer_cond_dim: int | None = None,
    quadratic_last: bool = True,
    cross_terms_last: bool = True,
    last_layer_nn_width: int | None = None,
    last_layer_nn_depth: int | None = None,
    # Σ_X coupling layer — set sigmax_cond_dim=5 to enable
    sigmax_cond_dim: int | None = None,
    sigmax_nn_width: int = 32,
    sigmax_nn_depth: int = 2,
    # No default: guessing this is exactly the mistake POINT_SOURCE already
    # made once. Compute it from the TARGET catalog's own Sigma_X via
    # `sigmax_log_scale_stats`, never from the (noiseless) training templates.
    sigmax_log_scale_mean: float | None = None,
    sigmax_log_scale_std: float | None = None,
) -> Transformed:
    """Construct a masked autoregressive normalizing flow for BFD galaxy moments.

    Builds ``flow_layers - 1`` unconditional
    :class:`EquivariantAutoregressiveLayer` steps (each followed by a random
    equivariant permutation) and a single conditional
    :class:`ExplicitPolyLast` final layer.  If ``sigmax_cond_dim=5``, a
    :class:`SigmaXCouplingLayer` is appended after ``ExplicitPolyLast`` so
    that the prior condition becomes ``[g1, g2, log_scale, e1, e2]``.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    base_dist : AbstractDistribution
        Base distribution (typically a standard multivariate Gaussian).
    flow_layers : int, optional
        Total number of coupling/autoregressive layers.  Default is 8.
    nn_width : int, optional
        Hidden width for the early-layer networks.  Default is 50.
    nn_depth : int, optional
        Number of hidden layers in the early networks.  Default is 1.
    nn_activation : callable, optional
        Activation function.  Default is ``jax.nn.relu``.
    invert : bool, optional
        If ``True`` (default), wrap the bijection in
        :class:`~flowjax.bijections.Invert` so that ``sample`` runs in the
        forward direction.
    last_layer_cond_dim : int or None, optional
        Conditioning dimension for :class:`ExplicitPolyLast`.
    quadratic_last : bool, optional
        Whether to include ``|g|²`` features in the last layer.  Default
        is ``True``.
    cross_terms_last : bool, optional
        Whether to include ``g1·g2`` cross terms.  Default is ``True``.
    last_layer_nn_width : int or None, optional
        Hidden width for the last-layer networks.  Falls back to
        ``nn_width`` if ``None``.
    last_layer_nn_depth : int or None, optional
        Number of hidden layers for the last-layer networks.  Falls back
        to ``nn_depth`` if ``None``.
    sigmax_cond_dim : int or None, optional
        If not ``None``, appends a :class:`SigmaXCouplingLayer` conditioned
        on a ``sigmax_cond_dim``-dimensional vector.  Set to 5 to enable
        full Σ_X conditioning.
    sigmax_nn_width : int, optional
        Hidden width for the Σ_X coupling network.  Default is 32.
    sigmax_nn_depth : int, optional
        Number of hidden layers for the Σ_X coupling network.  Default is 2.
    sigmax_log_scale_mean, sigmax_log_scale_std : float
        Required when ``sigmax_cond_dim`` is set.  From
        `sigmax_log_scale_stats` applied to the TARGET catalog's Sigma_X, not
        the (noiseless) training templates.

    Returns
    -------
    Transformed
        A flowjax ``Transformed`` distribution (base dist + bijection chain).
    """
    dim = base_dist.shape[-1]
    _last_width = last_layer_nn_width if last_layer_nn_width is not None else nn_width
    _last_depth = last_layer_nn_depth if last_layer_nn_depth is not None else nn_depth
    use_sigmax = sigmax_cond_dim is not None
    if use_sigmax and (sigmax_log_scale_mean is None or sigmax_log_scale_std is None):
        raise ValueError(
            "sigmax_cond_dim is set but sigmax_log_scale_mean/std were not "
            "given -- compute them with sigmax_log_scale_stats(target_sigma_x) "
            "from the TARGET catalog, not the training templates.")

    def _make_layer_uncond(layer_key, layer_idx):
        bij_key, _ = jr.split(layer_key)
        bijection = EquivariantAutoregressiveLayer(
            bij_key, nn_width, nn_depth, nn_activation
        )
        perm = jnp.where(
            layer_idx % 2 == 1, jnp.array([1, 0, 2, 3]), jnp.array([0, 1, 2, 3])
        )
        return Chain([bijection, Permute(perm)]).merge_chains()

    # one extra key when sigmax layer is enabled
    n_keys = flow_layers + (1 if use_sigmax else 0)
    keys = jr.split(key, n_keys)
    early_layers = [
        _make_layer_uncond(k, i)
        for i, k in enumerate(keys[: -1 if not use_sigmax else -2])
    ]

    k_last = keys[-2] if use_sigmax else keys[-1]
    k_perm, k_poly = jr.split(k_last)

    explicit = ExplicitPolyLast(
        k_poly,
        dim,
        last_layer_cond_dim,
        quadratic_last,
        cross_terms_last,
        _last_width,
        _last_depth,
        nn_activation,
    )
    explicit_with_perm = (
        Chain([_add_equivariant_permute(explicit, k_perm)]).merge_chains()
        if dim > 1
        else explicit
    )

    if use_sigmax:
        sigmax = SigmaXCouplingLayer(
            keys[-1],
            nn_width=sigmax_nn_width,
            nn_depth=sigmax_nn_depth,
            activation=nn_activation,
            full_cond_dim=sigmax_cond_dim,
            log_scale_mean=sigmax_log_scale_mean,
            log_scale_std=sigmax_log_scale_std,
        )
        last = Chain([explicit_with_perm, sigmax]).merge_chains()
    else:
        last = explicit_with_perm

    bijection = EarlyChain(early_layers, last)

    if invert:
        bijection = Invert(bijection)
    return Transformed(base_dist, bijection)
