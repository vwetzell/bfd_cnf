"""
models/bijections.py
====================
Custom flowjax bijection classes used in the bfd_cnf normalizing-flow model:

  - RawMomentStandardize   — transforms raw moments to standardized coordinates
  - BoundedAffine          — affine bijection with bounded scale parameter
  - Spin0AutoregressiveLayer / Spin2CouplingLayer / EquivariantAutoregressiveLayer
  - ShearTaylorLast / SigmaXCouplingLayer / EarlyChain
  - new_masked_autoregressive_flow

Helper functions: _bounded_log_scale, _bounded_scale, _inv_bounded_scale,
_raw2std_jacobian_single_jax, propagate_cov_to_std_jax.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flowjax.bijections import AbstractBijection, Chain, Invert, Permute
from flowjax.distributions import AbstractDistribution, Transformed
from flowjax.utils import arraylike_to_array
from jaxtyping import Array, ArrayLike
from paramax import AbstractUnwrappable, Parameterize

# ---------------------------------------------------------------------------
# Bounded-scale helpers
# ---------------------------------------------------------------------------


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


def _bounded_loc(u: jax.Array, max_loc: float = 8.0) -> jax.Array:
    """Map an unconstrained parameter to a location bounded to (-max_loc, max_loc).

    2026-08-05: Spin0AutoregressiveLayer's net_ls1 emits BOTH a log-scale (already
    hard-bounded via _bounded_log_scale) AND a location/shift -- but the shift was
    used RAW, unbounded. A pure translation has Jacobian 1 (zero log-det cost), so
    nothing in the loss penalises an extreme shift; a sufficiently precisely-
    optimized run (full LR decay instead of the usual floor-and-hold) found and
    exploited this, driving the size channel to +-20 in standardized units for a
    thin slice of the base distribution -- a classic normalizing-flow "unpenalized
    location term chases an outlier" pathology, unrelated to the (separately
    verified clean) ShearTaylorLast log-det work earlier in this investigation.
    tanh gives a smooth, everywhere-differentiable saturation (near-linear/
    identity for |u| << max_loc, so normal in-distribution behaviour is
    unaffected) instead of a hard clip -- same design as _bounded_log_scale's
    sigmoid, and reuses this file's existing 8-sigma "definitely off-manifold"
    convention (_SHEAR_COEFF_INPUT_CLIP).
    """
    return max_loc * jnp.tanh(u / max_loc)


# ---------------------------------------------------------------------------
# Jacobian helpers for raw → standardized coordinate change
# ---------------------------------------------------------------------------


def _raw2std_jacobian_single_jax(
    raw_moments: jax.Array, std_scales: jax.Array
) -> jax.Array:
    """Compute the Jacobian matrix of the raw-to-standardised coordinate transform.

    The transform is
    ``m = [(log10 Mf - mean[0])/s[0],  (Mr/Mf - mean[1])/s[1],
           (M1/Mr - mean[2])/s[2],      (M2/Mr - mean[3])/s[3]]``
    and this function returns ``J = diag(1/s) @ dt/dx``.

    Parameters
    ----------
    raw_moments : jax.Array, shape (4,)
        Raw moments ``[Mf, Mr, M1, M2]``.
    std_scales : jax.Array, shape (4,)
        Standardisation scales (``raw2standard.std``).

    Returns
    -------
    jax.Array, shape (4, 4)
        Jacobian ``dm/dx`` incorporating the standardisation scaling.
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


class RawMomentStandardize(AbstractBijection):
    """
    Raw x = [Mf, Mr, M1, M2]
    Transformed t = [log10(Mf), Mr/Mf, M1/Mr, M2/Mr]
    Then standardized: m = (t - mean) / std
    """

    mean: jax.Array
    std: jax.Array

    def __init__(self, mean=None, std=None):
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
            self.mean = jnp.zeros(4)
        else:
            self.mean = jnp.asarray(mean)
        if std is None:
            self.std = jnp.ones(4)
        else:
            self.std = jnp.asarray(std)

    @property
    def shape(self):  # type: ignore[override]
        """Shape of the bijection input/output: ``(4,)``."""
        return (4,)

    @property
    def cond_shape(self):  # type: ignore[override]
        """Conditioning shape: ``None`` (unconditional)."""
        return None

    def _forward_transform(self, x):
        """Apply the raw-to-standardised coordinate change.

        Parameters
        ----------
        x : jax.Array, shape (..., 4)
            Raw moments ``[Mf, Mr, M1, M2]``.

        Returns
        -------
        m : jax.Array, shape (..., 4)
            Standardised coordinates.
        t : jax.Array, shape (..., 4)
            Intermediate un-standardised transformed coordinates.
        """
        Mf, Mr, M1, M2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        t0 = jnp.log10(Mf)
        t1 = Mr / Mf
        t2 = M1 / Mr
        t3 = M2 / Mr

        t = jnp.stack([t0, t1, t2, t3], axis=-1)
        m = (t - self.mean) / self.std
        return m, t

    def _inverse_transform(self, m):
        """Invert the standardised-to-raw coordinate change.

        Parameters
        ----------
        m : jax.Array, shape (..., 4)
            Standardised coordinates.

        Returns
        -------
        x : jax.Array, shape (..., 4)
            Raw moments ``[Mf, Mr, M1, M2]``.
        t : jax.Array, shape (..., 4)
            Intermediate un-standardised coordinates.
        """
        t = m * self.std + self.mean
        log10_const = jnp.log(jnp.array(10.0, dtype=t.dtype))
        Mf = jnp.exp(t[..., 0] * log10_const)
        Mr = t[..., 1] * Mf
        M1 = t[..., 2] * Mr
        M2 = t[..., 3] * Mr
        x = jnp.stack([Mf, Mr, M1, M2], axis=-1).astype(t.dtype)
        return x, t

    def transform_and_log_det(self, x, condition=None):
        """Transform raw moments to standardised coordinates and compute the log-abs-det.

        Parameters
        ----------
        x : jax.Array, shape (..., 4)
            Raw moments.
        condition : ignored

        Returns
        -------
        m : jax.Array, shape (..., 4)
            Standardised output.
        log_abs_det : jax.Array, scalar
            Log absolute determinant of the Jacobian.
        """
        m, _ = self._forward_transform(x)
        Mf = x[..., 0]
        Mr = x[..., 1]
        ln10 = jnp.log(jnp.array(10.0, dtype=Mf.dtype))
        lad_geom = -(2.0 * jnp.log(Mf) + 2.0 * jnp.log(Mr) + jnp.log(ln10))
        lad_std = -jnp.sum(jnp.log(self.std))
        return m, lad_geom + lad_std

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
# Standardiser <-> flow provenance (sidecar)
# ---------------------------------------------------------------------------
# A flow .eqx is only valid with the exact (mean, std) it was trained with.
# Coupling them by adjacency — <flow>.eqx.stats.npz next to the weights —
# means you cannot grab the wrong stats, and a later FITS change cannot poison
# them (they are frozen at train time). See data/raw2standard_stats*.npz for
# the legacy uncoupled files this replaces.


def stats_sidecar_path(flow_path: str) -> str:
    """Path of the standardiser stats paired with ``flow_path`` by adjacency."""
    return flow_path + ".stats.npz"


def save_stats(flow_path: str, raw2standard: "RawMomentStandardize") -> str:
    """Write the flow's standardiser ``(mean, std)`` to its sidecar."""
    p = stats_sidecar_path(flow_path)
    np.savez(p, mean=np.asarray(raw2standard.mean), std=np.asarray(raw2standard.std))
    return p


def load_stats(
    flow_path: str, override: str | None = None, rtol: float = 1e-5
) -> "RawMomentStandardize":
    """Load the standardiser paired with ``flow_path``.

    Prefers the sidecar ``<flow_path>.stats.npz`` written at train time.  If
    ``override`` (an explicit ``--stats`` npz) is given it must agree with the
    sidecar when one exists; a disagreement raises rather than silently warping
    contours.  Raises if neither a sidecar nor an override is available.
    """
    side = stats_sidecar_path(flow_path)
    have_side = os.path.exists(side)
    if override is not None:
        o = np.load(override)
        if have_side:
            s = np.load(side)
            if not (
                np.allclose(o["mean"], s["mean"], rtol=rtol)
                and np.allclose(o["std"], s["std"], rtol=rtol)
            ):
                raise ValueError(
                    f"--stats {override} disagrees with the flow's sidecar "
                    f"{side}; they encode different standardisers, so the flow "
                    "would be evaluated in the wrong coordinates."
                )
        return RawMomentStandardize(
            mean=jnp.asarray(o["mean"]), std=jnp.asarray(o["std"])
        )
    if not have_side:
        raise FileNotFoundError(
            f"No standardiser sidecar {side} and no --stats override. "
            "Backfill it with `python -m bfd_cnf.backfill_stats`."
        )
    s = np.load(side)
    return RawMomentStandardize(mean=jnp.asarray(s["mean"]), std=jnp.asarray(s["std"]))


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
        for layer in self.layers:
            x = layer(x)
        return x


def _norm_rescale(v: jax.Array, budget: float) -> jax.Array:
    """Rescale ``v`` so its L2 norm is <= ``budget`` (same idea as
    ``_spectral_norm_rescale``, for a vector instead of a matrix).

    Uses ``sqrt(sum(v^2)+eps)`` rather than ``jnp.linalg.norm(v)``: plain
    ``norm``'s gradient is ``v/||v||``, an exact ``0/0`` at ``v=0`` -- which
    is precisely ``ShearTaylorLast``'s zero-initialised last layer's bias.
    Confirmed this was the cause of a NaN Sobolev gradient (the weight's
    spectral-norm rescale doesn't have this problem: its whole power
    iteration is stop-gradiented, so autodiff never differentiates through
    a norm there at all -- see ``_spectral_norm_rescale``). Adding eps
    INSIDE the sqrt keeps the gradient finite (-> 0) at v=0 instead of
    merely avoiding a div-by-zero in the already-differentiated result.
    """
    n = jnp.sqrt(jnp.sum(v * v) + 1e-12)
    return v * jnp.minimum(1.0, budget / n)


def _spectral_norm_rescale(w: jax.Array, sigma_max: float, n_iters: int = 20) -> jax.Array:
    """Rescale ``w`` (out,in) so its spectral norm (largest singular value)
    is <= ``sigma_max``, via power iteration (Miyato et al. 2018).

    An earlier version used an exact SVD (cheap enough for these tiny
    matrices) but caused NaN gradients through the Sobolev path: SVD's
    gradient has a ``1/(s_i^2-s_j^2)``-type term that blows up whenever two
    singular values coincide -- which they always do at ``ShearTaylorLast``'s
    zero-initialised last layer (the weight IS the zero matrix there, a
    fully degenerate spectrum). Power iteration sidesteps this exactly the
    way it's meant to: the iterated singular vectors are stop-gradiented, so
    autodiff only ever sees the final, smooth bilinear form ``v^T w u`` as a
    function of ``w`` -- never SVD's eigendecomposition, so no degenerate-
    eigenvalue singularity to differentiate through. ``n_iters=20`` easily
    converges for matrices this small (a handful of hidden units); recomputed
    fresh every call (no persistent buffer) since that's negligible here.
    """
    out_dim, in_dim = w.shape
    u0 = jnp.ones((in_dim,)) / jnp.sqrt(in_dim)

    def body(u, _):
        v = w @ u
        v = v / (jnp.linalg.norm(v) + 1e-12)
        u_new = w.T @ v
        u_new = u_new / (jnp.linalg.norm(u_new) + 1e-12)
        return u_new, None

    u_final, _ = jax.lax.scan(body, u0, xs=None, length=n_iters)
    u_final = jax.lax.stop_gradient(u_final)
    v_raw = w @ u_final
    v_final = jax.lax.stop_gradient(v_raw / (jnp.linalg.norm(v_raw) + 1e-12))
    sigma = v_final @ (w @ u_final)
    return w * jnp.minimum(1.0, sigma_max / (sigma + 1e-12))


class LipschitzCoeffNet(eqx.Module):
    """``CoeffNet`` with a hard, provable global Lipschitz-constant bound.

    2026-08-06: replaces a ``jnp.tanh``-based output clamp (see the removed
    note in ``ShearTaylorLast._coeffs``/``_SHEAR_COEFF_OUTPUT_CLIP`` history)
    that stopped ``ShearTaylorLast.transform_and_log_det``'s ``log|det(I+J)|``
    from diverging to infinity, but only by capping the SYMPTOM (coefficient
    magnitude) -- training still walked every coefficient to the cap, because
    nothing addressed the actual mechanism: an unconstrained coefficient net
    gives ``J = d(shift)/dx`` (hence ``det(I+J)``, hence ``log_det``) no
    ceiling at all, and NLL has a real, measured, one-sided incentive (via
    that log-det term) to inflate it for the real ellipticity tail.

    This instead bounds the FUNCTION CLASS: every linear layer's weight is
    rescaled (via its exact spectral norm, ``_spectral_norm_rescale``) to a
    fixed per-layer operator-norm budget, chosen so the WHOLE composed net is
    ``<= target_lipschitz``-Lipschitz in its input, for ANY parameter values
    reachable by training -- standard technique (spectral normalisation,
    Miyato et al. 2018; used identically by i-ResNet, Behrmann et al. 2019,
    to guarantee invertibility and a provably bounded log-det for a residual
    map). Combined with the shear stencil's own small ``|g|<=0.05``, a small
    Lipschitz constant here directly bounds ``J``'s operator norm (see
    ``ShearTaylorLast._coeffs`` for the derivation), which bounds
    ``det(I+J)`` between ``(1-L)^d`` and ``(1+L)^d`` -- a mathematical
    consequence of the construction, not a numeric value someone chose to
    clamp outputs to.

    Bounding only the weights' spectral norm is NOT sufficient by itself --
    caught by an adversarial stress test (weights AND biases set to ~1e6):
    spectral normalisation bounds the net's SLOPE (``d(coeffs)/dx``, which is
    what the INDIRECT, chain-rule contribution to ``J`` scales with), but
    says nothing about the net's VALUE at any fixed point, which the biases
    control directly and additively, with zero effect on the derivative.
    ``_shift``'s equivariant tensors are not all G^2-suppressed either -- the
    spin-2 ``Tge2`` term's own x-derivative grows linearly with ``|e|``, so a
    coefficient with an unconstrained VALUE (regardless of how flat its
    slope is) can still blow up that DIRECT contribution to ``J``. Each
    layer's bias is therefore ALSO norm-rescaled (``_norm_rescale``) to the
    same per-layer budget as its weight's spectral norm -- giving both a
    bounded rate of change (Lipschitz) and a bounded value at any reference
    point (hence over the whole, already-bounded, input domain), by the same
    kind of construction.

    The activation is pinned to ``relu`` (exactly 1-Lipschitz) regardless of
    what the rest of the flow uses elsewhere -- e.g. SiLU's derivative peaks
    just above 1, which would silently violate the per-layer budget this
    class exists to guarantee.
    """

    layers: tuple
    sigma_per_layer: float = eqx.field(static=True)

    def __init__(self, key, in_dim, out_dim, width, depth, target_lipschitz):
        ks = jr.split(key, depth + 1)
        h = max(in_dim, 1)
        layers = []
        for i in range(depth):
            layers.append(eqx.nn.Linear(h, width, key=ks[i]))
            h = width
        layers.append(eqx.nn.Linear(h, out_dim, key=ks[-1]))
        self.layers = tuple(layers)
        n_linear = depth + 1
        self.sigma_per_layer = float(target_lipschitz) ** (1.0 / n_linear)

    def __call__(self, x):
        if x.shape[-1] == 0:
            x = jnp.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
        for i, layer in enumerate(self.layers):
            w = _spectral_norm_rescale(layer.weight, self.sigma_per_layer)
            b = _norm_rescale(layer.bias, self.sigma_per_layer)
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = jnn.relu(x)
        return x


# ---------------------------------------------------------------------------
# Spin0AutoregressiveLayer
# ---------------------------------------------------------------------------


class Spin0AutoregressiveLayer(AbstractBijection):
    """Autoregressive bijection on the spin-0 (flux/size) components.

    Transforms components 0 and 1 of the input vector with an affine
    autoregressive map:

    * ``y[0] = x[0] * exp(-s0)`` — unconditional scale.
    * ``y[1] = (x[1] - loc1(y[0])) * exp(-s1(y[0]))`` — conditioned on y[0].

    Leaves components 2 and 3 unchanged.

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

    log_scale0: jax.Array
    net_ls1: CoeffNet

    def __init__(self, key, nn_width, nn_depth, activation):
        (k1,) = jr.split(key, 1)
        self.log_scale0 = jnp.zeros(())
        self.net_ls1 = CoeffNet(k1, 1, 2, nn_width, nn_depth, activation)

    @property
    def shape(self):
        return (4,)

    @property
    def cond_shape(self):
        return None

    def _spin0_fwd(self, x):
        ls0 = _bounded_log_scale(self.log_scale0)
        y0_t = x[0] * jnp.exp(-ls0)
        out1 = self.net_ls1(jnp.array([y0_t]))
        loc1 = _bounded_loc(out1[0])
        ls1 = _bounded_log_scale(out1[1])
        y1_t = (x[1] - loc1) * jnp.exp(-ls1)
        return y0_t, y1_t, loc1, ls0, ls1

    def transform_and_log_det(self, x, condition=None):
        y0_t, y1_t, _, ls0, ls1 = self._spin0_fwd(x)
        y = x.at[0].set(y0_t).at[1].set(y1_t)
        return y, -(ls0 + ls1)

    def inverse_and_log_det(self, y, condition=None):
        ls0 = _bounded_log_scale(self.log_scale0)
        x0 = y[0] * jnp.exp(ls0)
        y0_t = y[0]
        out1 = self.net_ls1(jnp.array([y0_t]))
        loc1 = _bounded_loc(out1[0])
        ls1 = _bounded_log_scale(out1[1])
        x1 = y[1] * jnp.exp(ls1) + loc1
        x = y.at[0].set(x0).at[1].set(x1)
        return x, ls0 + ls1


# ---------------------------------------------------------------------------
# Spin2CouplingLayer
# ---------------------------------------------------------------------------


class Spin2CouplingLayer(AbstractBijection):
    """Coupling bijection on the spin-2 (shape) components conditioned on spin-0.

    Transforms components 2 and 3 with a single shared bounded log-scale
    ``s2`` conditioned on the already-transformed spin-0 components (0, 1).
    Both components are scaled by ``exp(-s2)``, which preserves the spin-2
    equivariance of the distribution.

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

    def __init__(self, key, nn_width, nn_depth, activation):
        self.net = CoeffNet(key, 2, 1, nn_width, nn_depth, activation)

    @property
    def shape(self):
        return (4,)

    @property
    def cond_shape(self):
        return None

    def _log_scale_each(self, y0_t, y1_t):
        out = self.net(jnp.array([y0_t, y1_t]))
        s2 = _bounded_log_scale(out[0])
        return -s2  # log alpha_2 = -s2, the per-component log scale

    def transform_and_log_det(self, x, condition=None):
        log_det_each = self._log_scale_each(x[0], x[1])
        y2 = x[2] * jnp.exp(log_det_each)
        y3 = x[3] * jnp.exp(log_det_each)
        y = x.at[2].set(y2).at[3].set(y3)
        return y, log_det_each + log_det_each

    def inverse_and_log_det(self, y, condition=None):
        log_det_each = self._log_scale_each(y[0], y[1])
        x2 = y[2] * jnp.exp(-log_det_each)
        x3 = y[3] * jnp.exp(-log_det_each)
        x = y.at[2].set(x2).at[3].set(x3)
        return x, -(log_det_each + log_det_each)


# ---------------------------------------------------------------------------
# EquivariantAutoregressiveLayer
# ---------------------------------------------------------------------------


class EquivariantAutoregressiveLayer(AbstractBijection):
    """Composed bijection combining :class:`Spin0AutoregressiveLayer` and :class:`Spin2CouplingLayer`.

    The spin-0 layer transforms components 0–1 autoregressively and the
    spin-2 layer then transforms components 2–3 conditioned on the spin-0
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

    def __init__(self, key, nn_width, nn_depth, activation):
        k0, k2 = jr.split(key)
        self.spin0 = Spin0AutoregressiveLayer(k0, nn_width, nn_depth, activation)
        self.spin2 = Spin2CouplingLayer(k2, nn_width, nn_depth, activation)

    @property
    def shape(self):
        return (4,)

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
    output ``(y2,y3)`` then feeds, as the bounded invariant
    ``L=log1p(y2^2+y3^2)`` (same saturating form ShearTaylorLast uses -- its
    gradient w.r.t. ``y2,y3`` VANISHES as ``|y|`` grows, so this input can't
    blow up the flux Jacobian row even for outlier ellipticities), a
    correction to the flux shift ``s0``. Safe with NO circularity in either
    direction:

    * forward: ``(y2,y3)`` come from ``x0,x1`` (given) alone, computable
      before ``y0`` needs them.
    * inverse: ``(y2,y3)`` are literally part of the given ``y`` -- no
      recovery needed at all -- so the correction is immediate; ``x0`` comes
      out via one subtraction, then ``x1``, then finally ``(x2,x3)`` via the
      SAME closed-form ellipticity inverse ``SigmaXCouplingLayer`` uses.

    ``g_s`` (size log-scale) is deliberately left UNCHANGED (depends on ``x0``
    only): letting it read the final ellipticity would need
    ``kappa=exp(g_s)`` to determine that same ellipticity's value
    (``y2,y3 = .../kappa``) -- a genuine circular dependency, unlike flux's
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
    """

    net_flux: CoeffNet     # (log_scale_n, ehat2)          -> (1,)  s0 base (SAME as SigmaXCouplingLayer)
    net_size: CoeffNet     # (x0, log_scale_n, ehat2)      -> (1,)  g_s (UNCHANGED, no owne correction)
    net_dipquad: CoeffNet  # (x0, x1, log_scale_n, ehat2)  -> (2,)  D, c (UNCHANGED)
    net_flux_e: CoeffNet   # (log1p(|e_final|^2),)         -> (1,)  s0 "own final ellipticity" correction (NEW)
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
        k0, k1, k2, k3 = jr.split(key, 4)
        nets = [
            CoeffNet(k0, 2, 1, nn_width, nn_depth, activation),  # net_flux
            CoeffNet(k1, 3, 1, nn_width, nn_depth, activation),  # net_size
            CoeffNet(k2, 4, 2, nn_width, nn_depth, activation),  # net_dipquad
            CoeffNet(k3, 1, 1, nn_width, nn_depth, activation),  # net_flux_e
        ]
        # Zero each net's final layer -> all coefficients start at 0, so the
        # layer is the identity at init (same convention as SigmaXCouplingLayer).
        nets = [_zero_last_layer(n) for n in nets]
        self.net_flux, self.net_size, self.net_dipquad, self.net_flux_e = nets

    @property
    def shape(self):
        return (4,)

    @property
    def cond_shape(self):
        return (self._cond_dim,)

    def _unpack(self, condition):
        log_scale_n = condition[2] - self._log_scale_mean
        e1, e2 = condition[3], condition[4]
        e_mag_sq = e1**2 + e2**2
        e_mag_sq_n = e_mag_sq / (self._e_mag_sq_scale + 1e-8)
        T_n = jnp.exp(log_scale_n)  # analytic C_X scaling factored out, as in SigmaXCouplingLayer
        return log_scale_n, e1, e2, e_mag_sq, e_mag_sq_n, T_n

    def _ellipticity(self, x0, x1, x2, x3, e1, e2, log_scale_n, e_mag_sq_n, T_n):
        """UNCHANGED from SigmaXCouplingLayer: reads only the RAW (x0,x1), so it
        is safe to compute before the flux transform in BOTH directions
        (forward: x0,x1 given; inverse: recovered first, see below)."""
        g_s = self._g_s_max * jnn.tanh(
            T_n * self.net_size(jnp.array([_bound_coeff_input(x0), log_scale_n, e_mag_sq_n]))[0]
        )
        kappa = jnp.exp(g_s)
        dq_in = jnp.array([_bound_coeff_input(x0), _bound_coeff_input(x1), log_scale_n, e_mag_sq_n])
        D_raw, c_raw = self.net_dipquad(dq_in)
        D = T_n * D_raw
        c = jnn.tanh(T_n**2 * c_raw)
        m2 = (1.0 + c * e1) * x2 + c * e2 * x3 + D * e1
        m3 = c * e2 * x2 + (1.0 - c * e1) * x3 + D * e2
        return m2 / kappa, m3 / kappa, kappa, D, c

    def _s0(self, y2, y3, log_scale_n, e_mag_sq_n, T_n):
        """Flux shift: base (C_X only, as in SigmaXCouplingLayer) + a bounded
        correction reading the galaxy's OWN final ellipticity (y2,y3) --
        always available (given directly on inverse, computed first on
        forward)."""
        s0_base = T_n * self.net_flux(jnp.array([log_scale_n, e_mag_sq_n]))[0]
        L = jnp.log1p(y2 * y2 + y3 * y3)
        s0_owne = self._s0_e_max * jnn.tanh(T_n * self.net_flux_e(jnp.array([L]))[0])
        return s0_base + s0_owne

    def _raw_transform(self, x, condition):
        log_scale_n, e1, e2, _, e_mag_sq_n, T_n = self._unpack(condition)
        x0, x1, x2, x3 = x
        y2, y3, kappa, _, _ = self._ellipticity(x0, x1, x2, x3, e1, e2, log_scale_n, e_mag_sq_n, T_n)
        s0 = self._s0(y2, y3, log_scale_n, e_mag_sq_n, T_n)
        c1 = self._size_loc
        y0 = x0 + s0
        y1 = kappa * x1 + c1 * (kappa - 1.0)
        return jnp.stack([y0, y1, y2, y3])

    def transform_and_log_det(self, x, condition=None):
        y = self._raw_transform(x, condition)
        jac = jax.jacfwd(self._raw_transform, argnums=0)(x, condition)
        _, log_det = jnp.linalg.slogdet(jac)
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        # Fully closed-form (no iteration anywhere): y2,y3 are GIVEN, so s0
        # (and hence x0) is immediately computable; x1 follows from x0; x2,x3
        # invert the SAME closed-form ellipticity block SigmaXCouplingLayer uses.
        log_scale_n, e1, e2, e_mag_sq, e_mag_sq_n, T_n = self._unpack(condition)
        y0, y1, y2, y3 = y

        s0 = self._s0(y2, y3, log_scale_n, e_mag_sq_n, T_n)
        x0 = y0 - s0

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
        r2 = kappa * y2 - D * e1
        r3 = kappa * y3 - D * e2
        det_e = 1.0 - c**2 * e_mag_sq
        x2 = ((1.0 - c * e1) * r2 - c * e2 * r3) / det_e
        x3 = (-c * e2 * r2 + (1.0 + c * e1) * r3) / det_e

        x = jnp.stack([x0, x1, x2, x3])
        jac = jax.jacfwd(self._raw_transform, argnums=0)(x, condition)
        _, log_det_fwd = jnp.linalg.slogdet(jac)
        return x, -log_det_fwd


# ---------------------------------------------------------------------------
# ShearTaylorLast
# ---------------------------------------------------------------------------

# ShearTaylorLast's coefficient nets (CoeffNet: plain unbounded ReLU MLPs) are
# queried on (flux, size) moment coordinates that are normally O(1) standardised
# values, but can occasionally be an off-manifold many-sigma outlier (e.g. an
# untrained q's garbage draw during ELBO fine-tuning). Queried there, an unbounded
# MLP extrapolates to huge outputs, which (even with _spin2_transform's EXACT
# log-det, not the old O(g^2)-truncated polynomial this once fed) still means
# querying the coeff nets at a wildly extrapolated point nobody trained them for
# — a real, physically meaningless value, not just a numerically unstable one.
# Flux/size get this treatment via a hard clip. A hard clip is exact identity
# for any in-distribution template and only ever engages off-manifold, so it
# is a no-op for a trained model's normal behaviour.
_SHEAR_COEFF_INPUT_CLIP = 8.0


def _bound_coeff_input(x):
    return jnp.clip(x, -_SHEAR_COEFF_INPUT_CLIP, _SHEAR_COEFF_INPUT_CLIP)


# 2026-08-06: a hard clip + a trained synthetic-probe penalty on the
# ellipticity input (log1p(e1^2+e2^2)) were both tried here, after measuring
# that ShearTaylorLast's coefficient net blows up 2-3 orders of magnitude for
# realistic (not rare-outlier) standardised ellipticity -- std_e is much
# smaller than std_x0/std_x1, so physical |e_raw|<=1 standardises to |e_std|
# routinely 3-9 for REAL, common templates. Neither held up: clipping/probing
# one region just relocated the same-magnitude blowup to wherever was left
# unclipped/unprobed (e.g. e_std~1-6, squarely inside real data), rather than
# fixing the underlying instability. Removed rather than iterated on further
# per explicit request -- no clipping or cleaning on e1, e2 here; they reach
# the coefficient net exactly as computed.
#
# The actual mechanism (found by checking checkpoints across a training run
# where the loss suddenly dropped -- see ShearTaylorLast._coeffs for the
# full account) is that transform_and_log_det's log|det(I+J)| has no upper
# bound: J scales with these coefficients, and an unbounded coefficient net
# gives NLL a literal unbounded-below direction (inflate the Jacobian
# determinant at real data instead of fitting the physical shear response).
#
# A first attempt bounded the coefficients THEMSELVES via a tanh saturation
# in _coeffs. That stopped the divergence but was a clamp on the symptom:
# a follow-up measurement (scaling the trained net's raw output from 0 to 1
# on a real batch) showed NLL has a genuine, if small on average, one-sided
# preference for inflated coefficients -- concentrated in the real
# high-ellipticity tail (up to +0.9 nats for a single template) -- so
# training walked every coefficient to whatever cap existed regardless of
# where the cap was set. Replaced by LipschitzCoeffNet: a hard, provable
# Lipschitz-constant bound on the coefficient net ITSELF (spectral
# normalisation of every linear layer), which bounds J's operator norm --
# and therefore det(I+J) and log_det -- as a mathematical consequence of
# the function class, not a value someone chose to clamp outputs to.
_SHEAR_COEFF_NET_LIPSCHITZ = 1.0
#
# Outer ring radius of the training g-stencil (bfd_cnf/models/flows.py:
# make_elbo_loss/make_nll_loss's `_radii` array -- MUST be
# kept equal to max(_radii) in BOTH of those; this constant does not
# derive from them automatically, so a stencil change that doesn't also
# update this one silently desyncs the two -- see git history 2026-08-05.
# All 2nd-order (B) coefficients are reparametrised to "shift at the outer
# ring" units (comparable scale to the 1st-order (A) coefficients, which
# operate at O(|g|^1)) and converted to raw units only at the point of use:
# Adam normalises each parameter's step by its OWN gradient RMS, so a raw-unit
# B would get inflated ~1/OUTER_G^2 more than A to reach a comparable step in
# the ACTUAL predicted shift, amplifying whatever noise sits in that already-
# small gradient by the same factor (see 2026-08-05 investigation).
_SHEAR_STENCIL_OUTER_G = 0.05

# Picard-iteration count for inverse_and_log_det's fixed-point substitution
# inverse (x = y - shift(x;g), iterated from x=y). Empirically geometric
# convergence (~3x/pass) even for adversarially large coefficients -- this is
# a cheap accuracy safety margin, not load-bearing for the realistic
# (near-zero-init) training regime. See test_bijection_roundtrip_and_identity_at_g0.
_INVERSE_PICARD_ITERS = 4


class ShearTaylorLast(AbstractBijection):
    """Structural shear layer: an additive 2nd-order Taylor-in-g moment displacement.

    ``y = x + A(x)·g + 0.5 g^T B(x) g``, ``A`` (4,2) = ``dy/dg``, ``B`` (4,2,2) =
    ``d^2y/dg^2``, both evaluated at ``g=0`` (i.e. at the input ``x`` itself --
    the textbook definition of a 2nd-order Taylor expansion). Zero-initialised
    coefficient nets ⇒ identity at init, and identity at ``g=0`` always (so the
    unconditional bulk layers before this one set the ``g=0`` prior shape, and
    this layer only imprints the shear displacement on top).

    Equivariance: ``A`` and ``B`` are NOT free 4x2 / 4x2x2 tensors -- rotating
    the coordinate system must rotate ``g`` and the galaxy's own ellipticity
    ``e=(M1,M2)`` together and leave the physical answer unchanged, which
    constrains ``A(x)·g`` and ``g^T B(x) g`` to specific combinations of ``g``
    and ``e`` for each output channel's spin:

    * Flux, size (``x0,x1``, spin-0 scalars): the shift must be rotation
      INVARIANT. The only invariant tensor linear in ``g`` once own-``e`` (also
      spin-2) is available to contract against is ``e·g = M1 g1 + M2 g2``; the
      only two invariant tensors quadratic in ``g`` are the isotropic ``|g|²``
      and the own-``e``-oriented ``(e·g)²``. So each of flux, size gets exactly
      3 free (invariant-scalar) coefficients: ``cA, c0, c1`` with
      ``shift = cA·(e·g) + 0.5·(c0·|g|² + c1·(e·g)²)``. A net with no ``e``
      input at all (the pre-fix design) can only ever emit ``cA=0`` and an
      isotropic ``B`` (``c1=0``) -- see memory shear-taylor-flux-size-blind-
      to-orientation and the 2026-08 own-e first-order fix. This is REAL: the
      real BFD Pqr derivatives show flux/size have a first-order response
      dominated by (and highly correlated with, r>0.96) own ``(M1,M2)``, and a
      second-order response with a substantial (~10%) own-e-oriented
      component riding on top of the dominant isotropic dilation term.
    * Ellipticity (``x2,x3=M1,M2`` as one complex ``e``, spin-2): by the same
      argument (worked out via Wirtinger derivatives on the exact reduced-
      shear map, and independently by parity: reflecting the whole system
      ``g->ḡ, e->ē`` must map any physical response to its own conjugate)
      there are exactly 4 equivariant tensors up to 2nd order in ``g``:
      ``g`` and ``ḡe²`` (1st order), ``|g|²e`` and ``g²ē`` (2nd order) -- each
      with a REAL (not complex) invariant-scalar coefficient. 4 free
      coefficients total: ``c_g, c_ge2, c_g2, c_g2b``.

    ONE coefficient net (``net_coeffs``, 10 outputs: ``cA_f,c0_f,c1_f`` for
    flux, ``cA_s,c0_s,c1_s`` for size, ``c_g,c_ge2,c_g2,c_g2b`` for
    ellipticity) covers all three channels, conditioned JOINTLY on the full
    invariant set ``(flux, size, |e|²)``. An earlier version split this into
    two nets (flux+size vs. ellipticity) solely so a since-removed
    ``--freeze-shear-response`` diagnostic could stop-gradient the ellipticity
    response alone; with that gone there's no reason not to use one net for
    all ten coefficients. Before that, an even earlier per-channel design
    conditioned on non-overlapping SUBSETS of the invariants (an artifact of
    an incremental, block-triangular-log-det-preserving design) -- which is
    why the measured 2nd-order own-e fit only reached r≈0.75 against real Pqr
    B12/(B11-B22) (flux/size never saw their own |e|² jointly with their own
    scale). Joint conditioning here fixes that structurally.

    The Jacobian/log-det and inverse are both GENERIC (not per-block
    closed-form): ``transform_and_log_det`` computes the exact Jacobian of the
    single ``_shift`` function via ``jax.jacfwd`` (the earlier design used
    separate closed-form derivations for the flux/size and spin-2 blocks,
    composed as if block-triangular -- an approximation that silently dropped
    a real cross-term once flux/size started depending on the spin-2 OUTPUT).
    ``inverse_and_log_det`` Picard-iterates the fixed point
    ``x = y - shift(x; g)`` from ``x=y`` (see ``_INVERSE_PICARD_ITERS``) instead
    of a hand-derived closed-form substitution per block -- geometric
    convergence, no differentiation involved anywhere in the iteration, so no
    nested-AD fragility risk (the concern that ruled out an actual Newton
    solve here previously).

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    dim : int
        Input dimensionality (must be 4 for galaxy moments).
    raw_cond_dim : int
        Conditioning dimensionality (>= 2 for ``[g1, g2]``; extra dims ignored).
    last_width, last_depth : int
        Hidden width / depth of the coefficient net.
    activation : callable
        Activation function.
    """

    dim: int = eqx.field(static=True)
    raw_cond_dim: int = eqx.field(static=True)
    net_coeffs: LipschitzCoeffNet  # (x0,x1,Lval) -> (10,): (cA_f,c0_f,c1_f, cA_s,c0_s,c1_s, c_g,c_ge2,c_g2,c_g2b)

    def __init__(self, key, dim, raw_cond_dim, last_width, last_depth, activation):
        if dim != 4:
            raise ValueError(
                "ShearTaylorLast is defined for the 4 galaxy moments (dim=4)."
            )
        self.dim = dim
        self.raw_cond_dim = raw_cond_dim
        # `activation` (the rest of the flow's shared nn_activation, e.g. SiLU) is
        # accepted for call-site compatibility but NOT used here -- LipschitzCoeffNet
        # pins its own activation to relu, which the Lipschitz-bound derivation
        # requires (see LipschitzCoeffNet's docstring). _SHEAR_COEFF_NET_LIPSCHITZ
        # is what actually closes the log-det exploit; see ShearTaylorLast._coeffs.
        self.net_coeffs = _zero_last_layer(
            LipschitzCoeffNet(key, 3, 10, last_width, last_depth, _SHEAR_COEFF_NET_LIPSCHITZ)
        )

    @property
    def shape(self):
        return (self.dim,)

    @property
    def cond_shape(self):
        return (self.raw_cond_dim,)

    @staticmethod
    def _cmul(xr, xi, yr, yi):
        return xr * yr - xi * yi, xr * yi + xi * yr

    def _coeffs(self, x0, x1, e1, e2):
        """Invariant-scalar coefficients at base point (x0,x1,e1,e2), jointly
        conditioned on all three invariants (flux, size, |e|^2).

        ``e1, e2`` here are STANDARDISED (M1/Mr, M2/Mr) -- the same units as
        ``x0, x1`` -- not raw ellipticity. No clipping or cleaning is applied
        to e1, e2: they feed ``log1p(e1^2+e2^2)`` exactly as computed (see the
        2026-08-06 note above ``_SHEAR_COEFF_INPUT_CLIP`` for why a clip was
        tried and removed here).

        2026-08-06: ``ShearTaylorLast.transform_and_log_det``'s
        ``log|det(I+J)|`` has NO upper bound if ``J`` (the Jacobian of
        ``_shift``, which calls this method) is left free to scale with an
        unconstrained coefficient net. Verified directly on a trained
        checkpoint: at a routine (not rare) point (e_std=5), ``log_det`` at
        the training stencil's outer ring grew from +7 to +18.5 between
        steps 30k-45k of an otherwise-ordinary NLL run, exactly tracking a
        sharp drop in the training loss -- i.e. NLL is UNBOUNDED BELOW along
        this direction: gradient descent can buy arbitrary fake
        log-likelihood by inflating the Jacobian determinant at real, dense
        data, with no need to actually fit the physical shear response.

        Two value-side fixes were tried and rejected: a synthetic OOD probe
        penalty and an input clip only relocated the same blowup elsewhere;
        a tanh output clamp stopped the divergence but training still walked
        every coefficient to whatever cap existed -- a follow-up measurement
        (scaling the trained net's raw output from 0 to 1 on a real batch)
        found NLL has a genuine, if small on average, one-sided preference
        for inflated coefficients, concentrated in the real high-ellipticity
        tail (up to +0.9 nats for a single template), so any fixed value cap
        just becomes the new wall regardless of where it's set.

        The actual fix is architectural: ``self.net_coeffs`` is a
        :class:`LipschitzCoeffNet`, whose spectral-normalised weight layers
        give it a hard, provable global Lipschitz constant
        (``_SHEAR_COEFF_NET_LIPSCHITZ``) -- ``||coeffs(x)-coeffs(x')|| <=
        L*||x-x'||`` for every ``x, x'``, by construction, regardless of what
        training finds. Combined with the shear stencil's own small
        ``|g|<=0.05``, this directly bounds ``J``'s operator norm (the
        coefficients enter ``_shift`` multiplied by ``g``/``g^2``, so their
        LOCAL RATE OF CHANGE w.r.t. x -- not their raw magnitude -- is what
        controls ``J``), which bounds ``det(I+J)`` between ``(1-L)^4`` and
        ``(1+L)^4`` -- a mathematical consequence of the function class, not
        a value anyone chose to clamp outputs to. No clamp is applied here
        anymore; the returned coefficients are whatever the (Lipschitz-safe)
        net computes.
        """
        inv = jnp.array([
            _bound_coeff_input(x0), _bound_coeff_input(x1),
            jnp.log1p(e1 * e1 + e2 * e2),
        ])
        cA_f, c0_f, c1_f, cA_s, c0_s, c1_s, c_g, c_ge2, c_g2, c_g2b = self.net_coeffs(inv)
        s2 = _SHEAR_STENCIL_OUTER_G ** 2
        # c0,c1,c_g2,c_g2b are all 2nd-order (B) coefficients -- nets emit them
        # in "shift at the outer ring" units (see _SHEAR_STENCIL_OUTER_G above).
        return (cA_f, c0_f / s2, c1_f / s2, cA_s, c0_s / s2, c1_s / s2,
                c_g, c_ge2, c_g2 / s2, c_g2b / s2)

    def _shift(self, x, g1, g2):
        """Full (4,) shift = A(x).g + 0.5 g^T B(x) g at base point x=(x0,x1,e1,e2)."""
        x0, x1, e1, e2 = x[0], x[1], x[2], x[3]
        cA_f, c0_f, c1_f, cA_s, c0_s, c1_s, c_g, c_ge2, c_g2, c_g2b = (
            self._coeffs(x0, x1, e1, e2)
        )

        # spin-0 (flux, size): A.g = cA*(e.g); B: c0*|g|^2 + c1*(e.g)^2 -- the
        # unique equivariant tensors, see class docstring.
        eg = e1 * g1 + e2 * g2
        gmag = g1 * g1 + g2 * g2

        def spin0_shift(cA, c0, c1):
            return cA * eg + 0.5 * (c0 * gmag + c1 * eg * eg)

        s_flux = spin0_shift(cA_f, c0_f, c1_f)
        s_size = spin0_shift(cA_s, c0_s, c1_s)

        # spin-2 (M1, M2): the 4 equivariant tensors {g, ḡe², |g|²e, g²ē}, real
        # invariant-scalar coefficients (parity forces the imaginary parts to
        # exactly 0 -- see class docstring).
        cm = self._cmul
        e2r, e2i = e1 * e1 - e2 * e2, 2.0 * e1 * e2
        gg_r, gg_i = g1 * g1 - g2 * g2, 2.0 * g1 * g2
        Tge2 = cm(g1, -g2, e2r, e2i)
        Tg2b = cm(gg_r, gg_i, e1, -e2)
        tensors = [(g1, g2), Tge2, (gmag * e1, gmag * e2), Tg2b]
        coeffs = [(c_g, 0.0), (c_ge2, 0.0), (c_g2, 0.0), (c_g2b, 0.0)]
        dr, di = 0.0, 0.0
        for (cr, ci), (tr, ti) in zip(coeffs, tensors):
            pr, pi = cm(cr, ci, tr, ti)
            dr, di = dr + pr, di + pi

        return jnp.array([s_flux, s_size, dr, di])

    def transform_and_log_det(self, x, condition=None):
        g1, g2 = condition[0], condition[1]

        def shift_fn(xx):
            return self._shift(xx, g1, g2)

        y = x + shift_fn(x)
        J = jax.jacfwd(shift_fn)(x)
        det = jnp.linalg.det(jnp.eye(4) + J)
        log_det = jnp.log(jnp.maximum(jnp.abs(det), 1e-10))
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        g1, g2 = condition[0], condition[1]

        def shift_fn(xx):
            return self._shift(xx, g1, g2)

        x = y
        for _ in range(_INVERSE_PICARD_ITERS):
            x = y - shift_fn(x)
        J = jax.jacfwd(shift_fn)(x)
        det = jnp.linalg.det(jnp.eye(4) + J)
        log_det = jnp.log(jnp.maximum(jnp.abs(det), 1e-10))
        return x, -log_det

    def shear_derivs(self, x):
        """First/second moment shear-response at ``x`` (a std moment at ``g=0``).

        Returns ``(A, B)`` with ``A`` shape ``(4, 2)`` = ``dy/dg`` and ``B``
        shape ``(4, 2, 2)`` = ``d^2y/dg^2`` at ``g=0``. Plain autodiff of
        ``_shift`` at the FIXED base point ``x`` -- no hand-derived cross-term
        needed (the earlier per-block design needed one once flux/size started
        depending on the spin-2 OUTPUT rather than a fixed base point).
        """
        def sh(gv):
            return self._shift(x, gv[0], gv[1])
        A = jax.jacfwd(sh)(jnp.zeros(2))
        B = jax.jacfwd(jax.jacfwd(sh))(jnp.zeros(2))
        return A, B

    def shear_derivs_generative(self, x):
        """Data-space *generative* shear-response ``d m / d g``, ``d^2 m / d g^2``.

        :meth:`shear_derivs` returns the layer's transform (encode-direction)
        coefficients ``A, B``. The generative response — how the *decoded*
        moment moves when the flow is conditioned on ``g`` — is the derivative
        of the inverse map. Implicit-differentiating ``m = shear⁻¹(u, g)`` at
        fixed ``u`` and ``g=0`` gives, in closed form from ``A, B`` and the
        coeff-net Jacobian ``∂A/∂x``::

            A_gen        = -A
            B_gen[i,a,b] = -B[i,a,b] + Σ_j (∂A[i,b]/∂x_j)·A[j,a]
                                     + Σ_j (∂A[i,a]/∂x_j)·A[j,b]

        This is this LAYER's own local generative response, at its own local
        coordinate ``x`` (its input/output at ``g=0``, since it's the identity
        there) — NOT necessarily the whole flow's data-space response, since
        :class:`SigmaXCouplingLayer` is chained after this layer (data-adjacent;
        see :class:`EarlyChain`) and is itself a nonlinear function of this
        layer's output. Composing this response through Σ_X's own local
        Jacobian/Hessian to get the full data-space response is done in
        ``models/flows.py::_shear_response``. ``∂A/∂x`` is a local Jacobian of
        the small coeff nets — not a whole-flow autodiff.
        Returns ``(A_gen (4,2), B_gen (4,2,2))``.
        """
        A, B = self.shear_derivs(x)  # (4,2), (4,2,2)
        dAdx = jax.jacfwd(lambda z: self.shear_derivs(z)[0])(x)  # (4,2,4)
        corr = jnp.einsum("ibj,ja->iab", dAdx, A) + jnp.einsum("iaj,jb->iab", dAdx, A)
        return -A, -B + corr


# ---------------------------------------------------------------------------
# EarlyChain
# ---------------------------------------------------------------------------


class EarlyChain(AbstractBijection):
    """Bijection chain of unconditional ``early`` layers plus one conditional ``last``.

    ``transform`` maps data->base (the density direction, since the flow wraps this
    in :class:`~flowjax.bijections.Invert`).  The log-absolute-determinant is the
    sum of all constituent determinants.

    ``last`` always runs data-adjacent: ``transform`` applies ``last`` first (data
    end) then ``early`` (unconditional bulk, base end) — the physically-ordered
    generative stack ``base -> bulk -> shear -> Σ_X -> data``. ``last`` is itself a
    (possibly merged) :class:`~flowjax.bijections.Chain` of the two conditional
    layers, so it is :class:`SigmaXCouplingLayer` (currently the innermost/
    data-adjacent sub-bijection of ``last``) whose own coefficients are directly
    the data-space response, with no bulk-Jacobian entanglement; ``ShearTaylorLast``
    sits one step further from the data (see ``models/flows.py::_shear_response``,
    which chains through Σ_X's local Jacobian to get the shear layer's own
    data-space response).

    Parameters
    ----------
    early : iterable of AbstractBijection
        Sequence of unconditional bijections.
    last : AbstractBijection
        The single conditional bijection.  Its ``cond_shape`` determines the
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
        x, lad_total = self.last.transform_and_log_det(x, condition)
        for b in self.early:
            x, lad = b.transform_and_log_det(x)
            lad_total = lad_total + lad
        return x, lad_total

    def inverse_and_log_det(self, y, condition=None):
        lad_total = jnp.zeros(y.shape[:-1], dtype=y.dtype)
        for b in reversed(self.early):
            y, lad = b.inverse_and_log_det(y)
            lad_total = lad_total + lad
        y, lad = self.last.inverse_and_log_det(y, condition)
        lad_total = lad_total + lad
        return y, lad_total


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
    last_layer_nn_width: int | None = None,
    last_layer_nn_depth: int | None = None,
    # Σ_X coupling layer — set sigmax_cond_dim=5 to enable
    sigmax_cond_dim: int | None = None,
    sigmax_nn_width: int = 32,
    sigmax_nn_depth: int = 2,
    sigmax_log_scale_mean: float = 2.0 * 5.991464547107982,  # 2*log(400)
    sigmax_log_scale_std: float = 1.0,
    sigmax_e_max: float = 0.2,
    sigmax_size_loc: float = 0.0,
    sigmax_layer_kind: str = "autoregressive",  # "autoregressive" | "block"
) -> Transformed:
    """Construct a masked autoregressive normalizing flow for BFD galaxy moments.

    Builds ``flow_layers - 1`` unconditional
    :class:`EquivariantAutoregressiveLayer` steps (each followed by a random
    equivariant permutation) followed by the conditional
    :class:`ShearTaylorLast` layer.  If ``sigmax_cond_dim=5``, a
    :class:`SigmaXCouplingLayer` is chained after ``ShearTaylorLast`` (i.e.
    data-adjacent; see :class:`EarlyChain`) so that the generative order is
    ``base -> bulk -> shear -> Σ_X -> data`` and the prior condition becomes
    ``[g1, g2, log_scale, e1, e2]``.

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
        Conditioning dimension for :class:`ShearTaylorLast`.
    last_layer_nn_width : int or None, optional
        Hidden width for the last-layer networks.  Falls back to
        ``nn_width`` if ``None``.
    last_layer_nn_depth : int or None, optional
        Number of hidden layers for the last-layer networks.  Falls back
        to ``nn_depth`` if ``None``.
    sigmax_cond_dim : int or None, optional
        If not ``None``, appends a :class:`SigmaXCouplingLayer` (data-adjacent,
        after ``ShearTaylorLast``) conditioned on a ``sigmax_cond_dim``-dimensional
        vector.  Set to 5 to enable full Σ_X conditioning.
    sigmax_nn_width : int, optional
        Hidden width for the Σ_X coupling network.  Default is 32.
    sigmax_nn_depth : int, optional
        Number of hidden layers for the Σ_X coupling network.  Default is 2.
    sigmax_layer_kind : {"autoregressive", "block"}, optional
        ``"autoregressive"`` (default) uses :class:`SigmaXCouplingLayer` (closed-form
        triangular inverse; ``s0``/``g_s`` coefficients are blind to flux/size/size
        respectively). ``"block"`` uses :class:`SigmaXBlockLayer`: a single joint
        coefficient net sees the full moment vector, with an exact autodiff log-det
        and an iterative (unrolled Newton) inverse instead of a closed-form one.

    Returns
    -------
    Transformed
        A flowjax ``Transformed`` distribution (base dist + bijection chain).
    """
    dim = base_dist.shape[-1]
    _last_width = last_layer_nn_width if last_layer_nn_width is not None else nn_width
    _last_depth = last_layer_nn_depth if last_layer_nn_depth is not None else nn_depth
    use_sigmax = sigmax_cond_dim is not None

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
    _, k_shear = jr.split(k_last)

    shear = ShearTaylorLast(
        k_shear,
        dim,
        last_layer_cond_dim,
        _last_width,
        _last_depth,
        nn_activation,
    )
    # No equivariant permute for the Taylor layer: it relies on the fixed
    # spin-0/spin-2 component order (0,1 spin-0; 2,3 spin-2), which a 2<->3 swap
    # would scramble.
    shear_with_perm = shear

    if use_sigmax:
        if sigmax_layer_kind == "block":
            sigmax = SigmaXBlockLayer(
                keys[-1],
                nn_width=sigmax_nn_width,
                nn_depth=sigmax_nn_depth,
                activation=nn_activation,
                full_cond_dim=sigmax_cond_dim,
                log_scale_mean=sigmax_log_scale_mean,
                e_max=sigmax_e_max,
                size_loc=sigmax_size_loc,
            )
        elif sigmax_layer_kind == "autoregressive":
            sigmax = SigmaXCouplingLayer(
                keys[-1],
                nn_width=sigmax_nn_width,
                nn_depth=sigmax_nn_depth,
                activation=nn_activation,
                full_cond_dim=sigmax_cond_dim,
                log_scale_mean=sigmax_log_scale_mean,
                log_scale_std=sigmax_log_scale_std,
                e_max=sigmax_e_max,
                size_loc=sigmax_size_loc,
            )
        else:
            raise ValueError(f"Unknown sigmax_layer_kind: {sigmax_layer_kind!r}")
        # NO spin-2-swapping permute between the two conditional layers: both
        # ShearTaylorLast (shear g) and SigmaXCouplingLayer (centroid ellipticity
        # e) couple to EXTERNAL spin-2 vectors in the *data* frame, so their spin-2
        # inputs must stay aligned with (M1, M2). A 2<->3 swap would rotate the
        # e-coupling 90 degrees (off-diagonal response: e1->M2, e2->M1) and make
        # the dipole/quadrupole terms unable to match the diagonal target — the
        # directional terms then never train. Use `shear` directly.
        #
        # Chain order [sigmax, shear] here means, in the GENERATIVE (inverse)
        # direction Chain applies reversed(bijections) = [shear, sigmax] — i.e.
        # bulk -> shear -> Σ_X -> data, matching the physical process (apply the
        # shear response to the intrinsic galaxy, THEN marginalise the centroid
        # measurement uncertainty on top). SigmaXCouplingLayer is therefore the
        # data-adjacent layer (see EarlyChain); ShearTaylorLast sits right after
        # the unconditional bulk. This was previously the other way around
        # (Σ_X before shear, shear data-adjacent) — see git history / memory
        # "flow-layer-order-shear-sigmax-inverted".
        last = Chain([sigmax, shear]).merge_chains()
    else:
        last = shear_with_perm

    bijection = EarlyChain(early_layers, last)

    if invert:
        bijection = Invert(bijection)
    return Transformed(base_dist, bijection)
