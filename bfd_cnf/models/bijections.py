"""
models/bijections.py
====================
Custom flowjax bijection classes used in the bfd_cnf normalizing-flow model:

  - RawMomentStandardize   — transforms raw moments to standardized coordinates
  - BoundedAffine          — affine bijection with bounded scale parameter
  - Spin0AutoregressiveLayer / Spin2CouplingLayer / EquivariantAutoregressiveLayer
  - ExplicitPolyLast / ShearTaylorLast / SigmaXCouplingLayer / EarlyChain
  - new_masked_autoregressive_flow

Helper functions: _bounded_log_scale, _bounded_scale, _inv_bounded_scale,
_raw2std_jacobian_single_jax, propagate_cov_to_std_jax.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, ClassVar

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
        loc1 = out1[0]
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
        loc1 = out1[0]
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
# ExplicitPolyLast
# ---------------------------------------------------------------------------


class ExplicitPolyLast(AbstractBijection):
    """Conditional last-layer bijection using explicit polynomial shear features.

    Implements an affine coupling layer whose scale and shift coefficients
    are polynomials in the shear components ``(g1, g2)`` with network-
    generated coefficients conditioned on the transformed spin-0 inputs.
    Optionally includes quadratic (``|g|²``) features.

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

    def __init__(
        self,
        key,
        dim,
        raw_cond_dim,
        use_quadratic,
        last_width,
        last_depth,
        activation,
    ):
        self.dim = dim
        self.raw_cond_dim = raw_cond_dim
        self.use_quadratic = use_quadratic
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
    by :class:`ExplicitPolyLast`).

    The transform is a *physically constrained* map — it allows only the changes
    the centroid marginalisation produces to leading order (analogous to
    :class:`ExplicitPolyLast` for shear, but a different effect):

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
# |e|^2 already gets this treatment via log1p (see _spin2_shift); flux/size need
# the same. A hard clip is exact identity for any in-distribution template and
# only ever engages off-manifold, so it is a no-op for a trained model's normal
# behaviour.
_SHEAR_COEFF_INPUT_CLIP = 8.0


def _bound_coeff_input(x):
    return jnp.clip(x, -_SHEAR_COEFF_INPUT_CLIP, _SHEAR_COEFF_INPUT_CLIP)


class ShearTaylorLast(AbstractBijection):
    """Structural shear layer: an additive Taylor-in-g moment displacement.

    The final conditional shear layer — the structural alternative to
    :class:`ExplicitPolyLast`.  Instead of folding ``g`` into affine scale/shift
    coefficients (which mixes the shear orders), it applies an *explicit*
    second-order Taylor displacement in the shear ``g = (g1, g2)``::

        y[i] = x[i] + A[i]·g + 1/2 g^T B[i] g

    so the first- and second-order shear responses are read straight off the named
    coefficient fields ``A`` (``dy/dg`` at ``g=0``) and ``B`` (``d^2y/dg^2`` at
    ``g=0``) — the moments' shear response, exposed via :meth:`shear_derivs` for
    diagnostics and Sobolev supervision.

    Structure / equivariance:

    * The shift depends on ``g`` and on the *preceding* components only (masked-
      autoregressive), so ``dy/dx`` is unit lower-triangular and ``log|det J| = 0``:
      a valid, closed-form-invertible layer in both directions.
    * Spin-0 components (flux ``x0``, size ``x1``) get NO linear term (``A=0``): a
      spin-0 moment has no first-order response to the spin-2 shear.  Only the even
      ``|g|^2`` second-order term acts on them.
    * Spin-2 components (ellipticity ``x2, x3``) carry the dipole ``A·g`` (leading
      shear response) plus the second-order term.
    * Zero-initialised coeff nets ⇒ the layer is the **identity at init, and at
      ``g=0`` always**.  So the ``g=0`` prior shape comes entirely from the earlier
      (unconditional) layers and this layer only imprints the shear displacement on
      top.  That clean separation is what lets :meth:`shear_derivs(x)` be evaluated
      at a data moment directly (at ``g=0`` the layer's input *is* the data moment).

    ``g1, g2`` are read from ``condition[:2]``; extra dims (the Σ_X
    ``[log_scale, e1, e2]``) are ignored — Σ_X is handled by
    :class:`SigmaXCouplingLayer`, chained after this one.

    ponytail: the spin-2 dipole coefficient depends on the preceding components
    (flux, size, and — for M2 — M1) but not on a component's own value, so a single
    layer cannot make the first-order response depend on the galaxy's own |e|.  The
    Sobolev supervision (which uses the whole flow's response) carries the own-|e|
    dependence; add an own-|e| term here only if that proves insufficient.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key (split for the four coefficient nets).
    dim : int
        Input dimensionality (must be 4 for galaxy moments).
    raw_cond_dim : int
        Conditioning dimensionality (>= 2 for ``[g1, g2]``; extra dims ignored).
    last_width, last_depth : int
        Hidden width / depth of the coefficient nets.
    activation : callable
        Activation function.
    """

    dim: int = eqx.field(static=True)
    raw_cond_dim: int = eqx.field(static=True)
    own_e: bool = eqx.field(static=True)
    split_ab: bool = eqx.field(static=True)
    spin2_owne: bool = eqx.field(static=True)
    net_flux: CoeffNet  # ()          -> (3,)  B0 (spin-0, even only)
    net_size: CoeffNet  # (x0,)       -> (3,)  B1 (spin-0, even only)
    net_m1: CoeffNet  # (x0,x1)       -> (5,) A2+B2, or (2,) A2 only when split_ab
    net_m2: CoeffNet  # (x0,x1,x2)    -> (5,) A3+B3, or (2,) A3 only when split_ab
    net_m1_B: Any  # (x0,x1)          -> (3,)  B2, or None (only when split_ab)
    net_m2_B: Any  # (x0,x1,x2)       -> (3,)  B3, or None (only when split_ab)
    net_m1_e: Any  # (x0,x1,m2_final) -> (5,)  own-|e| M1 correction, or None

    def __init__(self, key, dim, raw_cond_dim, last_width, last_depth, activation,
                 own_e=False, split_ab=False, spin2_owne=False):
        if dim != 4:
            raise ValueError(
                "ShearTaylorLast is defined for the 4 galaxy moments (dim=4)."
            )
        self.dim = dim
        self.raw_cond_dim = raw_cond_dim
        self.own_e = bool(own_e)
        # split_ab: give the spin-2 SECOND-order coeff B its OWN net instead of sharing
        # the A trunk. The first-order NLL+sob1 gradient dominates a shared trunk and
        # starves the sob2 (2nd-order) supervision, pinning B at ~0; a dedicated B trunk
        # lets sob2 train it. STATIC structure — needs --from-scratch + fresh --prior-out.
        self.split_ab = bool(split_ab)
        # spin2_owne: condition the SPIN-2 (M1,M2) shear coefficients on the rotation-
        # invariant |e|^2 = M1^2 + M2^2, so the shift can depend on the galaxy's OWN
        # ellipticity — the info a masked layer legally can't see (M1's 2nd-order response
        # is 100% own-|e|; R^2=0 from flux,size alone). The (M1,M2) block becomes a JOINT
        # coupling with a real (non-unit) log-det and an iterative inverse — the price for
        # own-|e| while keeping spin-2 equivariance (|e|^2 is a spin-0 invariant).
        self.spin2_owne = bool(spin2_owne)
        k0, k1, k2, k3, k4, k5, k6 = jr.split(key, 7)
        if self.spin2_owne:
            # Equivariant spin-2 block: the (M1,M2) shift is built from equivariant tensors
            # of e=(M1,M2) and g, with complex INVARIANT-scalar coefficients (functions of
            # flux,size,|e|^2). net_m1 -> 1st-order coeffs (c_g, c_ge2 = 4 reals);
            # net_m2 -> 2nd-order coeffs (c_g2, c_g2b = 4 reals). No masked A/B split.
            m1_in = m2_in = 3
            m1_out = m2_out = 4
        else:
            m1_in, m2_in = 2, 3
            m1_out = 2 if self.split_ab else 5  # A only vs A+B
            m2_out = 2 if self.split_ab else 5
        nets = [
            CoeffNet(k0, 0, 3, last_width, last_depth, activation),           # net_flux (B0)
            CoeffNet(k1, 1, 3, last_width, last_depth, activation),           # net_size (B1)
            CoeffNet(k2, m1_in, m1_out, last_width, last_depth, activation),  # net_m1
            CoeffNet(k3, m2_in, m2_out, last_width, last_depth, activation),  # net_m2
        ]
        # Zero each net's final layer -> all coefficients start at 0, so the layer is
        # the identity at init (a small perturbation of the g=0 base shape).
        nets = [_zero_last_layer(n) for n in nets]
        self.net_flux, self.net_size, self.net_m1, self.net_m2 = nets
        if self.split_ab and not self.spin2_owne:
            self.net_m1_B = _zero_last_layer(
                CoeffNet(k5, 2, 3, last_width, last_depth, activation)      # (x0,x1)->B2
            )
            self.net_m2_B = _zero_last_layer(
                CoeffNet(k6, 3, 3, last_width, last_depth, activation)      # (x0,x1,x2)->B3
            )
        else:
            self.net_m1_B = None
            self.net_m2_B = None
        # own-|e|: a second-stage dipole/quad correction to M1 conditioned on the FINAL
        # M2 (spin-2 magnitude the masked net_m1 can't see). Conditioning on the already-
        # shifted y[3] keeps a closed-form, unit-determinant inverse (see (in)transform).
        if self.own_e:
            self.net_m1_e = _zero_last_layer(
                CoeffNet(k4, 3, 5, last_width, last_depth, activation)  # (x0,x1,m2)->A2e+B2e
            )
        else:
            self.net_m1_e = None

    @property
    def shape(self):
        return (self.dim,)

    @property
    def cond_shape(self):
        return (self.raw_cond_dim,)

    @staticmethod
    def _shift(A, B, g1, g2):
        # A: (2,) linear coeffs; B: (3,) = (B11, B22, B12) second-order coeffs.
        # y - x = A0 g1 + A1 g2 + 1/2 B11 g1^2 + 1/2 B22 g2^2 + B12 g1 g2.
        return (
            A[0] * g1
            + A[1] * g2
            + 0.5 * B[0] * g1**2
            + 0.5 * B[1] * g2**2
            + B[2] * g1 * g2
        )

    def _coeffs(self, x0, x1, x2):
        """Per-component (A, B) coefficient tuples at the given preceding components."""
        x0, x1, x2 = _bound_coeff_input(x0), _bound_coeff_input(x1), _bound_coeff_input(x2)
        B0 = self.net_flux(jnp.zeros(0))  # (3,)
        B1 = self.net_size(jnp.array([x0]))  # (3,)
        x01 = jnp.array([x0, x1])
        x012 = jnp.array([x0, x1, x2])
        zeroA = jnp.zeros(2)
        if self.split_ab:
            A2, B2 = self.net_m1(x01), self.net_m1_B(x01)        # (2,), (3,)
            A3, B3 = self.net_m2(x012), self.net_m2_B(x012)      # (2,), (3,)
        else:
            o2, o3 = self.net_m1(x01), self.net_m2(x012)         # (5,), (5,)
            A2, B2, A3, B3 = o2[:2], o2[2:], o3[:2], o3[2:]
        A = (zeroA, zeroA, A2, A3)
        B = (B0, B1, B2, B3)
        return A, B

    @staticmethod
    def _cmul(xr, xi, yr, yi):
        return xr * yr - xi * yi, xr * yi + xi * yr

    def _spin2_shift(self, x0, x1, e1, e2, g1, g2):
        """Equivariant spin-2 (M1,M2) shift as a fn of e=(e1,e2) at fixed (x0,x1,g).

        Represents e,g as complex numbers. The shift is a sum of the equivariant tensors
        {g, ḡe², |g|²e, g²ē} — each transforming as spin-2 — with complex INVARIANT-scalar
        coefficients (c_g, c_ge2 from net_m1; c_g2, c_g2b from net_m2) that depend only on
        (flux, size, |e|²). Rotation-equivariant by construction, yet own-|e| aware.
        Returns (δM1, δM2)."""
        cm = self._cmul
        # Feed a SOFT-BOUNDED |e|^2 (log1p) so pathological large-|e| samples (e.g. an
        # untrained flow's garbage draws) can't blow the coeffs up and flip det(I+J)<0.
        # Monotonic in the invariant |e|^2 ⇒ equivariance + representational power intact.
        # x0, x1 (flux, size) get the same off-manifold guard (see _bound_coeff_input).
        inv = jnp.array([
            _bound_coeff_input(x0), _bound_coeff_input(x1),
            jnp.log1p(e1 * e1 + e2 * e2),
        ])
        a = self.net_m1(inv)                 # (c_g_r, c_g_i, c_ge2_r, c_ge2_i)
        b = self.net_m2(inv)                 # (c_g2_r, c_g2_i, c_g2b_r, c_g2b_i)
        e2r, e2i = e1 * e1 - e2 * e2, 2.0 * e1 * e2          # e^2
        gg_r, gg_i = g1 * g1 - g2 * g2, 2.0 * g1 * g2        # g^2
        gmag = g1 * g1 + g2 * g2                             # |g|^2
        Tge2 = cm(g1, -g2, e2r, e2i)                         # ḡ e^2
        Tg2b = cm(gg_r, gg_i, e1, -e2)                       # g^2 ē
        tensors = [(g1, g2), Tge2, (gmag * e1, gmag * e2), Tg2b]
        coeffs = [(a[0], a[1]), (a[2], a[3]), (b[0], b[1]), (b[2], b[3])]
        dr, di = 0.0, 0.0
        for (cr, ci), (tr, ti) in zip(coeffs, tensors):
            pr, pi = cm(cr, ci, tr, ti)
            dr, di = dr + pr, di + pi
        return dr, di

    def _spin2_transform(self, x, g1, g2):
        """(y2,y3) and log|det d(y2,y3)/d(x2,x3)| for the equivariant spin-2 block.

        EXACT closed form for det(I+J), J = d(shift)/d(e) — not the O(g^2)-truncated
        polynomial this replaced (see git history for why THAT was itself a fix for
        slogdet's near-singular-J gradient blowup, and for why the truncation turned
        out to be its own bug: log|det(I+J)| grows only ~log(coeff) in the coeff
        nets' output magnitude, but a truncated `tr(J)-1/2 tr(J^2)` — missing the
        OUTER log entirely — grows ~coeff^2, an unbounded free reward for the
        optimizer to exploit by inflating the coeff nets without bound; this is what
        broke the sob0_narrowsx training run).

        Derivation: cm = self._cmul. Write shift as
            delta(e) = c_g(L)*g + c_ge2(L)*(gbar*e^2) + c_g2(L)*(|g|^2*e) + c_g2b(L)*(g^2*ebar)
        with complex coeffs c_* depending on e ONLY through L = log1p(|e|^2), and split
        J = J_a + w*p^T:
          * J_a is the Jacobian holding the coeffs FIXED (only e's explicit appearances
            differentiated) — a complex-linear-plus-antilinear map delta ~ alpha*de +
            beta*conj(de), alpha = 2*c_ge2*gbar*e + c_g2*|g|^2, beta = c_g2b*g^2. A
            standard result (re-derived and verified here, not assumed) for such a map:
                det(I+J_a) = |1+alpha|^2 - |beta|^2   (exact, no truncation)
            — matches the classic reduced-shear composition Jacobian; its log grows
            correctly (~log|coeff|), closing the exploit above.
          * w*p^T is the RANK-1 correction from d(c_i)/dL != 0 (own_e-style |e|^2
            sensitivity): p = kappa*(2e1,2e2), kappa=d(log1p)/d|e|^2 = 1/(1+|e|^2);
            w = sum_i (d c_i/dL) * T_i(e) (the same 4 equivariant tensors T_i as the
            shift itself). d(c_i)/dL is a LOCAL Jacobian of net_m1/net_m2 w.r.t. their
            OWN scalar L input — cheap, not a whole-flow autodiff, same pattern as
            shear_derivs_generative's dA/dx for own_e.
        The matrix determinant lemma then gives the FULL exact determinant with no
        matrix inverse anywhere (division-free):
            det(I+J) = det(I+J_a) + p^T . adj(I+J_a) . w
        `adj` is the 2x2 adjugate, itself closed-form from alpha,beta. log|det| still
        has a genuine (not fake) gradient singularity exactly where the map becomes
        non-invertible — floored below like other near-singular quantities in this
        codebase (e.g. BFD's own detj floor).
        """
        cm = self._cmul
        x0, x1, e1, e2 = x[0], x[1], x[2], x[3]
        s = e1 * e1 + e2 * e2
        Lval = jnp.log1p(s)
        inv = jnp.array([_bound_coeff_input(x0), _bound_coeff_input(x1), Lval])
        a = self.net_m1(inv)                     # (c_g_r, c_g_i, c_ge2_r, c_ge2_i)
        b = self.net_m2(inv)                     # (c_g2_r, c_g2_i, c_g2b_r, c_g2b_i)
        dadL = jax.jacfwd(self.net_m1)(inv)[:, 2]  # d(a)/dL — local coeff-net Jacobian
        dbdL = jax.jacfwd(self.net_m2)(inv)[:, 2]  # d(b)/dL

        e2r, e2i = e1 * e1 - e2 * e2, 2.0 * e1 * e2          # e^2
        gg_r, gg_i = g1 * g1 - g2 * g2, 2.0 * g1 * g2        # g^2
        gmag = g1 * g1 + g2 * g2                             # |g|^2
        T1 = (g1, g2)                                        # g
        Tge2 = cm(g1, -g2, e2r, e2i)                         # gbar . e^2
        T3 = (gmag * e1, gmag * e2)                          # |g|^2 . e
        Tg2b = cm(gg_r, gg_i, e1, -e2)                       # g^2 . ebar
        coeffs = [(a[0], a[1]), (a[2], a[3]), (b[0], b[1]), (b[2], b[3])]
        tensors = [T1, Tge2, T3, Tg2b]

        # shift value (same formula _spin2_shift computes; done here directly since
        # a,b are already in hand)
        dr, di = 0.0, 0.0
        for (cr, ci), (tr, ti) in zip(coeffs, tensors):
            pr, pi = cm(cr, ci, tr, ti)
            dr, di = dr + pr, di + pi

        # alpha, beta of the coeffs-fixed piece J_a
        two_cge2_gbar_e = cm(2.0 * a[2], 2.0 * a[3], *cm(g1, -g2, e1, e2))
        cg2_gmag = cm(b[0], b[1], gmag, 0.0)
        alpha_r = two_cge2_gbar_e[0] + cg2_gmag[0]
        alpha_i = two_cge2_gbar_e[1] + cg2_gmag[1]
        beta_r, beta_i = cm(b[2], b[3], gg_r, gg_i)

        det_a = (1.0 + alpha_r) ** 2 + alpha_i ** 2 - beta_r ** 2 - beta_i ** 2
        m00 = 1.0 + alpha_r + beta_r
        m01 = -alpha_i + beta_i
        m10 = alpha_i + beta_i
        m11 = 1.0 + alpha_r - beta_r

        # rank-1 correction from d(coeffs)/dL
        dcoeffs_dL = [(dadL[0], dadL[1]), (dadL[2], dadL[3]), (dbdL[0], dbdL[1]), (dbdL[2], dbdL[3])]
        W_r, W_i = 0.0, 0.0
        for (dcr, dci), (tr, ti) in zip(dcoeffs_dL, tensors):
            pr, pi = cm(dcr, dci, tr, ti)
            W_r, W_i = W_r + pr, W_i + pi
        kappa = 1.0 / (1.0 + s)
        p1, p2 = kappa * 2.0 * e1, kappa * 2.0 * e2
        # p^T . adj(I+J_a) . w, adj(M)=[[m11,-m01],[-m10,m00]]
        correction = p1 * (m11 * W_r - m01 * W_i) + p2 * (m00 * W_i - m10 * W_r)

        det_total = det_a + correction
        log_det = jnp.log(jnp.maximum(jnp.abs(det_total), 1e-10))
        return x[2] + dr, x[3] + di, log_det

    def transform_and_log_det(self, x, condition=None):
        g1, g2 = condition[0], condition[1]
        if self.spin2_owne:
            # spin-0 (0,1): masked, unit-triangular (no det contribution).
            B0 = self.net_flux(jnp.zeros(0))
            B1 = self.net_size(jnp.array([_bound_coeff_input(x[0])]))
            y0 = x[0] + self._shift(jnp.zeros(2), B0, g1, g2)
            y1 = x[1] + self._shift(jnp.zeros(2), B1, g1, g2)
            # spin-2 (2,3): equivariant joint coupling with a real 2x2 log-det.
            y2, y3, log_det = self._spin2_transform(x, g1, g2)
            return jnp.stack([y0, y1, y2, y3]), log_det
        A, B = self._coeffs(x[0], x[1], x[2])
        shifts = jnp.stack([self._shift(A[i], B[i], g1, g2) for i in range(4)])
        y = x + shifts
        if self.own_e:
            # Stage 2: extra M1 shift conditioned on the FINAL M2 (y[3], unchanged here),
            # giving M1's response an own-|e| dependence. det stays 1 (see class docstring).
            oe = self.net_m1_e(jnp.array([
                _bound_coeff_input(x[0]), _bound_coeff_input(x[1]), _bound_coeff_input(y[3]),
            ]))  # (5,) = A2e + B2e
            y = y.at[2].add(self._shift(oe[:2], oe[2:], g1, g2))
        return y, jnp.zeros(())

    def inverse_and_log_det(self, y, condition=None):
        g1, g2 = condition[0], condition[1]
        if self.spin2_owne:
            B0 = self.net_flux(jnp.zeros(0))
            x0 = y[0] - self._shift(jnp.zeros(2), B0, g1, g2)
            B1 = self.net_size(jnp.array([_bound_coeff_input(x0)]))
            x1 = y[1] - self._shift(jnp.zeros(2), B1, g1, g2)
            # spin-2 inverse: closed-form substitution inverse — NOT a Newton solve,
            # and NOT routed through shear_derivs_generative's extra derivative layer
            # either (that needed d(A)/dx via a further jacfwd on top of the jacfwd/
            # jacfwd already inside shear_derivs — 3 nested AD levels, then a 4th from
            # training's own reverse-mode grad on top: numerically fragile enough to
            # itself produce NaN *gradients* for an off-manifold y, even though the
            # forward *value* stayed finite — see the 2026-07-27 investigation).
            # Instead: evaluate the (exact, closed-form) shift function directly AT y
            # instead of at the unknown x — a single function call, no differentiation
            # of any kind needed for this step. Since shift is O(g), x = y-shift(x;g)
            # and x_approx = y-shift(y;g) differ only by shift(x;g)-shift(y;g) =
            # O(g)*O(x-y) = O(g)*O(g) = O(g^2) — negligible at the |g|<=0.02-0.05 this
            # layer is used at (0.02^2 = 4e-4), and still an order better than the
            # model's own g^3 truncation floor is trying to protect anyway. Composed
            # of only coeff-net + polynomial evaluations: provably finite value AND
            # gradient for any finite input, with no nested-AD amplification at all.
            def shift23(e):
                return jnp.stack(self._spin2_shift(x0, x1, e[0], e[1], g1, g2))
            d = shift23(jnp.array([y[2], y[3]]))
            xrec = jnp.array([x0, x1, y[2] - d[0], y[3] - d[1]])
            _, _, fwd_ld = self._spin2_transform(xrec, g1, g2)
            return xrec, -fwd_ld
        # Sequential: shift[i] depends only on the already-recovered x[<i] (and, for the
        # own-|e| M1 correction, on the FINAL M2 = y[3], which is given).
        B0 = self.net_flux(jnp.zeros(0))
        x0 = y[0] - self._shift(jnp.zeros(2), B0, g1, g2)
        B1 = self.net_size(jnp.array([_bound_coeff_input(x0)]))
        x1 = y[1] - self._shift(jnp.zeros(2), B1, g1, g2)
        x01 = jnp.array([_bound_coeff_input(x0), _bound_coeff_input(x1)])
        if self.split_ab:
            A2, B2 = self.net_m1(x01), self.net_m1_B(x01)
        else:
            o2 = self.net_m1(x01)
            A2, B2 = o2[:2], o2[2:]
        m1_shift = self._shift(A2, B2, g1, g2)
        if self.own_e:
            oe = self.net_m1_e(jnp.array([
                _bound_coeff_input(x0), _bound_coeff_input(x1), _bound_coeff_input(y[3]),
            ]))  # final M2 = y[3], directly known
            m1_shift = m1_shift + self._shift(oe[:2], oe[2:], g1, g2)
        x2 = y[2] - m1_shift
        x012 = jnp.array([_bound_coeff_input(x0), _bound_coeff_input(x1), _bound_coeff_input(x2)])
        if self.split_ab:
            A3, B3 = self.net_m2(x012), self.net_m2_B(x012)
        else:
            o3 = self.net_m2(x012)
            A3, B3 = o3[:2], o3[2:]
        x3 = y[3] - self._shift(A3, B3, g1, g2)
        return jnp.stack([x0, x1, x2, x3]), jnp.zeros(())

    def shear_derivs(self, x, own_e_stop_grad=False):
        """First/second moment shear-response at ``x`` (a std moment at ``g=0``).

        Returns ``(A, B)`` with ``A`` shape ``(4, 2)`` = ``dy/dg`` and ``B`` shape
        ``(4, 2, 2)`` = ``d^2y/dg^2`` at ``g=0``.  Because the layer is the identity
        at ``g=0``, its input there IS the data moment, so ``x`` may be passed as a
        data moment directly.

        ``own_e_stop_grad``: when ``self.own_e``, the M1 correction ``net_m1_e`` is
        conditioned on ``u3 = x[3]`` playing the role of the layer's OWN output for
        channel 3 (``y[3]`` in :meth:`transform_and_log_det`) — a quantity held FIXED
        in :meth:`shear_derivs_generative`'s implicit differentiation (it IS the
        component of the fixed target ``u``, not the unknown ``m3`` being solved
        for), even though at ``g=0`` it numerically equals ``x[3]``.  Pass ``True``
        when this method's *Jacobian* w.r.t. ``x`` is about to be taken (as
        :meth:`shear_derivs_generative` does), so ``∂/∂x[3]`` doesn't spuriously
        pick up ``net_m1_e``'s dependence on that slot — the VALUE is unaffected
        either way (``stop_gradient`` is the identity in forward mode).
        """
        if self.spin2_owne:
            B0 = self.net_flux(jnp.zeros(0))
            B1 = self.net_size(jnp.array([x[0]]))
            # spin-2 A,B by autodiff of the equivariant shift w.r.t. g at g=0.
            def sh(gv):
                return jnp.stack(self._spin2_shift(x[0], x[1], x[2], x[3], gv[0], gv[1]))
            A23 = jax.jacfwd(sh)(jnp.zeros(2))                    # (2,2) d(M1,M2)/d(g)
            B23 = jax.jacfwd(jax.jacfwd(sh))(jnp.zeros(2))        # (2,2,2) d^2/dg^2

            def _Bm(bb):
                return jnp.array([[bb[0], bb[2]], [bb[2], bb[1]]])

            A_mat = jnp.stack([jnp.zeros(2), jnp.zeros(2), A23[0], A23[1]])
            B_mat = jnp.stack([_Bm(B0), _Bm(B1), B23[0], B23[1]])
            return A_mat, B_mat
        A, B = self._coeffs(x[0], x[1], x[2])
        if self.own_e:
            u3 = jax.lax.stop_gradient(x[3]) if own_e_stop_grad else x[3]
            oe = self.net_m1_e(jnp.array([
                _bound_coeff_input(x[0]), _bound_coeff_input(x[1]), _bound_coeff_input(u3),
            ]))  # (5,) = A2e + B2e
            A = (A[0], A[1], A[2] + oe[:2], A[3])
            B = (B[0], B[1], B[2] + oe[2:], B[3])
        A_mat = jnp.stack(A)  # (4, 2)

        def _Bmat(b):
            return jnp.array([[b[0], b[2]], [b[2], b[1]]])  # (B11,B22,B12) -> 2x2

        B_mat = jnp.stack([_Bmat(B[i]) for i in range(4)])  # (4, 2, 2)
        return A_mat, B_mat

    def shear_derivs_generative(self, x):
        """Data-space *generative* shear-response ``d m / d g``, ``d^2 m / d g^2``.

        :meth:`shear_derivs` returns the layer's transform (encode-direction)
        coefficients ``A, B``.  The generative response — how the *decoded* moment
        moves when the flow is conditioned on ``g`` — is the derivative of the
        inverse map.  Implicit-differentiating ``m = shear⁻¹(u, g)`` at fixed ``u``
        and ``g=0`` gives, in closed form from ``A, B`` and the coeff-net Jacobian
        ``∂A/∂x``::

            A_gen        = -A
            B_gen[i,a,b] = -B[i,a,b] + Σ_j (∂A[i,b]/∂x_j)·A[j,a]
                                     + Σ_j (∂A[i,a]/∂x_j)·A[j,b]

        This equals the whole-flow autodiff response (``_flow_shear_derivs``) EXACTLY
        **iff this layer is data-adjacent** (``EarlyChain.conditional_first=True``);
        base-adjacent, the response also threads through the unconditional bulk and
        this identity does not hold.  ``∂A/∂x`` is a local Jacobian of the small coeff
        nets — not a whole-flow autodiff.  Returns ``(A_gen (4,2), B_gen (4,2,2))``.

        Also correct under ``own_e``: ``A[2],B[2]`` from :meth:`shear_derivs` already
        fold in the ``net_m1_e`` correction, so this formula applies unchanged — with
        ONE exception. ``net_m1_e`` is conditioned on ``u3`` (fixed, not ``x[3]``
        treated as the unknown ``m3``; see :meth:`shear_derivs`'s docstring), so the
        ``∂A/∂x`` Jacobian must be taken with ``own_e_stop_grad=True`` or the
        ``i=2,j=3`` entry of ``∂A/∂x`` would spuriously include ``net_m1_e``'s
        dependence on that slot, corrupting ``B_gen`` for M1 (the very channel
        ``own_e`` exists to fix). No other entry of ``∂A/∂x`` is affected: no other
        channel's coefficient net takes ``x[3]`` as an input.
        """
        A, B = self.shear_derivs(x)  # (4,2), (4,2,2)
        dAdx = jax.jacfwd(lambda z: self.shear_derivs(z, own_e_stop_grad=True)[0])(x)  # (4,2,4)
        corr = jnp.einsum("ibj,ja->iab", dAdx, A) + jnp.einsum("iaj,jb->iab", dAdx, A)
        return -A, -B + corr

    def coeff_ood_penalty(self, key, n=16, lo=4.0, hi=_SHEAR_COEFF_INPUT_CLIP):
        """Mean squared coefficient magnitude at synthetic off-template probes.

        ``_bound_coeff_input`` gives a hard, structural guarantee that the coeff
        nets are never QUERIED beyond ``hi`` sigma; it says nothing about what they
        output there, which — for an unbounded ReLU/SiLU MLP trained only on the
        real (|x0|,|x1| ~ few sigma) template population — is arbitrary extrapolation.
        This term trains the nets themselves to decay toward zero on the (lo, hi)
        transition band nobody's real template ever visits, so the region the hard
        clip folds everything past ``hi`` onto is a place the nets have actually
        learned to be small, not just wherever training happened to leave it.
        Probes sample (flux, size) outside (-lo, lo) and |e| up to 1; only the coeff
        nets get evaluated (no shift/log-det), so this is cheap relative to the
        surrounding ELBO/NLL computation.
        """
        k0, k1, k2, k3 = jr.split(key, 4)

        def _signed_probe(k):
            ks, kv = jr.split(k)
            sign = jnp.where(jr.bernoulli(ks), 1.0, -1.0)
            return sign * jr.uniform(kv, (), minval=lo, maxval=hi)

        x0 = jax.vmap(_signed_probe)(jr.split(k0, n))
        x1 = jax.vmap(_signed_probe)(jr.split(k1, n))
        e1 = jr.uniform(k2, (n,), minval=-1.0, maxval=1.0)
        e2 = jr.uniform(k3, (n,), minval=-1.0, maxval=1.0)

        if self.spin2_owne:
            def _sq(x0i, x1i, e1i, e2i):
                B1 = self.net_size(jnp.array([x0i]))
                inv = jnp.array([x0i, x1i, jnp.log1p(e1i * e1i + e2i * e2i)])
                a, b = self.net_m1(inv), self.net_m2(inv)
                return jnp.sum(B1**2) + jnp.sum(a**2) + jnp.sum(b**2)

            return jnp.mean(jax.vmap(_sq)(x0, x1, e1, e2))

        def _sq(x0i, x1i, e1i):
            A, B = self._coeffs(x0i, x1i, e1i)  # e1i doubles as the M1 "x2" probe
            return sum(jnp.sum(a**2) for a in A) + sum(jnp.sum(b**2) for b in B)

        return jnp.mean(jax.vmap(_sq)(x0, x1, e1))


# ---------------------------------------------------------------------------
# EarlyChain
# ---------------------------------------------------------------------------


class EarlyChain(AbstractBijection):
    """Bijection chain of unconditional ``early`` layers plus one conditional ``last``.

    ``transform`` maps data->base (the density direction, since the flow wraps this
    in :class:`~flowjax.bijections.Invert`).  The log-absolute-determinant is the
    sum of all constituent determinants.

    ``conditional_first`` selects where the conditional ``last`` layer sits:

    * ``False`` (default): ``early`` (unconditional) run first, ``last`` last, so the
      conditional layer is adjacent to the **base** Gaussian.  This is the historical
      order the w128 .eqx flows were trained with.
    * ``True``: ``last`` runs first, ``early`` after, so the conditional shear / Σ_X
      layer is adjacent to the **data** moments.  Then its coefficients ARE the
      data-space response (no bulk-Jacobian entanglement) — the physically-ordered
      generative stack ``base -> bulk -> Σ_X -> shear -> data``.

    ``conditional_first`` is a STATIC part of the flow structure, so a saved flow only
    deserialises with the value it was trained with; switching it requires retraining.

    Parameters
    ----------
    early : iterable of AbstractBijection
        Sequence of unconditional bijections.
    last : AbstractBijection
        The single conditional bijection.  Its ``cond_shape`` determines the
        ``cond_shape`` of the whole chain.
    conditional_first : bool, optional
        Place ``last`` at the data end (``True``) or the base end (``False``, default).
    """

    early: tuple
    last: AbstractBijection
    conditional_first: bool = eqx.field(static=True)

    def __init__(self, early, last, conditional_first=False):
        object.__setattr__(self, "early", tuple(early))
        object.__setattr__(self, "last", last)
        object.__setattr__(self, "conditional_first", bool(conditional_first))

    @property
    def shape(self):
        return self.last.shape

    @property
    def cond_shape(self):
        return self.last.cond_shape

    def transform_and_log_det(self, x, condition=None):
        lad_total = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        if self.conditional_first:
            x, lad = self.last.transform_and_log_det(x, condition)
            lad_total = lad_total + lad
        for b in self.early:
            x, lad = b.transform_and_log_det(x)
            lad_total = lad_total + lad
        if not self.conditional_first:
            x, lad = self.last.transform_and_log_det(x, condition)
            lad_total = lad_total + lad
        return x, lad_total

    def inverse_and_log_det(self, y, condition=None):
        lad_total = jnp.zeros(y.shape[:-1], dtype=y.dtype)
        if not self.conditional_first:
            y, lad = self.last.inverse_and_log_det(y, condition)
            lad_total = lad_total + lad
        for b in reversed(self.early):
            y, lad = b.inverse_and_log_det(y)
            lad_total = lad_total + lad
        if self.conditional_first:
            y, lad = self.last.inverse_and_log_det(y, condition)
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
    shear_layer_kind: str = "poly",
    conditional_first: bool = False,
    shear_own_e: bool = False,
    shear_split_ab: bool = False,
    shear_spin2_owne: bool = False,
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
    k_perm, k_shear = jr.split(k_last)

    if shear_layer_kind == "taylor":
        shear = ShearTaylorLast(
            k_shear,
            dim,
            last_layer_cond_dim,
            _last_width,
            _last_depth,
            nn_activation,
            own_e=shear_own_e,
            split_ab=shear_split_ab,
            spin2_owne=shear_spin2_owne,
        )
        # No equivariant permute for the Taylor layer: it relies on the fixed
        # spin-0/spin-2 component order (0,1 spin-0; 2,3 spin-2), which a 2<->3 swap
        # would scramble.
        shear_with_perm = shear
    elif shear_layer_kind == "poly":
        shear = ExplicitPolyLast(
            k_shear,
            dim,
            last_layer_cond_dim,
            quadratic_last,
            _last_width,
            _last_depth,
            nn_activation,
        )
        shear_with_perm = (
            Chain([_add_equivariant_permute(shear, k_perm)]).merge_chains()
            if dim > 1
            else shear
        )
    else:
        raise ValueError(
            f"unknown shear_layer_kind {shear_layer_kind!r} (want 'poly' or 'taylor')"
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
            e_max=sigmax_e_max,
            size_loc=sigmax_size_loc,
        )
        # NO spin-2-swapping permute between the two conditional layers: both
        # ExplicitPolyLast (shear g) and SigmaXCouplingLayer (centroid ellipticity
        # e) couple to EXTERNAL spin-2 vectors in the *data* frame, so their spin-2
        # inputs must stay aligned with (M1, M2).  _add_equivariant_permute swaps
        # 2↔3, which rotates the e-coupling 90° (off-diagonal response: e1→M2,
        # e2→M1) and makes the dipole/quadrupole terms unable to match the diagonal
        # target — the directional terms then never train.  Use `shear` directly.
        last = Chain([shear, sigmax]).merge_chains()
    else:
        last = shear_with_perm

    bijection = EarlyChain(early_layers, last, conditional_first=conditional_first)

    if invert:
        bijection = Invert(bijection)
    return Transformed(base_dist, bijection)
