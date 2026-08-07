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


class RawMomentStandardize(AbstractBijection):
    """
    Raw x = [Mf, Mr, M1, M2, Mc]   (bfd's even-moment order)
    Transformed z0 = [log10(Mf), Mr/Mf, Mc/Mr, M1/Mr, M2/Mr]
    Then standardized: z = (z0 - mean) / std

    The transformed order groups the three SPIN-0 coordinates first (0, 1, 2)
    and the spin-2 pair last (3, 4); the equivariant layers and the
    permutations between them rely on that split.  Mc/Mr carries the same units
    as Mr/Mf (both are k^2), so the two size-like coordinates share a scale.
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
            self.mean = jnp.zeros(5)
        else:
            self.mean = jnp.asarray(mean)
        if std is None:
            self.std = jnp.ones(5)
        else:
            self.std = jnp.asarray(std)

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
        z0_1 = Mr / Mf
        z0_2 = Mc / Mr
        z0_3 = M1 / Mr
        z0_4 = M2 / Mr

        z0 = jnp.stack([z0_0, z0_1, z0_2, z0_3, z0_4], axis=-1)
        z = (z0 - self.mean) / self.std
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
        z0 = z * self.std + self.mean
        log10_const = jnp.log(jnp.array(10.0, dtype=z0.dtype))
        Mf = jnp.exp(z0[..., 0] * log10_const)
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
        ln10 = jnp.log(jnp.array(10.0, dtype=Mf.dtype))
        lad_geom = -(2.0 * jnp.log(Mf) + 3.0 * jnp.log(Mr) + jnp.log(ln10))
        lad_std = -jnp.sum(jnp.log(self.std))
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

    * ``z[0] = x[0] * exp(-s0)`` — unconditional scale.
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

    log_scale0: jax.Array
    net_ls1: CoeffNet
    net_ls2: CoeffNet

    def __init__(self, key, nn_width, nn_depth, activation):
        k1, k2 = jr.split(key, 2)
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
        z0 = x[0] * jnp.exp(-ls0)
        loc1, ls1 = self._loc_scale(self.net_ls1, [z0])
        z1 = (x[1] - loc1) * jnp.exp(-ls1)
        loc2, ls2 = self._loc_scale(self.net_ls2, [z0, z1])
        z2 = (x[2] - loc2) * jnp.exp(-ls2)
        y = x.at[0].set(z0).at[1].set(z1).at[2].set(z2)
        return y, -(ls0 + ls1 + ls2)

    def inverse_and_log_det(self, y, condition=None):
        ls0 = _bounded_log_scale(self.log_scale0)
        x0 = y[0] * jnp.exp(ls0)
        loc1, ls1 = self._loc_scale(self.net_ls1, [y[0]])
        x1 = y[1] * jnp.exp(ls1) + loc1
        loc2, ls2 = self._loc_scale(self.net_ls2, [y[0], y[1]])
        x2 = y[2] * jnp.exp(ls2) + loc2
        x = y.at[0].set(x0).at[1].set(x1).at[2].set(x2)
        return x, ls0 + ls1 + ls2


# ---------------------------------------------------------------------------
# Spin2CouplingLayer
# ---------------------------------------------------------------------------


class Spin2CouplingLayer(AbstractBijection):
    """Coupling bijection on the spin-2 (shape) components conditioned on spin-0.

    Transforms components 3 and 4 with a shared affine scale conditioned
    on the already-transformed spin-0 components (0, 1, 2).  The single shared
    scale is what preserves spin-2 equivariance: anything treating the two
    components differently would pick out a direction on the sky.

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
        self.net = CoeffNet(key, 3, 2, nn_width, nn_depth, activation)

    @property
    def shape(self):
        return (5,)

    @property
    def cond_shape(self):
        return None

    def _coeffs(self, z0_t, z1_t, z2_t):
        out = self.net(jnp.array([z0_t, z1_t, z2_t]))
        log_scale = _bounded_log_scale(out[0])
        c = 0.9 * jnn.tanh(out[1])
        log_one_minus_c = jnp.log1p(-c)
        return log_scale, c, log_one_minus_c

    def transform_and_log_det(self, x, condition=None):
        log_scale, c, log_one_minus_c = self._coeffs(x[0], x[1], x[2])
        log_det_each = log_one_minus_c - log_scale
        y3 = x[3] * jnp.exp(log_det_each)
        y4 = x[4] * jnp.exp(log_det_each)
        y = x.at[3].set(y3).at[4].set(y4)
        return y, log_det_each + log_det_each

    def inverse_and_log_det(self, y, condition=None):
        log_scale, c, log_one_minus_c = self._coeffs(y[0], y[1], y[2])
        log_det_each = log_one_minus_c - log_scale
        x3 = y[3] * jnp.exp(-log_det_each)
        x4 = y[4] * jnp.exp(-log_det_each)
        x = y.at[3].set(x3).at[4].set(x4)
        return x, -(log_det_each + log_det_each)


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

    def __init__(self, key, nn_width, nn_depth, activation):
        k0, k2 = jr.split(key)
        self.spin0 = Spin0AutoregressiveLayer(k0, nn_width, nn_depth, activation)
        self.spin2 = Spin2CouplingLayer(k2, nn_width, nn_depth, activation)

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


# ---------------------------------------------------------------------------
# SigmaXCouplingLayer
# ---------------------------------------------------------------------------


class SigmaXCouplingLayer(AbstractBijection):
    """PSF noise covariance (Σ_X) conditioned coupling bijection.

    Provides an additional bijection layer conditioned on the PSF noise
    covariance parameters ``[g1, g2, log_scale, e1, e2]`` where
    ``log_scale = 0.5 * log det(Σ_X)``.  This allows the flow to learn
    how galaxy shape probabilities vary with PSF ellipticity and size.

    The transform is factored into:

    * **Spin-0 autoregressive**: components 0–1 scaled by networks
      conditioned on ``log_scale``.
    * **Spin-2 coupling**: components 2–3 transformed by the linear map
      ``A = s·I + c·E`` where ``E = [[e1, e2], [e2, -e1]]``, ensuring
      first-order sensitivity to PSF ellipticity.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    nn_width : int, optional
        Hidden width.  Default is 32.
    nn_depth : int, optional
        Number of hidden layers.  Default is 2.
    activation : callable, optional
        Activation function.  Default is ``jax.nn.silu``.
    full_cond_dim : int, optional
        Total conditioning dimension (must be 5 for
        ``[g1, g2, log_scale, e1, e2]``).  Default is 5.
    log_scale_mean : float, optional
        Prior mean of ``log_scale`` for normalisation.  Default is 12.
    log_scale_std : float, optional
        Prior std of ``log_scale`` for normalisation.  Default is 3.
    e_max : float, optional
        Maximum PSF ellipticity magnitude used to normalise ``e_mag²``.
        Default is 0.1.
    """

    net_s0: CoeffNet  # (log_scale_n,)                          → (1,)
    net_s1: CoeffNet  # (x0, log_scale_n)                       → (1,)
    net_avg: CoeffNet  # (x0, x1, log_scale_n, e_mag_sq_n)      → (1,)
    net_c: CoeffNet  # same inputs                             → (1,)  coupling strength
    _cond_dim: int = eqx.field(static=True)
    _log_scale_mean: float = eqx.field(static=True)
    _log_scale_std: float = eqx.field(static=True)
    _e_mag_sq_scale: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        nn_width=32,
        nn_depth=2,
        activation=jnn.silu,
        full_cond_dim=5,
        log_scale_mean=12.0,
        log_scale_std=3.0,
        e_max=0.1,
    ):
        self._cond_dim = full_cond_dim
        self._log_scale_mean = float(log_scale_mean)
        self._log_scale_std = float(log_scale_std)
        self._e_mag_sq_scale = float(e_max**2)
        k0, k1, k2, k3 = jr.split(key, 4)
        self.net_s0 = CoeffNet(k0, 1, 1, nn_width, nn_depth, activation)
        self.net_s1 = CoeffNet(k1, 2, 1, nn_width, nn_depth, activation)
        self.net_avg = CoeffNet(k2, 4, 1, nn_width, nn_depth, activation)
        self.net_c = CoeffNet(k3, 4, 1, nn_width, nn_depth, activation)

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
        log_scale_n = (log_scale - self._log_scale_mean) / (self._log_scale_std + 1e-8)
        e_mag_sq_n = e_mag_sq / (self._e_mag_sq_scale + 1e-8)
        return log_scale_n, e1, e2, e_mag_sq, e_mag_sq_n

    def _spin2_coeffs(self, x0, x1, log_scale_n, e_mag_sq, e_mag_sq_n):
        s2_in = jnp.array([x0, x1, log_scale_n, e_mag_sq_n])

        s = jnp.exp(-_bounded_log_scale(self.net_avg(s2_in)[0]))  # > 0

        # FIX: c = tanh * s  (first-order in ellipticity, not second-order).
        # Previously c = tanh * s * e_mag, which made the directional asymmetry
        # O(e^2) — ~0.8% for e_mag=0.09 and invisible in any plot.
        #
        # With c = tanh * s:
        #   A = s*I + c*E,  E = [[e1,e2],[e2,-e1]]
        #   det(A) = s^2 - c^2 * e_mag^2
        #          = s^2 * (1 - tanh^2 * e_mag^2)
        #          >= s^2 * (1 - e_max^2)  > 0  for e_max < 1
        # Bijectivity is preserved; directional asymmetry is now first-order.
        c = jnn.tanh(self.net_c(s2_in)[0]) * s

        return s, c

    def _spin0_coeffs(self, x0, log_scale_n):
        ls0 = _bounded_log_scale(self.net_s0(jnp.array([log_scale_n]))[0])
        ls1 = _bounded_log_scale(self.net_s1(jnp.array([x0, log_scale_n]))[0])
        return ls0, ls1

    def transform_and_log_det(self, x, condition=None):
        log_scale_n, e1, e2, e_mag_sq, e_mag_sq_n = self._unpack(condition)

        # spin-0 (autoregressive)
        ls0, ls1 = self._spin0_coeffs(x[0], log_scale_n)
        y0 = x[0] * jnp.exp(-ls0)
        y1 = x[1] * jnp.exp(-ls1)

        # spin-2 coupling — A = s*I + c*E,  E = [[e1,e2],[e2,-e1]]
        s, c = self._spin2_coeffs(x[0], x[1], log_scale_n, e_mag_sq, e_mag_sq_n)
        y2 = s * x[2] + c * (e1 * x[2] + e2 * x[3])
        y3 = s * x[3] + c * (e2 * x[2] - e1 * x[3])

        # det(A) = s^2 - c^2 * |e|^2,  guaranteed > 0 by construction
        log_det_spin2 = jnp.log(s**2 - c**2 * e_mag_sq)

        return jnp.stack([y0, y1, y2, y3]), -(ls0 + ls1) + log_det_spin2

    def inverse_and_log_det(self, y, condition=None):
        log_scale_n, e1, e2, e_mag_sq, e_mag_sq_n = self._unpack(condition)

        # spin-0: exact autoregressive inverse
        ls0 = _bounded_log_scale(self.net_s0(jnp.array([log_scale_n]))[0])
        x0 = y[0] * jnp.exp(ls0)
        ls1 = _bounded_log_scale(self.net_s1(jnp.array([x0, log_scale_n]))[0])
        x1 = y[1] * jnp.exp(ls1)

        # spin-2: A^{-1} = (s*I - c*E) / det(A)
        s, c = self._spin2_coeffs(x0, x1, log_scale_n, e_mag_sq, e_mag_sq_n)
        det = s**2 - c**2 * e_mag_sq  # > 0 by construction
        x2 = (s * y[2] - c * (e1 * y[2] + e2 * y[3])) / det
        x3 = (s * y[3] - c * (e2 * y[2] - e1 * y[3])) / det

        log_det_spin2 = jnp.log(det)

        return jnp.stack([x0, x1, x2, x3]), ls0 + ls1 - log_det_spin2


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
    sigmax_log_scale_mean: float = 2.0 * 5.991464547107982,  # 2*log(400)
    sigmax_log_scale_std: float = 1.0,
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
