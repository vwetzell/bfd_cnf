"""
convergence_metrics.py
======================
Per-stage convergence diagnostics for the prior normalizing flow.

The prior flow is a composition of three trainable stages, each carrying a
*disjoint* slice of the model's behaviour:

  1. ``EquivariantAutoregressiveLayer``  (×9, unconditional) — the base galaxy
     moment *shape* ``p(z)`` at the reference condition.
  2. ``ExplicitPolyLast``                (conditional on ``[g1, g2]``) — the
     entire **shear** response of the prior.
  3. ``SigmaXCouplingLayer``             (conditional on ``[log_scale, e1, e2]``)
     — the entire **centroid-covariance** ``C_X`` response (flux/size monopole
     + ellipticity dipole/quadrupole).

Because the prior's conditioning vector ``[g1, g2, log_scale, e1, e2]`` is read
by exactly one conditional stage per slot (``g`` → ExplicitPolyLast,
``log_scale/e`` → SigmaXCouplingLayer) and the early layers are unconditional,
the gradient ``∇_cond log p(z | cond)`` evaluated at a fixed reference condition
*cleanly partitions* into a per-stage functional signature:

  * ``∂ log p / ∂[g1, g2]``           → ExplicitPolyLast signature.
  * ``∂ log p / ∂[log_scale, e1, e2]`` → SigmaXCouplingLayer signature.
  * ``log p`` itself                  → base-shape (early-layer) signature.

This module provides two complementary convergence views per stage:

  * **Parameter plateau** — relative L2 change of the stage's flattened
    trainable parameter vector between consecutive checkpoints.  This is the
    cleanly *isolated* "have the parameters stopped moving" signal.
  * **Functional plateau** — relative change of the stage's functional
    signature, binned across moment space (the standardised flux ``z0`` and
    size ``z1`` axes).  This captures whether the stage's *behaviour* has
    stopped moving, which can plateau before/after the raw parameters do.

A stage is considered converged when *both* its parameter and functional
relative deltas fall below their thresholds.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .models.bijections import (
    EquivariantAutoregressiveLayer,
    ExplicitPolyLast,
    SigmaXCouplingLayer,
)

# Stage identifiers (order = report order).
STAGE_EQUIV = "EquivariantAutoregressiveLayer"
STAGE_POLY = "ExplicitPolyLast"
STAGE_SIGMAX = "SigmaXCouplingLayer"
STAGE_ORDER = (STAGE_EQUIV, STAGE_POLY, STAGE_SIGMAX)

_STAGE_TYPES = {
    STAGE_EQUIV: EquivariantAutoregressiveLayer,
    STAGE_POLY: ExplicitPolyLast,
    STAGE_SIGMAX: SigmaXCouplingLayer,
}

# Condition-vector layout: [g1, g2, log_scale, e1, e2].  Which columns each
# conditional stage reads (and therefore which gradient slice is its signature).
_POLY_COND_COLS = (0, 1)          # ExplicitPolyLast reads g
_SIGMAX_COND_COLS = (2, 3, 4)     # SigmaXCouplingLayer reads [log_scale, e1, e2]


# ---------------------------------------------------------------------------
# Parameter-plateau metrics
# ---------------------------------------------------------------------------


def collect_stage_params(prior: Any) -> dict[str, jax.Array]:
    """Return the flattened trainable parameter vector of each prior stage.

    Walks the prior pytree and, for every instance of a target stage class,
    harvests its inexact-array leaves (concatenated, leaf order is the stable
    pytree-traversal order so vectors align across checkpoints of the same
    architecture).

    Parameters
    ----------
    prior : Transformed
        The prior normalizing flow.

    Returns
    -------
    dict[str, jax.Array]
        Mapping ``stage_name -> flat_param_vector`` for the three stages.
    """
    types = tuple(_STAGE_TYPES.values())
    buckets: dict[str, list[jax.Array]] = {name: [] for name in STAGE_ORDER}
    name_of = {t: name for name, t in _STAGE_TYPES.items()}

    def rec(o: Any) -> None:
        for t in types:
            if isinstance(o, t):
                leaves = jax.tree_util.tree_leaves(
                    eqx.filter(o, eqx.is_inexact_array)
                )
                buckets[name_of[t]].extend(jnp.ravel(leaf) for leaf in leaves)
                return  # do not descend further into a matched stage
        if isinstance(o, eqx.Module):
            for f in dataclasses.fields(o):
                rec(getattr(o, f.name))
        elif isinstance(o, (list, tuple)):
            for x in o:
                rec(x)
        elif isinstance(o, dict):
            for x in o.values():
                rec(x)

    rec(prior)
    out: dict[str, jax.Array] = {}
    for name in STAGE_ORDER:
        parts = buckets[name]
        out[name] = (
            jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=jnp.float32)
        )
    return out


def param_plateau_metrics(
    cur: dict[str, jax.Array], prev: dict[str, jax.Array] | None
) -> dict[str, dict[str, float]]:
    """Compute per-stage parameter norms and relative change from the previous checkpoint.

    Parameters
    ----------
    cur : dict[str, jax.Array]
        Current flattened stage parameter vectors (from :func:`collect_stage_params`).
    prev : dict[str, jax.Array] or None
        Previous checkpoint's vectors, or ``None`` for the baseline block.

    Returns
    -------
    dict[str, dict[str, float]]
        ``stage -> {param_norm, abs_delta, rel_delta}``.  ``abs_delta`` /
        ``rel_delta`` are ``nan`` for the baseline block.
    """
    out: dict[str, dict[str, float]] = {}
    for name in STAGE_ORDER:
        v = np.asarray(cur[name])
        norm = float(np.linalg.norm(v))
        if prev is None:
            abs_d = float("nan")
            rel_d = float("nan")
        else:
            p = np.asarray(prev[name])
            abs_d = float(np.linalg.norm(v - p))
            rel_d = abs_d / max(float(np.linalg.norm(p)), 1e-12)
        out[name] = {"param_norm": norm, "abs_delta": abs_d, "rel_delta": rel_d}
    return out


# ---------------------------------------------------------------------------
# Functional signatures across moment space
# ---------------------------------------------------------------------------


def _logprob_and_cond_grad(prior: Any, z: jax.Array, cond: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-point ``log p`` and its gradient w.r.t. the conditioning vector.

    Parameters
    ----------
    prior : Transformed
        The prior flow.
    z : jax.Array, shape (4,)
        A single standardised moment vector.
    cond : jax.Array, shape (5,)
        Conditioning vector ``[g1, g2, log_scale, e1, e2]``.

    Returns
    -------
    logp : jax.Array, scalar
    grad_cond : jax.Array, shape (5,)
    """
    def lp(c: jax.Array) -> jax.Array:
        return prior.log_prob(z, condition=c)

    logp, grad_cond = jax.value_and_grad(lp)(cond)
    return logp, grad_cond


def functional_signatures(
    prior: Any,
    z_val: jax.Array,
    c_ref: jax.Array,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the per-stage functional signatures on a validation set.

    Parameters
    ----------
    prior : Transformed
        The prior flow.
    z_val : jax.Array, shape (Nval, 4)
        Fixed validation set of standardised moments.
    c_ref : jax.Array, shape (5,)
        Reference conditioning vector ``[0, 0, log_scale_ref, 0, 0]``.
    chunk : int, optional
        Rows evaluated per device call (bounds memory).  Default 4096.

    Returns
    -------
    logp : np.ndarray, shape (Nval,)
        Reference-condition log-density (base-shape signature).
    grad_cond : np.ndarray, shape (Nval, 5)
        ``∂ log p / ∂cond`` per point.  Columns 0:2 are the ExplicitPolyLast
        (shear) signature; columns 2:5 are the SigmaXCouplingLayer (C_X)
        signature.
    """
    f = eqx.filter_jit(
        jax.vmap(lambda z, c: _logprob_and_cond_grad(prior, z, c), in_axes=(0, 0))
    )
    n = z_val.shape[0]
    cond_full = jnp.broadcast_to(c_ref[None, :], (n, c_ref.shape[0]))
    logps, grads = [], []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        lp, gc = f(z_val[start:end], cond_full[start:end])
        logps.append(np.asarray(lp))
        grads.append(np.asarray(gc))
    return np.concatenate(logps, axis=0), np.concatenate(grads, axis=0)


# ---------------------------------------------------------------------------
# Moment-space binning
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MomentSpaceBins:
    """Fixed 2-D binning of moment space by standardised flux (z0) and size (z1)."""

    bin_idx: np.ndarray          # (Nval,) flattened bin index in [0, nbins)
    counts: np.ndarray           # (nbins,) points per bin
    nx: int
    ny: int
    x_edges: np.ndarray
    y_edges: np.ndarray
    min_count: int = 20          # bins with fewer points are masked from rel-deltas

    @property
    def nbins(self) -> int:
        return self.nx * self.ny

    @property
    def valid_mask(self) -> np.ndarray:
        return self.counts >= self.min_count


def make_moment_space_bins(
    z_val: jax.Array, nx: int = 8, ny: int = 8, q_lo: float = 1.0, q_hi: float = 99.0
) -> MomentSpaceBins:
    """Build a fixed (z0, z1) 2-D binning from the validation set quantiles.

    Parameters
    ----------
    z_val : jax.Array, shape (Nval, 4)
        Validation standardised moments.
    nx, ny : int, optional
        Number of flux / size bins.  Default 8 each.
    q_lo, q_hi : float, optional
        Percentile edges of the binned range (outliers clipped into edge bins).

    Returns
    -------
    MomentSpaceBins
    """
    z = np.asarray(z_val)
    x, y = z[:, 0], z[:, 1]
    x_edges = np.linspace(*np.percentile(x, [q_lo, q_hi]), nx + 1)
    y_edges = np.linspace(*np.percentile(y, [q_lo, q_hi]), ny + 1)
    ix = np.clip(np.digitize(x, x_edges[1:-1]), 0, nx - 1)
    iy = np.clip(np.digitize(y, y_edges[1:-1]), 0, ny - 1)
    bin_idx = ix * ny + iy
    counts = np.bincount(bin_idx, minlength=nx * ny)
    return MomentSpaceBins(bin_idx, counts, nx, ny, x_edges, y_edges)


def bin_means(values: np.ndarray, bins: MomentSpaceBins) -> np.ndarray:
    """Average ``values`` within each moment-space bin.

    Parameters
    ----------
    values : np.ndarray, shape (Nval,) or (Nval, C)
        Per-point scalar or vector field.
    bins : MomentSpaceBins

    Returns
    -------
    np.ndarray, shape (nbins,) or (nbins, C)
        Per-bin mean (``nan`` for empty bins).
    """
    values = np.asarray(values)
    nb = bins.nbins
    if values.ndim == 1:
        sums = np.bincount(bins.bin_idx, weights=values, minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            field = sums / bins.counts
        return field
    C = values.shape[1]
    out = np.full((nb, C), np.nan)
    for c in range(C):
        sums = np.bincount(bins.bin_idx, weights=values[:, c], minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, c] = sums / bins.counts
    return out


def _field_rel_delta(cur: np.ndarray, prev: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Relative + absolute L2 change of a binned field over valid bins."""
    c = np.asarray(cur)
    p = np.asarray(prev)
    if c.ndim == 1:
        m = mask
    else:
        m = mask[:, None] * np.ones((1, c.shape[1]), dtype=bool)
    cm = c[m]
    pm = p[m]
    good = np.isfinite(cm) & np.isfinite(pm)
    cm, pm = cm[good], pm[good]
    abs_d = float(np.linalg.norm(cm - pm))
    rel_d = abs_d / max(float(np.linalg.norm(pm)), 1e-12)
    return rel_d, abs_d


# ---------------------------------------------------------------------------
# Block-level metric assembly
# ---------------------------------------------------------------------------


def compute_functional_fields(
    prior: Any, z_val: jax.Array, bins: MomentSpaceBins, c_ref: jax.Array, chunk: int = 4096
) -> dict[str, Any]:
    """Compute the three per-stage functional fields + scalar summaries.

    Returns a dict with, per stage:
      * ``field``      — binned signature, shape (nbins,) or (nbins, C).
      * ``mean_mag``   — mean magnitude of the per-point signature (scalar).
    Stage 1 (base shape) uses ``log p``; stages 2/3 use the relevant gradient
    slice (``|·|`` taken over channels for the magnitude summary).
    """
    logp, grad_cond = functional_signatures(prior, z_val, c_ref, chunk=chunk)

    g_sig = grad_cond[:, list(_POLY_COND_COLS)]      # (Nval, 2)
    cx_sig = grad_cond[:, list(_SIGMAX_COND_COLS)]   # (Nval, 3)

    fields = {
        STAGE_EQUIV: {
            "field": bin_means(logp, bins),
            "mean_mag": float(np.mean(logp)),
        },
        STAGE_POLY: {
            "field": bin_means(g_sig, bins),
            "mean_mag": float(np.mean(np.linalg.norm(g_sig, axis=1))),
        },
        STAGE_SIGMAX: {
            "field": bin_means(cx_sig, bins),
            "mean_mag": float(np.mean(np.linalg.norm(cx_sig, axis=1))),
        },
    }
    # Keep raw per-point arrays for optional plotting at the final block.
    fields["_raw"] = {"logp": logp, "g_sig": g_sig, "cx_sig": cx_sig}
    return fields


def functional_plateau_metrics(
    cur_fields: dict[str, Any], prev_fields: dict[str, Any] | None, bins: MomentSpaceBins
) -> dict[str, dict[str, float]]:
    """Per-stage functional relative change between consecutive blocks."""
    out: dict[str, dict[str, float]] = {}
    mask = bins.valid_mask
    for name in STAGE_ORDER:
        mean_mag = cur_fields[name]["mean_mag"]
        if prev_fields is None:
            rel_d, abs_d = float("nan"), float("nan")
        else:
            rel_d, abs_d = _field_rel_delta(
                cur_fields[name]["field"], prev_fields[name]["field"], mask
            )
        out[name] = {"mean_mag": mean_mag, "rel_delta": rel_d, "abs_delta": abs_d}
    return out


def stage_converged(
    param_m: dict[str, dict[str, float]],
    func_m: dict[str, dict[str, float]],
    tau_param: float,
    tau_func: float,
) -> dict[str, bool]:
    """Per-stage boolean: both param and functional rel-deltas below threshold."""
    out: dict[str, bool] = {}
    for name in STAGE_ORDER:
        pr = param_m[name]["rel_delta"]
        fr = func_m[name]["rel_delta"]
        out[name] = (
            np.isfinite(pr) and np.isfinite(fr) and pr < tau_param and fr < tau_func
        )
    return out
