"""
training.py
===========
ELBO loss setup, model training via ``fit_to_key_based_loss``, and
model serialisation / deserialisation utilities.

The main entry point is ``load_or_train()``, which either loads pre-trained
weights from disk or runs training from scratch.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from tqdm import tqdm

from .config import (
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    batch_size,
    c_idx,
    c_off,
    g_scale,
    max_scale,
    min_scale,
    num_samples,
    prior_early_nn_depth,
    prior_early_nn_width,
    prior_flow_layers,
    prior_last_nn_depth,
    prior_last_nn_width,
    prior_sigmax_log_scale_mean,
    prior_sigmax_log_scale_std,
    prior_sigmax_nn_depth,
    prior_sigmax_nn_width,
    q_flow_layers,
    q_nn_depth,
    q_nn_width,
    r_idx,
    r_off,
    train_chunk_size,
    use_nda_weight,
)
from .config import (
    e_max as _e_max,
)
from .config import (
    log_scale_range as _log_scale_range,
)
from .data import transform_dataset_to_standard
from .models.flows import (
    batch_cholesky_of_sym,
    build_flows,
    cov2corr,
    make_elbo_loss,
    make_nll_loss,
)

# ---------------------------------------------------------------------------
# Standardisation statistics helper
# ---------------------------------------------------------------------------


def compute_std_stats(
    raw2standard: Any,
    moments_jnp: jax.Array,
    cov_jnp: jax.Array,
    chunk_size: int = 50_000,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute normalisation statistics for the Cholesky conditioning features.

    Transforms raw moments and covariances into standardised coordinates,
    then computes the mean and standard deviation of the log-diagonal
    elements of the Cholesky factor and of the off-diagonal correlation
    coefficients.  These statistics are used to whiten the conditioning
    features passed to the flow networks.

    Processed in chunks to avoid materialising the full (N, 4, 4, 4, 4)
    Hessian intermediate on the GPU at once.

    Parameters
    ----------
    raw2standard : RawMomentStandardize
        Bijection from raw to standardised moment coordinates.
    moments_jnp : jax.Array, shape (N, 4)
        Raw template moments.
    cov_jnp : jax.Array, shape (N, 4, 4)
        Per-object raw-moment covariance matrices.
    chunk_size : int, optional
        Number of rows processed per GPU call.  Default is 50 000.

    Returns
    -------
    mean_log_diag : jax.Array, shape (4,)
        Mean of ``log(diag(L))`` over the training set.
    std_log_diag : jax.Array, shape (4,)
        Standard deviation of ``log(diag(L))``.
    mean_off : jax.Array, shape (6,)
        Mean of the off-diagonal correlation elements.
    std_off : jax.Array, shape (6,)
        Standard deviation of the off-diagonal correlation elements.
    """
    n = moments_jnp.shape[0]
    log_diag_chunks, off_chunks = [], []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        _, Sigma_chunk = transform_dataset_to_standard(
            raw2standard, moments_jnp[start:end], cov_jnp[start:end]
        )
        L_chunk = batch_cholesky_of_sym(Sigma_chunk)
        diag_chunk = jnp.diagonal(L_chunk, axis1=-2, axis2=-1)
        log_diag_chunks.append(jnp.log(diag_chunk + 1e-12))
        corr_chunk = cov2corr(Sigma_chunk)
        off_chunks.append(corr_chunk[..., r_off, c_off])

    log_diag_all = jnp.concatenate(log_diag_chunks, axis=0)
    off_all = jnp.concatenate(off_chunks, axis=0)

    mean_log_diag = jnp.mean(log_diag_all, axis=0)
    std_log_diag = jnp.std(log_diag_all, axis=0)
    mean_off = jnp.mean(off_all, axis=0)
    std_off = jnp.std(off_all, axis=0)

    return mean_log_diag, std_log_diag, mean_off, std_off


# ---------------------------------------------------------------------------
# Training loop (shared by train_model / continue_training)
# ---------------------------------------------------------------------------


def _make_train_step(elbo: Any, optimizer: Any) -> Any:
    """Build the eager single-step update used by the per-step fallback loop."""

    @eqx.filter_jit
    def train_step(
        model, data_y, data_Sigma, data_dg, data_d2g, data_X, opt_state, key
    ):
        loss_val, grads = eqx.filter_value_and_grad(
            lambda m: elbo(m, data_y, data_Sigma, data_dg, data_d2g, data_X, key)
        )(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(model, updates), new_opt_state, loss_val

    return train_step


def _make_chunk_runner(elbo: Any, optimizer: Any, chunk_len: int) -> Any:
    """Build a function that runs ``chunk_len`` update steps in one ``lax.scan``.

    The whole chunk lowers to a single XLA dispatch, so the GPU runs the steps
    back-to-back with no per-step host round-trip (the source of the per-step
    stall).  The large data arrays are passed as dynamic arguments — never
    closed over — so they are not embedded as compile-time constants.

    The equinox model is partitioned into array leaves + static structure so it
    can be carried through the scan (``eqx.partition`` / ``eqx.combine``).
    ``opt_state`` is a plain JAX pytree (optax NamedTuples) and is passed
    directly in the carry — routing it through ``eqx.partition`` causes equinox
    to reconstruct optax NamedTuples as plain tuples, breaking ``state.count``
    lookups when a schedule function is used (e.g. cosine decay).
    """

    @eqx.filter_jit
    def run_chunk(
        model, opt_state, key, data_y, data_Sigma, data_dg, data_d2g, data_X
    ):
        model_arrays, model_static = eqx.partition(model, eqx.is_array)

        def body(carry, _):
            model_arrays, opt_state, key = carry
            model = eqx.combine(model_arrays, model_static)
            key, subkey = jr.split(key)
            loss_val, grads = eqx.filter_value_and_grad(
                lambda m: elbo(m, data_y, data_Sigma, data_dg, data_d2g, data_X, subkey)
            )(model)
            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
            )
            model = eqx.apply_updates(model, updates)
            model_arrays, _ = eqx.partition(model, eqx.is_array)
            return (model_arrays, opt_state, key), loss_val

        (model_arrays, opt_state, key), losses = jax.lax.scan(
            body, (model_arrays, opt_state, key), xs=None, length=chunk_len
        )
        model = eqx.combine(model_arrays, model_static)
        return model, opt_state, key, losses

    return run_chunk


def _run_training_loop(
    model_tuple: Any,
    opt_state: Any,
    optimizer: Any,
    elbo: Any,
    data: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    key: jax.Array,
    steps: int,
    chunk_size: int,
    runners: dict[int, Any] | None = None,
) -> tuple[Any, Any, list[float]]:
    """Run ``steps`` ELBO updates, fusing ``chunk_size`` steps per dispatch.

    ``data`` is ``(moments, cov, dm_dg, d2m_dg2, centroid)``.  With
    ``chunk_size <= 1`` this is the original eager per-step loop (one
    device→host sync per step); otherwise steps are fused via
    :func:`_make_chunk_runner`, with one host sync per chunk.  RNG threading is
    identical in both paths, so results match bit-for-bit at ``chunk_size=1``.

    Returns ``(model_tuple, opt_state, losses)``.
    """
    moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp, centroid_moments_jnp = data
    losses: list[float] = []

    if chunk_size <= 1:
        train_step = _make_train_step(elbo, optimizer)
        pbar = tqdm(range(steps))
        for _ in pbar:
            key, subkey = jr.split(key)
            model_tuple, opt_state, loss_val = train_step(
                model_tuple, moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp,
                centroid_moments_jnp, opt_state, subkey,
            )
            loss_f = float(loss_val)
            losses.append(loss_f)
            pbar.set_postfix(loss=f"{loss_f:.4f}")
        return model_tuple, opt_state, losses

    # Cache one compiled runner per distinct chunk length (full + final remainder).
    # A caller-supplied dict is REUSED across blocks so the jitted runner (and its
    # compiled XLA executable) persists — rebuilding it per block leaks GPU memory
    # (each fresh eqx.filter_jit compiles a new executable that is never freed).
    if runners is None:
        runners = {}
    pbar = tqdm(total=steps)
    done = 0
    while done < steps:
        n = min(chunk_size, steps - done)
        if n not in runners:
            runners[n] = _make_chunk_runner(elbo, optimizer, n)
        model_tuple, opt_state, key, chunk_losses = runners[n](
            model_tuple, opt_state, key,
            moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp, centroid_moments_jnp,
        )
        chunk_losses = np.asarray(chunk_losses)  # single host sync for the chunk
        losses.extend(chunk_losses.tolist())
        done += n
        pbar.update(n)
        pbar.set_postfix(loss=f"{chunk_losses[-1]:.4f}")
    pbar.close()
    return model_tuple, opt_state, losses


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    key: jax.Array,
    moments_jnp: jax.Array,
    centroid_moments_jnp: jax.Array,
    cov_jnp: jax.Array,
    dm_dg_jnp: jax.Array,
    d2m_dg2_jnp: jax.Array,
    weights: jax.Array,
    raw2standard: Any,
    *,
    steps: int = 20_000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 0.5,
    log_scale_range: tuple[float, float] = _log_scale_range,
    e_max: float = _e_max,
    chunk_size: int = train_chunk_size,
    nda: jax.Array | None = None,
    use_nda_weight: bool = use_nda_weight,
) -> tuple[Any, Any, list[float]]:
    """Build flows, construct the ELBO loss, and run training.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    moments_jnp : jax.Array, shape (N, 4)
        Raw template moments.
    centroid_moments_jnp : jax.Array, shape (N, 2)
        First-order Fourier moments (centroid moments) for each template.
        Passed as ``data_X`` to ``make_elbo_loss`` and used to compute the
        centroid likelihood weight ``log N(X_G; 0, C_X)`` during training.
    cov_jnp : jax.Array, shape (N, 4, 4)
        Per-object raw-moment covariance matrices.
    dm_dg_jnp : jax.Array, shape (N, 4, 2)
        First shear derivatives of the moments.
    d2m_dg2_jnp : jax.Array, shape (N, 4, 2, 2)
        Second shear derivatives of the moments.
    weights : jax.Array, shape (N,)
        Importance weights for training.
    raw2standard : RawMomentStandardize
        Bijection from raw to standardised moment coordinates.
    steps : int, optional
        Number of gradient steps.  Default is 20 000.
    learning_rate : float, optional
        AdamW learning rate.  Default is ``1e-4``.
    weight_decay : float, optional
        AdamW weight decay.  Default is ``1e-5``.
    grad_clip : float, optional
        Global gradient-norm clip.  Default is 0.5.
    log_scale_range : tuple of float, optional
        ``(min, max)`` of ``0.5 * log det(Σ_X)`` for Σ_X conditioning.
    e_max : float, optional
        Maximum ellipticity magnitude for Σ_X conditioning.
    chunk_size : int, optional
        Number of gradient steps fused into one ``jax.lax.scan`` dispatch
        (≈1.7x faster, identical math).  Default from ``config.train_chunk_size``.
        Set to 1 for the eager per-step loop (e.g. under ``jax_debug_nans``).
    nda : jax.Array or None, optional
        Per-template BFD ``nda = sky_density × da`` weight (the FITS ``weight``
        column).  When ``use_nda_weight`` is True, each template's loss
        contribution is scaled by ``nda`` so the flow learns the *nda-weighted*
        template prior, matching the traditional BFD integration.  ``None``
        (default) leaves the loss unweighted.
    use_nda_weight : bool, optional
        Master switch for the ``nda`` weighting.  Defaults to
        ``config.use_nda_weight``.

    Returns
    -------
    prior_trained : Transformed
        Trained prior normalizing flow.
    q_trained : Transformed
        Trained variational (q) normalizing flow.
    losses : list of float
        Training loss values recorded by ``fit_to_key_based_loss``.
    """
    # NOTE: jax_debug_nans is intentionally NOT forced on here — it makes every step
    # ~10-50x slower, which is prohibitive for a 20k-step run.  The debugging entry
    # point ``train_from_scratch.py`` enables it explicitly when localising a NaN.

    mean_log_diag, std_log_diag, mean_off, std_off = compute_std_stats(
        raw2standard, moments_jnp, cov_jnp
    )

    key, sub = jr.split(key)
    prior_flow, q_flow = build_flows(
        sub,
        latent_dim=4,
        cond_dim=16,
        raw2standard=raw2standard,
        prior_flow_layers=prior_flow_layers,
        prior_early_nn_width=prior_early_nn_width,
        prior_early_nn_depth=prior_early_nn_depth,
        prior_last_nn_width=prior_last_nn_width,
        prior_last_nn_depth=prior_last_nn_depth,
        prior_sigmax_nn_width=prior_sigmax_nn_width,
        prior_sigmax_nn_depth=prior_sigmax_nn_depth,
        prior_sigmax_log_scale_mean=prior_sigmax_log_scale_mean,
        prior_sigmax_log_scale_std=prior_sigmax_log_scale_std,
        q_flow_layers=q_flow_layers,
        q_nn_width=q_nn_width,
        q_nn_depth=q_nn_depth,
        min_scale=min_scale,
        max_scale=max_scale,
    )

    weights_np = np.array(weights)
    weights_np = weights_np / weights_np.sum()

    elbo = make_elbo_loss(
        N=moments_jnp.shape[0],
        batch_size=batch_size,
        num_samples=num_samples,
        weights=(jnp.asarray(weights_np) if use_nda_weight else None),
        log_scale_range=log_scale_range,
        e_max=e_max,
        use_sx=(log_scale_range is not None),
        raw2standard=raw2standard,
        mean_log_diag=mean_log_diag,
        std_log_diag=std_log_diag,
        mean_off=mean_off,
        std_off=std_off,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )

    model_tuple = (prior_flow, q_flow)
    opt_state = optimizer.init(eqx.filter(model_tuple, eqx.is_inexact_array))

    model_tuple, opt_state, losses = _run_training_loop(
        model_tuple,
        opt_state,
        optimizer,
        elbo,
        (moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp, centroid_moments_jnp),
        key,
        steps,
        chunk_size,
    )

    prior_trained, q_trained = model_tuple
    return prior_trained, q_trained, losses


def continue_training(
    key: jax.Array,
    prior_trained: Any,
    q_trained: Any,
    moments_jnp: jax.Array,
    centroid_moments_jnp: jax.Array,
    cov_jnp: jax.Array,
    dm_dg_jnp: jax.Array,
    d2m_dg2_jnp: jax.Array,
    weights: jax.Array,
    raw2standard: Any,
    *,
    steps: int = 5_000,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 0.3,
    log_scale_range: tuple[float, float] = _log_scale_range,
    e_max: float = _e_max,
    chunk_size: int = train_chunk_size,
    nda: jax.Array | None = None,
    use_nda_weight: bool = use_nda_weight,
) -> tuple[Any, Any, list[float]]:
    """Continue training from saved model weights.

    Reconstructs the ELBO loss, creates a new optimiser, and resumes
    training from the provided flow instances.  Saves the updated weights
    to disk on completion.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    prior_trained : Transformed
        Previously trained prior flow to resume from.
    q_trained : Transformed
        Previously trained q flow to resume from.
    moments_jnp : jax.Array, shape (N, 4)
        Raw template moments.
    centroid_moments_jnp : jax.Array, shape (N, 2)
        First-order Fourier moments (centroid moments) for each template.
        Passed as ``data_X`` to ``make_elbo_loss`` for centroid likelihood
        weighting during training.
    cov_jnp : jax.Array, shape (N, 4, 4)
        Per-object raw-moment covariance matrices.
    dm_dg_jnp : jax.Array, shape (N, 4, 2)
        First shear derivatives of the moments.
    d2m_dg2_jnp : jax.Array, shape (N, 4, 2, 2)
        Second shear derivatives of the moments.
    weights : jax.Array, shape (N,)
        Importance weights for training.
    raw2standard : RawMomentStandardize
        Bijection from raw to standardised moment coordinates.
    steps : int, optional
        Number of additional gradient steps.  Default is 10 000.
    learning_rate : float, optional
        AdamW learning rate.  Default is ``1e-3``.
    weight_decay : float, optional
        AdamW weight decay.  Default is ``1e-4``.
    grad_clip : float, optional
        Global gradient-norm clip.  Default is 0.3.
    log_scale_range : tuple of float, optional
        ``(min, max)`` of ``0.5 * log det(Σ_X)`` for Σ_X conditioning.
    e_max : float, optional
        Maximum ellipticity magnitude for Σ_X conditioning.
    chunk_size : int, optional
        Number of gradient steps fused into one ``jax.lax.scan`` dispatch
        (≈1.7x faster, identical math).  Default from ``config.train_chunk_size``.
        Set to 1 for the eager per-step loop (e.g. under ``jax_debug_nans``).
    nda : jax.Array or None, optional
        Per-template BFD ``nda = sky_density × da`` weight (the FITS ``weight``
        column).  When ``use_nda_weight`` is True, each template's loss
        contribution is scaled by ``nda`` so the flow learns the *nda-weighted*
        template prior, matching the traditional BFD integration.  ``None``
        (default) leaves the loss unweighted.
    use_nda_weight : bool, optional
        Master switch for the ``nda`` weighting.  Defaults to
        ``config.use_nda_weight``.

    Returns
    -------
    prior_trained : Transformed
        Updated prior flow.
    q_trained : Transformed
        Updated q flow.
    losses : list of float
        Training loss values.
    """
    mean_log_diag, std_log_diag, mean_off, std_off = compute_std_stats(
        raw2standard, moments_jnp, cov_jnp
    )

    weights_np = np.array(weights)
    weights_np = weights_np / weights_np.sum()

    elbo = make_elbo_loss(
        N=moments_jnp.shape[0],
        batch_size=batch_size,
        num_samples=num_samples,
        weights=(jnp.asarray(weights_np) if use_nda_weight else None),
        log_scale_range=log_scale_range,
        e_max=e_max,
        use_sx=(log_scale_range is not None),
        raw2standard=raw2standard,
        mean_log_diag=mean_log_diag,
        std_log_diag=std_log_diag,
        mean_off=mean_off,
        std_off=std_off,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )

    model_tuple = (prior_trained, q_trained)
    opt_state = optimizer.init(eqx.filter(model_tuple, eqx.is_inexact_array))

    model_tuple, opt_state, losses = _run_training_loop(
        model_tuple,
        opt_state,
        optimizer,
        elbo,
        (moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp, centroid_moments_jnp),
        key,
        steps,
        chunk_size,
    )

    prior_trained, q_trained = model_tuple

    save_models(prior_trained, q_trained, PRIOR_FLOW_PATH, Q_FLOW_PATH, raw2standard)

    return prior_trained, q_trained, losses


def pretrain_prior(
    key: jax.Array,
    moments_jnp: jax.Array,
    centroid_moments_jnp: jax.Array,
    cov_jnp: jax.Array,
    dm_dg_jnp: jax.Array,
    d2m_dg2_jnp: jax.Array,
    weights: jax.Array,
    raw2standard: Any,
    prior_flow: Any,
    *,
    steps: int = 80_000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 0.5,
    log_scale_range: tuple[float, float] = _log_scale_range,
    e_max: float = _e_max,
    chunk_size: int = train_chunk_size,
    nda: jax.Array | None = None,
    use_nda_weight: bool = use_nda_weight,
) -> tuple[Any, list[float]]:
    """Pre-train only the prior flow using direct NLL on template moments (m ≈ y).

    Gives all three prior stages — base shape, shear response, C_X response —
    a clean gradient signal without Q-sampling noise.  Run this before
    converge_train.py (joint ELBO fine-tuning) to give both flows a warm start.

    Parameters mirror train_model; prior_flow is the (possibly freshly built)
    prior to train in-place.  Returns (prior_trained, losses).
    """
    weights_np = np.array(weights)
    weights_np = weights_np / weights_np.sum()

    nll = make_nll_loss(
        N=moments_jnp.shape[0],
        batch_size=batch_size,
        weights=(jnp.asarray(weights_np) if use_nda_weight else None),
        log_scale_range=log_scale_range,
        e_max=e_max,
        use_sx=(log_scale_range is not None),
        raw2standard=raw2standard,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(prior_flow, eqx.is_inexact_array))

    prior_trained, _, losses = _run_training_loop(
        prior_flow,
        opt_state,
        optimizer,
        nll,
        (moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp, centroid_moments_jnp),
        key,
        steps,
        chunk_size,
    )
    return prior_trained, losses


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_models(
    prior_trained: Any,
    q_trained: Any,
    prior_path: str = PRIOR_FLOW_PATH,
    q_path: str = Q_FLOW_PATH,
    raw2standard: Any = None,
) -> None:
    """Serialise trained flow models to disk using equinox.

    Parameters
    ----------
    prior_trained : Transformed
        Trained prior normalizing flow.
    q_trained : Transformed
        Trained variational (q) normalizing flow.
    prior_path : str, optional
        Output path for the prior flow weights.  Defaults to
        ``config.PRIOR_FLOW_PATH``.
    q_path : str, optional
        Output path for the q flow weights.  Defaults to
        ``config.Q_FLOW_PATH``.
    raw2standard : RawMomentStandardize, optional
        The standardiser the flow was trained with.  When given, its
        ``(mean, std)`` are written to a sidecar next to each flow
        (``<path>.stats.npz``) so inference/plot tools load the matching
        standardiser by adjacency.  Strongly recommended.
    """
    # Save the trained models
    eqx.tree_serialise_leaves(prior_path, prior_trained)
    eqx.tree_serialise_leaves(q_path, q_trained)
    if raw2standard is not None:
        from .models.bijections import save_stats
        save_stats(prior_path, raw2standard)
        save_stats(q_path, raw2standard)


def load_models(
    prior_flow: Any,
    q_flow: Any,
    prior_path: str = PRIOR_FLOW_PATH,
    q_path: str = Q_FLOW_PATH,
) -> tuple[Any, Any]:
    """Load previously serialised flow weights into template pytrees.

    Parameters
    ----------
    prior_flow : Transformed
        An unweighted (freshly initialised) prior flow with the same
        architecture as the saved model.
    q_flow : Transformed
        An unweighted q flow with the same architecture as the saved model.
    prior_path : str, optional
        Path to the saved prior flow weights.  Defaults to
        ``config.PRIOR_FLOW_PATH``.
    q_path : str, optional
        Path to the saved q flow weights.  Defaults to ``config.Q_FLOW_PATH``.

    Returns
    -------
    prior_trained : Transformed
        Prior flow with loaded weights.
    q_trained : Transformed
        Q flow with loaded weights.
    """
    # Loading the models back
    prior_trained = eqx.tree_deserialise_leaves(prior_path, prior_flow)
    q_trained = eqx.tree_deserialise_leaves(q_path, q_flow)
    return prior_trained, q_trained


def load_or_train(
    key: jax.Array,
    moments_jnp: jax.Array,
    centroid_moments_jnp: jax.Array,
    cov_jnp: jax.Array,
    dm_dg_jnp: jax.Array,
    d2m_dg2_jnp: jax.Array,
    weights: jax.Array,
    raw2standard: Any,
    prior_path: str = PRIOR_FLOW_PATH,
    q_path: str = Q_FLOW_PATH,
    **train_kwargs: Any,
) -> tuple[Any, Any, list[float]]:
    """Load saved flow models if they exist, otherwise train from scratch and save.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    moments_jnp : jax.Array, shape (N, 4)
        Raw template moments.
    centroid_moments_jnp : jax.Array, shape (N, 2)
        First-order Fourier moments (centroid moments) for each template.
        Forwarded to :func:`train_model` as ``data_X`` for centroid likelihood
        weighting.
    cov_jnp : jax.Array, shape (N, 4, 4)
        Per-object raw-moment covariance matrices.
    dm_dg_jnp : jax.Array, shape (N, 4, 2)
        First shear derivatives of the moments.
    d2m_dg2_jnp : jax.Array, shape (N, 4, 2, 2)
        Second shear derivatives of the moments.
    weights : jax.Array, shape (N,)
        Importance weights for training.
    raw2standard : RawMomentStandardize
        Bijection from raw to standardised moment coordinates.
    prior_path : str, optional
        Path for the prior flow weights file.
    q_path : str, optional
        Path for the q flow weights file.
    **train_kwargs
        Additional keyword arguments forwarded to :func:`train_model`.

    Returns
    -------
    prior_trained : Transformed
        Prior flow (loaded or freshly trained).
    q_trained : Transformed
        Q flow (loaded or freshly trained).
    losses : list of float
        Training loss values (empty list when loading from disk).
    """
    import os

    key, sub = jr.split(key)
    prior_flow, q_flow = build_flows(
        sub,
        latent_dim=4,
        cond_dim=16,
        raw2standard=raw2standard,
        prior_flow_layers=prior_flow_layers,
        prior_early_nn_width=prior_early_nn_width,
        prior_early_nn_depth=prior_early_nn_depth,
        prior_last_nn_width=prior_last_nn_width,
        prior_last_nn_depth=prior_last_nn_depth,
        prior_sigmax_nn_width=prior_sigmax_nn_width,
        prior_sigmax_nn_depth=prior_sigmax_nn_depth,
        prior_sigmax_log_scale_mean=prior_sigmax_log_scale_mean,
        prior_sigmax_log_scale_std=prior_sigmax_log_scale_std,
        q_flow_layers=q_flow_layers,
        q_nn_width=q_nn_width,
        q_nn_depth=q_nn_depth,
        min_scale=min_scale,
        max_scale=max_scale,
    )

    if os.path.exists(prior_path) and os.path.exists(q_path):
        print(f"Loading models from {prior_path} and {q_path}")
        prior_trained, q_trained = load_models(prior_flow, q_flow, prior_path, q_path)
        return prior_trained, q_trained, []
    else:
        print("Training models from scratch...")
        prior_trained, q_trained, losses = train_model(
            key,
            moments_jnp,
            centroid_moments_jnp,
            cov_jnp,
            dm_dg_jnp,
            d2m_dg2_jnp,
            weights,
            raw2standard,
            **train_kwargs,
        )
        save_models(prior_trained, q_trained, prior_path, q_path, raw2standard)
        return prior_trained, q_trained, losses
