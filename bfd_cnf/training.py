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
)
from .config import (
    e_max as _e_max,
)
from .config import (
    log_scale_range as _log_scale_range,
)
from .config import (
    n_sx_train as _n_sx_train,
)
from .data import transform_dataset_to_standard
from .models.flows import (
    batch_cholesky_of_sym,
    build_flows,
    cov2corr,
    make_elbo_loss,
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
    n_sx_train: int = _n_sx_train,
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
        Maximum PSF ellipticity magnitude for Σ_X conditioning.
    n_sx_train : int, optional
        Number of Σ_X conditions sampled per gradient step.  The ELBO is
        evaluated at each and the losses are averaged, reducing gradient
        variance from the single-PSF estimate.  Default is 1 (original
        behaviour); try 4–8 to widen the learned marginal distribution.

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
        weights=jnp.asarray(weights_np),
        log_scale_range=log_scale_range,
        e_max=e_max,
        n_sx_train=n_sx_train,
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

    losses = []
    pbar = tqdm(range(steps))
    for _ in pbar:
        key, subkey = jr.split(key)
        model_tuple, opt_state, loss_val = train_step(
            model_tuple,
            moments_jnp,
            cov_jnp,
            dm_dg_jnp,
            d2m_dg2_jnp,
            centroid_moments_jnp,
            opt_state,
            subkey,
        )
        loss_f = float(loss_val)
        losses.append(loss_f)
        pbar.set_postfix(loss=f"{loss_f:.4f}")

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
    n_sx_train: int = _n_sx_train,
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
        Maximum PSF ellipticity magnitude for Σ_X conditioning.
    n_sx_train : int, optional
        Number of Σ_X conditions sampled per gradient step.  Losses are
        averaged over all conditions.  Default is 1 (original behaviour);
        try 4–8 to widen the learned marginal distribution.

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
        weights=jnp.asarray(weights_np),
        log_scale_range=log_scale_range,
        e_max=e_max,
        n_sx_train=n_sx_train,
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

    losses = []
    pbar = tqdm(range(steps))
    for _ in pbar:
        key, subkey = jr.split(key)
        model_tuple, opt_state, loss_val = train_step(
            model_tuple,
            moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp, centroid_moments_jnp,
            opt_state, subkey,
        )
        loss_f = float(loss_val)
        losses.append(loss_f)
        pbar.set_postfix(loss=f"{loss_f:.4f}")

    prior_trained, q_trained = model_tuple

    eqx.tree_serialise_leaves(PRIOR_FLOW_PATH, prior_trained)
    eqx.tree_serialise_leaves(Q_FLOW_PATH, q_trained)

    return prior_trained, q_trained, losses


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_models(
    prior_trained: Any,
    q_trained: Any,
    prior_path: str = PRIOR_FLOW_PATH,
    q_path: str = Q_FLOW_PATH,
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
    """
    # Save the trained models
    eqx.tree_serialise_leaves(prior_path, prior_trained)
    eqx.tree_serialise_leaves(q_path, q_trained)


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
        save_models(prior_trained, q_trained, prior_path, q_path)
        return prior_trained, q_trained, losses
