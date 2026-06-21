"""
pretrain_prior.py
=================
Pre-train only the prior flow using direct NLL on template moments (z ≈ y).

This warmup phase gives all three prior stages a clean, Q-free gradient signal
before joint ELBO fine-tuning with converge_train.py.  The prior learns the base
galaxy moment distribution, shear response, and C_X response from the template
data directly — without the Q-coupling noise that slows convergence in joint
training, and without the weak indirect gradient that starves the SigmaX stage.

After this script completes, run converge_train.py to jointly fine-tune both
flows with the ELBO.  It will load the pre-trained prior automatically.

Usage
-----
    cd /home/vwetzell/gitrepos/bfd_cnf

    # Default: 80k steps, constant LR 1e-4, resume prior if checkpoint exists
    python -m bfd_cnf.pretrain_prior

    # From scratch, cosine LR decay, save Q init alongside prior
    python -m bfd_cnf.pretrain_prior --from-scratch --steps 80000 \\
        --lr-schedule cosine --lr-final 1e-6 --save-fresh-q

    # Smoke test
    python -m bfd_cnf.pretrain_prior --from-scratch --steps 100 --no-backup
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .config import (
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    e_max,
    key as _base_key,
    log_scale_range,
    n_sx_train,
    nda_clip_percentile,
    train_chunk_size,
    use_nda_weight,
)
from .data import load_training_dataset
from .models.flows import build_flows, make_nll_loss
from .training import _run_training_loop


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup(path: str, tag: str) -> None:
    if os.path.exists(path):
        shutil.copy2(path, f"{path}.bak-{tag}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=80_000,
                   help="Gradient steps for the NLL warmup (default 80 000).")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                   help="LR schedule: constant (default) or cosine decay to --lr-final.")
    p.add_argument("--lr-final", type=float, default=1e-6,
                   help="Final LR for cosine schedule (default 1e-6).")
    p.add_argument("--from-scratch", action="store_true",
                   help="Initialise a fresh prior even if a checkpoint exists.")
    p.add_argument("--prior-in", type=str, default=PRIOR_FLOW_PATH,
                   help="Path to load an existing prior checkpoint (if not --from-scratch).")
    p.add_argument("--prior-out", type=str, default=PRIOR_FLOW_PATH,
                   help="Path to save the pre-trained prior.")
    p.add_argument("--save-fresh-q", action="store_true",
                   help="Also save a freshly initialised Q flow to --q-out so "
                        "converge_train.py can load it without --from-scratch.")
    p.add_argument("--q-out", type=str, default=Q_FLOW_PATH,
                   help="Path to save the fresh Q flow (only used with --save-fresh-q).")
    p.add_argument("--no-backup", action="store_true",
                   help="Do not write .bak copies before overwriting checkpoints.")
    p.add_argument("--subsample", type=int, default=1,
                   help="Keep 1/subsample of templates (same as converge_train.py).")
    args = p.parse_args()

    print(f"devices: {jax.devices()}")
    print(f"NLL prior pre-training: {args.steps} steps, LR={args.learning_rate:.1e}, "
          f"schedule={args.lr_schedule}")
    print(f"Σ_X conditioning: log_scale_range={log_scale_range}  e_max={e_max}  "
          f"n_sx_train={n_sx_train}")

    key = _base_key

    # ------------------------------------------------------------------ data
    print("Loading data...")
    data = load_training_dataset(key=key, subsample=args.subsample)
    key = data["key"]
    moments_jnp = data["moments_jnp"]
    cov_jnp = data["cov_jnp"]
    dm_dg_jnp = data["dm_dg_jnp"]
    d2m_dg2_jnp = data["d2m_dg2_jnp"]
    centroid_moments_jnp = data["centroid_moments_jnp"]
    weights = data["weights"]
    nda = data["nda"]
    raw2standard = data["raw2standard"]
    N = moments_jnp.shape[0]
    print(f"  N templates = {N}")

    # ---------------------------------------------------------- build / load
    key, k_build = jr.split(key)
    prior_flow, q_flow = build_flows(k_build, latent_dim=4, cond_dim=16)

    if args.from_scratch:
        prior = prior_flow
        print("Initialising fresh prior from scratch.")
    elif os.path.exists(args.prior_in):
        prior = eqx.tree_deserialise_leaves(args.prior_in, prior_flow)
        print(f"Loaded prior from {args.prior_in}.")
    else:
        prior = prior_flow
        print(f"No checkpoint at {args.prior_in}; starting fresh.")

    # ----------------------------------------------------------- backup
    if not args.no_backup:
        tag = f"pre-nll-{_ts()}"
        _backup(args.prior_out, tag)
        print(f"  backed up prior with tag '{tag}'")

    # ----------------------------------------------------------- NLL loss + optimizer
    from .config import batch_size
    weights_np = np.array(weights)
    weights_np = weights_np / weights_np.sum()

    nll = make_nll_loss(
        N=N,
        batch_size=batch_size,
        weights=jnp.asarray(weights_np),
        nda=(jnp.asarray(nda) if use_nda_weight else None),
        nda_clip_percentile=nda_clip_percentile,
        log_scale_range=log_scale_range,
        e_max=e_max,
        n_sx_train=n_sx_train,
        use_sx=(log_scale_range is not None),
        raw2standard=raw2standard,
    )

    if args.lr_schedule == "cosine":
        lr_sched = optax.cosine_decay_schedule(
            init_value=args.learning_rate,
            decay_steps=args.steps,
            alpha=args.lr_final / args.learning_rate,
        )
        print(f"LR: cosine {args.learning_rate:.1e} → {args.lr_final:.1e} over {args.steps} steps")
    else:
        lr_sched = args.learning_rate
        print(f"LR: constant {args.learning_rate:.1e}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=lr_sched, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(prior, eqx.is_inexact_array))

    # ----------------------------------------------------------- training loop
    key, k_train = jr.split(key)
    prior_trained, _, losses = _run_training_loop(
        prior,
        opt_state,
        optimizer,
        nll,
        (moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp, centroid_moments_jnp),
        k_train,
        args.steps,
        train_chunk_size,
    )
    losses_arr = np.asarray(losses)
    finite = losses_arr[np.isfinite(losses_arr)]
    print(f"\nDone.  loss: first={losses_arr[0]:.4f}  "
          f"median={np.median(finite):.4f}  last={finite[-1]:.4f}  "
          f"n_nonfinite={int(np.sum(~np.isfinite(losses_arr)))}")

    # ----------------------------------------------------------- save
    eqx.tree_serialise_leaves(args.prior_out, prior_trained)
    print(f"Saved pre-trained prior → {args.prior_out}")

    if args.save_fresh_q:
        if os.path.exists(args.q_out) and not args.no_backup:
            _backup(args.q_out, f"pre-nll-{_ts()}")
        eqx.tree_serialise_leaves(args.q_out, q_flow)
        print(f"Saved fresh Q init → {args.q_out}")
    else:
        print("Q flow not saved (pass --save-fresh-q to write a fresh Q checkpoint "
              "so converge_train.py can load both flows without --from-scratch).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
