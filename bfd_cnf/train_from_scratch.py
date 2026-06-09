"""
train_from_scratch.py
=====================
Train the prior and q normalizing flows from scratch with JAX NaN-checking
turned on, so the first NaN produced anywhere in the ELBO / flow forward or
backward pass aborts immediately with a traceback pointing at the offending op.

This isolates the *training* step of the pipeline (run.py also runs the long
grid-inference stage afterwards and turns jax_debug_nans back off).

Usage
-----
    # from the repo root so the package import resolves:
    cd /home/vwetzell/gitrepos/bfd_cnf
    python -m bfd_cnf.train_from_scratch                # full 20k-step run
    python -m bfd_cnf.train_from_scratch --steps 200    # quick smoke run

Notes
-----
* ``jax_debug_nans=True`` makes every primitive's output checked for NaN/Inf.
  When one is found JAX raises ``FloatingPointError`` and re-runs the op in
  eager mode to localise it.  This is slow — expect a large per-step slowdown
  versus a normal run — but it is exactly what catches the NaN at its source.
* On a NaN the script prints the approximate step (from the tqdm counter) and
  re-raises so the full traceback is visible.
* Weights are only saved if training completes without a NaN.
"""

from __future__ import annotations

import argparse
import sys

import jax

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .config import (
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    e_max,
    key as _base_key,
    log_scale_range,
    n_sx_train,
)
from .data import load_data
from .training import save_models, train_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=20_000,
        help="Number of gradient steps (default: 20000, matching run.py).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="AdamW learning rate."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="AdamW weight decay."
    )
    parser.add_argument(
        "--grad-clip", type=float, default=0.5, help="Global grad-norm clip."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Train but do not write .eqx weights to disk.",
    )
    parser.add_argument(
        "--debug-nans",
        action="store_true",
        help="Enable jax_debug_nans (≈10-50x slower; for localising a NaN at its "
        "source).  Off by default so full 20k-step runs are fast.",
    )
    args = parser.parse_args()

    # Must be set before any tracing/compilation (incl. the jnp ops in load_data).
    jax.config.update("jax_debug_nans", args.debug_nans)
    print(f"jax_debug_nans = {jax.config.jax_debug_nans}")
    print(f"devices: {jax.devices()}")

    key = _base_key

    # ------------------------------------------------------------------ data
    print("Loading data...")
    data = load_data(key=key)
    key = data["key"]

    # ------------------------------------------------------------- training
    print(f"Training from scratch for {args.steps} steps "
          f"(NaN-checking {'ON' if args.debug_nans else 'OFF'})...")
    try:
        prior_trained, q_trained, losses = train_model(
            key,
            data["moments_jnp"],
            data["centroid_moments_jnp"],
            data["cov_jnp"],
            data["dm_dg_jnp"],
            data["d2m_dg2_jnp"],
            data["weights"],
            data["raw2standard"],
            steps=args.steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            log_scale_range=log_scale_range,
            e_max=e_max,
            n_sx_train=n_sx_train,
        )
    except FloatingPointError:
        # jax_debug_nans raises this the moment a NaN/Inf is produced.
        print(
            "\n*** NaN/Inf detected during training (jax_debug_nans). ***\n"
            "The traceback below points at the operation that produced it.\n",
            file=sys.stderr,
        )
        raise

    losses_arr = np.asarray(losses)
    n_bad = int(np.sum(~np.isfinite(losses_arr)))
    print(
        f"Training finished: {len(losses)} steps, "
        f"final loss = {losses_arr[-1]:.4f}, "
        f"min = {np.nanmin(losses_arr):.4f}, "
        f"non-finite losses = {n_bad}"
    )

    if args.no_save:
        print("--no-save set; not writing weights.")
    else:
        save_models(prior_trained, q_trained, PRIOR_FLOW_PATH, Q_FLOW_PATH)
        print(f"Saved weights to:\n  {PRIOR_FLOW_PATH}\n  {Q_FLOW_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
