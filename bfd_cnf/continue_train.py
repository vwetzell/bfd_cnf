"""
continue_train.py
==================
Continue training the prior + q normalizing flows from their saved checkpoints
for a number of additional gradient steps.

Use this to adapt the flows to an updated Σ_X conditioning range
(``config.log_scale_range`` / ``config.e_max``) without retraining from
scratch — when the shift is small, a short fine-tune is much faster.

It loads the existing ``prior_flow_xy.eqx`` / ``q_flow_xy.eqx``, resumes via
:func:`bfd_cnf.training.continue_training` (which re-serialises the updated
weights to the **same** paths on completion), and reports the loss trajectory.
Because the save happens only after the loop finishes, a crash mid-training
leaves the checkpoints untouched.

Usage
-----
    # from the repo root so the package import resolves:
    cd /home/vwetzell/gitrepos/bfd_cnf
    python -m bfd_cnf.continue_train                 # 5000-step fine-tune
    python -m bfd_cnf.continue_train --steps 200     # quick smoke run
"""

from __future__ import annotations

import argparse

import jax
import jax.random as jr
import numpy as np

from .config import (
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    e_max,
    key as _base_key,
    log_scale_range,
)
from .data import load_training_dataset
from .models.flows import build_flows
from .training import continue_training, load_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=5000,
        help="Number of additional gradient steps (default: 5000).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="AdamW learning rate (default 1e-4, the run.py continue setting).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="AdamW weight decay."
    )
    parser.add_argument(
        "--grad-clip", type=float, default=0.5, help="Global grad-norm clip."
    )
    parser.add_argument(
        "--debug-nans", action="store_true",
        help="Enable jax_debug_nans (localises NaNs, but much slower).",
    )
    args = parser.parse_args()

    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)

    print(f"devices: {jax.devices()}")
    print(f"Sigma_X conditioning: log_scale_range={log_scale_range}  e_max={e_max}")

    key = _base_key

    # ------------------------------------------------------------------ data
    print("Loading data (reads the FITS template table)...")
    data = load_training_dataset(key=key)
    key = data["key"]

    # ----------------------------------------------------- load checkpoints
    print("Building architecture and loading existing checkpoints...")
    key, k_build = jr.split(key)
    prior_flow, q_flow = build_flows(k_build, latent_dim=4, cond_dim=16, raw2standard=data["raw2standard"])
    prior_trained, q_trained = load_models(
        prior_flow, q_flow, PRIOR_FLOW_PATH, Q_FLOW_PATH
    )

    # --------------------------------------------------------- continue train
    print(f"Continuing training for {args.steps} steps...")
    key, k_train = jr.split(key)
    prior_trained, q_trained, losses = continue_training(
        k_train,
        prior_trained,
        q_trained,
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
    )

    losses_arr = np.asarray(losses)
    n_bad = int(np.sum(~np.isfinite(losses_arr)))
    print(
        f"Done: {len(losses)} steps, final loss = {losses_arr[-1]:.4f}, "
        f"min = {np.nanmin(losses_arr):.4f}, non-finite losses = {n_bad}"
    )
    print(f"Updated weights saved to:\n  {PRIOR_FLOW_PATH}\n  {Q_FLOW_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
