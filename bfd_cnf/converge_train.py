"""
converge_train.py
=================
Resume prior + q flow training in fixed-size blocks (default 10 000 steps),
checkpointing and measuring **per-stage convergence** after each block, and
stopping automatically once every stage of the prior flow has plateaued.

Convergence is assessed independently for the three trainable stages of the
prior (see :mod:`bfd_cnf.convergence_metrics`):

  * ``EquivariantAutoregressiveLayer`` — base moment shape.
  * ``ExplicitPolyLast``              — shear (g) response.
  * ``SigmaXCouplingLayer``           — centroid-covariance (C_X) response.

For each stage and each block we record:

  * **param rel-delta** — relative L2 change of the stage's trainable
    parameter vector vs. the previous checkpoint (the "parameters have stopped
    moving" signal the user asked for).
  * **functional rel-delta** — relative change of the stage's functional
    signature binned across moment space (flux × size), so we can see *where*
    in moment space each stage is still moving.

A stage is "converged" for a block when both rel-deltas are below their
thresholds.  Training stops when all three stages are converged AND the block
loss has stopped improving, for ``--patience`` consecutive blocks (or at
``--max-blocks``).

Unlike :func:`bfd_cnf.training.continue_training`, the optimizer (Adam) state is
carried across blocks, so blocks are just checkpoint/measure boundaries on one
continuous run — no per-block Adam warm-up transient to contaminate the
plateau signal.

Usage
-----
    cd /home/vwetzell/gitrepos/bfd_cnf
    python -m bfd_cnf.converge_train                         # real run, 10k blocks
    python -m bfd_cnf.converge_train --block-steps 50 \
        --max-blocks 2 --n-val 2000 \
        --prior-out /tmp/p.eqx --q-out /tmp/q.eqx --no-backup # smoke test
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .config import (
    PLOTS_DIR,
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    batch_size,
    e_max,
    key as _base_key,
    log_scale_range,
    n_sx_train,
    num_samples,
    prior_sigmax_log_scale_mean,
    train_chunk_size,
    use_nda_weight,
)
from .convergence_metrics import (
    STAGE_ORDER,
    collect_stage_params,
    compute_functional_fields,
    functional_plateau_metrics,
    make_moment_space_bins,
    param_plateau_metrics,
    stage_converged,
)
from .data import load_training_dataset, transform_dataset_to_standard
from .models.bijections import save_stats
from .models.flows import build_flows, make_elbo_loss, make_nll_loss
from .training import _run_training_loop, compute_std_stats, load_models

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup(path: str, tag: str) -> None:
    if os.path.exists(path):
        shutil.copy2(path, f"{path}.bak-{tag}")


def _fmt(v: float) -> str:
    return "   nan " if not np.isfinite(v) else f"{v:8.5f}"


def _print_block_table(block: int, total_steps: int, param_m, func_m, conv, loss_stats) -> None:
    print(
        f"\n┌─ block {block}  (total steps = {total_steps})  "
        f"loss median={loss_stats['median']:.4f} "
        f"mean={loss_stats['mean']:.4f} last={loss_stats['last']:.4f} "
        f"Δloss_rel={_fmt(loss_stats['rel_improve'])}"
    )
    print(f"│ {'stage':32s} {'param|θ|':>10s} {'paramΔrel':>10s} "
          f"{'funcMag':>10s} {'funcΔrel':>10s}  conv")
    for name in STAGE_ORDER:
        pm = param_m[name]
        fm = func_m[name]
        print(
            f"│ {name:32s} {pm['param_norm']:10.4f} {_fmt(pm['rel_delta']):>10s} "
            f"{fm['mean_mag']:10.4f} {_fmt(fm['rel_delta']):>10s}  "
            f"{'YES' if conv[name] else 'no '}"
        )
    print("└" + "─" * 78)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block-steps", type=int, default=10_000, help="Gradient steps per block.")
    p.add_argument("--max-blocks", type=int, default=10, help="Hard cap on number of blocks.")
    p.add_argument("--min-blocks", type=int, default=2, help="Always run at least this many blocks.")
    p.add_argument("--start-steps", type=int, default=None,
                   help="Step count already trained (for labelling/backups only). "
                        "Default: 0 with --from-scratch, else 50000.")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--lr-schedule", choices=["constant", "cosine", "plateau"], default="constant",
                   help="LR schedule. 'cosine' decays learning_rate -> lr_final over "
                        "lr_decay_steps.  'plateau' = reduce-LR-on-plateau: hold LR until the "
                        "loss flattens, then drop by lr_gamma; terminate at lr_min + gates "
                        "(adaptive — ensures convergence without guessing a horizon).")
    p.add_argument("--lr-final", type=float, default=None,
                   help="Final LR for the cosine schedule (default = learning_rate).")
    p.add_argument("--lr-decay-steps", type=int, default=None,
                   help="Steps over which cosine decay runs (default = block_steps*max_blocks).")
    p.add_argument("--lr-gamma", type=float, default=0.3,
                   help="plateau: multiply LR by this when the loss plateaus (default 0.3 — "
                        "fewer halvings from the start LR down to lr_min than the old 0.5).")
    p.add_argument("--lr-patience", type=int, default=2,
                   help="plateau: blocks that fail to beat the running-best loss (by tau_loss) "
                        "before an LR drop.")
    p.add_argument("--lr-cooldown", type=int, default=0,
                   help="plateau: after an LR drop, ignore this many blocks before counting "
                        "non-improving blocks again (lets the new LR settle; default 0 = off).")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="plateau: LR floor; convergence requires LR at this floor + gates.")
    p.add_argument("--n-val", type=int, default=100_000, help="Validation-set size for functional metrics.")
    # Defaults: 1500 < Mf < 90000 (in log10) and 2.2 < Mr/Mf < 3.5.
    p.add_argument("--val-logmf-min", type=float, default=np.log10(1500), help="Min log10(Mf) for validation sample.")
    p.add_argument("--val-logmf-max", type=float, default=np.log10(90000), help="Max log10(Mf) for validation sample.")
    p.add_argument("--val-mrmf-min", type=float, default=2.2, help="Min Mr/Mf for validation sample.")
    p.add_argument("--val-mrmf-max", type=float, default=3.5, help="Max Mr/Mf for validation sample.")
    p.add_argument("--tau-param", type=float, default=0.03,
                   help="Param rel-delta below this = stage params plateaued.")
    p.add_argument("--tau-func", type=float, default=0.05,
                   help="Functional rel-delta below this = stage behaviour plateaued.")
    p.add_argument("--tau-loss", type=float, default=0.003,
                   help="Relative loss-improvement margin. plateau: a block counts as progress "
                        "only if it beats the running-best block loss by this fraction; "
                        "constant/cosine: consecutive-block rel-improvement below this = plateaued.")
    p.add_argument("--patience", type=int, default=2,
                   help="Consecutive plateaued blocks required to stop.")
    p.add_argument("--tag", type=str, default="converge", help="Backup/label tag.")
    p.add_argument("--prior-out", type=str, default=PRIOR_FLOW_PATH)
    p.add_argument("--q-out", type=str, default=Q_FLOW_PATH)
    p.add_argument("--prior-in", type=str, default=PRIOR_FLOW_PATH)
    p.add_argument("--q-in", type=str, default=Q_FLOW_PATH)
    p.add_argument("--loss", choices=["elbo", "nll"], default="elbo",
                   help="Training objective. 'elbo' (default) jointly trains prior+q. "
                        "'nll' trains the prior alone on direct template NLL (z≈y, no q flow) "
                        "with the same per-stage convergence gates — for isolating ELBO/q issues.")
    p.add_argument("--from-scratch", action="store_true",
                   help="Initialise fresh flows (skip load_models) and train from random init "
                        "rather than resuming from a checkpoint.")
    p.add_argument("--no-backup", action="store_true", help="Do not write per-block .bak copies.")
    p.add_argument("--subsample", type=int, default=1,
                   help="Keep roughly 1/subsample of the training templates (sampled before cuts). "
                        "E.g. --subsample 8 reduces ~40M rows to ~5M, cutting GPU data memory "
                        "from ~7.5 GB to ~0.95 GB with no effect on gradient quality.")
    p.add_argument("--metrics-json", type=str,
                   default=os.path.join(_LOG_DIR, "converge_metrics.json"))
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    if args.start_steps is None:
        args.start_steps = 0 if args.from_scratch else 50_000

    print(f"devices: {jax.devices()}")
    print(f"Σ_X conditioning: log_scale_range={log_scale_range} e_max={e_max} "
          f"n_sx_train={n_sx_train}")
    print(f"blocks: {args.block_steps} steps each, max {args.max_blocks}; "
          f"thresholds τ_param={args.tau_param} τ_func={args.tau_func} τ_loss={args.tau_loss}")

    key = _base_key

    # --------------------------------------------------------------- data
    print("Loading data (reads the configured training template set)...")
    data = load_training_dataset(key=key, subsample=args.subsample)
    key = data["key"]
    moments_jnp = data["moments_jnp"]
    cov_jnp = data["cov_jnp"]
    dm_dg_jnp = data["dm_dg_jnp"]
    d2m_dg2_jnp = data["d2m_dg2_jnp"]
    centroid_moments_jnp = data["centroid_moments_jnp"]
    weights = data["weights"]   # batch-sampling proposal ∝ nda (BFD template weight)
    raw2standard = data["raw2standard"]
    N = moments_jnp.shape[0]
    print(f"  N templates = {N}")

    # ------------------------------------------------ build / load flows
    key, k_build = jr.split(key)
    prior_flow, q_flow = build_flows(k_build, latent_dim=4, cond_dim=16, raw2standard=raw2standard)
    if args.from_scratch:
        print("Initialising FRESH flows from scratch (random init, no checkpoint load).")
        prior, q = prior_flow, q_flow  # build_flows pulls arch from config → canonical structure
    else:
        print("Building architecture and loading checkpoints...")
        prior, q = load_models(prior_flow, q_flow, args.prior_in, args.q_in)

    # ------------------------------------------------ validation set
    key, k_val = jr.split(key)
    # Restrict candidates by raw log10(Mf) and Mr/Mf before sampling n_val.
    # cols: [0]=Mf, [1]=Mr, [2]=M1, [3]=M2
    m_all = np.asarray(moments_jnp)
    log_mf = np.log10(np.maximum(m_all[:, 0], np.finfo(np.float32).tiny))
    mr_mf = m_all[:, 1] / m_all[:, 0]
    keep = np.ones(N, bool)
    if args.val_logmf_min is not None: keep &= log_mf >= args.val_logmf_min
    if args.val_logmf_max is not None: keep &= log_mf <= args.val_logmf_max
    if args.val_mrmf_min is not None: keep &= mr_mf >= args.val_mrmf_min
    if args.val_mrmf_max is not None: keep &= mr_mf <= args.val_mrmf_max
    cand = np.flatnonzero(keep)
    if cand.size < N:
        print(f"  validation cut: {cand.size}/{N} templates pass log10(Mf)/Mr/Mf limits")
    n_val = min(args.n_val, cand.size)
    val_idx = cand[jr.choice(k_val, cand.size, shape=(n_val,), replace=False)]
    z_val, _ = transform_dataset_to_standard(
        raw2standard, moments_jnp[val_idx], cov_jnp[val_idx]
    )
    z_val = jnp.asarray(z_val)
    bins = make_moment_space_bins(z_val)
    c_ref = jnp.array([0.0, 0.0, float(prior_sigmax_log_scale_mean), 0.0, 0.0])
    print(f"  validation set: {n_val} pts, {bins.nx}×{bins.ny} moment-space bins "
          f"({int(bins.valid_mask.sum())} valid)")

    # ------------------------------------------------ ELBO + optimizer (kept across blocks)
    mean_log_diag, std_log_diag, mean_off, std_off = compute_std_stats(
        raw2standard, moments_jnp, cov_jnp
    )
    # Proposal ∝ nda (use_nda_weight=False ⇒ uniform sampling).  The per-copy weight is
    # L(X|C_X) alone inside the loss — see make_nll_loss / make_elbo_loss.
    w_np = np.asarray(weights)
    w_np = w_np / w_np.sum()
    sampling_weights = jnp.asarray(w_np) if use_nda_weight else None
    if args.loss == "nll":
        # q-free: prior is supervised directly on log p(y_std | g, C_X) (z≈y).
        loss_fn = make_nll_loss(
            N=N, batch_size=batch_size,
            weights=sampling_weights,
            log_scale_range=log_scale_range, e_max=e_max,
            n_sx_train=n_sx_train, use_sx=(log_scale_range is not None),
            raw2standard=raw2standard,
        )
        print("Loss: NLL (q-free, prior-only direct template NLL)")
    else:
        loss_fn = make_elbo_loss(
            N=N, batch_size=batch_size, num_samples=num_samples,
            weights=sampling_weights,
            log_scale_range=log_scale_range, e_max=e_max,
            n_sx_train=n_sx_train, use_sx=(log_scale_range is not None),
            raw2standard=raw2standard, mean_log_diag=mean_log_diag,
            std_log_diag=std_log_diag, mean_off=mean_off, std_off=std_off,
        )
        print("Loss: ELBO (joint prior+q)")
    def make_opt(lr):
        # A scalar LR is applied at update time (not stored in Adam's state), so the
        # opt_state structure is identical for any scalar lr — letting the 'plateau'
        # mode rebuild the optimizer with a new LR each drop while REUSING opt_state
        # (Adam moments stay warm).
        return optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(learning_rate=lr, weight_decay=args.weight_decay),
        )

    current_lr = args.learning_rate          # mutated by the 'plateau' schedule
    # Reduce-LR-on-plateau state (noise-robust, ReduceLROnPlateau semantics): track the
    # best block loss seen so far and drop LR after `lr_patience` blocks fail to beat it
    # by `tau_loss`.  The old scheme counted *consecutive* sub-threshold blocks and reset
    # on any noisy block, so a single uptick stalled the decay (the last from-scratch run
    # sat at LR=1e-3 for ~160k steps for exactly this reason).
    best_block_loss = float("inf")
    bad_blocks = 0
    lr_cooldown_left = 0
    if args.lr_schedule == "cosine":
        lr_final = args.lr_final if args.lr_final is not None else args.learning_rate
        decay_steps = args.lr_decay_steps or (args.block_steps * args.max_blocks)
        base_lr = optax.cosine_decay_schedule(
            init_value=args.learning_rate, decay_steps=decay_steps,
            alpha=lr_final / args.learning_rate,
        )
        print(f"LR schedule: cosine {args.learning_rate:.1e} -> {lr_final:.1e} over {decay_steps} steps")
    elif args.lr_schedule == "plateau":
        base_lr = None
        print(f"LR schedule: plateau — start {args.learning_rate:.1e}, ×{args.lr_gamma} after "
              f"{args.lr_patience} blocks not beating best by {args.tau_loss} "
              f"(cooldown {args.lr_cooldown}), floor {args.lr_min:.1e}")
    else:
        base_lr = args.learning_rate
        print(f"LR schedule: constant {args.learning_rate:.1e}")

    # The optimised model is the (prior, q) pair for ELBO, or the prior alone for NLL.
    model_tuple = (prior, q) if args.loss == "elbo" else prior
    _init_lr = base_lr if args.lr_schedule == "cosine" else args.learning_rate
    opt_state = make_opt(_init_lr).init(eqx.filter(model_tuple, eqx.is_inexact_array))

    # ------------------------------------------------ pristine backup of the resume point
    if not args.no_backup:
        tag0 = f"{args.tag}-pre{args.start_steps // 1000}k-{_ts()}"
        _backup(args.prior_out, tag0)
        if args.loss == "elbo":
            _backup(args.q_out, tag0)
        print(f"  backed up resume checkpoint with tag '{tag0}'")

    # ------------------------------------------------ baseline (block 0) metrics
    prev_params = collect_stage_params(prior)
    prev_fields = compute_functional_fields(prior, z_val, bins, c_ref)
    param_m0 = param_plateau_metrics(prev_params, None)
    func_m0 = functional_plateau_metrics(prev_fields, None, bins)
    print("\nBaseline (resume point) functional magnitudes:")
    for name in STAGE_ORDER:
        print(f"  {name:32s} |θ|={param_m0[name]['param_norm']:.4f} "
              f"funcMag={func_m0[name]['mean_mag']:.4f}")

    history: list[dict] = [{
        "block": 0, "total_steps": args.start_steps,
        "loss": {"median": float("nan"), "mean": float("nan"),
                 "last": float("nan"), "rel_improve": float("nan")},
        "param": {n: param_m0[n] for n in STAGE_ORDER},
        "func": {n: {k: v for k, v in func_m0[n].items()} for n in STAGE_ORDER},
        "converged": {n: False for n in STAGE_ORDER},
        "fields": {n: np.asarray(prev_fields[n]["field"]).tolist() for n in STAGE_ORDER},
    }]

    # ------------------------------------------------ block loop
    total_steps = args.start_steps
    prev_block_median = None
    consecutive_ok = 0
    fields_for_plot = prev_fields  # updated each block; final used for field plots
    prev_fields_for_delta = prev_fields

    # Build the optimizer + jitted runner ONCE and reuse them across blocks.  For
    # cosine/constant the optimizer never changes (the LR varies via opt_state.count
    # or is fixed), so rebuilding per block only recompiles a fresh XLA executable
    # that is never freed -> GPU OOM after ~10 blocks.  For 'plateau' the optimizer
    # is rebuilt ONLY when the LR actually drops, and the runner cache is cleared
    # then (the runner closes over the optimizer) with the stale executable freed.
    optimizer = make_opt(current_lr if args.lr_schedule == "plateau" else base_lr)
    last_opt_lr = current_lr
    runners: dict = {}

    for block in range(1, args.max_blocks + 1):
        if args.lr_schedule == "plateau" and current_lr != last_opt_lr:
            optimizer = make_opt(current_lr)   # reuses opt_state — Adam moments warm
            runners.clear()
            jax.clear_caches()                 # free the stale compiled runner
            last_opt_lr = current_lr
        key, k_train = jr.split(key)
        model_tuple, opt_state, losses = _run_training_loop(
            model_tuple, opt_state, optimizer, loss_fn,
            (moments_jnp, cov_jnp, dm_dg_jnp, d2m_dg2_jnp, centroid_moments_jnp),
            k_train, args.block_steps, train_chunk_size, runners=runners,
        )
        prior = model_tuple[0] if args.loss == "elbo" else model_tuple
        if args.loss == "elbo":
            q = model_tuple[1]
        total_steps += args.block_steps

        losses_arr = np.asarray(losses)
        finite = losses_arr[np.isfinite(losses_arr)]
        med = float(np.median(finite)) if finite.size else float("nan")
        rel_improve = (
            (prev_block_median - med) / max(abs(prev_block_median), 1e-12)
            if prev_block_median is not None else float("nan")
        )
        loss_stats = {
            "median": med, "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "last": float(finite[-1]) if finite.size else float("nan"),
            "rel_improve": rel_improve,
            "n_nonfinite": int(np.sum(~np.isfinite(losses_arr))),
        }

        # checkpoint (only overwrite canonical paths if the block stayed finite)
        if loss_stats["n_nonfinite"] == 0:
            eqx.tree_serialise_leaves(args.prior_out, prior)
            save_stats(args.prior_out, raw2standard)
            if args.loss == "elbo":
                eqx.tree_serialise_leaves(args.q_out, q)
                save_stats(args.q_out, raw2standard)
            if not args.no_backup:
                tag = f"{args.tag}{total_steps // 1000}k-{_ts()}"
                _backup(args.prior_out, tag)
                if args.loss == "elbo":
                    _backup(args.q_out, tag)
        else:
            print(f"  WARNING: {loss_stats['n_nonfinite']} non-finite losses in block "
                  f"{block}; NOT overwriting canonical checkpoint.")

        # metrics
        cur_params = collect_stage_params(prior)
        cur_fields = compute_functional_fields(prior, z_val, bins, c_ref)
        param_m = param_plateau_metrics(cur_params, prev_params)
        func_m = functional_plateau_metrics(cur_fields, prev_fields_for_delta, bins)
        conv = stage_converged(param_m, func_m, args.tau_param, args.tau_func)

        _print_block_table(block, total_steps, param_m, func_m, conv, loss_stats)

        history.append({
            "block": block, "total_steps": total_steps, "loss": loss_stats,
            "lr": float(current_lr),
            "param": {n: param_m[n] for n in STAGE_ORDER},
            "func": {n: {k: v for k, v in func_m[n].items()} for n in STAGE_ORDER},
            "converged": {n: bool(conv[n]) for n in STAGE_ORDER},
            "fields": {n: np.asarray(cur_fields[n]["field"]).tolist() for n in STAGE_ORDER},
        })
        with open(args.metrics_json, "w") as fh:
            json.dump({"args": vars(args), "history": history}, fh, indent=2)

        # convergence test (+ reduce-LR-on-plateau scheduling)
        all_stage_ok = all(conv[n] for n in STAGE_ORDER)
        loss_ok = np.isfinite(rel_improve) and rel_improve < args.tau_loss
        dropped = False
        improved = False

        if args.lr_schedule == "plateau":
            at_floor = current_lr <= args.lr_min * 1.0001
            # Noise-robust plateau detection: a block is "progress" only if it beats the
            # BEST block loss so far by the relative margin tau_loss.  A non-improving
            # block increments bad_blocks instead of resetting a consecutive counter, so
            # a single noisy uptick no longer stalls the decay.  bad_blocks resets only on
            # genuine progress; an optional cooldown skips counting right after a drop.
            if np.isfinite(med) and med < best_block_loss * (1.0 - args.tau_loss):
                best_block_loss = med
                bad_blocks = 0
                improved = True
            elif np.isfinite(med):
                if lr_cooldown_left > 0:
                    lr_cooldown_left -= 1
                else:
                    bad_blocks += 1
            if bad_blocks >= args.lr_patience and not at_floor:
                current_lr = max(current_lr * args.lr_gamma, args.lr_min)
                bad_blocks = 0
                lr_cooldown_left = args.lr_cooldown
                dropped = True
                at_floor = current_lr <= args.lr_min * 1.0001
            # converged only once LR is parked at the floor AND all stage gates pass
            block_ok = all_stage_ok and at_floor
        else:
            block_ok = all_stage_ok and loss_ok

        consecutive_ok = consecutive_ok + 1 if block_ok else 0

        if args.lr_schedule == "plateau":
            print(f"  → LR={current_lr:.2e}{'  ↓dropped' if dropped else ''}  "
                  f"best={best_block_loss:.4f} improved={improved}  "
                  f"bad={bad_blocks}/{args.lr_patience}"
                  f"{f' cooldown={lr_cooldown_left}' if lr_cooldown_left else ''}  "
                  f"all-stages={all_stage_ok}  consecutive_ok={consecutive_ok}/{args.patience}")
        else:
            print(f"  → all-stages={all_stage_ok}  loss_plateau={loss_ok}  "
                  f"consecutive_ok={consecutive_ok}/{args.patience}")

        prev_params = cur_params
        prev_fields_for_delta = cur_fields
        fields_for_plot = cur_fields
        prev_block_median = med

        if block >= args.min_blocks and consecutive_ok >= args.patience:
            print(f"\n✓ CONVERGED after block {block} (total steps = {total_steps}).")
            break
    else:
        print(f"\n⚠ Reached max-blocks ({args.max_blocks}) without full plateau "
              f"(total steps = {total_steps}).")

    print(f"\nMetrics written to {args.metrics_json}")

    # ------------------------------------------------ plots
    if not args.no_plots:
        try:
            _make_plots(history, fields_for_plot, bins, args.tag)
        except Exception as exc:  # plotting must never lose a finished run
            print(f"  (plotting failed: {exc!r})")

    return 0


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _make_plots(history, final_fields, bins, tag) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    blocks = [h for h in history if h["block"] > 0]
    steps = [h["total_steps"] for h in blocks]
    colors = {STAGE_ORDER[0]: "C0", STAGE_ORDER[1]: "C1", STAGE_ORDER[2]: "C2"}
    short = {STAGE_ORDER[0]: "Equiv (base shape)",
             STAGE_ORDER[1]: "PolyLast (shear)",
             STAGE_ORDER[2]: "SigmaX (C_X)"}

    # ---- dashboard ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax = ax.ravel()

    ax[0].plot(steps, [h["loss"]["median"] for h in blocks], "o-k")
    ax[0].set_title("block median ELBO loss"); ax[0].set_xlabel("total steps")
    ax[0].set_ylabel("loss"); ax[0].grid(alpha=0.3)

    for name in STAGE_ORDER:
        ax[1].plot(steps, [h["param"][name]["rel_delta"] for h in blocks],
                   "o-", color=colors[name], label=short[name])
    ax[1].set_yscale("log"); ax[1].set_title("per-stage PARAM rel-Δ (plateau ↓)")
    ax[1].set_xlabel("total steps"); ax[1].set_ylabel("‖Δθ‖/‖θ‖")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)

    for name in STAGE_ORDER:
        ax[2].plot(steps, [h["func"][name]["rel_delta"] for h in blocks],
                   "o-", color=colors[name], label=short[name])
    ax[2].set_yscale("log"); ax[2].set_title("per-stage FUNCTIONAL rel-Δ across moment space (plateau ↓)")
    ax[2].set_xlabel("total steps"); ax[2].set_ylabel("‖ΔF‖/‖F‖")
    ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)

    for name in STAGE_ORDER:
        ax[3].plot(steps, [h["func"][name]["mean_mag"] for h in blocks],
                   "o-", color=colors[name], label=short[name])
    ax[3].set_title("per-stage functional magnitude (settles to a value)")
    ax[3].set_xlabel("total steps"); ax[3].set_ylabel("mean |signature|")
    ax[3].grid(alpha=0.3); ax[3].legend(fontsize=8)

    fig.suptitle("Prior-flow per-stage convergence", fontsize=14)
    fig.tight_layout()
    f1 = os.path.join(PLOTS_DIR, f"convergence_dashboard_{tag}.png")
    fig.savefig(f1, dpi=130); plt.close(fig)
    print(f"  saved {f1}")

    # ---- final across-moment-space field maps ----
    extent = [bins.x_edges[0], bins.x_edges[-1], bins.y_edges[0], bins.y_edges[-1]]
    titles = {
        STAGE_ORDER[0]: "base shape:  log p(z | ref)",
        STAGE_ORDER[1]: "shear resp:  |∂logp/∂g|",
        STAGE_ORDER[2]: "C_X resp:  |∂logp/∂(logS,e1,e2)|",
    }

    def _field_mag(name):
        f = np.asarray(final_fields[name]["field"])
        if f.ndim == 1:
            return f.reshape(bins.nx, bins.ny)
        return np.linalg.norm(f, axis=1).reshape(bins.nx, bins.ny)

    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.6))
    for k, name in enumerate(STAGE_ORDER):
        m = _field_mag(name).T
        im = ax2[k].imshow(m, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        ax2[k].set_title(f"{short[name]}\n{titles[name]}", fontsize=10)
        ax2[k].set_xlabel("std flux z0"); ax2[k].set_ylabel("std size z1")
        fig2.colorbar(im, ax=ax2[k], fraction=0.046)
    fig2.suptitle(f"Final per-stage signatures across moment space "
                  f"(total steps = {blocks[-1]['total_steps']})", fontsize=13)
    fig2.tight_layout()
    f2 = os.path.join(PLOTS_DIR, f"convergence_fields_{tag}.png")
    fig2.savefig(f2, dpi=130); plt.close(fig2)
    print(f"  saved {f2}")


if __name__ == "__main__":
    raise SystemExit(main())
