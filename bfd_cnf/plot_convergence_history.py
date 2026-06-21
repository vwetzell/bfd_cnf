"""
plot_convergence_history.py
===========================
Stitch the per-block convergence metrics from one or more
:mod:`bfd_cnf.converge_train` phases into a single 50k→end timeline figure.

Each phase JSON (written by ``converge_train``) holds ``{"args":..., "history":
[...]}`` where every block records the median loss and, per prior stage, the
parameter and functional relative-deltas plus the functional magnitude.  This
script concatenates the phases by ``total_steps`` and draws a 4-panel dashboard:

  * median ELBO loss
  * per-stage parameter rel-Δ (plateau ↓), with the strict gate line
  * per-stage functional rel-Δ across moment space (plateau ↓), with gate line
  * per-stage functional magnitude (shows which stages are still *building*)

Phase boundaries (e.g. the switch from fixed LR to the cosine-decay finish) are
shaded and labelled, so the LR-decay collapse of the noise ball is visible.

Usage
-----
    cd /home/vwetzell/gitrepos/bfd_cnf
    python -m bfd_cnf.plot_convergence_history          # default 3-phase stitch
    python -m bfd_cnf.plot_convergence_history \
        --phase fixed:logs/converge_metrics.json:'fixed LR 1e-4' \
        --phase finish:logs/converge_finish.json:'cosine 1e-4->1e-6'
"""

from __future__ import annotations

import argparse
import json
import os

from .config import PLOTS_DIR
from .convergence_metrics import STAGE_ORDER

_SHORT = {
    STAGE_ORDER[0]: "Equiv (base shape)",
    STAGE_ORDER[1]: "PolyLast (shear)",
    STAGE_ORDER[2]: "SigmaX (C_X)",
}
_COLORS = {STAGE_ORDER[0]: "C0", STAGE_ORDER[1]: "C1", STAGE_ORDER[2]: "C2"}

# Default phases: (label, json path, LR description, shade color)
_DEFAULT_PHASES = [
    ("fixed", "logs/converge_metrics.json", "fixed LR 1e-4", None),
    ("cosine-test", "logs/converge_cosine10k.json", "cosine 1e-4→1e-5", "0.93"),
    ("finish", "logs/converge_finish.json", "cosine 1e-4→1e-6 (finish)", "0.85"),
]


def _load_phase(path: str) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)["history"]


def _series(blocks: list[dict], picker) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for h in blocks:
        if h["block"] == 0:  # baseline rows have nan deltas
            continue
        v = picker(h)
        if v is None:
            continue
        xs.append(h["total_steps"])
        ys.append(v)
    return xs, ys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", action="append", default=None,
                   help="label:json_path:lr_desc  (repeatable; overrides defaults)")
    p.add_argument("--out", default=os.path.join(PLOTS_DIR, "convergence_full_history.png"))
    p.add_argument("--tau-param", type=float, default=0.015)
    p.add_argument("--tau-func", type=float, default=0.025)
    args = p.parse_args()

    if args.phase:
        phases = []
        for spec in args.phase:
            label, path, lr = spec.split(":", 2)
            phases.append((label, path, lr, None))
    else:
        phases = _DEFAULT_PHASES

    loaded = []
    for label, path, lr, shade in phases:
        if not os.path.exists(path):
            print(f"  (skipping missing phase {label}: {path})")
            continue
        loaded.append((label, _load_phase(path), lr, shade))
    if not loaded:
        print("No phase JSONs found.")
        return 1

    # phase boundaries (min/max total_steps of each phase's non-baseline blocks)
    bounds = []
    for label, blocks, lr, shade in loaded:
        steps = [h["total_steps"] for h in blocks if h["block"] > 0]
        if steps:
            bounds.append((label, min(steps), max(steps), lr, shade))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax = ax.ravel()

    def shade_phases(a):
        for label, lo, hi, lr, shade in bounds:
            if shade is not None:
                a.axvspan(lo - 5000, hi + 0, color=shade, alpha=0.5, zorder=0)

    # --- loss ---
    for label, blocks, lr, shade in loaded:
        xs, ys = _series(blocks, lambda h: h["loss"]["median"])
        ax[0].plot(xs, ys, "o-", label=f"{label} ({lr})")
    shade_phases(ax[0])
    ax[0].set_title("block median ELBO loss"); ax[0].set_xlabel("total steps")
    ax[0].set_ylabel("loss"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    # --- param rel-delta ---
    for name in STAGE_ORDER:
        xs, ys = [], []
        for _, blocks, _, _ in loaded:
            a, b = _series(blocks, lambda h: h["param"][name]["rel_delta"])
            xs += a; ys += b
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax[1].plot([xs[i] for i in order], [ys[i] for i in order],
                   "o-", color=_COLORS[name], label=_SHORT[name])
    ax[1].axhline(args.tau_param, ls="--", color="k", alpha=0.6, label=f"gate {args.tau_param}")
    shade_phases(ax[1])
    ax[1].set_yscale("log"); ax[1].set_title("per-stage PARAM rel-Δ  (plateau ↓)")
    ax[1].set_xlabel("total steps"); ax[1].set_ylabel("‖Δθ‖/‖θ‖")
    ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8)

    # --- func rel-delta ---
    for name in STAGE_ORDER:
        xs, ys = [], []
        for _, blocks, _, _ in loaded:
            a, b = _series(blocks, lambda h: h["func"][name]["rel_delta"])
            xs += a; ys += b
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax[2].plot([xs[i] for i in order], [ys[i] for i in order],
                   "o-", color=_COLORS[name], label=_SHORT[name])
    ax[2].axhline(args.tau_func, ls="--", color="k", alpha=0.6, label=f"gate {args.tau_func}")
    shade_phases(ax[2])
    ax[2].set_yscale("log"); ax[2].set_title("per-stage FUNCTIONAL rel-Δ across moment space  (plateau ↓)")
    ax[2].set_xlabel("total steps"); ax[2].set_ylabel("‖ΔF‖/‖F‖")
    ax[2].grid(alpha=0.3, which="both"); ax[2].legend(fontsize=8)

    # --- func magnitude ---
    for name in STAGE_ORDER:
        xs, ys = [], []
        for _, blocks, _, _ in loaded:
            a, b = _series(blocks, lambda h: h["func"][name]["mean_mag"])
            xs += a; ys += b
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax[3].plot([xs[i] for i in order], [ys[i] for i in order],
                   "o-", color=_COLORS[name], label=_SHORT[name])
    shade_phases(ax[3])
    ax[3].set_title("per-stage functional magnitude  (still rising = still building)")
    ax[3].set_xlabel("total steps"); ax[3].set_ylabel("mean |signature|")
    ax[3].grid(alpha=0.3); ax[3].legend(fontsize=8)

    # annotate phase boundaries
    for label, lo, hi, lr, shade in bounds:
        for a in ax:
            a.axvline(lo - 5000, color="gray", ls=":", alpha=0.5)

    fig.suptitle("Prior-flow per-stage convergence — full history (fixed-LR plateau → cosine-decay finish)",
                 fontsize=14)
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(args.out, dpi=130); plt.close(fig)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
