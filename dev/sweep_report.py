"""Tabulate `dev/sweep_capacity.jsonl`.

Two things this has to get right, both learned the hard way in the run it was
written for:

  * Compare PAIRED, seed by seed, never a single point against a multi-seed
    mean.  Every capacity axis in this sweep looked like a win at seed 0 and
    evaporated when the same architecture was retrained at seeds 1 and 2.
  * Never compare across shear learning rates.  lr moves RMS(m1) 5x, which is
    larger than any architectural effect measured, so a cross-lr comparison
    reads the optimiser and calls it the model.
"""
import json
import sys
from collections import defaultdict

import numpy as np

ROWS = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1
                                    else "dev/sweep_capacity.jsonl")]


def ok(r):
    return r.get("converged", r["shear_val_nll"] < 2 * r["bulk_val_nll"])


def lr(r):
    return r.get("shear_lr", 3e-3)


print("ALL RUNS")
print(f"{'config':>16s} {'L':>3s} {'W':>4s} {'D':>2s} {'params':>8s} {'s_lr':>6s} "
      f"{'bulk nll':>9s} {'m1(all)':>9s} {'RMS':>7s} {'ptp':>7s} {'min':>5s}")
for r in ROWS:
    h = r["heldout"]
    body = (f"{h['all']:+9.4f} {h['rms']:7.4f} {h['ptp']:7.4f}" if ok(r)
            else f"{'DIVERGED':>9s} {'':>7s} {'':>7s}")
    print(f"{r['name']:>16s} {r['layers']:3d} {r['width']:4d} {r['depth']:2d} "
          f"{r['bulk_params']:8d} {lr(r):6.0e} {r['bulk_val_nll']:9.4f} {body} "
          f"{r['minutes']:5.1f}")

# ---- paired comparison, at the baseline lr only -------------------------
ARMS = {"base": (8, 64, 2), "width128": (8, 128, 2), "depth4": (8, 64, 4)}
by_arm = defaultdict(dict)
for r in ROWS:
    if not ok(r) or lr(r) != 3e-3 or r["bulk_steps"] != 4000:
        continue
    for arm, dims in ARMS.items():
        if (r["layers"], r["width"], r["depth"]) == dims:
            by_arm[arm][r["seed"]] = r["heldout"]["rms"]

seeds = sorted(set(by_arm["base"]))
print(f"\n\nPAIRED vs base, same seed, shear lr 3e-3 only")
print(f"{'arm':>10s} " + " ".join(f"{'s' + str(s):>9s}" for s in seeds) +
      f" {'mean':>9s} {'sd':>8s} | {'mean d':>9s} {'sem':>8s} {'sigma':>6s}")
for arm in ARMS:
    if arm not in by_arm:
        continue
    v = [by_arm[arm].get(s, np.nan) for s in seeds]
    row = f"{arm:>10s} " + " ".join(f"{x:9.4f}" for x in v)
    row += f" {np.nanmean(v):9.4f} {np.nanstd(v, ddof=1):8.4f}"
    if arm != "base":
        d = np.array(v) - np.array([by_arm["base"][s] for s in seeds])
        sem = np.nanstd(d, ddof=1) / np.sqrt(np.sum(~np.isnan(d)))
        row += f" | {np.nanmean(d):+9.4f} {sem:8.4f} {abs(np.nanmean(d))/sem:6.1f}"
    print(row)

# ---- the confound that dominates everything -----------------------------
print(f"\n\nSHEAR LR, at fixed architecture (8/64/2, seed 0 unless noted)")
print(f"{'lr':>8s} {'RMS':>18s}")
for want in (3e-3, 1e-3, 3e-4):
    v = [r["heldout"]["rms"] for r in ROWS
         if ok(r) and lr(r) == want and (r["layers"], r["width"], r["depth"])
         == (8, 64, 2) and r["bulk_steps"] == 4000]
    print(f"{want:8.0e} {np.mean(v):9.4f}"
          + (f" +/- {np.std(v, ddof=1):.4f} (n={len(v)})" if len(v) > 1 else ""))

# ---- diagnostics vs the thing we care about -----------------------------
good = [r for r in ROWS if ok(r)]
print(f"\n\nDYNAMIC RANGE across all converged runs (n={len(good)})")
for key, lab in (("bulk_val_nll", "bulk val nll"), ("shear_val_nll", "shear val nll")):
    v = [r[key] for r in good]
    print(f"  {lab:>14s}  {min(v):9.4f} .. {max(v):9.4f}   span {max(v)-min(v):.4f}")
v = [r["heldout"]["rms"] for r in good]
print(f"  {'RMS(m1)':>14s}  {min(v):9.4f} .. {max(v):9.4f}   "
      f"span {max(v)/min(v):.1f}x")

print(f"\n\nheld-out m1 per Mr/Mf octile (converged runs)\n{'config':>16s} " +
      " ".join(f"{b[0]:>8.2f}" for b in good[0]["heldout"]["bins"]))
for r in good:
    print(f"{r['name']:>16s} " +
          " ".join(f"{b[1]:+8.4f}" for b in r["heldout"]["bins"]))
