"""Train until a held-out objective stops improving, instead of a fixed step count.

Every trainer here used `optax.cosine_decay_schedule(lr, steps)`: the step
count was a guess, and a cosine tail flattens the loss whether or not the fit
has converged, so a flat tail proved nothing.  `fit` replaces the schedule with
a linear warmup to a constant LR plus reduce-on-plateau, and stops on a
statistical test rather than a budget:

  every `check_every` steps the trainer's `val_fn` returns the PER-POINT
  objective on a fixed evaluation set; the mean improvement over the reference
  (the last checkpoint that significantly improved) is tested PAIRED, point by
  point, against 2x its standard error.  `patience` checks without a
  significant improvement cut the LR by `factor`; a plateau after `n_decays`
  cuts stops training.  The best-mean checkpoint is returned.

The paired test is what makes the stop scale-free: it needs no per-trainer
tolerance, only the evaluation set's own point-to-point scatter.
"""
from __future__ import annotations

import numpy as np


def fit(run_chunk, carry, params_of, val_fn, *, chunk=500, check_every=5000,
        warmup=2000, patience=2, factor=0.3, n_decays=3, max_steps=3_000_000,
        min_steps=0, report=None):
    """`run_chunk(carry, lr_mult, n) -> (carry, aux)` runs `n` steps with the
    optimiser's updates scaled by `lr_mult` (a traced scalar, so no recompile);
    `params_of(carry)` is what gets validated and returned (the EMA params when
    the trainer keeps one); `val_fn(params)` is the per-point objective, shape
    (n_eval,).  `report(step, aux)` prints the trainer's own per-chunk line.

    Returns `(best_params, history)`; history rows are
    (step, lr_mult, val mean, improvement vs reference, its SE)."""
    # The reference is the FIRST check, not the init: an untrained flow's
    # per-point loss has 1e5-nat outliers that swamp every later SE.
    ref, best_params, best_mean = None, None, np.inf
    print(f"converge: check every {check_every}, patience {patience}, factor {factor}, "
          f"{n_decays} decays", flush=True)
    lr, stall, decays, step, hist = 1.0, 0, 0, 0, []
    while step < max_steps:
        for _ in range(max(check_every // chunk, 1)):
            mult = lr * min(1.0, (step + chunk) / warmup) if warmup else lr
            carry, aux = run_chunk(carry, np.float32(mult), chunk)
            step += chunk
            if report:
                report(step - 1, aux)
        p = params_of(carry)
        cur = np.asarray(val_fn(p), np.float64)
        # A lone tail point can go non-finite between checks (seen on bulk at
        # step 20000 with a healthy training NLL); drop it from the test and
        # report it, but stop if it is more than a stray.
        n_bad = int((~np.isfinite(cur)).sum())
        if n_bad > 0.01 * len(cur):
            print(f"converge: step {step}  {n_bad} non-finite val points -- stopping, "
                  f"returning best", flush=True)
            break
        if ref is None:
            ref = cur
        ok = np.isfinite(cur) & np.isfinite(ref)
        # Winsorised at 0.1/99.9%: a single tail point must not set the SE.
        d = ref[ok] - cur[ok]
        d = np.clip(d, *np.percentile(d, [0.1, 99.9]))
        imp, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        sig = imp > 2 * se
        cm = cur[np.isfinite(cur)].mean()
        if cm < best_mean and not n_bad:
            best_params, best_mean = p, cm
        if sig:
            ref, stall = cur, 0
        else:
            stall += 1
        hist.append((step, lr, cm, imp, se))
        print(f"converge: step {step}  lr x{lr:.4g}  val {cm:.6f}  "
              f"d vs ref {imp:+.2e} +/- {se:.1e}  {'IMPROVED' if sig else f'stall {stall}/{patience}'}"
              + (f"  ({n_bad} non-finite)" if n_bad else ""), flush=True)
        if stall >= patience and step >= min_steps:
            if decays == n_decays:
                print(f"converge: CONVERGED at step {step}  best val {best_mean:.6f}", flush=True)
                break
            lr, stall, decays = lr * factor, 0, decays + 1
            print(f"converge: plateau -> lr x{lr:.4g} (decay {decays}/{n_decays})", flush=True)
    else:
        print(f"converge: NOT CONVERGED -- hit max_steps {max_steps}", flush=True)
    return best_params, hist


if __name__ == "__main__":
    # Self-check on a noisy quadratic: must converge, and to near the optimum.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4096, 3)); w_true = np.array([1.0, -2.0, 0.5])
    y = X @ w_true + 0.1 * rng.normal(size=4096)
    Xv = rng.normal(size=(2048, 3)); yv = Xv @ w_true + 0.1 * rng.normal(size=2048)

    def run_chunk(w, mult, n):
        for _ in range(n):
            i = rng.integers(0, 4096, 64)
            w = w - 0.05 * mult * 2 * X[i].T @ (X[i] @ w - y[i]) / 64
        return w, None

    w, h = fit(run_chunk, np.zeros(3), lambda w: w, lambda w: (Xv @ w - yv) ** 2,
               chunk=100, check_every=500, warmup=200)
    assert np.abs(w - w_true).max() < 0.01, w
    assert h[-1][1] < 1.0 and len(h) < 200
    print("ok", w)
