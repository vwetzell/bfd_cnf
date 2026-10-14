"""Train until a held-out objective stops improving, instead of a fixed step count.

Every trainer here used `optax.cosine_decay_schedule(lr, steps)`: the step
count was a guess, and a cosine tail flattens the loss whether or not the fit
has converged, so a flat tail proved nothing.  `fit` replaces the schedule with
a linear warmup to a constant LR plus reduce-on-plateau, and stops on a
statistical test rather than a budget:

  every `check_every` steps the trainer's `val_fn` returns the PER-POINT
  objective on a fixed evaluation set; the improvement over the reference (the
  last checkpoint that significantly improved) is tested PAIRED, point by point,
  with a Wilcoxon signed-rank test at one-sided z > 2.  Ranks, not a mean: a
  handful of junk eval points (tiny faint templates, misfits 1e2-1e3x the
  median) swung the old winsorised mean's SE ~10x and stopped runs early
  (2026-10-13, desv centroid and shear stages).  `patience` checks without a
  significant improvement cut the LR by `factor`; a plateau after `n_decays`
  cuts stops training.  The LAST params are returned, never a best-of-checkpoints pick:
  picking among checkpoints by a noisy validation number is not reproducible on a new
  population (2026-10-13: it shipped 25-35k-step snapshots of 60k-step centroid runs).

The paired test is what makes the stop scale-free: it needs no per-trainer
tolerance, only the evaluation set's own point-to-point scatter.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def wilcoxon_z(d):
    """Signed-rank statistic of paired improvements `d` as a normal z (ties averaged, zeros dropped)."""
    d = d[d != 0]
    n = len(d)
    if n < 2:
        return 0.0
    rk = rankdata(np.abs(d))
    _, cnt = np.unique(rk, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (cnt ** 3 - cnt).sum() / 48
    return float((rk[d > 0].sum() - n * (n + 1) / 4) / np.sqrt(var))


def fit(run_chunk, carry, params_of, val_fn, *, chunk=500, check_every=5000,
        warmup=2000, patience=2, factor=0.3, n_decays=3, max_steps=3_000_000,
        min_steps=0, report=None, on_check=None, state_out=None):
    """`run_chunk(carry, lr_mult, n) -> (carry, aux)` runs `n` steps with the
    optimiser's updates scaled by `lr_mult` (a traced scalar, so no recompile);
    `params_of(carry)` is what gets validated and returned (the EMA params when
    the trainer keeps one); `val_fn(params)` is the per-point objective, shape
    (n_eval,).  `report(step, aux)` prints the trainer's own per-chunk line.

    Returns `(last_params, history)`; history rows are
    (step, lr_mult, val mean, median improvement vs reference, its signed-rank z).
    `on_check(step, params)` runs at every check; `state_out` (a dict) gets the final
    `carry`, `lr` and `step`, so a caller can keep training past the stop."""
    # The reference is the FIRST check, not the init: an untrained flow's
    # per-point loss has 1e5-nat outliers that swamp every later SE.
    ref = None
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
        if on_check:
            on_check(step, p)
        cur = np.asarray(val_fn(p), np.float64)
        # A lone tail point can go non-finite between checks (seen on bulk at
        # step 20000 with a healthy training NLL); drop it from the test and
        # report it, but stop if it is more than a stray.
        n_bad = int((~np.isfinite(cur)).sum())
        if n_bad > 0.01 * len(cur):
            raise RuntimeError(f"converge: step {step}  {n_bad} non-finite val points")
        if ref is None:
            ref = cur
        ok = np.isfinite(cur) & np.isfinite(ref)
        d = ref[ok] - cur[ok]
        z = wilcoxon_z(d)
        sig = z > 2
        imp = np.median(d)
        cm = cur[np.isfinite(cur)].mean()
        if sig:
            ref, stall = cur, 0
        else:
            stall += 1
        hist.append((step, lr, cm, imp, z))
        print(f"converge: step {step}  lr x{lr:.4g}  val {cm:.6f}  "
              f"median d vs ref {imp:+.2e}  z {z:+.1f}  {'IMPROVED' if sig else f'stall {stall}/{patience}'}"
              + (f"  ({n_bad} non-finite)" if n_bad else ""), flush=True)
        if stall >= patience and step >= min_steps:
            if decays == n_decays:
                print(f"converge: CONVERGED at step {step}  final val {cm:.6f}", flush=True)
                break
            lr, stall, decays = lr * factor, 0, decays + 1
            print(f"converge: plateau -> lr x{lr:.4g} (decay {decays}/{n_decays})", flush=True)
    else:
        print(f"converge: NOT CONVERGED -- hit max_steps {max_steps}", flush=True)
    if state_out is not None:
        state_out.update(carry=carry, lr=lr, step=step)
    return params_of(carry), hist


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
    from scipy.stats import wilcoxon
    dd = rng.normal(0.1, 1, 500)
    from scipy.stats import norm
    p = wilcoxon(dd, alternative="greater", method="approx", correction=False).pvalue
    assert abs(norm.sf(wilcoxon_z(dd)) - p) < 1e-9, (norm.sf(wilcoxon_z(dd)), p)
    print("ok", w)
