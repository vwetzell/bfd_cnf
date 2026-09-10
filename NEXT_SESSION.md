# Next session prompt

Paste the block below to start.  `GUIDING_PRINCIPLES.md` is the standing
reference for what counts as evidence and in what order to spend effort -- read
it first.  The previous prompt (the gauss2_v3d 500k run, now DONE) is preserved
as `NEXT_SESSION_20260906_500k.md`; its Traps and State sections still apply and
are condensed below.

---

Branch `feat/centroid-shear-conditioning`, repo `bfd_cnf` (sim repo
`../bfd_cnf_imsims`).

**Task: cut JAX compilation cost without changing a single output bit.**

**Start by searching the web for current best practice on JAX/XLA compilation.**
This codebase was written without that reference and the numbers below suggest
it is leaving a lot on the floor.  Look for prevailing wisdom on: reducing the
number of compiled executables, `jax.jit` donation and `static_argnums`,
avoiding retracing from varying shapes, `eqx.filter_jit` idioms, AOT
(`.lower().compile()`), `jax.lax.scan`/`map` versus Python loops, and the
persistent compilation cache.  Prefer primary sources (JAX docs, XLA docs,
Equinox docs, JAX GitHub issues) over blog posts, and say which claims you
verified by measurement here rather than by reading.

You are explicitly allowed to **rework the flow's structure and retrain** if
that is what cuts compile time.  Retraining is on the table; changing the
numbers is not.

## THE MEASUREMENTS -- do not re-derive these

Fit over four runs spanning 1000x in N (`logs/ovh_*.log`, `logs/gx_a05.log`,
the 500k run):

    wall = 575 s + 26.3 ms x N_target-arms

Residuals +/-30 s; the 500k point lands within 1 s.  **Compile is 80-85% of
that 575 s** -- established by the persistent-cache experiment, not inferred.

| run | target-arms | effective ms/target-arm |
|---|---|---|
| N=256 | 512 | 1137 |
| N=2000 | 4,000 | 178 |
| N=8000 | 16,000 | 60.7 |
| 500k | 507,722 | 27.5 |
| asymptote | - | **26.3** |

**Persistent cache works: 4.6x on small runs** (`logs/cache*_all.log`).

| | wall | entries |
|---|---|---|
| no cache (x3) | 582 / 647 / 650 s | - |
| cold (populating) | 462 s | 602 |
| warm (still populating) | 375 s | 1203 |
| **warm2 / warm3** | **139 / 144 s** | 1203, **no growth** |

Keys ARE stable; it needs TWO priming runs (first pass compiles the plus arm,
second picks up the minus arm and the selection-terms path).  Cached fixed cost
is ~127 s against ~575-640 s.  Enable with

    set -gx JAX_COMPILATION_CACHE_DIR ~/.cache/jax     # fish

**GATE: a bit-identity check was still running when this was written.**
`logs/bitcmp_all.log` compares `pqr/bit_nocache.npz` against `pqr/bit_cached.npz`
element-for-element at N=2000.  Read it first.  All printed numbers already
matched across no-cache/cold/warm at N=256 -- including `P_s`, `Q_s`, `R_s`, both
windowed m1s, the unwindowed m1, and the zero-weight counts.  If the arrays
differ, the cache is off for paired science regardless of the speedup.

## THE SMELL: 1203 cache entries for ONE run

That is the headline. A program with a handful of distinct computations should
compile a handful of executables, not twelve hundred.  Every one of those is a
separate XLA compilation.  Candidate causes, in the order worth checking:

1. **`pqr_streamed` builds its `eqx.filter_jit` wrappers INSIDE the call**
   (`dev/bench_pqr.py:99-102` documents this) -- so every call is a fresh jit
   cache entry and pays full compile.  `bias.py` calls it once per arm.
2. **Unequal last chunk.**  `selection_terms`' docstring says chunks are
   accumulated "weighted by chunk size, so an unequal last chunk is handled
   correctly".  A different trailing shape is a different compilation.  Padding
   to a fixed shape is the standard fix.
3. `_merge_finish`'s un-jackknifed fallback path (`[[road-to-1e-3]]`: it fires
   when there is a single chunk) is a second trace of the same region.
4. `--gauge auto`'s per-target `_blend_lambda` fit and the `safe_point` /
   `in_domain` masking may be introducing shape- or branch-dependent traces.
5. The 65,536-draw guard pilot is its own compilation.

Measure before fixing: dump the compilation log
(`JAX_LOG_COMPILES=1`, or `jax.log_compiles(True)`) and count what is actually
being compiled and why.  That is the first experiment of the session.

## THE VERIFICATION PROTOCOL -- output must not change

Anything you do is only acceptable if the outputs are unchanged.  Use, in
increasing order of strictness:

1. **Zero-weight counts** are the cheapest tripwire.  They are a THRESHOLD test
   on importance weights, so they move on perturbations far below printed
   precision -- this is exactly what `[[tf32-breaks-paired-runs]]` broke.
2. All printed numbers: `P_s`, `Q_s`, `R_s`, windowed uncorrected/corrected m1,
   unwindowed m1.
3. **Bit-identical `--save-pqr` arrays.**  `plus_q`, `plus_r`, `minus_q`,
   `minus_r` compared with `np.array_equal(..., equal_nan=True)`.  This is the
   gate.  `/tmp/.../scratchpad/bitcmp.sh` is the harness; copy it.

If you retrain a flow, bit-identity against the old flow is impossible by
construction -- then the standard is a PAIRED comparison via `bias.compare()`
(paired bar ~170x tighter than unpaired, measured) showing the difference is
consistent with zero, plus unchanged behaviour on the checks above.

## Related findings from 2026-09-07, already paid for

* **`dev/bench_pqr.py` IS BROKEN -- do not trust it.**  Its comment says it runs
  at n and 2n and differences to remove compile, but the code calls
  `measure(m)` twice at the SAME n, so compile stays in its "MARGINAL".  It
  reports 370.9 ms/target/arm against the production path's 26.3.  It also
  self-reports `peel: centroid layer ABSENT` -- which is true, but true of
  production too, so that is not the discrepancy.  Fixing this script is a
  reasonable first task; it is the natural harness for this whole session.
* **The centroid peel never fires.**  Every trained flow has
  `cond_shape=(5,)` and `bias.py:935` refuses to peel it.  Forcing the peel
  (scratchpad `peel_run.py`) gives **m1 = -1.00006 +/- 0.00009**: Q inflates to
  ~2.5e9 and R11 to ~9e14, and the inflated values are ARM-IDENTICAL (median
  `|q1_plus - q1_minus|` is exactly 0), so the response cancels and
  `ghat_plus = ghat_minus`.  The guard is load-bearing -- LEAVE IT.  New clue
  for whoever root-causes it: the contamination does not depend on the target
  data at all, which points inside
  `Invert(Chain(bij[2:]).merge_chains())`.  Worth ~30% of MARGINAL cost
  (842 s vs 971 s at N=8000), so it is a correctness curiosity, not a
  performance item.
* **Never benchmark this pipeline below ~50,000 target-arms.**  Under that you
  are measuring XLA, not the estimator.  Both the "bench says 370 ms" and the
  "small runs say 60 ms" puzzles were the same fixed-cost artifact.

## STILL OPEN -- unfixed bugs carried forward

1. **`dev/window_scan.py` counts `n_ns` over the PQR file's rows, not the
   catalog** (line 216 point estimate, line 240 bootstrap).  On a pre-filtered
   PQR it under-counts by 64%, moving m1 by **0.015** while leaving the error
   bar unchanged -- so it is invisible in the bars.  This is trap #1 from the
   old prompt; `bias.py` was fixed for it, `window_scan.py` never was.  **FIX:**
   have `save_pqr` write the per-target `prefilter_w` (`bias.py:3079` already
   computes it) and have `window_scan.py` use `w[~sp].sum()` in both places,
   defaulting to ones when the key is absent.  Weights, not a scalar
   `n_out_full`, because a scan re-solves at DIFFERENT windows.  ~5 lines.
2. The corrected 500k answer, re-solved by hand with the right `n_ns`:
   **m1 = -0.0030 +/- 0.0031** (galaxy 0.00306, selection MC 0.00036),
   consistent with zero.  The `+0.0121` in `logs/resolve_g2v3d_500k_s*.log` is
   the bug above, NOT a result.
3. `[[road-to-1e-3]]` was updated 2026-09-07: truth.py is UNBLOCKED but is
   noiseless/centred only, so it cannot reference the recentring that gates
   1e-3.  `sigma(m1) ~ 1.03/sqrt(N_in-window)`; 1e-3 needs ~1.1M in-window /
   ~4.7M catalog (~36 GPU-h).
4. `--gauge prior` and `--alpha 1.0` are both REJECTED, reconfirmed on
   gauss2_v3d 2026-09-07 -- see `[[r-is-an-is-artifact]]`'s last section.
   Gauge prior is 2.0-2.3x faster and gives **unwindowed m1 = -1.10 +/- 0.007**.
   Do not revisit as a speed lever.

## Traps -- all paid for already

* **ONE JAX GPU JOB AT A TIME.**  A second launch while the first finishes is an
  instant OOM.
* **Launch long runs with `setsid`.**  Poll a log file, never `pgrep -f`
  (it matches the waiter's own command line and hangs).
* **`--batch-budget` bounds MEMORY independently of `--samples`.**  Omitting it
  gives a LARGER batch and the card OOMs at 13 GiB.
* **`--floor-eps 1e-3 --window-guard 100` on every gauss2 run.**
* **The Bash tool runs zsh, which does NOT word-split `$var`.**  `set -- $spec`
  fails silently in a direct tool call; it is fine inside `bash foo.sh`.
* `/usr/bin/time` is NOT installed -- use the shell's `$SECONDS`.
* **Quote no `m1` without its window AND its draw count.**
* **Read the UNWINDOWED m1 first.**  A windowed-only comparison has no power:
  the window excises exactly where a broken estimator fails.
* An identical rebuild moves noisy `m1` by sd 0.0008; no-cache wall time varies
  ~10% run to run.

## State

Catalogs (`../bfd_cnf_imsims/data`): `moments_gauss2_fwd_{g2v3,g2v3d}.fits`,
`targets_{g2v3,g2v3d}_{g0,g1p02,g1m02}_500k.fits`,
`copies_gauss2_fwd_{g2v3,g2v3d}.fits`.  bulgedisc has only 200k targets.

Flows: `flows/{bulk,shear,centroid}_{g2v3,g2v3d}.eqx`; bulgedisc's are
`{bulk,shear}_v10` + `centroid_v11`.

PQR: `pqr/g2v3d_500k.npz` is the headline run (253,861 pre-filtered rows).
`pqr/gx_{a05,p05,a10,p10,peel}.npz` are the 2026-09-07 gauge/alpha/peel probes
at N=8000; `pqr/bit_{nocache,cached}.npz` are the cache bit-check pair.
Logs: `logs/{gx_*,ovh_*,cache*,bitcmp,peel}*.log`.

Nothing is committed.  The working tree carries several sessions of changes.

---

# SESSION RESULTS 2026-09-07 (compile-cost task) -- DONE

## 1. THE GATE'S PREMISE IS WRONG -- bit-identity is unachievable

`logs/bitcmp_all.log` returned "DIFFERS -- do NOT use the cache". **That
verdict is an artifact of a missing control.**  I ran nocache-vs-nocache
(`pqr/det_a.npz` vs `pqr/det_b.npz`, same code, same seeds, cache off both):

| pair | plus_q max\|d\| | minus_q max\|d\| | dm1 unwind | dm1 wind |
|---|---|---|---|---|
| nocache vs **nocache** | 3.662e-04 | 5.646e-04 | **-2.95e-06** | 1.35e-08 |
| nocache vs cached | 3.663e-04 | 5.646e-04 | -3.45e-07 | 1.90e-09 |

The `q` maxima agree to four digits.  **`bias.py` is not run-to-run
bit-reproducible on this GPU**, and the persistent cache perturbs m1 an ORDER
OF MAGNITUDE LESS than a plain rerun does.

* **The cache is SAFE. Enable it** -- `set -gx JAX_COMPILATION_CACHE_DIR
  ~/.cache/jax`.  4.6x on small runs; needs two priming runs.
* **Strike gate #3 (bit-identical `--save-pqr` arrays) from the protocol.**
  The right standard is a TOLERANCE: m1 to `<< 1e-4`, with **3e-6 the measured
  run-to-run floor at N=2000**.  Zero-weight counts (gate #1) are still a fine
  tripwire but will now trip on noise -- treat a move as signal only if m1 does.

## 2. Compile cost: it is ONE kernel, not 1203 entries

Measured with `JAX_LOG_COMPILES=1` / `JAX_EXPLAIN_CACHE_MISSES=1`.
**The "1203 entries" smell is the wrong metric** -- ~200 of them are eager
op-by-op trivia (`convert_element_type` 46, `stage` 33, ...) worth a few
seconds total.  Compile TIME is one kernel: `jit(one)` / `jit(one_b)`.

The cause is the **short trailing chunk** = a second shape = a second full
compile, once per arm.  Fixed in two places:

| path | before | after |
|---|---|---|
| non-streamed (`_over_targets`) | 119.2 s | **74.8 s (-37%)** |
| **streamed (production)** | 280.6 s | **135.1 s (-52%)** |

`one_b` 4 compiles -> 2.  **GPU wall at N=2000: 491 s vs the 582-650 s
no-cache baseline (~20-25%).**  Verified against the tolerance above:
padded vs det_a = **6.07e-07**, det_b = 3.55e-06 -- i.e. INSIDE the
run-to-run scatter.  Check: `tests/test_over_targets.py`.

Key mechanism (`_fixed_batch`): pad at the **jit boundary only**.  Padding
`m_b` itself would reseed `mixture_draws` for the trailing batch and move
those targets for real.

## 3. Corrections to the prompt's candidates

* **#1 is right, but not for the stated reason.** `eqx.filter_jit` (0.13.8)
  DOES key its `jax.jit` on the function's qualname -- but it passes the
  function as a **static arg hashed by identity**, so a rebuilt wrapper
  recompiles anyway.
* **Memoizing `psi` across arms is a NULL** (74.8 -> 77.5 s, noise).  The
  per-arm recompile is `batched` itself being rebuilt, not `psi`.
* **#4 (`_blend_lambda`) is real**: 43.7 s of 280 s, halved by the same fix.
* `models/bijections.py:1080` -- `step` redefined per call, so `lax.scan`
  retraces its body.  JAX flags it explicitly.  TRACING cost only (17 s of
  75 s); left alone.

## 4. `window_scan.py` n_ns -- FIXED

`save_pqr` now writes per-target `prefilter_w`; `window_scan.py` uses
`w[~sp].sum()` in both the point estimate and the bootstrap, falling back to
ones when the key is absent.  Verified on `pqr/g2v3d_500k.npz` (weights
reconstructed from the log's pad 20% / sample 15%; `in_pad` = 210656, matching
the log to the galaxy):

| | corrected m1 | +/- gal |
|---|---|---|
| fallback (= old behaviour) | +0.01223 | 0.00310 |
| weighted | **-0.00276** | 0.00304 |

Rows out of window 136,551 vs weighted 381,379 -- **under-count of 64.2%**,
with the bar essentially unmoved.  **The 500k headline is m1 = -0.0028 +/-
0.0030.**  The `+0.0121` in `logs/resolve_g2v3d_500k_s*.log` is the bug.

## 4b. TRAP ADDED -- do not run JAX on CPU beside a GPU job

I used `JAX_PLATFORMS=cpu` to run probes alongside the GPU runs.  **It crashed
the machine**: swap 48 GB -> 16 GB from ~12:06, page-allocation stalls, reboot
at 12:16.  The CPU backend puts in HOST RAM what the GPU keeps in VRAM, and
XLA's CPU compile of `one_b`/`one_k` is memory-hungry on its own (25-38 s
each).  ONE JAX job at a time means CPU too.  Queue probes behind the GPU run.

No result in this file is affected -- every measurement completed and was
written to disk before the crash, and all of `pqr/det_a.npz`, `pqr/det_b.npz`,
`pqr/pad_a.npz` and the edits survived.

## 5. Left undone

* **Test suite not run** (interrupted).  Re-run `pytest tests/ -q` before
  committing.  Gauge `prior` and `kernel` were smoke-tested clean.
* Last per-arm recompile (`one_b` x2, ~26 s CPU) needs `batched` memoized
  ACROSS `pqr_streamed` calls.  Landmine: closed-over arrays are static, so a
  wrong cache key would silently bake a stale `cinv`.  Not attempted.
* Cached re-solves still import JAX and fight a running GPU job -- use
  `JAX_PLATFORMS=cpu`.
* Nothing committed.
