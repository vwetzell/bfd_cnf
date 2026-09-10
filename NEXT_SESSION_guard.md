# NEXT SESSION — the guard truncation is the binding error, not the statistics

Written 2026-09-09. **Supersedes `NEXT_SESSION_1e3.md`**, whose premise (that
σ(m1) = 1e-3 is the thing to buy) is now measurably wrong: the selection
guard moves the central value by 4.2e-3, four times the statistical bar that
plan spends 26 GPU-h to reach.

---

## TL;DR

`gauss2_v3e` slice 0 (920k targets, 215,715 in window) gives

| selection term | m1 |
|---|---|
| unguarded, 2^26, 6 seeds | **−0.00869 ± 0.00049** |
| guard 100, 2^26, 4 seeds | **−0.01290 ± 0.00069** |
| guard 100, 2^28, 2 seeds | **−0.01227 ± 0.00055** |

plus a galaxy bootstrap bar of ±0.0022 on all of them.  So the honest number
is **m1 ≈ −0.010 ± 0.002 (stat) ± 0.002 (truncation)** — a real ~1% negative
bias at 4–6σ, consistent with the centroid spin-2 floor `[[road-to-1e-3]]`
names as the blocker.  It is NOT a σ = 1e-3 measurement and should not be
quoted as one.

**Do not run slices 1–4 until the guard is settled.**  They buy σ 0.0022 →
0.0010 under a 0.0042 systematic.

---

## The one open question

`m1` slides monotonically with the guard factor and has not plateaued:

| guard | m1 (2^26) | seeds |
|---|---|---|
| 30 | −0.01540 | 2 |
| 100 | −0.01290 | 4 |
| 300 | −0.01221 | 1+ |
| 1000 | *running* | 2 |
| 3000 | *running* | 2 |
| ∞ (off) | −0.00869 | 6 |

`dev/rs_guard_scan.sh` was running at the time of writing; results land in
`logs/rs_g{30,300,1000,3000}_26_s{0,7}.log` and `logs/rs_guard_scan.outer.log`.

**Read the shape, not one value.**  If m1 flattens over a decade of guard,
quote the plateau and the residual spread as the truncation systematic.  If it
slides all the way from 30 to ∞, the guard is choosing the answer and no
plateau exists to quote — go upstream instead (next section).

### The sign is backwards and nobody has explained it

Truncating a heavy-tailed mean should LOWER it.  This guard RAISES both
`R_s11` (0.0619 → 0.0727) and `P_s` (0.2326 → 0.2416).  So it is not clipping
a fat positive tail; it is cutting draws whose `R + QQ^T` is large and
NEGATIVE.  "Drop the biggest |Q|, |R|" may simply be the wrong cut.  Anyone
picking this up should look at the sign and size distribution of the dropped
draws' `R + QQ^T` before defending any particular factor.

### Upstream fix, if there is no plateau

`[[rs-tail-is-unphysical-draws]]` already root-caused this: `in_support` is a
BOX, the manifold is CURVED, and the tail draws invert to no galaxy.  The
2^28 unguarded failure is a clean example — one draw in 268M took 100% of
`R_s11` at **Mf = 1999, Mr/Mf = 1.741**, i.e. BELOW both window cuts (2500 and
2.2).  It contributes only through the smoothed window probability `F`, with
an unphysical score.  A support test that follows the manifold would remove
those draws on physics rather than on magnitude, and would not need a factor.

---

## What changed in the code this session

All four are committed to the working tree, none are committed to git.

1. **`bias.py` — `--target-offset`** (default 0).  `slice(off, off + n)`, so a
   30 h run splits into resumable slices.  All three arms take the same slice,
   so the ± pairing holds.  `load_cov` asserts C_M is row-invariant, so
   nothing else in the path is index-dependent.
2. **`bias.py` — `CATALOGS["gauss2_v3e"]` + `TRAIN_DATA["gauss2_v3e"]**.  The
   train data deliberately points at `moments_gauss2_fwd_g2v3d.fits`:
   `RawMomentStandardize`'s `mean`/`std` are pytree leaves overwritten from the
   checkpoint, and a fresh render of the prior came out BYTE-IDENTICAL to
   g2v3d's, so this is the same population, not a transfer.
3. **`dev/window_scan.py` — `--window-guard`** (default 0.0, matching
   `bias.py`).  **This path never passed a guard.**  `bias.py:3368` passes
   `guard=a.window_guard` (the runs use 100); `window_scan.py` passed nothing,
   so EVERY offline re-solve in this project — including the 500k resolves —
   was a different, unguarded estimator from the in-run number.  Silently.
4. **`dev/window_scan.py` — `--draw-chunk`** (default 0 = old behaviour).
   2^28 used to OOM on ONE allocation: `base_dist.sample(key, (n,))` asks for
   10.02 GiB on a 16 GB card.  Chunked, it runs.  **Two levels are required**:
   the outer bounds the sample, the inner 16384 bounds the transform — feeding
   a whole draw-chunk into the vmapped transform replaces a 10 GiB failure
   with a 20 GiB one (measured).  Chunks use `jr.fold_in` per chunk because
   `sample` at a FIXED key NESTS: k calls with one key return k IDENTICAL
   blocks, which would have turned 2^28 draws into 2^24 repeated 16 times and
   reported a 4x-too-tight answer with no error.  The stream therefore differs
   from an unchunked run, and the terms cache name carries `_c<chunk>` and
   `_g<guard>` so the variants cannot collide.
5. **`dev/window_scan.py` — refuses a PQR with no `prefilter_w`** instead of
   silently substituting `ones()`.  `pqr/g2v3d_500k.npz` has no `prefilter_w`,
   which is why its resolve logs read +0.0121 against a true −0.0028.

New scripts: `dev/g2v3e_1e3.sh` (smoke/slice/null/run/cat/resolve),
`dev/pqr_cat.py` (+ `demo` self-check), `dev/rs_ladder_g2v3e.sh`,
`dev/rs_2p28.sh`, `dev/rs_guard.sh`, `dev/rs_guard_scan.sh`.

---

## What was measured, and what it settled

**Catalog.**  `targets_g2v3e_{g0,g1p02,g1m02}_4600k` rendered at SIGMA=1.86,
POP=gauss2_fwd, seed 1, 2.25 GB each.  Median flux S/N 9.7 against the 500k's
9.6.  `cov`/`cov_odd` bit-identical to g2v3d's, headers match, `Sigma_X`
identical, chart constants are literals.  `P_s` reproduces to four digits.
Nothing is stale; the two catalogs are the same population.

**Slice 0** (`pqr/g2v3e_s0.npz`, 467,697 integrated, 51.0%): uncorrected
m1 = +0.01442.  6.6 h.

**Draw-seed null** (same galaxies, `--draw-seed 4242`): 3.4e-4 uncorrected,
3.2e-4 corrected.  Kernel draw noise at `--samples 8192` is NOT a constraint.

**The uncorrected bootstrap bar in the logs understates by ~3.4x.**  It is
0.00067; a paired bootstrap that resamples all galaxies and re-derives each
arm's mask gives 0.00227, and 60 disjoint split-halves give 0.00265 — both
matching the CORRECTED bar of 0.00217.  Cause: `bias.bootstrap`'s no-`ns`
branch resamples within the selected subset, which assumes both arms select
the same galaxies.  **1.71% of the window (3,682 galaxies) is inside one arm's
window and outside the other's**, and those contribute shape noise the ±
pairing never cancels.  Quote the corrected-style bar, or per-slice logs read
as 5σ detections when they are 1.5σ.

**The 500k-vs-slice-0 gap is not a discrepancy.**  +0.02079 vs +0.01442 looks
like 6.4σ on the log's bars and is 1.7σ on honest ones.  No host-vs-device
anomaly; no control run needed.

**Two dead ends, recorded so nobody re-runs them.**  (a) I read three
single-seed `R_s11` values as a monotone drift; the 2^24 scatter is 10.6%, and
a 7-seed vs 6-seed comparison puts the 2^24→2^26 drift at 1.21σ with the sd
ratio at 2.13 against the 1/√N prediction of 2.00.  (b) I then concluded the
heavy tail was absent — wrong in the other direction: it is real, it just
needs 2^28 draws to surface, and it is invisible at 2^24/2^26.

---

## Do this next, in order

1. **Read the guard scan.**  `grep -E "===|nominal" logs/rs_guard_scan.outer.log`.
   Column order is `nominal n_in P_s R_s11 R_s12 tracels m1_unc m1_corr ±gal`
   — index 3 is `R_s11`, index 7 is `m1_corr`.  Getting this wrong produces
   nonsense that looks plausible; it happened once already.
2. **Decide plateau or no plateau** by the rule above.
3. **If plateau:** quote m1 = plateau ± 0.0022 (galaxy) ± (spread across the
   plateau).  Then, and only then, is it worth deciding whether slices 1–4 are
   worth 26 GPU-h — and they are only worth it if the truncation systematic
   came in under ~1e-3.
4. **If no plateau:** stop measuring and fix `in_support`.  Census the dropped
   draws first (sign and size of `R + QQ^T`, where they sit relative to the
   window) — `dev/rs_census.py` and `dev/clamp_census.py` already exist.
5. **Either way, write the result into `[[road-to-1e-3]]`** and correct
   `[[rs-tail-is-unphysical-draws]]` with the 2^28 evidence.

---

## Traps (all of these bit during this session)

* **ONE JAX job at a time, GPU *and* CPU.**
* **Do NOT set `JAX_PLATFORMS=cpu` on `window_scan`.**  `NEXT_SESSION_1e3.md`
  Step 5 says to; the CPU backend is ~20x slower and a 2^24 re-solve that
  takes 4 min on the GPU ran 5 h before I killed it.
* **Never edit a bash script while it is running** — bash re-reads the file
  and dies with `unexpected EOF`.
* **`setsid` + poll a LOG FILE or a bracketed `pgrep -f "foo[.]sh"`.**  A
  plain `pgrep -f` matches the waiter's own command line.  A watcher that
  polls `! pgrep window_scan` fires in the GAP between two sequential python
  invocations — watch the outer script, not the inner process.
* **A completion marker outside the `case` block** fires on the first
  sub-stage.  Three watchers reported "done" on a job that had not started.
* **`--batch-budget 65536` is mandatory on `bias.py`** or the card OOMs.
* `--floor-eps 1e-3`, `--window-size 2.2 3.2`, `--window-terms score`,
  `--gauge auto` — all mandatory, all previously documented.

---

## Ready to run, if the guard question resolves in favour of more targets

Catalog, code and scripts are all in place.  `dev/g2v3e_1e3.sh slice N` for
N = 1..4 (~6.6 h each), then `bash dev/g2v3e_1e3.sh cat`, then `resolve`.
Slice 0 is already on disk.  Expect σ(m1) → ~0.0010 at the full 5 slices,
since k = σ√N_in = 1.054 measured on slice 0.
