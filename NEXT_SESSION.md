# Next session — bfd_cnf

Written 2026-09-13, updated 2026-09-14 after a session that FIXED Thread 2
(see "Thread 2 — RESOLVED" below), then found a new, real, reproducible
~-4 to -6% multiplicative bias behind it (**Thread 3**), then a follow-up
session the same day that ISOLATED Thread 3's bias to the window's bright
flux tail and made a working decision to proceed on a restricted window
rather than chase it further right now (see "Thread 3 — isolated to
flux_hi" below). Originally written after a repo cleanup that cut `dev/`
from ~150 scripts to the ~40 still-live ones (see `git log` for the two
cleanup commits on `feat/centroid-shear-conditioning`, and the removed
`NEXT_SESSION_*.md` / `NEXT_PROMPT.txt` files this supersedes). `README.md`
is still the canonical description of the pipeline; this file is just the
open-thread task list.

**User's stated priority (2026-09-14, end of session)**: Thread 3 is bounded
well enough to move on. Resume **Thread 1 (PSFE leak)** next session, but
**only using the restricted, validated-null window** (`2.2 < Mr/Mf < 3.2`,
`1500 < Mf < 20000`) and the exact target-integration/selection-term
configuration validated below — see "Thread 1 — RESUME CONFIG" for the
literal flags. Do not silently fall back to the old `dev/bias_psfe_g2v3d.sh`
window/flow (2500-50000 flux, the pre-jac-weight `sigmaxblock_multiscale`
flow); that script is now stale and needs updating first.

## Where things stand

`bulk_g2v3d_full2.eqx` / `shear_g2v3d_full2.eqx` (2026-09-12) is the current
best flow: `dm/dg` RMS residual against bfd's exact derivatives is 1.4-2.7%
(`logs/shear_g2v3d_full2.log`), after `models/shear.py` got a split
coefficient bound (`_COEFF_MAX_SPIN0`) and a point-source edge taper
(`_edge_factor`) — see commit `4e48ade`. `bias.py` also gained a
nearest-neighbour support-density guard (`build_support_density`,
`in_support_density`) for selection-term draws, in the same commit.

Three threads now: Thread 1 (open, blocked), Thread 2 (RESOLVED 2026-09-14),
Thread 3 (new, the priority).

## Thread 2 — leader-draw / R_s instability — RESOLVED 2026-09-14

Everything below the original writeup (now folded into history further down
this file) led to a working fix. Summary for anyone not reading the blow-by-
blow: the R_s score-function estimator's instability was NOT one bad draw —
it was a genuine local-Jacobian steepness in the bulk flow's coupling layers,
present in MULTIPLE layers' conditioner nets (not just the one `bend=True`
layer `dev/bend_layer_check.py` first found), so a penalty on only that one
net fixed the ONE known leader but left the estimator just as broken once a
realistic `bias.py` run (centroid layer on, 2^24 selection draws) surfaced
the NEXT-worst leader living in a different layer entirely.

**The fix, `bulk.py`'s `--jac-weight`**: an explicit Jacobian penalty (mean
squared gradient) on every conditioner net (`Spin0AutoregressiveLayer.
net_ls1`/`net_ls2`, `Spin2CouplingLayer.net`) in all 8 bulk layers, not just
one — see `bulk.bulk_jacobian_penalty`'s docstring for the full derivation
and the two failed attempts that came before it:
  - v1 (single-layer, no stop-gradient): the optimiser collapsed
    `RawMomentStandardize.std[4]` to exactly 0 to cheat the penalty for free —
    a cancellation exploit of the same kind `[[score-weighted-deriv-loss]]`
    found for `--score-weight`.
  - v2 (single-layer, stop-gradient added): closed that gradient path, but a
    SECOND, structural collapse of `std[3]`/`std[4]` still happened, because
    forcing one net flatter makes it unable to represent the true density and
    joint training "resolves" that by degenerating the chart instead.
  - v3 (single-layer, chart frozen at its OWN untrained init): training ran
    clean, but freezing the chart at a value far from what full NLL training
    converges to left the rest of the architecture mismatched, and pushed the
    bend layer's log-det formula (`2h + log1p(2 q dh)`) negative-domain (NaN)
    at one held-out validation point.
  - v4 (single-layer, chart warm-started from the CONVERGED baseline via the
    new `--chart-from`, then frozen): trained clean, 0/10000 non-finite val
    points, and cut the known leader's Lipschitz ratio 4.27e5 -> 1.98e4
    (21.6x). But re-run through the REAL estimator (centroid on, `bias.py` at
    2^24 draws) it made things WORSE: leader concentration 44% -> 115% of
    R_s11, because the fix only reached the ONE net the original diagnosis
    found, and `dev/jacobian_decomposition.py`-style analysis of the NEW
    worst leader showed its 37.8x dominant steepness came from the SECOND
    bulk layer's `spin2.net`, which has `bend=False` and was never touched.
  - **v5, generalized to all layers, chart-from the converged baseline,
    `--jac-weight 0.1` — WORKS.** `flows/bulk_g2v3d_full2_jac.eqx` /
    `flows/shear_g2v3d_full2_jac.eqx` / `flows/centroid_g2v3d_full2_jac.eqx`.
    Response fit unchanged (`dm/dg` RMS 1.89/1.46/2.62/2.66/1.81%, matches
    baseline to noise). Known leader's Lipschitz ratio: 4.27e5 -> 2.73e4
    (15.6x). Real `bias.py` run, centroid on, 2^24 draws: leader concentration
    44% -> **3.1%**, R_s traceless monitor (should be exactly 0 by isotropy)
    -5.3e-2 -> **+1.4e-6**. This is a real fix, not a narrower one that will
    fail the next re-run the way v4 did — see Thread 3 below for the
    (extensive) cross-checking that went into believing that.

**Retrain recipe**, for reference:
```
python bulk.py train --jac-weight 0.1 --chart-from flows/bulk_g2v3d_full2.eqx \
    --data ../bfd_cnf_imsims/data/moments_gauss2_fwd_g2v3d.fits \
    --flow flows/bulk_g2v3d_full2_jac.eqx --steps 150000
python shear.py train --init flows/bulk_g2v3d_full2_jac.eqx --deriv-weight 1e4 \
    --data ../bfd_cnf_imsims/data/moments_gauss2_fwd_g2v3d.fits \
    --flow flows/shear_g2v3d_full2_jac.eqx --steps 60000
python centroid.py train-sigmax --multi-scale --anisotropic \
    --copies ../bfd_cnf_imsims/data/copies_gauss2_fwd_g2v3d.fits \
    --init flows/shear_g2v3d_full2_jac.eqx --flow flows/centroid_g2v3d_full2_jac.eqx
```
`--jac-weight` freezes the chart automatically whenever it's nonzero
(`bulk._trainable`) — that is load-bearing, not a style choice; see v1/v2
above for what happens without it.

Also fixed in passing: `dev/window_scan.py` was calling `bias.
selection_terms_score(..., guard=a.window_guard)`, a kwarg that function no
longer has (replaced by the `density=` support-density filter well before
this session) — every offline re-solve through that path was silently
crashing until this session patched it to build and pass the density filter
instead, matching what `bias.py`'s own `main()` does.

## Thread 3 — real ~-4 to -6% multiplicative bias, cause unknown (NEW, the priority)

Once Thread 2's R_s instability was actually gone (not just quieter), the
`gauss2_v3d` windowed, selection-corrected `m1` stopped being noise-dominated
and consistent with zero, and turned into a small but real, reproducible,
significantly-negative number. This was NOT expected — see the discussion in
this session's transcript for why a nonzero result here was surprising enough
that a sim-level bug looked like the more likely explanation going in.

**The headline numbers**, all on `flows/centroid_g2v3d_full2_jac.eqx`
(Thread 2's fixed flow), `--pop gauss2_v3d`, window `(2.2,3.2)` in Mr/Mf x
`(2500,50000)` in Mf, `--no-zero-arm`, default gauge (`prior`):

| check | corrected m1 |
|---|---|
| 20k targets (offset 0), `--window-terms score` (the flow's own estimator) | -0.0416 +/- 0.0195 |
| same, `--gauge auto --floor-eps 1e-3` | -0.0422 +/- 0.0191 |
| same targets, `--window-terms templates` (bfd's EXACT analytic response for the selection term, no trained flow involved in that half at all) | -0.0441 +/- 0.0194 |
| 20k targets, offset 200000 (independent slice of the 500k catalog) | -0.0348 +/- 0.0312 |
| 20k targets, offset 400000 (independent slice) | -0.0619 +/- 0.0206 |

Four independent things all agree: two different SELECTION-TERM estimators
(the flow's score-function estimator vs. the exact closed-form templates
estimator) land within 0.003 of each other, and three independent,
non-overlapping 20k-target slices of the 500k catalog all land in the same
-0.03 to -0.06 range with no sign flips. This is not a fluke of one draw
seed, one estimator, or one slice of targets.

**What's been ruled out** (in the order it was checked, so the next session
doesn't re-derive it):
- **Not the leader/instability itself.** Manually excluding the single
  largest-|R11| selection-term draw from the OLD (pre-fix) flow's estimator
  leaves `m1` at +0.0187 +/- 0.0281 — still consistent with zero, and the
  isotropy-violating traceless monitor is UNCHANGED (-5.27e-2 either way).
  So the pre-fix "≈0" result was never a correct answer hidden by one bad
  draw; it was noise from a broken estimator that happened to average out.
  Script: none saved (`/tmp` scratch, `drop_leader_draw.py` — reproduce from
  this session's transcript if needed, it is ~80 lines using `bias.
  selection_terms_score`'s exact per-chunk formula).
- **Not a flow density-fit defect.** `--window-terms templates` bypasses the
  trained flow entirely for the selection term (uses bfd's exact analytic
  response on the real catalog) and reproduces the flow-based number to
  0.003. Whatever this is, it is not something unique to what the flow
  learned.
- **Not `--gauge`/`--floor-eps`.** `--gauge auto --floor-eps 1e-3` (the flags
  `dev/g2v3d_500k.sh` uses and memory's `[[gauss2-passes-with-floor-and-
  guard]]` calls mandatory for gauss2) tightened the per-quintile MC noise
  enormously (q2's error bar: +/-20.8 -> +/-0.012) but moved the corrected
  `m1` by only 0.0006. The instability those flags fix is a DIFFERENT,
  smaller effect layered on top of this one, not the same thing.
- **Not the response fit.** `shear.py check` on the fixed flow: `dm/dg` RMS
  1.89/1.46/2.62/2.66/1.81% (Mf/Mr/M1/M2/Mc) against bfd's exact derivative,
  matching the baseline (pre-fix) flow to noise. The response the shear layer
  learned did not change.
- **Not the centroid/SigmaXBlockLayer.** `dev/jacobian_decomposition.py`-style
  layer walk on the centroid-on chain: `SigmaXBlockLayer`'s own
  leader/control Jacobian-condition ratio was 0.78x (LESS steep than
  controls there) -- it contributes nothing to any steepness story, and this
  session confirmed `train-sigmax` (not the old `CentroidMarginalize`) is the
  architecturally correct layer type, so that is not a configuration mistake
  either.
- **Not statistical luck in the target draw** -- see the three-slice table
  above.
- **Believed (not yet directly confirmed) to be a WINDOW effect, not a
  population-wide one.** Reasoning: the UNWINDOWED and windowed-UNCORRECTED
  numbers are IDENTICAL between the score and templates methods (both are
  computed before the selection-term correction is ever applied, so of
  course they don't discriminate) -- but the windowed-uncorrected value
  itself (-0.0177 +/- 0.0102) is only mildly significant, while the
  CORRECTION (`R_s`, applied because the window discards ~75% of the
  population) is what pushes it out to -0.04 to -0.06 with high
  significance, and does so reproducibly across two very different
  correction estimators. That points at the size/flux cut's selection
  correction specifically, not something wrong with the whole population or
  flow. **Not yet run**: `dev/window_scan.py`'s own boundary-perturbation
  scan (drop `--no-scan`) measures d(m1)/d(boundary) at each window edge --
  if this is a window-boundary effect it should show a real, non-noise
  slope there, in the same spirit as `[[residual-is-the-size-edge]]` /
  `[[edge-bias-is-conditional-support-leak]]` for the OLD (non-`_full2`)
  flow chain. This is the concrete next step.
- **Abandoned, not conclusive**: an attempt to get a fully-exact answer (both
  the target-side noise integration AND the selection term computed with NO
  trained flow at all) by extending `truth.py`. This DID get built and
  works correctly (`truth.log_prob_sigma`, `truth.pqr_sigma`, `truth.
  BiasFlow`, and `bias.py --flow truth` -- composes the exact analytic
  gauss2 density with `models.centroid._transport`'s exact zero-parameter
  centroid-marginalisation transport, verified to reduce exactly to plain
  `truth.log_prob`/`pqr` at Sigma_X=0). It was abandoned for being far too
  slow to run at any useful `--n-targets` (a 30-step Newton solve per draw,
  vs. a flow's single forward pass -- confirmed via `nvidia-smi` to be
  genuinely GPU-bound at 100% utilization, not hung, just slow, which is
  reportedly why this was never built before). If revisited: the code is
  real and correct, just needs either a much smaller `--n-targets` (tried
  500, still impractically slow with `--chunk 16 --batch-budget 64` -- go
  smaller, like 50-100, or invest in vmapping `analytic.theta_of_m`'s Newton
  solve properly instead of `jax.lax.map`) or a cheaper substitute question:
  `truth.pqr_sigma` applied directly to noiseless catalog moments (no MC
  integration, no centroid-noise convolution, instant) answers a narrower
  but still useful question -- is the response/selection machinery itself
  unbiased in the noise-free limit.

**Recommended concrete next step**: run `dev/window_scan.py`'s local
boundary-perturbation scan (remove `--no-scan`, needs the cached selection
terms this session already wrote at `logs/phase_a/terms_gauss2_v3d_24_s0_
1w_g100*.npz` -- check which one corresponds to the FIXED flow, the BASELINE
ones from this session's leader-drop experiments are a different flow and
must not be reused for this) on the fixed flow, size window `(2.2,3.2)` and
flux window `(2500,50000)`, to see whether `m1` has a real slope with either
boundary. A strong slope at the SIZE boundary specifically would tie this to
the same size-edge mechanism `[[residual-is-the-size-edge]]` found on the old
flow chain, just not yet fixed by whatever `models/shear.py`'s edge taper
(commit `4e48ade`) was supposed to fix for `_full2`. A flat slope would argue
against a boundary mechanism and reopen the "is this actually a window issue
at all" question the transcript's own reasoning above is not 100% sure of.

## Thread 3 — isolated to flux_hi, 2026-09-14 follow-up session

Same day, a second session picked Thread 3 back up and worked through the
cause list systematically (F, then D, then E, then A/B/C in that order,
cheapest first). Summary for anyone not reading the blow-by-blow:

**Ruled out, in order:**
- **F (catalog/provenance mismatch)**: NEGATIVE. All Thread 3 runs use the
  identical flow (`centroid_g2v3d_full2_jac.eqx`), identical catalog
  (`gauss2_v3d` -> `targets_g2v3d_g1p02_500k` / `moments_gauss2_fwd_g2v3d.
  fits` for both targets and the R_s population), and consistent
  recentring-drop rates (2.1-2.7%) across every offset/gauge variant.
- **D (a combine-step bug in `bias.py`'s `ghat`)**: NEGATIVE, ruled out
  algebraically, not just empirically. `ghat`'s non-selection term (`Q =
  q.sum(0) - n_ns*qs/(1-ps)`, `R = obs + n_ns*(outer(qs,qs)/(1-ps)**2 +
  rs/(1-ps))`) is exactly the standard `log(1-P_s)` non-detection-term
  derivative (`d/dg ln(1-P_s) = -Q_s/(1-P_s)`, `d2/dg2 ln(1-P_s) =
  -R_s/(1-P_s) - Q_sQ_s^T/(1-P_s)^2`, sign-flipped once for the `obs = -SUM
  r` convention) -- sign and structure both check out by hand.
- **E ("is the machinery unbiased in the noise-free limit")**: answered, and
  NOT a null result -- see below. This was the pivot point of the session.

**E's answer, and why it matters**: a new scratch script (not committed;
recipe below) tested the windowed+corrected estimator with NO noise, NO MC,
and NO trained flow anywhere -- using only the population catalog's own
exact `dm/dg`/`d2m/dg2` (`shear.lens`, what `--window-terms templates`
already uses) for the selection correction, and `truth.py`'s exact analytic
gauss2 density (`truth.pqr_batch`) for the target-side Q,R on the SAME
catalog rows, Taylor-lensed to `g=+/-0.02` (and `+/-0.005` as a cross-check)
and treated as noiseless "observed" targets. Result: `m1` came out
**-0.021 at g=0.02, -0.052 at g=0.005** -- grows as g->0, which rules out
Taylor truncation in `shear.lens` as the explanation (a truncation artifact
shrinks as g->0, this grew). So the bias is not noise-driven, not
flow-density-driven, and not MC-driven -- it survives with literally
nothing but the exact catalog and the exact analytic density in the loop.
Both runs flagged the SAME thing Thread 2 originally found: one boundary
template near the flux ceiling (`Mf~=49820`, `Mr/Mf=2.876`) carried 47% of
`R_s11` -- a leader-concentration effect, present even with zero MC noise.

**A "strict window" test then partly REFUTED the obvious follow-up
(bigger population catalog = fixed R_s)**: a densely-populated window well
inside the bulk (`Mr/Mf` 2.45-2.75, `Mf` 6000-15000, ~1500-3000 templates,
nowhere near any percentile edge) showed comparable or WORSE
`templates`-mode leader concentration (41.5%/55.3%) than the original
boundary-adjacent window's 42% -- more local density did not fix it, at
least for the exact/`templates` estimator. The user does not care about
`templates` (production only uses `score`), so this was set aside rather
than chased further -- see the scan below for what actually matters for
`score`.

**The `score` (flow-based, production-relevant) boundary scan -- the actual
conclusion.** Pooling all three of today's independent 20k-target slices
(offset 0, 200000, 400000 -- `pqr/g2v3d_full2_jac_centroid_20k[_offset*].
npz`, 59851 targets total) against freshly-computed selection terms
(`--window-terms score`, 2^24 draws each) for a sequence of windows:

| window (Mr/Mf x Mf) | n_in/arm | R_s leader share | m1 (corrected) |
|---|---|---|---|
| (2.4,2.8) x (2000,20000) | 4309/4302 | 0.2% | +0.012 +/- 0.027 |
| (2.4,3.2) x (2000,20000) | 14237/14221 | 4.1% | -0.011 +/- 0.014 |
| **(2.2,3.2) x (1500,20000)** | **19129/19098** | **1.6%** | **-0.007 +/- 0.011 (NULL)** |
| (2.2,3.2) x (1500,50000) | 20791/20757 | 1.7% | **-0.037 +/- 0.011 (3.3 sigma)** |
| (2.2,3.2) x (2500,50000), original, single 20k slice | 4671/4666 | ~3% | -0.042 +/- 0.019 |

Raising `flux_hi` from 20000 to 50000, holding everything else fixed, flips
the result from a clean null to a 3.3-sigma negative bias -- with almost no
change in R_s leader share (1.6% -> 1.7%, i.e. the SCORE estimator is
well-converged at 2^24 draws in both cases, this is not an MC-noise
artifact). This isolates the driver to **`flux_hi`, not `size_hi`** --
`size_hi=3.2` alone (with `flux_hi=20000`) stayed null. Population density
falls off sharply approaching the 50000 ceiling: ~1677 templates per
1000-flux-unit in the bulk vs. ~36 near 48000-50000, a ~47x drop.

**Working hypothesis (user's, end of session, not yet directly tested)**:
the real templates are too sparse at the bright end for the FLOW's learned
density to be accurate there -- a genuine density-fit error in a
high-signal-to-noise, under-represented part of the population, not an
estimator/MC artifact (the leader-share/convergence evidence above argues
against "just needs more draws"; the templates-mode strict-window result
argues against "just needs more local templates" for the discrete estimator,
though that finding is about `templates`, not `score`). **Not yet tested**:
whether `score`'s own local density estimate (the flow's log-prob, or a
`build_support_density`-style nearest-neighbour check) actually degrades
specifically in the `Mf > 20000` tail, which would confirm this directly.

**Decision (user, end of session)**: don't chase this further right now.
Move on to Thread 1 (PSFE leak) using the validated-null window as a
boundary the analysis stays inside of. See "Thread 1 — RESUME CONFIG" below
for the exact configuration to use.

**Reproducing today's checks** (none of this is committed as a `dev/`
script -- it lived in the session's scratchpad and is not on disk in the
repo):
- The noise-free machinery check (hypothesis E): `truth.pqr_batch` on
  `shear.lens`-Taylored population rows, combined via `bias.ghat`/`bias.
  selection_terms`, non-finite rows dropped (a handful of in-window rows
  land outside gauss2's true analytic support after lensing and need
  filtering, same as `bias.py`'s own `|Q|`/`|R|` guard) -- see this
  session's transcript for the ~90-line script if it needs rebuilding.
- The pooled `score` boundary scan: for each window, `dev/window_scan.py
  --pop gauss2_v3d --flow flows/centroid_g2v3d_full2_jac.eqx --pqr
  pqr/g2v3d_full2_jac_centroid_20k.npz --size <lo> <hi> --flux <lo> <hi>
  --log2-draws 24 --no-scan --seed 0 --no-prefilter-weights --cache
  <distinct path>` to compute+cache selection terms for that window
  (**always pass a distinct `--cache` path per window** -- the tool's
  default cache filename does not encode the window, size, or flux, only
  `(pop, log2_draws, seed, n_windows)`, so two different windows at the same
  draw count/seed silently collide and the second run reads the first
  window's stale R_s without complaint), then a small pooling script that
  loads all three `pqr/g2v3d_full2_jac_centroid_20k[_offset200k/400k].npz`
  files, concatenates `plus_q/r`, `minus_q/r`, `obs_plus/minus` (window-
  independent, safe to concatenate across independent target slices),
  applies `bias.window_mask` with the target window, sums `n_ns` across all
  three slices, and calls `bias.bias(..., ns=(...))` plus a galaxy
  bootstrap over the pooled ~60k targets. ALSO not committed to `dev/`; a
  `dev/window_scan_multi.py` combining these two steps into one script,
  properly, is a reasonable thing to build before the next round of window
  experiments rather than re-writing scratch code again.
- **Also found in passing, not yet followed up**: `dev/window_scan.py`
  line ~342's summary print (`A1/A3 at 2^24, nominal window size ...`) is
  hardcoded to print the module's `NOMINAL_SIZE`/`NOMINAL_FLUX` constants
  regardless of what `--size`/`--flux` were actually passed. Cosmetic only
  -- the actual computation uses the real args correctly (verified against
  a standalone `P_s` check) -- but worth a one-line fix so it stops being
  misleading.

## Thread 1 — RESUME CONFIG (validated 2026-09-14, use exactly this)

Before picking Thread 1 back up: `dev/bias_psfe_g2v3d.sh` is STALE. It
currently points at the OLD flow (`flows/centroid_g2v3d_sigmaxblock_
multiscale.eqx`, pre-`--jac-weight`) and the OLD window (`--window-size 2.2
3.2 --window-flux 2500 50000`, the one Thread 3 just showed is
significantly biased). Update it to match what was validated null today
before trusting any new PSFE number against it:

  - **Flow**: `flows/centroid_g2v3d_full2_jac.eqx` (Thread 2's fixed flow),
    not `centroid_g2v3d_sigmaxblock_multiscale.eqx`.
  - **Window**: `--window-size 2.2 3.2 --window-flux 1500 20000` -- this
    EXACT window, not the old `2500 50000`. This is the one combination
    validated null (`m1 = -0.007 +/- 0.011` over 59851 pooled targets) in
    today's scan; neighbouring windows (narrower size, or `flux_hi` back at
    50000) were NOT null. Do not widen this window without re-validating it
    the same way (pooled multi-slice `score` re-solve, check leader share
    stays low AND `m1` stays consistent with zero).
  - **Target integration**: `--samples 8192 --alpha 0.5 --chunk 2048
    --batch-budget 16384 --n-targets 20000 --no-zero-arm --support
    --floor-eps 0` -- this is what today's base 20k run
    (`bias_g2v3d_full2_jac_centroid_20k.log`) used, NOT `--gauge auto
    --floor-eps 1e-3 --window-guard 100` (the old gauss2-mandatory flags
    from `[[gauss2-passes-with-floor-and-guard]]`, measured against the OLD
    pre-`--jac-weight`/pre-`--support`-flow catastrophe). The `--gauge auto
    --floor-eps 1e-3` variant WAS also checked today (on the OLD, wider
    window) and agreed with the `--support --floor-eps 0` number to 0.0006,
    so either should be fine, but `--support --floor-eps 0` is the one with
    a validated null result on THIS window -- use that one unless there is
    a specific reason to switch.
  - **Selection term**: `--window-terms score --window-draws 16777216`
    (2^24, not the script's current `1048576` / 2^20) -- today's leader-
    share numbers (0.2-4.1% clean) are all at 2^24; smaller draw counts are
    NOT validated on this window and the session's earlier quick checks at
    2^20 on the OLD window saw leader concentration as high as 341%, purely
    from undersampling.
  - Everything else in the script (per-orientation/amplitude `pop` loop,
    `--save-pqr`) can stay as-is.

## Thread 1 — PSF-anisotropy (PSFE) leak

Full reasoning in `PSFE_PROVENANCE.md` — read that first, including new §9
below. Short version: the ellipticity of the PSF leaks into `dc2` through the
`Sigma_X` channel (linear in `e_psf`, `[[psf-leak-is-the-sigma-x-channel]]`),
confirmed again on the `gauss2_v3d`/mcfix grid at ~3σ (`logs/bias_g2v3d_psfe*_
mcfix.log`, `m1 ~ +0.026-0.028` corrected, consistent across `psfe00`/`psfe1p05`).

**UPDATE 2026-09-14 (`PSFE_PROVENANCE.md` §9)**: reran the 10-config psfe grid
on the FIXED flow (`flows/centroid_g2v3d_full2_jac.eqx`) and the
validated-null restricted window (`(2.2,3.2) x (1500,20000)`, "Thread 1 —
RESUME CONFIG" below), plus updated `dev/psfe_paired_diff.py`/`dev/
psfe_joint_slope.py` to compute selection terms fresh (no terms-cache exists
for this flow/window). **The leak survives, comparable or slightly stronger,
not suppressed**: `dc1/d(psf_e1) = -0.0293 +/- 0.0032 (-9.1σ)`, `dc2/d(psf_e2)
= -0.0141 +/- 0.0055 (-2.6σ)`, vs. the old flow/window's `-0.0156 +/- 0.0030
(5.3σ)` / `-0.0115 +/- 0.0036 (3.2σ)`. Neither Thread 2's `--jac-weight` fix
nor Thread 3's window restriction touches this — it's a genuine,
flow-independent effect, not an artifact of either prior issue.

**UPDATE 2026-09-15 — selection term (`Q_s`) ruled out; `--no-centroid`
bisection run, RESOLVED: the centroid layer contributes ZERO suppression.**
Two follow-ups this session, both concrete answers, neither the driver:

- **Selection term (`Q_s`) checked and ruled out.** `Q_s` is not exactly zero
  across the psfe grid (a first linear fit through +/-`psf_e` pairs gave
  exactly 0.0000, but that is a fit-design artifact — by isotropy at `g=0`,
  `Q_s(Sigma_X)` can only depend on Sigma_X's rotation-INVARIANT part
  (eigenvalues, i.e. `|psf_e|^2`), never orientation, so a linear/odd fit
  against signed pairs cancels to exactly zero regardless of the underlying
  physics). The real, isotropy-consistent, orientation-independent quadratic
  signal is there (`dev/psfe_qs_slope.py`, parses the existing 10 grid logs,
  no rerun) but tiny: `+1e-7` to `+1.6e-6` across the amplitude range, ~1000x
  below both `Q_s`'s own MC error (`5.9e-4`) and the `c1`/`c2` leak
  (`~1e-2`). Not the driver.
- **`--no-centroid` bisection (proposed §8d, never run before now) — RUN,
  minimal rerun (baseline + `psfe1` x 3 amplitudes, `dev/
  bias_psfe_g2v3d_nocent.sh`, `logs/bias_g2v3d_full2_jac_psfe*_nocent.log`,
  `~30 min total`).** Corrected `c1` with the `SigmaXBlockLayer` entirely
  removed (`--flow flows/shear_g2v3d_full2_jac.eqx --no-centroid`) is
  statistically IDENTICAL to the centroid-layer run at every amplitude:

  | `psf_e1` | with centroid | without centroid | diff |
  |---|---|---|---|
  | 0.00 | -8.59e-3 +/- 3.9e-3 | -8.76e-3 +/- 4.0e-3 | +1.7e-4 |
  | 0.02 | -9.25e-3 +/- 4.1e-3 | -9.26e-3 +/- 4.0e-3 | +1e-5 |
  | 0.05 | -9.89e-3 +/- 4.6e-3 | -9.92e-3 +/- 4.1e-3 | +3e-5 |
  | 0.10 | -1.03e-2 +/- 4.0e-3 | -1.03e-2 +/- 4.1e-3 | ~0 |

  Same monotonic slope with or without the layer, every difference well under
  1σ. This is not "partial suppression" (the WORSE-without-it outcome) — the
  layer contributes **no suppression at all**. Rules out "centroid layer
  fails to fully capture the Sigma_X dependence" as the mechanism; whatever
  learns/fails to learn Sigma_X's anisotropy is not where this leak lives.

**UPDATE 2026-09-15 (cont.) — tightening the size ceiling to (2.2,2.8) does
NOT shrink the slope.** Reused the existing `pqr/g2v3d_full2_jac_psfe*.npz`
files (no target-integration rerun — `dev/window_scan.py --no-scan` re-solves
selection terms fresh off the cached per-target Q/R/obs at a new window,
~1-2 min/config at 2^24 draws instead of the ~7-8 min/config a full `bias.py`
run needs). `dc1/d(psf_e1)` (unweighted, 4-point, `window_scan`'s own
non-paired errors — NOT the tight joint-bootstrap the -0.0293+/-0.0032
headline used, so this is suggestive, not yet as rigorous):

| window | dc1/d(psf_e1) |
|---|---|
| (2.2,3.2), original | ~-0.0193 |
| (2.2,2.8), tightened | ~-0.0204 |

Cutting the outer size edge moved the *baseline* `c1` a lot (-8.59e-3 ->
-2.70e-3) but did NOT shrink the *slope* with `psf_e1` — if anything slightly
larger. This argues AGAINST the shared-cause-with-Thread-3's-size-edge
hypothesis: if the leak were riding the same outer-edge density failure,
cutting it out should have suppressed the slope, and it didn't. Caveat: needs
the proper paired-diff-from-`psfe00` bootstrap (same technique `dev/
psfe_joint_slope.py` uses for the headline number) before trusting this as
more than suggestive — see the plan below.

**Where that leaves Thread 1, end of 2026-09-15**: selection term ruled out,
centroid/Sigma_X layer ruled out, shared-cause-with-Thread-3's-size-edge
argued against (all direct tests now, not circumstantial). None of the
flow-side or estimator-side mechanisms tried have panned out. Time to step
back to the BFD formalism itself rather than iterate further on flow
architecture/training — see the closure plan immediately below.

### Thread 1 — closure plan: enumerate every remaining channel, close each definitively

Six ways an anisotropic PSF's ellipticity could leak into the recovered
shear's `c1`/`c2` within BFD's own pipeline, ordered cheapest/most-decisive
first. Two are already closed; the plan below is written to close the rest
without more flow-architecture guessing.

**Already closed (do not re-litigate; re-verify only if a later step implies
a contradiction):**
- Channel A, selection term (`Q_s`/`R_s`): [[qs-quadratic-in-psf-e]] — real
  but isotropy-forced quadratic, ~1000x too small.
- Channel B, centroid/`Sigma_X` marginalisation: [[psfe-no-centroid-bisection-
  resolved]] — bit-identical with the `SigmaXBlockLayer` removed entirely.

**Step 0 (minutes, zero compute) — PSF round-trip sanity check.** Before
anything else: confirm the SAME `PSF_E` global is used for both rendering
(`imsims/sim.py`'s image generation) and deconvolution (`_measure()`'s
`psf_cov()` call feeding `bfd.simpleImage`) for every psfe config used in the
grid. Read the code path, don't assume it — `PSF_E` is a module-level global
set once in `main()` (`imsims/sim.py:922-923`), so check there's no path where
render and measure see different values (e.g. multiprocessing worker import
order, or a stale global from a previous population in the same script run).
If this is somehow wrong, everything downstream is moot and this is the whole
answer. Expected: closed (PSF_E is one global read both places), but confirm
before spending the compute below.

**Step 1 (cheap, one script, reuses everything) — does BFD's own exact
machinery show the leak at all, independent of the trained flow?** This is
the single most decisive test and mirrors Thread 3's own precedent exactly:
[[residual-is-not-the-flow]] proved Thread 3's residual is NOT a flow defect
by showing BFD's own template sum (eq. 35-36, no flow, no IS) reproduces it.
The equivalent tool for Thread 1 existed and was cut in the `dev/` cleanup —
recover it rather than rebuild: `git show b93e27b:dev/prior_n.py` and
`git show b93e27b:dev/band_tmpl.py` (the `whitened_templates`/`template_pqr`
dependency). Adapt from `bulgedisc_deep_v2`/its old `--pqr` convention to the
psfe grid: run it PAIRED per config against `psfe00`, on the existing psfe
target catalogs, using each catalog's own exact `dm_dg`/`d2m_dg2` (no flow,
no shear.py, no bulk.py) for the target-side Q,R the same way `--window-terms
templates` already does for the SELECTION side. `ESSMIN` matters (see the old
docstring) — do not skip it.
  - **If the leak persists at comparable magnitude with NO flow in the loop
    at all**: it is baked into BFD's own moment/response definitions under an
    anisotropic PSF (channels 2 or 3 below), and no flow retraining,
    architecture change, or centroid-layer redesign can fix it — the fix, if
    any, is in the deconvolution/weight math or in how the response is
    defined near an anisotropic PSF. Escalate to Step 2.
  - **If the leak shrinks or vanishes with the flow removed**: the flow's own
    representation (chart, moment standardisation, the "isotropic-scalar
    model form" [[psfe-leak-is-not-an-optimization-problem]] already flagged)
    is introducing or failing to correct something the exact machinery
    handles fine. Escalate to Step 4 instead.

**Step 2 (moderate — fits-file audit, no new renders) — is `C_M`'s
PSF-induced anisotropy (off-diagonal even-moment covariance) actually
reaching the likelihood correctly?** Sibling check to this session's
Sigma_X audit: pull `cov` (packed `C_M`) directly from the psfe grid's own
fits catalogs (same way `cov_odd` was pulled straight from the files this
session) and confirm it genuinely picks up off-diagonal structure
(`Cov(M1,M2)` etc.) with `psf_e`, matching BFD's own math. Then read
`mixture_draws`/`log_conv_is`/`pqr_streamed`'s handling of `cov` end to end —
is the FULL matrix used, or is there a diagonal-only approximation, a
transposition/packing bug (`bulkUnpack`), or a place `C_M` is treated as
population-representative rather than per-target? This is a code-reading
task first, a numeric check second — no GPU needed unless it turns up
something that needs re-solving.

**Step 3 (hardest, only if Steps 1-2 come up clean) — deconvolution/weight
truncation.** The classical PSF-anisotropy-leakage failure mode: an elliptical
PSF's `T(k)` decays anisotropically, and BFD's weight function's implicit
noise cutoff bites harder along the PSF's minor axis, so ANY residual PSF
anisotropy not removed by the (algebraically exact, but weight-truncated)
deconvolution imprints directly on the moments before anything else touches
them. Hardest to test empirically; the cleanest route is an ANALYTIC
toy-population check in the style of `truth.py`'s exact gauss2 machinery —
does a KNOWN closed-form population, convolved with a KNOWN elliptical PSF
and deconvolved via BFD's own exact Fourier-domain formula (no sim noise, no
flow, no MC), reproduce a nonzero `c1`/`c2` at `g=0`? If yes, this is a
property of BFD's deconvolution algebra itself under anisotropic PSF, fully
independent of this codebase's flow/estimator machinery, and the fix belongs
in the weight-function/deconvolution choice, not in `bfd_cnf`.

**Step 4 (only if Step 1 shows the flow is the culprit) — chart/model-form
audit.** Go back to `[[psfe-leak-is-not-an-optimization-problem]]`'s "likely
D's isotropic-scalar model form" and pin down exactly which piece: walk
`RawMomentStandardize`/the chart construction in `models/`, and check whether
any transform implicitly assumes a PSF-independent (isotropic) relationship
between moments that isn't exactly true once the PSF carries ellipticity —
i.e. does the chart's own Jacobian/whitening choice bake in an isotropy
assumption that channels 1-3 above don't actually satisfy under `psf_e != 0`.

**Do not restart the size-edge/centroid-layer/selection-term threads** —
those are closed by this session's direct tests (see "already closed"
above and the size-2.8 result). If Step 1 or 3 come back positive, Thread 1's
right home is probably a `bfd` (the base package, not `bfd_cnf`)-level issue
or a documented limitation, not more work in this repo's flow architecture.

**UPDATE 2026-09-15 (cont.) — Step 1 run, DECISIVE POSITIVE: the leak survives
with NO FLOW at all.** Recovered `dev/prior_n.py`/`dev/band_tmpl.py` from
`git show b93e27b:...` (now re-added to `dev/`, previously cut in the cleanup).
Full re-render of `copies_gauss2_fwd_g2v3d.fits` per psf_e config was judged
too expensive (24.3M rows, ~3.5GB, and `imsims/copies.py` has no
`--psf-e1/--psf-e2` flag at all) — user's call: reuse the ONE existing
`psf_e=0` copies pool for template moments/derivatives, and inject each
config's REAL measured PSF anisotropy through the two channels BFD's own eq.
35-36 actually carries it in: `Sigma_X` (the copy weight, `log_weights`) and
`C_M` (the whitening covariance), both read directly from that config's own
already-rendered `cov_odd`/`cov` columns (confirmed genuinely anisotropic on
disk, e.g. psfe2p10's `cov_odd` off-diagonal ~2085 against ~49668 diagonal).
This does NOT capture a PSF-driven shift in the deconvolved template moments
themselves (that needs the real re-render) — only the weight/whitening
channels.

New script `dev/psfe_no_flow_check.py` (+ `dev/psfe_no_flow_slope.py` for the
paired bootstrap): computes per-target Q,R via BFD's exact template sum
(`band_tmpl.template_pqr`, no flow, no autodiff through a trained model) on
all 10 psfe grid configs, `--pop gauss2_v3d`, Thread-1's validated-null window
`(2.2,3.2)x(1500,20000)` + `ESSMIN=1000` (an unwindowed/unguarded first pass
was catastrophically noisy — raw m1 ~ -1.05, the same leader-draw instability
Thread 2 fixed for the flow, still present here since this bypasses the flow
entirely). All 10 configs share the same seed-1 galaxies/row order, so a
paired bootstrap needs no nearest-neighbor matching — just the intersection
of each config's own window/ESS-kept row indices (3946 of ~4040 common to all
10).

**Result: `dc1/d(psf_e1) = -0.0418 +/- 0.0008` (-52.8 sigma), `dc2/d(psf_e2) =
-0.0458 +/- 0.0012` (-39.3 sigma).** Same sign as the flow-based leak
(`-0.0293`/`-0.0141` from the `_full2_jac` grid), comparable-to-somewhat-larger
magnitude, and clean cross-axis nulls (`dc1/d(psf_e2)`, `dc2/d(psf_e1)` both
consistent with zero — no cross-talk, physically sane). Per-config point
estimates are monotonic in amplitude with no exceptions.

**Conclusion: the leak is baked into BFD's own Sigma_X/C_M-driven likelihood,
not into anything the trained flow learned or failed to learn.** This closes
off Channel 4 (chart/model-form audit) as the primary suspect and escalates
directly to **Step 2** (does `C_M`'s PSF-induced anisotropy correctly reach
the likelihood — a code-reading task, `mixture_draws`/`log_conv_is`/
`pqr_streamed`) and **Step 3** (deconvolution/weight-truncation leakage,
the classical elliptical-PSF failure mode) of the closure plan below. Caveat:
since this test used the Sigma_X/C_M channels only (not a real re-render), it
cannot yet distinguish "real, physical BFD-algebra leak" from "a bug in how
this codebase's whitening/weighting reads `cov`/`cov_odd`" — Step 2 is
specifically the code-reading pass that discriminates those two.

**UPDATE 2026-09-15 (cont.) — Step 3 (lightweight version, user's call): a
round/noiseless/known-centre toy shows the leak is NOT in static
deconvolution.** New script `dev/psfe_deconv_toy.py`: renders a single
PERFECTLY ROUND (`e1=e2=0`) Gaussian galaxy through `imsims.sim`'s own
render+measure path (`bfd.drawGauss` + `sim._measure(..., recenter=False)`,
i.e. NO noise, NO centroid solve, known centre) at `psf_e` up to 0.20 — far
past the grid's own 0.10 ceiling. Result: `M1`/`M2` (the spin-2 moments) are
EXACTLY zero at every amplitude (`dM1/d(psf_e1) = dM2/d(psf_e2) = 0.000000`,
noise floor ~1e-12 relative to `Mf`). **BFD's deconvolution + KBlackmanHarris
weight function is exact for a round, centred galaxy under an elliptical
PSF, at any tested amplitude.** This rules out the classical
"weight-truncation imprints residual PSF anisotropy on any galaxy's moments"
mechanism (Step 3's original hypothesis) as a shape-independent, always-on
effect.

Combined with Step 1's positive result, this sharpens rather than contradicts
the picture: the toy test is structurally incapable of probing the channel
Step 1 actually implicated, since `recenter=False` + no noise means there is
no `Sigma_X`/centroid uncertainty in the toy at all. **The leak is not in
static deconvolution — it is specifically in how PSF-driven `Sigma_X`
anisotropy enters the centroid-marginalisation math** (paper eq. 36's copy
weight `N(X_c; 0, Sigma_X)`, or the trained `SigmaXBlockLayer`'s
approximation of it), consistent with the original
`[[psf-leak-is-the-sigma-x-channel]]` finding — now confirmed two independent
ways (Step 1's flow-free reweighting test, and this toy's clean elimination
of the alternative). **Recommended next step, not yet run**: repeat
`psfe_deconv_toy.py` with `noise_sigma > 0` and `recenter=True` (so a genuine
`Sigma_X` exists and centroid marginalisation actually engages) on the SAME
round galaxy, across many noise realizations — if the round galaxy's
*recovered* M1/M2, averaged over noise, comes out nonzero under an
anisotropic PSF, that pins the leak to the centroid-solve/marginalisation
step specifically, with no galaxy-shape or population confound at all. This
is the natural, still-cheap continuation of today's toy and the most direct
remaining test before returning to flow/layer-level work.

**UPDATE 2026-09-15 (cont.) — noisy/recentred round-galaxy follow-up: weak,
not-yet-decisive lead.** `dev/psfe_deconv_toy_noisy.py`: same round galaxy as
the toy above, but now with real pixel noise and `recenter=True` (a genuine
centroid solve, so `Sigma_X`/centroid marginalisation actually engages),
`N=2000` PAIRED noise draws (same noise field reused across every `psf_e`
config, same antithetic-pairing trick as everywhere else — cut the per-draw
scatter from ~29-36 to ~0.15-0.65). Result: `dM1/d(psf_e1) = +4.06 +/- 1.81`
(2.2 sigma), `dM2/d(psf_e2) = +1.59 +/- 1.82` (0.9 sigma) — cross-terms null.
**Suggestive, not decisive**, and NOT directly comparable in units to Step
1's `dc1`/`dc2` (this is a raw first-moment shift on one galaxy, not a
population-level bias-corrected `c1` from `Q`/`R`) — do not quote this
alongside the -52.8/-39.3 sigma Step-1 number as if they were the same
statistic. Would need roughly 10x the draws (~hours, not run this session)
to get this past ~5 sigma on its own. **Not yet run**: the same test at
several galaxy sizes/`noise_sigma` values (to see if the effect scales the
way a centroid-marginalisation mechanism predicts), or just accept Step 1's
already-decisive result and move to auditing `SigmaXBlockLayer`'s own
math/training against eq. 36 directly instead of chasing more single-galaxy
significance.

**UPDATE 2026-09-15 (cont.) — user's question: is Step 1's leak a copies-grid
density/extent artifact? Real but modest effect, does not explain the leak
away.** The copy grid's shift extent/density (`imsims/copies.py`'s
`SIGMA_RANGE`/`SIGMA_MAX`/`SIGMA_STEP`) was built and calibrated for an
ISOTROPIC Sigma_X (`makeTemplates`'s `sigmaXY` is a scalar, and the grid's
inclusion boundary `chisq_xy = (MX^2+MY^2)/sigmaXY^2 <= sigmaMax^2` is a
circular cut in moment space) — Step 1 then reweighted that fixed grid by a
genuinely ANISOTROPIC Sigma_X. A real coverage/density mismatch was a live
possibility, not ruled out by Step 1 alone.

Tested directly: built two small (`n=3000` galaxies, same `--seed 0`, so the
SAME population in both) copies pools via `imsims.copies` — `narrow`
(`--sigma-range 1.4`, matching production's actual build args, confirmed via
`rebuild_v3.sh`'s `copies()` — no override there) and `wide` (`--sigma-range
3.0`, ~13x more copies/galaxy: 244 -> 3113). Reran the full 10-config
paired-bootstrap slope (same recipe as Step 1) against each:

| grid | dc1/d(psf_e1) | dc2/d(psf_e2) |
|---|---|---|
| narrow (range=1.4, matches production) | -0.0673 +/- 0.0016 (-42.3 sigma) | -0.0728 +/- 0.0023 (-31.6 sigma) |
| wide (range=3.0, ~13x denser/wider) | -0.0625 +/- 0.0015 (-41.7 sigma) | -0.0613 +/- 0.0021 (-29.8 sigma) |

The wide grid's slope is smaller by ~7% (`dc1`) to ~16% (`dc2`) than the
narrow one — a genuine, ~2-4 sigma shift (`|dc1_narrow - dc1_wide| = 0.0048`
against a combined error of `0.0022`; `|dc2_narrow - dc2_wide| = 0.0115`
against `0.0031`), so copy-grid density/extent DOES matter at some level, not
a null result. But it is a second-order correction, not the driver: both
settings stay overwhelmingly significant (30-42 sigma) at essentially the
same order of magnitude. A 13x denser/wider grid collapsing the signal by
only 7-16% argues against "insufficient copy coverage" as the primary
mechanism — if that were the whole story, a 13x improvement should have
moved the needle far more than this. Reads as a normal, mostly-converged
discretization systematic riding on top of a real, physical leak.

**Caveat, not yet resolved**: this used only `n=3000` template galaxies (vs.
production's 100000), so the absolute point estimates here (`dc1`/`dc2` both
somewhat larger in magnitude than the full-100k-pool Step 1 numbers,
`-0.0418`/`-0.0458`) are not directly comparable to Step 1's headline — this
test's value is the INTERNAL narrow-vs-wide comparison at fixed population,
not its absolute numbers. A `sigma-range` sweep PAST 3.0 (5.0, say) on the
same small population would show whether the ~10-16% shrinkage keeps going
(concerning) or has already leveled off (reassuring) — not run this session,
copies scale steeply with range (~13x cost from 1.4->3.0) so a further push
is a real but bounded compute commitment, not a full re-render.

**UPDATE 2026-09-15 (cont.) — real, confirmed bug found and fixed in the
offline diagnostic: a spurious per-copy Jacobian factor in `imsims/copies.py`'s
`log_weights`. Cuts the leak ~75% at full scale, but ~11-13 sigma survives.**

Traced BFD's own math (Bernstein+2015, arXiv:1508.05655) precisely: eq.
(35)/(36)/(38)'s `J(M)` is a per-TARGET constant (evaluated once at the
observed `M_i`, outside the copy sum) that provably cancels from `Q`, `R`,
`ĝ` -- it only ever contributes an additive, g-independent term to eq. (44)'s
`(const)`, never touching `Q_tot`/`R_tot`. `imsims/copies.py`'s `log_weights`
instead folds `|J(u)| = log_jacobian(copies["moments"])` in PER-COPY (varying
with each copy's own shifted moment, baked into `t["logw"]`, shared across
every target in `band_tmpl.py`'s `_logs`) -- a different, uncancelling
quantity. Confirmed against the base `bfd` package itself (`~/gitrepos/bfd`,
NOT `bfd_cnf`): `momentcalc.py`'s `makeTemplates` computes `detj` (the same
Jacobian) ONLY for a convex-region validity gate and the separate
`area_integral` diagnostic -- the actual per-copy weight it stores
(`tmpl.nda = tmpl.nda * da`, line 552) has NO Jacobian factor at all, and
`probabilities_jax.py:236-240` (the real P/Q/R assembly, `p = sum(p*nda)`,
`dpdg = einsum(nda, dpdm, dm_dg)`, etc.) uses that Jacobian-free `nda`
directly. So this is a confirmed, real deviation from how `bfd` itself
computes copy weights, not a defensible alternative convention.

**Fix**: drop `log_jacobian(copies["moments"])` from `log_weights` entirely
(`dev/psfe_no_flow_check.py`'s `NOJAC` env var; not yet ported back into
`imsims/copies.py` itself). Re-ran the full 10-config paired-bootstrap grid
on the FULL 24.3M-copy production pool (`copies_gauss2_fwd_g2v3d.fits`, not
a small proxy):

| | dc1/d(psf_e1) | dc2/d(psf_e2) |
|---|---|---|
| buggy (per-copy J, original Step 1 number) | -0.0418 +/- 0.0008 (-52.8 sigma) | -0.0458 +/- 0.0012 (-39.3 sigma) |
| fixed (no per-copy J) | **-0.0102 +/- 0.0008 (-13.3 sigma)** | **-0.0116 +/- 0.0011 (-10.8 sigma)** |

Cuts the leak ~75-76% at full scale (consistent with the ~65-71% seen on a
small n=3000-galaxy proxy pool first). **But a highly significant ~11-13
sigma residual survives** -- nowhere near the paper's own >3000x suppression
(`c` consistent with zero at `1e9` targets, their Table 1/eq. 59-60). So this
bug explains a large fraction of Step 1's original number but is NOT the
whole story.

**Not yet identified**: what accounts for the remaining ~11-13 sigma. Ruled
out as explanations so far (this session): PSF-dependence of `M(u)` itself
(exactly zero, `dev/psf_anisotropy_shift_probe.py`); copy-grid density/extent
alone (~10-16% effect, real but far too small on its own, `dev/copies_narrow
vs wide` test); the curvature/`Sigma_X` mechanism as a *general, unavoidable*
property of eq. 36 (contradicted by the paper's own clean result under an
equivalent anisotropic-PSF setup). Candidates not yet checked: whether the
FLOW's own training/inference path (`bulk.py`, `models/*`,
`mixture_draws`/`log_conv_is` in `bias.py`) has an analogous, independently-
introduced deviation from the base `bfd` formalism (it never calls
`imsims/copies.py`'s `log_weights` at all, so this specific bug can't be
the explanation for the ORIGINAL flow-based leak that started Thread 1 --
that agreement between the flow and Step 1's buggy diagnostic may be
coincidental, or may point to a shared, still-unfound issue). This is the
open thread for next time.

**Housekeeping**: `dev/psfe_no_flow_check.py` gained a `COPIES`/`CACHE_DIR`
env-var override (for testing against alternate copies pools) and a `NOJAC`
env var (drops the per-copy Jacobian). `dev/psfe_no_flow_slope.py` gained a
matching `CACHE_DIR` override. `dev/psf_anisotropy_shift_probe.py` (new) --
extends the existing `dev/psf_anisotropy_probe.py` gate-G0 check to nonzero
shift `u`, confirming `M(u)` is PSF-independent at every `u` tested, not just
`u=0`. None of today's `dev/_psfe_*_cache/` directories are committed (scratch
caches, regenerate via the scripts above).

Live tooling (all in `dev/`): `psfe_compare.py`, `psfe_joint_slope*.py`,
`psfe_qs_slope.py` (new, 2026-09-15 — `Q_s` vs `psf_e`, log-parsing only),
`psfe_angle_diagnostic.py`, `psfe_paired_diff.py`, `psfe_size_offline.py` (+
`_reseed.py`), `psfe_ess_by_orientation.py`, `psf_anisotropy_probe.py`, driven
by `bias_psfe*.sh` / `bias_psfe_g2v3d_nocent.sh` (new) / `render_psfe.sh` /
`psfe_afterburn*.sh`. Cached selection terms live in
`logs/phase_a/terms_*_mcfix.npz`.

**Not yet answered**: whether tightening the size ceiling shrinks `dc1`/`dc2`
(`psfe_size_offline.py`'s question, added 2026-09-11 — check if it ran to a
conclusion before picking this back up) and whether the leak's angular
dependence (`psfe_angle_diagnostic.py`, `psfe_ess_by_orientation.py`) narrows
it further. `NEXT_SESSION_psfe_rerender.md` (now removed) had a rerender plan
for `gauss2_fwd`/`noise_sigma=1.86` catalogs to match the trained
SigmaXBlockLayer's coverage — check whether that rerender happened before
re-deriving it; if not, it's still the right next step for testing this
against a population/depth the flow was actually trained on.

## Thread 2 detailed diagnosis history (superseded — see the RESOLVED summary above)

Kept for the record; the summary above is now the entry point. Everything
here predates and led up to the `--jac-weight` fix.

New 2026-09-12. `selection_terms_score`'s score-function estimator
concentrates on single draws: on `gauss2_v3d` one draw carried 61% of `R_s11`
before the density guard, 9% after it. The guard (`build_support_density`,
`bias.py`) rejects draws whose shape sits outside the training catalog's own
nearest-neighbour manifold — but the analysis in that function's docstring
found a real limit: the worst leader sits deep inside real-template density
(k-NN distance 0.46 vs. cutoff 4.6) despite having an unphysical exact
preimage (`log_p_theta = -inf` on `gauss2`'s known closed form) — the
reachable manifold folds back on itself near enough that no density estimate
over shape invariants alone can catch it.

Live tooling (`dev/`): `leader_census_logp.py`, `leader_logp_check.py` (does
the flow's own log-prob flag the leader as anomalous — if yes, a log-prob
cutoff might catch what the density guard can't), `lipschitz_check.py` (is
the leader's local Q/R sensitivity anomalous vs. real-template neighbors),
`fold_check.py` (is the Mr/Mf~3.0-3.3 fold a genuine feature of the true
moment map, checked analytically against `truth.py`), `boundary_shape_check.py`
(is the reachable-set boundary predictable from shape alone).

**UPDATE 2026-09-13 — all four checks now run, plus two follow-ups.** Results
below; `dev/lipschitz_check.py` had a bug (found and fixed this session — see
below) so re-pull before trusting an old run of it.

1. **`fold_check.py` — NEGATIVE.** The leader's own band (Mr/Mf 3.0-3.3,
   Mc/Mr 6.0-6.2) is analytically *better*-conditioned than a control band
   outside it (condition number p50 2378/max 8432 in-band vs. p50 1725/max
   3.75e6 outside). The true two-Gaussian moment map has no genuine fold
   there — whatever's happening is a property of the trained flow, not the
   physics.

2. **`leader_logp_check.py` — POSITIVE but partial.** The single known worst
   leader sits at the 2.73 local percentile of the flow's own log_prob
   (real signal). But `leader_census_logp.py`'s top-25-by-contribution census
   spreads 3-14% locally — a log-prob cutoff would catch roughly half these
   leaders and miss the rest. Not a complete guard on its own.

3. **`lipschitz_check.py` — HAD A BUG, NOW FIXED, STRONGEST SIGNAL FOUND.**
   The script defined `qr_vec` calling the un-jitted `one()` directly instead
   of the `chunked = eqx.filter_jit(jax.vmap(one))` it built but never used —
   every one of ~800 calls was retracing/recompiling `jax.linearize` from
   scratch, hanging for 20+ minutes. Fixed by jitting `qr_vec` itself
   (`one_jit = eqx.filter_jit(one)`); now runs in ~10s. Fixed result: the
   leader's local Q/R sensitivity (Lipschitz ratio 4.27e5) sits AT the 100th
   percentile of real-to-real sensitivity in-band — 1.55x the real-to-real
   max, ~1470x the median. This is the cleanest discriminator found so far,
   because it tests derivative steepness directly rather than proximity to
   training data (which is exactly what the k-NN density guard gets wrong on
   this leader: it's spatially unremarkable, k-NN distance 0.46 vs cutoff
   4.6).

4. **`boundary_shape_check.py` — POSITIVE.** `(Mr/Mf, Mc/Mr)` alone is
   insufficient to predict reachability: 18.7% of bins are genuinely mixed,
   and within mixed bins, ellipticity and flux carry a real, consistently-
   signed signal the 2D ratio projection can't see (reachable draws skew
   lower `|e|` and lower flux). Any 2D shape-invariant density guard is
   structurally underpowered; it needs these extra dimensions.

**Follow-up 1 — is this overfitting?** Tested concretely
(`lipschitz_vs_density.py`, not yet in `dev/` — see below): correlated each
point's local training-data density (10-NN distance) against its Lipschitz
ratio, across 600 real templates in-band. Result: Spearman ρ = −0.50
(p=6e-40) — steepness is HIGHER in denser regions, the opposite of the
overfitting/undertrained-sparse-region prediction. Worse for that theory: the
leader itself sits at the 96.7th percentile of LOCAL SPARSITY yet has the
single highest Lipschitz ratio observed — it's an outlier even relative to
points at its own density level, not an extension of a general density-driven
trend. Conclusion: not overfitting.

**Follow-up 2 — is any layer's scale factor saturated there?** Two candidate
mechanisms checked and BOTH RULED OUT (`coeff_ceiling_check.py`,
`scale_saturation_check.py`, not yet in `dev/`):
  - The shear layer's `_edge_factor`/`_COEFF_MAX` point-source taper: leader's
    spin-2 coefficient sits at 99.75% of its bound, but that's the norm in
    this band (median 97.85% across 3000 real templates, 43% above 99%) —
    and the coefficient's SENSITIVITY to a small ratio step is actually LOWER
    near the leader (median slope 736) than in a control region well below
    the ceiling (median slope 1137). Saturated regions are flat, not steep.
  - Every `_bounded_log_scale` sigmoid in all 8 bulk layers (both
    `Spin0AutoregressiveLayer` per-step scales and `Spin2CouplingLayer`'s
    stretch): the leader's per-layer activations all sit in 0.22-0.78, nowhere
    near the 0/1 pins, and its overall saturation score is at the 55.7th
    percentile of 300 in-band real templates — completely unremarkable.

**Not yet answered / recommended next step**: since no single layer is
saturated or ill-conditioned, but the FULL flow's composed Q/R response is
extremely steep at the leader (Lipschitz check), the likely explanation is a
composition effect — no one factor is at its bound, but the product of 8
bulk-layer Jacobians times the shear layer's can still amplify a small
moment-space perturbation multiplicatively. Next concrete step: measure the
FULL chain's Jacobian condition number layer-by-layer at the leader vs.
control points (extend `fold_check.py`'s SVD approach to the trained flow's
`transform_and_log_det` Jacobian composed layer-by-layer, not just the
analytic truth map), to find which layer(s) contribute the amplification even
though none are individually saturated. This is the same kind of
log-det-vs-score decomposition `[[spikiness-is-the-jacobian]]` used on the
bulgedisc bulk flow (`score = -(dT/dz)^T z_base + grad log|det|`), just not
yet run on this stack/leader.

Once that's understood, decide whether a usable second guard is: the
Lipschitz ratio itself (works today, ~10s per check after the fix), a
combined log-prob + Lipschitz threshold, or an expanded shape-invariant space
for `build_support_density` per finding 4 above.

**UPDATE 2026-09-13 (cont.) — layer-by-layer decomposition run, four more
follow-ups, and a guard-comparison table.** The composition-effect question
above is answered, and every single-metric guard tried since then
underperforms the composed Lipschitz ratio without fully replacing it.

5. **`jacobian_decomposition.py` (new) — the amplification is NOT spread
   across the stack.** Walking `flow.bijection.bijection.bijections` (data ->
   base: `RawMomentStandardize`, `ShearResponse`, then 8x
   `[EquivariantAutoregressiveLayer, Permute]`) and taking each layer's local
   Jacobian SVD at the leader vs. 20 in-band control points: the FIRST bulk
   layer alone (`bend=True`) is 87x steeper for the leader than the control
   median, and the first two bulk layers together produce a 649x
   leader/control gap in the cumulative condition number -- the peak of the
   whole stack. Every layer after that actually narrows the gap back down to
   20-52x; the tail of the flow partially corrects rather than compounding.
   `RawMomentStandardize`/`ShearResponse` are unremarkable (~1x). So the
   effect localizes to layers 2-4 of 18, not a diffuse composition effect.

6. **`bend_layer_check.py` (new) — the bend layer's own steepness is not
   sigmoid saturation or the `log1p(q)` small-q blowup.** The bend lives in
   `EquivariantAutoregressiveLayer.spin2` (a `Spin2CouplingLayer`, not the
   layer itself -- fixed a wrong attribute path this session). Its stretch is
   `_h(x,q) = _bounded_log_scale(net([x0,x1,x2, log1p(q)-W0]), ...) * 0.5`, a
   sigmoid, so steepest at sigmoid=0.5 and FLATTEST at saturation (already
   ruled out elsewhere, see item 4 in the original list above). Measured
   against 300 in-band real templates: the leader's `q=|e|^2` is only at the
   90th percentile (not extreme), and `1/(1+q)` -- the derivative of its
   4th input -- is actually LOW (10th percentile, works against steepness).
   The real driver is the net's raw pre-sigmoid output itself: 1.27 for the
   leader vs. a max of 0.63 across all 300 controls -- literally exceeds
   every one of them. A genuine net-generalization gap at this specific
   4-d input combination, not a mechanical saturation effect.

7. **`bend_conditioner_density_check.py` (new) — that gap correlates with
   LOCAL SPARSITY, opposite sign from the earlier full-chain finding.**
   Spearman(local 10-NN distance, `|net_out|`) = +0.447 (p=3e-147, n=3000)
   in the bend net's OWN 4-d conditioner space `[z0,z1,z2, log1p(q)-W0]` --
   net_out genuinely grows in sparser regions, the classic
   undertrained-extrapolation signature. This is the opposite sign from
   `lipschitz_vs_density.py`'s -0.50 on the full composed Lipschitz ratio in
   the 5-d shape-invariant space (which ruled out overfitting THERE) -- the
   two findings are about different spaces and do not contradict each other.
   The leader sits at the 98.3rd percentile of local sparsity in this 4-d
   space, consistent with the trend, but even the 51 real templates at least
   as sparse as the leader cap out at `|net_out|`=0.48 -- the leader's 1.27 is
   still 2.7x that ceiling, so density alone doesn't fully explain it either.

8. **`bend_density_guard_check.py` (new) -- a global-cutoff guard built in
   this 4-d space does NOT work, for a structural reason, not a tuning one.**
   Same recipe as `bias.build_support_density` (global max over every real
   template's own k-NN spacing) but in the bend-conditioner space: the known
   worst leader still tests `IN-SUPPORT` (it's sparser than most real
   templates but not sparser than the SINGLE sparsest one, which is what a
   global cutoff actually tests), and a fresh top-25-by-|contribution| census
   (2^20 draws) is only 1/25 rejected -- barely better than the existing
   shape-invariant guard's 0/25. A *global* cutoff structurally cannot use
   the density-vs-steepness correlation found in item 7; it would need a
   local/percentile-conditioned test instead, not tried yet.

9. **`lipschitz_guard_check.py` (new) -- the Lipschitz ratio, run against the
   same fresh top-25 census, generalizes only partially.** Real-to-real
   in-band baseline: p50=290, p99=6.6e4, max=2.75e5. A threshold at the
   real-to-real MAX flags 4/25 leaders; at the real-to-real p99 (a 2.3%
   false-accept rate on real templates) it flags 15/25. The single known
   worst leader from `lipschitz_check.py` (1470x median, 1.55x max) is far
   more extreme than this typical top-25 draw -- most of today's census sits
   at only 1-6x the real p99, not orders of magnitude beyond it. So
   "R_s leader" is not one failure mode with one clean signature; it is a
   population of draws of varying severity, and no single metric tried so
   far (shape density, bend-conditioner density, or Lipschitz ratio at either
   threshold) catches most of them.

**Guard comparison, same top-25 census:**

| guard | known worst leader | fresh top-25 census |
|---|---|---|
| shape-invariant density (existing, `bias.build_support_density`) | miss (by construction) | 0/25 |
| bend-conditioner density (new, item 8) | miss | 1/25 |
| Lipschitz ratio @ real-to-real max | catch | 4/25 |
| Lipschitz ratio @ real-to-real p99 | catch | 15/25 |

**Not yet answered / recommended next step**: no single-metric hard filter is
adequate. Two directions not yet tried: (a) a COMBINED score (e.g. Lipschitz
ratio x local density in the bend-conditioner space, or log-prob x
Lipschitz per `leader_logp_check.py`'s partial finding) might cover more of
the 25 than either alone; (b) accept that a hard filter cannot fully solve
this and instead characterize/bound the residual R_s instability that
survives the best available guard (e.g. re-run `window_scan_n_ns` or
`window-scan`'s cached terms with the p99 Lipschitz guard applied, and see
how much the m1/R_s error bar actually shrinks) -- that is the number that
determines whether Thread 2 is "good enough to resume Thread 1" per the
user's stated priority at the top of this file.

**Housekeeping**: `dev/lipschitz_vs_density.py`, `dev/coeff_ceiling_check.py`,
`dev/scale_saturation_check.py`, `dev/jacobian_decomposition.py`,
`dev/bend_layer_check.py`, `dev/bend_conditioner_density_check.py`,
`dev/bend_density_guard_check.py`, and `dev/lipschitz_guard_check.py` are all
uncommitted as of this writing (check `git status`) so they aren't lost; all
are one-off checks, not polished tools, so expect rough edges (e.g.
`scale_saturation_check.py` is unjitted per-sample and takes ~2 min for 300
control points -- jit it before scaling up the sample; the two census-based
guard-check scripts each take ~1 min per run at 2^20 draws and duplicate a
fair amount of the sampling/census boilerplate -- worth factoring out a
shared `dev/_census.py` if a third guard idea needs the same harness).

## Working rules (carried over from HANDOFF.md, still true)

- `shear.py train` on `bulgedisc` needs `--deriv-weight 1e4`.
- `--alpha 0.5` for any `_deep` population — `1.0` doesn't converge.
- `pqr_streamed` seeds chunk `c` as `seed + 7919*c`: raising `S` at a fixed
  seed nests the smaller draw set, so an `S` scan at one seed is not a
  convergence check.
- Two JAX processes do not fit on the 16GB card at `2^24` — serialize GPU
  jobs, and prefer `dev/window_scan.py`'s cache (`logs/phase_a/terms_*.npz`)
  for a re-solve so it doesn't touch the GPU at all.
