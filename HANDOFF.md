> This file was reset on 2026-08-25. Everything before that date is archived
> in `HANDOFF_archive_2026-08-25.md`, unchanged. Append new dated sections
> below rather than editing this one.

# Handoff: the centroid shear-redesign works as built, but doesn't resolve
the estimator disagreement -- and the search for why moved to the code, not
the physics

## 2026-08-25: `shear_delta_invariants`, its outlier failure mode, and what's
still unexplained

Branch `feat/centroid-shear-conditioning`, off `feat/shear-response-refactor`.

### What was built

`models/centroid.py`'s `g_invariants` (built from `p1 = Re(ēg)/G_MAX`, `p2 =
|g|²/G_MAX²`) was replaced with `shear_delta_invariants`: it delenses the
observed moments through the trained, frozen sibling `ShearResponse.unshear`
and feeds the coefficient net `(da, dq)`, the resulting shift in the size and
shape invariants, instead of a generic (e, g)-alignment proxy. `_SLOPE_MAX`
was retired; the slope terms are bounded by `_COEFF_MAX` (12.0), the same
bound `base` uses. `CentroidMarginalize` gained a frozen `shear` field
(`bulk.build_flow` wires it at construction; `bulk.sync_chart_constants`
re-points it after a warm-start graft, the same pattern already used for the
chart constants).

The no-op at known centers (`Σ_X = 0 ⇒ identity, for any coefficients`) was
verified algebraically -- `response` never adds a coefficient standalone,
only ever multiplies one by `t0` or `t2`, and both are exactly zero when
`Σ_X = 0` regardless of the coefficient -- and pinned by a new test,
`test_identity_at_zero_sigma_with_shear`, on an untrained net with its slope
heads deliberately woken up.

Full test suite passes: `tests/test_centroid.py` 13/13, `tests/test_shear.py`
14/14, the rest of `tests/` 39/39.

### The retrain and measurement

`python -u centroid.py train --copies
../bfd_cnf_imsims/data/copies_gauss2_deep.fits --flow
flows/centroid_gauss2_deep_sheardelta.eqx --init flows/shear_gauss2.eqx
--steps 16000` converged cleanly (NLL 42.3-42.7 throughout, off-chart
fraction ≤2e-3). `centroid.py check`'s static (g=0) ellipticity-response
overshoot came out 3.97x -- consistent with the pre-existing, already-known
4.36x/4.35x defect from the prior session, untouched by this change, not
investigated further tonight.

`bias.py` at 20k targets, gauss2_deep, `--samples 8192 --alpha 0.5 --chunk
4096 --batch-budget 65536 --window-size 2.2 3.2 --window-flux 2500 50000`
(same flags as the earlier pre-fix/post-fix comparison):

|  | windowed, uncorrected | windowed, corrected | unwindowed |
|---|---|---|---|
| pre-fix control (`flows/pre_chartsync/centroid_gauss2_deep.eqx`) | +0.00802 ± 0.00262 | −0.00851 ± 0.00347 | −0.02518 ± 0.00915 |
| broken post-fix (old p1/p2 slope, `flows/centroid_gauss2_deep.eqx`) | +0.03264 ± 0.00183 | +0.01530 ± 0.00284 | −0.02184 ± 0.00243 |
| new (`flows/centroid_gauss2_deep_sheardelta.eqx`) | −0.01405 ± 0.00140 | **−0.02980 ± 0.00264** | +0.08211 ± 0.04946 |

Reproduced byte-identically on a second run with `--save-pqr` added
(`logs_deep_cent_sheardelta.txt`, `logs_deep_cent_sheardelta_savepqr.txt`,
repo root) -- deterministic, not a fluke of one seed.

Confirmed the mechanism is genuinely engaged, not quietly inert: slope-head
weights moved well off zero-init (max |weight| 0.59, max |bias| 0.32 on the
trained net), `da`/`dq` have real per-galaxy magnitude (median 0.003/0.014 at
g=0.02 on a 2000-galaxy probe), and the `Σ_X=0` no-op still holds exactly on
the *trained* checkpoint.

### The outlier investigation

`--save-pqr` to `pqr_sheardelta.npz` gave per-target `Q`, `R`, truth, and
observed moments. Per-target `|R|`: median 43.5, 99th pct 296, 99.9th pct
5052, worst 35804 -- a 200-800x spread among a couple dozen targets.

Root mechanism, traced directly: `shear_delta_invariants` runs the trained
shear layer's `unshear` on the target's *observed* (noisy) ellipticity. For
most targets that's a small, well-behaved perturbation. For a target whose
measured ellipticity is wildly noisy, `unshear` is extrapolating far outside
anything the shear network saw in training. The final coefficient still
comes out clamped to ±12 (`rational_bound` on the summed coefficient does its
job at any single g), but it swings between the two saturated extremes
*within* the tiny ±0.02 shear disc -- a near step function in g, whose
derivative at g=0 (exactly what `bias.py`'s `Q, R` measure) is enormous.

Quantified:
- `q = |e|²` (standardized units, layer's own normalization, typical ≈ 3.5):
  21.7% of targets have q > 10 in at least one arm, 2.7% have q > 50, 0.7%
  have q > 100, 0.2% have q > 200 (max seen: 350,000). 41/50 of the worst
  `|R|` outliers have q > 10 in some arm.
- Cross-referenced the top 30 `|R|`/`|Q|` outliers against the raw target
  catalog (`targets_gauss2_deep_g1p02_200k.fits` etc.): `badcenter = False`
  for every one of them -- the Newton recentring solve reports clean
  convergence on all of them, so the pipeline's own binary check does not
  catch this population.
- `nda = 1.0000` for every one of them too, but this is **not a detection
  signal** -- `nda` is hardcoded to 1.0 for every target row
  (`imsims/sim.py::fill_row`, `row["nda"] = 1.0  # uniform sky density`) and
  is semantically a template-only quantity (`bfd/multimoment.py`: "sky
  density of objects like this (only relevant for templates)"). A target's
  `nda` says nothing; treating it as evidence of anything was a mistake made
  and corrected within this same session.
- `xyshift` (the continuous displacement the recentring solve actually
  applied, distinct from the binary `badcenter` flag) IS elevated for this
  population: catalog median 0.044, 99th pct 0.346; the top 30 `|R|`
  outliers range 0.07-0.79, several above the 99th percentile.
- Strongly flux-correlated: q > 100 occurs *only* in the faintest quintile
  (3.6% there, 0% in quintiles 2-5); frac(q > 10) falls monotonically 58.6%
  (faintest quintile) → 1.8% (brightest); log(flux)-log(q) correlation
  −0.52; the top 30 `|R|` outliers have flux 122-1443 against a population
  median of 5542.

So: faint object → low S/N → the centroid solve converges (by the binary
check) but to a point well past its typical excursion → the moments measured
there have an absurd ellipticity → `shear_delta_invariants` extrapolates the
shear layer's delensing map into territory built entirely from well-measured
galaxies → an enormous, spurious coefficient sensitivity to g. This is an
expected feature of a deep/noisy catalog (`noise_sigma = 2.73`), not a data
bug -- the fix has to be in the model (bound what `shear_delta_invariants` is
willing to act on), not the catalog.

### Outlier exclusion does not explain the windowed number

Exploratory, explicitly not a statistically valid procedure (post-hoc
selection on the estimator's own inputs) -- run out of curiosity, results
recorded because they were informative about WHERE the −0.0298 comes from,
not to argue for the estimate an exclusion produces:

| exclusion | windowed, uncorrected | windowed, corrected | unwindowed |
|---|---|---|---|
| none (baseline) | −0.01405 ± 0.00140 | −0.02980 ± 0.00264 | +0.08211 ± 0.04946 |
| worst 100 by \|Q\| or \|R\| | −0.01405 (identical) | −0.02980 (identical) | −0.00214 ± 0.00356 |
| worst 5% (999) by `xyshift` | −0.01428 ± 0.00132 | −0.03042 ± 0.00263 | +0.01294 ± 0.02726 |

Under every exclusion rule tried, the unwindowed estimate collapses toward a
tight, unremarkable value -- confirming the outlier population fully
explains the unwindowed instability and the earlier wild `q1` flux-quintile
result. But **both windowed numbers are essentially unmoved by any of these
cuts**, including cuts that remove hundreds to a thousand targets -- because
the window (Mr/Mf ∈ [2.2, 3.2], flux ∈ [2500, 50000]) already excludes
essentially all of this outlier population on its own (0 of the worst 100
`|Q|`/`|R|` outliers were inside the window to begin with).

**Conclusion: the windowed-corrected −0.0298 vs. the priors' −0.0085/+0.0153
is NOT a tail artifact.** It is a real difference in how this centroid layer
behaves on the ordinary, well-centered, in-window population, and remains
unexplained.

### Artifacts

- `flows/centroid_gauss2_deep_sheardelta.eqx` -- the new checkpoint.
- `logs_deep_cent_sheardelta.txt`, `logs_deep_cent_sheardelta_savepqr.txt` --
  repo root, the two (identical) bias.py runs.
- Per-target diagnostics (`pqr_sheardelta.npz`, `qvals.npz`, `sel_terms.npz`)
  were written to this session's `/tmp` scratchpad and were NOT committed --
  they will not survive past this session; the commands to reproduce them
  are all given above and in the session transcript.

### Carried forward, not acted on

The open design question from the previous log (archived) stands and is now
doubly motivated: **should `RawMomentStandardize.mean/std` be trainable at
all?** Tonight's `CentroidMarginalize.shear` field is another instance of
exactly the fragility that question flags -- a frozen copy of another
layer's state that has to be re-synced by hand (`bulk.sync_chart_constants`)
after every warm-start graft, because the original is trainable. Freezing
`RawMomentStandardize.mean/std` would make that whole class of bug
impossible rather than merely patched each time it's hit. Still left for the
user to call -- it would change `bulk.train`'s optimum, so it is a real
experiment, not a cleanup.

The 3.97x/4.36x static ellipticity-response overshoot is real, pre-existing,
and untouched by anything in this session.

## 2026-08-26: `_Coeffs.u_mean`/`.u_white` were stale too -- and fixing that
narrows the "unexplained" windowed-corrected gap by roughly half

`bulk.sync_chart_constants` re-points `e_scale`/`chart_loc`/`chart_scale`/
`CentroidMarginalize.mean/std/shear` at the chart's TRAINED mean/std after a
warm-start graft, but never touched `ShearResponse.coeffs.u_mean/.u_white` --
the affine whitening `_Coeffs.__call__` applies to its four net inputs before
the coefficient net sees them. Those two were computed ONCE, in
`bulk.build_flow`, from the chart's PRE-TRAINING statistics
(`bulk.coeff_stats`). Since `RawMomentStandardize.mean/std` are trainable and
move substantially during `bulk.train`/`shear.py train` (the module's own
docstring cites a 3.9x spin-2-std shift on gauss2), the whitening was
reverting to the ill-conditioned state it exists to fix (correlation 0.997,
condition number ~2625) for the entire duration of every `--init` warm start
-- i.e. every gauss2/bulgedisc/sersic flow on disk, including
`_sheardelta` above.

### Fix

`bulk.coeff_stats(t, mean, std)` now takes the chart's mean/std explicitly
(defaults to `t`'s own, preserving `build_flow`'s init-time behaviour).
`bulk.sync_chart_constants(flow, m_train=None)` takes the raw training
moments and, when given, recomputes `coeffs.u_mean`/`.u_white` against the
CURRENT (post-graft) chart and re-points every `ShearResponse` at them, in
the same `eqx.tree_at` call as the existing three fields. `m_train=None`
keeps the old (stale) behaviour for any caller that doesn't have the data at
hand. `shear.py`'s and `centroid.py`'s `--init` call sites now pass
`m_train=train_set[0]` / `m_train=m_train` respectively -- both already had
the raw moments in scope. New test:
`test_sync_chart_constants_follows_a_grafted_chart` (tests/test_shear.py)
extended to check `u_mean`/`u_white` follow the moved chart when `m_train` is
given and stay put when it isn't. Full `tests/test_shear.py` +
`tests/test_centroid.py`: 28/28.

### Retrain and measurement

Reused `flows/bulk_gauss2.eqx` (no `ShearResponse` in a bulk-only flow, so
unaffected). Pre-fix checkpoints saved to `flows/pre_usync/`. Rebuilt the
chain with the fixed `sync_chart_constants`:

`shear.py train --data ../bfd_cnf_imsims/data/gauss2_g0_1M.fits --flow
flows/shear_gauss2.eqx --init flows/bulk_gauss2.eqx --steps 60000
--deriv-weight 1e4 --seed 0` -- val NLL 42.41, dm/dg RMS residual/truth
0.26/0.60/0.25/0.25/0.78% (Mf/Mr/M1/M2/Mc), d2m/dg2 0.07/0.18/0.21/0.18/0.20%
-- both tighter than the u-stale `_sheardelta` chain's 0.29/0.66/0.32/0.32/0.87%
and 0.12/0.20/0.25/0.25/0.27%.

`centroid.py train --copies ../bfd_cnf_imsims/data/copies_gauss2_deep.fits
--flow flows/centroid_gauss2_deep_sheardelta.eqx --init flows/shear_gauss2.eqx
--steps 16000` -- converged cleanly (NLL 42.3-42.6, off-chart fraction
≤2e-3), same profile as before. Static ellipticity-response overshoot: catalog
+8.36e-3, layer +3.76e-2 (4.5x) -- consistent with the pre-existing, untouched
3.97x/4.36x defect noted above.

`bias.py --flow flows/centroid_gauss2_deep_sheardelta.eqx --pop gauss2_deep
--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536 --n-targets
20000 --window-size 2.2 3.2 --window-flux 2500 50000` (same flags as the prior
session's comparison; this OVERWRITES the u-stale numbers logged above under
the identical filename):

|  | windowed, uncorrected | windowed, corrected | unwindowed |
|---|---|---|---|
| pre-fix control | +0.00802 ± 0.00262 | −0.00851 ± 0.00347 | −0.02518 ± 0.00915 |
| `u`-stale (2026-08-25, `_sheardelta`) | −0.01405 ± 0.00140 | −0.02980 ± 0.00264 | +0.08211 ± 0.04946 |
| **`u`-synced (this session)** | **+0.00437 ± 0.00142** | **−0.01199 ± 0.00269** | **−0.01746 ± 0.00149** |

Two things moved. The unwindowed error bar shrank **33x** (±0.04946 →
±0.00149) -- the stale, ill-conditioned whitening was making the unwindowed
estimate essentially unusable, not just biased. And windowed-corrected m1
moved from −0.02980 back to −0.01199, roughly halving the gap to the pre-fix
control that the 2026-08-25 section called "not a tail artifact... remains
unexplained." It has NOT closed the gap (−0.01199 vs −0.00851 is still a
~1σ-ish difference given the ±0.0027-0.0035 errors, and the estimator
disagreement this whole investigation started from is presumably still
there) -- but a meaningful fraction of what looked like a genuine centroid-
layer effect was actually this staleness bug.

### Artifacts

- `flows/shear_gauss2.eqx`, `flows/centroid_gauss2_deep_sheardelta.eqx` --
  overwritten with the u-synced retrain.
- `flows/pre_usync/{shear_gauss2,centroid_gauss2_deep_sheardelta}.eqx` --
  the u-stale checkpoints, preserved for comparison.
- `logs_usync_shear_gauss2.txt`, `logs_usync_centroid_gauss2_deep_sheardelta.txt`,
  `logs_usync_bias_gauss2_deep_sheardelta.txt` -- repo root.

### Carried forward, not acted on

Same open design question as 2026-08-25, now with a THIRD instance behind it:
should `RawMomentStandardize.mean/std` be trainable at all? Three frozen
copies (`ShearResponse.e_scale/chart_loc/chart_scale`,
`CentroidMarginalize.mean/std/shear`, and now
`ShearResponse.coeffs.u_mean/.u_white`) have all had to be hand-resynced
after the fact, on three separate occasions, because the thing they copy is
trainable. `sync_chart_constants` fixes all three now, but there is no
mechanical guarantee a fourth won't be added the same way. Still left for the
user to call.

Not re-investigated: the outlier mechanism, the ellipticity-response
overshoot, and whether the residual −0.01199 vs −0.00851 gap is real or noise
-- next natural step is a seed-repeat of this exact retrain (per the
project's rule to quote gauss2 m1 with seed spread, not a single run) before
reading anything further into the remaining gap.

## 2026-08-26 (cont.): selection-correction magnitude -- checked, not
overcorrecting

The u-sync retrain's windowed m1 (uncorrected +0.00437 -> corrected
−0.01199, a −0.0164 shift) looked suspicious: a correction is moving an
already-near-zero number substantially further from zero. Investigated
whether `bias.ghat`'s eq.(45)-(46) `ns` term (`n_ns * (Q_s Q_s^T/(1-P_s)^2 +
R_s/(1-P_s))`, added to `R`) is oversized.

It is not, checked three ways (temporary diagnostics added to and then
removed from `bias.py`/`bias.selection_terms`, not committed):

1. **`R_s`'s own precision.** It drives ~99.97% of the correction (`Q_s` is
   consistent with zero by isotropy and contributes ~0.03%). Chunk-to-chunk
   SEM over the ~1M-row template catalog: R_s = 0.5367 ± 0.0014 (0.26%).
2. **Cross-validated two independent ways.** `--window-terms templates`
   (exact bfd `dm/dg` lensing of the template catalog) gives R_s = 0.537,
   P_s = 0.632; `--window-terms flow` (draws from the trained flow's own
   density) gives R_s = 0.549, P_s = 0.628 -- agreement to ~2%, from methods
   that share nothing but the population.
3. **The correction is genuinely small in absolute terms**: `|R_corr|` is
   only ~1.65% of `|R_kept|` (measured: R_kept_p norm 9.11e5, R_corr_p norm
   1.50e4). The large m1 shift is arithmetic, not mis-sizing: `n_ns/(1-P_s)
   ≈ N_total` for both arms (they differ by only ~0.18%, via `n_ns`), so the
   correction acts as a near-COMMON multiplicative shrink on `ghat` in both
   arms -- and because `m1 = (gp-gm)/(2g) - 1` references a "-1" baseline
   rather than a pure difference, a small common-mode R change lands almost
   entirely in m1 instead of cancelling between the +g/-g arms. This is
   exactly what the paper's derivation (a genuine incomplete-data likelihood
   term for `N_ns` known non-detections) predicts, not a bug.

**The positive signal:** windowed-corrected (−0.01199 ± 0.00269) and the
independent unwindowed estimate (−0.01746 ± 0.00149) now agree to ~1.8σ,
versus >10σ apart before the u-sync fix. A correctly-sized selection
correction reconciling two independently-computed estimators is what should
happen -- not what overcorrection looks like.

**Conclusion: the selection-correction machinery is cleared as the source of
the remaining ~−0.015 residual.** The search for that residual moves
elsewhere. `flows/`, `bias.py`, and `bias.selection_terms` are unchanged by
this investigation (diagnostics were added and then reverted); no new
retrain was needed.

## 2026-08-26 (cont.): the remaining residual is real, not the estimator --
self-consistency control comes back clean

Ran `dev/check_selfconsistency_noisy.py` against the u-synced checkpoint
(`flows/centroid_gauss2_deep_sheardelta.eqx --centroid --train-data
gauss2_g0_1M.fits --cov-from targets_gauss2_deep_g0_200k.fits -n 20000
--samples 8192 --alpha 0.5`, matching the `bias.py` run's depth/samples/alpha):
targets drawn straight from the flow itself, so the model is exact by
construction and the pipeline's own numerics (streaming IS, ESS, jackknife)
are the only thing left that could produce a nonzero m1.

**Overall control m1 = +0.0043 ± 0.0022 -- consistent with zero.** This rules
out the estimator/numerics as the source of the ~−0.015 residual (unwindowed
−0.01746 ± 0.00149, windowed-corrected −0.01199 ± 0.00269): if streaming IS or
ESS starvation at this depth/sample count were producing a spurious m1, the
control would show it too, since it runs the identical pipeline. It doesn't.

(The control's OWN per-bin printout shows a −0.174 ± 0.012 "hump" swinging
+0.077 -> −0.150 across Mr/Mf -- this is the KNOWN hidden-variable-binning
artifact from [[noisy-binned-m1-is-half-methodology]]/`check_selfconsistency_
noisy.py`'s own docstring: binning on the TRUE Mr/Mf, invisible to a noisy
estimator, biases per-bin numbers even under a perfect model. Not new, not
evidence of anything here -- the OVERALL (unbinned) control number is what's
diagnostic, and it's clean.)

**Conclusion: the remaining bias is real** -- a genuine mismatch between the
trained flow's density/response and the true simulated gauss2_deep
population, not a measurement/estimator artifact. Leading candidate, already
on record but never chased: `centroid.py check`'s static (g=0)
ellipticity-response overshoot, catalog +8.3649e-3 vs layer +3.7583e-2 -- a
**4.5x mismatch**, flagged as "pre-existing, untouched" across the last three
sessions' logs without anyone testing whether it's actually the driver of the
shear-bias residual. Natural next step, not yet done: perturb or ablate
whatever produces that overshoot (or scale it down post-hoc) and re-measure
m1, to see if it moves.

## 2026-08-26 (cont.): the ellipticity-response overshoot is not a red
herring -- it's population-wide and concentrates exactly where the m1
residual does, but the causal link to m1 is still inferred, not measured

Chased the lead above with `flows/centroid_gauss2_deep_sheardelta.eqx` (the
u-synced checkpoint) and `copies_gauss2_deep.fits`, ad hoc script off
`centroid.py check`'s own `weighted_copy_mean`/`dm_dsigma` call, not
committed (per-galaxy breakdown of the same `resp()` the CLI already prints
an aggregate of). n = 4000 galaxies, same as `check`'s default.

**Not diagnostic noise or an outlier artifact.** A 400-resample bootstrap
over galaxies gives layer resp = 0.0377 ± 0.0024, catalog resp = 0.0084 ±
0.0003, difference 0.0293 ± 0.0022 -- ~13σ from zero, an order of magnitude
past anything a noise floor could produce. It is also NOT concentrated in a
few extreme galaxies the way the shear-conditioning outlier mechanism
(2026-08-25, above) was: the top 20 |contribution| galaxies carry only 18%
of the total, top 100 only 42%, and trimming the most extreme 1% or 5% of
galaxies by |e| moves the layer number by <3% (0.0376 -> 0.0364 at 1%,
0.0369 at 5%) while the catalog number barely moves at all. This is a broad,
population-wide overshoot, not a spiky-few-galaxies problem like the g-swing
issue was.

**Strongly concentrated at exactly the population segment the shear-bias
residual has always concentrated in.** Binned the same per-galaxy `resp()`
by flux and by Mr/Mf quintile (same invariants `bias.py` already bins m1 by):

| flux quintile (faint->bright) | catalog resp | layer resp | overshoot ratio |
|---|---|---|---|
| 0 (faintest) | +2.80e-2 | +1.31e-1 | 4.7x |
| 1 | +7.12e-3 | +2.84e-2 | 4.0x |
| 2 | +3.34e-3 | +1.38e-2 | 4.1x |
| 3 | +1.58e-3 | +7.31e-3 | 4.6x |
| 4 (brightest) | +5.49e-4 | +1.41e-3 | 2.6x |

| Mr/Mf quintile (compact->extended) | catalog resp | layer resp | overshoot ratio |
|---|---|---|---|
| 0 (most compact) | +1.39e-2 | +8.23e-2 | 5.9x |
| 1 | +9.08e-3 | +4.15e-2 | 4.6x |
| 2 | +7.30e-3 | +2.31e-2 | 3.2x |
| 3 | +5.80e-3 | +1.93e-2 | 3.3x |
| 4 (most extended) | +5.25e-3 | +1.90e-2 | 3.6x |

The absolute size of the overshoot (layer minus catalog) is ~12x larger in
the faintest quintile than the brightest, and the compactness gradient
(quintile 0 vs 4) tracks the same "near the resolution/support edge" region
flagged by `[[mrmf-floor-is-misalignment-not-fixable]]`,
`[[support-edge-is-sparse-tail-score-error]]`, and
`[[resolution-edge-bias-is-real]]` in past sessions -- exactly where the
shear-response fit (a different layer, but trained the same way on the same
sparse tail) is independently known to degrade.

**Corroborating, not new: the 2026-08-24 2x2 ablation** (commit `e85d2f7`)
already showed centroid OFF makes the two shear-flow variants agree
(-0.00771 ± 0.00264 vs -0.01084 ± 0.00268) while centroid ON is where the
+0.0153/-0.0298-ish numbers and the windowed/unwindowed estimator
disagreement both live. That is consistent with the layer being the driver,
though (as that entry itself notes) it is not by itself proof -- `--no-
centroid` measures a structurally different model (no Sigma_X
marginalisation at all), so a null result there wouldn't have exonerated the
layer, and a positive one doesn't yet pin down the mechanism either.

**Assessment: this is very unlikely to be a red herring, but the causal link
from "wrong E[shift] at g=0" to "wrong m1 at g=0.02" is still inferred by
correlation, not shown mechanistically.** What was NOT done, and is the
honest next step, given time spent tonight: (a) an actual perturb-and-
remeasure -- scale the layer's predicted shift by e.g. 8.36e-3/3.76e-2 ≈
0.22 post-hoc (no retrain needed, `dm_dsigma`'s output can be rescaled
before it's used) and rerun `bias.py` to see if m1 moves toward the pre-fix
control; (b) distinguishing fitting (undertrained on the sparse faint/
compact tail -- same shape as `[[shear-layer-undertrained-at-6k]]`) from
structural (the coefficient net's `_COEFF_MAX` bound saturating there, same
shape as `[[centroid-tanh-gradient-trap]]`/`[[rho-coefficient-saturates-
both-populations]]`) -- a first attempt at reading `_Coeffs`' `base` output
against `_COEFF_MAX` on this same faint quintile hit a shape bug in the ad
hoc script and was not resolved tonight, so this is open, not answered.

### Artifacts

Ad hoc script, NOT saved to the repo (session `/tmp` scratchpad only):
per-galaxy breakdown of `centroid.py check`'s `resp()`, bootstrap SE, and
flux/Mr-Mf-quintile binning, all against `flows/centroid_gauss2_deep_
sheardelta.eqx` + `copies_gauss2_deep.fits`. No flow, no `flows/`, no
`bias.py`-facing code changed.

## 2026-08-26 (cont.): perturb-and-remeasure -- the overshoot is causal, not
just correlated

Ran the step-1 experiment the last entry deferred. `models/centroid.py`'s
`response()` now scales `t0`/`t2` (both `shift` and `de_raw` are linear in
them before their saturating nonlinearities) by `_RESPONSE_SCALE`, an
`os.environ["CENTROID_RESPONSE_SCALE"]`-controlled knob defaulting to 1.0
(no-op; committed as such, `tests/test_centroid.py` 14/14 unaffected). This
is IN the generative path `bias.py` differentiates through, not a diagnostic-
only rescale.

Set it to 0.2226 (= catalog/layer ellipticity response, 8.3649e-3/3.7583e-2)
and reran `bias.py` on the identical 20k targets/window as the u-synced
baseline logged above (`--samples 8192 --alpha 0.5 --chunk 4096 --batch-
budget 65536 --window-size 2.2 3.2 --window-flux 2500 50000`):

|  | windowed, uncorrected | windowed, corrected | unwindowed |
|---|---|---|---|
| u-synced baseline (scale=1.0) | +0.00437 ± 0.00142 | −0.01199 ± 0.00269 | −0.01746 ± 0.00149 |
| **rescaled (scale=0.2226)** | **+0.00819 ± 0.00142** | **−0.00831 ± 0.00268** | **−0.01117 ± 0.00152** |
| pre-fix control (reference) | +0.00802 ± 0.00262 | −0.00851 ± 0.00347 | −0.02518 ± 0.00915 |

Both estimators moved substantially toward zero in the same direction, by
far more than the error bars: windowed-corrected −0.01199 → −0.00831 (lands
almost exactly on the pre-fix control's −0.00851), unwindowed −0.01746 →
−0.01117 (36% of the way to zero). `centroid.py check` at this scale:
ellipticity response layer +6.99e-3 vs catalog +8.36e-3 (now a slight
undershoot, since a single uniform scale calibrated to the ellipticity ratio
doesn't separately correct Mf/Mr/Mc, which had their own, different
overshoot ratios before scaling).

**Conclusion: causal, not coincidental.** The overshoot is a real
contributor to the m1 residual -- this is the first result in the
investigation that moves m1 by actually touching the suspected mechanism,
rather than by correlation or ablation-of-a-different-model. It does not
fully close the gap (both numbers are still negative, and the rescale was a
crude uniform multiplier, not a fit), so there is likely still more than one
thing wrong -- but the centroid layer's Sigma_X-response magnitude is now an
established, not just suspected, piece of the ~-0.015 residual.

Not yet done: fitting the RIGHT scale (per-moment, not a single global
ellipticity-calibrated number) and re-measuring, and the fitting-vs-
structural question (undertraining vs. `_COEFF_MAX` saturation) on the
faint/compact tail where the overshoot concentrates -- both carried forward
to `.session/next_prompt.md`.

### Artifacts

- `logs_perturb_scale0.2226.txt` -- repo root, the rescaled `bias.py` run.
- The `_RESPONSE_SCALE` env-var knob used for this test was REMOVED after
  the run (user instruction: no tuning parameters in the final
  implementation). `models/centroid.py` is back to its pre-experiment state;
  `tests/test_centroid.py` 14/14 confirmed clean after the revert.

## 2026-08-26 (cont.): root cause -- the layer's first-order-in-T expansion
is invalid exactly where the overshoot lives, and it is not a capacity or
training-duration problem

Measured `T0 = (Mr/Mf) * Sigma_u` (the dimensionless perturbation parameter
`response`'s docstring and `_COEFF_MAX`'s own header comment assume is
"~1e-3, so nothing real comes close to \[the bound\]") directly on
`copies_gauss2_deep`'s `galaxies["moments"]`, by flux quintile:

| flux quintile | T0 median | T0 max |
|---|---|---|
| 0 (faintest) | 0.0267 | 0.571 |
| 1 | 0.0093 | 0.018 |
| 2 | 0.0046 | 0.008 |
| 3 | 0.0021 | 0.004 |
| 4 (brightest) | 0.00078 | 0.0016 |

The brightest quintile matches the assumed ~1e-3 scale. The faintest reaches
T up to **0.57 -- over 500x** that scale. This follows directly from
`displacement_covariance`/`_tensor`: `Sigma_u ~ Sigma_X / Mr^2` and
`T ~ Sigma_X / (Mr . Mf)`, so faint AND compact galaxies (small Mr, small
Mf) blow T up -- exactly the population segment (flux quintile 0, Mr/Mf
quintile 0) where the ellipticity-response overshoot concentrates (prior
entry, above).

Two capacity/training explanations were checked and ruled out:
- **Not `_COEFF_MAX` saturation.** `|base coefficient| / _COEFF_MAX` by
  flux quintile: mean fraction 0.1648 (faintest) rising MONOTONICALLY to
  0.3461 (brightest) -- the opposite direction from the overshoot. The net's
  output is not pinned against its bound where the overshoot lives; it has
  headroom it isn't using.
- **Not generic undertraining.** An undertrained net would show scatter/
  noise broadly, not a clean monotone flux dependence that lines up exactly
  with where the model's own stated small-T assumption is violated.

**Root cause: `response()`'s `shift = s0 @ [t0, Re(ec.t2)]` is LINEAR in T,
then merely saturated by `tanh` at `_EXP_MAX` post-hoc -- it is a first-order
Taylor approximation to the true marginalisation integral (eq. 36), valid
only near T = 0.** At T ~ 1e-3 (bright quintile) the linear term is the
whole story and the fit is accurate. At T ~ 0.03-0.5 (faint quintile) the
TRUE nonlinear marginalisation self-averages/saturates (the copy weights
spread mass over many origins), while the fitted linear-in-T coefficient,
optimised by NLL against training data that includes this same large-T
tail, keeps extrapolating past where the linear approximation holds --
producing exactly the observed pattern: catalog response flattens (+0.0084
aggregate) while the model's linear term overshoots it (+0.0376). The
`tanh(shift/_EXP_MAX)` saturation caps the SIZE of the output but does
nothing about the WRONG FUNCTIONAL FORM of the input to it -- it was sized
to stop the map from ceasing to be a diffeomorphism, not to correct
first-order-in-T error.

**This is a structural defect in the model, not a fitting one.** No amount
of extra `centroid.py train` steps or capacity fixes it -- the functional
form itself (first order in T) does not represent the population's true
response once T leaves the ~1e-3 regime the model was built around. A real
fix needs a higher-order or explicitly-saturating dependence on T itself
(not merely on the post-multiplication `shift`) -- not attempted this
session; this is a design decision for the user, not something to retrain
into the current architecture.

### Artifacts

None -- diagnostic only (T0/T2 magnitude, `_COEFF_MAX` saturation, all by
flux quintile), ad hoc `/tmp` scripts, not saved. No code changed by this
entry (the `_RESPONSE_SCALE` knob from the previous entry was already
reverted before this investigation).

## 2026-08-26 (cont.): the T-saturation fix -- implemented, retrained,
partial close (windowed-corrected -0.01199 -> -0.00966, unwindowed
-0.01746 -> -0.01499), and a new asymmetry it introduces

Implemented the fix the previous entry left open: `response()`
(`models/centroid.py`) no longer feeds T = (t0, t2) into `shift`/`de_raw`
raw. It now saturates T ITSELF first, `t0 -> _rational_bound(t0, t0_scale)`
and the equivalent complex-magnitude form for `t2`, with `t0_scale, t2_scale
> 0` two NEW outputs of `_Coeffs` (via `softplus`), learned per (f, a, b, q)
like every other coefficient -- not a hand-picked constant. Design reasoning:

- The diagnosed defect was the LINEAR-in-T Taylor term extrapolating past
  where it's valid (T ~ 0.03-0.57 vs. the ~1e-3 regime it's accurate in);
  `_EXP_MAX`'s `tanh` already caps the OUTPUT magnitude but does nothing
  about the wrong functional form of the input, so this session's other
  option (a hand-derived 2nd-order term from the eq. 36 integral) was
  rejected in favour of matching the true integral's qualitative behaviour
  (self-averaging/saturating at large T) with a learnable saturation on T,
  cheaper and no new physics derivation needed.
- T = 0 (Sigma_X = 0, or a round galaxy for t2) saturates to 0 at ANY scale
  value, so `test_identity_at_zero_sigma_with_shear`'s no-op and
  `test_response_is_first_order_in_sigma_x`'s small-T linearity both still
  hold with no special-casing -- confirmed, both pass (14/14 in
  `tests/test_centroid.py`, updated for `_Coeffs`'/`response`'s new return/
  argument, 6/6 in `tests/test_shear.py` untouched).
- `t0_scale, t2_scale` depend only on (f, a, b, q), NOT `da, dq`: saturation
  is a property of the marginalisation integral, not of shear-conditioning,
  so giving it a slope on da, dq would be capacity with no motivation.

Retrained at the established floor, `centroid.py train --copies
../bfd_cnf_imsims/data/copies_gauss2_deep.fits --flow flows/centroid_
gauss2_deep_sheardelta.eqx --init flows/shear_gauss2.eqx --steps 16000`
(NLL 42.3-42.6, off-chart <=2e-3, same convergence profile as every prior
run of this recipe -- no step-count search).

**`centroid.py check`: the static ellipticity-response overshoot narrowed,
did not close.** Catalog +8.3649e-3 (unchanged, a property of the fixed
catalog), layer +2.8521e-2 -- 3.4x, down from the u-synced baseline's 4.5x
(+3.7583e-2). **New asymmetry the fix introduced: the three spin-0 shifts
(Mf, Mr, Mc) flipped from OVERSHOOTING the catalog to UNDERSHOOTING it**,
e.g. Mr layer -2.268e-3 vs. catalog -1.131e-2 (a ~5x undershoot, was an
overshoot before); Mf -5.896e-4 vs. -5.922e-3 (~10x); Mc -3.992e-3 vs.
-1.592e-2 (~4x). A single learned T-saturation scale per invariant, shared
by whatever mixture of galaxies trains it, does not separately get the
spin-0 and spin-2 responses right at once -- the same shape of limitation
`--RESPONSE_SCALE`'s uniform multiplier had (2026-08-26, "perturb-and-
remeasure"), now inside the trained model instead of a post-hoc knob.

**`bias.py`, identical 20k-target/window config throughout this
investigation** (`--pop gauss2_deep --samples 8192 --alpha 0.5 --chunk 4096
--batch-budget 65536 --n-targets 20000 --window-size 2.2 3.2 --window-flux
2500 50000`):

|  | windowed, uncorrected | windowed, corrected | unwindowed |
|---|---|---|---|
| u-synced baseline (pre-fix) | +0.00437 +/- 0.00142 | -0.01199 +/- 0.00269 | -0.01746 +/- 0.00149 |
| **T-saturation fix** | **+0.00676 +/- 0.00141** | **-0.00966 +/- 0.00268** | **-0.01499 +/- 0.00203** |
| pre-fix control (reference) | +0.00802 +/- 0.00262 | -0.00851 +/- 0.00347 | -0.02518 +/- 0.00915 |

Both estimators moved toward the control/toward zero, by more than the
error bars but not by much: windowed-corrected covered ~19% of the gap from
baseline to the control value (-0.01199 -> -0.00966 against a -0.00851
target), unwindowed covered ~14% of the gap to zero (-0.01746 -> -0.01499).
**Neither closed. Windowed-corrected is still -0.00966 +/- 0.00268, 3.6sigma
from zero; unwindowed is -0.01499 +/- 0.00203, 7.4sigma from zero.** This is
a smaller move than the crude uniform `_RESPONSE_SCALE = 0.2226` rescale
produced (that landed windowed-corrected almost exactly on the control,
-0.00831) -- consistent with the check-level result above: a single
per-invariant learned scale is a blunter instrument than a scale hand-tuned
to the ellipticity ratio specifically, and it now has to serve Mf/Mr/Mc too,
which it visibly does not get right at the same time (the new undershoot).

**Assessment: real, structural, and correctly targeted -- but a partial
fix, not a resolution.** The direction and mechanism are validated twice
now (the crude rescale in the prior entry, and this trained, principled
version), so the T-saturation diagnosis itself is confirmed, not merely
plausible. What is NOT resolved:

- The spin-0/spin-2 tradeoff above -- a shared `t0_scale` is being asked to
  fit both the Mf/Mr/Mc shifts (which now undershoot) and, via the same t0
  entering `shift`'s `s0` combination, contribute to the ellipticity
  response (which still overshoots). Separate scales per OUTPUT rather than
  per INPUT (t0), or a scale that is itself a function of da, dq after all,
  were not tried.
- Whatever remains of the ~-0.015 unwindowed / ~-0.010 windowed-corrected
  residual after this fix is still 3.6-7.4sigma from zero and needs a next
  lead, not a re-tune of this one -- this session did not identify what
  that is.

### Artifacts

- `models/centroid.py`: `_Coeffs.__call__` now returns `(coeffs, t0_scale,
  t2_scale)`; `response`'s signature gained `t0_scale, t2_scale` before
  `z`; both committed, not knobs.
- `tests/test_centroid.py`: `test_response_is_first_order_in_sigma_x`
  updated for the new return/argument shapes; no test's intent changed.
- `flows/centroid_gauss2_deep_sheardelta.eqx`: retrained, OVERWRITES the
  u-synced checkpoint the baseline numbers above were measured on.
- Training log: `/tmp` scratchpad only, not committed (`step ... nll ...`
  plus the `check`-style shift table this run's `centroid.py train` prints
  at exit).
- `bias.py` log: `/tmp` scratchpad only, not committed.

## 2026-08-27: the centroid layer is now an exact, zero-parameter analytic
transport -- derivation, four implementation bugs, and the residual mostly
closes

Replaced the entire coefficient-net response in `models/centroid.py`
(`_Coeffs`, `response`, `_tensor`, `shear_delta_invariants`) with a
closed-form transport derived from bfd's own moment definitions, at the
user's explicit direction to "do it right" rather than patch the
first-order model again. Full derivation, validation, and design discussion
are in this session's conversation; summary below.

### The physics

bfd moments are `M_a(u) = Re[Sum_k W(k) kernel_a(k) I(k) e^{ik.u}]`
(`bfd/momentcalc.py`), so averaging over the target's own centroid-error
distribution `u ~ N(0, Sigma_u)` (the SAME linearisation this module always
used) is EXACTLY a Gaussian damping `exp(-1/2 k^T Sigma_u k)` applied in
k-space -- no profile-shape assumption needed for that step. Modelling
`W(k) I(k)` itself as Gaussian in k (exact for a single Gaussian profile
times a Gaussian weight; `gauss2` is literally a Gaussian mixture) then
gives a closed form for all five moments in terms of one real-space "width"
matrix `R`, solved EXACTLY from the galaxy's own measured `(Mf, Mr, M1,
M2)`:

    R = (1/(2 Mf)) [[Mr+M1, M2], [M2, Mr-M1]]     (== -J/Mf, bfd's own
                                                     xyJacobian)
    P = R @ Sigma_u
    R~ = (I+P)^-1 @ R          (NOT R @ (I+P)^-1 -- see bug 1 below)
    Mf' = Mf/sqrt(det(I+P)),  Mr' = Mf'*tr(R~),  M1' = Mf'*(R~xx-R~yy),
    M2' = Mf'*2 R~xy
    Mc' = Mc + [(2Mr'^2+M1'^2+M2'^2)/Mf' - (2Mr^2+M1^2+M2^2)/Mf]

`unmarginalize` (data -> base) is the SAME formula with `P -> -P` and `R`
solved from the DATA point's own moments -- exact by the self-consistency
of the Gaussian-in-k family under composition of dampings, so there is no
fixed-point iteration left in this layer at all. Zero free parameters: `R`
comes from each galaxy's own measurement, not a population fit, so nothing
here needs recalibrating for a different real population -- the only thing
that changes is how good the Gaussian-in-k approximation is, which is
diagnosable (the `Mc` self-consistency ratio) rather than assumed. No
shear-conditioning: unlike the retired net, which modelled a
population-averaged `E[marginalisation | m]` and so needed to know shear
shifts which profiles land at a given m, `_transport` solves the ansatz
exactly for the one galaxy in front of it -- there is no population mixture
whose composition shear could change.

### Four bugs, caught before and during validation

1. **Matrix ordering.** First implementation used `R @ inv(I+P)` for `R~`;
   the correct form is `inv(I+P) @ R` (they coincide only when `R` and
   `Sigma_u` commute, i.e. only for ISOTROPIC `Sigma_X` -- exactly the case
   every existing test and the `gauss2_deep` catalog uses, so it would have
   shipped silently broken for any future anisotropic-PSF population).
   Caught by hand-deriving the correct symmetric form and checking it
   numerically before writing the JAX code; a NEW test
   (`test_round_trip`) now exercises a genuinely anisotropic `Sigma_X`
   specifically to keep this from regressing silently.
2. **`_safe_logit`'s rational bound.** The chart's `logit(Mr/(r* Mf))` /
   `logit(Mc/(rc* Mr))` need a NaN guard (`_transport`'s output can exceed
   the point-source ceiling for a real, not-exactly-Gaussian galaxy). A
   smooth `_rational_bound(2v-1, ~1)` rescale seemed like the "no dead
   gradient" answer, but real galaxies span most of `(0,1)`, not a
   neighbourhood of 0.5, so the bound measurably distorted EVERY target,
   not just edge cases -- measured as a median z-space shift of ~0.5-1.2
   (should be ~1e-3, matching `T`'s own scale) on
   `targets_gauss2_deep_g0_200k.fits`. Fixed by reverting to a plain
   `jnp.clip`, which is an exact no-op for any value not already at the
   boundary (a hard clip's zero gradient there is fine -- `bias.py` already
   drops non-finite Q/R, the correct outcome for a target this ansatz
   cannot represent).
3. **Missing output cap.** With no cap on `_transport`'s output magnitude,
   a rare barely-resolved target's EXACT (not wrong, just large) shift
   could land the frozen bulk+shear flow -- never retrained against this
   layer's new output distribution -- in a steep-curvature part of its own
   density and dominate `bias.py`'s Q/R sum past its outlier guard.
   Restored `_EXP_MAX = 0.5`, the retired architecture's own constant
   (reused, not re-derived), as a `tanh`-saturated cap on the z-space
   output of both `unmarginalize` and `marginalize`.
4. **Peeling a chained layer broke Q/R, not log_prob.** Since
   `CentroidMarginalize` no longer reads `g` at all, `bias.split_centroid`
   was changed to peel it unconditionally (previously refused for a
   chained `(5,)`-conditioned layer, because the OLD coefficient net
   genuinely depended on `g`). This is mathematically sound at the
   `log_prob` level -- checked directly, `flow.log_prob` and `peeled
   rest.log_prob(centroid_transform(...)) + log-det` agree to float32
   roundoff for well-behaved targets -- but `bias.py`'s actual Q/R (`jax.
   grad`/`jax.hessian` of log_prob w.r.t. g, taken THROUGH the
   reconstructed `rest = Invert(Chain(bij[2:]).merge_chains())`) came out
   catastrophically wrong (`gauss2_deep` windowed-corrected m1 ~ -0.68,
   c2 ~ -0.088) even with `_transport` monkeypatched to a literal identity
   -- proving the bug is in autodiff through the peeled sub-chain, not in
   this layer's math. **Root cause not found.** Fixed by restricting
   `split_centroid` back to standalone `(3,)`-conditioned layers only,
   matching the original architecture's behaviour (`bias.py`'s
   `split_centroid` docstring carries the full account). This is a
   real open question -- a latent autodiff/flowjax fragility when
   reconstructing a sliced sub-chain under `jax.hessian` -- worth
   understanding before this peeling pattern is trusted elsewhere; it is
   currently just avoided, not resolved. Cost of avoiding it: the
   g-autodiff now traverses this (parameter-free, closed-form, no large
   network) layer directly instead of hoisting its log-det out, which is
   fine since there is no more coefficient net to differentiate through.

### Validation

`tests/test_centroid.py`: 12/12, rewritten for the new architecture
(round-trip now checked at an anisotropic `Sigma_X`, which would have
caught bug 1; `test_saturates_instead_of_overflowing` checks finiteness,
not round-trip precision, in the regime the safety floor is actively
engaged). `tests/test_shear.py`: 6/6, unaffected.

`centroid.py check` (`copies_gauss2_deep.fits`, u-synced flow warm-started
from `flows/shear_gauss2.eqx`, zero training steps -- there is nothing left
to train):

|  | layer shift | catalog shift |
|---|---|---|
| Mf | -5.764e-3 | -5.922e-3 |
| Mr | -1.074e-2 | -1.131e-2 |
| Mc | -1.561e-2 | -1.592e-2 |
| ellipticity response | +5.872e-3 | +8.365e-3 |

Mf/Mr/Mc match the catalog to 2-5% with zero trained parameters (the old
architecture's best-fit numbers, after a full 16k-step train, were off by
3-6x on these same channels in the tail). Ellipticity response still
undershoots (0.70x), the honest residual of the Gaussian-in-k
approximation for a real (not exactly Gaussian) profile.

`bias.py`, identical 20k-target/window config throughout this
investigation (`--pop gauss2_deep --samples 8192 --alpha 0.5 --chunk 4096
--batch-budget 65536 --n-targets 20000 --window-size 2.2 3.2 --window-flux
2500 50000`):

|  | windowed, uncorrected | windowed, corrected | unwindowed |
|---|---|---|---|
| pre-fix control | +0.00802 +/- 0.00262 | -0.00851 +/- 0.00347 | -0.02518 +/- 0.00915 |
| T-saturation fix (2026-08-26) | +0.00676 +/- 0.00141 | -0.00966 +/- 0.00268 | -0.01499 +/- 0.00203 |
| **analytic transport (this entry)** | **+0.01400 +/- 0.00139** | **-0.00279 +/- 0.00268** | **-0.00646 +/- 0.00153** |

**Windowed-corrected is now -0.00279 +/- 0.00268 -- 1.0sigma from zero**,
against 3.6sigma (T-saturation fix) and 2.5sigma (control). Unwindowed is
-0.00646 +/- 0.00153 -- 4.2sigma from zero, still not fully closed but
substantially better than either prior number (7.4sigma and 2.75sigma
respectively) and less negative than the control itself. By every measure
in this investigation, this is the best result to date -- most, though not
quite all, of the residual bias closes with a zero-free-parameter physics
model, not a fit.

**Assessment.** Not a full resolution: unwindowed m1 is still 4.2sigma from
zero, and the honest candidates for what remains are (a) the ellipticity
response's residual 30% undershoot (a real Gaussian-profile-approximation
error, not a bug -- see the `check` table), and (b) whatever the pre-fix
control's own -0.0043 +/- 0.0022 self-consistency floor already represented
(`[[residual-bias-is-real-not-estimator]]`). Bug 4 (peeling) is flagged
above as a genuinely open architectural question, not swept under the
"fixed" label -- it happened to not affect correctness once avoided, but
its root cause is unknown and could resurface elsewhere.

### Artifacts

- `models/centroid.py`: rewritten -- `_transport`, `_safe_logit`,
  `_rational_bound`/`_rational_scale`, `raw_from_standard`/
  `standard_from_raw` extended for Mc, `CentroidMarginalize` reduced to
  `(mean, std)` only (no coefficients, no shear reference).
- `bulk.py`: `CentroidMarginalize` construction drops `shear=`; `sync_
  chart_constants` no longer re-points a (now nonexistent) `.shear` copy.
- `centroid.py`: `train()` detects "nothing trainable" (correctly, via
  paramax's `NonTrainable` `is_leaf`, not the pre-existing `_trainable`
  helper's boolean spec, which does not use it) and skips the loop rather
  than burning steps on batch noise.
- `bias.py`: `split_centroid` restricted to standalone layers (bug 4);
  `layer_is_g_conditioned` removed (its check is now load-bearing for a
  different reason than originally written, absorbed inline).
- `tests/test_centroid.py`: rewritten, 12/12 (see Validation above).
- `flows/centroid_gauss2_deep_analytic.eqx`: the new checkpoint these
  numbers were measured on (u-synced bulk+shear warm-started from `flows/
  shear_gauss2.eqx`, centroid layer has no weights to train).
- `bias_final20k.log`, `bias_capped.log`, `bias_reverted.log` etc.: `/tmp`
  scratchpad only, not committed.

## 2026-08-27 (cont.): the analytic transport generalises to bulgedisc --
`check` confirms the physics, `bias.py` is noisy but not broken

User asked to test the new analytic centroid transport on `bulgedisc`, the
population it was NOT designed around -- `gauss2` is literally a Gaussian
mixture, `bulgedisc` is bulge+disc profiles, a real test of the
Gaussian-in-k approximation's robustness.

### Retraining was needed first, and surfaced a real setup gap

`flows/shear_bulgedisc.eqx` (Aug 7) predates several chart/architecture
fixes since and would not even deserialise against current code. Retrained
from scratch:

- `bulk.py train --data moments.fits --steps 20000` -- clean, val nll 40.42.
- `shear.py train --steps 60000` with NO `--deriv-weight`: dm/dg residuals
  128-172% RMS/RMS truth, 520-4142% on d2m/dg2 -- unusable. `shear.py
  --help` documents this outcome exactly: NLL alone provably cannot
  identify the spin-0 response at any batch/steps/g_max, and
  `--deriv-weight 1e4` is not a tuning knob but a requirement. Missed on
  the first pass since gauss2's committed checkpoints already had it baked
  in from an earlier session and nothing in THIS session's workflow
  re-derived that requirement from scratch.
- Retrained with `--deriv-weight 1e4`: Mf 2.81%, Mr 26.12%, M1 2.09%, M2
  2.13%, Mc 40.73% dm/dg residual (d2m/dg2 2-25%). Mr/Mc residuals notably
  higher than gauss2's -- plausibly bulgedisc's own known bulge/disc
  misalignment floor (`[[mrmf-floor-is-misalignment-not-fixable]]`), not a
  new defect; not independently re-verified against that memory's specific
  numbers this session.

### `centroid.py check` (`copies_bulgedisc_deep.fits`) -- the physics holds up

|  | layer shift | catalog shift |
|---|---|---|
| Mf | -4.040e-3 | -4.124e-3 |
| Mr | -7.937e-3 | -8.020e-3 |
| Mc | -1.198e-2 | -1.156e-2 |
| ellipticity response | +1.1284e-2 | +1.7774e-2 |

Mf/Mr/Mc match the catalog to 2-4% -- as good as or better than gauss2_deep's
match, with STILL ZERO trained parameters. Ellipticity response undershoots
at 0.635x, in the same ballpark as gauss2_deep's 0.70x. **This is the
strongest evidence yet that the closed form is not overfit to gauss2's
Gaussian-mixture structure** -- bulge+disc profiles are a genuinely
different shape family and the approximation degrades only mildly, not
catastrophically.

### `bias.py` (`--pop bulgedisc_deep`, 20k targets, 8192 samples, UNWINDOWED
-- no window bounds have ever been established for this population)

    19974 targets: m1 = +0.1627 +/- 0.0735

By Mf quintile: q1 +0.044+/-0.063, q2 -1.22+/-6.10, q3 +0.32+/-0.76, q4
+0.035+/-0.020, q5 -0.0007+/-0.0059. The overall number is 2.2sigma from
zero but the error bar is dominated by q2/q3's essentially uninformative
variance (a handful of unstable targets, not a systematic pull -- q1/q4/q5,
which have real precision, all sit within ~1sigma of zero). Contrast with
gauss2_deep's OWN unwindowed number this session, -0.00646 +/- 0.00153,
which was tight without any window at all -- the difference here is not
"missing a window fixes it" so much as bulgedisc_deep genuinely having
noisier/more outlier-prone Q,R at this sample depth, consistent with it
being the harder, more realistic population throughout this project's
history. No prior bulgedisc_deep bias.py m1 baseline was found in
HANDOFF/HANDOFF_archive to compare against directly.

**Assessment: inconclusive on `bias.py`, not evidence of a defect.** The
`check` results are the informative signal here and they are genuinely
good. The `bias.py` number is too noisy at this (unwindowed) depth to say
anything about the residual bias one way or the other -- establishing a
bulgedisc-appropriate window (mirroring the tuning gauss2_deep's window
went through over many prior sessions) would be the natural next step if
more precision is wanted here, not attempted this session since it was not
asked for.

### Artifacts

- `flows/bulk_bulgedisc.eqx`, `flows/shear_bulgedisc_fresh.eqx`: fresh
  checkpoints, current code, `--deriv-weight 1e4`.
- `flows/centroid_bulgedisc_deep_analytic.eqx`: warm-started from the above,
  no centroid weights to train.
- `flows/shear_bulgedisc.eqx` (Aug 7): left in place, untouched, now known
  stale -- superseded by `shear_bulgedisc_fresh.eqx` for anything current.
- `shear_bulgedisc_train.log` (broken, no deriv-weight), `shear_bulgedisc_
  train2.log`, `bias_bulgedisc2.log`: `/tmp` scratchpad only, not
  committed.

## 2026-08-27 (cont.): `bulgedisc_deep` q2/q3 instability isolated to the
centroid layer, not ESS -- alpha=1 confirmed unreliable, the old T0
mechanism ruled out, root cause still open

Follow-up to the previous entry's "inconclusive" `bias.py` number. Added a
quintile-stratified ESS/drop-count diagnostic to `bias.py` (the single
2000-target ESS sample was replaced with a per-quintile stratified sample;
`sane_targets`' finite/outlier drop counts are now also broken out by flux
quintile) and ran four `--pop bulgedisc_deep --samples 8192 --chunk 4096
--batch-budget 65536 --n-targets 20000` configurations.

### Finding 1: `--alpha 1.0` (the default) is ESS-starved here, exactly as
established for `gauss2_deep` at this same noise depth -- don't trust it

|  | median ESS | 5th pct ESS | frac(ESS<10) | m1 |
|---|---|---|---|---|
| alpha=1.0 (default), with centroid | 71 | 4 | 0.168 | -0.0951 +/- 0.0473 |
| alpha=0.5, with centroid | 396 | 65 | 0.003 | +0.1627 +/- 0.0735 |

The two disagree in SIGN and by more than either's stated error. This is
not new physics -- `[[alpha1-ess-does-not-scale]]` already established
alpha=0.5 as the honest choice for `gauss2_deep`'s noise_sigma (2.73), and
`bulgedisc_deep` uses the identical depth/kernel. The alpha=1 number above
should be discarded as ESS-starved-estimator bias, not treated as a second
independent measurement; alpha=0.5 is the one to quote.

Per-quintile ESS at alpha=1 (the starved case) shows q2/q3 -- the two
unstable quintiles from the prior entry -- have the worst tails (frac<10 =
0.297, 0.220) against q1/q4/q5's 0.15, 0.15, 0.02. At alpha=0.5 this
concentration disappears entirely: q1-q5 median ESS are 413/552/357/232/328,
all frac<10 <= 0.01. **ESS is not what is driving q2/q3's instability once
alpha=0.5 is used** -- the alpha=1 quintile pattern was itself just a
symptom of general starvation, not a clue to the real cause.

### Finding 2: with ESS controlled for (alpha=0.5 throughout), the centroid
layer itself -- not sampling -- is what blows up q2/q3

Ran the `--no-centroid` baseline this population never had on disk
(`flows/shear_bulgedisc_fresh.eqx`, `--no-centroid`, same alpha=0.5 config):

| Mf quintile | no centroid | with centroid (analytic transport) |
|---|---|---|
| q1 | +0.089 +/- 0.121 | +0.044 +/- 0.063 |
| q2 | -0.185 +/- 0.055 | -1.221 +/- 6.100 |
| q3 | -0.052 +/- 0.011 | +0.321 +/- 0.759 |
| q4 | -0.026 +/- 0.008 | +0.035 +/- 0.020 |
| q5 | -0.021 +/- 0.010 | -0.001 +/- 0.006 |
| **overall** | **-0.0361 +/- 0.0132** | **+0.1627 +/- 0.0735** |

Both runs have essentially identical ESS statistics (ESS is computed on the
pre-peel kernel draws, upstream of whether the centroid layer is even
applied) -- so ESS cannot explain this. q1, q4, q5 are comparable or
slightly better with the centroid layer on. **q2 and q3 alone inflate by
~100x in variance when the centroid layer is switched on**, while
`sane_targets`' |Q|/|R| > 1000x-median guard drops only 0-6 targets per
quintile in either run -- a few outlier targets are surviving that guard
and dominating the sum, the same failure shape as bug 3 in the previous
entry (`_EXP_MAX` caps the z-space OUTPUT but a target can still land the
frozen flow in a steep-curvature pocket), just not fully caught here.

### Finding 3: it is NOT the T-saturation/large-T mechanism from the
gauss2_deep investigation

Checked the leading hypothesis directly: `T0 = trace(Sigma_X) / (Mr * Mf)`
(the same dimensionless perturbation parameter diagnosed in
`[[bfd-second-derivs-are-float32]]`'s neighbourhood, `models/centroid.py`'s
`response`-era T) computed on `targets_deep_g0_200k.fits`'s own moments and
`cov_odd`, by Mf quintile:

| quintile | T0 median | T0 p95 | T0 max |
|---|---|---|---|
| q1 | 0.0070 | 0.0348 | 13.818 |
| q2 | 0.0038 | 0.0052 | 0.0142 |
| q3 | 0.0024 | 0.0034 | 0.0081 |
| q4 | 0.0013 | 0.0019 | 0.0046 |
| q5 | 0.0003 | 0.0007 | 0.0022 |

T0 is monotone decreasing q1->q5 as expected (faint = large T), and q1
alone carries a genuine extreme tail (max 13.8, far past anything in
q2-q5). But **q1 is the STABLE quintile** in both tables above, while
q2/q3 -- whose T0 is modest and unremarkable, nowhere near q1's tail --
are the unstable ones. Whatever is inflating q2/q3 is not large-T,
barely-resolved targets in the sense the earlier gauss2_deep investigation
found; it is something else, not yet identified.

### Assessment

Not resolved. What's established:
- **Always use `--alpha 0.5` for `bulgedisc_deep`** (matches existing
  `gauss2_deep` practice at this depth) -- `--alpha 1.0`'s result is not
  usable evidence one way or the other.
- With alpha=0.5, the well-determined quintiles (q1, q4, q5) show the
  centroid layer is neutral-to-mildly-helpful, consistent with the `check`
  table's good match in the previous entry.
- q2/q3 carry a real, reproducible (not resampling noise -- confirmed
  across two independent full runs, this entry's and the previous entry's)
  variance blowup specific to the centroid layer, that is NOT explained by
  ESS starvation and NOT explained by the T-saturation mechanism that
  explained gauss2_deep's earlier residual. The overall unwindowed m1
  (+0.1627 +/- 0.0735) is not trustworthy as reported -- it is a weighted
  average dominated by two quintiles whose own numbers are close to
  uninformative.
- **Next step, not done here:** pull the specific q2/q3 targets whose |Q|
  or |R| survive `sane_targets`' 1000x-median guard but still dominate the
  per-quintile sum (via `--save-pqr` + an offline per-target norm scan,
  the same idiom `sane_targets`'s own docstring describes), and check what
  they share structurally (Mr/Mf specifically rather than Mf, ellipticity,
  bulge/disc flux ratio if available) -- Finding 3 above only rules out one
  candidate, it does not identify the real one. No bulgedisc-appropriate
  `--window-size`/`--window-flux` was established this entry; doing so
  blindly before the mechanism is known risks cutting around the symptom
  rather than the cause.

### Artifacts

- `bias.py`: `main()` -- the single-slice ESS diagnostic replaced with a
  flux-quintile-stratified one, reported per quintile; `sane_targets`'
  finite/outlier drop counts now also broken out by flux quintile when any
  are dropped. Diagnostic-only, no change to any Q/R/m1 computation.
- `flows/shear_bulgedisc_fresh.eqx` used with `--no-centroid`: no new
  checkpoint, first use of this flow without the centroid layer on
  `bulgedisc_deep`.
- Logs: `/tmp` scratchpad only (this session's own scratchpad, not the
  shared one referenced in the previous entry), not committed.

## 2026-08-27 (cont.): matching `bulgedisc` to real DES/COSMOS templates --
flux-size correlation fixed, ellipticity is a structural limit of the
Gaussian-profile family, not a knob

Follow-up to the q2/q3 investigation above: compared `bulgedisc`'s moment
distribution against real target moments
(`~/Documents/BFD_cNF/summary_templates_new.fits`, a COSMOS/DES-like
template catalog with the same 5-moment `[Mf, Mr, M1, M2, Mc]` layout, cut
to S/N > 10) via a corner plot in the flow's chart coordinates
(`dev/check_bulgedisc_vs_real.py`, companion to `bulk.py corner`). All work
is in `../bfd_cnf_imsims` (uncommitted).

### Flux-size correlation: fixed, and the fix generalises to `gauss2_real`

Real data has `Mr/Mf` anti-correlated with flux (corr -0.32; brighter
galaxies measure more compact, fainter ones larger) because a MORE
resolved profile's own k-space falloff pulls power away from the weight's
high-k tail -- an unresolved/point-like galaxy sits at the point-source
ceiling, not the floor. `bulgedisc`'s independent flux/size power laws
had this backwards (+0.1 to +0.25). Fixed in `imsims/sim.py`:
`_coupled_sigma` (shared by `sample_population` and `fit_gauss2_p0.py`)
scales the drawn `sigma` by `(flux / SIZE_FLUX_PIVOT) ** SIZE_FLUX_SLOPE`
(brighter -> bigger -> more resolved -> lower Mr/Mf), `SIZE_BETA` flattened
2.5 -> 1.3 so the rescale doesn't pile mass onto `SIZE_RANGE`'s floor.
Calibrated by rendering test catalogs, not a closed-form fit (`Mr/Mf` has
no clean analytic form in `sigma` once `bulge_frac`/`bulge_ratio` are
folded in). Result: corr -0.27 (target -0.32), median `Mr/Mf` 3.51 (target
3.50), quintile trend now falls with flux as real's does.

`fit_gauss2_p0.py` gained `--prefix`, and now draws its fit sample through
`_coupled_sigma` too, so refitting P0 off the corrected box was one
command. Added `GAUSS2_REAL_MU`/`_COV` and a `"gauss2_real"` population
(parametrised `sample_population_gauss2(n, rng, mu, cov)` rather than
duplicated) -- `GAUSS2_MU`/`GAUSS2_COV`/`"gauss2"` are untouched, so
`flows/shear_gauss2.eqx` and everything trained on it stay valid. The
refit's spin-0 covariance has a negative (log Mf, logit Mr/Mf) cross-term
(-0.090 vs. the old +0.055) -- the anti-correlation is now in the chosen
density itself, though `gauss2_real`'s Gaussian-in-chart-space P0
structurally cannot carry skew (real: -0.58; `gauss2_real`: ~0), a ceiling
of the family, not a miss.

### Ellipticity: NOT fixable by tuning -- a genuine structural limit

User's assessment after the first corner plot: "not nearly similar
enough." The dominant remaining gap is ellipticity: `bulgedisc` measures
median `|e| = 0.012-0.03` against real's `0.129` at matched S/N -- a
4-10x gap that PREDATES this session's changes (the original,
pre-`SIZE_FLUX_SLOPE` population already measured `|e| = 0.026`).

Two fixes tried and both fail to close it:
- **`BULGE_ELLIP_ALIGN`** (new): the bulge's ellipticity was drawn
  independently of the disc's (magnitude AND orientation) -- "a
  deliberately harsh test, not a realistic population" per the original
  comment. Added a knob interpolating to fully aligned (1, matching
  `gauss2`'s own long-standing co-elliptical choice) -- physically the
  right call, kept at 1, but moves measured `|e|` only 0.026 -> 0.029.
  Independent orientation was never the dominant suppressor.
- **Raising `E_SIGMA`** (intrinsic ellipticity width): tested 0.3 through
  0.9 (near-maximal intrinsic distortion) at fixed size -- measured `|e|`
  saturates around 0.043 and goes no higher. The bottleneck isn't
  intrinsic shape, it's resolution: at the sizes `SIZE_FLUX_SLOPE`
  needs to hit the correlation/median above, most of the population sits
  only marginally resolved against `PSF_SIGMA`/`WEIGHT_SIGMA`, and an
  unresolved source measures little ellipticity regardless of its true
  one.
- **Sizing up to fix resolution directly** (tested `SIZE_RANGE` from
  `(0.25,1.5)` through `(0.7,3.0)`): measured `|e|` does rise (up to
  ~0.11 at the largest range tried) -- but median `Mr/Mf` falls with it
  (well-resolved galaxies measure LOWER `Mr/Mf`, the same relation the
  correlation fix above relies on), landing at 2.4-2.5 against real's
  3.16, and the flux correlation's sign becomes unstable across the
  tested (size, slope, clip) combinations. Six combinations were tried;
  none gets within ~2x of real on both `Mr/Mf` and `|e|` simultaneously.

**Assessment: this looks like a limit of the Gaussian-profile family, not
an untried knob.** Real (non-Gaussian) profiles carry more high-k shape
power per unit `Mr/Mf` than a Gaussian does, so a real galaxy can be
simultaneously as compact (in `Mr/Mf`) and as elliptical (in measured
`|e|`) as observed, while two co-elliptical Gaussians measured through
this weight/PSF apparently cannot -- at least not via `FLUX_RANGE`,
`SIZE_RANGE`, `SIZE_BETA`, `SIZE_FLUX_SLOPE`, `E_SIGMA`, or
`BULGE_ELLIP_ALIGN` alone. User's call, given the options (accept the
partial match / run a real multi-parameter optimisation against the
combined ceiling / try the existing `sersic` population instead, whose
non-Gaussian profile might carry more shape power at the same
resolution): **accept the partial match, document the limit, stop
tuning.** `SIZE_FLUX_SLOPE`/`SIZE_BETA` were kept at the values that best
match the correlation/median (0.45/1.3), not retuned toward the
ellipticity-favouring, correlation-losing end of the trade-off.

Flux normalisation was left unaddressed in this entry -- see the
follow-on below, same session.

### Artifacts

- `../bfd_cnf_imsims/imsims/sim.py`: `SIZE_BETA` 2.5 -> 1.3;
  `SIZE_FLUX_PIVOT`/`SIZE_FLUX_SLOPE` (new); `_coupled_sigma` (new, shared
  helper); `BULGE_ELLIP_ALIGN` (new, =1); `GAUSS2_REAL_MU`/`_COV` (new);
  `"gauss2_real"` in `POPULATIONS`; `sample_population_gauss2` gained
  `mu`/`cov` parameters (default `GAUSS2_MU`/`GAUSS2_COV`, unchanged
  behaviour for `"gauss2"`).
- `../bfd_cnf_imsims/fit_gauss2_p0.py`: `--prefix` flag; draws its fit
  sample's size through `_coupled_sigma`.
- `../bfd_cnf_imsims/tests/test_sim.py`: `test_population_shapes` updated
  for `sigma` no longer being a clean power law within `SIZE_RANGE`; all
  23 tests pass.
- `dev/check_bulgedisc_vs_real.py` (new, in this repo): the corner-plot
  comparison, reuses `bulk.py`'s `to_coords`/chart constants.
- `dev/moments_bulgedisc_realmatch.fits` (this repo): 40k-galaxy render
  used for the final plot.
- `plots/bulgedisc_vs_real_corner.png` (this repo): the comparison plot.
- None of this was rendered at catalog scale or retrained downstream --
  only affects what `imsims.sim`/`copies.py` produce going forward.
  `bulgedisc`, `bulgedisc_deep`, `bulgedisc_noisy`'s existing `.fits` files
  and every flow trained on them still reflect the OLD distribution.

## 2026-08-27 (cont.): the flux/Mr-Mf mismatch was half a plotting bug --
fixed the comparison, then fixed the flux marginal for real

User pushed back on the corner plot ("not nearly similar enough") and
asked specifically why the flux-size (`log10 Mf` vs. `Mr/Mf`) plane didn't
overlap even though the correlation direction had been fixed. Two
separate causes, only one of them a real population mismatch.

### Bug: `in_chart`'s row-drop silently biased the plotted real sample

`dev/check_bulgedisc_vs_real.py`'s original `in_chart` filter dropped a
row if EITHER chart logit (`Mr/Mf` or `Mc/Mr`) was invalid, since a corner
plot needs one consistent 5-d point across every panel. Measured:
**50.9%** of the real S/N>10 sample fails the `Mc/Mr` chart bound alone
(real galaxies routinely exceed this chart's `Mc/Mr` ceiling -- a real,
separate finding about the chart's own coverage of real data, not chased
here). That failure correlates with size, so the drop was not neutral:
real's Mr/Mf-valid-only median is 3.32 (matching what bulgedisc had
actually been calibrated against), but requiring BOTH logits valid pulled
the surviving median down to 3.16 -- making an apples-to-oranges
comparison look like bulgedisc's population itself was off.

Fixed by replacing the drop with `to_coords_clipped`: clip `Mr/Mf` and
`Mc/Mr` to just inside `(0, ceiling)` before the logit instead of dropping
the row. No row disappears from panels it's otherwise fine in; the
boundary-piled rows show up as a spike at the chart edge instead (itself
informative -- makes the 50.9% `Mc/Mr` overflow visible rather than
silently discarded). Also needed: the plot's axis range, previously set
from `real`'s own percentiles, was blown out by that same spike once rows
stopped being dropped (>50% of `Mc/Mr` mass sitting at the clip boundary
moves the 99.5th percentile there); fixed by computing the range from the
non-piled-up bulk only (`|t| < 8` on the two logit columns).

Net effect: with the filtering bug fixed, `Mr/Mf` and `Mc/Mr` overlap
markedly better between real and bulgedisc -- most of the apparent
mismatch on those two axes in the previous entry's plot was this bug, not
a population defect.

### Real fix: `FLUX_RANGE`/`FLUX_ALPHA`/`SIZE_FLUX_SLOPE` retuned jointly

The `log10 Mf` offset survived the plotting fix (confirmed on the
unfiltered real sample directly, not just the corner plot), so it is
real: bulgedisc measured median 3.75 vs. real's 3.32, std 0.26 vs. 0.49.
Retuned by rendering (not closed-form -- `SIZE_FLUX_SLOPE`'s coupling
changes how much flux the weight window captures as well as the flux
draw itself, so the two interact and had to be retuned together):

- `FLUX_RANGE`: `(300, 3e5)` -> `(100, 4e5)`
- `FLUX_ALPHA`: `2.5` -> `2.0`
- `SIZE_FLUX_SLOPE`: `0.45` -> `0.30` (re-lowered because the wider/
  shifted flux draw alone pushed the Mr/Mf correlation past target)

Result (8k-galaxy render, matched to the real S/N>10 sample): `log10 Mf`
median 3.37 (target 3.32), std 0.40 (target 0.49, better than the prior
0.26 though still narrow); `Mr/Mf` median 3.52 (target ~3.50), corr(log
Mf, Mr/Mf) -0.36 (target -0.32) -- essentially unchanged from before this
retune, confirming the two knobs interact but didn't have to trade off
against each other here. `gauss2_real`'s P0 was refit
(`fit_gauss2_p0.py --prefix GAUSS2_REAL`) against the corrected box and
the constants updated in `sim.py`.

Ellipticity (`M1/Mr`, `M2/Mr`) is unaffected by either fix, as expected --
still the documented structural limit from the previous entry.

### Artifacts

- `dev/check_bulgedisc_vs_real.py`: `in_chart` replaced by
  `to_coords_clipped`; plot range computed from the non-piled-up bulk.
- `../bfd_cnf_imsims/imsims/sim.py`: `FLUX_RANGE`, `FLUX_ALPHA`,
  `SIZE_FLUX_SLOPE` retuned (values above); `GAUSS2_REAL_MU`/`_COV`
  refit to match.
- `dev/moments_bulgedisc_realmatch.fits`, `plots/bulgedisc_vs_real_corner.png`:
  regenerated against the retuned population.
- `../bfd_cnf_imsims` tests: 23/23 pass, unaffected (no shape assertions
  on `flux` itself beyond its own power-law CDF, which still holds under
  the new `FLUX_RANGE`/`FLUX_ALPHA`).

## 2026-08-27 (cont.): the flux floor was a weight-function gain, not a
range problem -- and E_SIGMA's width does keep paying off past the
resolution ceiling

Two more user questions on the corner plot: why doesn't bulgedisc's flux
marginal reach as faint as real's, and can the ellipticity width
(`E_SIGMA`) just be raised to help.

### The flux floor: a measured ~10-13x gain between `flux` and `Mf`

Even with `FLUX_RANGE`'s true-flux floor at 100 (previous entry), measured
`log10 Mf`'s 1st percentile was 2.94 -- nowhere near real's ~2.4-2.5.
Rendered the ten faintest true-flux galaxies (`flux = 100`) directly:
measured `Mf` came back **1070-1280**, a 10.7-12.8x GAIN, not a loss.
Confirmed on a larger sample, binned by `sigma`: the gain runs ~12.7x for
compact/faint objects down to ~9.6x for the largest, best-resolved ones.
This is a genuine property of the weight function's own k-space
normalisation (an overall scale between the `flux` parameter, in
`NOISE_SIGMA` units, and the moment `Mf`), not a bug -- but it meant the
true-flux floor needed to drop by roughly that gain to reach real's faint
end, not just get pushed down a little.

Retuned `FLUX_RANGE` `(100, 4e5)` -> `(25, 2e5)`, `FLUX_ALPHA` `2.0` ->
`1.6`, `SIZE_FLUX_SLOPE` `0.30` -> `0.22` (jointly, by rendering -- the
three interact through the same weight-window mechanism). No single
setting matched median, std AND the 1st-percentile floor together; this
one trades the median (measured 2.96 vs. real 3.32, ~0.36 dex short) for
getting the floor (2.47 vs. ~2.4-2.5) and std (0.64 vs. real's 0.49, now
too WIDE rather than 2x too narrow) both close. Visually: the corner
plot's `log10 Mf` leading edge now lines up with real's.

### E_SIGMA: median still capped, but the SPREAD keeps responding

The previous entry found median measured `|e|` saturates ~0.043 by
`E_SIGMA = 0.9` -- correct, and still true. But re-checked the SPREAD (not
just the median) under the corrected flux config: std(`M1/Mr`) grows
0.036 -> 0.048 and the 90th-percentile `|e|` grows 0.076 -> 0.106 going
from `E_SIGMA = 0.3` to `0.9`, with the gain flattening out around 0.7.
So raising `E_SIGMA` is not fully dead -- it widens the TAILS even though
it can't move the CENTRE past the resolution ceiling. Set to `0.7`
(diminishing returns past that point). `tests/test_sim.py`'s ellipticity
bounds were loosened to match (true, unmeasured `e`: median 0.35 -> 0.55,
frac>0.6 0.1 -> 0.3 -- both scale with `E_SIGMA` by construction, not
arbitrarily loosened).

Median measured `|e|` is still the same structural gap documented in the
previous entry -- this is a real, partial improvement to the shape of the
distribution, not a claim the gap closed.

### Artifacts

- `../bfd_cnf_imsims/imsims/sim.py`: `FLUX_RANGE`, `FLUX_ALPHA`,
  `SIZE_FLUX_SLOPE`, `E_SIGMA` retuned again (values above);
  `GAUSS2_REAL_MU`/`_COV` refit to match.
- `../bfd_cnf_imsims/tests/test_sim.py`: ellipticity bounds updated for
  `E_SIGMA = 0.7`; 23/23 pass.
- `dev/moments_bulgedisc_realmatch.fits`, `plots/bulgedisc_vs_real_corner.png`:
  regenerated.

## 2026-08-27 (cont.): that over corrected -- the calibration target was
itself wrong

User's next call: "that over corrected." Checked the flux marginal's
actual real-vs-bulgedisc numbers (not just the plot) and found the
calibration target used above was wrong, not the retune itself.

`median 3.32, p1 ~2.4-2.5`, used as the real target for both flux
retunes this session, came from an ad hoc calculation earlier in the
session that (like `dev/check_bulgedisc_vs_real.py`'s original
`in_chart`, fixed two entries ago) additionally required the `Mc/Mr`
chart bound to hold before computing real's flux statistics -- the same
Mc/Mr-correlates-with-size bias that made `Mr/Mf` look mismatched
earlier. Measured the CORRECT way (matching what the corner plot itself
actually compares, `Mr > 0` only): real's `log10(Mf)` median is **3.19**
(not 3.32), std **0.43** (not 0.49), 1st percentile **2.77** (not
~2.4-2.5). The previous entry's retune chased the wrong, too-faint
target and visibly overshot real's marginal past it.

Retuned a third time against the correct numbers: `FLUX_RANGE` `(25,
2e5)` -> `(60, 3e5)`, `FLUX_ALPHA` `1.6` -> `1.9`, `SIZE_FLUX_SLOPE`
`0.22` -> `0.26`. Result: log10(Mf) median 3.18 (target 3.19), std 0.45
(target 0.43), p1 2.82 (target 2.77), Mr/Mf median 3.51 (target 3.50),
corr -0.34 (target -0.33) -- close on every axis checked this time, and
the corner plot's `log10 Mf` peaks now visibly line up. `gauss2_real`
refit again. `E_SIGMA` (ellipticity width) was untouched this round --
its target wasn't affected by the flux-target bug.

**Lesson for next time:** when computing a real-data target statistic
for calibration, always go through the same filtering path the actual
comparison will use (here: `Mr > 0` only, no `Mc/Mr` requirement) rather
than a fresh ad hoc calculation -- an isolated "let me just check X"
snippet is exactly where a stale/inconsistent filter creeps back in.

### Artifacts

- `../bfd_cnf_imsims/imsims/sim.py`: `FLUX_RANGE`, `FLUX_ALPHA`,
  `SIZE_FLUX_SLOPE` retuned a third time (values above);
  `GAUSS2_REAL_MU`/`_COV` refit to match.
- `dev/moments_bulgedisc_realmatch.fits`, `plots/bulgedisc_vs_real_corner.png`:
  regenerated.
- `../bfd_cnf_imsims` tests: unaffected by this entry, 23/23 pass.

## 2026-08-27 (cont.): matching the Mr/Mf marginal's shape, not just its
median and correlation

User: "match the size marginal a bit more accurately." Prior entries had
`Mr/Mf` median (3.51) and correlation (-0.34) close to real's (3.50,
-0.33), but the DISTRIBUTION's shape was still off: std 0.35 vs. real's
0.47 (25% too narrow), skew -2.3 vs. real's -0.60 (piled hard against the
point-source ceiling, real isn't).

Widened `SIZE_RANGE` `(0.25, 1.5)` -> `(0.15, 2.5)` and flattened
`SIZE_BETA` `1.3` -> `0.9`, giving `SIZE_FLUX_SLOPE` more genuine spread
in the base size draw to work with instead of concentrating everything
near the resolution ceiling. Widening the range alone dilutes the
coupling's relative pull (correlation fell toward zero at the same
slope), so `SIZE_FLUX_SLOPE` had to go back up (`0.26` -> `0.38`) to hold
it. Checked the flux marginal wasn't disturbed by the size retune -- it
wasn't (log10 Mf median/std stayed within 0.01 of target).

Result: `Mr/Mf` std 0.46-0.48 (target 0.47, essentially closed, up from
0.35), median 3.55 (target 3.50, small overshoot), corr -0.29 to -0.34
(target -0.33). **Skew did not respond** -- every (range, beta, slope)
combination tried that kept median/correlation near target stayed in the
-2 to -3 band; it moved only when the correlation was allowed to collapse
toward zero (a genuine tension: the coupling that produces the real
anti-correlation is exactly what piles mass against the ceiling and
produces the negative skew). This looks like the same kind of structural
ceiling as the ellipticity gap documented earlier -- both the "not
fixable by tuning" entry and the flux-target correction two entries ago
apply the same lesson here: this is a genuine limit of the two-knob
(range, slope) family at this level of investment, not a specific
setting left untried. Visible trade-off in the plot: the wider spread
pushes bulgedisc's bright tail past real's, so `Mr/Mf`'s peak now sits
slightly right of real's rather than sitting under it.

### Artifacts

- `../bfd_cnf_imsims/imsims/sim.py`: `SIZE_RANGE`, `SIZE_BETA`,
  `SIZE_FLUX_SLOPE` retuned (values above); `GAUSS2_REAL_MU`/`_COV`
  refit to match.
- `dev/moments_bulgedisc_realmatch.fits`, `plots/bulgedisc_vs_real_corner.png`:
  regenerated.
- `../bfd_cnf_imsims` tests: 23/23 pass, unaffected.

## 2026-08-27 (cont.): "that overdid it again" -- reverted the size-marginal
widening, skew is a structural limit like ellipticity

User's next call on the previous entry's plot. Dialed the fix back
(`SIZE_RANGE` `(0.2, 2.0)`/`(0.2, 1.8)`, `SIZE_BETA` `1.1`/`1.15`, milder
slopes `0.30`-`0.32`) to see if a smaller step would keep the std gain
without the peak shift -- it didn't: skew stayed in the same -2.0 to -2.6
band as the full widening at every one of these intermediate settings
too, while median stayed ~3.54 (still overshooting real's 3.50). The std
improvement was never separable from the skew/peak-shift side effect
across the whole range tried.

Reverted fully to the pre-widening state: `SIZE_RANGE` back to `(0.25,
1.5)`, `SIZE_BETA` back to `1.3`, `SIZE_FLUX_SLOPE` back to `0.26`.
This is the best MEDIAN alignment found this session (3.51 vs. real's
3.50) at the cost of `Mr/Mf`'s std staying narrow (0.35 vs. real's 0.47).
`gauss2_real` refit to match (back to the "that over corrected" entry's
values). Treating `Mr/Mf`'s std/skew gap as belonging in the same bucket
as the ellipticity gap from earlier -- a real, documented limit of this
population family at the level of investment spent chasing it this
session, not a specific untried setting.

### Artifacts

- `../bfd_cnf_imsims/imsims/sim.py`: `SIZE_RANGE`, `SIZE_BETA`,
  `SIZE_FLUX_SLOPE` reverted to `(0.25, 1.5)`, `1.3`, `0.26`;
  `GAUSS2_REAL_MU`/`_COV` refit to match.
- `dev/moments_bulgedisc_realmatch.fits`, `plots/bulgedisc_vs_real_corner.png`:
  regenerated.
- `../bfd_cnf_imsims` tests: 23/23 pass, unaffected.

## 2026-08-27 (cont.): "the ellipticity marginal is still too narrow" --
deliberately re-opened the Mr/Mf-vs-|e| trade-off, on the user's call

User pushed on the ellipticity gap once more. Before touching anything,
ran two clean tests to make sure there wasn't a cheap untried lever:
raising `E_SIGMA` to 1.5 (nearly maximal intrinsic distortion) moved
median measured `|e|` 0.0199 -> 0.0213; widening `BULGE_SIZE_RATIO` to
`(0.05, 0.95)` (almost arbitrary bulge/disc size mismatch) moved it
0.0199 -> 0.0223. Both essentially flat -- reconfirmed the resolution
ceiling from two entries ago is real and neither of those levers touches
it.

Presented the only remaining option (re-opening the `SIZE_RANGE` trade-off
just reverted, in the OPPOSITE direction -- toward ellipticity, away from
`Mr/Mf`) and asked the user directly rather than guessing; they chose it.

Retuned deliberately onto the ellipticity side this time: `SIZE_RANGE`
`(0.25, 1.5)` -> `(0.9, 4.2)`, `SIZE_BETA` `1.3` -> `0.45`,
`SIZE_FLUX_SLOPE` `0.26` -> `0.35` (retuned up to keep some negative
correlation once sizes grew -- without it, corr collapses toward zero or
flips positive, as measured at several intermediate settings). Median
`sigma` moved from ~0.26" (below where `|e|` responds at all) to ~0.8",
clearing the resolution floor. `_coupled_sigma`'s clip ceiling raised
`3.0 -> 6.0` to stop silently re-capping the new, larger sizes.

Result: `std(M1/Mr)` 0.12 (real: 0.14) -- the ellipticity panels visibly
converge with real's in the corner plot, the biggest single improvement
of the whole ellipticity investigation. Cost, accepted explicitly:
`Mr/Mf` median drops to ~2.75-2.77 (real: 3.50), correlation weakens to
~-0.18 (real: -0.33), `log10(Mf)` softens slightly too (median ~3.04 vs.
the prior entry's 3.18). `gauss2_real` refit to match -- its spin-2
variance jumped from 1.5e-3 to 1.2e-2, a direct reflection of the same
choice.

**Where this leaves the population:** `bulgedisc` now sits on the
ellipticity side of a trade-off this session established is inherent to
the Gaussian-profile family, not a bug -- getting real's `Mr/Mf` needs
small, marginally-resolved sizes; getting real's `|e|` needs sizes that
clear the PSF/weight resolution floor; this weight/PSF setup cannot
deliver both from the same size distribution. If a future session wants
`Mr/Mf` back, the `(0.25, 1.5)`/`1.3`/`0.26` values from two entries ago
are the other end of this same trade-off, not a bug to re-diagnose.

### Artifacts

- `../bfd_cnf_imsims/imsims/sim.py`: `SIZE_RANGE`, `SIZE_BETA`,
  `SIZE_FLUX_SLOPE` retuned to `(0.9, 4.2)`, `0.45`, `0.35`;
  `_coupled_sigma`'s clip ceiling `3.0 -> 6.0`; `GAUSS2_REAL_MU`/`_COV`
  refit to match.
- `../bfd_cnf_imsims/tests/test_sim.py`: sigma-range assertion's upper
  bound updated to match the new clip ceiling; 23/23 pass.
- `dev/moments_bulgedisc_realmatch.fits`, `plots/bulgedisc_vs_real_corner.png`:
  regenerated.

## 2026-08-27 (cont.): decoupling flux/size from ellipticity with a
Gaussian copula -- caught and fixed an implementation bug along the way

User: "can we readjust the flux and size now without changing the
ellipticity distribution?" The multiplicative coupling (`sigma *=
(flux/pivot)**slope`) used all session couldn't do this even in
principle -- retuning `FLUX_RANGE` changes the ratio `flux/pivot` for
every galaxy, which rescales `sigma`, which moves measured `|e|`, exactly
the entanglement this session's flux and size retunes kept running into.

### The fix: a Gaussian copula

Replaced it with `_flux_sigma_copula`: draw `flux` and `sigma` from their
own INDEPENDENT power laws (`FLUX_RANGE`/`FLUX_ALPHA`,
`SIZE_RANGE`/`SIZE_BETA`), draw a pair of correlated standard normals
`(z1, z2)` with correlation `SIZE_FLUX_RHO`, then assign each galaxy the
flux/sigma values at the RANK of its own `z1`/`z2` (`sort(x)[rank_of_z]`).
This reorders both draws to induce a Spearman-style correlation between
them without rescaling either -- so each marginal is EXACTLY preserved
(same set of values, just permuted), and `SIZE_FLUX_RHO` is the only
handle on the correlation. Verified by rendering: median `|e|` stayed
flat at 0.096-0.097 across `SIZE_FLUX_RHO` in `[-0.9, 0.9]` and across
every `FLUX_RANGE` tried at fixed `SIZE_FLUX_RHO` -- genuine decoupling,
confirmed empirically, not just by construction.

### A bug, caught by the corner plot itself

The first implementation used `sigma = _powerlaw(rng, n, *SIZE_RANGE,
SIZE_BETA)` directly for the copula's sigma marginal -- but the
`SIZE_RANGE`/`SIZE_BETA` values on disk (`(0.9, 4.2)`/`0.45`) were fit for
the OLD multiplicative coupling, where the realised sigma was that base
draw MULTIPLIED by a shrinking `(flux/pivot)**slope` factor (typically <
1). Dropping the multiply without re-deriving the base range left sigma
far too large: rendered, median `|e|` came out 0.233 (real: 0.129, so
this OVERSHOT it by nearly 2x) and `Mr/Mf` collapsed to 1.43 with its
correlation near zero. Caught immediately from the rendered corner plot
(`M1/Mr`/`M2/Mr` visibly wider than real's, not just matching) rather than
assumed correct from the isolated test script that validated the design.
Root cause: the isolated test that validated the copula's decoupling
property used a DIFFERENT, frozen internal sigma-generating step than
what got written into `sim.py` -- the design was right, the
implementation didn't match what was tested.

### Recalibrated cleanly under the fixed design

With flux and size now genuinely independent, recalibrating was a normal
two-axis search rather than the three-way entangled one earlier in this
session: `FLUX_RANGE` `(90, 4e5)`, `FLUX_ALPHA` `1.65`; `SIZE_RANGE`
`(0.4, 2.2)`, `SIZE_BETA` `0.6`; `SIZE_FLUX_RHO` `0.85`. Result:

- `log10(Mf)`: median 3.24 (target 3.19), std 0.53 (target 0.43).
- `Mr/Mf`: corr -0.29 (target -0.33); median stays ~2.4-2.8 regardless of
  `SIZE_FLUX_RHO` -- reconfirmed structural (tied to sigma's marginal,
  which is now fixed for ellipticity), not a copula limitation.
- **`|e|`: median 0.126 against real's 0.129** -- the closest ellipticity
  match of the entire investigation, better than the multiplicative
  coupling's best (0.095-0.097) ever reached, purely from being able to
  pick `SIZE_RANGE`/`SIZE_BETA` for ellipticity without also having to
  carry the flux correlation through the same numbers.

`gauss2_real` refit to match. All 23 `bfd_cnf_imsims` tests pass
(`test_sim.py`'s sigma-marginal assertion rewritten to check its own
clean power-law CDF, the same way flux's already was, since the copula no
longer clips or rescales it).

### Artifacts

- `../bfd_cnf_imsims/imsims/sim.py`: `_coupled_sigma` replaced by
  `_flux_sigma_copula` (new function, new `SIZE_FLUX_RHO` constant,
  `SIZE_FLUX_PIVOT`/`SIZE_FLUX_SLOPE` retired); `FLUX_RANGE`, `FLUX_ALPHA`,
  `SIZE_RANGE`, `SIZE_BETA` retuned to the values above;
  `sample_population` updated; `GAUSS2_REAL_MU`/`_COV` refit to match.
- `../bfd_cnf_imsims/fit_gauss2_p0.py`: updated to call
  `sim._flux_sigma_copula` instead of the retired `_coupled_sigma`.
- `../bfd_cnf_imsims/tests/test_sim.py`: sigma assertion rewritten for a
  clean power-law CDF check (was a bounds-only check under the old
  clip-based coupling); 23/23 pass.
- `dev/moments_bulgedisc_realmatch.fits`, `plots/bulgedisc_vs_real_corner.png`:
  regenerated.

## 2026-08-27 (cont.): "there must be, you are wrong" -- the trade-off
wasn't structural, it was two arbitrary parameterisation choices

User rejected the "structural trade-off" conclusion outright: "the bulge
disc model has more than enough free parameters to produce any moments we
are interested in, the only thing preventing that is your parameterization
on the distribution." Correct, and it took a controlled single-galaxy
test to see it -- every prior test in this investigation had varied
POPULATION-level knobs (E_SIGMA, BULGE_SIZE_RATIO, bulge/disc dominance,
even the `sersic` population) and read off POPULATION medians, which
conflates "can one galaxy have both" with "does the typical galaxy in
this particular sample have both."

### The direct test that overturned it

Rendered single galaxies at FIXED sigma, sweeping only `e`:

| sigma | Mr/Mf(e=0) | Mr/Mf(e=0.9) | \|e\|(e=0.9) |
|---|---|---|---|
| 0.3 | 3.247 | 3.288 | 0.121 |
| 0.4 | 2.958 | 3.062 | 0.198 |
| 0.5 | 2.646 | 2.844 | 0.279 |

At fixed size, `Mr/Mf` barely moves with `e` -- a single galaxy at
sigma=0.4 measures Mr/Mf=3.06 AND `|e|`=0.20 simultaneously, nothing
conflicting about it. The population-level anti-correlation this whole
investigation kept finding was real but came from elsewhere.

### Two arbitrary parameterisation choices, found and fixed

1. **`_ellipticity`'s eq. 58 prior caps median intrinsic e at ~0.45**
   regardless of scale. Measured directly: median e = 0.29 / 0.41 / 0.45 /
   0.45 at `sigma_e` = 0.3 / 0.7 / 1.5 / 10.0 -- the `(1-e^2)^2` term
   structurally bounds the typical value, independent of the scale
   parameter this session kept turning up (`E_SIGMA`). Every earlier
   "E_SIGMA saturates" finding this session was this ceiling, misdiagnosed
   as a resolution limit. Fixed: new `_ellipticity_wide`,
   `P(e) ~ e exp(-e^2/2 sigma_e^2)` (no suppression term) -- its median
   scales freely (0.35 / 0.53 / 0.61 at sigma_e = 0.3 / 0.5 / 0.7). Used
   only by `sample_population` (bulgedisc); `_ellipticity`/eq. 58 is
   untouched and still used by `sersic` and the `gauss2`/`gauss2_real`
   family, where it remains the physically appropriate choice.
2. **`BULGE_FRAC_RANGE = (0, 1)` let bulge-dominated galaxies dilute the
   population's measured `|e|`** independently of both sigma and the
   e-prior: a galaxy whose flux is mostly in a SMALL bulge
   (`bulge_ratio * sigma`, well under the disc's own sigma) measures
   suppressed ellipticity from that component regardless of its intrinsic
   e, dragging the population median down. Fixed: narrowed to `(0, 0.3)`
   -- disc-dominated galaxies only. Not a claim this matches real bulge
   fractions; it removes an artifact specific to this two-Gaussian
   construction, not a real sub-population.

### Recalibrated with both fixes in place

`SIZE_RANGE` narrowed `(0.4, 2.2) -> (0.35, 0.55)`, `SIZE_BETA`
`0.6 -> 2.0` (concentrate mass at the now-favourable sigma), `FLUX_RANGE`
`(90, 4e5) -> (70, 3e5)`, `FLUX_ALPHA` `1.65 -> 1.75`, `SIZE_FLUX_RHO`
`0.85 -> 0.4` (the narrower size range needs much less rank-correlation
to reach the same Mr/Mf-vs-flux slope). Result, compared to the entry
right before this one:

|  | Mr/Mf median (target 3.50) | corr (target -0.33) | median \|e\| (target 0.129) |
|---|---|---|---|
| previous entry | 2.43 | -0.29 | 0.126 |
| **this entry** | **3.05** | -0.26 | **0.122** |

Both `Mr/Mf` and `|e|` improved (or held near-flat) together -- the
outcome the user said should be possible and the previous six entries in
this investigation incorrectly concluded was not. `log10(Mf)`: median
3.24 (target 3.19), std 0.55 (target 0.43) -- close, not chased further
this entry. `gauss2_real` refit to match. All 23 `bfd_cnf_imsims` tests
pass (`test_sim.py`'s ellipticity bounds updated for the new, deliberately
wider distribution).

**Lesson:** when several independent levers all point the same direction
("this is capped"), check whether they're independent knobs on the SAME
underlying mechanism (here: population median e, capped by one prior's
functional form) before concluding the underlying quantity itself is
capped. A single controlled test at the individual level (not the
population level) would have caught this much earlier.

### Artifacts

- `../bfd_cnf_imsims/imsims/sim.py`: new `_ellipticity_wide` function and
  `ELLIP_SIGMA_WIDE` constant; `BULGE_FRAC_RANGE` narrowed to `(0, 0.3)`;
  `FLUX_RANGE`, `FLUX_ALPHA`, `SIZE_RANGE`, `SIZE_BETA`, `SIZE_FLUX_RHO`
  retuned (values above); `sample_population` now calls
  `_ellipticity_wide` instead of `_ellipticity`; `GAUSS2_REAL_MU`/`_COV`
  refit to match.
- `../bfd_cnf_imsims/fit_gauss2_p0.py`: updated to call
  `sim._ellipticity_wide` instead of `sim._ellipticity`.
- `../bfd_cnf_imsims/tests/test_sim.py`: ellipticity bounds widened again
  for the new distribution; 23/23 pass.
- `dev/moments_bulgedisc_realmatch.fits`, `plots/bulgedisc_vs_real_corner.png`:
  regenerated.

## 2026-08-27 (cont.): "still not good enough" / "keep trying the bulge
disc" -- a third stale target, then a log-normal sigma closes most of the
remaining gap

Two more rounds on `Mr/Mf`, both starting from checking the corner plot
directly (per the user's explicit instruction this time, not just summary
stats).

### Round 1: the |e| target was ALSO stale (a third time)

Before iterating further, re-verified the |e| target against the same
filtered sample the comparison plot actually uses (the same discipline
that caught the flux-target bug earlier). It was wrong again: **real's
median |e| is 0.089, not 0.129** -- the 0.129 figure had been carried
since very early in the session from an unfiltered ad hoc calculation and
never re-checked once the plot's own filtering was fixed. This is the
THIRD time this exact class of bug hit this investigation (flux target,
twice; now |e|). Since bulgedisc's `|e|` (0.122 at the time) was already
OVERSHOOTING the correct target, there was real room to trade back toward
`Mr/Mf`: lowered `ELLIP_SIGMA_WIDE` 0.68 -> 0.3, widened `BULGE_FRAC_RANGE`
0.3 slightly and `SIZE_RANGE` back out. Checked the corner plot after --
`Mr/Mf` improved modestly (median 2.43 -> 2.76-2.84) but the visual gap
was still large and a further small nudge (`BULGE_FRAC_RANGE` -> 0.4,
`ELLIP_SIGMA_WIDE` -> 0.34) barely moved the rendered plot at all.

### Round 2: the power law itself was the ceiling

User: "we are getting closer keep trying the bulge disc." Diagnosed why
the previous round stalled: `SIZE_RANGE`/`SIZE_BETA` is a BOUNDED POWER
LAW, whose median and dynamic range are set by the same two numbers (lo,
hi, beta) -- narrowing the range to push the median up always shrank the
std, and widening it back to fix std always dragged the median back down.
Confirmed across a dozen (range, beta) combinations this session; none
cleared both `Mr/Mf`'s median (3.50) and std (0.47) simultaneously.

Replaced sigma's marginal with **log-normal**: `sigma = exp(SIZE_LOGMEDIAN
+ SIZE_LOGSTD * N(0,1))`, clipped to `[0.05, 5.0]` for numerical safety
only. Log-normal's median (`exp(SIZE_LOGMEDIAN)`) and spread
(`SIZE_LOGSTD`) are genuinely independent parameters -- unlike the power
law, moving one doesn't structurally constrain the other. At
`SIZE_LOGMEDIAN = log(0.4)`, `SIZE_LOGSTD = 0.5`:

|  | Mr/Mf median (target 3.50) | Mr/Mf std (target 0.47) | \|e\| median (target 0.089) |
|---|---|---|---|
| power law, best | 2.84 | 0.41 | 0.09 |
| **log-normal** | **3.14** | **0.43** | **0.090** |

`ELLIP_SIGMA_WIDE` retuned to 0.7, `SIZE_FLUX_RHO` to 0.6 alongside the
new sigma marginal. Checked the corner plot: `Mr/Mf` and `Mc/Mr`'s shape
(not just summary stats) visibly closer to real -- the 2D contours now
show real overlap between the two populations rather than a small orange
blob sitting inside a much larger blue one. `gauss2_real` refit to match.
All 23 tests pass (`test_sim.py`'s sigma-marginal check rewritten for
log-normal instead of the retired power-law CDF check).

**Not fully closed:** `Mr/Mf` median is still short (3.14 vs. 3.50) and
`Mc/Mr` remains visibly narrower than real. Further tuning of this same
family (flux/sigma copula + log-normal sigma + `_ellipticity_wide`) is
reasonable next-session work; no new structural blocker was found this
round, just a family (power law) that couldn't reach the target and one
(log-normal) that gets much closer.

**Lesson, reinforced a third time:** ANY numeric target pulled into a
calibration loop must be re-verified against the exact same filtering
path as the comparison being optimized, every time it's used, not just
the first time it's introduced -- stale targets from earlier, differently
filtered computations kept re-entering this investigation as ad hoc
sanity checks.

### Artifacts

- `../bfd_cnf_imsims/imsims/sim.py`: `SIZE_RANGE`/`SIZE_BETA` (power law)
  retired, replaced by `SIZE_LOGMEDIAN`/`SIZE_LOGSTD` (log-normal) in
  `_flux_sigma_copula`; `SIZE_FLUX_RHO`, `ELLIP_SIGMA_WIDE`,
  `BULGE_FRAC_RANGE`, `BULGE_SIZE_RATIO` retuned across both rounds
  (values above); `GAUSS2_REAL_MU`/`_COV` refit twice, once per round.
- `../bfd_cnf_imsims/tests/test_sim.py`: sigma-marginal check rewritten
  for log-normal; 23/23 pass throughout both rounds.
- `dev/moments_bulgedisc_realmatch.fits`, `plots/bulgedisc_vs_real_corner.png`:
  regenerated after each round.

## 2026-08-28: training a flow on the retuned `bulgedisc` surfaces a
catastrophic `m1 ~ -1` on `bulgedisc_deep` -- extensively narrowed down,
NOT resolved, and the leading hypothesis is disputed by the user

Picked up the retuned-population task from `.session/next_prompt.md`:
render fresh `bulgedisc` training/target catalogs against the real-data
retune (previous several entries), retrain bulk/shear/centroid, and measure
`bulgedisc_deep`'s multiplicative bias. Steps 1-5 (render 100k noiseless
training catalog, `bulk.py train --steps 20000`, `shear.py train
--deriv-weight 1e4 --steps 60000`, render 200k deep targets, render copies,
warm-start centroid) all went cleanly and matched or beat the pre-retune
numbers: `dm/dg` residuals Mf 0.35%/Mr 2.31%/Mc 4.90% (vs. the prior
population's 2.81%/26.12%/40.73%), `centroid.py check`'s ellipticity-response
ratio 0.708 (vs. the prior population's 0.635, `gauss2_deep`'s 0.70).

Then `bias.py --pop bulgedisc_deep_v2 --samples 8192 --alpha 0.5 --chunk 4096
--batch-budget 65536 --n-targets 20000` (added a `"bulgedisc_deep_v2"` entry
to `CATALOGS`/`TRAIN_DATA` pointing at the new files, since the existing
`"bulgedisc_deep"` entry's filenames are the STALE pre-retune catalogs --
those are left alone and still work): `m1 = -1.01112 +/- 0.00023`. Not a
noisy/inconclusive number like the prior population's `+0.1627 +/- 0.0735` --
this is TIGHT and pinned near -1 (zero measured shear response) for three of
five `Mf` quintiles (q1 -1.0011+/-0.0001, q2 -1.0028+/-0.0005, q3
-1.0201+/-0.0008), q4 sign-flipped (+1.12+/-0.16), only q5 sane
(-0.016+/-0.006).

### What was ruled out, with evidence

1. **`Mc/Mr` ceiling excess.** The retuned population's noisy deep targets
   have 14.2% of rows above `POINT_SOURCE_MC` (6.662089) vs. the old
   population's 3.8% (both 0% in the noiseless training catalog -- it is
   noise pushing them over). Excluding those targets from the `ghat` sum
   offline: m1 still -1.02. Not the cause.
2. **Noise depth mismatch (real, but not sufficient).** The retune dropped
   the training catalog's median `Mf` 5067 -> 1859 (a deliberate, correct
   consequence of matching real DES/COSMOS flux). `noise_sigma = 2.73` was
   calibrated years ago against the OLD population to put flux S/N's 5th
   percentile at 10, median at 19; at the retuned population it now gives
   median S/N 6.8 -- below the old population's OWN 5th percentile.
   Recalibrated to `noise_sigma = 0.9` (median S/N 19.8-20.4, 5th pct ~9.5,
   matching the old target), re-rendered all three deep-target catalogs and
   `copies_bulgedisc_v2.fits` at the corrected depth (same filenames,
   overwriting the wrong-depth versions -- the noise_sigma=2.73 deep
   catalogs no longer exist on disk), re-warm-started centroid. `bias.py`
   at the corrected depth: `m1 = -1.26963 +/- 0.01158` -- WORSE, not fixed.
   Real bug, real fix (this depth is now the correct one to use going
   forward), but not the cause of the `m1~-1` catastrophe.
3. **Jackknife/chunk-count MC bias** (`pqr_streamed`'s own documented
   `O(1/S)` bias mechanism). `--chunk 4096` (2 chunks) vs `--chunk 1024` (8
   chunks) on the same 4000-target subset: `m1 = -1.27901` both ways, no
   material change. Not the cause.
4. **Outlier/spike domination.** The top 20 targets by |R| carry only
   8.9% of `R`'s ensemble sum; it takes 1206/19998 targets (6%) to reach 90%
   of the sum. This is a broad effect across a real fraction of the
   population, not a handful of `sane_targets`-evading spikes.
5. **Training-data sparsity.** Rendered 1M noiseless galaxies (10x), retrained
   bulk/shear/centroid from scratch on that at the corrected depth. `dm/dg`
   residuals barely moved (0.35/2.26/0.60/0.60/4.83 vs. 0.35/2.31/0.60/0.61/
   4.90). Unwindowed `bias.py`: `m1 = -1.17832 +/- 0.01002` -- still
   catastrophic. More data did not help.
6. **Ellipticity extremity.** Targets with pathological `R` (diag > 100) have
   LOWER measured `|e|` than the rest (0.091 vs 0.127 median) -- rounder, not
   more elliptical; `corr(rdiag, |e|) = -0.089`. `gauss2_deep`'s own noisy
   `|e|` reaches as high as 378 (pure noise excursion) and it is still clean
   unwindowed. Not ellipticity.
7. **`Mc/Mr`-ceiling PROXIMITY (not excess -- the whole distribution sitting
   near the boundary).** `gauss2_deep`'s own `Mc/Mr` median is 5.76, all but
   identical to the retuned `bulgedisc_deep`'s 5.90 at the corrected depth --
   and `gauss2_deep` is clean. Ruled out as the differentiator.
8. **`safe_point`/`in_domain` masking leaking a poisoned gradient through
   `jnp.where`.** Measured directly: a `bulgedisc_deep` target with 24.4% of
   its 8192 kernel draws out-of-domain has `flow.log_prob` at `safe_point`'s
   dummy as extreme as -3.4e15. Suspected this leaks into the Hessian via
   JAX's `where`-gradient semantics. Directly disproved: pulled
   `gauss2_deep`'s OWN worst-masked target (93% of draws masked, dummy
   `log_prob` = -2.35e18, even more extreme) and computed its real `R` via
   `pqr_streamed` -- `rdiag = 13.9`, completely sane. Masking is handled
   correctly in both populations.
9. **Low ESS / weight concentration among the surviving in-domain draws.**
   Measured directly for the same pathological target: ESS = 500.6 (healthy),
   top-5 weights ~0.004 each (not dominated by 1-2 draws). Not degenerate.
10. **Centroid transport's log-det swamping the g-dependent signal in float32
    precision.** Measured `|log-det|` of the centroid layer's Jacobian
    directly for both "bad" `bulgedisc` targets and `gauss2` targets: all
    values are small (<0.4) for both populations, nowhere near large enough
    to swamp anything.
11. **Streaming/jackknife-merge artifact in `pqr_streamed`.** Computed the
    FULL, unchunked Hessian of `log_conv_is` directly via `jax.hessian` on all
    8192 draws at once for the same pathological target, bypassing
    `pqr_streamed`'s chunking entirely: `rdiag = 823.6`, still huge. The
    pathological curvature is present in the CORE Hessian-of-logsumexp
    computation itself, not a streaming/merge artifact.

### Where this was left, and an explicit disagreement to resolve first

The session's own working hypothesis (not the user's) was that `bulgedisc`'s
non-Gaussian bulge+disc profile makes `models/centroid.py`'s analytic
Gaussian-in-k transport only APPROXIMATE (exact for `gauss2`, which is
literally a Gaussian-in-k mixture), and that the retuned population's wider
`Mr/Mf` spread pushes enough targets into a regime where that approximation
error, small in the aggregate `check` shift, is large enough per-target to
land the shear-response network's g-Hessian in a genuinely bad, sparsely-
resolved pocket -- consistent with `--no-centroid` being sane
(`m1 = -0.0836 +/- 0.0227`, comparable to the old population's own
`--no-centroid` baseline of `-0.0361 +/- 0.0132`) while `--with-centroid`
is catastrophic.

**The user explicitly disagrees with this**: the Gaussian-in-k approximation
should be good enough not to cause a failure this large, and there is likely
ANOTHER bug or difference between the `gauss2_deep` setup (known to work) and
the retuned `bulgedisc_deep_v2` setup (broken) that has not yet been found --
not a graceful approximation-quality effect. This is a live disagreement, not
settled, and the next session should treat the approximation-quality
hypothesis as unproven rather than default to it.

**The eleven-item elimination list above is not gospel either.** It was
produced under time pressure across many rounds of ad hoc diagnostic
scripts, at least one of which (the first `safe_point` clip-rate test) was
initially methodologically wrong -- it bypassed `in_domain` masking and gave
a misleading result until redone correctly. Re-verify a ruled-out item's own
measurement before leaning on it if the next session's work starts to
contradict it, rather than assuming it was airtight. See
`.session/next_prompt.md` for where to pick this up.

### Artifacts

- `bias.py`: added `"bulgedisc_deep_v2"` to `CATALOGS` (targets:
  `targets_deep_g1p02_200k_v2`/`g1m02`/`g0`) and `TRAIN_DATA`
  (`moments_bulgedisc_v2.fits`) -- the ONLY code change this session. NOT
  YET COMMITTED. `"bulgedisc_deep"`'s own entries are untouched and still
  point at the pre-retune catalogs.
- `../bfd_cnf_imsims/data/moments_bulgedisc_v2.fits` (100k, noiseless),
  `moments_bulgedisc_v2_1M.fits` (1M, noiseless): fresh renders against the
  committed real-data retune.
- `../bfd_cnf_imsims/data/targets_deep_g1p02_200k_v2.fits`/`g1m02`/`g0`,
  `copies_bulgedisc_v2.fits`: rendered TWICE this session, first at
  `noise_sigma = 2.73` (matching the old population's depth, WRONG for this
  one), then re-rendered under the SAME filenames at the corrected
  `noise_sigma = 0.9` -- only the corrected-depth version exists on disk now.
- `flows/bulk_bulgedisc_v2.eqx`, `shear_bulgedisc_v2.eqx`,
  `centroid_bulgedisc_v2.eqx`: 100k-galaxy training, `--deriv-weight 1e4`,
  centroid warm-started twice (stale-depth copies, then corrected-depth
  copies) -- the checkpoint on disk reflects the corrected depth.
- `flows/bulk_bulgedisc_v2_1M.eqx`, `shear_bulgedisc_v2_1M.eqx`,
  `centroid_bulgedisc_v2_1M.eqx`: the 10x-training-data retrain, corrected
  depth throughout, used to rule out data sparsity (point 4 above).
- No `models/centroid.py`, `shear.py`, or `bulk.py` changes this session --
  the investigation was entirely measurement/diagnosis, not a code fix.
- Numerous `/tmp` scratchpad diagnostic scripts and `.npz`/`.log` dumps
  (`pqr_v2_corrected.npz`, `pqr_v2_1M.npz`, `pqr_old_centroid.npz`, etc.),
  not committed, not preserved past the session's scratchpad directory.

## 2026-08-28 (cont.): the centroid approximation is NOT the cause -- the
ensemble R has flipped sign at the faint end, and the Fisher identity is the
diagnostic that finds it

Picked up `.session/next_prompt.md`'s five steps against the previous entry's
open disagreement.  The user's position -- that the Gaussian-in-k centroid
transport is good enough and something else is different between
`gauss2_deep` and `bulgedisc_deep_v2` -- is CORRECT, and the previous entry's
working hypothesis is now directly falsified.

### Step 1: the approximation-quality hypothesis, measured directly -- FALSIFIED

Per-target, on each population's own g=0 deep targets (20k rows, both arms of
the comparison run through the SAME code):

| quantity (p50 / p99)                    | bulgedisc_deep_v2 | gauss2_deep    |
|-----------------------------------------|-------------------|----------------|
| `trace(P)` (the transport's strength)   | 0.0050 / 0.038    | 0.0047 / 0.129 |
| round-trip \|z_rt - z\| (unmarg->marg)  | 4e-4 / 0.707      | 5e-4 / 0.501   |
| `\|dz\|` before the `_EXP_MAX` tanh cap | 0.033 / 9.52      | 0.057 / 18.1   |
| frac of targets with `trace(P) > 0.5`   | 0.0000            | 0.0005         |
| centroid log-det spread over 512 kernel draws (p50 / p99 nats) | 0.16 / 3.10 | 0.10 / 3.55 |

The transport is not merely comparable, it is MILDER on `bulgedisc_deep_v2`
than on the population that works -- gauss2 is the one with the heavier tail
in every column.  `Sigma_X / (Mf Mr)`, the dimensionless combination that sets
`trace(P)`, is 1.2e-3 for bulgedisc_v2 and 1.15e-3 for gauss2: the corrected
`noise_sigma = 0.9` depth matched the two populations almost exactly in the
variable that actually drives this layer.  There is no regime difference for
an approximation error to be large in.

### Step 2: every constant and config difference -- CLEAN

- Catalog headers are identical except `NOISESIG`/`SIG_XY`: `PIXSCALE 0.2`,
  `WTSIGMA 0.65`, `PSFSIGMA 0.4`, `SEED 0`, `IMGNOISE T`, same 9 columns, same
  `G1 = +/-0.02`, in all three populations.
- `SIG_XY` and `cov_odd` scale EXACTLY with `noise_sigma` (326.02/107.48 =
  3.0333 = 2.73/0.9), and the even-moment `cov` scales exactly with
  `noise_sigma^2` (73924.016 * (0.9/2.73)^2 = 8038 vs the 8034.249 on disk).
  The re-render at the corrected depth is self-consistent.
- Chart constants inside each checkpoint agree across all three layers:
  `RawMomentStandardize.mean/std` == `CentroidMarginalize.mean/std` ==
  `ShearResponse.chart_loc/chart_scale`, byte-identical, in
  `centroid_bulgedisc_v2.eqx`, `centroid_bulgedisc_v2_1M.eqx` AND
  `centroid_gauss2_deep_analytic.eqx`.  No repeat of the 2026-08-24 stale-chart
  bug.
- The centroid warm-start graft is complete: `centroid_bulgedisc_v2.eqx`'s
  chart and all 8 bulk layers are BIT-IDENTICAL to `shear_bulgedisc_v2.eqx`'s,
  and the shear layer differs in exactly 2 of 13 leaves -- `coeffs.u_mean`
  (max 3.0e-3) and `coeffs.u_white` (max 2.9e-2), the expected consequence of
  `sync_chart_constants` being handed the copies catalog's GALAXIES table
  rather than the 90% training slice.  0.3%, not a bug.
- Note for whoever reads `split_centroid` next: it returns `(flow, None)` for
  every flow `bulk.build_flow(shear=True, centroid=True)` produces, because
  those give the layer `cond_dim = 5`.  The peel is DEAD CODE on both
  populations -- so `bias.py` differentiates through the centroid layer in
  both, identically.  Not a difference between them, but it means the
  `centroid_transform`/log-det-in-the-weight path is never exercised.

### Step 3: moment-file structure -- CLEAN

Training and target moment distributions agree within each population
(v2 train Mf p1/50/99 = 728/1770/186278 vs targets 693/1776/179275;
Mr/Mf p50 3.149 vs 3.131).  No structural anomaly, no degenerate `cov`.

### What is actually wrong: R's SIGN, via the Fisher identity

The BFD estimator needs `sum_targets R` negative; the check that it is is the
Fisher identity `sum q^2 / sum (-r)`, which should be ~1 when the flow matches
the population the targets were drawn from.  Measured on 2000 g=0 deep targets
at S=8192, alpha=0.5:

| population        | sum q1^2 | sum -R11 | ratio  |
|-------------------|----------|----------|--------|
| gauss2_deep       | 70491    | +73601   | +0.958 |
| bulgedisc_deep_v2 | 24450    | -89793   | -0.272 |

`sum R11` is POSITIVE for bulgedisc_v2.  `ghat = -R^-1 Q` off a
wrong-signed R is what "`m1` pinned at -1 with a 1e-4 error bar" IS -- and it
explains why the number was tight rather than noisy.  **This is a far cheaper
and sharper health check than `m1`: it needs one arm, not three, and it is
diagnostic rather than a single summary number.  Use it going forward.**

Broken down by flux quintile (same 2000 targets):

| Mf quintile        | bulgedisc_v2 ratio | gauss2 ratio |
|--------------------|--------------------|--------------|
| q1 (faintest)      | **-0.031**         | +0.998       |
| q2                 | **-0.191**         | +0.926       |
| q3                 | +1.018             | +0.908       |
| q4                 | +1.150             | +0.939       |
| q5 (brightest)     | +1.194             | +1.007       |

The failure is ENTIRELY in the two faintest flux quintiles (`Mf < ~1450`,
flux S/N < ~16).  q3-q5 are as healthy as gauss2's.  gauss2's own q1 reaches
`Mf = 254` (S/N ~ 0.9) and is clean, so this is not "faint" per se.

### What it is NOT (measured this session, on top of the previous entry's list)

1. **Not ESS or Monte-Carlo bias.**  `sum R11` over the same 500 targets at
   S = 2048 / 8192 / 32768: +18477 / +21901 / +19589 -- flat over a 16x range.
   gauss2 over the same scan: -18069 / -18127 / -18269.  Kernel ESS is the
   same in both (median 153 vs 149 per 2048 draws; 5th pct 17.7 vs 32.3).
2. **Not the ceiling-violating targets alone.**  Dropping every target with
   `Mr/Mf > POINT_SOURCE` or `Mc/Mr > POINT_SOURCE_MC`, plus a 3% margin
   (13.5% of the catalog), still leaves `sum R11 = +4795`; gauss2 at the same
   target count is -72691.  NOTE this SUPERSEDES the previous entry's item 4:
   the top 20 targets by |R| carry 8.9% of `sum |R|`, but the top 20 by
   SIGNED R11 carry **58%** of `sum R11` and the top 100 carry **92%**.  The
   earlier test measured the wrong quantity.  All 20 of those targets sit at
   `Mf = 780-1080` with `Mr/Mf = 3.0-4.4` and `Mc/Mr = 6.0-7.3`.  The worst
   (R11 = 2.26e4) is just BELOW `sane_targets`' own guard (1000 x median |R|
   norm = 22197) -- the guard is calibrated far too loose for this population.
3. **Not the second-order shear response.**  A single-point grid scan at
   `M1 = M2 = 0`, `Mf = 900` looked conclusive -- `R11` runs -28 flat up to
   `Mr/Mf = 3.2`, crosses zero at ~3.33 and reaches +143 by 3.45, and zeroing
   the second-order spin-0 coefficients (`s0`'s `p2`/`p3` columns) flattens it
   to a stable -9.  It does NOT survive the ensemble.  Full ablation at 4000
   targets x 3 arms:

   | ablation        | m1       | Fisher ratio |
   |-----------------|----------|--------------|
   | full            | -1.28915 | -0.354       |
   | no 2nd spin0    | -1.34367 | -0.419       |
   | no 2nd spin2    | -1.29118 | -0.356       |
   | no 2nd at all   | -1.34654 | -0.422       |

   Every ablation is slightly WORSE.  A clean negative -- do not re-run it.
   (Control through the identical harness: `gauss2_deep` full = **-0.00767**,
   ratio +0.985.  The measurement path is correct.)
4. **Not the shear layer's fit.**  `dm/dg` and `d2m/dg2` against bfd's exact
   per-template truth, on held-out rows, binned by flux AND by `Mr/Mf`: the
   worst bin for bulgedisc_v2 is 6.4% (Mr) / 14.7% (Mc) at the smallest
   `Mr/Mf`, and in the faint quintiles where the estimator dies it is
   0.4%/1.9%/3.3%/3.1%/4.1%.  The response is fit well exactly where R is
   wrong.
5. **Not training-data volume.**  See below -- the 1M retrain has an identical
   density-steepness profile, consistent with the previous entry's item 5.

### The mechanism: the bulk density's SCORE, and where it peaks

`R = E_w[H_g] + Var_w[score_g]` over the importance-weighted draws, and
`score_g = -grad_z log p_bulk . A` with `A = d(unshear)/dg`.  `||A||` is
comparable in the two populations (median 6.2 vs 4.3, p99 34 vs 53), so
`Var_w[score_g]` is set by `|grad_z log p_bulk|`.  Measured on each
population's OWN noiseless training galaxies (4000 rows):

| flow                       | p50   | p90    | p99   |
|----------------------------|-------|--------|-------|
| gauss2                     | 6.4   | 11.6   | 88.6  |
| bulgedisc OLD (pre-retune) | 32.4  | 86.4   | 351   |
| bulgedisc_v2               | 18.2  | 103.2  | 483   |
| bulgedisc_v2_1M            | 18.9  | 97.7   | 476   |

and by flux quintile (p50):

| Mf quintile | gauss2 | bulgedisc OLD | bulgedisc_v2 |
|-------------|--------|---------------|--------------|
| q1 faintest | 6.8    | 17.4          | **33.6**     |
| q2          | 6.1    | 35.1          | 26.9         |
| q3          | 6.0    | 33.8          | 19.7         |
| q4          | 6.2    | 36.8          | 15.5         |
| q5          | 6.9    | 39.7          | 10.4         |

Two things read off this.  First, `bulgedisc_v2` and `bulgedisc_v2_1M` are
identical, so the steepness is a property of the population as this flow
parameterises it, not of sample size or overfitting.  Second -- and this is
the actual difference from the OLD population, which measured a merely-bad
`m1 = +0.16` rather than -1 -- the retune **inverted the flux dependence**.
gauss2's score is flat in flux, the old bulgedisc's RISES with flux (steepest
where the noise kernel is narrowest, i.e. harmless), and the retuned one
FALLS with flux: its density is steepest exactly at the faint end where the
noise kernel is widest.  The retune also moved the population's median flux
down 3x (5090 -> 1770), so that faint end is now where most of the catalog
lives.

A per-point decomposition of `R11 = A^T H A + grad_z(log p_bulk) . U11 +
d2/dg1^2 logdet` confirms which term carries it (the three sum to the full
`R11` to printed precision):

```
bulgedisc_v2, Mf=900     Mr/Mf   full R11   A^T H A   grad.U11   |grad log p|
                          2.800     -28.49    -7.335     -20.70          20.8
                          3.200     -25.18    -7.058     -18.03          47.3
                          3.400     +60.10    -5.065     +69.89         164.3
                          3.500     +80.41    -8.744    +100.60         215.3
gauss2, Mf=1500           2.800    -105.40   -95.750      -4.30           5.3
                          3.300     -21.65   -25.080      +1.17           4.6
```

gauss2's `R11` is carried by the healthy `A^T H A` term with a score of ~5;
bulgedisc_v2's is carried entirely by the score term with a score up to 215.

### Where to pick this up

The lever is the BULK density's steepness at faint flux, not the centroid
layer and not the shear layer.  Concretely, in rough order:

1. Ask whether the retuned population's density really is that sharp or
   whether the flow is manufacturing structure: sample the flow at faint flux
   and compare its marginals against the catalog's, and look at the flux
   floor specifically -- `moments_bulgedisc_v2.fits` has 22% of its galaxies
   between `Mf = 562` and `1103` above a hard cut near 300, i.e. a
   near-discontinuity in `log10 Mf` that the flow must represent as a cliff.
   The OLD population had no such pile-up at its faint end.
2. If the density is genuinely that sharp, the convolution is the problem, not
   the fit: the noise kernel at `Mf ~ 900` is `sqrt(cov00) = 89.6`, i.e. +/-10%
   in flux, straddling the cliff.  Consider whether the retune's flux floor
   should be softened (it is a rendering choice, not real-sky physics) or the
   depth reconsidered so the kernel does not span it.
3. `sane_targets`' `factor = 1000` guard is far too loose here (threshold
   22197 against a worst target of 22735, which it therefore keeps).  It was
   calibrated on a population with `max/median ~70` for |R|.  Worth revisiting
   on its own terms, though it is a symptom-catcher, not the fix.
4. Re-check anything that depends on `split_centroid` actually peeling: it
   never does for a chained flow (see Step 2's last bullet).

### Artifacts

- `bias.py`'s `bulgedisc_deep_v2` `CATALOGS`/`TRAIN_DATA` entries, previously
  uncommitted, are committed with this entry.  No other code changed --
  this session was measurement only.
- Scratchpad diagnostics (not committed): transport round-trip, chart/graft
  comparison, per-target Q/R dumps, the S scan, the `Mr/Mf` grid scans, the
  `R11` decomposition, the flux/`Mr/Mf`-binned `dm/dg` check, the bulk-score
  profile, and the second-order ablation.

## 2026-08-28 (cont.): the flow's density is the defect, not the population --
BFD's own template-sum prior is healthy on exactly the targets the flow fails,
and selection cannot be used to escape it

Follow-up to the entry above, on the user's instruction that `bulgedisc_v2` is
the realistic population and the METHOD must adapt to it.  Also corrects one
of that entry's own tests.

### Correction: the S scan was not an independent convergence test

The previous entry claimed `sum R11` "flat over a 16x range in S".  That used
ONE seed, and `pqr_streamed` seeds chunk `c` as `seed + 7919*c`, so S = 2048 /
8192 / 32768 are NESTED draw sets, not independent ones.  Redone with an
independent seed per point:

| proposal | S = 8192 | 32768 | 131072 |
|----------|----------|-------|--------|
| alpha 0.5 | +59704   | +51771 | +51643 |
| alpha 1.0 | +2074    | -3366  | +6016  |

So the conclusion SURVIVES, and is now properly grounded: `alpha 0.5` is
converged by S = 32768, `alpha 1.0` is not converged at any of these (matching
the `alpha1-ess-does-not-scale` memory).  Four independent seeds at S = 8192,
alpha 0.5, on 500 faint targets: +49565 / +62799 / +49685 / +54062 --
reproducible, so this is estimator BIAS-free convergence to a positive R, not
variance.  gauss2 over the same seeds: -8606 to -8907, ratio 0.945-0.978.

### The sign constraint: what it actually rests on, and it is intact

`sum_i r_i` is the ensemble log-likelihood Hessian.  It is forced negative only
through `E[q^2] = E[-r]`, which needs (a) `Z(g) = INT P(M|g) dM = 1` for all g
and (b) the targets to be drawn from the model.  (a) is what a normalising flow
guarantees -- but only while every layer stays injective, because
`transform_and_log_det` uses `slogdet(...)[1] = log|det|` and will happily keep
going through a fold.

Both checked:

- **The shear map DOES fold.**  `det(d unshear/dz) <= 0` for 0.07-0.11% of
  noisy targets at `g = +/-0.02` (0.00% of noiseless training galaxies, and
  exactly 1.0 at g = 0 in both populations, as it must be).  The folded rows
  sit near the ceiling -- median `Mr/Mf` 3.63, `z1` 2.81.  bulgedisc's folds are
  far more violent than gauss2's (min det -81.6 vs -2.8) but occur at the same
  RATE, so this is not the discriminator.
- **Normalisation survives them.**  `Z(g)` estimated with BOUNDED weights
  (defensive mixture `q = (p_0 + p_g)/2`, so `p_g/q <= 2`; the naive
  `E_{p0}[p_g/p_0]` has unbounded weights and diverges -- its max log-ratio is
  +1128 on bulgedisc and +607699 on gauss2, so do not use it):

  | g1 | bulgedisc_v2 Z | gauss2 Z |
  |------|----------------|----------|
  | 0.000 | 1.00000 (control) | 1.00000 (control) |
  | 0.005 | 0.99984 | 1.00000 |
  | 0.020 | 0.99946 | 1.00011 |
  | 0.040 | 0.99968 | 1.00009 |

  1 to within 5e-4.  The folds are too rare to move the integral.

So the mechanical half of the identity holds, and the zero crossing IS genuine
model misspecification.  Worth stating plainly because it is easy to expect a
hard sign constraint here: there is none per target (P is a Gaussian mixture in
g, not log-concave), and the ensemble constraint is an expectation that assumes
the model is right.

### Where the defect lives: the Mc/Mr ceiling, not the flux floor

The bulk score, on each population's own noiseless training galaxies, sorted by
the slot-2 chart coordinate `v = Mc / (POINT_SOURCE_MC * Mr)`:

| v | bulgedisc_v2 score p50 | gauss2 score p50 |
|---------------|------|------|
| < 0.80        |  8.9 |  5.8 |
| 0.80 - 0.90   | 14.7 |  6.2 |
| 0.90 - 0.95   | 35.0 |  7.4 |
| 0.95 - 0.99   | 96.3 | 14.3 |
| > 0.99        | 355  | none |

Below v = 0.80 bulgedisc_v2 is gauss2-like.  It has **15.5%** of its training
population (17.3% of its noisy targets) above v = 0.95, against gauss2's 1.0%,
and reaches v = 0.998 where gauss2 stops at 0.977.  The flux dependence
reported in the previous entry is a CORRELATE of this: the top 1% by score have
median Mf 1037, Mr/Mf 3.556, v 0.970.  This supersedes the "flux floor / cliff
in log10 Mf" reading -- the flux floor was the wrong suspect.

### Selection is not an escape -- eq. (40) re-imports the error

Re-derived the window for this population (the committed `FLUX_WINDOW =
(2500, 50000)` was calibrated when the median flux was 5090; it is now 1770).
Fisher ratio on 6000 g=0 targets, scanning the cut:

| S/N cut | Mf cut | kept | sum -R11 | ratio |
|---------|--------|------|----------|-------|
| none    | -      | 95%  | -2.38e5  | -0.28 |
| 10      | 896    | 85%  | -1.19e5  | -0.53 |
| 12      | 1076   | 73%  | +9533    | +6.15 |
| 15      | 1345   | 58%  | +4.16e4  | +1.27 |
| 20      | 1793   | 45%  | +3.84e4  | +1.13 |

`sum -R11` passes through ZERO between S/N 10 and 12, which is why the ratio
poles there.  gauss2 sits at 0.99-1.01 for EVERY window including no cut.
Cutting on `v` instead does NOT work (at S/N 0, v < 0.90 keeps 60% and still
gives ratio +2.55, having crossed zero); at S/N >= 15 the ratio is 1.22-1.26 at
every v cut.  So flux/S-N is the operating selection and v is the correlate.

Full measurement, 20000 targets, `--window-size 2.2 3.5 --window-flux 1345 1e9`:

```
  windowed, uncorrected: m1 = -0.01649 +/- 0.00486
  windowed, corrected:   m1 = +0.87631 +/- 0.03566
  unwindowed:            m1 = -1.28003 +/- 0.01342
  quintiles: q1 -1.027, q2 -1.167, q3 +0.100, q4 -0.024, q5 -0.006
```

The cut alone gives an ordinary-looking 1.6% bias.  **The eq. (40)/(45)-(46)
correction then takes it to +0.88**, and this is not a bug in the correction:
`P_s = 0.5486` and `Q_s` consistent with zero are both sane, and the terms came
from `--window-terms templates`, i.e. the true population, not the flow.  The
correction's job is to add back what the CUT galaxies would have contributed --
which is precisely the faint population the flow gets wrong.  So a selection
cannot be used to dodge a density error; eq. (40) puts it straight back.  The
windowed-uncorrected number is only the shear of the selected subpopulation.

### The decisive experiment: BFD's own prior on the same targets

The paper's prior (eq. 38) is a template sum, `P(M|g) = (1/N) SUM_G
N(M - m_G(g); C)` -- a kernel density estimate whose bandwidth IS the noise
covariance, so its score is bounded by construction and no fitted-density spike
can exist.  Its Q and R are exact autodiff of that sum: no flow, no importance
sampling, no proposal, no ESS.  100000 templates, 400 targets per bin:

| targets        | bulgedisc_v2 sum -R11 | ratio | gauss2 sum -R11 | ratio |
|----------------|-----------------------|-------|-----------------|-------|
| faintest 20%   | **+1971**             | +1.26 | +4048           | +1.15 |
| middle 20%     | +4115                 | +0.96 | +14943          | +0.96 |
| brightest 20%  | +4.26e8               | +2.3e5| +92791          | +1.18 |

**At the faint end the template sum is healthy (+1971, ratio +1.26) exactly
where the flow gives -9.3e4 and ratio -0.03.**  The population is not
intrinsically pathological and the noise level is not too deep -- the fitted
density is the defect.

(bulgedisc's BRIGHTEST bin breaks the template sum instead, ratio 2.3e5: 100000
templates cannot cover a flux tail that reaches 2.7e6, so the KDE has no
support out there.  That is the failure mode the flow exists to fix, and it is
a real argument for the flow -- just not at the faint end.)

### What this means for adapting the method

The flow has to reproduce, at the faint/unresolved end, what the template sum
does there.  The property the template sum has and the fitted flow lacks is
that its density is band-limited to the noise scale: BFD's prior is never
sharper than C, so `grad log P` is bounded, whereas the flow is free to put
structure at any scale and does (score p99 ~ 1e5 in z units, and identical on
the 1M retrain, so it is structural rather than overfitting).

Directions, none of them yet built:

1. Make the fitted prior no sharper than the template sum.  Note that "no
   sharper than C" is not a tuned bandwidth -- it is the bandwidth BFD's own
   prior already has.  A noise-split (`train on m + N(0, eps C)`, evaluate with
   kernel `(1-eps)C`) is exact in `P(M|g)` for ANY eps and in a pre-test cut the
   faint-end score p99 from 282878 to 154 at eps = 0.02 while barely moving
   gauss2 -- but eps is a free parameter, and the user has ruled that out as
   tuning.  Any version of this needs eps pinned by an argument, not chosen.
2. Attack the chart instead.  The score climbs with `v -> 1`, i.e. against
   `POINT_SOURCE_MC`.  A population that genuinely reaches v = 0.998 sits at
   `z2 = +6.2` in a logit chart, in a tail with almost no training mass.  A
   smooth population under a logit link would give a BOUNDED score there
   (`grad_z log p -> -1`), so the measured 355 says the flow is not
   representing that tail, not that the chart is wrong in principle.  Worth
   checking whether the z2 marginal is actually fit out there before
   redesigning anything.
3. Use the template sum as the supervision target for the density itself, the
   way `--deriv-weight` already uses bfd's exact derivatives -- the faint-end
   `log P` from the template sum is computable and correct.

### Artifacts

- No code changed.  All measurement.
- Scratchpad (not committed): the window/Fisher scans, the independent-seed and
  alpha convergence scans, the fold determinant test, the bounded-weight
  normalisation test, the score-vs-v localisation, and the template-sum Q/R.
- `.session/next_prompt.md` updated.

## 2026-08-28 (cont.): the selection machinery is VALIDATED and works for
arbitrary windows -- one real bug fixed in it, and the remaining failure is
still the density

Against the requirement that the flow must accept an arbitrary window and
return an unbiased corrected windowed mean shear.  Templates are noiseless and
the PSF is exact, so there is no excuse for eq. (40)/(45)-(46) not to deliver
that; this entry establishes that it DOES, and that what blocks it on
`bulgedisc_v2` is the fitted density, not the correction.

### The correction is correct: three independent checks

**1. The formula.**  `ghat`'s correction is the truncated-data likelihood
`L = PROD_sel P(M_i|g) * PROD_unsel (1 - P_s(g))`, whose first two g-derivatives
give exactly `Q = SUM q - N_ns Q_s/(1-P_s)` and
`R = -SUM r + N_ns (Q_s Q_s^T/(1-P_s)^2 + R_s/(1-P_s))`.  That is what is coded.

**2. `P_s`, `Q_s`, `R_s` against finite differences** of `P_s(g)` computed
directly on the same templates (common random numbers, so only the g dependence
is differenced):

| window | P_s autodiff / direct | R_s11 autodiff / finite-diff | ratio |
|--------|-----------------------|------------------------------|-------|
| size (2.2,3.5)      | 0.945279 / 0.945279 | +4.235e-2 / +4.053e-2 | 1.045 |
| flux > 2500         | 0.876065 / 0.876065 | -1.854e-1 / -1.836e-1 | 1.010 |
| size (2.6,3.3)      | 0.706952 / 0.706952 | +1.645e-1 / +1.627e-1 | 1.011 |
| boxed (2.4,3.4)x(3000,20000) | 0.706959 / 0.706959 | -7.70e-3 / -8.94e-3 | 0.861 |

`P_s` exact to six digits; `R_s` to 1-4% on the three windows where it is
resolvable.  The boxed window's ratio is NOT a discrepancy -- its `R_s` is 20x
smaller and `R_s h^2/2 = 3.8e-7` sits at the float32 noise floor of the
difference.  `window_prob` handles its own moving boundary correctly (the
quadrature substitution carries `t0`, `t1` and the `0.5*(t1-t0)` Jacobian, all
functions of Mf), which was the obvious suspect and is not the problem.

**3. End to end on `gauss2_deep`, seven arbitrary windows** (20000 targets,
S=8192; Q and R do not depend on the window so one pass serves all of them):

| window                | kept  | P_s   | uncorr m1 | corr m1  |
|-----------------------|-------|-------|-----------|----------|
| size (2.2,3.5)        | 0.947 | 0.945 | -0.00524  | -0.00635 |
| flux > 2500           | 0.880 | 0.876 | -0.00899  | -0.00409 |
| flux > 4000           | 0.683 | 0.682 | -0.01185  | -0.00242 |
| size (2.2,3.5) & >2500| 0.852 | 0.847 | -0.00753  | -0.00461 |
| size (2.6,3.3)        | 0.715 | 0.707 | +0.00633  | +0.00064 |
| flux < 12000          | 0.872 | 0.873 | -0.00422  | -0.00979 |
| boxed                 | 0.711 | 0.707 | -0.00222  | -0.00197 |

against unwindowed `m1 = -0.00698 +/- 0.00145`.  The correction cuts the
window-to-window spread from 0.018 to 0.010 and pulls each window toward the
unwindowed value -- most visibly the tight size cut, +0.00633 -> +0.00064.
Residual window-dependence is ~0.005 in m1, which is the accuracy of the
machinery as it stands and is worth knowing against an LSST budget of ~1e-3.

### Bug found and fixed: `selection_terms` NaNs on one bad draw

`--window-terms flow` returned `P_s = NaN` for EVERY window on
`bulgedisc_v2`.  Cause: about 1 prior draw in 262144 overflows float32 (the
flow's `log10 Mf` tail reaches `Mf = 2.1e8`), and one non-finite row NaNs the
whole `jnp.mean` and with it `P_s`, `Q_s` and `R_s`.  `gauss2` produces none,
which is why it was never seen.

Fixed in `selection_terms` by dropping non-finite draws from the sample at
g = 0, OUTSIDE the autodiff, rather than masking in-graph -- measured
directly, an in-graph `jnp.where` mask fixes the VALUE but leaves the
g-Hessian NaN.  Finiteness ONLY, not `in_domain`: `window_prob` needs Mf and
Mr, not chart-domain membership, and `draw` is not required to return
chart-representable moments (`tests/test_selection.py` hands it an analytic
Gaussian population, half of which is off-chart -- adding `in_domain` silently
redefined `P_s` and failed that test).  `gauss2`'s numbers are unchanged to
every printed digit; the count is reported when it fires; 65/65 tests pass.

### With the flow branch working, it is a density diagnostic

`--window-terms flow` vs `templates` measures the flow's density error near
the window edge, which is what that option's docstring says it is for:

| window | R_s11 flow | R_s11 templates | ratio |
|--------|------------|-----------------|-------|
| bulgedisc_v2 size (2.2,3.5) | +2.83e-1 | -2.59e0  | **-0.11** |
| bulgedisc_v2 flux > 1345    | -2.38e-1 | -2.61e-1 | +0.91 |
| bulgedisc_v2 size (2.6,3.3) | -1.52e0  | +2.96e-1 | **-5.14** |
| gauss2 size (2.2,3.5)       | +1.13e-2 | +4.24e-2 | +0.27 |
| gauss2 flux > 1345          | -5.49e-2 | -4.84e-2 | +1.13 |
| gauss2 size (2.6,3.3)       | +1.27e-1 | +1.65e-1 | +0.77 |

On a FLUX window the flow's selection response is right to 9%.  On a SIZE
window it has the WRONG SIGN on `bulgedisc_v2` -- the `Mr/Mf` boundary runs
through the region where the density is spiky (`v = Mc/(rc Mr) -> 1`, previous
entry), and the flow's response to shear across that boundary is not merely
inaccurate but inverted.

### The acceptance test on bulgedisc_v2: still fails, as expected

Same seven-window sweep, `--window-terms templates` (so the correction terms
are exact and only `SUM q`, `SUM r` come from the flow):

| window                | kept  | uncorr m1 | corr m1  |
|-----------------------|-------|-----------|----------|
| size (2.2,3.5)        | 0.807 | -2.61291  | -2.15448 |
| flux > 1345           | 0.633 | +0.03590  | +0.07275 |
| flux > 2000           | 0.446 | -0.05037  | -0.01988 |
| size (2.2,3.5) & >1345| 0.547 | -0.00907  | +0.89979 |
| size (2.6,3.3)        | 0.529 | +5.96962  | +4.24066 |
| flux < 12000          | 0.897 | -1.22832  | -1.22879 |
| boxed (2.4,3.4)x(1500,20000) | 0.385 | -0.02000 | -0.01377 |

Corrected `m1` ranges over -2.15 to +4.24.  Every window that keeps the faint
population is catastrophic; the ones with a flux floor and a bounded bright
end land near zero.  Since the correction terms are exact here, the
window-dependence is entirely `SUM q`, `SUM r` from the flow over the selected
targets: different windows admit different amounts of the region the density
gets wrong, so the answers scatter.  This is the same conclusion as the
previous entry, now demonstrated across windows rather than at one.

### Where this leaves the requirement

The requirement is achievable and the machinery for it is in place and
validated.  What it needs is a prior that is accurate over the WHOLE
population, because `P_s`, `Q_s` and `R_s` are population integrals: the
correction deliberately reaches outside the window, so the flow cannot be made
correct only where the targets are kept.  That is the remaining work, and it
is the density problem of the previous entry, unchanged.

### Artifacts

- `bias.py`: `selection_terms` drops non-finite draws (the only code change).
- Scratchpad: the seven-window sweeps for both populations, the `R_s`
  finite-difference validation, and the flow-vs-templates `R_s` comparison.

## 2026-08-28 (cont.): the selection terms ARE a boundary flux -- verified, and
it makes a flux-windowed measurement on the realistic population work

The user's proposal: the selection terms are a measure of the flux across the
window boundary, so they should be obtainable without the flow being accurate
in the extremes that lie outside the window.  Examined critically, then tested.
It HOLDS, with one direction-dependent caveat, and it CORRECTS an earlier claim
in this file.

### The structural claim, verified

`P_s(g) = E_{m ~ p_0}[F(m_g(m))]` with `F(m) = Pr[m + n in S]`, so
`Q_s = E[grad F . dm/dg]` and `R_s = E[grad F . d2m/dg2 + (dm/dg)^T H_F
(dm/dg)]`.  `F` saturates at 1 deep inside S and 0 deep outside, so `grad F`
and `H_F` live only within a few `sigma_C` of the boundary.  Measured, by
decomposing `R_s` into per-template contributions for a flux window at
`Mf > 1345` (`sigma_Mf = 89.6`):

| \|distance to boundary\| (sigma) | share of R_s |
|--------------------------------|--------------|
| 0 - 1   | 0.663 |
| 1 - 2   | 0.281 |
| 2 - 3   | 0.052 |
| 3 - 5   | 0.004 |
| 5 - 10  | 0.000 |
| > 10    | -0.000 |

Deep inside and deep outside contribute EXACTLY zero.  99.6% of `R_s` comes
from within 3 sigma of the boundary.  The selection terms are a boundary flux.

**This corrects the claim in the previous entry that eq. (40) "re-imports the
faint-end density error" by adding back what the cut galaxies contribute.**  It
does not: it never integrates the prior over the excluded interior.  That
statement was wrong.

### What the prior is actually needed for

- `Q_s`, `R_s`: the boundary collar only (above).
- `P_s`: the window interior plus global normalisation -- NOT boundary-local.
  Minor: it enters only through `1/(1 - P_s)`, and can be replaced by the
  measured selected fraction, accurate to `R_s g^2 / 2 ~ 5e-4`.
- `SUM q`, `SUM r` over the SELECTED targets: each target's convolution reaches
  a few `sigma_C` around itself, so this needs the prior on
  **window UNION a ~5 sigma collar**.

Net requirement: the prior must be right on the window plus a collar, and
nowhere else.  The far tails are genuinely irrelevant.  That is a real
relaxation of "correct everywhere".

Note this could NOT have explained the previous entry's failing sweep, which
used `--window-terms templates`: `P_s`, `Q_s`, `R_s` there already came from
the true noiseless templates with bfd's exact derivatives.  A few-percent error
in `R_s` moves the worst window's m1 by ~0.03 against a scatter of 6.4.  The
failure was, and is, `SUM q` / `SUM r` over the selected targets.

### The falsifiable prediction, and it holds in FLUX

Pre-registered from the flux-quintile Fisher ratios (q1/q2 negative below
`Mf = 1452`, q3 = +1.02 above): with a 5-sigma collar of 448, corrected m1
should be flat for `flux_lo >~ 1900` and break below it.  20000 targets,
`--window-terms templates`, no size cut:

| flux_lo | collar (5 sig) | kept | uncorr m1 | corr m1 |
|---------|----------------|------|-----------|---------|
| 1200 |  752 | 0.701 | +0.30042 | **+0.35793 +/- 0.038** |
| 1400 |  952 | 0.611 | +0.00905 | +0.04429 +/- 0.018 |
| 1600 | 1152 | 0.545 | -0.03662 | -0.00315 +/- 0.015 |
| 1800 | 1352 | 0.488 | -0.04899 | -0.01825 +/- 0.016 |
| 2000 | 1552 | 0.446 | -0.05037 | -0.01988 +/- 0.016 |
| 2200 | 1752 | 0.413 | -0.04691 | -0.01571 +/- 0.018 |
| 2500 | 2052 | 0.372 | -0.06624 | -0.03429 +/- 0.019 |
| 3000 | 2552 | 0.318 | -0.06572 | -0.03415 +/- 0.020 |

Stable at about -0.02 from 1600 up, blowing up to +0.36 by 1200.  **This is a
working measurement on the realistic population: `m1 ~ -0.02 +/- 0.016` keeping
45-55% of the catalog, against -1.28 unwindowed.**  Control on `gauss2_deep`,
where the density is good everywhere: corrected m1 = -0.0059 / -0.0040 /
-0.0035 / -0.0024 / +0.0007 for `flux_lo` 1000 -> 5000 against unwindowed
-0.00698, i.e. FLAT, while uncorrected drifts -0.0063 -> -0.0109.  The
correction is what delivers the window-independence.

### It does NOT hold in the SIZE direction, and here is why

Adding the size window `(2.2, 3.5)` to the same flux sweep makes corrected m1
DIVERGE -- +1.08, +0.91, +1.05, +1.26, +1.57, +1.99, +2.89, +10.15 as `flux_lo`
goes 1200 -> 3000 -- while UNCORRECTED stays sane at -0.02 to -0.06.

The reason is that the assumption "the window falls a few sigma inside the
well-modelled region" is not satisfiable in `Mr/Mf` on this population.  The
collar width against the population's own spread:

| cut direction | 5-sigma collar | population p25-p75 | ratio |
|---------------|----------------|--------------------|-------|
| Mf                    | 448   | 2980 (1103-4083) | 0.15 |
| Mr/Mf at Mf = 4083    | 0.285 | 0.552 | 0.52 |
| Mr/Mf at Mf = 1770    | 0.661 | 0.552 | **1.20** |
| Mr/Mf at Mf = 1103    | 1.069 | 0.552 | **1.94** |

At the median flux a 5-sigma collar in `Mr/Mf` is WIDER than the population's
own interquartile spread, and at the lower quartile it is nearly twice it
(`sigma(Mr/Mf) = (sigma_Mr/Mf) sqrt(1 - 2 rho k + k^2)`, `k = (Mr/Mf)
sigma_Mf/sigma_Mr`, `rho = 0.743`).  There is nowhere to put an `Mr/Mf`
boundary that is a few sigma clear of the badly-modelled region, because a few
sigma spans the whole population.  In flux the collar is 15% of the
interquartile range, so the same window is comfortably satisfiable.

(`gauss2` has an equally wide `Mr/Mf` collar -- 0.640 against a p25-p75 spread
of 0.406 -- and its size window is fine, because a wide collar is harmless when
there is no badly-modelled region for it to reach into.  The collar width only
matters relative to the region being avoided.)

### Standing caveats

- The stable region still drifts -0.003 -> -0.034 between `flux_lo` 1600 and
  3000, larger than the ~0.005 window-dependence the machinery shows on
  `gauss2`.  So a real residual of a few percent survives even with a clean
  collar; this is not a null result.
- Q and R here are S = 8192 (cached).  The windowed samples are bright, where
  `alpha 0.5` is well converged, but the sweep has not been repeated at 32768.
- Rows are nested subsets of one target set, so adjacent rows are correlated
  and the quoted bootstrap errors are not independent between rows.

### Artifacts

- No code change in this entry.
- Scratchpad: the `R_s` boundary decomposition, the flux-collar sweeps for both
  populations, and the collar-vs-spread calculation.

## 2026-08-28 (cont.): how far off at the point-source limit -- the CENTROID
layer's `_safe_logit` clip, firing on 2.2% of the population

Question: how well does the flow match the data as the point-source limits are
approached?  (Above them it is exactly zero by construction -- `raw_from_
standard` applies `POINT_SOURCE * sigmoid(z1) * Mf` -- so only the approach
from below is at issue.  Confirmed too that `shear.lens`'s Taylor-lensed
templates never cross either ceiling: 0/100000 at g = 0.04, and `max u`
actually FALLS with shear, 0.99753 -> 0.99740.  So both `selection_terms`
branches already satisfy "P = 0 above the limit"; no constraint is owed there.)

### The deficit, measured

Flow samples vs the training catalog, mass per bin, ratio flow/catalog:

| bin | bulgedisc_v2 u | bulgedisc_v2 v | gauss2 v |
|-----|----------------|----------------|----------|
| 0.90-0.95 | 1.017 | 1.028 | 0.919 |
| 0.95-0.98 | 0.898 | 0.890 | 0.735 |
| 0.98-0.99 | **0.507** | **0.509** | (empty) |
| 0.99-0.995| **0.203** | **0.254** | (empty) |
| 0.995-1.0 | 0.207 | 0.184 | (empty) |

Matches to 1-3% up to 0.95, then loses half the mass by 0.98 and ~75% by 0.99.
`gauss2` shows the same qualitative deficit but holds only ~1% of its catalog
above 0.95 and never reaches 0.98; `bulgedisc_v2` holds **15.4%** above 0.95
and 2.2% above 0.98.  Same defect, 15x the exposure.

### It is NOT the bulk fit -- it is the centroid layer's clip

`log_prob` on the flow's OWN training galaxies, split three ways:

| v bin | n | bulk only | centroid, Sigma_X = 0 | centroid, Sigma_X real | frac v_out >= 1 |
|-------|---|-----------|------------------------|-------------------------|-----------------|
| < 0.80      | 19461 | -44.87 | -44.87 | -44.86    | 0.0000 |
| 0.80-0.90   | 37956 | -39.50 | -39.50 | -39.55    | 0.0000 |
| 0.90-0.95   | 27122 | -34.25 | -34.25 | -35.90    | 0.0000 |
| 0.95-0.98   | 13248 | -29.72 | -29.72 | -49.61    | 0.0000 |
| 0.98-0.99   |  1869 | -25.61 | -25.61 | **-653.91**  | 0.0054 |
| 0.99-1.00   |   344 | -22.87 | -22.87 | **-4891.54** | **0.7209** |

The bulk flow fits the ceiling region PERFECTLY -- `log p` rises monotonically
-44.9 -> -22.9 as `v -> 1`, exactly as a well-fit density should, and at
`Sigma_X = 0` the centroid layer is the identity and reproduces it bit for bit.
Switching on the real `Sigma_X` collapses it by 4800 nats.

Mechanism: `_transport`'s data -> base direction pushes `v = Mc/(rc Mr)` PAST
1, where `standard_from_raw`'s `_safe_logit` hard-clips.  `frac v_out >= 1` is
0.54% in the 0.98-0.99 bin and **72.1%** in the 0.99-1.00 bin.  A maximum-
likelihood fit could not have left 0.34% of its own training data at
`log p = -4892` (that alone is 16.6 nats of a ~35 nat mean NLL), which is what
first said the bulk was not the culprit.

`_safe_logit`'s docstring anticipates exactly this and argues the affected rows
should be DROPPED -- "the right outcome for a target this ansatz genuinely
cannot represent, not a bug to paper over", on the grounds that the clip zeroes
the gradient so the `jacfwd` log-det comes back `-inf` and `bias.py` discards
the row.  In practice they are NOT dropped: the clip yields a FINITE
`log p ~ -4892`, so the rows survive `sane_targets` and enter the ensemble sums
as essentially-zero-density galaxies.

### Relation to the earlier "centroid falsified" entry

That entry measured the transport's MAGNITUDE -- `trace(P)` p50/p99
0.005/0.038, round-trip residual, log-det spread over kernel draws -- and found
`bulgedisc_v2` milder than `gauss2` in every column.  **That stands.**  What it
never tested was whether the CLIP guarding the ansatz's failure mode fires.  It
fires on 2.2% of this population, concentrated exactly where the Fisher ratio
inverts and where the top-20 R-dominating targets sit (`Mc/Mr` 6.0-7.3, i.e.
`v` 0.90-1.10).  So the Gaussian-in-k approximation's SIZE is not the problem,
as the user said; its failure HANDLING is.

### Why this explains the size-window result

`Mr/Mf = 3.5` is `u = 3.5/POINT_SOURCE = 0.948` -- the size window's upper edge
sits exactly at the knee where the fit starts degrading, so its boundary collar
is in the damaged region while its SAMPLE is clean (the cut removes 100% of
`u > 0.95` by construction, yet 8.0% of the kept targets still have
`v > 0.95`).  The flux window keeps 5-8% of `u > 0.95` and `v > 0.95` targets
and still works, because its boundary is in a well-modelled place.  That is the
boundary-flux argument in its sharpest form: what matters is where the BOUNDARY
sits, not whether the sample contains badly-modelled galaxies.

### Next

The concrete defect is now a handful of rows whose centroid transport leaves
the chart.  Options, none tested: honour the docstring's own intent and make
those rows genuinely non-finite so they are dropped (changes which targets are
measured, and 2.2% is not negligible); bound the transport so `v_out < 1` by
construction the way `_transport`'s `sign < 0` branch already bounds
`trace(P)`; or reconsider the chart at the ceiling as the user suggested, so a
population that piles up against the point-source limit is not mapped to
`z = +inf`.

### Artifacts

- No code change in this entry.
- Scratchpad: flow-vs-catalog mass ratios near both ceilings, the
  bulk/Sigma_X=0/Sigma_X-real `log_prob` split, the lensed-template ceiling
  check, and the selected-sample composition table.

## 2026-08-28 (cont.): dropping the off-chart rows is a NO-OP -- and the
previous entry's inference from `log p = -4892` was wrong

Tried the fix the previous entry proposed and `_safe_logit`'s docstring argues
for: instead of clipping a transport whose base point leaves the chart, give
that row zero density.  Implemented in `CentroidMarginalize.transform_and_log_
det` as `jnp.where(on_chart(_transport(...)), log_det, -inf)` -- applied to the
LOG-DET, not the output, so `y` stays finite and the `-inf` is a g-INDEPENDENT
additive constant (zero weight in `log_conv_is`'s logsumexp, zero gradient, no
0 * inf).

Rerun of the identical configuration (2000 g = 0 targets, S = 8192, alpha 0.5,
seed 12345):

| | clip (before) | -inf drop (after) |
|--------------------------|---------|---------|
| bulgedisc_v2 `sum R11`   | +89793  | +88006  |
| bulgedisc_v2 Fisher ratio| -0.272  | -0.278  |
| gauss2 `sum R11`         | -73601  | -73578  |
| gauss2 Fisher ratio      | +0.958  | +0.958  |

**No change.**  The clip fires often -- 7.8% of `bulgedisc_v2` targets' own
transports leave the chart (3.3% for gauss2), 2.2% of in-domain kernel draws
do, and 40.7% of targets have at least one such draw (32.8% for gauss2) -- and
it makes no difference, because `log p ~ -4892` ALREADY underflows to exactly
zero weight in float32.  Replacing zero with zero.

### Correcting the previous entry

It concluded that the clipped rows "survive `sane_targets` and enter the
ensemble sums as essentially-zero-density galaxies", implying they poison the
sums.  **They do not.**  They enter with zero weight and contribute nothing.
The `-4892` measurement was a POINT-DENSITY evaluation on noiseless training
galaxies, which is not how `bias.py` uses the flow -- it uses it inside a
weighted logsumexp over kernel draws, where such a row is simply absent.  The
centroid clip is NOT the cause of the R sign flip.

What that entry established and which still stands: the bulk flow fits the
ceiling region well (median `log p` rising monotonically -44.9 -> -22.9 as
`v -> 1`, and identical at `Sigma_X = 0`), and the centroid layer at the real
`Sigma_X` collapses the POINT density there.  The collapse is real; its
consequence for Q and R is nil.

The change was REVERTED: it costs an extra `_transport` per draw in the hot
path (`transform_and_log_det` runs inside `log_conv_is` for every draw) and
buys nothing measurable.  `tests/test_centroid.py` 12/12 after the revert.

Also note the user's objection, which is correct and is why the alternative was
not attempted: `v_out < 1` is NOT guaranteed by anything, so bounding the
transport to enforce it would be inventing a constraint rather than restoring
one.

### What is still unexplained

The R sign flip at the faint/unresolved end.  Not the centroid approximation's
magnitude, not its clip, not the selection machinery, not ESS/MC, not the
second-order shear response, not training-data volume.  The flow's SAMPLING
deficit near the ceilings (flow/catalog mass ratio 0.51 at 0.98, 0.20-0.25 at
0.99) has not been attributed -- it was measured at the real `Sigma_X`, so it
could be the centroid layer's generative direction rather than the bulk, and
that split has not been made.  That is the obvious next measurement: repeat the
flow-vs-catalog mass comparison with `Sigma_X = 0`, where the bulk is known to
fit.

### Artifacts

- No net code change (the experiment was reverted).
- Scratchpad: `dropclip.py`, which reports the clip rates and the before/after
  Fisher ratios.

## 2026-08-28 (cont.): the near-ceiling "deficit" is the physical centroid
marginalisation -- the bulk fit is clean, and that claim is withdrawn

Closing the gap left open above.  The earlier deficit was measured by sampling
the flow at the REAL Sigma_X and comparing against the NOISELESS catalog, which
is not apples-to-apples: centroid marginalisation physically lowers Mf, Mr/Mf
and Mc/Mr, so the marginalised population should hold fewer galaxies near the
ceilings.  At `Sigma_X = 0` the transport is the identity and the comparison is
a clean test of the bulk.

Mass per bin, ratio to the catalog, `bulgedisc_v2` (500k samples each):

| v bin | catalog | bulk only | centroid Sigma_X = 0 | centroid Sigma_X real |
|-------|---------|-----------|----------------------|-----------------------|
| 0.00-0.80 | 0.19461 | 1.00 | 1.00 | 1.03 |
| 0.80-0.90 | 0.37956 | 1.00 | 1.00 | 1.03 |
| 0.90-0.95 | 0.27122 | 1.01 | 1.01 | 1.03 |
| 0.95-0.98 | 0.13248 | 1.01 | 1.01 | 0.89 |
| 0.98-0.99 | 0.01869 | 1.00 | 1.00 | 0.51 |
| 0.99-1.00 | 0.00344 | 0.87 | 0.87 | 0.26 |

and the same for `u` (1.00/1.00/1.01/1.01/0.99/0.86 bulk).

**The bulk fits the ceiling approach essentially perfectly** -- 1.00-1.01 in
every bin down to 0.98, with only the last bin (0.34% of the catalog, a 5.6
sigma deficit on 344 galaxies) at 0.86-0.87.  Bulk-only and `Sigma_X = 0` agree
to the last printed digit, which independently confirms the centroid layer is
exactly the identity at `Sigma_X = 0`.

**The whole deficit is the centroid marginalisation, and it is physical.**  Mass
is depleted at the top AND enriched at the bottom (1.02-1.03 in the low bins) --
conserved and moving downward, exactly the direction the module docstring gives.
`gauss2` shows the same pattern in its populated bins (u 1.04/0.98/0.90/0.86;
v 1.13/1.00/0.92/0.74) with a LARGER median shift than `bulgedisc_v2`
(median v 0.8625 -> 0.8584, i.e. -0.0041, against 0.8851 -> 0.8819, -0.0032).
Normal behaviour of a working layer.

**The claim "the flow under-populates the last 5% before the ceiling" is
therefore WITHDRAWN**, along with the reading of it as a fit failure the chart
might need changing to fix.  What remains true from that entry: the flow is
exactly zero above both ceilings by construction, and lensed templates never
cross them.

### A new inconsistency, flagged not asserted

`selection_terms`' two branches draw from DIFFERENT populations.  `templates`
lenses the noiseless, UN-marginalised templates; `flow` draws the centroid-
marginalised population at the real `Sigma_X`.  The targets are centroid-
affected, so in principle the marginalised one is right and the templates
branch is missing a step.  The data does not cleanly pick a side -- observed
kept-fraction minus `P_s`, over three windows:

| window | templates | flow |
|--------|-----------|------|
| size (2.2,3.5)  | +0.0005 | -0.0081 |
| flux > 1345     | -0.0024 | +0.0029 |
| size (2.6,3.3)  | +0.0065 | -0.0013 |

Both are within ~0.8% and neither is uniformly better.  Worth resolving on its
own terms, but it is not the size of effect that explains anything measured
here.

### Where the R sign flip stands

Still unexplained.  Now also NOT the bulk's fit near the ceilings, NOT the
centroid clip, NOT the centroid transport's magnitude, NOT the selection
machinery, NOT ESS/MC or the proposal, NOT the second-order shear response, NOT
training-data volume.  What is established positively: BFD's template-sum prior
is healthy on the same faint targets where the flow inverts (Fisher ratio +1.26
against -0.03), and a flux window whose collar clears `Mf ~ 1450` gives a stable
`m1 ~ -0.02`.

### Artifacts

- No code change.
- Scratchpad: `gapclose.py`.

## 2026-08-28 (cont.): flow vs template sum on the COMPOSITE -- log P is right,
R is not.  A retrain is ruled out.

Every diagnostic before this checked COMPONENTS (marginals, dm/dg, d2m/dg2,
transport magnitude, log-det, ESS) and all of them pass in the failing region,
which is why "train longer" kept looking plausible.  This compares the thing
`bias.py` actually consumes -- `log P(M|g)`, `Q`, `R` at g = 0 -- target by
target against BFD's template sum, both as absolute densities in 5-D moment
space so there is no free constant.

200 targets per flux group, S = 8192, alpha 0.5, 100000 templates:

| population / bin | quantity | flow | templates | median diff | corr |
|------------------|----------|------|-----------|-------------|------|
| bulgedisc faintest 20% | log P | -37.2367 | -37.2822 | +0.0099 | **0.995** |
| bulgedisc faintest 20% | Q1    |   0.0699 |   0.0722 | +0.0378 | 0.896 |
| bulgedisc faintest 20% | R11   | **+56.54** | **-3.83** | +60.33 | **0.319** |
| bulgedisc faintest 20% | sum R11 | +3.155e4 | -1024 | | |
| bulgedisc middle 20%   | log P | -39.5099 | -39.4899 | -0.0370 | 0.989 |
| bulgedisc middle 20%   | R11   | -11.7675 |  -8.6440 | -0.8544 | 0.085 |
| bulgedisc middle 20%   | sum R11 | -1887 | -2984 | | |
| gauss2 faintest 20%    | log P | -42.7795 | -42.7171 | -0.0167 | 0.995 |
| gauss2 faintest 20%    | Q1    |   0.2273 |   0.0958 | +0.0016 | 0.983 |
| gauss2 faintest 20%    | R11   |  -9.2728 |  -8.9939 | -0.1617 | 0.516 |
| gauss2 faintest 20%    | sum R11 | **-1956** | **-1976** | | |
| gauss2 middle 20%      | R11   | -32.4002 | -32.2882 | +0.1688 | 0.965 |
| gauss2 middle 20%      | sum R11 | -7320 | -7381 | | |

**The flow's `log P` is right** -- correlation 0.995 and a median offset of 0.01
nats against an exact template sum.  `Q` is broadly right (0.896).  `R` is
wrong in sign and magnitude, correlation 0.32.  The gauss2 control reproduces
the template sum's `R` to **1%** (sum -1956 vs -1976; middle bin -7320 vs
-7381, corr 0.965), so the comparison itself is sound and the flow CAN get R
right when the population allows it.

(The bulgedisc BRIGHTEST row is the template sum failing, not the flow:
`sum R11 = -2.67e8`, the known sparse-coverage breakdown at a flux tail
reaching 2.7e6.  Disregard it; the flow is the trustworthy side there.)

### Consequence: a retrain is not the answer

Training drives `log P`, and `log P` already matches an exact reference to 0.01
nats in the bin where the estimator inverts.  More steps or a different
schedule optimises a quantity that is already correct.  That is consistent with
the three nulls already on record -- the 23-run capacity sweep, the 10x data
retrain, and shear steps past the 60k plateau -- none of which moved the bias,
and it explains WHY they were null rather than leaving it a coincidence.

### What the defect actually is

`R = E_w[d2 log p] + Var_w[d log p]` over the importance-weighted draws.  The
flow reproduces the integral (`log P`) and the first moment of the score (`Q`)
but not the second.  The measured bulk score is 18-34 in z units against
gauss2's 6.4, while the template sum's score is BOUNDED by construction --
its bandwidth is exactly the noise covariance C, so it cannot carry structure
finer than the kernel it is convolved with.  The flow can and does, and the
weighted VARIANCE of that structure is what inflates R positive.

So this is a SMOOTHNESS problem, not a fit problem.  That returns to the
band-limiting idea, previously set aside because `eps` was a free knob -- but
now with a knob-free statement of the target: the flow's score across the
kernel should behave like the template sum's, whose bandwidth is set by C
rather than chosen.  How to impose that without introducing a tuned parameter
is the open design question, and nothing here settles it.

### Artifacts

- No code change.
- Scratchpad: `composite.py`.  Note `bias.pqr` returns only (Q, R);
  `bias.pqr_full` is the one that also returns `log P`.

## 2026-08-28 (cont.): where the spikiness comes from -- the JACOBIAN, forced
by heavy-tailed chart coordinates.  Not band-limiting.

### The base coordinates are textbook -- the flow normalises correctly

Catalog pushed to base space through the BULK checkpoint only (`rest.inverse`
is the data -> base direction; `rest.transform` is not, checked empirically by
which gives robust sd ~ 1):

| | bulgedisc_v2 | gauss2 | reference |
|---|---|---|---|
| `\|z_base\|` p50 / p99 / p99.9 / max | 2.137 / 3.692 / 4.246 / 7.65 | 2.097 / 3.836 / 4.457 / 6.91 | chi_5 p99.9 = 4.42 |
| per-slot robust sd | 0.996 0.997 0.992 1.195 1.188 | 1.001 1.001 1.004 1.030 1.021 | 1 |
| frac any slot > 3.09 | 0.0059 | 0.0091 | 0.002 |

Indistinguishable, and bulgedisc has FEWER extreme points than gauss2.  An
earlier reading in this file of base-space `p99.9 = 113` was the CENTROID flow
with a handful of outliers and is withdrawn.

### The score is the log-det term

`log p(z) = log p_base(T(z)) + SUM_i log|det J_i|`, so the score splits into
`-(dT/dz)^T T(z)` and `grad_z SUM log|det|`:

| | base pullback p50 | log-det p50 | total p50 |
|---|---|---|---|
| bulgedisc_v2 | 24.17 | **20.53** | 18.91 |
| gauss2       |  7.88 | **2.51**  |  6.36 |

The log-det term is **8x** larger; the two partially cancel, which is why the
total is below the base part.  And by concentration bin the score climbs
8.91 -> 14.72 -> 35.01 -> 89.72 -> 233.65 while `|z_base|` moves only
2.249 -> 3.191: **the score grows 26x while the base position grows 1.4x**, so
all of the growth is in the Jacobian, none in where the point lands.
`corr(|score|, |z_base|) = 0.370`.

### Why the Jacobian has to be that stiff

The chart coordinates the bulk must Gaussianise:

| slot | bulgedisc skew | bulgedisc exc kurt | gauss2 skew | gauss2 exc kurt |
|------|----------------|--------------------|-------------|-----------------|
| 0 log10 Mf | **1.775** | **3.807** | -0.014 | 0.007 |
| 1 logit u  | 0.184 | 0.110 | 0.275 | -0.468 |
| 2 logit v  | 0.239 | 0.261 | 0.215 | -0.605 |
| 3 spin-2   | 0.035 | **3.435** | -0.002 | -0.084 |
| 4 spin-2   | -0.016 | **3.328** | -0.012 | -0.059 |

gauss2's are already near-Gaussian (|exc kurt| < 0.61); bulgedisc's are heavily
SKEWED in flux and heavy-TAILED in ellipticity.  The bulk is a stack of AFFINE
MAF steps (`Spin0AutoregressiveLayer` loc + `_bounded_log_scale`,
`Spin2CouplingLayer` shared scale -- no splines), and an affine map can only
Gaussianise a heavy tail by varying its scale rapidly.  `grad log|det| =
-grad SUM ls` IS that rate.  The stiffness is the price of compressing heavy
tails with affine maps, and the score inherits it.

Note the two heavy slots are exactly what the real-data retune introduced: the
flux power law spanning 3.5 decades, and the `|e|` marginal widened to match
DES/COSMOS (2026-08-27 entries).

Worth noting the near-ceiling coordinates (slots 1 and 2) are NOT the heavy
ones -- the score's correlation with `v` is real but `v` is a correlate of
faintness, not the cause.  The earlier reading that the chart's logit stretch
at the ceiling was to blame is not supported by this.

### What this implies

The defect is a BASE / TRANSFORM-FAMILY mismatch, not a density-fit problem and
not something to fix by smoothing the density.  Structural options, none tested
and none carrying a tuned bandwidth:

1. A heavier-tailed base (e.g. Student-t) so the transport does not have to
   compress a kurtosis-3.8 coordinate into a Gaussian at all.
2. A chart that pre-Gaussianises slot 0 the way the logit already handles the
   two size ratios -- the flux power law is the largest single offender
   (skew 1.775).
3. A transform family that can absorb tails without a steep scale.

### Artifacts

- No code change.
- Scratchpad: `spikesrc.py`.  Note `Invert(Chain(bulk)).inverse` is the
  data -> base direction, not `.transform`.

## 2026-08-28 (cont.): capacity is a hard null, and a knob-free route --
Fisher scoring on a cross-fit OPG metric instead of the broken R

### More layers/width/depth: NULL, as the decomposition predicted

`grad log|det J| = grad log p - J^T grad log p_base` is fixed once the DENSITY
is fixed, so architecture can only move it by changing the fitted density --
and the density already matches the template sum to 0.01 nats.  Tested anyway,
bulk only, 20000 steps, same data and seed:

| | held-out NLL | score p50 at the Mf 900-1100 peak |
|---|---|---|
| L8 W64 D2 (current) | 38.9748 | 36.50 |
| L16 W128 D3 | 38.9709 | 37.13 |

4x the parameters and 3.4x the training time buys **-0.004 nats** and leaves
the score marginally HIGHER.  Every flux bin agrees within ~3%.  Together with
the retrain null (`log P` already right) this closes the "train bigger/longer"
direction on this population.

### Where the stiffness actually sits: slots 1 and 2, at the flux TURNOVER

Per-slot decomposition of the score, `bulgedisc_v2`:

| Mf bin | \|score\| p50 | slot0 | slot1 | slot2 | spin2 | slot0 share of q^2 | dlnN/dlog10Mf |
|--------|---------------|-------|-------|-------|-------|--------------------|---------------|
| < 700       | 14.83 | 7.27 |  7.87 |  6.70 | 0.90 | **0.495** | **+22.3** |
| 700-900     | 30.54 | 3.47 | 19.06 | 17.91 | 2.52 | 0.164 | +22.1 |
| **900-1100**| **36.56** | 1.65 | **25.92** | **24.61** | 3.36 | **0.001** | **-0.13** |
| 1100-1300   | 27.14 | 1.22 | 19.54 | 18.48 | 2.69 | 0.000 | -2.34 |
| 3000-5000   | 14.53 | 1.12 | 10.46 |  9.87 | 1.59 | 0.000 | -1.80 |

This REFUTES "the flow cannot learn the steep faint-end die-off".  At the score
peak slot 0 carries 0.1% of the squared score; it is slots 1 and 2 (the logit
size and concentration coordinates) at 25.9 and 24.6.  And the peak sits where
the flux marginal is FLAT (`dlnN/dlog10Mf = -0.13`, the turnover), not where it
is steep.  In the genuine die-off (Mf < 900, slope +22) slot 0 DOES carry the
score (share 0.50 and 0.16) exactly as that hypothesis predicts -- but the total
score there is LOWER and those bins hold 9.6% of the catalog.  gauss2's slots 1
and 2 are flat at ~1.9-2.1 everywhere, a 13x gap.

Since slots 1 and 2 have near-Gaussian MARGINALS (excess kurtosis 0.11, 0.26),
this is CONDITIONAL structure: `p(z1|z0)` and `p(z2|z0,z1)` must swing hard as
flux crosses the turnover.  By flux, the score is 30.5 at S/N 7.8-10, peaks at
36.6 at S/N 10-12.3, and falls monotonically to 9.6 above S/N 111; gauss2 is
FLAT at 5.9-7.2 across its whole range including its faintest bin at S/N 7.4,
so this is not a depth effect -- at matched S/N ~ 10 it is 36.6 against 6.3.

### A knob-free route: Fisher scoring, not Newton

`bias.ghat` takes a Newton step `ghat = -(SUM r)^-1 SUM q`, i.e. it consumes
the one quantity verified BROKEN while `SUM q` is verified right (corr 0.896
against the template sum, 0.983 on gauss2).  Fisher scoring solves the same
likelihood equation with the outer-product (BHHH/OPG) metric,

    ghat = (SUM q q^T)^-1 SUM q

which has no free parameter and is positive-definite by construction.  On the
cached 20000-target run:

| | Newton (r) | OPG (q q^T) |
|---|---|---|
| bulgedisc unwindowed | **-1.27730** | **-0.15594** |
| bulgedisc flux > 1600 | -0.03662 | -0.17674 |
| gauss2 unwindowed | -0.00698 | **-0.03605** |
| gauss2 flux > 2200 | -0.00739 | -0.03469 |

Unwindowed bulgedisc goes from -128% to -16% with no window and no parameter.
OPG carries its own **-3.5% bias on gauss2**, stable across windows -- and that
bias is the standard noise inflation `E[qhat qhat^T] = q q^T + Var[eps]`, which
overestimates the information and shrinks `ghat`.  Confirmed by its S scaling
(3000 targets, seed 4242):

| | S = 8192 | S = 32768 |
|---|---|---|
| gauss2 Newton | -0.00391 | -0.00907 |
| gauss2 OPG | -0.01187 | **-0.00979** |
| bulgedisc Newton | -1.27184 | -1.28456 |
| bulgedisc OPG | -0.22423 | **-0.19192** |

gauss2's OPG converges onto Newton's answer as S grows, so the excess is
O(1/S), not a modelling error.

**That makes it exactly removable with no tuned constant**: cross-fit the outer
product over disjoint draw sets, `E[qhat^(A) qhat^(B)T] = q q^T` when A and B
are independent, so the `eps eps^T` term vanishes in expectation rather than
being subtracted by a fitted correction.  `pqr_streamed` already stores
per-chunk `B/A` values for its delete-one jackknife, so the independent halves
are in hand.

Caveats, unresolved: OPG equals Newton only for a correctly specified model, so
this sidesteps the broken `R` rather than fixing the density; bulgedisc's OPG
is still -0.19 at S = 32768 and falling, and whether cross-fitting takes it to
zero is untested; and `ghat`'s selection branch assembles `R` from `-SUM r`
plus the eq. (40) terms, so it would need the OPG form threaded through
consistently.

### Artifacts

- No code change.  Scratchpad: `byflux.py`, `whichslot.py`, `capacity.py`
  (which also writes two bulk checkpoints), and the OPG comparisons.

## 2026-08-28 (cont.): the three OPG caveats, settled -- one pass, one fail,
one that inherits the failure

Implemented Fisher scoring with a cross-fit outer-product metric
(`_merge_opg`, `pqr_streamed(opg=True)`, `ghat(..., opg=)`, `bias(..., opg=)`
-- opt-in, inert by default, 65/65 tests pass) and settled all three open
caveats.

### 1. Correct specification -- PASSES

Does cross-fit OPG reproduce Newton where Newton is right?  `gauss2_deep`,
3000 targets, seed 4242:

| S | Newton | OPG plain | OPG cross-fit |
|---|--------|-----------|---------------|
| 8192  | -0.00391 | -0.01187 | -0.00964 |
| 32768 | -0.00907 | -0.00979 | **-0.00931** |

At S = 32768, where both are converged (Newton itself moves 0.005 between the
two rows, so the 8192 row is not a fair reference), cross-fit OPG agrees with
Newton to **2.4e-4**.  The -3.5% plain-OPG inflation is gone.  The estimator is
sound where the model is.

### 2. Does cross-fitting take bulgedisc to zero -- NO

| S | Newton | OPG plain | OPG cross-fit |
|---|--------|-----------|---------------|
| 8192  | -1.27183 | -0.22423 | -0.17529 |
| 32768 | -1.28457 | -0.19192 | **-0.18691** |

It plateaus at about **-0.19**, stable over 4x in S and drifting slightly AWAY
from zero.  So OPG turns a -128% catastrophe into a -19% bias -- a large
mitigation, still ~190x an LSST budget.  In hindsight this is what the
composite comparison predicted: OPG consumes `Q`, and `Q` itself correlates
only 0.896 with the template sum on the failing bin.

### 3. The selection branch -- threads through, but inherits a worse problem

Mechanically fine: only the `-SUM r` term is replaced, the analytic `Q_s`/`R_s`
are untouched, and the corrected windowed m1 is window-STABLE under OPG.
`gauss2_deep`, 20000 targets, S = 8192, flux windows:

| flux_lo | kept | Newton corr | OPG cross corr |
|---------|------|-------------|----------------|
| 0    | 1.000 | -0.00483 | -0.03409 |
| 900  | 0.996 | -0.00340 | -0.03331 |
| 1200 | 0.987 | -0.00294 | -0.03293 |
| 1600 | 0.965 | -0.00241 | -0.03177 |
| 2000 | 0.933 | -0.00221 | -0.03080 |

Spread 0.003 for OPG against Newton's 0.0026 -- equally window-independent.
But it sits at a **-0.033 offset** on the same population where the n = 3000
run gave -0.0096.  A bias must not depend on sample size, and this one does.

**Cause, measured:** `SUM q q^T` is far more outlier-dominated than `-SUM r`,
because it SQUARES `Q`.  On `gauss2_deep` (19976 sane targets):

| | share of `SUM q1^2` | share of `SUM -R11` |
|---|---|---|
| top 1 target    | 0.0028 | 0.0005 |
| top 100         | 0.0849 | 0.0259 |
| top 1000        | 0.3865 | 0.1778 |
| max / median    | **181.2** | 13.4 |

So as the catalog grows, more extreme `|Q|` enter, `SUM q q^T` inflates faster
than `SUM -R`, `ghat` shrinks and m1 goes more negative -- exactly the observed
-0.0096 (n = 3000) -> -0.033 (n = 20000).  Suppressing it needs a tightened
`|Q|` guard (`sane_targets`' factor 1000 is nowhere near tight enough for a
squared metric), i.e. precisely the tuned constant this route existed to avoid.

### Verdict

Cross-fit OPG is a genuine, knob-free mitigation and a sound estimator on a
correct model, but it is NOT the answer: -0.19 on the target population, and a
sample-size-dependent bias that can only be closed with a tuned cut.  Recorded
in `ghat`'s docstring so it is not adopted naively.  The code is kept because it
is opt-in, inert by default, and the only way to reproduce these numbers.

For comparison, the best measurement on this population remains the flux window
under the ordinary Newton step: `m1 ~ -0.02 +/- 0.016` for `flux_lo` >= 1600,
stable to 3000 (2026-08-28 collar entry).

### Artifacts

- `bias.py`: `_merge_opg`, `pqr_streamed(opg=True)`, `ghat(..., opg=)`,
  `bias(..., opg=)`.  All opt-in; no default behaviour changes.
- Scratchpad: `opgtest.py`.

## 2026-08-28 (cont.): the road to 1e-3 -- the gate is a ~5e-3 floor on
gauss2, and it is NOT draws, targets, or the estimator

Asked how to reach `|m1| ~ 1e-3`.  The honest starting point is that NOTHING
measured in this session reaches it, including on the population where the
method works: `gauss2_deep` sits at `-0.0070 +/- 0.0015` at S = 8192, ~5 sigma
from zero and 7x the target.  So `bulgedisc_v2` is not the only obstacle, and
until the gauss2 floor is understood no fix to it can be validated -- a 1e-3
improvement is invisible under a 7e-3 floor.

### S-convergence: settled, and it closes the "more draws" route

Same 19976 targets, same seed, so the draw sets are NESTED across S and the
shape noise is common:

| S | chunks k | jackknife | m1 | delta |
|---|----------|-----------|----|-------|
| 2048  | 1  | **off** | -0.015682 +/- 0.003100 | -- |
| 8192  | 4  | on | -0.006978 +/- 0.001509 | +0.00870 |
| 32768 | 16 | on | **-0.006190 +/- 0.001205** | **+0.00079** |

Two readings, and the first is a confound worth recording:

* **`chunk = 2048` makes S = 2048 a SINGLE chunk, and `_merge_finish` falls
  back to the un-jackknifed plain estimator (`if not jackknife or k < 2`).**
  So that row is not "fewer draws", it is "no bias correction", and the +0.0087
  step to S = 8192 measures what the jackknife is WORTH, not an S trend.  Any
  run with `--samples <= --chunk` is silently uncorrected.
* Between the two JACKKNIFED points, 4x the draws bought **+7.9e-4**.  If that
  residual is O(1/S) the asymptote is about **-0.0052**, so only ~1.1e-3 of the
  S = 8192 number is estimator residual.

So: more draws will not reach 1e-3 (the S residual is ~1e-3 and nearly
exhausted by S = 32768), and more targets will not either -- they shrink the
+/-0.0012, not the -0.0062 central value, which is 5.1 sigma from zero.  The
Fisher ratio is 1.002-1.008 at every S, so the model is not the issue here
either.

### What the ~5e-3 floor probably is

`models/centroid.py`'s own header records that on `copies_gauss2_deep` -- where
the Gaussian-in-k transport is claimed EXACT, gauss2 being literally a Gaussian
mixture -- the ellipticity response still shows "a roughly flat 25-30%
UNDERSHOOT across the whole population".  The centroid layer is worth +0.0138
in m1 (`centroid-layer-status` memory).  A 30% shortfall on +0.0138 is ~4e-3,
which is the measured floor to within its error.

That is a specific, cheap test rather than a hypothesis: `--no-centroid` on
`gauss2_deep` at S = 32768.  If the layer supplies ~+0.014 of a needed ~+0.020,
the arithmetic closes and the 1e-3 target becomes a question about the
transport's spin-2 response, not about `bulgedisc_v2` at all.  NOT YET RUN.

### The three gaps to 1e-3, with sizes

| gap | size | status |
|-----|------|--------|
| model error on the realistic population | -0.02 (windowed, flux >= 1600) | diagnosed, not fixed |
| method floor on a GOOD model (gauss2) | **-0.0052 asymptotic** | suspect identified, untested |
| estimator O(1/S) residual after jackknife | ~1.1e-3 at S = 8192 | measured; halve it with S = 32768 |
| statistical, 20000 targets | +/-0.0012 at S = 32768 | 200k targets are on disk; ~10x more is affordable but SLOW (a 200k x 2-arm run at S = 8192 projects to ~30 h) |

Order of attack: the gauss2 floor first (it gates validation of everything
else), then the realistic-population model error, and only then statistics.

### Artifacts

- No code change.  Scratchpad: `floor.py`, `floor_S*.npz` (per-target Q, R and
  observed moments at each S, for offline reuse).

## 2026-08-28 (cont.): the gauss2 floor IS the centroid layer's known
undershoot; and the analytic route is blocked by the stale truth.py chart

### Can the remaining bias be determined analytically?

For `gauss2` in principle YES -- `truth.py` gives exact `P(m|g)`, `Q`, `R` in
closed form, so the decomposition is available:

    m1(exact Q, R)  = the estimator's INTRINSIC bias with a perfect model
    m1(flow  Q, R)  = the total
    difference      = model error, isolated

(For the noisy `_deep` targets it needs `INT P(m|g) k(M-m) dm` with
`truth.log_prob` in place of the flow -- the same `log_conv_is` structure, so
the machinery exists.)

**It is blocked.**  The 2026-08-20 stale-chart bug is unfixed, and re-measured:

| `bulk.to_coords` slot | `sim.GAUSS2_MU` (what P0 assumes) | catalog mean (what was generated) | offset |
|---|---|---|---|
| 0 log10 Mf | 3.7874 | 3.7407 | 0.16 sd |
| 1 | 2.0477 | 1.5074 | 0.75 sd |
| **2** | **5.9740** | **1.8615** | **8.3 sd** |

`truth.py` evaluates P0 in a chart where slot 2 sits 8.3 sigma from where the
catalog actually lives.  Fixing it is NOT a mean/cov swap: the catalog's
coordinates are mildly non-Gaussian in the CURRENT chart (slot 2 skew 0.215,
excess kurtosis -0.605), which is what a population drawn Gaussian in the OLD
chart looks like after the slot-2 logit was redefined.  It needs the actual
chart composition.  Worth doing -- it is the most powerful diagnostic available
for this whole problem and would settle the model/estimator split exactly.

### Meanwhile, the floor is attributed empirically

`gauss2_deep`, S = 32768, 19976 targets:

| | m1 |
|---|---|
| with the centroid layer | **-0.00619 +/- 0.00121** |
| `--no-centroid` (`flows/shear_gauss2_60k.eqx`) | **-0.01316 +/- 0.00533** |
| the layer's contribution | +0.0070 +/- ~0.0055 |

The layer supplies about 53% of the +0.0132 needed to reach zero.
`models/centroid.py`'s own header records "a roughly flat 25-30% UNDERSHOOT"
of the ellipticity response measured on `copies_gauss2_deep` -- which predicts
it should supply 70-75%.  Those agree within the large error on the
no-centroid arm, and the residual has the right sign and order either way.

So the ~5e-3 gauss2 floor is consistent with the centroid transport's KNOWN
spin-2 response shortfall -- a modelling deficiency its own docstring already
flags, not an estimator or training artifact.  Note this is the Gaussian-in-k
ansatz being incomplete, NOT the earlier (falsified) claim about its magnitude
or its clip: on gauss2 the transport is supposed to be exact, and the
undershoot is measured there anyway.

### Where that leaves the 1e-3 target

| gap | size | status |
|-----|------|--------|
| centroid spin-2 undershoot (gauss2 floor) | ~5e-3 | ATTRIBUTED, unfixed -- the gate |
| model error on `bulgedisc_v2` | -0.02 windowed | diagnosed, unfixed |
| estimator O(1/S) after jackknife | ~1.1e-3 at S = 8192 | halved by S = 32768 |
| statistics at 20000 targets | +/-0.0012 | 200k on disk; a 200k 2-arm run projects to ~30 h |

The order stands: close the centroid undershoot first (it gates validation of
everything else and it is the largest term on the clean population), then the
realistic-population density error, then statistics.  Unblocking `truth.py`
would make the first step exactly measurable rather than inferred from a
`--no-centroid` difference with a +/-0.0055 error bar.

### Artifacts

- No code change.  Scratchpad: `g2_nocent.log`, and the truth.py chart check.

## 2026-08-28 (cont.): truth.py FIXED -- the analytic ground truth is usable
again, and it supersedes the Fisher-identity conclusion

### The bug

`truth.to_coords` used slot 2 = `logit(Mc/(POINT_SOURCE_MC Mr))`, deliberately
duplicating `bulk.to_coords` -- its docstring called that an invariant and
`tests/test_truth.py::test_to_coords_matches_bulk` enforced it.  But `log_p0`
evaluates a Gaussian at `sim.GAUSS2_MU`/`GAUSS2_COV`, and those are defined in
SIM's chart (`sim.py` line 377, and `sim._gauss2_m_of_t` whose inverse this
is):

    sim:  t = [log10 Mf, logit(Mr/(POINT_SOURCE Mf)), **Mc/Mr**, M1/Mr, M2/Mr]
    bulk: t = [log10 Mf, logit(Mr/(POINT_SOURCE Mf)), **logit(Mc/(rc Mr))**, ...]

So it compared `mu[2] = 5.974`, an `Mc/Mr` value, against a coordinate worth
about 2.16.  `log_prob` was a Gaussian in the wrong variable and every `Q`,
`R` derived from it was wrong with it.  That is the 2026-08-20 "truth chart is
stale" entry, now located exactly.

**Correcting this session's own earlier diagnosis**: the "8.3 sigma" table a
few entries above compared `sim.GAUSS2_MU` against `bulk.to_coords` means --
two different charts -- so it was not itself evidence of anything.  In sim's
own chart the catalog sits at [3.7413, 1.5089, 5.7046] against
`GAUSS2_MU` [3.7874, 2.0477, 5.9740], i.e. -0.15/-0.75/-0.54 sd, with a
NARROWER spread (0.296/0.499/0.371 against 0.301/0.719/0.497) -- exactly the
documented ~30% rejection truncating and shifting P0.  The catalog and
`GAUSS2_MU` were always consistent; only `to_coords` was wrong.

### Validated externally

| | before | after |
|---|---|---|
| `truth.log_prob(m, 0)` median on its own catalog | -311.97 | **-43.26** |
| Fisher identity `SUM q^2 / SUM(-r)` from exact Q, R | 1.94 (recorded) | **1.0038** |

The density is now in line with the flow's -41.6 on the same population
instead of 270 nats low, and the analytic Fisher identity closes to 0.4% on
4000 noiseless targets.

**That supersedes the standing conclusion that gauss2 fails the Fisher identity
at 1.94 "because the catalog is P0 restricted".**  The restriction is real and
correctly handled -- `log_prob` evaluates P0 at the LATENT moments, so its
normalising constant is g-independent and cancels out of Q and R -- but it was
never the cause.  The chart was.

### Why it survived so long

Every other test in `tests/test_truth.py` is self-consistent within whichever
chart `to_coords` happens to use: the change-of-variables check, the
finite-difference check on `pqr`, even the end-to-end "recovers an injected
shear from exact Q, R".  The only test that reached outside `truth.py` was
`test_to_coords_matches_bulk`, and it was pointing at the wrong reference.  It
is replaced by `test_to_coords_matches_sim`, which round-trips through
`sim._gauss2_m_of_t` and asserts the chart does NOT equal `bulk`'s.  65/65
pass.

### What this unlocks

The exact decomposition is now available on `gauss2_deep`:

    m1 from exact (Q, R)   = the estimator's INTRINSIC bias with a perfect model
    m1 from flow  (Q, R)   = the total
    difference             = model error, isolated

For the noisy targets it needs `INT P(m|g) k(M-m) dm` with `truth.log_prob` in
place of the flow -- the same `log_conv_is` structure, so the machinery exists;
cost is `truth.log_prob`'s 30-step Newton solve per draw, which argues for a
few thousand targets rather than 20000.  That is the measurement that would
turn the -0.0062 gauss2 floor from "consistent with the centroid layer's
25-30% undershoot" into an exact attribution.  NOT YET RUN.

### Artifacts

- `truth.py`: `to_coords` slot 2 is now `Mc/Mr`.
- `tests/test_truth.py`: `test_to_coords_matches_sim` replaces
  `test_to_coords_matches_bulk`.

## 2026-08-28 (cont.): the exact split, and the budget closes -- the estimator
is unbiased to 1.7e-4 and the whole floor is the centroid layer

With `truth.py` fixed, the model/estimator split is now arithmetic rather than
inference.  Done on NOISELESS `gauss2` (`CATALOGS["gauss2"]`, the 1M
catalogs), where `P(M|g)` is a point evaluation: no draws, no importance
sampling, no convolution, and no acceptance-boundary subtlety, since every
catalog target is interior to the rejection region.  200000 targets, paired
(same targets for both models, so the shape noise cancels in the difference).

| | m1 |
|---|---|
| **truth, exact Q and R** | **-0.000171 +/- 0.000107** |
| flow, bulk+shear (`shear_gauss2_60k.eqx`) | -0.000784 +/- 0.000462 |
| difference, paired bootstrap | **-0.00061 +/- 0.00048** |

per-target agreement `corr(Q1) = 0.9844`, `corr(R11) = 0.8542`.

Three things follow, all at or below the 1e-3 target:

1. **The BFD estimator is intrinsically unbiased to 1.7e-4.**  The single
   Newton step, the +/- antisymmetrisation, the ensemble sums and the |Q|/|R|
   guard together carry no 1e-3 bias.  That retires the whole
   "Newton-truncation / finite-g" class of worry.
2. **bulk+shear contribute -6.1e-4 +/- 4.8e-4**, consistent with zero at 1.3
   sigma.  On gauss2 the fitted prior and its shear response are essentially
   exact at the level that matters.
3. So the entire -0.0062 noisy floor is introduced by the two things present
   ONLY in the noisy path: the centroid layer and the IS convolution.

### The budget, closed

| configuration | m1 |
|---|---|
| noiseless (no centroid, no convolution) | -0.0008 |
| noisy, `--no-centroid` (S = 32768) | -0.0132 +/- 0.0053 |
| noisy, with centroid (S = 32768) | -0.0062 +/- 0.0012 |

Centroid marginalisation costs **-0.0124** if left unmodelled; the layer
recovers **+0.0070** of it, i.e. 53%; the residual -0.0062 is its 47%
shortfall.  The "needed +0.0132" is now ANCHORED by the noiseless baseline
rather than inferred, which is what the earlier `--no-centroid` difference
alone could not do.

For orientation, `models/centroid.py`'s header quotes a 25-30% undershoot of
the ELLIPTICITY RESPONSE; that is a different quantity from a 47% shortfall in
m1, but the same order, and both point at the same place.

### What to do next

The target is now singular and quantitative: **the centroid layer captures 53%
of the centroid effect and needs to capture ~100%.**  Everything else in the
chain is measured and clean at the 1e-3 level on this population.  Two ways in:

* The Gaussian-in-k ansatz's spin-2 response is the documented weak point, and
  `centroid.py check` already measures it against the copy catalog -- that is
  the loop to close, and it needs no `bias.py` run to iterate.
* The convolution's own contribution has NOT been separated from the centroid
  layer's within the noisy path.  The clean way is the same exact split done
  here, but with `truth.log_prob` inside `log_conv_is` on the noisy targets --
  affordable (measured ~2600 evals/s, so 2000 targets x 8192 draws is ~1.8 h
  per arm), with one caveat to handle first: `sample_population_gauss2` uses
  rejection at ~30% acceptance, so the true density is P0 RESTRICTED, and
  draws landing outside the accepted region should get zero density where
  `truth.log_prob` returns a finite one.  The acceptance test is explicit
  (`resid < GAUSS2_RESID_TOL`, sigma/rho in range, `|e| < GAUSS2_E_MAX`) so it
  can be masked, but the boundary is g-DEPENDENT (shear moves `|e|`) and a hard
  `jnp.where` has no gradient there, so the masked convolution would miss a
  boundary term.  Worth thinking through before spending the hours.

### Artifacts

- Scratchpad: `exact_split.py` (three modes: `flow`, `truth`, `combine` --
  separate processes because importing `truth` enables x64 and the flow
  checkpoints are float32), `split_{flow,truth}_{plus,minus}.npz`.

## 2026-08-28 (cont.): the centroid layer's 30% deficit is SPIN-2 ONLY, and it
is neither Sigma_u nor the single-Gaussian collapse

With the budget closed onto the centroid layer (previous entry), two candidate
mechanisms were tested against the exact copy-weighted marginalisation on 4000
`copies_gauss2_deep` galaxies.  Both are NEGATIVE, and what survives is sharper.

### Sigma_u is not the problem

The layer uses `Sigma_u = J^-1 Sigma_X J^-T`, the LINEARISATION of
`X^G(u) ~ J u`, and treats `|J(u)|` as constant.  The true density is
`w(u) = d2u |J(u)| N(X(u); 0, Sigma_X)`, which the copy catalog carries
exactly (`u`, `xy`, `da` per copy).  Exact weighted second moment vs the
linearisation:

| trace(Sigma_u) exact / linearised | p10 | p50 | p90 |
|---|---|---|---|
| all | 1.0014 | **1.0087** | 1.0557 |
| faintest 25% | | 1.0471 | |
| brightest 25% | | 1.0017 | |

1-5%, not 25-30%.  A response linear in `Sigma_u` cannot lose 30% here.

### The deficit is confined to spin-2

Ansatz (`_transport(m, sigma_x, +1)`) against the exact copy-weighted mean:

| channel | ansatz / exact |
|---|---|
| Mf fractional shift | 0.976 |
| Mr fractional shift | 1.004 |
| Mc fractional shift | 1.089 |
| **ellipticity response** | **0.702** |

The spin-0 channels are right to ~2% (Mc to 9%).  Only the spin-2 response is
short, and by 30% -- reproducing the module docstring's own "25-30%
undershoot" number, now localised to one channel.

### And it is NOT the single-Gaussian collapse

`gauss2` is a two-Gaussian mixture, so `W(k)I(k)` is a SUM of two Gaussians in
k while the ansatz solves for ONE width matrix `R` from `(Mf, Mr, M1, M2)`.
That predicts the error should grow with the profile's non-Gaussianity.  Using
`nu = Mc Mf / (2 Mr^2 + M1^2 + M2^2)`, which is exactly 1 for a single Gaussian
in k (it is the ansatz's own `mc_ansatz` ratio):

| nu bin | n | exact response | ansatz response | ratio |
|--------|---|----------------|-----------------|-------|
| 0.909-0.929 | 800 | 4.945e-3 | 3.851e-3 | 0.779 |
| 0.929-0.942 | 800 | 5.545e-3 | 4.286e-3 | 0.773 |
| 0.942-0.958 | 800 | 8.039e-3 | 6.303e-3 | 0.784 |
| 0.958-0.984 | 800 | 1.036e-2 | 6.667e-3 | 0.644 |
| 0.984-1.066 | 800 | 1.279e-2 | 8.156e-3 | 0.638 |

The ratio gets WORSE as `nu -> 1`, i.e. as the profile becomes MORE Gaussian --
backwards from the hypothesis.  (Caveat: `nu` correlates with size and flux, so
this is suggestive rather than airtight, but it certainly does not support the
collapse being the mechanism.)

### What that leaves, and the test to run next

`Sigma_u` right, spin-0 algebra right, profile model not the driver, deficit
flat at ~30% in one channel.  **A flat factor in a single channel looks like an
algebra error rather than an approximation** -- candidates being the
`(I+P)^-1 R` versus `R (I+P)^-1` ordering the docstring itself flags as
non-commuting, or a factor in `M1' = Mf'(R~xx - R~yy)` / `M2' = 2 Mf' R~xy`.

The decisive test: **a single elliptical Gaussian is a case where the ansatz is
supposed to be EXACT.**  Evaluate the damping integral
`INT W(k) kernel_a(k) I(k) exp(-1/2 k^T Sigma_u k)` numerically for one and
compare against `_transport`.  Disagreement is a bug in the closed form;
agreement means the ansatz is right and the 30% really is the two-component
profile, which the `nu` trend argues against.  `gauss2` cannot serve as this
test because it is a TWO-Gaussian mixture -- the unit test needs a
single-Gaussian galaxy built for the purpose.  NOT YET RUN.

Note `centroid.py check` gives a fast iteration loop for any fix: it measures
this against the copy catalog directly, with no `bias.py` run in between.

### Artifacts

- Scratchpad: `sigma_u.py`, `ansatz_err.py`.

## 2026-08-28 (cont.): the centroid closed form is EXACT -- no algebra bug.
The 30% is the weight function.

The previous entry's leading suspect was an algebra slip in the spin-2 branch
(a flat 30% in one channel with the spin-0 channels right to 2%).  It is not.

### Worked analytically, then verified numerically

For a Gaussian weight times a Gaussian galaxy, `W(k)I(k) = c exp(-1/2 k^T A k)`
and bfd's kernels (1, k^2, kx^2-ky^2, 2 kx ky, k^4) give, with `B = A^-1`:

    Mf = 2 pi c / sqrt(det A)
    Mr = Mf tr B,   M1 = Mf (B00 - B11),   M2 = 2 Mf B01
    Mc = Mf (3 B00^2 + 3 B11^2 + 2 B00 B11 + 4 B01^2)

so the ansatz's `R = (1/(2 Mf)) [[Mr+M1, M2],[M2, Mr-M1]]` is exactly `B =
A^-1`.  Damping by `exp(-1/2 k^T Sigma_u k)` is exactly `A -> A + Sigma_u`, and

    R~ = (I + R Sigma_u)^-1 R = (I + A^-1 Sigma_u)^-1 A^-1
       = [A (I + A^-1 Sigma_u)]^-1 = (A + Sigma_u)^-1

which is the correct new width matrix -- **the `(I+P)^-1 R` ordering the
docstring flags is right**.  `Mf' = Mf / sqrt(det(I+P))` equals
`sqrt(det A / det(A + Sigma_u))`, and `(2 Mr^2 + M1^2 + M2^2)/Mf` works out to
`Mc` identically, so the differential Mc correction is exact too.

Numerically:

| check | result |
|---|---|
| the moment formulas above vs 2-D quadrature | agree to 3e-15 |
| `_transport` vs exact, 200 random Gaussians | worst rel. err **1.8e-14** |
| ellipticity response ratio, same setup | **1.000000** |

against 0.702 on `copies_gauss2_deep`.  So the closed form reproduces the
marginalisation to machine precision in the spin-2 channel too.

### Consequence

The entire 25-30% undershoot is `W(k) I(k)` not being Gaussian in k.  And note
this is NOT only about the galaxy: the weight is **KBlackmanHarris**, not a
Gaussian, so the product is non-Gaussian even for a single-Gaussian galaxy.
The ansatz is therefore never exact in practice -- the docstring's "exact for a
single Gaussian profile times a Gaussian-shaped weight" is true but the second
half of that condition never holds here.

That also explains the otherwise-backwards `nu` trend in the previous entry:
`nu` measures the GALAXY's radial non-Gaussianity, but the dominant departure
is the WEIGHT's, which `nu` does not track.

### Where a fix would come from

The ansatz fixes a one-parameter radial k-scale from `Mr/Mf` alone and then
treats `Mc` differentially.  The moments already in hand give THREE radial
k-moments of `W(k)I(k)` -- `<1>`, `<k^2>`, `<k^4>` via Mf, Mr, Mc -- so a
two-parameter radial family fixed by both `Mr/Mf` AND `Mc/Mr` would use
information that is currently discarded, with no free constant.  Whether that
captures the KBlackmanHarris window's shape well enough is untested.
`centroid.py check` is the loop.

### Artifacts

- `tests/test_centroid.py::test_transport_is_exact_for_a_gaussian_in_k` --
  pins the exactness claim at 1e-10 so a future edit to the closed form cannot
  quietly break the one case it is supposed to nail.  13/13 pass.
- Scratchpad: `gauss_exact.py`.

## 2026-08-28 (cont.): the 30% is the CIRCULAR WEIGHT times an ELLIPTICAL
galaxy.  Matching Mc will not fix it; carrying W(k) explicitly would.

Two candidate fixes tested by quadrature against `_transport`.  In every test a
circular GAUSSIAN window is the control, since it keeps the product Gaussian and
the ansatz must stay exact there -- it does, to 1.0000, which validates each
setup.

### Matching Mc/Mr (a two-parameter radial law): DEAD, and backwards

Model `W(k)I(k) = f(q)`, `q = k^T A k`, with `f` a Gamma family
`q^(a-1) exp(-q/theta)` -- `a = 1` IS the Gaussian, so the family contains the
current ansatz.  Sweeping `a` and reading off `nu = Mc Mf/(2Mr^2+M1^2+M2^2)`:

| a | nu | I2 I0 / I1^2 | ansatz/exact |
|---|-----|--------------|--------------|
| 1.00 | 1.0000 | 2.000 | **1.0000** |
| 1.05 | 0.9762 | 1.952 | 1.0242 |
| 1.10 | 0.9545 | 1.909 | **1.0472** |
| 1.35 | 0.8704 | 1.741 | 1.1475 |
| 0.80 | 1.1249 | 2.250 | 0.8901 |

The real data has `nu` p10/p50/p90 = 0.923/0.949/1.005, i.e. `a ~ 1.1`, where
the radial correction is **+4.7%** -- it would make the ansatz OVER-predict,
while the data shows it UNDER-predicting by 30%.  Over the whole plausible `nu`
range the effect is at most +/-15%.  **Matching `Mc/Mr` cannot close the gap and
pushes the wrong way.**  Do not build it.

### The circular weight times an elliptical galaxy: CONFIRMED

The ansatz assumes the k-space isophotes are concentric SIMILAR ellipses -- one
ellipticity for all k.  `W(k)` is circular and `I(k)` elliptical, so the
product's ellipticity VARIES with k.  Elliptical Gaussian galaxy times a
circular window, isotropic `Sigma_X` as the sims have (so
`Sigma_u = J^-1 Sigma_X J^-T` is anisotropic and aligned with the galaxy):

| circular window | ansatz/exact | product ellipticity by \|k\| shell |
|---|---|---|
| **Gaussian (control)** | **1.0000** | +0.011 -> +0.238 |
| `exp(-(k/k0)^4)` | **0.877** | +0.011 -> +0.196 |
| `(1-(k/kmax)^2)^3` | **0.862** | +0.011 -> +0.179 |
| top hat | **0.802** | +0.011 -> +0.201 |
| *real data* | *0.702* | |

Right sign, right order.  The shells show the mechanism: the product's
ellipticity rises steeply with `|k|`, and a circular non-Gaussian window
compresses that rise; a single k-independent ellipticity cannot represent the
gradient.

**A caution recorded because it bit here**: an earlier version of this test
imposed isotropic `Sigma_u` and got ratios 1.55-2.79, i.e. the WRONG direction.
The sims have isotropic `Sigma_X`, not `Sigma_u`, and the resulting alignment of
`Sigma_u` with the galaxy dominates the spin-2 response.  Any future test of
this layer must impose `Sigma_X`.

### The fix this points to, and its cost

Stop absorbing `W(k)` into the Gaussian: model only the GALAXY `I(k)` as an
elliptical Gaussian and keep the known `W(k)` explicit in

    M_a(Sigma_u) = INT W(k) kernel_a(k) I_gal(k) exp(-1/2 k^T Sigma_u k) d2k

`W` is KBlackmanHarris and known exactly, so this adds no free constant and is
exact for a Gaussian GALAXY -- a strictly weaker assumption than the present
one, which needs a Gaussian PRODUCT.

Cost, which is the reason not to start it casually: recovering `I_gal`'s
parameters from the measured moments becomes a small per-galaxy nonlinear solve
(3 unknowns from `Mr/Mf`, `M1/Mr`, `M2/Mr`), and the damping integral needs
quadrature -- replacing closed-form 2x2 algebra that currently sits INSIDE the
g-autodiff and runs once per draw in `log_conv_is`.  Worth costing against the
~5e-3 m1 it would buy before committing.

### Artifacts

- Scratchpad: `radial.py`, `angular.py`.

## 2026-08-28 (cont.): prototype -- carrying W(k) explicitly closes 27% of the
centroid gap, not all of it

Tested before implementing, against the copy-weighted ground truth on 300
`copies_gauss2_deep` galaxies.

Setup validated first: the weight rebuilt from bfd's own coefficients
(`[0.349792, 0.487396, 0.150208, 0.012604]`, `kmax = 1.07635 pi / 0.65`) gives
`<k^2>` under W alone = **3.692575**, matching `POINT_SOURCE` to every digit --
so the weight, kmax, units and quadrature grid are all right.

| model | ellipticity response | ratio to exact |
|---|---|---|
| exact (copies) | +8.730e-3 | -- |
| current ansatz (Gaussian PRODUCT) | +6.155e-3 | 0.705 |
| **W explicit, Gaussian GALAXY** | **+6.856e-3** | **0.785** |
| W explicit + `beta` deformation fixed by Mc | +6.856e-3 | 0.785 (solve failed) |

So the proposed fix is real and in the right direction -- it closes **27% of
the gap** (0.295 -> 0.215) -- but it is NOT sufficient by itself.

The `beta` extension (galaxy model `(1 + beta q) exp(-q/2)`, `beta` fixed by
`Mc`, exactly determined by the five measured moments) could not be solved: the
family cannot match `Mc/Mr` to better than ~3% (residual 0.166 against a target
of ~5.75), so it is too rigid.  That is consistent rather than surprising --
the earlier Gamma-family sweep already showed radial deformations move the
response by at most +/-15% and by ~5% at the measured `nu`.

### What is still missing

The residual 21.5% is not radial (two independent tests now say so) and not
co-elliptical structure: a mixture of co-elliptical Gaussians shares one shape
matrix, so all its isophotes are similar ellipses and it falls in the
"elliptical with arbitrary radial law" class the Gamma sweep already bounded at
+/-15%.  The remaining candidates are things that make the k-space isophote
SHAPE vary with k, which the prototype's single elliptical Gaussian galaxy
cannot represent.  A circular PSF times an elliptical galaxy would do exactly
that, and `PSFSIGMA = 0.4` is circular in these sims -- worth checking whether
the moments are of the PSF-convolved or PSF-deconvolved profile before assuming
it, since a circular GAUSSIAN PSF times an elliptical Gaussian is still a single
elliptical Gaussian and would change nothing.

### Cost, if it is pursued

Even the partial fix replaces closed-form 2x2 algebra with a 3-parameter
nonlinear solve per galaxy plus a 2-D quadrature, inside the g-autodiff and
once per draw in `log_conv_is`.  For 27% of a 5e-3 bias -- about 1.4e-3 -- that
is a poor trade as it stands.  It only becomes worth building if the remaining
21.5% is also identified and fixed, taking the whole 5e-3.

### Artifacts

- Scratchpad: `wexplicit.py` (self-checking: it prints `<k^2>` under W and
  should always read 3.692575).

## 2026-08-28 (cont.): SOLVED in principle -- two-Gaussian galaxy + exact W
reproduces the centroid response to 99.05%

Question raised: the weight is a simple known function, so why can we not get
this right with it?  Answer: we can.  The weight was never the obstacle.

| model | ellipticity response | ratio to exact |
|---|---|---|
| exact (copy-weighted truth) | +8.730e-3 | -- |
| current ansatz: single Gaussian PRODUCT | +6.155e-3 | 0.705 |
| single Gaussian GALAXY + exact W | +6.856e-3 | 0.785 |
| **two-Gaussian GALAXY + exact W** | **+8.647e-3** | **0.9905** |

300 `copies_gauss2_deep` galaxies.  The last row closes **96.8%** of the gap;
the residual ~1% is consistent with the `Sigma_u` linearisation measured at
1-5% earlier.

The two-Gaussian row uses `analytic.theta_of_m` to recover the galaxy's own
(flux, sigma, rho, e1, e2) from its moments, then damps BOTH components --
exact, since `exp(-1/2 k^T Sigma_u k)` times a Gaussian is `C_i -> C_i +
Sigma_u` for each -- and contracts with bfd's own weighted kernels.  No new
quadrature beyond what `analytic.moments` already does.

### Correcting the previous entry

It concluded "the residual 21.5% is not radial", citing the Gamma sweep's
+/-15% bound.  **That bound does not apply and the conclusion was wrong.**  The
sweep deformed the radial law of the PRODUCT `W I` with W absorbed; here the
GALAXY's radial law is deformed with W carried separately, and
`W(|k|) f(k^T S k)` is not of the form `g(k^T S k)`, so it is a strictly larger
model class.  The residual WAS radial -- in the galaxy, not in the product.

### Why the current ansatz cannot get there

Collapsing `W I` into ONE Gaussian forces a single k-scale AND a single
k-independent isophote shape.  The real product has neither: W is circular
while the galaxy is elliptical, so the shape varies with k, and a bulge+disc
galaxy has two scales.  Separating W from I fixes the first (0.705 -> 0.785)
and giving the galaxy two scales fixes the second (0.785 -> 0.9905).

### The recipe, and what it costs

Exactly determined and knob-free: a co-elliptical two-Gaussian galaxy has five
parameters (F, sigma, rho, e1, e2) against five measured moments -- which is
precisely the solve `analytic.theta_of_m` already performs.  `BULGE_FRAC = 0.5`
is a basis convention that makes the count work, not a tuned constant.

Cost is the open question, and it is steeper than the earlier estimate: a 5-D
Newton solve PLUS a k-grid contraction per evaluation, and since
`split_centroid` never peels (see the 2026-08-28 config entry) that runs once
per DRAW inside `log_conv_is`.  One mitigation: the centroid layer is
g-independent, so JAX propagates symbolic zeros and the Newton solve is never
differentiated with respect to g.

For the value: closing this buys the ~5e-3 gauss2 floor, i.e. it is the
difference between a method that stops at 6e-3 and one that can reach the 1e-3
target -- the noiseless split already showed the estimator and bulk+shear are
clean at 1.7e-4 and 6e-4.

### Artifacts

- Scratchpad: `ceiling2g.py`.

## 2026-08-28 (cont.): can the centroid fix be analytic, with no Newton solve?

Not with the moments as they stand, for TWO independent reasons -- but there is
a clean route that trades run-time compute for a few more measured moments.

### Obstacle 1: four of the five fourth moments are not measured

To first order in `Sigma_u`, `dM_a = -1/2 Sigma_u,ij <W kernel_a k_i k_j I>`, so
the response is a contraction of `M4_ijkl = <W k_i k_j k_k k_l I>`.  In 2D that
tensor has five independent components, decomposing by spin as

| part | components | measured? |
|------|------------|-----------|
| spin-0 | `<k^4>` | **yes -- Mc** |
| spin-2 | `<(kx^2-ky^2) k^2>`, `<2 kx ky k^2>` | no |
| spin-4 | `<kx^4 - 6 kx^2 ky^2 + ky^4>`, `<4 kx ky (kx^2-ky^2)>` | no |

Everything else the first-order response needs is already in hand -- for kernel
1, `<W k_i k_j I>` IS `(Mr, M1, M2)`.  So the ONLY missing input is the spin-2
and spin-4 parts of `M4`.  The Gaussian ansatz supplies them by Wick from the
second moments, which is exactly where its 30% goes.

### Obstacle 2: first order is not accurate enough anyway

On 300 `copies_gauss2_deep` galaxies, ellipticity-response ratio to the exact
copy-weighted truth:

| method | ratio |
|--------|-------|
| current ansatz | 0.705 |
| 1st order, `M4` all Gaussian | 0.777 |
| 1st order, `M4` spin-0+2 exact | 1.145 |
| 1st order, `M4` FULLY exact | **1.092** |
| non-perturbative, two-Gaussian galaxy + exact W | 0.9905 |

Even given the exact `M4`, first order overshoots by 9%.  Against the 0.9905
row -- same galaxy information, evaluated non-perturbatively -- that 9% is
purely the truncation.  The series converges slowly because the damping
`exp(-1/2 k^T Sigma_u k)` bites hardest on the high-k tail, where the exponent
is not small even though the galaxy-averaged `trace(P)` is 0.005.

### The analytic, solve-free route: change the basis

Expand `W(k) I(k)` in a Gauss-Hermite (shapelet) basis of fixed scale.  Two
properties line up:

* the coefficients are LINEAR functionals of the image, so bfd can measure them
  exactly like any other moment -- the same k-sum with more kernels;
* Gaussian damping maps that basis onto itself ANALYTICALLY and
  non-perturbatively -- a scale change plus a known finite mixing of
  coefficients.

That yields exact damped moments with no Newton solve, no run-time quadrature
and no model assumption; the cost moves out of the estimator and into the
catalog.  UNTESTED -- the mixing coefficients and how many orders are needed for
1% have not been worked out, and it would need `imsims.copies`/bfd to record the
extra moments.

### The cheaper middle ground

By equivariance the response coefficients depend only on `(Mr/Mf, Mc/Mr, |e|)`,
so they can be precomputed once on a 3-D grid and interpolated: no solve at run
time, fully differentiable, cheap.  It keeps a model assumption (whatever
generated the table) but removes the cost objection to the two-Gaussian fix.

### Artifacts

- Scratchpad: `fourth.py`, which also contains a spin decomposition of `M4` and
  a first-order response builder, both reusable.

## 2026-08-28 (cont.): the pragmatic centroid fix -- a binned table OVERFITS;
only a single constant transfers

Tested the "precompute a correction table in `(Mr/Mf, Mc/Mr, |e|)`" idea, whose
whole appeal was avoiding the per-draw Newton solve.  Correction defined as
`kappa = (exact response) / (ansatz response)`, with the copy-weighted
marginalisation as ground truth on both populations, 9000 galaxies each.

The test that matters is TRANSFER -- build `kappa` on `gauss2`, apply to
`bulgedisc_v2`:

| | uncorrected | 3-D table (48 bins) | single constant |
|---|---|---|---|
| gauss2 (self) | 0.7142 | 1.0000 | 1.0000 |
| **bulgedisc_v2 (transfer)** | 0.7021 | **0.9595** | **0.9830** |

**The binned table transfers WORSE than a single number** -- 4.1% residual
against 1.7%.  It is overfitting `gauss2`.  The per-quartile structure shows
why: the two populations do not share the dependence.

| kappa by Mr/Mf quartile | q1 | q2 | q3 | q4 |
|---|---|---|---|---|
| gauss2 | 1.445 | 1.312 | 1.497 | 1.329 |
| bulgedisc_v2 | 1.514 | 1.330 | 1.254 | 1.217 |

gauss2's is non-monotonic (sampling noise); bulgedisc's is cleanly monotonic.
So `(Mr/Mf, Mc/Mr, |e|)` does NOT capture the physics across populations and
the table learned gauss2's noise.  **The tabulated version as proposed is
dead.**

### What survives, and its caveats

A single constant `kappa = 1.4001`, COMPUTED from the copy catalogs rather than
tuned against `m1`, corrects the ensemble ellipticity response to 1.7% across a
population change.  Two things to hold against it:

* bulgedisc's genuine 1.514 -> 1.217 trend across `Mr/Mf` means a constant is
  right on the ENSEMBLE average and wrong per galaxy by 10-20%.  A selection
  that cuts on size would expose exactly that structure -- and the whole reason
  windows matter here is the boundary-flux result, so this is not hypothetical.
* the `m1` gain cannot be read off `kappa`: the layer captures 53% of the m1
  effect while its ellipticity response sits at 70%, so the two measures do not
  map onto one another.  It needs a `bias.py` run to settle.

### Standing recommendation

The only option measured to reach ~1% per galaxy is still the two-Gaussian
galaxy with `W(k)` carried exactly (0.9905), and its cost is a 5-D Newton solve
plus a k-grid contraction per draw.  Everything cheaper that has been tried --
matching `Mc/Mr`, a first-order expansion even with the exact `M4`, and now a
binned correction table -- either fails or transfers badly.  The one untried
route that is both cheap at run time and model-free is the shapelet basis
(previous entry), which moves the cost into the catalog by measuring a few more
linear moments.

### Artifacts

- Scratchpad: `kappa.py`.

## 2026-08-29: it is NOT the weight -- KBH costs 2.7%, the galaxy's second
component costs the rest.  Corrects the 2026-08-28 attribution.

Asked whether `kappa` is just the Gaussian-vs-KBlackmanHarris width difference.
It cannot be a WIDTH mismatch in the second-moment sense -- the ansatz solves
`R` from `(Mf, Mr, M1, M2)`, so its fitted Gaussian matches `<k_i k_j>` of the
true `W I` exactly.  And it turns out not to be the weight's SHAPE either.

Controlled decomposition, single elliptical Gaussian galaxy throughout so the
galaxy is not a confound, Sigma_X isotropic:

| case | ansatz/exact |
|------|--------------|
| 1. Gaussian weight matched to KBH's `<k^2>` (control) | **1.0000** |
| 2. KBH weight -- the WEIGHT's share | **0.9734** |
| 3. KBH weight + TWO-Gaussian galaxy (rho = 0.4) | **0.8344** |

`<k^4>/<k^2>^2` is 1.8042 for KBH against 1.9884 for the matched Gaussian -- a
9% kurtosis difference that costs only **2.7%** of response error.  Adding a
second galaxy component costs another **14%** at rho = 0.4, and more at
realistic rho.

### This corrects the 2026-08-28 attribution

That entry concluded "the 30% is the circular weight times an elliptical
galaxy", on the strength of circular windows giving 0.877 / 0.862 / 0.802.  But
those test windows (`exp(-(k/k0)^4)`, `(1-(k/kmax)^2)^3`, a top hat) are far
more aggressive than KBH actually is.  **The real weight is gentle -- 2.7% --
and the deficit is dominated by the galaxy's multi-scale structure.**  That is
also the consistent reading of the two-Gaussian result (0.9905): it was the
GALAXY component doing the work, not carrying W explicitly.

Practical consequence: a fix that only carries `W` exactly buys ~3%, not 30%.
The 0.785 measured for "single Gaussian galaxy + exact W" on real galaxies is
mostly still galaxy error, not weight error.

### Is it "a width" operationally?

A `Sigma_u` rescale can mimic it -- the response is essentially linear in the
rescale (ansatz/exact = 0.9734, 1.1679, 1.3623, 1.5566 at x1.0, 1.2, 1.4, 1.6),
so x1.03 fixes case 2 and x1.20 fixes case 3.  But the required factor tracks
the GALAXY's internal structure (rho), not the weight.  That is exactly why the
single `kappa` carries a real `Mr/Mf` trend (1.51 -> 1.22) and a depth
dependence rather than being universal, and why it is a per-population
calibration rather than a constant of the method.

### Artifacts

- Scratchpad: `isitwidth.py`.

## 2026-08-29 (cont.): there is NO better approximation from these five moments
-- the response is only half-determined by them, and the rest is Var[.|m]

Asked whether there is a better way to approximate the centroid marginalisation.
Measured answer: not from `(Mf, Mr, M1, M2, Mc)`.

### The five moments determine only about half the response

Per-galaxy `kappa = (exact response)/(ansatz response)` on 4243 well-determined
`copies_gauss2_deep` galaxies (cut to the upper 60% in `|e|` and in the ansatz's
own projection, so the ratio is not dominated by division noise):

| | robust sd |
|---|---|
| overall | **0.227** |
| within fine bins of `(Mr/Mf, Mc/Mr, |e|)` -- 216 bins | **0.162** |

Conditioning on all the moment information removes 0.227 -> 0.162, i.e. explains
**~49% of the variance**.  About **+/-11% per galaxy survives** and cannot be
predicted by ANY function of the five moments.  (Part of the 0.162 is
estimation noise in the per-galaxy ratio, so it is an upper bound -- but the
conclusion does not turn on the exact value.)

### Why -- and why `Mc` in particular cannot rescue it

A controlled two-component scan (single elliptical shape, varying bulge/disc
size ratio `rho`, real KBH weight):

| rho | `nu = Mc Mf/(2Mr^2+M1^2+M2^2)` | ansatz/exact |
|-----|--------------------------------|--------------|
| 1.00 | 0.9759 | 0.9537 |
| 0.70 | 1.0185 | 0.8658 |
| 0.55 | 1.0680 | 0.7949 |
| 0.40 | 1.0939 | 0.7931 |
| 0.30 | 1.0746 | 0.8363 |

Within the scan `nu` does track the error.  But the REAL catalog has
`nu` p10/p50/p90 = 0.923/0.949/1.005 -- BELOW the single-Gaussian value of
0.976 -- which the scan maps to `rho ~ 1` and an error of ~0.95, while the real
error is 0.70.  So the mapping is broken by other degrees of freedom (size
relative to `kmax`, the PSF, the flux ratio).  That is also why the earlier
Gamma-family correction pointed the wrong way.

### This is the layer's own documented floor, now quantified

`models/centroid.py`'s "What it cannot carry" already names it: "the same
`Var[.|m]` floor as the shear layer -- two galaxies with identical moments
marginalise differently, and a deterministic transport can only carry the
conditional mean of that."  The +/-11% IS that floor.  It is missing
information, not a modelling failure to be out-argued.

### The structural route that follows

The layer applies a deterministic TRANSPORT -- it moves the mean.  But the
marginalisation is genuinely stochastic at fixed `m`: a spread, not a shift.
The structurally honest treatment carries it as extra CONVOLUTION COVARIANCE
beside `C_M`:

    P(M|g) = INT p(m|g) N(M - m - Delta(m); C_M + C_cent(m)) dm

with `Delta` the mean shift the current layer already computes and `C_cent` the
induced moment covariance.  Three properties the transport lacks: it represents
the physics correctly; the +/-11% becomes a MODELLED variance instead of an
unmodelled error; and it needs only the SECOND moment of the scatter, a far
weaker requirement than predicting each galaxy's own shift.

UNTESTED, and the honest catch is that `C_cent` plausibly needs the same high-k
information the mean shift does, so it may inherit the same problem.  But it is
the only route left that does not require predicting per-galaxy behaviour the
moments provably do not determine.

### Artifacts

- Scratchpad: the per-galaxy kappa scatter test and the rho scan (inline).

## 2026-08-29 (cont.): the template sum does NOT have this problem, and with
Sigma_X dependence required the route is a copies-trained conditional flow

Asked whether BFD's template sum suffers the same limitation.  It does not, and
the reason identifies what to do instead.

### Why the template sum is immune

Eq. (36) ENUMERATES the copies: `P(M,s|G,g) = J(M) SUM_u d2u L[X^G(u); 0,
Sigma_X] L[M - M^G(u)]`.  Its effective prior over latent moments is the copy
CLOUD -- each galaxy contributes a spread of points, not one -- and it never
inverts moments -> profile, because it holds the actual galaxy.  Our centroid
layer is a deterministic MAP on moments: it can shift a density but not broaden
it, so it structurally cannot reproduce a copy cloud.  That is a consequence of
having replaced templates with a density over moment space, not a bug.

What the map omits, measured on 3000 `copies_gauss2_deep` galaxies:

| moment | copy-cloud sd | vs C_M width | vs population sd |
|--------|---------------|--------------|------------------|
| Mf | 22.1  | 8.1%  | 0.4% |
| Mr | 119.4 | 11.3% | 0.7% |
| M1 | 57.6  | 7.7%  | **5.3%** |
| M2 | 57.6  | 7.7%  | **5.3%** |
| Mc | 930.6 | 11.2% | 0.9% |

Correcting the previous entry's emphasis: the BROADENING dominates, not the
uncertainty on the mean shift.  The unpredictable part of the shift is +/-11% of
~5.7e-4 in `e`, i.e. ~6e-5, while the copy-cloud spread in `e` is ~3.4e-3 --
fifty times larger.

### Sigma_X dependence is clean; the covariance is not predictable

| sigma scale | Sigma_X rel | mean shift | Var[M1] |
|---|---|---|---|
| 0.80 | 0.640 | 1.000 | 1.000 |
| 1.00 | 1.000 | 1.600 | 2.352 |
| 1.25 | 1.562 | 2.580 | 5.405 |

`Delta ~ Sigma_X` and `C_cent ~ Sigma_X^2`, exactly as parity predicts: for the
EVEN moments `dM_a/du = 0` at u = 0 (the integrand is real and even), so the
shift is second order in u and the variance fourth.  So the Sigma_X dependence
is a known power law, not something that has to be learned.

**But `C_cent` is not predictable from the moments.**  Fractional scatter of
`Var[M1]` is 1.964 overall and 1.916 within fine `(Mr/Mf, Mc/Mr, |e|)` bins --
conditioning removes ~5% of the variance, against 49% for the mean shift.  That
kills the `C_cent(m, Sigma_X)` model proposed in the previous entry: it cannot
be tabulated because the moments do not determine it.

### What survives, given Sigma_X dependence is required

Train the flow on COPY moments, conditioned on Sigma_X.  Reweighting is free --
`w(u)` depends on Sigma_X, so one copy grid yields training data at any Sigma_X
in its validity range (roughly a factor 1.4 in sigma_XY), which is what
`centroid.py --sigma-scale` already does.  Sample Sigma_X per batch and the flow
learns `p(m | g, Sigma_X)` directly from exact eq. (36) weights.

Two properties that matter:

* it is a DENSITY over copy moments, not a bijection applied to a galaxy
  density, so it CAN represent the broadening no deterministic layer can;
* the Sigma_X dependence is learned from exact weights rather than derived from
  an ansatz, and the `Sigma_X` / `Sigma_X^2` scalings above give the right
  functional form to learn against.

This means dropping `CentroidMarginalize` and making bulk+shear conditional on
`(g, Sigma_X)`.  A real architecture change, but it removes the transport
approximation, the 30% spin-2 deficit and the +/-11% floor together, because
nothing is inferred from moments any more.  The copies catalog already carries
`moments`, `dm_dg` and `d2m_dg2` PER COPY -- exactly what `bulk.train` and
`shear.train` consume -- and `CopySampler` already draws copies by weight
(though it fixes Sigma_X at construction and would need that lifted).

NOT YET ATTEMPTED.

### Artifacts

- Scratchpad: the copy-cloud spread and Sigma_X-scaling measurements (inline).
