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

## 2026-08-29 (cont.): centring the layer at a nominal Sigma_X cuts its error
10-50x -- this is the route to 1e-3

Proposal tested: bulk+shear learn the COPY-moment density at a nominal
`Sigma_X0` (so the full marginalisation at the operating point, broadening
included, is learned exactly from the copies), and the centroid layer carries
only the DIFFERENTIAL Sigma_X dependence around it.

The k-space damping composes -- damping by `Sigma_u` then `Sigma_u'` equals
damping by `Sigma_u + Sigma_u'` -- so stepping `Sigma_X0 -> Sigma_X` is the same
`_transport` applied to the already-marginalised moments with the difference.
Measured against the copy grid, which gives truth at any Sigma_X by reweighting
(4000 `copies_gauss2_deep` galaxies):

| sigma scale | Sigma_X rel | exact resp | differential | ratio | abs err | vs current |
|---|---|---|---|---|---|---|
| *(current, 0 -> Sigma_X)* | 1.000 | 8.365e-3 | 5.875e-3 | 0.7024 | 2.49e-3 | 1.00 |
| 1.10 | 1.210 | 1.522e-3 | 1.466e-3 | **0.9635** | 5.6e-5 | **0.02** |
| 1.25 | 1.562 | 3.909e-3 | 3.862e-3 | **0.9878** | 4.8e-5 | **0.02** |
| 0.90 | 0.810 | -1.447e-3 | -1.354e-3 | 0.9357 | 9.3e-5 | 0.04 |
| 0.80 | 0.640 | -2.805e-3 | -2.591e-3 | 0.9236 | 2.1e-4 | 0.09 |

Two gains, both real:

* **relative accuracy 92-99% against 70%** -- starting from the
  already-marginalised moments puts the transport at a better effective
  profile, and by the composition property the residual is second order in the
  STEP rather than in the full depth;
* **the ABSOLUTE error, which is what reaches m1, falls to 2-9%** of the current
  scheme's.  The 2.49e-3 response error is what produces the -5.2e-3 m1 floor,
  so scaling by 0.02-0.09 puts the residual at roughly **1e-4 to 5e-4** -- at or
  below the 1e-3 target.

### Why this also dissolves the Var[.|m] floor

The +/-11% floor and the missing copy-cloud broadening are both about a
deterministic map having to MANUFACTURE the marginalisation from moments.  Here
the marginalisation at `Sigma_X0` is in the learned density, so nothing has to
manufacture it; the layer only perturbs an already-correct distribution.

Also correcting an over-statement two entries above: "a deterministic map cannot
broaden a density" is too strong.  Any two smooth densities on the same support
are diffeomorphic, so a LEARNED Sigma_X-conditioned bijection has no structural
barrier.  What `_transport` specifically cannot do is produce the right marginal,
because it is derived as a per-galaxy mean shift and applied pointwise.

### Caveats

* Tested on `copies_gauss2_deep` only; `copies_bulgedisc_v2` not checked.
* The error grows with depth range -- already 9% at `sigma x 0.8` -- so a survey
  spanning much more than +/-25% needs re-checking.
* The copy grid is valid over roughly a factor 1.4 in `sigma_XY`, which bounds
  both the training data and the span the layer can be asked to cover.
* This addresses the gauss2 floor.  `bulgedisc_v2`'s separate -0.02 windowed
  density error is untouched by it.

### What it would take

`bulk.train`/`shear.train` consume `moments`, `dm_dg`, `d2m_dg2`, which the
copies catalog carries PER COPY, and `CopySampler` already draws copies by their
eq. (36) weight -- so training on copies at a fixed `Sigma_X0` needs no new
machinery.  The layer's contract changes from `0 -> Sigma_X` to
`Sigma_X0 -> Sigma_X`, i.e. `_transport` receives the differential and is the
identity at `Sigma_X = Sigma_X0`.

NOT YET IMPLEMENTED.

### Artifacts

- Scratchpad: `differential.py`.

## 2026-08-29 -- The four caveats checked: the differential layer cures gauss2, not bulgedisc

`scratchpad/diff_checks.py`, 4000 galaxies of each copy catalog, sigma scan
[0.6, 1.6].  `ratio` = differential/exact ellipticity response, `abs err` is
what propagates to m1, `edge wt` is the fraction of copy weight in the outer
10% of each galaxy's own u grid.

```
=== gauss2_deep      (nominal edge wt 1.2e-4, ESS median 25)
  current 0 -> Sigma_X0:  ratio 0.7024,  abs err 2.49e-3
  scale   ratio   abs err  vs cur   edge wt   ESS  grid
   0.60  0.9014  5.13e-4   0.206  2.3e-05     9   ok
   0.70  0.9122  3.56e-4   0.143  4.0e-05    12   ok
   0.80  0.9236  2.14e-4   0.086  6.2e-05    16   ok
   0.90  0.9357  9.31e-5   0.037  8.8e-05    20   ok
   1.10  0.9635  5.56e-5   0.022  1.7e-04    30   ok
   1.25  0.9878  4.75e-5   0.019  3.4e-04    39   ok
   1.40  1.0163  1.04e-4   0.042  8.7e-04    49   ok
   1.60  1.0637  6.15e-4   0.247  2.8e-03    63   marginal

=== bulgedisc_v2     (nominal edge wt 4.2e-6, ESS median 25)
  current 0 -> Sigma_X0:  ratio 0.7084,  abs err 1.19e-3
   0.60  0.7526  6.60e-4   0.555  7.0e-08     9   ok
   0.70  0.7487  5.36e-4   0.451  3.0e-07    12   ok
   0.80  0.7454  3.84e-4   0.323  8.0e-07    16   ok
   0.90  0.7432  2.05e-4   0.172  1.7e-06    20   ok
   1.10  0.7440  2.24e-4   0.188  1.5e-05    30   ok
   1.25  0.7509  5.75e-4   0.484  1.1e-04    39   ok
   1.40  0.7644  9.07e-4   0.763  4.9e-04    49   ok
   1.60  0.7945  1.23e-3   1.032  2.0e-03    64   marginal
```

**1. bulgedisc: the relative test fails.**  The ratio is pinned at 0.74-0.79 at
EVERY scale, against gauss2's 0.90-0.99.  Centring does not cure the 30%
deficit; it shrinks the STEP, so the same fractional error rides a smaller
response.  This is the split the controlled decomposition already found: the KBH
weight is worth 2.7%, the galaxy's second component the rest -- and bulgedisc's
misaligned bulge+disc is still present in the copy-mean moments that the
differential transport starts from.  What survives is the absolute error, 5.8x
smaller at +/-10% depth (2.0e-4 vs 1.19e-3), not 50x.

**2. Depth range.**  gauss2 stays within 10% of current over `sigma` in
[0.7, 1.4] (a factor 4 in Sigma_X), and its error is a U with the minimum at
x1.25 rather than at x1.0 -- the Gaussian ansatz changes sign near there.
bulgedisc's useful band is only [0.9, 1.1]; at x1.4 it is 0.76x current and at
x1.6 it buys nothing.

**3. Copy grid: valid, and slightly wider than the factor-1.4 rule.**  Edge
weight stays below 1e-3 through x1.4 on both populations and only reaches
2-3e-3 at x1.6.  The binding constraint at the LOW end is not the grid but the
ESS: median 25 at nominal, 9 at x0.6, so the copy "truth" is itself noisiest
exactly where the gauss2 error grows again -- part of the 0.60/0.70 rise is
that noise, not the scheme.  Usable range `sigma` in [0.7, 1.4].

**4. bulgedisc's -0.02 windowed error is untouched**, now by measurement and not
only by argument.  That bias sits in the flow's `R` (corr 0.319 vs the template
sum, wrong sign, in the two faintest flux quintiles) while `log P` correlates at
0.995.  The differential scheme only moves where the centroid marginalisation
happens; it leaves the bulk density alone, and it now also fails to remove
bulgedisc's own centroid deficit.  Both bulgedisc problems survive.

Net: real, but oversold from one population.  gauss2's -5.2e-3 centroid floor
goes away.  bulgedisc gains ~5x in absolute error over a narrow +/-10% band,
moving its centroid contribution from ~5e-3 toward ~1e-3, and does not touch the
larger -0.02.  Getting bulgedisc to 1e-3 still needs the two-component ansatz.

### Artifacts

- Scratchpad: `diff_checks.py` (supersedes `differential.py`).

## 2026-08-29 -- Two-Gaussian + exact W on bulgedisc: 0.90, not 0.99

`scratchpad/twogauss_bd.py`, 600 galaxies per catalog, ratio to copy truth,
+- is a 400-resample galaxy bootstrap.  Every rung is driven by the five
measured moments alone.

```
=== gauss2_deep (control)            exact response +9.06e-03
     0. current _transport   0.7412 +- 0.0327   abs err 2.35e-3
     1. 1-Gauss + exact W    0.8286 +- 0.0374   abs err 1.55e-3
     2. 2-Gauss + exact W    1.0368 +- 0.0400   abs err 3.34e-4

=== bulgedisc_v2                     exact response +4.27e-03
     0. current _transport   0.7245 +- 0.0164   abs err 1.18e-3
     1. 1-Gauss + exact W    0.7938 +- 0.0189   abs err 8.80e-4
     2. 2-Gauss + exact W    0.8976 +- 0.0122   abs err 4.37e-4
```

Solves: 600/600 on gauss2, 586/600 on bulgedisc (14 galaxies have moments
outside the co-elliptical two-Gaussian's reachable set); residuals ~3e-16.

The gauss2 control is consistent with 1.0 at 0.9 sigma, retiring the earlier
0.9905 (same quantity, 300 galaxies).  But that control was never a real test:
gauss2's galaxy IS two co-elliptical Gaussians at `BULGE_FRAC = 0.5`, so
`analytic.theta_of_m` inverts the generative parameters and rung 2 is close to a
self-consistency check.  bulgedisc_v2 is a Sersic bulge + exponential disc,
MISALIGNED -- wrong profile, not co-elliptical -- and there it stalls at
0.8976 +- 0.0122, i.e. 8.4 sigma from closing.

Budget of bulgedisc's 27.6% deficit: exact W buys 6.9 points, the second
Gaussian 10.4, and **10.2 +- 1.2 points survive**.  The survivor is what a
co-elliptical model structurally cannot hold -- the bulge/disc misalignment,
the same thing that pins Mr/Mf at 3.0-3.2.  Relaxing co-ellipticity is 7
parameters against 5 measured moments: underdetermined without new moments,
which the standing constraint rules out.

**Not yet measured, and the obvious next move:** the two failure modes are
independent.  The differential scheme shrinks the STEP (bulgedisc abs err
2.0e-4 at +/-10% depth); this rung improves the fractional accuracy OF a step
(0.72 -> 0.90).  Composing them -- a 2-Gauss + exact-W transport carrying only
the `Sigma_X0 -> Sigma_X` differential -- projects to ~8e-5.

### Artifacts

- Scratchpad: `twogauss_bd.py`.

## 2026-08-29 -- The 0.898 stall is the analytic class, not the five moments

Scope: fixed isotropic Sigma_X (both catalogs carry exactly one isotropic
`cov_odd` row).  `scratchpad/learned_ceiling.py`, 6000 galaxies/catalog.

Construction is layer-shaped and equivariant: keep the ansatz's spin-2
DIRECTION, learn only a scalar gain `kappa` on the four rotation invariants
(log10 Mf, Mr/Mf, Mc/Mr, |e|).  Rotating the galaxy leaves the invariants fixed
and rotates the direction with it -- the `Spin2CouplingLayer` trick.

```
                                    ens ratio     |err|  per-gal scatter
--- bulgedisc_v2: fit on own train half, tested on held-out half (n=2906)
  0. current _transport                0.7266    0.2734            0.274
     + constant kappa                  1.0000    0.0000            0.377
     + learned kappa(4 invariants)     0.9960    0.0040            0.368
  2. 2-Gauss + exact W                 0.9076    0.0924            0.358
     + constant kappa                  1.0149    0.0149            0.401
     + learned kappa(4 invariants)     1.0049    0.0049            0.381

--- TRANSFER: fit on gauss2, applied to bulgedisc (n=5812)
  0. current _transport                0.7266    0.2734            0.332
     + gauss2 constant kappa           1.0274    0.0274            0.469
     + gauss2 learned kappa            0.9590    0.0410            0.482
  2. 2-Gauss + exact W                 0.9006    0.0994            0.401
     + gauss2 constant kappa           0.8936    0.1064            0.398
     + gauss2 learned kappa            0.8211    0.1789            0.417
```

**The information IS in the five moments.**  Held out, bulgedisc goes 0.7266 ->
0.9960.  The 10-point misalignment residual that no co-elliptical two-Gaussian
could reach is recoverable; the analytic model could not REPRESENT it, not
could not see it.

**One number does it.**  A single constant beats the 4-invariant model
within-population, so nothing elaborate is warranted -- the layer stays
closed-form with one calibrated gain.

**It does not transfer.**  gauss2's constant gives 1.0274 on bulgedisc and
gauss2's learned kappa 0.9590 (worse than the constant); on top of 2-Gauss + W
the transfer actively degrades it to 0.8211.  Same overfitting that killed the
binned table.  The within-population 1.0000 is a random split of ONE population,
so it tests sampling stability, not generalisation.  Consequence: the gain must
be calibrated per population, against that population's own copy catalog.
That trades "zero-parameter analytic" for a fit against ground truth -- same
status as bulk/shear, and copies are constructible for real data since they are
shifts of the noiseless deep templates.  USER'S CALL under the no-tuning rule.

**Unresolved.**  Per-galaxy scatter RISES under correction (0.274 -> 0.368);
unbiased scatter largely cancels in the ensemble but its effect on the density
is untested, and part of it is copy MC noise at median ESS 25.  The ensemble
spin-2 response is a scalar proxy: necessary for m1, not proof that the full
density and its g-derivatives come right.

If it holds, bulgedisc's centroid response error drops 1.18e-3 -> ~2e-5, taking
the centroid layer off the table and leaving the -0.02 windowed density error as
the whole of the bulgedisc problem.

### Artifacts

- Scratchpad: `learned_ceiling.py`.

## 2026-08-29 -- Centroid gain implemented and calibrated; NULL on bulgedisc's bias

Implemented the population-calibrated spin-2 gain from the previous entry.

- `models/centroid.py`: `_transport(m, sigma_x, sign, gain=1.0)` scales the
  ELLIPTICITY displacement `e = (M1 + i M2)/Mr` only.  Mf and Mr come out
  exactly as the ansatz computes them -- the spin-0 channels were already right
  to 2-9%, and rescaling `Sigma_u` instead would have moved them by the full
  gain.  `CentroidMarginalize.gain` is a STATIC field, so it stays off the leaf
  list and every checkpoint written before today still deserialises.
- `bulk.build_flow(..., centroid_gain=1.0)`; `--centroid-gain` on both
  `centroid.py` and `bias.py`.
- `centroid.py calibrate`: fits the gain on half the galaxies, reports the other
  half.  Held out because the binned kappa table overfitted this once and the
  in-sample ratio is 1 by construction.

Calibration on `copies_bulgedisc_v2.fits` (100k galaxies):

```
  fitted on 10000 galaxies, tested on 10000
  uncorrected ratio  train 0.7121   HELD OUT 0.7109
  corrected ratio    train 1.0000   HELD OUT 0.9984
  centroid gain = 1.4043
```

`centroid.py check` at that gain, end to end through the real layer:
ellipticity response ratio **0.708 -> 0.9929**, spin-0 shifts undisturbed
(Mf -3.755e-3 vs catalog -3.886e-3, Mr -7.340e-3 vs -7.213e-3).

Tests: 69/69 pass.  Three new ones pin that gain 1.0 is bit-identical to the old
path, that the gain scales the ellipticity shift while leaving Mf/Mr untouched,
that the round trip survives a gain at anisotropic Sigma_X (residual < 1e-3
against the bare layer's 1e-5), and that rotation equivariance holds.

### The bias: a matched pair, and a NULL

`--pop bulgedisc_deep_v2 --samples 8192 --alpha 0.5 --chunk 4096
--batch-budget 65536 --n-targets 20000 --flow flows/centroid_bulgedisc_v2.eqx`,
run sequentially so they did not contend for the card.

```
                     gain 1.0                gain 1.4043
  m1            -1.26963 +/- 0.01158    -1.27050 +/- 0.01163
  c1            -1.21e-03               -1.22e-03
  q1 (faintest)  -1.0250                 -1.0254
  q2             -1.1512                 -1.1514
  q3             +0.1352                 +0.1315
  q4             +0.0028                 +0.0024
  q5             -0.0010                 -0.0011
```

The shift is -0.0009 against a +/-0.0116 error bar: a null.

**This is the expected outcome, not a failed fix.**  The exact-split budget put
the whole centroid contribution at ~5e-3, three orders below the 1.27 here.
What dominates is the documented R sign inversion at the faint end -- q1 and q2
pinned at -1.03 and -1.15 (zero measured shear response) while q4/q5 are clean
at ~0.001-0.003.  Correcting the centroid's ellipticity response cannot touch a
quintile whose response has collapsed for a different reason.

The centroid layer is now correct on bulgedisc and is OFF the critical path.  It
was worth fixing -- a real 30% error that would surface at the 1e-3 level -- but
the binding constraint is the flow's density: R correlates 0.319 with the
template sum and flips sign in the two faintest flux quintiles, where log P
still correlates at 0.995.

### Artifacts

- Scratchpad: `bias_gain1.log`, `bias_gain140.log`, `run_bias.sh`.

## 2026-08-29 -- The R collapse is Var[score], and the template sum sizes it

`R` for a convolved target splits exactly, with `pi_s = softmax(log w_s + log p_s)`:

    Q = E_pi[d_g log p]    R = E_pi[d2_g log p] + Var_pi[d_g log p]
                               \_____ A ______/   \______ B _______/

A is the curvature (negative for a sane model), B the score variance (positive
definite).  R flipping sign means B beat A.  `scratchpad/rsplit.py` (flow) and
`scratchpad/tsplit.py` (template sum, the reference).

```
=== FLOW, bulgedisc_deep_v2 (n=2000, median draw ESS 267)
 Mf quintile  E[d2logp] A  Var[score] B      R=A+B  Fisher R  Fisher A only
          q1   -5.789e+01     2.042e+02  1.463e+02    -0.051          0.130
          q2   -4.558e+01     8.179e+01  3.620e+01    -0.219          0.174
          q3   -4.075e+01     2.975e+01 -1.100e+01     1.177          0.318
          q4   -2.366e+01     9.399e+00 -1.426e+01     1.195          0.720
          q5   -1.702e+01     1.732e+00 -1.529e+01     1.345          1.208
         ALL   -3.698e+01     6.538e+01  2.839e+01    -0.465          0.357
  top 1% of targets carry 27.7% of sum(B)

=== TEMPLATE SUM, bulgedisc_v2 (exact, 100k templates, no IS)
      flux group            A            B        R=A+B    B/|A|  Fisher R    ESS
    faintest 20%   -1.186e+01   6.930e+00   -4.928e+00    0.584     1.258   7402
      middle 20%   -7.144e+01   6.115e+01   -1.029e+01    0.856     0.955    565
   brightest 20%   -1.069e+06   4.002e+03   -1.065e+06    0.004      93.07      1

=== TEMPLATE SUM, gauss2_deep
    faintest 20%   -1.259e+01   2.468e+00   -1.012e+01    0.196     1.154   5015
      middle 20%   -6.841e+01   3.105e+01   -3.736e+01    0.454     0.961   1629
   brightest 20%   -5.239e+02   2.920e+02   -2.320e+02    0.557     1.178     28
```

**B is the disease, and it is now sized against truth.**  At the faint end the
flow gives `B/|A| = 3.53` and R goes positive; the template sum on the SAME
population gives 0.584 and stays negative at Fisher 1.258.  So the flow's score
variance is **29x too large** (204 vs 6.93) and its curvature **4.9x too large**
(57.9 vs 11.9).  `B/|A|` runs 3.53, 1.79, 0.73, 0.40, 0.10 across flux
quintiles -- monotone, and tracking exactly where m1 collapses.

**Q is roughly right.**  Backing mean q^2 out of the Fisher columns: flow 7.53
(= 0.130 x 57.89) against the template sum's 6.20 (= 1.258 x 4.928), i.e. 21%
high in q^2, ~10% in Q.  First derivative fine, BOTH second-order quantities
badly inflated -- consistent with log P matching templates to 0.01 nats, since
getting the density right constrains neither the roughness of its g-response
nor its curvature.

**Mechanism this points at:** `d_g log p = grad_z log p_bulk . d_g z +
d_g log|det|`.  At faint flux the bulk density is steep, so `grad_z log p_bulk`
is large and any error in the shear displacement is multiplied by it -- the
VARIANCE over the importance cloud more than the mean, which is why B is
inflated 29x while Q is inflated 1.1x.  The template sum cannot do this: its
bandwidth IS C, so its score is bounded by construction.

**Caveats.**  The brightest template row is meaningless (ESS 1 -- 100k templates
give the brightest targets no near neighbours).  The flow's quintiles are of its
own 2000-target sample, the template groups of 40000, so the selections are
close but not identical; immaterial at 29x.

**Not yet done:** split B by where the score comes from -- `grad_z log p_bulk`
(bulk steepness) versus the shear layer's own `d_g log|det|` -- which is what
decides whether the fix belongs in the chart, the bulk, or the shear layer.

### Artifacts

- Scratchpad: `rsplit.py`, `tsplit.py`.

## 2026-08-29 -- B is the BULK's score, not the shear layer's log-det (99.6%)

g enters the chain only through the shear layer, so with `z_c` the
(g-independent) output of raw2standard + centroid,

    d_g log p =  d_g log|det J_shear|        <- "logdet"
               + grad log p_rest . d_g S_g   <- "transport"

and `B = Var[logdet] + Var[transport] + 2 Cov`.  `scratchpad/bsplit.py`:

```
chain reconstruction vs flow.log_prob, on each target's top-weight draw:
  max RELATIVE diff = 4.39e-06

 Mf quintile  Var[logdet]  Var[transp]        2Cov     B total  logdet %  transp %
          q1    1.622e-01    2.034e+02   6.040e-01   2.042e+02       0.1      99.6
          q2    1.531e-01    8.171e+01  -7.354e-02   8.179e+01       0.2      99.9
          q3    8.592e-02    2.958e+01   8.086e-02   2.975e+01       0.3      99.4
          q4    5.769e-02    9.252e+00   8.917e-02   9.399e+00       0.6      98.4
          q5    1.141e-02    1.697e+00   2.361e-02   1.732e+00       0.7      98.0
         ALL    9.406e-02    6.514e+01   1.448e-01   6.538e+01       0.1      99.6
```

The cross term is negligible everywhere -- the two channels are effectively
independent -- and TRANSPORT carries 98-99.6% of B in every quintile.  The B
totals match `rsplit.py` to four digits.

**CORRECTS the emphasis in the "spikiness is the Jacobian" entry.**  The log-det
gradient of 20.5 (vs gauss2's 2.5) was a real measurement of the density's
roughness, but the log-det's own contribution to the SCORE VARIANCE is 0.1%.
The variance arrives through `grad_z log p_bulk` contracted with the shear
displacement, and the displacement is a smooth network output -- so the
roughness is in the BULK density's gradient field.

**Consequence: the repair belongs in the bulk density or its chart, not the
shear layer.**  That is also why every shear-side lever in this repo's history
came back null -- more shear steps, the score term, band-weighting, the
second-order ablation.  They were all acting on 0.1% of the problem.

**Verification note.**  The first run's self-check reported max |diff| = 5.7e7
and was a BAD CHECK, not a bad split: it compared an arbitrary draw's log p at
magnitudes ~1e10 (masked / far-tail draws), where float32 has ~1e3 of absolute
resolution and a jit reordering alone moves it by 1e7.  Checked properly
standalone, the manual walk matches `flow.log_prob` EXACTLY (diff 0.0, value and
gradient, over 40 targets); the in-run check now uses the top-weight draw and a
relative tolerance.

**Next:** separate WHY `grad_z log p_bulk` is bad -- too LARGE in magnitude (the
chart's heavy tails making the density genuinely steep at faint flux, log10 Mf
skew 1.78) or too ROUGH at fixed magnitude (affine MAF steps unable to
represent a smooth steep density).  Compare against the template sum's
`grad_M log P` at the same points, in magnitude and in local variation.

### Artifacts

- Scratchpad: `bsplit.py`.

## 2026-08-29 -- The bulk's score OSCILLATES in the weight-carrying bulk; the ceiling tail is a red herring

`scorefield.py` (flow score vs a template KDE) and `tail.py`/`tail2.py` (where
the blow-ups are, unweighted vs pi-weighted).

```
=== flow score vs template KDE, 22174 in-domain points, 200 faint targets
  flow |score| per noise sigma: median 16.9, p90 3.28e+08
  KDE h   |score| med   flow/KDE med   cos angle   corr per-dim
   0.25          11.9           1.42       0.440          0.000
   0.50          3.09           5.47       0.436          0.000
   1.00          1.21          13.96       0.328          0.000

=== |score| per noise sigma, 409600 draws over 400 faint targets
  UNWEIGHTED   q50/q90/q99/q99.9:  34.1  3.6e6  6.5e13  1.7e15
  pi-WEIGHTED  q50/q90/q99/q99.9:  20.3  323.5  1953    10011
  unweighted fraction |score| > 1e4:  25.6%
  pi-weighted  fraction |score| > 1e4:   0.105%
  top 50% of weight (7213 of 409600 draws):  40.9% of B
  top 90% of weight (27597 draws):           91.5% of B
  pi-weighted median 1-u = 1.049e-01, 1-v = 8.682e-02
```

**RED HERRING, recorded so it is not chased again.**  Unweighted, 25.6% of
in-domain draws carry |score| > 1e4, the p99.9 reaches 1e15, and it correlates
with the chart's point-source ceiling (corr(log|score|, log(1-u)) = -0.513).
Those draws have log p ~ -1e14, so pi = 0 EXACTLY and they cannot enter
B = Var_pi[score]: under the weights they hold 0.105% of the mass, and the
weight-carrying draws sit at 1-u = 0.105, nowhere near the ceiling.  The chart's
logit divergence is real and IRRELEVANT to the bias.

**B is carried by the mainstream of the cloud** -- the draws holding the top 50%
of the weight carry 41% of B, the top 90% carry 91.5%.  Not an outlier artifact.

**What is actually wrong.**  In the weight-carrying region the flow's
g-direction score has sd sqrt(204) = 14.3 against the template sum's
sqrt(6.93) = 2.63 -- **5.4x too much variation**.  |grad log p| runs median 20.3
with p90 324, a 16x spread inside a single noise kernel.  Against a fine
template KDE (h = 0.25) the median MAGNITUDE is only 1.42x high, but the
DIRECTION agrees at cos = 0.44.

So: not too steep, and not a singularity -- the fitted density's GRADIENT FIELD
OSCILLATES through the ordinary bulk of the population, ~5x more than the true
density's, with the direction half wrong.  Magnitude approximately right is
exactly why log P matches templates to 0.01 nats while its derivative does not.

**Points at the bulk's REPRESENTATION, not the chart's tails.**  Affine MAF
couplings fitting a heavy-tailed coordinate reproduce a density accurately in
value while its gradient rings.  Two candidate repairs, both architecture and
neither tuning: a smoother flux coordinate (log10 Mf skew 1.78), or monotonic
rational-quadratic splines in place of the affine steps.

### Artifacts

- Scratchpad: `scorefield.py`, `tail.py`, `tail2.py`.

## 2026-08-29 -- Correction: the score is SMOOTH and ringing, not kinked. NLL is the reason

The user pushed back on the previous entry's spline rationale: silu should make
the affine transforms smooth.  CORRECT, and the previous entry's mechanism
("affine coupling's log-density is piecewise-linear, so its score is a step
function") was WRONG.  `bulk.build_flow` passes `jax.nn.silu` to every
`EquivariantAutoregressiveLayer`, so `mu_i(x_<i)` and `s_i(x_<i)` are
C-infinity; an affine coupling's log-density is QUADRATIC in its own coordinate;
and the only clip in the path (`_safe_logit`, bijections.py:147) bites
out-of-range draws, which carry zero weight.  There is no kink to find.

Measured directly (`scratchpad/ring.py`) -- walk +/-3 noise sigma through each
target's highest-weight draw and count oscillations of the g1-score, against a
template KDE at h = 0.25 (the same object: a PRIOR's g-score, using each
template's own dm/dg):

```
24 lines, +/-3.0 noise sigma, 401 points each
            local extrema  totalvar/range  sd of score  wavelength/sigma
      flow           16.8            1.83    1.214e+13             0.649
KDE h=0.25            5.5            1.90        4.352             2.400
```

`totalvar/range` is essentially IDENTICAL (1.83 vs 1.90) -- same shape
character, not a jagged field.  What differs is FREQUENCY: the flow's score
turns over every 0.649 noise sigma against the true 2.400, ~3.7x faster, i.e.
structure finer than the kernel it is integrated against.  (Non-finite points,
where a line leaves the chart, are dropped; including them the raw extrema count
is 64.4, which is why the first pass reported NaN magnitudes.)

**The real mechanism is the OBJECTIVE, not the architecture.**  NLL constrains
`log p` pointwise and says nothing about `grad log p`.  Many densities fit the
templates equally well in value while differing in gradient, and maximum
likelihood has no preference among them.  That is exactly the measured
signature: log P matches templates to 0.01 nats at corr 0.995, while the score's
direction agrees at cos 0.44 and its variance is 29x too large.  Value pinned,
derivative free.

**Consequence for the two candidate repairs.**  The spline case now rests on a
compositional argument, not a smoothness one: an affine coupling needs many
stacked layers to build a non-Gaussian conditional, the composed Jacobian is a
product across all of them, and high-frequency content compounds; a spline
builds the same conditional in one layer.  UNMEASURED.  The COORDINATE route has
the stronger evidence, and it is the same evidence that argues against blaming
the layer type: the identical stack with the same silu gives gauss2 a log-det
gradient of 2.5 and Fisher 0.96 against bulgedisc's 20.5.  Same network, same
activations -- what changed is the chart's coordinate distributions (flux skew
1.78, spin-2 kurtosis 3.4 vs gauss2's textbook ones).  With the null capacity
sweep and the 1M retrain that ruled out data sparsity, that points at the
coordinate straining the fit.

### Artifacts

- Scratchpad: `ring.py`.

## 2026-08-29 -- Gaussianising the flux axis: a C2 cubic quantile spline, 32 knots

`scratchpad/gaussianise.py`, `fluxshape.py`, `c2spline.py`.

**Flux is the worst axis, but NOT the only one.**  `bulk.to_coords` per slot,
bulgedisc_v2 vs gauss2 (which the same stack fits at Fisher 0.96):

```
              slot   bd skew   bd kurt   g2 skew   g2 kurt
          log10 Mf     1.775     3.807    -0.012     0.013
    logit Mr/PS.Mf     0.184     0.110     0.276    -0.467
   logit Mc/PSc.Mr     0.239     0.261     0.215    -0.605
             M1/Mr     0.035     3.435     0.007    -0.068
             M2/Mr    -0.016     3.328    -0.008    -0.066
```

The two logit slots are FINE.  The spin-2 pair carries excess kurtosis ~3.4,
comparable to flux's 3.8 -- so it wants the same treatment, but RADIAL (a
monotone map of |e| with the direction untouched) or the equivariance breaks.

**Why two parameters cannot do the flux axis.**  A fitted sinh-arcsinh zeroes
skew AND kurtosis exactly and still leaves Anderson-Darling at 144 (current
log10 Mf: 919; rank-transform ceiling: 0.15).  Box-Cox is worse: lambda = -0.79
zeroes skew but drives kurtosis to -1.  The shape is not skewness -- it is a
sharp lower edge plus a long upper tail:

```
  percentile      0.1      1      5     50     95     99   99.9
  log10 Mf      2.750  2.862  2.930  3.248  4.460  5.270  5.983
  matched Gauss 1.832  2.222  2.570  3.409  4.249  4.597  4.987
```

i.e. a detection/flux cut piled against a floor, with a power-law tail above.
No two-moment family reaches a truncation.

**The transform must be C2, not merely monotone.**  `log p(x) = log p_z(T(x)) +
log|T'(x)|`, so the SCORE carries `T''/T'`.  A PCHIP is only C1 and would inject
a T'' jump of ~560 at every knot -- discontinuities in exactly the quantity
`ring.py` showed is currently smooth.  A natural cubic spline is C2 and, though
monotonicity is not guaranteed by construction, it holds here:

```
  knots    kind     skew     kurt      AD    min T'    max T'  max |T''| jump
     16   pchip   0.0624  -0.0252    1.24    0.8577    13.548        428.862
     16   cubic   0.0394  -0.0210    0.59    0.9274    13.012          0.003
     32   cubic   0.0157  -0.0154    0.33    0.9093    14.847          0.010
     64   cubic   0.0076  -0.0471    0.17    0.7926    15.074          0.033
    128   cubic   0.0054  -0.0592    0.15    0.7937    15.653          0.166
```

**Recommendation: 32-knot natural cubic spline on the flux quantiles.**  AD
919 -> 0.33, min T' = 0.909, T'' jump 0.010.

Implementation notes: knots span [2.750, 5.983] while the data reach
[2.477, 6.438] and draws go further, so the extrapolation must continue C2
(match value, slope AND curvature at the ends -- NOT linearly).  The constants
are fitted to the template flux marginal, the same status as
`RawMomentStandardize`'s mean/std (which are TRAINABLE) -- a population
statistic, not a knob tuned against m1, but that is the user's call.

**Still a hypothesis.**  That Gaussianising the marginals reduces the score's
ringing needs a bulk retrain and a re-run of `rsplit.py` to confirm B falls.

### Artifacts

- Scratchpad: `gaussianise.py`, `fluxshape.py`, `c2spline.py`.

## 2026-08-31 -- NEGATIVE: Gaussianising the flux axis makes B 25x WORSE

Implemented the sinh-arcsinh warp of the chart's flux axis (commit 5639186,
identity by default), retrained the whole chain on bulgedisc_v2 at
`--flux-sas 3.409382,0.510387,0.708531,0.487389`, and measured B.

The warp does what it says on the marginal: slot 0's skew 1.775 -> -3e-13 and
excess kurtosis 3.807 -> +5.5e-13, exactly.  It does NOT help the bias.

```
=== bulgedisc OLD chart (log10 Mf)          n=2000, median draw ESS 267
 Mf quintile  E[d2logp] A  Var[score] B      R=A+B  Fisher R  Fisher A only
          q1   -5.789e+01     2.042e+02  1.463e+02    -0.051          0.130
          q2   -4.558e+01     8.179e+01  3.620e+01    -0.219          0.174
          q5   -1.702e+01     1.732e+00 -1.529e+01     1.345          1.208
         ALL   -3.698e+01     6.538e+01  2.839e+01    -0.465          0.357

=== bulgedisc NEW chart (sinh-arcsinh)
          q1   -3.513e+03     5.133e+03  1.621e+03    -0.050          0.023
          q2   -3.572e+03     3.672e+03  9.982e+01    -0.528          0.015
          q5   -2.098e+01     3.038e+00 -1.794e+01     1.256          1.074
         ALL   -2.145e+03     2.475e+03  3.301e+02    -0.153          0.024
```

B goes 204 -> 5133 in the faintest quintile (25x worse), A blows up 61x with
it, and the Fisher ratio is unchanged where it matters (q1 -0.051 -> -0.050)
and worse in q2 (-0.219 -> -0.528).

**Mechanism, and the flaw in the reasoning that motivated this.**  Shear's
`dm/dg` residual on Mf went **0.35% -> 51.58%** in the retrain.  `d_g log p` on
RAW moments is chart-INDEPENDENT for a perfectly fitted density -- the chart is
internal and its log-det is g-independent, so it cancels out of Q and R
entirely.  The only way a chart change can move B is by changing how well the
flow FITS.  So the warp did not add roughness; it made the SHEAR RESPONSE far
harder to represent, because `models/shear.py` expresses the response in chart
coordinates and a chart whose Jacobian varies 22x across the population turns a
smooth response function into a violently varying one.  Better bulk marginal,
much worse response, ~25:1 against.

**This also kills the 32-knot spline**, not just this instance: its T' varies
0.79 -> 15.1, the same order, so it would strain the response the same way.  The
whole "Gaussianise the flux axis" line is CLOSED.

**And it retires the inference that motivated it.**  The gauss2-vs-bulgedisc
contrast (flux skew 1.78 vs 0.01, log-det gradient 20.5 vs 2.5) was CORRELATION,
not causation.  Removing the flux axis's non-Gaussianity entirely does not
reduce the score's ringing.

`bias.py` was not run: the Fisher ratio is the runbook's cheap proxy for exactly
this, and at q1 = -0.050 it is still sign-flipped, so m1 would still be near -1.

**Code status.**  The warp stays in (identity by default, 71/71 tests pass,
byte-identical to the old chart when `flux_sas=None`) so the negative is
reproducible, but it should NOT be enabled.  `flows/bulk_sas.eqx`,
`shear_sas.eqx`, `centroid_sas.eqx` are the warped-chart checkpoints; the
centroid gain recalibrates to 1.4042, i.e. unchanged, as expected since the
centroid layer's deficit is chart-independent.

### Artifacts

- Scratchpad: `train_sas.sh`, `t_bulk.log`, `t_shear.log`, `t_cent.log`,
  `t_calib.log`, `rsplit.py` (now runs both charts).

## 2026-09-01 -- The residual ~1% is the Mr/Mf support edge; the selection
correction is sound but its error bar is missing; three hypotheses died

Branch `feat/centroid-shear-conditioning`. All work offline on the saved
200k run (`dev/pqr_200k.npz`, gauge auto, S=8192, alpha 0.5,
`flows/centroid_bulgedisc_v2.eqx`) plus one controlled training triple.

### 1. The residual is ONE sub-population, and it is flux-independent

`dev/split_residual.py`, slicing the flux-windowed (`Mf >= 1600`) residual on
the clean truth moments:

| Mr/Mf | share | m1 |
|---|---|---|
| < 3.26 | 75% | within +/-0.012 |
| 3.26-3.43 | 15% | -0.050 to -0.067 |
| 3.43-4.12 | 10% | **-0.16 to -0.24** |

The 4x5 flux x size table spans Mf 1600 to 3.1e6 and every flux row repeats the
same profile (-0.24/-0.23/-0.16/-0.16 in the top size bin).  Arithmetic closes:
8.6% of the window x -0.13 = the -0.012 seen.  Ellipticity and
`v = Mc/(POINT_SOURCE_MC Mr)` splits are the same effect through correlated
coordinates.  **This CORRECTS the earlier "residual bias is flux-dependent"
reading: inside the flux window there is no flux dependence, the apparent trend
was the size mix.**

Cutting the edge works.  `dev/size_offline.py` (offline size sweep, one
`selection_terms` call per point; `TERMS=flow|templates`, `FD`, `FLUX_LO` envs):

| Mr/Mf hi | kept | corrected m1, flow route | templates route |
|---|---|---|---|
| none | 54.8% | -0.01723 | -0.01378 |
| 3.60 | 53.8% | -0.01717 | -0.01405 |
| 3.45 | 50.1% | -0.00986 | -0.00196 |
| 3.30 | 43.0% | -0.00098 | -0.00368 |
| 3.20 | 37.5% | -0.00327 | -0.00302 |
| 3.00 | 26.8% | +0.01667 | +0.02557 |

(262144 draws, `fd=0.02`.)  Paired drift vs no size cut, flow route:
+0.0074 +/- 0.0014 at 3.45, +0.0162 +/- 0.0021 at 3.30.

### 2. Why the corrected m1 climbs at a tight ceiling -- and why it is NOT
truncation

`dev/window_leverage.py`.  `D_sel/D` = -3.1/-1.0/+2.0/+5.0/+6.4% at ceilings
none/3.45/3.30/3.20/3.00, and `dm1` per 1% of `R_s11` =
+0.0003/+0.0001/-0.0002/-0.0005/-0.00065 (~-0.13 to -0.21 per UNIT `R_s11`).
Uncorrected m1 at 3.00 is +0.086 and the correction cancels 0.070 of it.

**`R_s`'s h -> 0 limit is DIVERGENT for a size window**, as
`selection_terms`'s own docstring warned (per-draw `d2F/dg2 ~ Mf^2` against
occupancy falling as 1/Mf).  With `fd=None` at 3.00 one template carries
**656%** of `R_s11` and the half-split spread is **354%**.  The jvp value is
not ground truth.

`P_s(g)` IS genuinely non-quadratic at a tight ceiling -- `dev/ps_ladder.py`
(fixed draw set, even/odd split) gives `R_eff = 2 even(g)/g^2` flat to 0.2%
over 16x in g at the flux-only window and moving 15-18% between g=0.02 and
0.04 at 3.2/3.0.  **But that does not explain the climb.**  `dev/profile_ps.py`
solves the estimating equation with the selection term evaluated AT the
operating shear -- `(c(u) I - R) g = Q - N_ns Q_s/(1-P_s)`,
`c = 2 N_ns P_even'(u)/(1-P_s)`, `P_even'` a LOCAL central difference in
`u = |g|^2`, no per-draw `dF` and no Taylor from zero.  `c(0) = N_ns
R_s/(1-P_s)` exactly, so eq. (45)-(46) is its `u -> 0` case.  Over 3 draw seeds
at 2^20 draws:

| ceiling | profile | eq. (45)-(46) | seed sd profile / eq |
|---|---|---|---|
| none | -0.0171 | -0.0170 | 0.0005 / 0.0003 |
| 3.30 | +0.0012 | +0.0014 | 0.0011 / 0.0011 |
| 3.20 | +0.0074 | +0.0080 | 0.0018 / 0.0009 |
| 3.00 | +0.0101 | +0.0099 | 0.0014 / **0.0045** |

**Same mean everywhere; the profile buys VARIANCE (3x at 3.00), not bias.**  A
0.0064 shift seen on one seed was a fluctuation.  The h-scan's monotone ramp
overstates the systematic because its small-h end is noise amplified by 1/h^2.

**The real gap in every number quoted in this repo: `P_s`/`Q_s`/`R_s` are held
FIXED across bootstrap draws, so the correction's own MC error is in nobody's
error bar** -- 0.0005 to 0.0018 (profile), up to 0.0045 (eq. 45-46) at 3.00.

Two implementation traps, both of which produced plausible-looking wrong tables
before a control caught them: (a) `np.polyfit` on `u` spanning [0, 2.5e-3]
against `P_s ~ 0.5` is conditioned ~1e10 and returns a junk slope; (b)
evaluating `P_s` along `+g1` only folds the anisotropy into the "even" curve
and corrupts the derivative by `Q_s/g` ~ 12% of `R_s`.  `dev/profile_ps.py` now
runs the flux-only window FIRST and exits unless `2 P_even'` reproduces the
ladder's -0.2306.

### 3. The flow's selection response is ~12% weak, and it is sharpest at the
flux-only window

`dev/rs_compare.py`, 1M flow, 3 seeds x 2^20 draws, templates' own half-split
as their error:

| ceiling | flow R_s11 | tmpl R_s11 | flow-tmpl | sigma |
|---|---|---|---|---|
| none | -0.2199 +/- 0.0008 | -0.2492 +/- 0.0004 | -0.0293 | **~30** |
| 3.45 | -0.0677 +/- 0.0107 | -0.1312 +/- 0.0028 | +0.0635 | 5.8 |
| 3.20 | +0.2323 +/- 0.0279 | +0.3084 +/- 0.0070 | -0.0761 | 2.6 |
| 3.00 | +0.3459 +/- 0.0071 | +0.2671 +/- 0.0696 | +0.0788 | 1.1 |

**The 3.00 ceiling settles nothing** -- its template reference has a 52%
half-split.  An earlier claim in-session of a "3 sigma gap at 3.00" was an
error (the FLOW's 15% half-split was used as the TEMPLATES'); it is 1.1 sigma.
The flux-only window is where the discrepancy is real and precise: the flow's
selection response is 11.8% weak at ~30 sigma, worth ~+0.0035 in m1 there.

### 4. Density is right; the g-evolution is not; steps and data do not fix it

`dev/shell_compare.py` separates `R_s`'s two ingredients against the templates
(bfd's exact `dm/dg`, the true population).

**Density: flow/template mass ratio 0.98-1.01 in EVERY `Mr/Mf` bin from 2.4 to
3.6, on both the 100k and 1M flows.**  Coverage, tilted training and stratified
rendering are all aimed at the wrong thing.  Sole exception: `[3.60, 3.80)` at
0.841 in BOTH flows, unmoved by 10x data -- structural, and 0.4% of the mass.

Response spread (means are ~0 for both by isotropy; only the spread is
meaningful):

| shell (+/-0.1) | 100k flow / tmpl | 1M flow / tmpl |
|---|---|---|
| 3.00 | 1.158 | 1.080 |
| 3.20 | 1.072 | 1.056 |
| 3.45 | 1.006 | 0.991 |

10x training data halves the excess at 3.00 -- so it is not a coefficient
bound.  **But `R_s` does not follow it:** the 1M flow's `R_s11` at the 3.00
ceiling is 0.3459 +/- 0.0071, unchanged from the 100k flow's ~0.34.  The
`R_s ~ E[(dv/dg)^2]` link predicted ~+16% and delivered 0%.  **Falsified; do
not reuse.**

**Shear steps are exhausted.**  Controlled triple from
`flows/bulk_bulgedisc_v2.eqx`, same seed, `--deriv-weight 1e4`, only `--steps`
varying (`flows/shear_v2_s{20000,60000,150000}.eqx`):

| steps | dm/dg Mf | Mr | M1 | M2 | Mc |
|---|---|---|---|---|---|
| 20k | 0.42% | 2.39% | 0.67% | 0.67% | 4.91% |
| 60k | 0.34% | 2.31% | 0.60% | 0.61% | 4.89% |
| 150k | 0.35% | 2.31% | 0.60% | 0.60% | 4.90% |

3.00-shell spread ratio 1.134 / 1.160 / 1.160 -- flat, marginally WORSE with
more steps.  This qualifies the older "response fit keeps improving past 60k":
that was with `--deriv-weight 0`, where NLL alone fights an O(g^2) spin-0
signal.  With 1e4 it is converged at 20k.

### 5. Free cross-fit Fisher check, and why OPG plateaus

Two saved runs over the SAME galaxies and noise with different `--draw-seed`
give independent `qhat`, so `SUM qhat_a qhat_b` is a cross-fit `SUM q^2` at no
cost (`dev/fisher_cross.py`).  At bright flux `SUM q^2 / SUM -r` =
1.081/1.125/1.233 (q3/q4/q5) with plain/cross 1.003/1.005/1.001 -- the
Fisher-identity violation is real, not MC noise.  Writing
`m1_N = E[QS]/E[-R] - 1` and `m1_F = E[QS]/E[Q^2] - 1`, the numbers say
`E[-R] ~ E[QS] ~ 0.85 E[Q^2]`: the model's score carries ~15-25% of variance
orthogonal to the true score.  **That explains OPG's -0.19 plateau** -- it is
dividing by a contaminated metric, not converging to a residual bias.  Do not
adopt OPG to chase this.

### What to quote

Cut at **Mr/Mf <= 3.30** (flux `Mf >= 1600`): corrected m1 = **+0.0012 +/-
0.0011 (selection MC) +/- 0.0036 (galaxy bootstrap)**, consistent with zero.
Do NOT quote 3.20 or 3.00 without the selection MC error, and do not use the
3.00 ceiling to compare the flow against the templates.

### Hypotheses that died today

- Truncation of eq. (45)-(46) in `g` explains the climb at 3.00 -- WITHDRAWN,
  same mean across seeds.
- A 3 sigma flow-vs-template `R_s` gap at 3.00 -- ERROR, 1.1 sigma once the
  templates' own half-split is used.
- Response spread drives `R_s` -- FALSIFIED by the 1M flow.
- Training the shear layer longer -- NULL, converged at 20k.

### Artifacts

All offline on saved runs unless noted.  `dev/split_residual.py` (residual by
moment coordinate), `dev/size_offline.py` (offline size sweep, both routes),
`dev/window_leverage.py` (leverage + fd/jvp convergence), `dev/ps_ladder.py`
(`P_s(+/-g)` even/odd ladder), `dev/profile_ps.py` (profile-`P_s` estimator,
control-gated), `dev/rs_compare.py` (`R_s` vs templates), `dev/shell_compare.py`
(density vs response split; `CENTROID=0` for shear-stage flows),
`dev/fisher_cross.py` and `dev/fisher_split.py` (cross-fit Fisher).
`dev/rs_converge.py` was written but superseded by `rs_compare.py` and not run.
New checkpoints: `flows/shear_v2_s{20000,60000,150000}.eqx` (shear stage only,
no centroid stage run on them).

Note: launching two JAX GPU jobs concurrently kills them both
(`cuSolver ... gpusolverDnCreate failed`, autotuner "No configs could be
compiled").  Sequence them, and wait on a PID rather than `pgrep -f`.

## 2026-09-02 -- The spin-0 response lead is DEAD, and the `R_s` gap it was
chasing was overstated ~4x by a broken error bar

Branch `feat/centroid-shear-conditioning`.  All offline, minutes per run.

### 1. The response is not what makes `R_s` weak (`dev/rs_swap.py`)

The templates are the true population and carry bfd's exact derivatives, so
replacing ONE column of their `dm_dg` with the flow's own response at the same
moments isolates that coordinate's contribution to `R_s`.  At the flux-only
window (`Mf >= 1600`), `R_s11`:

| swapped | dm/dg only | + d2m/dg2 |
|---|---|---|
| exact | -0.2492 | -0.2492 |
| Mf | -0.2492 | -0.2495 |
| Mr | -0.2492 | -0.2492 |
| Mc | -0.2492 | -0.2492 |
| all five | -0.2492 | **-0.2495** |

**The flow's ENTIRE response error, first and second order, on every
coordinate, is worth 0.1% of `R_s11`.**  In hindsight it could not have been
otherwise: a flux-only window's `F` depends on `Mf` alone, so only the flux
response enters -- the flow's *best* coordinate (0.35%) -- and it enters as
`F'' (dMf/dg)^2 + F' d2Mf/dg2`, i.e. squared.  `Mr`'s 2.3% and `Mc`'s 4.9% do
not touch it at all.  The link the last session proposed is broken; do not
re-run it, and do not reopen `models/shear.py`'s spin-0 parameterisation on
these grounds.

### 2. The templates' half-split error bar is 10x too small

`dev/rs_blocks.py`.  Per-template `d2F/dg1^2` is heavy-tailed, so a half-split
is a terrible variance estimate for it.  On `moments_bulgedisc_v2.fits`
(100k templates), flux-only window:

- half-split (what `rs_compare.py` printed): **0.0004**
- 10 blocks of 10k -> sem of the catalog mean: **0.0039**
- plain sem: 0.0039

Fixed in `dev/rs_compare.py`: the template error is now a 10-block sem.  The
flow's 3-seed sd is just as unreliable an estimate (3 seeds of the 100k flow
give 0.0028 where the old table quoted 0.0008) -- treat any `R_s` sd from <10
samples as a lower bound.

### 3. What the gap actually is

Against the 1M template catalog (the reference with a usable error bar), flux
-only window, 1M draws per seed:

| | `R_s11` | error |
|---|---|---|
| templates, 1M catalog | -0.2348 | 0.0012 (10 blocks) |
| templates, 100k catalog | -0.2492 | 0.0039 |
| flow `centroid_bulgedisc_v2` | -0.2273 | 0.0028 (3 seeds: -0.2277/-0.2243/-0.2298) |
| flow `centroid_bulgedisc_v2_1M` | -0.2203 | (2 seeds: -0.2188/-0.2217) |

**So the flow is 3.2% weak at ~2.4 sigma, not 11.8% at 30 sigma**, worth
~0.0016 in m1 rather than ~0.0035.  The 100k catalog's -0.2492 is itself a
high fluctuation (3-4 sigma against the 1M catalog).  Note the 1M-trained flow
is if anything slightly WORSE than the 100k one, consistent with the last
session's finding that 10x data does not move `R_s`.

### 4. Where the residual 3% lives: the flux threshold, in the density

`dev/rs_bins.py` splits `R_s11 = sum_b w_b f_b` over bins of the draw's own
unsheared `Mf`, with `w_b` the population share and `f_b` the mean per-draw
`(F(+h) + F(-h) - 2F(0))/h^2`.  All of `R_s` comes from `Mf` in [1500, 1900]
-- it is a boundary flux, ~4% of the population.  In 100-wide bins there, flow
/ template:

| Mf bin | w ratio | f ratio |
|---|---|---|
| 1500-1600 | 0.987 | 0.927 |
| 1600-1700 | 0.992 | 1.002 |
| 1700-1800 | 0.984 | 1.066 |
| 1800-1900 | 0.981 | 0.976 |

The per-bin curvature matches (that is item 1 again, seen locally); the flow
carries ~2% too little mass in the band.  **Coarse bins lie here** -- with
400-wide bins the "response ratio" reads 0.98/0.82, which is pure within-bin
density structure, since `f` swings from -0.03 to -4.3 across a single coarse
bin.  Any future flow-vs-template comparison at a window edge must bin finer
than the scale on which `f` varies.

### What this means for the residual m1

The `R_s` route is now worth ~0.0016, inside the errors already quoted on the
recommended `Mr/Mf <= 3.30` number (+0.0012 +/- 0.0011 +/- 0.0036).  It is no
longer a candidate for the ~-0.015 unwindowed residual.  The size-edge
population (top 10% in `Mr/Mf`, m1 -0.16 to -0.24) is still unexplained and is
where the next session should go -- and note that whatever is wrong there is
NOT the shear layer's response fit either.  Run at size ceilings, where the
`Mr` response DOES enter `F`, `dev/rs_swap.py` gives:

| ceiling | exact | swap Mr | swap all |
|---|---|---|---|
| 3.45 | -0.1312 | -0.1246 | -0.1233 |
| 3.30 | +0.1507 | +0.1524 | +0.1508 |

At 3.45 `Mr` does carry the whole swap effect, but the whole effect is 0.0079
against an observed flow-vs-template gap of 0.0635 -- **12%**.  At 3.30 it is
under 1%.  The spin-0 response lead is dead at every window.

### Artifacts

`dev/rs_swap.py` (column-swap of the templates' response), `dev/rs_bins.py`
(density/response split in `Mf` bins; `EDGES`, `TREF` envs), `dev/rs_blocks.py`
(block error bars for `R_s`, templates or a flow).  `dev/rs_compare.py`'s
template error bar fixed.

## 2026-09-02 (evening) -- The size edge splits in two: a real bias at
`Mr/Mf` 3.26-3.59 and a pure noise bin above it

Branch `feat/centroid-shear-conditioning`.  Three 20k-target runs, one flow
(`flows/centroid_bulgedisc_v2.eqx`), gauge auto, alpha 0.5, draw-seed 1001.

### 1. The MC error explodes exactly where m1 does -- but that is a coincidence
for the bins that matter

Two runs over the SAME galaxies and noise with different `--draw-seed` give the
per-target MC error for free (`dev/edge_noise.py`), binned by `Mr/Mf`
percentile at `Mf >= 1600`, `sane_targets` guard on:

| size pct | Mr/Mf | sd(R)/\|R\| | sd(Q)/\|Q\| | m1 |
|---|---|---|---|---|
| 0-25 | 2.40 | 0.01 | 0.01 | -0.003 |
| 25-50 | 2.86 | 0.02 | 0.01 | +0.000 |
| 50-75 | 3.13 | 0.05 | 0.02 | -0.007 |
| 75-90 | 3.34 | 0.15 | 0.05 | -0.064 |
| 90-98 | 3.49 | **0.48** | 0.24 | -0.153 |
| 98-100 | 3.64 | **0.97** | 0.83 | -0.261 |

`R = C/A - (B/A)(B/A)^T`, so `-R` is inflated by `Var_MC(Qhat)`, which the seed
pair measures directly -- a parameter-free prediction (`dev/edge_budget.py`):
`dm1 = -SUM Var_MC(Qhat) / SUM(-R)` = -0.0026 / -0.0334 / -0.2690 in the last
three bins, against -0.064 / -0.153 / -0.261 seen.  It lands on the top bin and
is 4-25x short elsewhere.  **That looked like the answer and it is not.**

### 2. Two controlled runs kill it

`--chunk 512` at fixed `--samples 8192` (16 jackknife chunks instead of 2;
`--chunk 4096` was what every earlier run used, so `_merge_finish`'s delete-one
jackknife had only TWO chunks and 198 targets fell back to the plain
estimator).  And `--samples 32768 --chunk 4096`, same draw-seed so the 8192
draws nest inside -- a paired 4x-draws comparison.

| size pct | Mr/Mf | S=8192 | S=32768 | chunk 512 | S32k - S8192 |
|---|---|---|---|---|---|
| 0-25 | 2.40 | -0.0029 | -0.0022 | -0.0049 | +0.0007 +/- 0.0036 |
| 25-50 | 2.86 | +0.0002 | +0.0009 | +0.0016 | +0.0007 +/- 0.0005 |
| 50-75 | 3.13 | -0.0072 | -0.0046 | -0.0036 | +0.0026 +/- 0.0028 |
| 75-90 | 3.34 | -0.0648 | -0.0646 | -0.0744 | **+0.0002 +/- 0.0101** |
| 90-98 | 3.49 | -0.1520 | -0.1722 | -0.1278 | -0.0202 +/- 0.0378 |
| 98-100 | 3.64 | -0.2633 | -0.0830 | -1.6603 | +0.18 +/- 5.44 |

**75-90 and 90-98 do not move with 4x the draws.**  That band -- `Mr/Mf` 3.26
to 3.59, 23% of the flux-windowed catalog -- is a REAL bias, not finite-S.

**98-100 is a noise bin and nothing else.**  It reads -0.26 / -0.08 / -1.66
across three configurations, its paired error bar is +/- 5.4, its top five
targets carry 19-49% of both sums, and `mean -R` flips sign under two of the
three.  Do not interpret it, do not quote it, and do not let it into a fit.

More chunks is NOT the lever: the fallback count drops 198 -> 16-24 (the
jackknife does engage), but at the edge the delete-one sums are unstable and
the top bin blows up to -1.66.  The jackknife hypothesis is closed.

### 3. What this leaves

The residual at `Mf >= 1600` is a genuine bias concentrated in `Mr/Mf`
3.26-3.59, insensitive to draws, and -- by this morning's `dev/rs_swap.py`
result -- not the shear layer's response fit either (`Mr` response error is
worth 12% of the `R_s` gap at a 3.45 ceiling, under 1% at 3.30).  Density mass
matches the templates to 2% out to 3.6 (2026-09-01 `dev/shell_compare.py`).  So
the three obvious explanations are all separately excluded, and the next test
has to compare the flow's per-target `Q`, `R` against BFD's own template sum
IN THAT BAND -- the one comparison never done at the edge.

### Artifacts

`dev/edge_noise.py` (per-target MC error from a draw-seed pair, by size),
`dev/edge_budget.py` (the `Var_MC(Qhat)` prediction vs the m1 seen, plus how
concentrated each bin's sums are), `dev/fisher_cross.py` extended with `BINBY`,
`FLUX_LO`, `GUARD`, `PCT` envs.  New runs: `dev/pqr_auto_20k_c512.npz`,
`dev/pqr_auto_20k_S32k.npz` and their logs.

Note: `--batch-budget 65536` now OOMs on this GPU (the Hessian wants 11.7 GiB
of 16); use 32768.  It costs throughput -- the S=32768 run took ~100 min.

## 2026-09-02 (late) -- The size-edge bias is NOT the flow: BFD's own template
sum reproduces it, and most of it is the binning's own uncorrected selection

Branch `feat/centroid-shear-conditioning`.  All offline, minutes per run.

### 1. The comparison the last session asked for (`dev/band_tmpl.py`)

BFD's own estimator, on the same targets, with no flow anywhere in it: the
paper's eq. (35)-(36) sum over the 22.7M shifted template COPIES of
`copies_bulgedisc_v2.fits`,

    P(M|g) ~ SUM_c w_c N(M - m_c(g); C_M),   w_c = d2u |J| N(X_c; 0, Sigma_X)

so the centroid marginalisation is carried by ENUMERATION -- exactly where the
flow carries it with a deterministic `CentroidMarginalize` transport.  `Q` and
`R` are the first two `g`-derivatives of `log P` in closed form (whitened
residual `u = A(M - m_c)`, `A = chol(C_M)^-1`):

    d_i s_c = u . Aq_i        d_ij s_c = -Aq_i . Aq_j + u . Ar_ij

then the softmax-weighted first and second cumulants.  Validated against
autodiff through the same sum to 3e-6 relative (`SELFCHECK=1`).  Targets and
bins are `dev/pqr_auto_20k_S32k.npz`'s, `Mf >= 1600`, binned by the zero arm's
`Mr/Mf` percentile -- identical to `dev/edge_noise.py`'s bins.

| size pct | Mr/Mf | kept | ESS | slope Q | slope R | **m1 flow** | **m1 templates** |
|---|---|---|---|---|---|---|---|
| 0-50 | 2.66 | 172 | 3519 | 1.19 | 0.61 | +0.0044 | +0.0021 |
| 50-75 | 3.13 | 448 | 7209 | 1.14 | 0.67 | +0.0000 | +0.0117 |
| 75-90 | 3.34 | 605 | 10680 | 1.18 | 0.79 | **-0.0383** | **-0.0425** |
| 90-98 | 3.49 | 643 | 12598 | 1.10 | 0.92 | **-0.1958** | **-0.2053** |
| 98-100 | 3.64 | 184 | 13101 | 1.08 | 0.97 | -0.4948 | -0.4385 |

**BFD's exact template sum has the same bias, bin for bin.**  Reproduced on a
second 1000-target subsample (75-90: -0.0356 vs -0.0412; 90-98: -0.1958 vs
-0.2053).  Per-target `Q` correlates at 0.88-0.96 with a slope of 1.08-1.19
that is FLAT across bins -- the flow's `Q` runs ~15% high everywhere, not in
the band -- and the `R` slope IMPROVES with size (0.61 -> 0.97), i.e. the flow
and BFD agree BEST exactly where the "bias" is worst.

So the band is not a flow density error, not a flow response error, and not the
`--gauge auto` blend.  Together with 2026-09-02's items 1-3, every flow-side
explanation is now excluded by measurement.

**`kept` is an ESS cut** (`ESSMIN`, default 1000).  100k galaxies do not cover
the bright/compact corner: a target 60 whitened sigma from its nearest copy
gets ESS ~25 out of 12M and `|Q| = 120`, which is the same bright-end failure
[[flow-density-vs-template-sum]] recorded.  Coverage is WORST in the 0-50 bin
(17% kept) and best in the band (60-65%), so the cut does not favour the
conclusion.  Without it the outliers put the template sum's Fisher ratio at
1e2-1e5 and every correlation at zero -- do not read an unguarded template sum.

### 2. Why both are biased: binning by size IS a selection on M

Sorting flux-windowed targets into `Mr/Mf` bins is a cut on the observed
moments, so eq. (40) applies to it: `E[SUM_{i in S} Q_i] = d_g P(s|g) != 0`.
Every per-size-bin `m1` in this repo was computed WITHOUT that correction.
`dev/size_bands.py` applies it band by band on `dev/pqr_200k.npz` (each arm
masked on its OWN observed moments, which is what eq. (40) corrects,
`--window-fd 0.02`, selection terms from the 1M template catalog):

| Mr/Mf band | kept | P_s | R_s11 | uncorrected | corrected |
|---|---|---|---|---|---|
| < 3.005 | 27.1% | 0.2699 | +0.336 | +0.08225 +/- 0.00357 | **+0.0078 +/- 0.0053** |
| 3.005-3.252 | 13.4% | 0.1351 | -0.131 | -0.10522 +/- 0.00183 | **-0.0298 +/- 0.0156** |
| 3.252-3.407 | 7.9% | 0.0786 | -0.235 | -0.29289 +/- 0.00678 | **-0.0081 +/- 0.0311** |
| 3.407-3.530 | 4.2% | 0.0414 | -0.131 | -0.64857 +/- 0.01266 | -0.4338 +/- 0.0483 |
| > 3.530 | 2.3% | 0.0222 | -0.075 | -0.62778 +/- 0.03720 | -0.1525 +/- 0.1339 |
| no size cut | 54.8% | 0.5476 | -0.249 | -0.04644 +/- 0.00240 | -0.01378 +/- 0.00306 |

The last row reproduces the known `Mf >= 1600` number to five digits, so the
machinery is the same one.  **`Mr/Mf` 3.252-3.407 -- the lower half of the
"real bias" band -- is consistent with zero once corrected**, and the
uncorrected number it replaces was -0.293.

The two bands above 3.407 do not clear, but their corrected values are NOT
measured: swapping the 100k template catalog for the 1M one moved them by
-0.067 and +0.086, several times the quoted bootstrap bar, and on the 100k
catalog `selection_terms` fires its boundary-dominated warning in three of the
five bands.  The correction's own MC error is still missing from these bars
(see [[selection-correction-g-truncation]]).  Treat everything above
`Mr/Mf = 3.407` as unmeasured, not as a residual.

### 3. What this costs the previous session's conclusion

"`Mr/Mf` 3.26-3.59 is a REAL bias, unmoved by draws" is now: unmoved by draws,
yes; but reproduced by BFD's exact estimator, and mostly an artifact of binning
without eq. (40).  **A per-bin `m1` is not an estimate of a bias.**  It has no
meaning until the bin's own selection is corrected, and it does not decompose
the total: the total `-0.01378 +/- 0.00306` at `Mf >= 1600` already carries the
flux window's correction.

### 4. Two negatives worth not repeating

* **Restricting the template prior sharply to the band is wrong** (`RESTRICT`
  with a hard cut, first version): it made the healthy bins worse
  (0-50: +0.0021 -> -0.0669) because the targets' bin is a SOFT, noise-smeared
  selection.
* **The soft version (`RESTRICT=1`, reweighting each copy by
  `bias.window_prob(m)`) only half-works**: 90-98 goes -0.2053 -> -0.0961 and
  98-100 -0.4385 -> -0.2709, but 0-50 goes +0.0021 -> -0.0555.  That is the
  right object for a ZERO-arm-binned selection but the wrong one for what
  eq. (40) corrects, and it is not a route to a number -- use
  `dev/size_bands.py` instead.

### Artifacts

`dev/band_tmpl.py` (copy-sum `Q`/`R` per target vs the flow's, by size bin;
envs `PQR`, `PCT`, `NPB`, `ESSMIN`, `SEED`, `RESTRICT`, `SELFCHECK`, `VERBOSE`;
caches the whitened 12M-copy array to the scratchpad, ~2 min to build, then
seconds per bin).  `dev/size_bands.py` (per-BAND m1 with the eq. (40)
correction; `TMPL` selects the template catalog).

## 2026-09-02 (night) -- The selection correction's own MC error, propagated:
negligible for the flux window, and it DOUBLES every size band's bar

Branch `feat/centroid-shear-conditioning`.  `dev/size_bands.py`, offline,
~2 min per table.  75/75 tests pass.

### 1. Method

`P_s`, `Q_s`, `R_s` are sample MEANS over the template catalog.  `ghat`'s
docstring called them "exact analytic derivatives ... not Monte-Carlo
estimates" -- true of each template's own `dF/dg`, `d2F/dg2`, false of their
mean, and no quoted bar has ever carried the difference.  Now it does:

* `dev/size_bands.py:per_template` runs `selection_terms`' own fd stencil
  WITHOUT taking the chunk mean, so any subset's `(P_s, Q_s, R_s)` is one
  `np.mean` away.  One jitted stencil per band, reused.
* 20 equal blocks of the 1M catalog, bootstrapped with replacement (keeping
  the three terms' correlations), re-solving `bias.bias` at each draw with the
  GALAXY sample held fixed.  Different catalogs, so the two errors add in
  quadrature.
* Checked: the block means average back to `selection_terms`' own `R_s11`
  (assert in the loop).
* Do NOT call `selection_terms` once per block -- it builds a fresh
  `eqx.filter_jit` closure per call and the accumulated CUBINs OOM the GPU at
  about the 20th.  That is why `per_template` exists.

### 2. Result -- the headline number is untouched

`dev/pqr_200k.npz`, `Mf >= 1600`, no size cut, 1M template catalog, fd = 0.02:

| | m1 | galaxy bootstrap | selection MC |
|---|---|---|---|
| uncorrected | -0.04644 | 0.00240 | -- |
| **corrected** | **-0.01577** | **0.00305** | **0.00016** |

**The selection MC error contributes 0.05% of the variance** -- `R_s11` is
-0.235 +/- 0.0012 there, 0.5%, because a flux-only window averages the whole
catalog.  So the ~5 sigma residual is real and is NOT the correction's noise.
Task 1 of the last handoff is answered in the negative: the residual survives.

Note the point estimate moved -0.01378 -> -0.01577 purely by swapping the 100k
template catalog for the 1M one.  The 100k catalog's own block error on a size
band is ~0.026, so that is a 100k fluctuation, not a change of method.  **Use
`moments_bulgedisc_v2_1M.fits` for every selection term from now on.**

### 3. Result -- per band, it is the DOMINANT error

Same run, corrected, `+/- galaxy +/- selection`:

| Mr/Mf band | kept | R_s11 | sd(R_s11) | corrected m1 | total |
|---|---|---|---|---|---|
| < 3.005 | 27.1% | +0.336 | 0.026 | +0.00784 +/- 0.00527 +/- 0.00537 | 0.0075 |
| 3.005-3.252 | 13.4% | -0.128 | 0.033 | -0.03114 +/- 0.01559 +/- 0.02129 | 0.0264 |
| 3.252-3.407 | 7.9% | -0.237 | 0.018 | -0.00453 +/- 0.03125 +/- 0.03098 | 0.0440 |
| 3.407-3.530 | 4.2% | -0.129 | 0.008 | **-0.43906** +/- 0.04788 +/- 0.02001 | 0.0519 |
| > 3.530 | 2.3% | -0.076 | 0.002 | -0.12802 +/- 0.13939 +/- 0.02293 | 0.1413 |

A narrow band averages `R_s` over few templates, so its relative error is
15-26% instead of 0.5%, and the selection term is as large as the galaxy
bootstrap or larger.  **Three of the five bands are consistent with zero once
both errors are carried** -- including 3.252-3.407, whose uncorrected -0.293
started this whole line of work.

### 4. The one band that survives: 3.407-3.530

-0.439 +/- 0.052, 8 sigma, on 4.2% of the flux-windowed targets.  Checked
against the finite-difference step, since `one_chunk_fd` carries an O(fd^2)
bias:

| fd | < 3.005 | 3.005-3.252 | 3.252-3.407 | **3.407-3.530** | > 3.530 |
|---|---|---|---|---|---|
| 0.01 | +0.0123 | -0.0612 | +0.0416 | **-0.4344** | -0.1279 |
| 0.02 | +0.0078 | -0.0311 | -0.0045 | **-0.4391** | -0.1280 |
| 0.04 | +0.0031 | -0.0102 | -0.0284 | **-0.4256** | -0.1280 |

The middle bands drift by up to 0.07 across the step -- but every drift is
inside the fd = 0.01 selection bar (0.044, 0.054), so it is that estimate's
noise, not an `fd^2` systematic.  **3.407-3.530 is stable to 0.013 across a
4x change in the step**, so its -0.44 is not a finite-difference artifact
either.  It is the only per-band bias left standing, and by the afternoon's
`dev/band_tmpl.py` result BFD's own template sum reproduces the flow there
(90-98 size percentile, median `Mr/Mf` 3.49: -0.2053 vs -0.1958).

### Artifacts

`dev/size_bands.py` gains `per_template`, `selection_error`, `NBLOCK`,
`TBOOT`; the corrected column now prints both errors and their quadrature sum.
`bias.ghat`'s docstring corrected on the "not Monte-Carlo estimates" claim.

### 5. The nominal window

**TEMPLATE-ROUTE NUMBERS -- superseded by section 7.**  Kept as the
flow-vs-templates comparison it turned out to be.

`2500 < Mf < 50000`, `2.2 < Mr/Mf < 3.2`, on `dev/pqr_200k.npz` with the 1M
template catalog.  45014/44950 targets kept (22.5%), `P_s = 0.2258`,
`R_s11 = +0.152 +/- 0.021`, `Q_s = (-3.7e-4, -8.4e-4)` -- consistent with the
zero isotropy demands, which is why the correction moves `c1`/`c2` by < 1e-4.

| | uncorrected | corrected | +/- galaxy | +/- selection | total |
|---|---|---|---|---|---|
| **m1** | +0.04664 +/- 0.00059 | **+0.00260** | 0.00458 | 0.00602 | **0.00757** |
| c1 | +0.00020 +/- 0.00134 | +0.00029 | 0.00125 | 0.00016 | 0.00126 |
| c2 | +0.00185 +/- 0.00139 | +0.00200 | 0.00129 | 0.00017 | 0.00130 |

**m1 is 0.34 sigma from zero**, and the selection MC error is again the larger
of the two components.  The correction does all the work (+0.047 -> +0.003), so
the finite-difference step was checked:

| fd | R_s11 | corrected m1 |
|---|---|---|
| 0.01 | +0.170 | -0.00234 +/- 0.00455 +/- 0.01030 |
| 0.02 | +0.152 | +0.00260 +/- 0.00458 +/- 0.00602 |
| 0.04 | +0.137 | +0.00666 +/- 0.00459 +/- 0.00322 |

The 0.009 spread is monotone in `fd` and every point is inside the others'
selection bars, so it is noise-consistent but may carry an O(fd^2) component.
Quoted as a systematic: **m1 = +0.003 +/- 0.008 (stat) +/- 0.005 (fd step)**.
One mild concentration warning at `fd = 0.01` (one template at 5% of `R_s11`,
`Mf = 4.98e4`, on the flux ceiling) -- not the 20-100% pathology the size bands
show, and the block bootstrap already prices it in.

The flux-only window's -0.0158 +/- 0.0031 and this +0.0026 +/- 0.0076 are not
in tension: `Mr/Mf < 3.2` excludes the 3.407-3.530 band that carries -0.44.

`dev/size_bands.py` gained `FLUX_HI` for this.

### 6. Is the nominal window's answer sensitive to where its boundaries sit?

**TEMPLATE-ROUTE NUMBERS -- superseded by section 7**, which redoes the whole
scan on the flow.  The method description below is unchanged and correct.

`dev/window_shift.py`: each of the four boundaries scanned over `+/-2%`,
`+/-4%` with the other three held nominal.  Two errors, answering different
questions -- the ABSOLUTE bar (galaxy + selection block bootstrap, in
quadrature) is "what is m1 in this window", the PAIRED bar is the error on the
SHIFT against nominal, from one galaxy resample and one template-block
resample evaluated at all five windows of a panel.  The windows are overlapping
subsets of one catalog and one template catalog, so the absolute bars are
heavily correlated and their quadrature difference is the wrong yardstick for a
drift -- the same distinction `dev/size_offline.py` draws for nested ceilings.

| boundary | value | kept | m1 | +/- abs | drift | +/- paired |
|---|---|---|---|---|---|---|
| Mf floor | 2400 | 23.22% | +0.00409 | 0.00742 | +0.00149 | 0.00379 |
| | 2450 | 22.86% | +0.00609 | 0.00766 | +0.00349 | 0.00383 |
| | **2500** | 22.51% | **+0.00260** | 0.00763 | -- | -- |
| | 2550 | 22.17% | +0.00859 | 0.00775 | +0.00599 | 0.00388 |
| | 2600 | 21.83% | +0.00171 | 0.00785 | -0.00089 | 0.00400 |
| Mf ceiling | 48000 | 22.44% | -0.00108 | 0.00713 | -0.00368 | 0.00669 |
| | 49000 | 22.47% | +0.00599 | 0.00699 | +0.00338 | 0.00750 |
| | 51000 | 22.55% | +0.01541 | 0.00698 | +0.01281 | 0.00654 |
| | 52000 | 22.58% | +0.00197 | 0.00782 | -0.00063 | 0.00797 |
| Mr/Mf floor | 2.112 | 23.12% | +0.00560 | 0.00707 | +0.00300 | 0.00225 |
| | 2.156 | 22.84% | +0.00733 | 0.00748 | +0.00473 | 0.00217 |
| | 2.244 | 22.15% | +0.00241 | 0.00746 | -0.00020 | 0.00243 |
| | 2.288 | 21.76% | +0.00319 | 0.00764 | +0.00059 | 0.00301 |
| Mr/Mf ceiling | 3.072 | 18.09% | +0.00037 | 0.01049 | -0.00223 | 0.00762 |
| | 3.136 | 20.28% | +0.00060 | 0.00773 | -0.00200 | 0.00735 |
| | 3.264 | 24.70% | -0.00355 | 0.00657 | -0.00615 | 0.00582 |
| | 3.328 | 26.73% | -0.00515 | 0.00639 | -0.00775 | 0.00614 |

**Every one of the 20 windows is consistent with m1 = 0**, the whole scan spans
-0.005 to +0.015, and no drift exceeds 2 sigma of its paired error.  The
conclusion does not depend on where these boundaries are placed to the percent.

Two things worth knowing anyway:

* **The `Mr/Mf` CEILING is the only boundary with a monotone trend**: m1 falls
  from +0.0004 at 3.072 to -0.0052 at 3.328, drift -0.0078 +/- 0.0061 at +4%.
  That is the large-size population being let in, and it is the same direction
  as the 3.407-3.530 band's -0.44.  `R_s11` falls 0.222 -> 0.036 across the
  scan and the top template's share of it climbs to 6.9% at 3.328, so pushing
  the ceiling further runs into the boundary-dominated regime.
* **The `Mf` CEILING is the least stable boundary.**  Its `R_s11` wobbles
  0.161/0.141/0.152/0.103/0.151 over `+/-4%` -- a flux ceiling sits on the
  bright template tail -- which is why its paired bars (0.0065-0.0080) are
  twice the flux floor's and the +0.0128 excursion at 51000 (2.0 sigma)
  appears.  Non-monotone, so noise, but it is the boundary to re-measure with
  more templates if a number at the 0.005 level is ever needed.

`plots/window_shift.png` has the four panels.

## 2026-09-02 (night, cont.) -- CORRECTION: sections 1-6 used the TEMPLATE
selection terms.  The flow's are the right ones, and they move the answer

Eq. (40)'s `P(s|g)` is an expectation under the SAME prior the targets' `Q`,
`R` came from.  Taking it from the template catalog instead mixes two priors --
and `bias.py`'s own `--window-terms` help already recorded that the two
disagree (the flow runs `P_s` ~4% high).  `bias.py` still defaults to
`templates`; **`dev/size_bands.py` now defaults to `flow`** (`TERMS=templates`
gets the old route back, and it remains useful as a density diagnostic).

Everything below is `flows/centroid_bulgedisc_v2.eqx`, 2^20 prior draws in
float64, seed 31, `dev/pqr_200k.npz`.

### 7a. The three headline numbers, both routes

| window | | templates | **flow** |
|---|---|---|---|
| `Mf >= 1600`, no size cut | `R_s11` | -0.235 +/- 0.0012 | -0.228 +/- 0.0050 |
| | corrected m1 | -0.01577 +/- 0.00305 +/- 0.00016 | **-0.01688 +/- 0.00305 +/- 0.00063** |
| nominal window | `R_s11` | +0.152 +/- 0.021 | +0.1225 +/- 0.021 |
| | corrected m1 | +0.00260 +/- 0.00458 +/- 0.00602 | **+0.01088 +/- 0.00461 +/- 0.00573** |
| | c1 | +0.00029 +/- 0.00126 | -0.00000 +/- 0.00127 |
| | c2 | +0.00200 +/- 0.00130 | +0.00200 +/- 0.00133 |

The flux-only number barely moves (its `R_s` averages the whole prior, and the
two priors agree there to 3%).  **The nominal window moves by +0.008, about one
sigma**, because the flow's `R_s11` is 19% weaker than the templates' on that
window.  `c1`/`c2` are untouched: `Q_s` is ~7e-4, consistent with the zero
isotropy demands, so the correction is carried entirely by `R_s`.

`fd` scan on the nominal window, flow route:

| fd | R_s11 | corrected m1 | top draw's share of R_s11 |
|---|---|---|---|
| 0.01 | +0.1250 | +0.01017 +/- 0.00461 +/- 0.00979 | 7.7% |
| 0.02 | +0.1225 | +0.01088 +/- 0.00461 +/- 0.00573 | 3.4% |
| 0.04 | +0.1025 | +0.01654 +/- 0.00464 +/- 0.00351 | 1.2% |

Same shape as the template route: a 0.006 spread, monotone, every point inside
the others' selection bars, and the concentration climbing as `fd` shrinks
exactly as `one_chunk_fd`'s docstring says it will.  **Quote the nominal window
as m1 = +0.011 +/- 0.007 (stat) +/- 0.005 (fd step)** -- 1.5 sigma, still
consistent with zero, but no longer the 0.3 sigma the template route gave.

### 7b. The bands, flow route

| Mr/Mf band | kept | R_s11 | corrected m1 (+/- galaxy +/- selection) | templates |
|---|---|---|---|---|
| < 3.005 | 27.1% | +0.316 | +0.0119 +/- 0.0053 +/- 0.0056 | +0.0078 |
| 3.005-3.252 | 13.4% | -0.154 | -0.0148 +/- 0.0159 +/- 0.0215 | -0.0311 |
| 3.252-3.407 | 7.9% | -0.177 | **-0.0981 +/- 0.0282 +/- 0.0250** | -0.0045 |
| 3.407-3.530 | 4.2% | -0.141 | -0.4030 +/- 0.0505 +/- 0.0313 | -0.4391 |
| > 3.530 | 2.3% | -0.072 | -0.1782 +/- 0.1317 +/- 0.0224 | -0.1280 |

**One conclusion from section 3 has to be withdrawn.**  On the template route
`Mr/Mf` 3.252-3.407 sat at -0.005 +/- 0.044 and was called consistent with
zero; on the flow route it is -0.098 +/- 0.038, **2.6 sigma**.  The band's
`R_s11` is -0.177 (flow) against -0.237 (templates), a 25% disagreement, and
that is the whole difference.  Two bands still sit at zero (`< 3.005`,
3.005-3.252); the size edge is now 3.25-3.53 rather than 3.41-3.53.

### 7c. The boundary scan, flow route (`plots/window_shift.png` regenerated)

| boundary | value | kept | m1 | +/- abs | drift | +/- paired |
|---|---|---|---|---|---|---|
| Mf floor | 2400 | 23.22% | +0.01260 | 0.00737 | +0.00172 | 0.00391 |
| | 2450 | 22.86% | +0.01455 | 0.00789 | +0.00367 | 0.00387 |
| | **2500** | 22.51% | **+0.01088** | 0.00781 | -- | -- |
| | 2550 | 22.17% | +0.01631 | 0.00782 | +0.00543 | 0.00389 |
| | 2600 | 21.83% | +0.00844 | 0.00796 | -0.00245 | 0.00404 |
| Mf ceiling | 48000 | 22.44% | +0.01280 | 0.00817 | +0.00192 | 0.00442 |
| | 49000 | 22.47% | +0.00846 | 0.00897 | -0.00242 | 0.00614 |
| | 51000 | 22.55% | +0.01242 | 0.00908 | +0.00154 | 0.00558 |
| | 52000 | 22.58% | +0.01115 | 0.00858 | +0.00026 | 0.00780 |
| Mr/Mf floor | 2.112 | 23.12% | +0.01214 | 0.00753 | +0.00126 | 0.00345 |
| | 2.156 | 22.84% | +0.01260 | 0.00761 | +0.00172 | 0.00363 |
| | 2.244 | 22.15% | +0.01692 | 0.00781 | +0.00603 | 0.00351 |
| | 2.288 | 21.76% | +0.01455 | 0.00774 | +0.00367 | 0.00454 |
| Mr/Mf ceiling | 3.072 | 18.09% | +0.00694 | 0.00920 | -0.00394 | 0.00783 |
| | 3.136 | 20.28% | +0.01986 | 0.01049 | +0.00897 | 0.01025 |
| | 3.264 | 24.70% | +0.01365 | 0.00592 | +0.00276 | 0.00643 |
| | 3.328 | 26.73% | -0.00313 | 0.00657 | **-0.01401** | 0.00592 |

**The stability conclusion survives**: every point sits in +0.007 to +0.020
except the `Mr/Mf` ceiling at +4%, and the whole scan is inside +/-0.02.  Two
changes from the template route:

* **The `Mf` ceiling is now the STEADIEST boundary**, drifts <= 0.0024 against
  the template route's +0.0128 excursion.  The flow's prior sample has no
  bright-tail sparsity to fluctuate on, which is exactly what made the template
  route wobble there.  The earlier "least stable boundary" reading was a
  property of the 1M catalog, not of the window.
* **The `Mr/Mf` ceiling is the one boundary that matters**: -0.0140 +/- 0.0059
  at +4% (2.4 sigma), and the top draw's share of `R_s11` climbs 1.4% -> 16.1%
  across the scan as the ceiling walks into the boundary-dominated regime.
  Consistent with 7b -- pushing the ceiling up admits the 3.25-3.53 band.

### Implementation

`dev/size_bands.py` split into `stencil_moments` (the nine lensed moment sets,
window-INDEPENDENT, computed once) and `per_template` (`window_prob` over them,
per window).  That is what makes a 17-window scan affordable on the flow route:
otherwise every window recompiles the whole bijection stack.  Two traps met on
the way, both now fixed in place:

* `per_template` with a 131072-row batch costs a **6-minute** XLA compile of
  one reduce fusion (9 `window_prob`s x 64 Gauss-Legendre nodes).  16384 is
  seconds.  Do not raise it.
* boolean-indexing the `(9, 1e6, 5)` stencil on the DEVICE costs another
  5-minute compile of a gather fusion.  The stencil is kept on the host; 380 MB
  moves back per window in ~0.04 s.

`selection_terms` is no longer called by `size_bands.py` -- `per_template`'s
means ARE it, checked equal while both were running, and on the flow route
calling it too would double the cost.

## 2026-09-02 (night, cont. 2) -- "The flow's R_s disagrees with the templates'"
was mostly the flow's own draw count.  Raise it, and only ONE window still
disagrees

`dev/rs_agree.py`.  `R_s` is a boundary flux -- only draws within a noise width
of the window edge contribute -- so a narrow window uses a few percent of the
sample and its `R_s` carries a correspondingly large error.  At the flow's old
2^20 draws that error was 17% on the nominal window, the same size as the
apparent gap.  **The asymmetry that matters: the templates' error is set by the
CATALOG (1M galaxies, and there is no more), the flow's by the number of
DRAWS, which is free.**

At 2^23 flow draws against the 1M template catalog, `fd = 0.02`:

| window | kept | `R_s11` flow | `R_s11` templates | diff | sigma | top draw f/t |
|---|---|---|---|---|---|---|
| flux only `Mf>=1600` | 54.8% | -0.2243 +/- 0.0013 | -0.2348 +/- 0.0010 | **+0.0105** | **6.5** | 0.3% / 0.0% |
| nominal | 22.5% | +0.0935 +/- 0.0083 | +0.1531 +/- 0.0194 | -0.0596 | 2.8 | 0.6% / 1.6% |
| band < 3.005 | 27.1% | +0.3673 +/- 0.0101 | +0.3364 +/- 0.0262 | +0.0309 | 1.1 | 0.2% / 0.7% |
| band 3.005-3.252 | 13.4% | -0.1859 +/- 0.0076 | -0.1305 +/- 0.0370 | -0.0554 | 1.5 | 0.2% / 1.9% |
| band 3.252-3.407 | 7.9% | -0.1942 +/- 0.0061 | -0.2350 +/- 0.0236 | +0.0409 | 1.7 | 0.2% / 1.1% |
| band 3.407-3.530 | 4.2% | -0.1382 +/- 0.0031 | -0.1308 +/- 0.0094 | -0.0074 | 0.7 | 0.2% / 1.9% |

**Which side is noisy has flipped.**  On every windowed case the templates' block
sem is now 2 to 6 times the flow's, and the template estimate is far more
concentrated (top draw 1-2% of `R_s11` against the flow's 0.2%).  With enough
draws the FLOW is the better-determined estimator of `R_s`; more flow draws can
no longer improve the comparison.

**Only the flux-only window is a resolved disagreement**: the flow is 4.5% low
at nominal 6.5 sigma.  Discount that: the 100k catalog gives -0.2492 against
the 1M's -0.2348, which is 4.5 sigma of the 100k's implied sem, so the template
block sem looks optimistic by ~1.5-2x.  Call it **4.5% low at >= 3 sigma** --
real, and the only one.  Every windowed case is 0.7-2.8 sigma, i.e. NOT
resolved, and cannot be resolved from this side.

### What this changes

`NDRAW` now defaults to 2^23 in `dev/size_bands.py` (~15 min for the stencil).
On the nominal window that takes the selection MC term on m1 from 0.0057 to
**0.0024**, below the galaxy bootstrap -- and moves the answer:

| nominal window, flow terms | m1 | +/- galaxy | +/- selection |
|---|---|---|---|
| 2^20 draws | +0.01088 | 0.00461 | 0.00573 |
| **2^23 draws** | **+0.01906** | **0.00471** | **0.00240** |

+0.0191 +/- 0.0053 is **3.6 sigma from zero** on the statistical errors alone.
Do NOT read the 2^20 -> 2^23 move as a convergence test: threefry is
counter-based, so the 2^20 sample NESTS inside the 2^23 one and the two
estimates are correlated ([[pqr-streamed-seed-nests]] again).  The block sem at
2^23 is a valid error on its own -- the blocks are disjoint -- so the number
stands; it is the *comparison* between draw counts that is not a test.

The band numbers move the same way (2^23, flow): `< 3.005` +0.0014 +/- 0.0052
+/- 0.0019, 3.005-3.252 +0.0063 +/- 0.0162 +/- 0.0047, 3.252-3.407 -0.0731 +/-
0.0290 +/- 0.0088, 3.407-3.530 -0.4129 +/- 0.0500 +/- 0.0089.

### How to close the remaining 4.5%, and what NOT to do

* **It is a density error on a thin shell, not a response error.**
  `dev/rs_swap.py` (2026-09-02 morning) put the flow's entire response, both
  orders, into the templates and moved flux-window `R_s11` by 0.1%.
  `dev/rs_bins.py` localised it: all of `R_s` comes from `Mf` in [1500, 1900],
  ~4% of the population, where the per-bin curvature MATCHES and the flow
  carries ~2% too little MASS.  So the target is the flow's marginal mass on a
  400-wide flux shell, not the shear layer, not capacity, not more training
  data -- all three are already null.
* **Do NOT Richardson-extrapolate the `fd` dependence to zero.**  It is
  tempting (the nominal window's fd spread, +/-0.005, is now the largest error
  in the budget) and it is wrong: for a size window the exact `d2F/dg2` has no
  usable mean at all (Hill index 0.74, `one_chunk_fd`'s docstring), so `fd -> 0`
  extrapolates toward a quantity that does not exist.  `fd = 0.02` is the
  OPERATING POINT -- `ghat` solves at `|g| ~ 0.02` and eq. (45)-(46)'s quadratic
  stands in for `P_s` across that range, not at an infinitesimal.
* **The principled way to remove that last +/-0.005** is therefore not a better
  derivative but dropping the quadratic: evaluate `P_s` at each arm's actual
  `+/-g` instead of reconstructing it from `Q_s`, `R_s`.  That is an algebra
  change in `ghat`, untried.

### Artifacts

`dev/rs_agree.py` (both stencils, one build each, every window; `NDRAW`,
`TMPL`, `BOOT`).  `dev/size_bands.py`'s `NDRAW` default raised to 2^23.

## 2026-09-02 (night, cont. 3) -- CORRECTION: the nominal window's "+2%" is a
SIZE-CEILING correction failing window-independence, not a bias

The user's reaction to `+0.019` was right and the check is one line: the
unwindowed m1, which needs no selection machinery at all.

    NO WINDOW, no correction:  m1 = +0.02206 +/- 0.00819

That looked like it VALIDATED the nominal window's +0.0191.  It does not --
it is the faint end, and the agreement is a coincidence.  Per `Mf` quintile,
unwindowed and uncorrected:

| Mf quintile | median Mf | m1 | share of `SUM q1` | share of `SUM -R11` |
|---|---|---|---|---|
| q1 | 907 | **+0.5199 +/- 0.2032** | 16.4% | 5.4% |
| q2 | 1207 | +0.0890 +/- 0.0281 | 15.0% | 13.1% |
| q3 | 1776 | -0.0297 +/- 0.0031 | 18.6% | 22.2% |
| q4 | 3281 | -0.0242 +/- 0.0029 | 23.9% | 28.5% |
| q5 | 12309 | -0.0087 +/- 0.0053 | 26.0% | 30.8% |

q1 carries 16% of the numerator against 5% of the denominator, so its +0.52
alone puts +0.028 into the total -- the whole unwindowed number.  That is
[[r-is-an-is-artifact]]'s unconstrained faint end again: **the windowed number
is the measurement and the unwindowed one is not a baseline.**  The nominal
window starts at `Mf > 2500` and never sees q1.

### The ladder (`dev/rs_agree.py`, flow terms, 2^23 draws)

Each row adds ONE boundary to the row above.  Eq. (45)-(46)'s entire content is
that the corrected m1 should not move.

| window | kept | corrected m1 | +/- galaxy | +/- selection |
|---|---|---|---|---|
| `Mf >= 1600` | 54.8% | -0.0174 | 0.0030 | 0.0002 |
| `Mf >= 2500` | 37.6% | -0.0191 | 0.0037 | 0.0003 |
| `Mf` 2500-50000 | 34.5% | -0.0202 | 0.0041 | 0.0014 |
| + size > 2.2 | 31.9% | -0.0159 | 0.0033 | 0.0017 |
| **nominal (+ size < 3.2)** | 22.5% | **+0.0191** | 0.0047 | 0.0024 |
| size 2.2-3.2 only, `Mf >= 1600` | 33.6% | +0.0187 | 0.0040 | 0.0024 |

**The flux floor, the flux ceiling and the size FLOOR all correct correctly** --
four nested windows agreeing to 0.004 over a 23-point spread in kept fraction.
**Adding the size CEILING jumps it by +0.035**, and the last row proves it is
the ceiling alone: a different flux window, the same jump.  The template route
does the same thing (+0.026, same sign, same boundary), so it is not a flow-vs-
templates issue.

`dev/window_shift.py`'s `Mr/Mf`-ceiling panel and the nominal window's `fd`
sensitivity were both symptoms of this, read at the time as noise.

### So what is the number

**Quote the flux-selected value: m1 ~ -0.018 +/- 0.004**, stable across four
nested windows.  Every "nominal window" m1 in sections 5-7 above -- +0.0026
(templates), +0.0109 (flow, 2^20), +0.0191 (flow, 2^23) -- is that number plus
a broken size-ceiling correction, and none of them is a bias measurement.

### Why the ceiling specifically

Both reasons were already on record and neither was connected to this:

* **`R_s` for a size ceiling has no `h -> 0` limit** (Hill index 0.74,
  [[selection-correction-g-truncation]]).  So `fd = 0.02` is not approximating a
  derivative -- eq. (45)-(46)'s QUADRATIC is standing in for `P_s(g)` across the
  whole `|g| = 0.02` the estimator solves at, and for this boundary that
  stand-in is wrong by 0.035 in m1.  `ghat` is one Newton step, i.e. a
  second-order Taylor expansion of `N_ns log(1 - P_s(g))`; nothing checks that
  the expansion holds out to the solution.
* **The targets the ceiling removes are the `Mr/Mf` 3.25-3.53 population**
  carrying m1 = -0.1 to -0.4 (section 7b).  The correction has to reinstate
  their contribution FROM THE PRIOR, at the support edge where the density is
  least trustworthy.

[[selection-terms-are-boundary-flux]]'s "size cuts WORK via `--window-fd 0.02`"
needs qualifying: they run without NaN, but they do not preserve
window-independence.

## 2026-09-02 (night, cont. 4) -- NEGATIVE: dropping the quadratic does NOT fix
the size ceiling.  The exact solve reproduces the estimator, jump and all

`dev/exact_ps.py`.  `ghat` takes one Newton step, i.e. expands
`N_ns log(1 - P_s(g))` to second order about `g = 0`.  `P_s(g)` is cheap to
evaluate exactly at any `g`, so keep the targets' quadratic (all `--save-pqr`
stores) and solve the untruncated scalar equation per arm:

    SUM_i q_i1 + g1 SUM_i r_i11 - N_ns P_s'(g1) / (1 - P_s(g1)) = 0

`P_s` on a 13-point grid over `g1` in [-0.03, 0.03] (`g2 = 0`; the window is
spin-0), cubic-splined, root-found with `brentq`.  The prior push-forward at
each grid `g1` is window-independent, so one grid serves every window.  4.2M
flow draws.  Built-in check: linearising the same solve about zero reproduces
`bias.ghat`'s 2x2 answer to 1e-4 in every row.

| window | production (fd 0.02 Newton) | **EXACT solve** |
|---|---|---|
| `Mf >= 1600` | -0.0174 | -0.0175 +/- 0.0030 |
| `Mf >= 2500` | -0.0191 | -0.0195 +/- 0.0038 |
| `Mf` 2500-50000 | -0.0202 | -0.0193 +/- 0.0042 |
| + size > 2.2 | -0.0159 | -0.0140 +/- 0.0033 |
| **nominal (+ size < 3.2)** | +0.0191 | **+0.0228 +/- 0.0047** |
| size 2.2-3.2 only | +0.0187 | +0.0141 +/- 0.0040 |

**The exact solve agrees with the production estimator to <= 0.005 in every
window and the ceiling jump survives at +0.037.**  Two things follow:

* **The quadratic is not the problem, and `fd = 0.02` is vindicated.**  The
  operating-point argument in `one_chunk_fd`'s docstring was made on principle;
  this measures it.  Note the local curvature is NOT the right input: the
  spline's `R_s(0)` differs from the `fd = 0.02` secant by 10% (flux floor) to
  a factor 2 (`Mf` 2500-50000), and feeding the spline's `R_s(0)` to `ghat`
  gives -0.0203/-0.0290/-0.0336/-0.0281/+0.0025/+0.0106 -- wrong by up to 0.017
  against the exact solve.  `P_s` is genuinely non-quadratic near zero; the
  0.02 secant is what the estimator needs and the exact solve confirms it.
* **So the ceiling failure is not an estimator truncation.**

### What it is instead

The template route jumps too (+0.026, section cont. 3), and the templates ARE
the true population -- so it is not the flow's prior either.  What is left is
the assumption eq. (45)-(46) makes about the objects it reinstates: that their
contribution is described by the same likelihood as the ones that were kept.
**The population a `Mr/Mf < 3.2` ceiling removes is exactly the 3.25-3.53 band
carrying m1 = -0.1 to -0.4** (section 7b), where the per-target `Q`, `R` are
demonstrably not right -- and where BFD's own template sum reproduces the flow
target for target (`dev/band_tmpl.py`).

**A size ceiling cannot be used to cut the biased population away: the
correction puts it back.**  That is a property of the estimator + the
population, not of any implementation here, and it is why the flux ladder is
flat and the size ceiling is not.

### Consequence for the quoted number

Unchanged: **m1 ~ -0.018 +/- 0.004** on the flux-selected windows, which are
window-independent across four nested cuts and now also reproduced by an
untruncated solve.  No number from a window with an `Mr/Mf` ceiling should be
quoted until the size-edge population's own bias is understood.

### Artifacts

`dev/exact_ps.py` (grid `P_s`, spline, untruncated per-arm solve; `GRID`,
`NDRAW`, `BOOT`).

## 2026-09-02 (night, cont. 5) -- The residual is NOT the flow.  BFD's own
template sum reproduces it, paired, to +0.0040 +/- 0.0041

Two new offline tools, `dev/edge_share.py` and `dev/prior_n.py`, both on
`dev/pqr_200k.npz` and the existing 22.7M-copy catalog.  69/69 tests in
`tests/` pass (`tests/test_truth.py` has a pre-existing collection error,
`module 'bfd' has no attribute 'KB...'`, untouched here).

### 1. A per-band m1 is not a bias, and this time there is a proof

Three sessions have chased the `Mr/Mf` 3.25-3.53 band's -0.1 to -0.4.  The
estimating equation is `SUM_i q_i + g SUM_i R_i = 0`, and `E[q] = 0` holds only
when the average runs over the WHOLE population -- `P(M|g)` is the marginal
over the prior, so no individual target's `q` has zero mean and no
subpopulation's does either.  A band's m1 is therefore nonzero for a perfectly
correct estimator.  Measured, at `Mf` in [1600, 2941) where the copy sum is
well covered (median ESS 16327), 400 targets per cell:

| cell (`Mr/Mf`) | flow m1 | TEMPLATE SUM m1 | flow `SUM q^2/SUM -r` | templates |
|---|---|---|---|---|
| 3.407-3.530 | -0.1767 +/- 0.0337 | **-0.1782 +/- 0.0138** | 0.537 | 0.431 |
| 3.252-3.407 | -0.0429 +/- 0.0220 | -0.0394 +/- 0.0137 | 0.865 | 0.691 |

BFD's eq. (35)-(36) sum -- no flow, no importance sampling, no gauge -- gives
the band the same -0.18, and violates the Fisher identity there HARDER than the
flow does.  Both statistics are properties of conditioning, not of a defect.
**Do not open the size edge again on the strength of a per-band number.**

### 2. What replaces it: contributions, not per-band m1 (`dev/edge_share.py`)

`ghat = R^-1 SUM q_i` is linear in the targets at fixed `R`, so with the window
held FIXED each target has an exact additive share `b_i` of the one estimate,
`SUM_i b_i = m1`, no selection term anywhere.  Binned on the `g = 0` arm's
moments (independent of both arms' noise and of `g`, so the binning induces no
correlation), `Mf >= 1600`, m1 = -0.02137:

| `Mr/Mf` band | share of D | contribution | +/- |
|---|---|---|---|
| < 3.005 | 60.4% | -0.00305 | 0.00218 |
| 3.005-3.252 | 22.3% | -0.00058 | 0.00076 |
| 3.252-3.407 | 10.9% | -0.00554 | 0.00040 |
| 3.407-3.530 | 4.6% | -0.00718 | 0.00052 |
| > 3.530 | 1.8% | -0.00502 | 0.00085 |

83% of the total sits above 3.25 on 17% of the weight -- but by section 1 that
is where the contributions ARE, not where an error is.

**Binning on the arm's OWN moments is what produced the old picture**: the same
table on the selected arm gives the `< 3.005` band +0.0659 (implied m1 +0.109)
against -0.0031 (-0.005).  That flip is the whole of the "size edge" as it was
originally seen.

### 3. `truth` in a saved run is NOT the truth

`bias.py` sets `truth = m["zero"]`, correctly commented "for binning only".  On
an image-noise catalog that is a THIRD noisy realization -- `targets_deep_g0`'s
`Mr/Mf` reaches 4.845 against the population's true ceiling 3.683, and `|e|`
reaches 6.17.  `save_pqr`'s docstring calls it "the clean unsheared moments";
that is wrong for every image-noise run and it cost an hour here (a census of
training coverage per (flux, size) cell, comparing NOISELESS training moments
against NOISY target ones, is meaningless).  Its value is that it is
independent of both arms, which is what section 2 needs.

### 4. The headline: the flow is exonerated to +/-0.004

`dev/prior_n.py`, `CELL=all`, 40000 random targets of the `Mf >= 1600` window,
flow and template sum on IDENTICAL targets so the paired difference beats
either error:

    14764 targets with ESS > 1000
    flow                 m1 = -0.03212
    templates - flow          +0.00400 +/- 0.00406

**BFD's own estimator, given the same prior population and the catalogs' own
exact `dm_dg`/`d2m_dg2`, reproduces the flow's bias.**  Every remedy aimed at
the flow -- retraining, capacity, the chart, the gauge, the centroid layer --
is competing for at most 0.004 of a 0.032.

`ESSMIN` is not optional and it is not an edge effect: `C_M` is IDENTICAL for
every target in these sims (checked: relative spread 0.0), so a BRIGHT target's
noise ball is tiny against the template spacing and the copy sum starves
everywhere -- median ESS 26 at `Mf >= 4778` in the BULK as much as at the edge,
and one bulk cell returned m1 = -7.9 +/- 5.4.  The cut keeps the faint 37% of
the window; the paired result is a statement about those targets.

Two subsidiary nulls from the same tool:

* **Finite prior is not it.**  Thinning the copy catalog by GALAXY 100k ->
  6250 does not produce a trend (the numbers are ESS-starved noise), and the
  Fisher ratio is flat along the flux axis at fixed size -- 0.59/0.70/0.80/
  0.78/0.79 across a 40x change in galaxies per cell.  It tracks SIZE only.
* **Not Monte-Carlo either.**  `SUM q^2 / SUM -r` by size band is unmoved by
  4x the draws (S = 8192 -> 32768: 0.710 -> 0.698, 0.514 -> 0.459), and the
  cross-fit inflation is < 0.3% outside the top 2%.

### 5. The one shared assumption that IS measurably wrong

Flow and templates share the prior's g-dependence: both lens a template by
`m + g.dm_dg + g.d2m_dg2.g/2`.  The rendered catalogs test that directly --
`(m_+ - m_-)/2g - dm_dg` is the third-order term, and noise averages out over
200k galaxies.  As a fraction of `<|dm_dg|>`, by the `g = 0` arm's `Mr/Mf`:

| band | Mf | Mr | **M1** | M2 | Mc |
|---|---|---|---|---|---|
| < 3.005 | +0.0001 | +0.0001 | +0.0027 | +0.0009 | +0.0003 |
| 3.005-3.252 | +0.0001 | +0.0003 | **+0.0109** | -0.0008 | +0.0004 |
| 3.252-3.407 | +0.0018 | -0.0005 | **+0.0107** | +0.0060 | -0.0019 |
| 3.407-3.530 | +0.0056 | +0.0061 | **-0.0179** | +0.0018 | +0.0045 |
| > 3.530 | -0.0056 | -0.0096 | **-0.2703 (40 sigma)** | -0.0079 | -0.0113 |

Only `M1` -- the component that carries `g1` -- is affected, and it is clean in
neither the bulk (+0.3%) nor the edge (-27%).  Weighted by each band's share of
`D` this is worth roughly 0.0005 in m1, so it is NOT the -0.018 on its own, but
it is a real, shared, 40-sigma defect in the prior's response model and it is
the third derivative bug in this repo's history ([[bfd-second-derivs-are-float32]],
[[stale-chart-constants-were-the-peak]]).

**The test that separates a genuine third-order term from a wrong `dm_dg`**:
render a small catalog at `g = 0.005` and repeat the table.  A true `g^2 d3m/dg3`
shrinks 16x; a wrong stored derivative does not move.

### Where this leaves the number and the next move

`m1 ~ -0.018 +/- 0.004` on the flux-selected windows is unchanged.  What
changed is where to look for it: **not in the flow, and not in a size band.**
The candidates that survive are the ones flow and templates SHARE -- the
prior's quadratic-in-g response (section 5, sized at ~0.0005 so far), the
recentring/`badcenter` selection, and the target-vs-prior population itself.

### Artifacts

`dev/edge_share.py` (contribution decomposition, `CUT`, `FLUX_LO`, `SIZE`,
`CORR`), `dev/prior_n.py` (flow vs copy sum, paired, `CELL`, `NGAL`, `ESSMIN`,
`FLUX_LO/HI`), one line in `dev/band_tmpl.py`'s cache to keep the `gal` column.

## 2026-09-02 (night, cont. 6) -- The M1 "-27% response error" was the
DIAGNOSTIC, not the pipeline.  With a clean reference it is +0.8%

`dev/response_g3.py`, plus seven fresh renders (200000 galaxies, `--seed 0
--pop bulgedisc`, the arguments recovered from `targets_deep_*_200k_v2`'s FITS
header): noisy `--g1 +/-0.005`, and a NOISELESS set at `g1 = 0, +/-0.005,
+/-0.02`.

### The retraction

Section cont. 5 reported `D(g) = (m_+ - m_-)/2g - dm_dg` reaching -27% in `M1`
at `Mr/Mf > 3.53`, and the `g = 0.005` render confirmed it does not shrink 16x,
so it is not the second-order truncation.  Both statements are true and both
are about a broken reference.  Two mistakes, both in the diagnostic:

* **The `dm_dg` column of a NOISY target catalog is contaminated.**  Same
  galaxies, rendered with and without `--add-noise`, the column differs -- in
  `M1`, by -0.25% / -0.55% / -2.1% / **-10.5%** over the bands from 3.005 up.
  It is evidently taken off the noisy recentred stamp.
* **Binning on the same arm's noisy `Mr/Mf` correlates the bin with the
  residual.**  Every arm uses `--seed 0`, so they share the noise FIELD;
  recentring leaves a little of it in `m_+ - m_-`, and the g = 0 arm's size
  carries the same field.

Production uses neither: `dev/band_tmpl.py` reads the COPIES (built from the
noiseless `moments_bulgedisc_v2`) and `shear.py` trains on the same noiseless
catalog.  The error was confined to cont. 5's table.

### The three measurements, `M1`, as a fraction of `<|dm_dg|>`

| `Mr/Mf` band | noisy arms, NOISY `dm_dg`, noisy bins | noiseless arms, clean ref | **noisy arms, CLEAN ref** |
|---|---|---|---|
| < 3.005 | +0.0027 | +0.0001 | **+0.0007** |
| 3.005-3.252 | +0.0109 | +0.0000 | **+0.0017** |
| 3.252-3.407 | +0.0107 | -0.0001 | **+0.0030** |
| 3.407-3.530 | -0.0179 | -0.0002 | **+0.0045** |
| > 3.530 | -0.2703 | -0.0003 | **+0.0077** |

Mf, Mr, M2, Mc are at 1e-4 in every column.

* **The stored derivative table is RIGHT.**  Noiseless, `dm_dg` reproduces the
  finite difference to ~1e-4 in every band and component, and the tiny `M1`
  residual that is left DOES scale as `g^2` (-0.0003 at g = 0.02 -> -0.0000 at
  g = 0.005), which is the truncation behaving exactly as advertised.
  `imsims/sim.py`'s "the first derivatives are fine at 0.009%" stands.
* **What is real is a RECENTRING effect**: with a clean reference the noisy,
  recentred measurement's `M1` response exceeds the noiseless template response
  by +0.07% in the bulk rising monotonically to +0.80% at the compact end, and
  it is IDENTICAL at g = 0.02 and g = 0.005, so it is not a shear-expansion
  term at all.  Weighted by band population it is ~+0.24%, i.e. worth roughly
  0.002 in m1 -- 5x cont. 5's estimate, still not the -0.018.

That effect is not unmodelled: the copy sum carries recentring by enumerating
shifts and the flow carries it in the centroid layer.  So it belongs to the
open centroid-residual topic ([[centroid-var-floor-quantified]],
[[centroid-gain-is-population-calibrated]]), not to a new bug -- and the
question it poses is whether either estimator gets this +0.8% right, which
`dev/prior_n.py`'s paired agreement suggests they get equally wrong or equally
right.

### Two things worth keeping

* **Same-seed arms make the pixel noise CANCEL** in `m_+ - m_-` (moments are
  linear in the image at fixed weight and centroid), which is why 200k galaxies
  resolve a 0.0001 effect at g = 0.005 where a naive noise budget says they
  cannot.  It is also why a contaminated reference reproduced to four decimals
  across a 4x change in `g` and looked so convincing.
* **Analytic ground truth is NOT available for bulgedisc.**
  `imsims/analytic.py` is the exact k-space closed form for a two-Gaussian
  bulge+disc but hard-codes `BULGE_FRAC = 0.5`, while `sim.py:319` draws
  `bulge_frac ~ U(0, 0.4)` per galaxy; only gauss2 sets
  `pop["bulge_frac"] = analytic.BULGE_FRAC` (`sim.py:510`).  Generalising
  `analytic.moments` to a per-galaxy bulge fraction is a one-parameter change
  and would give bulgedisc an exact `dm_dg`, `moments(theta, g)` and `cov_M` --
  the cheapest way to make any of this exact rather than rendered.

### Artifacts

`dev/response_g3.py` (`G`, `ZERO`, `PLUS`, `MINUS`, `BIN`, `SIZE`); `ZERO`
supplies both `dm_dg` and the binning, so point it at a NOISELESS render of the
same galaxies whenever the arms are noisy.  The seven renders live in the
session scratchpad -- they are diagnostics, not data.

## 2026-09-02 (night, cont. 7) -- The flow route's selection terms ARE recentred
and the template route's are not.  Measured, it is worth ~1% of `R_s`

`dev/rs_copies.py`.  The asymmetry is structural and `selection_terms`' own
docstring names it -- "`L(X^G)`, the detection factor, and its `Delta^2 u` sum
are what the centroid layer already carries (eq. 36)":

* FLOW route (`dev/size_offline.py:prior_draw`) builds with `centroid=True` and
  draws through the whole bijection stack, so its prior sample is the
  RECENTRED population.
* TEMPLATE route is `shear.lens(tm, tdm, td2m, g)` on
  `moments_bulgedisc_v2.fits` -- the noiseless, PERFECTLY CENTRED catalog
  moments lensed by the derivative table.  No shift enumeration at all.

`window_prob` convolves with `C_M` in both, so the noise is common; only the
recentring is not.  (Note `dev/band_tmpl.py` does not have this problem on the
TARGET side -- it sums the 22.7M copies, which carry recentring by
enumeration.  It is the selection-term template route that uses the plain
moments file.)

Test, without touching a checkpoint: run the template route on the COPIES,
importance-resampled by their eq. (36) weights to an equally-weighted set
(`selection_terms` takes a plain mean).  `copies` vs `moments_100k` is then the
same 100000 galaxies differing ONLY in whether the shifts are enumerated.

### `R_s11`, fd = 0.02

| window | `moments_100k` | **copies (recentred)** | delta | `moments_1M` | flow 2^23 |
|---|---|---|---|---|---|
| flux only `Mf>=1600` | -0.2492 | -0.2499 +/- 0.0021 | **-0.0007 (0.3%)** | -0.2348 | -0.2243 |
| nominal | +0.2633 | +0.2680 +/- 0.0032 | **+0.0047 (1.8%)** | +0.1531 | +0.0935 |
| band 3.005-3.252 | -0.1112 | -0.0919 | +0.0193 | -0.1305 | -0.1859 |
| band 3.252-3.407 | -0.2319 | -0.2426 | -0.0107 | -0.2350 | -0.1942 |
| band 3.407-3.530 | -0.1527 | -0.1522 | +0.0005 | -0.1308 | -0.1382 |

The `+/-` is the spread of two independent resample seeds (2^21 draws each,
1.46M distinct copies, all 100000 galaxies represented).

**Recentring is not the gap.**  It moves `R_s11` by 0.3% on the flux window --
not even sign-resolved against the resampling spread -- and 1.8% on the
nominal one, against a flow-vs-template gap of 4.5% and 39%.

**What dominates is the template route's own galaxy sampling.**  100k -> 1M,
both CENTRED, moves `R_s11` by 6% on the flux window and 72% on the nominal
one.  The 100k route also fires `selection_terms`' concentration warning on
three of five windows (one template carrying 9%, 11%, 22% of `R_s11`), so its
band numbers are not trustworthy and the two middle bands' larger deltas above
should not be read as recentring either.  Same conclusion as cont. 2 reached
from the other side: at 2^23 flow draws the TEMPLATES are the noisy estimator.

### Artifacts

`dev/rs_copies.py` (`ROUTE` in `copies|moments_100k|moments_1M`, `WINDOW`,
`NDRAW`, `WMIN`, `SEED`, `CACHE`).  ONE route and ONE window per process: the
`eqx.filter_jit` CUBIN accumulation OOMs the GPU at about the fifth call at 1M
rows.  The resample is cached so per-window reruns skip the 3 GB copies read.

## 2026-09-02 (cont. 8) -- Provenance audit: the `_v2` chain is CLEAN.  Then a
clean-slate rebuild to `_v3` at a derived depth, with the provenance made
mechanical

Two halves.  The audit was the task; the rebuild is what the user asked for on
the strength of it ("start from scratch just to be safe").

### Part 1 -- the audit.  Nothing stale was in use

`NEXT_SESSION.md`'s Lead 1 -- that `centroid.py:486` trains the centroid layer
at the TRAINING catalog's `Sigma_X` (119.4) while `bias.py:2096` evaluates at
the TARGET catalog's (107.5) -- **is wrong, and the mismatch does not exist.**
`galaxies` on that line comes from `load_copies(a.copies)` (`centroid.py:87`),
i.e. the COPIES file's `GALAXIES` extension, which was rendered at
`--noise-sigma 0.9`:

| | `cov_odd[0,0]` | sqrt |
|---|---|---|
| `copies_bulgedisc_v2` GALAXIES (centroid TRAINING) | 11551.9639 | 107.480 |
| `targets_deep_g0_200k_v2` (bias.py EVAL, `bias.py:2096`) | 11551.9639 | 107.480 |

Byte-identical.  `--sigma-scale` never left 1.0.  Nothing to measure.

The 1.0-vs-0.9 render difference is real but INERT.  `sim.NOISE_SIGMA`
defaulted to 1.0 and the two `moments_*` renders took the default while
targets/copies passed 0.9 explicitly, so those two files' `cov`/`cov_odd` are
23% wrong -- and **nothing reads them**: `bias.load_cov` is called on
`path(cat["zero"])` (`bias.py:2117`) and all ~20 `dev/*.py` `cov_odd` reads go
through `bias.CATALOGS[POP]['zero']`.  Direct proof `noise_sigma` cannot touch
the moments: `copies_bulgedisc_v2`'s `GALAXIES` (0.9) is BIT-IDENTICAL to
`moments_bulgedisc_v2` (1.0) in `moments`, `dm_dg`, `d2m_dg2`, `centroid`,
`xyshift`, `nda` and `badcenter`; only `cov` differs, by exactly
0.81 = (0.9/1.0)^2.

Lead 2, all confirmed on `POPULATION` (the noiseless drawn parameters, so no
noise confound):

* the three target arms are the same 200000 galaxies, exactly, in all 8
  columns -- the antithetic pairing is intact;
* `copies_bulgedisc_v2`'s `POPULATION` **and** its `GALAXIES["moments"]` equal
  `moments_bulgedisc_v2`'s exactly, so the copy sum and the flow share one
  prior galaxy for galaxy;
* `moments_bulgedisc_v2_1M`'s first 100000 rows are NOT the 100k catalog -- an
  independent draw, since `--n` changes the RNG stream.  Same population
  though: KS on all 8 parameters gives p = 0.31-0.85 against the 100k catalog
  and p = 0.43-0.94 against the targets.

Lead 3, the checkpoints.  Six distinct files (md5).  In every one the
`RawMomentStandardize.mean/std` matches its own `ShearResponse.chart_loc/
chart_scale` and `CentroidMarginalize.mean/std`, and `e_scale` matches the
symmetrised spin-2 std (0.11200 against slots 0.11208/0.11193; `_1M` 0.110495
against 0.11051/0.11048).  No repeat of [[stale-chart-constants-were-the-peak]]
or [[e-scale-was-unsymmetrised]].  Each centroid flow is grafted from its OWN
shear flow: `centroid_bulgedisc_v2`'s chart is 3.33482 = `shear_bulgedisc_v2`'s,
and the `_1M` pair's is 3.34471.

**Method note for the next person doing this.**
`RawMomentStandardize.mean/std` are Adam PARAMETERS (`bulk.train` hands both
spin-2 slots to the optimiser -- see the class docstring), so a checkpoint's
chart NEVER reproduces its catalog's raw `to_coords` statistics.  The `_v2`
drift is 0.075 in mean and 1.21x in slot-0 std.  Do not read that as staleness.
The discriminating comparison is against the WRONG catalog: old `moments.fits`
gives 3.783/0.304 where v2 gives 3.409/0.510, and the checkpoints are
unambiguously on v2.

Two provenance HOLES that the audit could not close, and that motivated part 2:

1. **No training log exists for any `_v2` flow.**  `logs/` has none, the fish
   history has none (they were run through a tool, not the user's shell), and
   `retrain*.sh` does not build them.  Their arguments survive only as prose in
   this file's 2026-08-27 entry.
2. **`CentroidMarginalize.gain` is `eqx.field(static=True)`** -- not
   serialised.  No checkpoint records the gain it was trained at.  It predates
   ed58d41 so it is 1.0, and `bias.py --centroid-gain` defaults to 1.0, so
   production matched; but passing 1.4043
   ([[centroid-gain-is-population-calibrated]]) evaluates the weights off their
   training point with nothing on disk to catch it.

### Part 2 -- the `_v3` rebuild

Everything previously generated was moved to `archive_20260902/` under
`../bfd_cnf_imsims/data` (130 entries, 93 GB), `flows/` (3143), `logs/` (47),
`plots/` (6) and `dev/` (66 `.npz`/`.fits`/`.log`).  All are gitignored, so git
is untouched; tracked `dev/*.py` stayed.  gauss2 and sersic went too.

**`rebuild_v3.sh {render|copies|check|train}` is the only place the `_v3`
artifacts are produced and the only record of how** -- that is the fix for hole
1 above.  Three changes against `_v2`:

* `--noise-sigma 0.93` on EVERY render including the noiseless prior, and
  `sim.NOISE_SIGMA`'s default moved 1.0 -> 0.93 so the slip cannot recur.
* prior and copies on `--seed 0`, all three target arms on `--seed 1`, so
  prior and targets are independent BY CONSTRUCTION rather than by the accident
  of differing `--n`.  The arms share a seed with each other, which is what
  makes +g and -g the same galaxies under the same noise field.
* bulk trains to 150000 steps, the converged value ([[bulk-was-undertrained]]);
  `_v2`'s bulk ran 20000, where the measured m1 is still a function of the
  schedule.

**The depth, derived rather than inherited.**  Reference is
`~/Documents/BFD_cNF/summary_templates_new.fits`, 1.37M rows, UNCUT: `Mf` p50
1199, `sqrt(Var Mf)` p50 60.9, flux S/N p[5,50,95] = 6.8 / **19.2** / 194.
Two defensible criteria disagree by a factor 1.5 --

| criterion | noise_sigma |
|---|---|
| match median flux S/N | **0.927** (used) |
| match absolute noise, `sqrt(Var Mf)` | 0.612 |

-- because the sim's `Mf` median sits 48% above the real UNCUT median: the flux
marginal was tuned against the S/N > 10 CUT sample, so it does not reproduce
the real catalog's undetected faint end.  S/N is the ratio the estimator
responds to and absolute moment units are arbitrary, so the S/N match wins.
**At NO depth does the S/N SHAPE match**: sim p5/p50 = 0.48 vs real 0.354,
p95/p50 = 16.3 vs 10.1.  That is the flux marginal, not the depth, and it is
the same unfinished business as
[[bulgedisc-real-match-ellipticity-limit]].  Verified at render: every `_v3`
catalog comes out at median flux S/N 19.1-19.2.

**`dev/check_provenance.py` turns each of the audit's five bug shapes into an
assertion** and gates `rebuild_v3.sh train`.  All 14 pass on `_v3`: one
`noise_sigma` everywhere; arms carry g1 = 0/+0.02/-0.02 and share a population
row for row; copies' `Sigma_X` == targets' (12334.9307); `C_M` constant;
copies' `GALAXIES` moments/`dm_dg`/`d2m_dg2` == the prior's; prior and targets
are different draws; prior inside both chart ceilings (max Mr/Mf 3.6835 vs
3.6926, max Mc/Mr 6.6488 vs 6.6621 -- tight, and a different draw could cross).
Copies: 23020358 from 100000 galaxies, detection integral 1.0000.

### The corner plot -- `dev/check_bulgedisc_vs_real.py`, now three-way

Real (measured) / `_v3` prior templates (NOISELESS) / `_v3` g=0 targets (image
noise + recentred), common S/N > 10 cut, `plots/bulgedisc_vs_real_corner.png`.

| | Mf p50 | Mr/Mf p50 | Mr/Mf sd | Mc/Mr p50 | Mc/Mr sd | \|e\| p50 |
|---|---|---|---|---|---|---|
| real | 1540 | 3.495 | 0.475 | 6.673 | 0.700 | 0.090 |
| v3 templates | 2072 | 3.132 | 0.433 | 5.875 | 0.614 | 0.094 |
| v3 targets | 2051 | 3.119 | 0.449 | 5.847 | 0.651 | 0.115 |

* The sim is **35% too bright and 10% too small**, and too narrow in both size
  coordinates.  Known, unfixed, out of scope here.
* **The real population's MEDIAN `Mc/Mr` is 6.673, ABOVE the chart's ceiling
  `POINT_SOURCE_MC` = 6.662089** -- which is why 53% of real rows are
  off-chart.  This has been noted before as "real galaxies routinely exceed
  this ceiling"; "routinely" undersells it.  More than HALF the real catalog
  lies outside the chart the flow is built on, and that ceiling is a hard
  boundary by construction.  The sim never approaches it (0% of templates, 5%
  of noisy targets).  For a method that is meant to run on the real catalog
  this is structural, not a tail effect.  NOT chased here.
* Templates vs targets separate only in `|e|` (0.094 -> 0.115), which is
  exactly what image noise plus recentring should do -- the check that the
  target render is the same population MEASURED, not a different population.

Also fixed: `bulk.COORD_LABELS` said `M_r/M_f` and `M_c/M_r` for slots that are
LOGITS against the point-source ceilings, so a plotted 2.1 looked comparable to
a catalog `Mr/Mf` of 3.15.  This affected `bulk.py corner` too.

## 2026-09-02 (cont. 9): the centroid layer's k^4 brackets -- steps 1 and 2 of
the NEXT_SESSION plan.  `gain` is gone, the coefficient is physical, and the
copy NLL turned out to be blind to it

Steps 1 and 2 of `NEXT_SESSION.md`'s plan, measured separately.  Baselines were
re-run first, not taken from the file, and reproduce it exactly.

### The decomposition that closes the system

Every first-order centroid response is one contraction,
`dM_a = -1/2 Sigma_u^{ij} <k_i k_j kernel_a>`.  Writing each symmetric 2x2
bracket as trace + traceless -- `s0 = tr Su`, `s1 = Su_xx - Su_yy`,
`s2 = 2 Su_xy` -- closes all four:

    dMf = -1/4 (s0 Mr + s1 M1 + s2 M2)
    dMr = -1/4 (s0 Mc + s1 N1 + s2 N2)
    dM1 = -1/4 (s0 N1 + s1 (Mc + K1)/2 + s2 K2/2)
    dM2 = -1/4 (s0 N2 + s1 K2/2       + s2 (Mc - K1)/2)

`Mf, Mr, M1, M2, Mc` are measured.  `N = N1 + i N2` is the k^4 SPIN-2 moment
(`int k^2 (kx^2-ky^2) G + i int k^2 2 kx ky G`) and `K = K1 + i K2` the k^4
spin-4; neither is.  Wick on the Gaussian-in-k ansatz gives
`N = 3 Mr (M1 + i M2)/Mf` and `K = 3 (M1 + i M2)^2 / Mf`, and those are exactly
what the closed-form backbone already used -- checked against its OWN
first-order expansion at Sigma_X * 1e-4, p50 agreement to 8 digits on all four
channels (the one large max-ratio outlier is a galaxy with M1 ~ 0).

`N` appears TWICE: as dMr's traceless half and as dM1/dM2's trace.  That is why
one coefficient moves the size and ellipticity responses together -- and why
step 1's two regressions turned out to be one object.

### Step 1 -- measured `Mc` in dMr's trace part.  Zero parameters

One line in `_transport`: `Mr_out -= sign * 0.25 * s0 * (Mc - Mc_ansatz)`.
`Mc/Mc_ansatz` has p50 0.928 on the v3 prior.

Ratios (layer/catalog) on the two NATIVE-depth grids:

| | Mf | Mr | Mc | spin-2 |
|---|---|---|---|---|
| sigma 0.67 before | 0.981 | 1.039 | 1.179 | 0.7144 |
| sigma 0.67 after | 0.981 | 0.968 | 1.072 | 0.6194 |
| sigma 1.30 before | 0.940 | 0.971 | 1.067 | 0.7125 |
| sigma 1.30 after | 0.940 | 0.912 | 0.984 | 0.6125 |

Mc improved on every axis and its per-galaxy `resid` nearly HALVED (1.89e-3 ->
1.04e-3 at sigma-scale 0.72; 3.07e-3 -> 1.70e-3 at nominal).  Spin-2 got worse
by a uniform ~0.095, as predicted, entirely through `e = M1/Mr`'s denominator
-- raw `M1_out` is untouched.

**The prediction that FAILED, and what it told us.**  `NEXT_SESSION` expected
Mr to move toward 1.0 on the strength of "dMr too big by 1.10".  It did not:
Mr was already right (1.039/0.971) and went uniformly 5-9% LOW.  So the
ansatz's dMr was right BY CANCELLATION -- the trace 7.2% low against a
traceless half ~7% high.  That traceless half is `N`, which step 2 then fitted.

**Invertibility, settled empirically.**  The backbone was never exactly
self-inverse: `Sigma_u = R^-1 Sigma_X R^-1 / Mf^2` is re-solved from whichever
point is handed in, so the round trip always carried an O(Su^2) error.
Measured on 5000 v3 galaxies at nominal depth, `Mr` round-trip |d|/m p50
5.52e-5 BEFORE -> 5.16e-5 AFTER (p99 3.09e-3 -> 3.05e-3) against a shift of
4.8e-3.  No Newton step needed; a first-order correction adds nothing to what
was already there.

### Step 2 -- one trained coefficient on `N`.  `gain` deleted

`c_spin2` multiplies the Gaussian `N`.  It is REAL, not complex: an imaginary
part rotates the k^4 spin-2 away from the k^2 one (isophote twist), which is
real per galaxy but averages to zero given parity-even conditioning, and every
invariant the coefficient can see (`Mr/Mf`, `Mc/Mr`, `|e|`) is parity even.  A
free imaginary part could only learn a parity-odd artifact of the training
sample -- the [[spin2-standardize-anisotropy]] failure class.  Test pins it.

`CentroidMarginalize.coeff` is a 3->32->1 tanh MLP with a ZEROED output layer,
so `c_spin2 == 1` exactly at init and an untrained layer reproduces the bare
ansatz bit for bit.  Inputs are the chart's own standardised `z[1]`, `z[2]` and
`|e|^2` in units of the chart's symmetrised spin-2 std -- no new population
constants, so no repeat of [[coeff-whitening-was-stale-too]].  `|e|^2` and not
`|e|`: `sqrt(M1^2+M2^2)` has an infinite gradient at a round galaxy, which
`test_gradients_are_finite_for_a_round_galaxy` caught immediately.

`gain` is deleted from `models/centroid.py`, `bulk.build_flow`, `--centroid-gain`
on `bias.py`/`centroid.py`, `dev/gauge.py`, and `centroid.py calibrate` (mode
removed).  **Old centroid checkpoints no longer deserialise** -- the layer has
leaves now.

**The copy NLL is blind to this coefficient.**  `NEXT_SESSION` expected the
existing NLL loop to just work.  It does not: 16000 steps wandered (nll 77-133,
no trend) and left the layer at a spin-2 response ratio of **-4.4**.
`dev/c_scan.py` settles why -- a CONSTANT `c` on 65536 copies:

| c | NLL | resp ratio |
|---|---|---|
| -1.0 | 113.74 | +6.60 |
| 0.90 | 113.51 | +1.03 |
| 1.0 | 113.38 | +0.65 |
| 3.0 | 113.22 | -5.47 |

0.7 nats of structureless noise while the response sweeps 12-fold.  Not a
tuning failure: the whole marginalisation changes `|e|` by ~0.5%, a negligible
perturbation of a density and an enormous one of a shear response.  Only an
estimator aimed at the response resolves it.

So `centroid.py train` now fits the catalog's own weighted copy shifts by least
squares (which estimates the CONDITIONAL MEAN -- exactly what a deterministic
transport can carry, with the `Var[.|m]` floor as residual, not bias).  Both
channels `N` touches, each normalised by its own catalog spread.  Cheaper than
NLL too: no log-det.  `--holdout` gives the within-population split.

**Results.**  Fitted `c_spin2` p5/p50/p95 = 0.785/0.843/0.890 -- essentially a
constant, and a closed-form estimate made BEFORE fitting said 0.87.

| | Mf | Mr | Mc | spin-2 |
|---|---|---|---|---|
| sigma 0.67 step 2 | 0.981 | 0.977 | 1.081 | 1.0046 |
| sigma 1.30 step 2 | 0.940 | 0.920 | 0.992 | 1.0142 |

Held-out half (fit on galaxies [0,50k), reported on [50k,100k)): spin-2 ratio
**1.0329**.  `dev/sigma_scan.sh`: 1.0233 -> 1.0045 -> 1.0164 across 2x in
sigma, against 0.724 -> 0.707 before.  The two native grids are DIFFERENT
galaxies at depths never trained on (Sigma_X 6394 and 24177 vs 12335), so one
coefficient fitted at one depth holds to 2% over 3.8x in Sigma_X -- because the
depth dependence is derived, not fitted.  That is what `gain = 1.4043`, a patch
on the net response, could never do.  M1/M2 per-galaxy `resid` also halved
(1.32e-4 -> 5.94e-5 at sigma 0.67).

Log-det consistency `|fwd + inv|` p50 **6.96e-4 with the trained c vs 6.70e-4
at c = 1** -- the coefficient adds nothing.  Full suite 58 passed, 1 skipped.

### What step 2 did NOT close, and what it hands to step 3

`Mr` is still 2-8% low (0.977 / 0.920).  Expected and quantified: the size
channel sees `N` only through `Sigma_u`'s traceless part, and `|s|/s0 = 0.178`
on this population -- which is `2|e|` to O(e^2), derived and confirmed
numerically.  ~35x less leverage than the spin-2 channels.  The scan makes the
tension explicit: the ellipticity channel wants `c ~ 0.905`, the size channel
wants `c ~ 0.36`, and the fitted net went to the ellipticity optimum because
that is where the leverage is.  One `N` cannot satisfy both, so the size
deficit lives in a bracket `N` cannot reach -- `K` (spin-4, step 3) or second
order (step 4).

Artifacts: `flows/centroid_v3.eqx` (all 100k, production),
`flows/centroid_v3_s2.eqx` (holdout 0.5, the validation run),
`logs/centroid_v3{,_s2}.log`.  `rebuild_v3.sh` updated.  Reproducing the step
0/1 numbers needs a `git stash` -- the old checkpoint will not load.

## 2026-09-02 (cont. 10): step 3, the k^4 spin-4 bracket.  Implemented and
correct; a NULL on this population, and provably so -- it is degenerate with
the spin-2 coefficient until the PSF becomes elliptical

### What was added

`K = c_spin4 * 3 (M1 + i M2)^2 / Mf`, the k^4 SPIN-4 moment, as the third and
last first-order bracket.  From the decomposition in cont. 9 it enters only
dM1 and dM2, and only against Sigma_u's TRACELESS part:

    dM1 += -1/8 (s1 dK1 + s2 dK2)      dM2 += -1/8 (s1 dK2 - s2 dK1)

`c_spin4` is a second head on the same trunk as `c_spin2` (3 invariants ->
32 -> 2, output layer zeroed so BOTH coefficients are exactly 1 at init).
Real, not complex, for the same parity argument as `c_spin2`.

**A correction to the plan's wording.**  `NEXT_SESSION` says the spin-4
coefficient is "zero for any co-elliptical G".  It is not: an ellipse HAS a
spin-4 moment, and the Gaussian/Wick value `3 (M1+iM2)^2 / Mf` is nonzero
whenever M1 or M2 is.  What is true is that for a co-elliptical G the spin-4
moment's PHASE is locked to twice the spin-2 moment's, so only a real magnitude
is left for `c_spin4` to carry.  Breaking the lock needs a SECOND spin-2
direction -- which is exactly what an elliptical PSF supplies and what this
population does not have.

### Wiring verified

`dev/c_scan.py` now scans both.  Across `c_spin4` from -1.9 to 3.0 the layer's
`dMr/Mr` is **-7.2865e-03 at every value, bit-identical** -- a trace cannot see
a spin-4 moment, and if a refactor ever gave `K` the trace contraction this is
what would catch it.  `dMf` likewise exactly 0.  `Mc` moves at most 5e-2
relative, entirely through `mc_ansatz_out`'s dependence on M1_out/M2_out, i.e.
the differential-Mc correction staying self-consistent.
`test_spin4_moves_only_the_spin2_channels` pins all of it, plus the requirement
that `c_spin4` be a no-op on a perfectly round galaxy.

### Why it is a null here: an |e|^2 degeneracy

Per galaxy, on 20000 v3 templates:

    d(response)/dc_spin4  /  d(response)/dc_spin2  =  k |e|^2,
    k = -0.976 (p50), 8.7% spread

which is what the algebra says: with Sigma_X isotropic, `s = -2 s0 e` and the
two brackets enter the ellipticity shift only as

    de  ~  (3/4) s0 (Mr/Mf) e  (-c_spin2 + c_spin4 |e|^2)

`|e|^2` is ALREADY one of the net's three inputs, so `c_spin2(|e|^2)` can
represent anything `c_spin4` adds.  Ensemble leverage ratio 12.1x
(-3.93 vs +0.32 per unit coefficient, OPPOSITE signs).

The one thing that is NOT degenerate is precisely what cont. 9 left open:
`c_spin4` moves the ellipticity response WITHOUT touching dMr.  But it cannot
reach far enough.  dMr wants `c_spin2 ~ 0.37`, which leaves the response ratio
at ~2.8; pulling that back to 1.0 needs `c_spin4 ~ -4.6` -- a spin-4 moment of
OPPOSITE SIGN and 4.6x the co-elliptical magnitude, and outside `_C_MAX`.  So
`K` does not close the Mr deficit either.  That deficit is second order or
`Var[.|m]`, not a first-order bracket.

### Measured, 3 seeds of the joint fit (holdout 0.5)

| seed | c_spin2 p50 | c_spin4 p50 | held-out spin-2 | native 0.67 spin-2 |
|---|---|---|---|---|
| 0 | 0.8444 | 0.618 | 1.0396 | 1.0095 |
| 1 | 0.8413 | 0.645 | 1.0364 | 1.0034 |
| 2 | 0.8434 | 0.660 | 1.0317 | 1.0039 |

Step 2, for comparison: c_spin2 0.843-0.859, held-out 1.0329, native 0.67
0.9993-1.0046.  **The ensemble response is unchanged.**  So is Mr (0.962 vs
0.960 held out; 0.979 vs 0.977 at native 0.67) and so is `sigma_scan`
(1.031 -> 1.012 -> 1.024, against step 2's 1.023 -> 1.005 -> 1.016; the ~0.008
offset is inside the c_spin2 scatter between two step-2 runs).

`c_spin4` is nonetheless well DETERMINED -- 0.641 +/- 0.021 over three seeds,
nowhere near a bound and not wandering -- because the degeneracy is only 91%
exact (the 8.7% spread in k above).  What is determined is the combination:
`c_eff = c_spin2 - c_spin4 |e|^2` comes out 0.831 / 0.824 / 0.827, sd 0.004,
against step 2's 0.835-0.851.  `corr(c_spin2, c_spin4) = +0.866` across
galaxies, which is the degeneracy showing up directly in the fit.

**The one thing that did move**: per-galaxy `resid`, reproducibly, at every
depth -- native 0.67 M1 5.94e-5 -> 5.32e-5 +/- 0.26e-5, M2 5.86-6.04e-5 ->
5.22e-5, Mr 2.82-2.89e-4 -> 2.52e-4 +/- 0.03e-4.  ~11%, outside both runs'
scatter.  Note `c_spin4` provably cannot touch Mr at all, so that part is
`c_spin2`'s refitted SHAPE, not new physics: with `c_spin4` free to carry the
|e|^2 dependence, `c_spin2` is fitted flatter in |e| (0.821 low-|e| vs 0.852
high-|e|) and the pair is better conditioned than one coefficient alone.

### Verdict

Keep it, on two grounds that are NOT "it improved the bias": it is the slot an
elliptical PSF needs (step 3's whole stated purpose), and the per-galaxy
conditional-mean residual is ~11% better across three seeds.  It is honestly a
NULL on every ensemble number, and if the extra weakly-identified parameter is
unwanted, deleting the second head is a one-line revert.  Do not expect it to
pay until Sigma_X and the PSF stop being circular and constant.

Artifacts: `flows/centroid_v3.eqx` (production, all 100k),
`flows/centroid_v3_s3{,_seed1,_seed2}.eqx` and their logs.

## 2026-09-03/04 -- The bulk flow could only make p(e | spin-0) an isotropic
Gaussian.  Fixing that removed the m1 flux gradient; then three separate
defects in the selection terms turned up behind it

### The audit finding, and it is architectural not a fit quality issue

`Spin2CouplingLayer` applied ONE shared scalar to slots 3, 4, and
`Spin0AutoregressiveLayer` never read them.  So the whole 8-layer stack
collapsed to

    b[0:3] = F(z[0:3])          b[3:] = z[3:] * S(z[0:3])

which, against an N(0, I) base, makes the modelled conditional shape density
EXACTLY an isotropic Gaussian and the spin-2 score EXACTLY linear in e.  Not a
tuning problem -- nothing in the parameter space could do otherwise.

Diagnostic (new, and the one to keep using): k-NN windows in spin-0,
`var(|e|^2)/mean(|e|^2)^2`.  The templates want **0.565** at K = 200; the old
flow gave **1.316** against an architecture FLOOR of 1.000.  Real |e| is a
narrow shell and the old stack could only err on the over-dispersed side.

### What shipped

A pure stretch whose profile in |e| the net learns:

    y[3:] = x[3:] * exp(h(z0, z1, z2, log1p(|e|^2)))

i.e. the conditioner MLP with |e|^2 as a fourth input.  No analytic family, no
basis, no spline.  Log-det `2h + log1p(2 q dh/dq)`.  Only the FIRST layer bends
(the bend composes multiplicatively; the interval-free form still wants the
data-adjacent coordinate).  Monotonicity is WATCHED, not enforced --
`check_monotone` measures 0.0000% on training data and on 200k flow samples.

| chain | bulk val nll | disp K=200 | unwindowed m1 |
|---|---|---|---|
| old arch | 38.9721 | 1.316 | -0.0391 +/- 0.0066 |
| power `a` (v6) | 38.7514 | 0.524 | -0.0191 +/- 0.0111 |
| **free net (v10)** | **38.6776** | **0.526** | **-0.0140 +/- 0.0073** |

Flux quintiles, the diagnostic that had survived every prior change:
old `-0.095, -0.127, -0.042, -0.023, +0.005` -> v10 `-0.048, -0.034, -0.015,
+0.002, -0.007` (bars 0.056, 0.017, 0.008, 0.016, 0.004).  No monotone trend
survives.

### Parameterisations that failed -- do not re-derive

* **Power tail** `T = (gamma beta/a) expm1(a log1p(q/beta))`.  Fit well but
  pinned `a` at a STABILITY ceiling (86.8% of templates at 150k steps).
  Raising the bound 4 -> 16 gives `val nll inf`: a pure power makes the density
  unevaluable off the data envelope, which is exactly where the next stage's
  draws land.
* **Asymptotically linear rational** `gamma q (beta + c q)/(beta + q)`.
  Un-pinned everything but fit worst (38.819, disp 0.650) -- a linear tail caps
  the total compression available.
* **Rational-quadratic spline.**  Rejected as over-parameterised (28 net-driven
  knots per galaxy); "fits better on this catalog" is the claim that would not
  transfer.

The free net beat all three.  It also removed a bug the analytic families
carried: forming `T(q)/q` as `expm1(z)/z * log1p(u)/u` overflows the first
factor at z ~ 88 while the product is ~1e31 and representable.

### Then the selection terms, which had THREE independent defects

1. **The `R_s` concentration warning said "template" for `--window-terms flow`
   runs, where the object is a flow DRAW.**  Cosmetic, but it made the warning
   un-diagnosable for three chains; `selection_terms` now takes `kind`.
2. **`--window-fd` is documented as REQUIRED for any `--window-size` cut and
   was not being passed.**  The reference invocation in the old NEXT_SESSION
   omitted it, so EVERY windowed number in the v3/v6/v10 comparison used the
   divergent pathwise estimator.  Symptom in hindsight: the windowed column
   bounced +0.013 / +0.050 / -0.005 while unwindowed moved smoothly.
3. **262144 draws is not converged**, and is the noisiest point of the scan.
   `--window-draws` now defaults to 2^20.

### The fix: a score-function estimator (`--window-terms score`)

`F` cuts on the MEASURED moment and shear acts on the prior, so `F` carries no
`g` at all.  Differentiating the DENSITY instead of the SAMPLE gives

    Q_s = E[F Q],   R_s = E[F (R + Q Q^T)],    m ~ P(.|0)

No `sigma` in the integrand, so the pathwise `Mf^2/sigma^2` divergence is
removed by construction rather than truncated.  Three-way comparison at 262144
draws (`dev/window_terms_compare.py`, minutes rather than a 40-min `bias.py`):

| estimator | R_s11 | R_s22 | Q_s1/err | top share | **Hill** |
|---|---|---|---|---|---|
| pathwise | -0.1624 | -0.2882 | 1.05 | 44.6% | 1.42 |
| fd = 0.02 | -0.1512 | -0.1668 | 1.19 | 6.1% | 1.20 |
| **score** | -0.1815 | -0.2159 | **0.28** | **4.4%** | **2.92** |

Only the score estimator has finite variance (Hill > 2).  `fd` is NOT the clean
fix its docstring implies for THIS window: that measurement was on a size
window with a ceiling, ours is size-floor-only plus a FLUX ceiling, and
differencing does not cure that (Hill 1.20).

**A free exactness check that was sitting unused: `R_s11 = R_s22` EXACTLY.**
The window cuts only on spin-0 quantities, so `P_s` can depend on `|g|^2`
alone.  Violation: pathwise 77%, fd 10%, score 17% at 262144 -- and for the
score estimator it converges away with draws, which is the proof that it
converges at all:

| draws | R_s11 | R_s22 | asymmetry |
|---|---|---|---|
| 2^16 | -0.2129 | -0.2359 | 10.2% |
| 2^18 | -0.1815 | -0.2159 | 17.3% |
| 2^20 | -0.1679 | -0.1651 | **1.7%** |
| 2^21 | -0.1622 | -0.1694 | **4.3%** |

Converged `R_s11 ~ -0.165`.  `fd = 0.02` gives -0.1512, a gap of 0.014 --
essentially exactly its independently-estimated O(h^2) bias.  Two estimators
with unrelated failure modes agreeing to a known systematic.

### Final numbers, `flows/centroid_v10.eqx`

* unwindowed `m1 = -0.01397 +/- 0.00731`
* windowed, corrected (score, 2^20 draws) `m1 = -0.00368 +/- 0.00964`
* `R_s11 = -0.165`, top draw 3.8%, no warning
* per-target PQRs saved to `pqr/v10_score.npz` -- `ghat` is affine in `R_s`, so
  any further selection variant is now a two-line evaluation with no GPU time.
  Predicted -0.0038 by extrapolation before the run, measured -0.00368.

The v3/v6 WINDOWED numbers are void (defect 2) and their checkpoints are gone
and incompatible, so there is no cross-architecture windowed comparison.  The
unwindowed series is unaffected.

### Open

* **`Q_s2 = -4.4e-03 +/- 2.0e-03`, i.e. 2.2 sigma from a value isotropy forces
  to be exactly zero.**  Largest seen; the standalone scan gave `Q_s1`/err of
  0.19, 0.28, -0.50, -1.03.  See the next-session prompt.
* Centroid ellipticity response ratio **1.0424** against 1.0054 (v6) and
  1.0078 (v3), with `c_spin4` scattered p5 0.577 to p95 1.209.
* The response fit is NOT the lever and did not move across any of these
  chains (dm/dg 0.29/2.29/0.56/0.57/4.91%).  `Var[Q|m]` is ~0.3% of the
  `E[QQ|m]` term in d2P/dg2, so the "59% irreducible" framing does not survive.
* The flux input to `_Coeffs` buys ~1% on this population (2.67 -> 2.63%).

---

## 2026-09-05: Phase A and B of the `GUIDING_PRINCIPLES` plan -- the C01
hypothesis is dead, two quoted numbers were wrong, and the window scan is
limited by galaxies not by draws

Branch `feat/centroid-shear-conditioning`.  Everything below is offline
re-solve or a dev script; no new target integration was run.  Test suite 116
passed / 1 skipped.

### B1: the `C01` hypothesis is FALSIFIED (and it was the whole story)

`dev/isotropy_check.py` gained `--rotate-sigma` and `--sigma-x`.  The old
script rotated the prior draws but held `Sigma_X` FIXED, which is only a
symmetry while it is isotropic.  `rot_sigma` rotates it properly: `X` is a
position, so `Sigma_X -> R Sigma_X R^T` with the ORDINARY angle, not the
doubled one the spin-2 moments turn through.  At `k = 1` that maps `e2p`'s
`[12676.665, 1084.720, 12676.665]` exactly onto `[11591.944, 0, 13761.385]`,
which is the RENDERED `e1m` value -- so the identity is checkable against an
independent catalog rather than only against itself.

Through the FULL chained flow, 65536 draws:

| | isotropic `Sigma_X` | `e2p`, `C01 != 0` |
|---|---|---|
| `\|dlogP\|` p50 / p99 | 7.11e-15 / 2.01e-12 | 7.11e-15 / 2.01e-12 |
| `\|dQ\|/\|Q\|` p50 / p99 | 5.64e-15 / 3.80e-12 | 5.69e-15 / 3.63e-12 |

The float64 floor, identical with and without the off-diagonal.

**The control that makes this non-vacuous was run.**  At fixed moments,
switching `C01` from 0 to 1084.72 moves `log P` by 1.5e-03 (p50, on
`|logP| ~ 37`) and `Q` by 1.3e-02 (p50, on `|Q| ~ 4.9`) -- comparable to
changing the whole `Sigma_X` trace (3.3e-03 and 9.6e-03).  The path is
exercised and it is exactly equivariant.

This was predictable from `_transport` and the code reads correctly:
`Sigma_u`'s traceless part `s = s1 + i s2` (spin 2) is contracted against the
spin-4 bracket as `conj(s) K` and against the spin-2 bracket as `conj(s) N`,
which is the only spin-consistent pairing available.

### Two numbers on the record were wrong

**1.  `psfe1m` is one galaxy.**  Inside the quoted window the top-1 share of
`sum |R11|` is:

| config | top-1 share | max/median |
|---|---|---|
| plain, psfe00, psfe1p, psfe2p | 0.13-0.14% | 8.5-9.7 |
| **psfe1m** | **7.87%** | **572** |

`sane_targets` fires at 1000x the median, so at 572x it keeps this target.
Effect (`dev/psfe_compare.py --drop-top 1`, and `dev/window_scan.py
--drop-top 1`):

| psfe1m | with | without |
|---|---|---|
| `dm1` vs psfe00, uncorrected | -8.37e-02 +/- 6.1e-02 | **-8.95e-03 +/- 4.6e-03** |
| corrected `m1` at 2^24 | -0.0722 +/- 0.0692 | **+0.0093 +/- 0.0084** |
| in-window Fisher ratio | 0.893 | -- |

Do NOT fix this by loosening the guard's factor: that is a tuned constant, and
the concentration report is the right net.  `--drop-top` is now on both
scripts and both numbers should be quoted.

**2.  "The `dc2` signal does not rotate" was quoted off two uncorrelated
bars.**  `A_c1` (the `e1p`/`e1m` antisymmetric part) and `dc2[e2p]` share the
`psfe00` baseline and the same galaxies, so their errors are correlated.
Bootstrapping the DIFFERENCE, 800 paired resamples:

    A_c1     = -2.776e-04 +/- 1.5e-04
    dc2[e2p] = -7.571e-04 +/- 2.1e-04
    gap      = -4.795e-04 +/- 2.7e-04   =  -1.8 sigma

Stable under `--drop-top 100 --drop-by Q2` (-1.7 sigma).  **There is no
measured failure to rotate.**  What survives is that `dc2` at `e2p` is real
(-3.4 sigma from zero, an uncorrected difference whose MC floor is 2e-10) and
consistent with an ordinary rotating leak of a few 1e-04 that 20k targets
cannot pin down.

The spin-0 counterpart PASSES outright.  `m1` can depend only on `|e_psf|^2`,
so all three orientations must agree; pairwise, outlier dropped, they do
(-0.8, +0.4, +1.2 sigma), and averaging them gives the bound

    dm1(|e_psf| = 0.2) = -5.6e-03 +/- 3.4e-03    (-1.7 sigma)

### B2: the Fisher identity's 24% violation is entirely outside the window

`sum q1^2 / sum(-R11)`, forced to 1.  Per flux quintile the shape is the same
on all six pqr files: q1 1.22-1.27, q2 0.97-1.00, q3 0.95-0.96, q4 0.92-0.97,
q5 0.98.  But q1's ceiling is `Mf ~ 1040` and q2's ~1435, both below the
`Mf > 2500` cut -- **nothing in q1 or q2 is ever quoted.**  Re-evaluated
INSIDE the window:

    plain 0.965   psfe00 0.940   psfe1p 0.940   psfe2p 0.940   psfe1m 0.893

a standing 3.5-6% deviation, not 24%.  `dev/fisher_split.py` bins by quintile
of the whole catalog and does not apply the window; the in-window number is
the one that is the precondition for trusting an `m1`.

Same run re-confirms 3.1's per-target convergence claim: `v11_psfe00` vs
`v11_psfe00_s7` (the MC null) differ by 1.1e-07 relative on `Q` and 6.1e-07 on
`R`, with `obs` bit-identical.

### A1/A3: everything re-quoted at 2^24

`dev/window_scan.py`.  `plain` and `psfe00` are bit-identical in BOTH `C_M`
and `Sigma_X`, so one pass over the draws serves all three circular-PSF pqr
files.  ~10 min per config, not the 40 estimated.

| config | corrected `m1` at 2^24 | `c1` | `c2` |
|---|---|---|---|
| plain | -0.00352 +/- 0.00966 | -1.93e-03 +/- 3.5e-03 | -5.07e-03 +/- 3.5e-03 |
| psfe00 | +0.01860 +/- 0.00799 | +3.50e-03 +/- 3.5e-03 | +1.84e-03 +/- 3.7e-03 |
| psfe1p | +0.01224 +/- 0.00787 | +3.30e-03 | +1.61e-03 |
| psfe2p | +0.01659 +/- 0.00705 | +3.74e-03 | +1.05e-03 |
| psfe1m (drop 1) | +0.00926 +/- 0.00839 | +3.85e-03 | +1.84e-03 |

Monitors at 2^24: traceless +2.2e-04 to +3.5e-04, `R_s12` +1.0e-03 to
+1.2e-03.  **The v10 headline `-0.00368` at 2^20 was fine** -- the 2^24 value
is -0.00352, a move of 1.2e-04 against the +/-4e-03 that was feared.

**A3 (`c = 0`)**: 1.4 sigma, not the 2 sigma on the record.  But the sample
floor is 3.5e-03, i.e. 3.5x `tau`, so this BOUNDS `c` rather than testing it
at `tau`.  Note `plain` and `psfe00` disagree by ~7e-03 in `c2` -- different
galaxies, and that gap IS the floor being visible.

**`R_s12` is a bigger MC channel than the traceless part, and section 3.1
quotes the wrong one.**  At seed 7, traceless is -5.9e-04 but `R_s12` is
-6.5e-03, 6x larger than at seed 0.  It evidently does not propagate strongly
into `m1` (see the null below), but the traceless part alone understates the
selection-term MC error.

### A2: local window stability -- no drift, but the test does not reach `tau`

Perturb each boundary by +/-5% and +/-10%, one at a time, 13 windows sharing
ONE pass over the draws (`selection_terms_score(windows=)`: the flow's `Q`,
`R` per draw do not depend on the window, only the cheap `window_prob`
quadrature does).  That also makes the scan PAIRED -- every window sees the
identical draw set -- and the re-solve uses one bootstrap index shared across
all windows, so the difference against nominal is paired galaxy for galaxy.
That mattered: a +/-10% move swaps a few hundred galaxies in or out, and the
unpaired +/-0.008 bar cannot tell that from a defect in the correction.

`d m1` per +/-10% of the boundary, with a bar from refitting the line inside
each paired resample:

| boundary | plain, seed 0 | plain, seed 7 | psfe00, seed 0 | psfe00, seed 7 |
|---|---|---|---|---|
| `size_lo` (2.2) | +6.5e-04 +/-3.5e-03 | +8.8e-04 +/-3.5e-03 | +1.67e-03 +/-3.6e-03 | +1.93e-03 +/-3.9e-03 |
| `flux_lo` (2500) | +4.20e-03 +/-5.3e-03 | +4.31e-03 +/-5.4e-03 | +4.84e-03 +/-4.9e-03 | +4.95e-03 +/-5.2e-03 |
| `flux_hi` (50000) | -2.7e-04 +/-1.7e-03 | -2.1e-04 +/-1.9e-03 | +7.0e-04 +/-1.5e-03 | +7.7e-04 +/-1.3e-03 |

**No slope is above 1.0 sigma.**  But every bar is 1.3e-03 to 5.4e-03, i.e.
1.3x to 5.4x `tau`, so by section 1 this has BOUNDED the window systematic at
roughly +/-5e-03, not shown it below `tau`.  Say that, not "it passed".

**The null run says the limit is galaxies, not draws.**  Re-running the whole
scan at draw seed 7 (`dev/phase_a_null.sh`) was not optional: `plain` and
`psfe00` are independent in their GALAXIES but share `Sigma_X` and `C_M` bit
for bit, so they were re-solved against IDENTICAL selection terms and a
term-side error would reproduce across them exactly.  Result:

* nominal corrected `m1` moves 7.3e-04 (plain) and 7.4e-04 (psfe00) between
  the seeds -- **the 7.41e-04 selection MC floor at 2^24 is confirmed to two
  digits, independently**;
* but the SLOPES move by only ~1e-04, far under their own +/-5e-03 bars.

So the slope is not selection-term MC error at all; its bar is pure galaxy
sample variance and falls as `1/sqrt(N)`.  The `flux_lo` central value
(+4.2e-03 to +5.0e-03, same sign in two independent galaxy samples, ~1.25
sigma combined) is the one to watch, and only more targets can resolve it.

**A free finding from the scan.**  At `size_lo -5%` and `-10%` the `R_s12`
monitor jumps to +0.0113 / +0.0131 against +0.0011 at nominal, and the
traceless part to -6.1e-03 / -3.0e-03 against +3.1e-04.  The selection terms
LOSE CONVERGENCE when the size floor is loosened -- a different failure from
window instability, and it explains why only that side of the size scan moves.
It is also the size-edge/support channel showing up in the selection machinery
for free.

### B3: amplitude-scan catalogs rendered, and the channel structure confirmed

`bash dev/render_psfe.sh 0.1` and `0.05`, 24 catalogs.  `render_psfe.sh` had
the missing-`pipefail` bug in another form -- a bare `wait` returns 0 whatever
the jobs did, so `set -e` never fired on a failed render; it now waits on each
pid.

Read off the catalogs alone, no bias run:

| quantity | 0.05 | 0.10 | 0.20 | fitted exponent |
|---|---|---|---|---|
| `Sigma_X` spin-2 split | 2.089e-02 | 4.198e-02 | 8.557e-02 | **+1.017** |
| `C_M` spin-2 split | 5.03e-04 | 2.03e-03 | 8.38e-03 | **+2.030** |

Clean separation, so Phase C2's fitted `d log|dc| / d log e_psf` localises the
channel: near 1 is `Sigma_X`, near 2 is `C_M`.  `e1p` and `e2p` give identical
`|Sigma_X spin-2|` at each amplitude with `e2p` purely off-diagonal -- the
exact 45-degree rotation.

**The scan is 4 runs, not 6.**  With `e_psf = 0` the amplitude does nothing,
and the `e00` catalogs at tags 05/10 are byte-identical in `moments` to tag
20, so `pqr/v11_psfe00.npz` is the baseline for every point.
`dev/bias_amplitude.sh` is written and ready.

### Tooling changes

* `bias.selection_terms_score(..., windows=[...])` -- many windows in one pass
  over the draws.  Returns a list; the single-window signature is unchanged.
* `dev/window_scan.py` -- A1/A2/A3 in one tool.  Caches the selection terms to
  `logs/phase_a/terms_*.npz` (they depend on the flow, `C_M`, `Sigma_X`, the
  window and the draw seed, and on nothing about the targets), and the flow is
  built INSIDE the cache-miss branch so a cached re-solve never touches the
  GPU.  `--drop-top`, `--pqr` takes several files.
* `dev/isotropy_check.py` -- `--rotate-sigma`, `--sigma-x`, `rot_sigma`.
* `dev/psfe_compare.py` -- `--drop-top`/`--drop-by`, and the rotation test with
  a PAIRED bar on the gap.
* `dev/phase_a.sh`, `dev/phase_a_null.sh`, `dev/bias_amplitude.sh`.
* `bias.CATALOGS`/`TRAIN_DATA` -- the four amplitude-scan configs.

### Traps paid for this session

* **Two JAX processes do not fit on the 16 GB card at 2^24.**  A concurrent
  `pytest` run killed a `window_scan` with `RESOURCE_EXHAUSTED ... 9 alive
  graphs`.  Both drivers now serialise, and cached re-solves avoid the GPU.
* **`pgrep -f "dev/phase_a.sh"` in a waiter matches the waiter's OWN command
  line** -- `[[background-hang-is-pgrep-selfmatch]]` again, and it cost 2.5
  hours of an idle GPU: the queued null run never started.  Guard on the
  PYTHON process, or on a marker string in the log, never on the script name.
* A bare `wait` in a shell script masks failed background jobs exactly the way
  a missing `pipefail` masks a failed pipeline.

## 2026-09-05 (cont.): Phase C2 -- the leak is the `Sigma_X` channel, and the
three-point scan would have said "artifact"

Four `bias.py` runs at `e_psf` 0.05 and 0.10 (`dev/bias_amplitude.sh`), then a
fifth at 0.02 added after the first fit came back wrong-looking.  All against
`pqr/v11_psfe00.npz`, which is the baseline for every amplitude because the
`e00` catalogs are byte-identical across tags.

### The result

`dc2`, `e2p` orientation, paired, `--drop-top 3`:

| `e_psf` | `dc2` | +/- paired | sigma |
|---|---|---|---|
| 0.02 | -1.921e-05 | 4.6e-05 | -0.4 |
| 0.05 | -5.118e-04 | 1.6e-04 | -3.2 |
| 0.10 | -5.773e-04 | 2.1e-04 | -2.7 |
| 0.20 | -7.577e-04 | 2.2e-04 | -3.4 |

`dc = A e^p` gives **p = 1.07 (1-sigma 0.85-1.34)**.  At fixed `p`, chi2 over
3 dof: constant 23.8, **linear 6.24**, quadratic 11.4.

* **no leak / `e_psf`-independent artifact (p = 0): excluded at 4.2 sigma**
* **`Sigma_X` (linear; the split runs as `e^1.017`): this is the channel**
* `C_M` (quadratic, `e^2.030`): disfavoured at 2.3 sigma

`dm1` is consistent with zero at every amplitude (all under 1 sigma), so the
spin-0 channel is bounded but not detected.

### The methodological point, which is the more valuable half

**The 0.05/0.10/0.20 scan ALONE gives p = 0.29 and reads as an artifact.**
That was the first conclusion off this data and it was wrong.  Leave-one-out on
the free-`p` fit:

| dropped | p | 1-sigma |
|---|---|---|
| e = 0.02 | **0.29** | 0.00-0.61 |
| e = 0.05 | 1.27 | 0.97-1.78 |
| e = 0.10 | 1.10 | 0.86-1.45 |
| e = 0.20 | 1.52 | 1.14-2.04 |

Three points spanning one decade at ~30% errors do not constrain an exponent.
A power law needs an anchor near zero, and the anchor is CHEAP here for a
reason worth remembering: the paired bar shrinks with the perturbation (4.6e-05
at 0.02 against 2.2e-04 at 0.20) because the two catalogs converge to identical
as `e_psf -> 0`.  One 40-minute run at small amplitude outweighed the three
that preceded it.

Both sanity checks that make the scan interpretable were run first and passed:

* the arms are genuinely paired -- target `|dM2|` p50 against the `e = 0`
  baseline is linear in `e_psf` to 5% (95.2, 95.5, 96.7, 100.6 per unit `e`),
  so the noise field is shared and the input really is a clean linear
  perturbation;
* the sim has no discrete switch -- `psf_cov` and galsim's `.shear(e1, e2)` are
  smooth in `e`, so "something turns on when `e_psf != 0`" had no mechanism.

### Bound at DES-like ellipticity

Directly measured at `e_psf = 0.05`: `dc2 = -5.1e-04 +/- 1.6e-04`.  The fitted
linear law would put it at -2.0e-04.  Either way below `tau = 1e-3` -- but it
is a 3.2-sigma DETECTION, not a null, and `chi2/dof = 2.08` for the linear fit
(the 0.05 point sits 2 sigma above it), so a sub-leading second component is
not excluded.

### What this closes

Section 5 axis 1's channel question, WITHOUT the no-centroid bisect: the
ellipticity reaches the estimator through `Sigma_X`, i.e. through the centroid
layer's conditioning, not through the noise covariance.  Taken with
[[c01-path-is-clean]] (the flow is exactly equivariant in `C01`) and
[[dc2-rotation-gap-is-1p8-sigma]] (the signal does rotate, 1.8 sigma), the
whole "mysterious non-rotating `e2p` anomaly" is resolved into an ordinary
linear `Sigma_X` leak of a few 1e-04.

### Tooling

* `dev/psfe_compare.py` -- `amplitude()`, and the amplitude-scan fit.  It fits
  `dc = A e^p` in the SIGNED variable, scanning `p` and least-squares-solving
  the single linear `A` at each step weighted by the paired bars.  A `log|dc|`
  fit was rejected: it discards the sign, cannot use a point consistent with
  zero (which is the most constraining one here), and is biased by the bar.
* `catalog_rows` now takes its path from `bias.CATALOGS` instead of rebuilding
  it with a hard-coded `20` tag -- verified a no-op on the original triplet.
* `bias.CATALOGS`/`TRAIN_DATA` -- `psfe2p02` and the four 05/10 configs.
