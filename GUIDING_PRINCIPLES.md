# Guiding principles

What the flow method has to achieve, what counts as evidence for it, and in
what order to spend effort.  This is the standing reference: when a decision
about what to build, measure or believe comes up, it is settled here rather
than re-argued.  `HANDOFF.md` records what happened; this records what we are
trying to do.

Written 2026-09-04, at the end of the fixed-elliptical-PSF study.
Updated 2026-09-05 with the Phase A/B measurements (sections 2, 3.1, 3.4, 4, 5, 6).

---

## 1. The claim, stated precisely

> For **isolated, unblended galaxies**, the flow-prior BFD estimator recovers
> shear with `|m| < tau` and `|c| < tau_c`, at stated confidence, under stated
> observing conditions.

Three things about that sentence govern everything below.

**"Unbiased" is not provable.**  It is a BOUND at a precision.  Every result is
quoted as a value with an error, and a result whose error exceeds `tau` has not
tested the claim regardless of where its central value sits.  We do not have
the option of reporting "consistent with zero" from a measurement that could
not have detected the effect.

**`tau` comes from the survey, not from us.**  ~2e-3 is DES-Y3-like; ~1e-3 is
LSST-like.  Pick one before measuring, not after.  Everything in section 3 is
a consequence of the number chosen.

**"Under stated conditions" is part of the claim, not a disclaimer.**  A bound
established at a fixed circular PSF on one population says nothing about a
varying PSF, and saying so is not hedging -- it is the result.

### Scope: isolated and unblended

This is a real narrowing and it should be used.  Out of scope, permanently
unless the scope changes: deblending systematics, neighbour-induced selection,
neighbour light in the stamp.  What remains in scope and must NOT be quietly
inherited into "isolated": detection/selection effects on isolated objects
(the flux and size window is exactly this and it is fully in scope), the PSF,
per-target noise, and the centroid.

---

## 2. Where we stand (2026-09-04, `flows/centroid_v11.eqx`)

| quantity | value | against `tau = 1e-3` |
|---|---|---|
| unwindowed `m1` | -0.0134 +/- 0.0068 | value 13x, error 7x |
| windowed corrected `m1`, 2^24 | -0.0035 +/- 0.0097 | error 10x |
| same, independent 20k draw | +0.0186 +/- 0.0080 | error 8x |
| windowed `c1`, `c2`, 2^24 | -1.9e-03, -5.1e-03 (+/- 3.5e-03) | ~1.4 sigma from a forced zero |
| window systematic, per +/-10% of a cut | bounded at ~+/-5e-03 | error 5x |

(Updated 2026-09-05.  The corrected numbers are now at 2^24 draws; the 2^20
values they replace were -0.0036 and +0.0179, so the feared +/-4e-03 never
materialised on the central value.)

**The binding constraint today is precision, not the central value.**  At 20k
targets the statistical error alone is 5-10x `tau`.  We cannot presently
distinguish `m1 = 0` from `m1 = 0.01`, so no amount of argument about the
central value is worth anything until section 3.1 is done.

---

## 3. What the claim requires

### 3.1 A noise floor below `tau`, DEMONSTRATED to integrate down

Three independent terms.  Every quoted bar must contain all three, and each
must be shown to fall with the resource that is supposed to control it.  A
floor that stops falling is a systematic wearing a statistical disguise, and
finding that out early is worth more than any single measurement.

| term | controlled by | status |
|---|---|---|
| galaxy sample variance | `N` targets, as `1/sqrt(N)` | MEASURED: +/-0.008 at 20k (bootstrap, block-jackknife and 20-subsample agree). Implies ~1.3M targets for 1e-3. |
| per-target MC draw noise | `S` draws/target, as `1/sqrt(S)` | MEASURED and NEGLIGIBLE at `S = 8192`: a same-data seed change moves windowed uncorrected `m1` by 1.0e-08 and `c1`, `c2` by ~2e-10, even though per-target `Q`, `R` move by up to 6e-04 / 7e-03.  A galaxy bootstrap IS therefore the complete bar for an uncorrected number. |
| selection-term MC error | `--window-draws`, as `1/sqrt(N)` | **DOMINANT.  Use 2^24, never the 2^20 default.**  Measured ladder (`dev/selection_convergence.py`, seeds 0 vs 7): corrected `m1` spread 4.07e-03 at 2^20 -> 7.41e-04 at 2^24; `R_s` traceless 1.36e-02 -> 5.94e-04.  Scales as `1/sqrt(N)`.  Every windowed corrected number taken at 2^20 carries an unquoted +/-4e-03. |

**The selection terms are the error budget.**  Measured 2026-09-04 and it
reverses the intuition: the per-target integral is converged to 10 decimal
places while the correction that is supposed to remove the window's effect
carries 4e-03 on `m1`.  Consequences, which apply to every windowed corrected
number this project has produced:

* an UNCORRECTED windowed difference is trustworthy at ~1e-10 and its galaxy
  bootstrap is complete;
* a CORRECTED one carries ~4e-03 on `m1` and ~1e-03 on `c2` that no galaxy
  bootstrap can see, and that must be quoted;
* `R_s11 = R_s22` AND `R_s12 = 0` (section 4) are the free convergence
  monitors -- both forced exactly, so their departures ARE the selection-term
  MC error.  Read them before believing any corrected number.  **Read BOTH:
  measured 2026-09-05, `R_s12` is the larger channel and the traceless part
  alone understates the error** -- at 2^24 seed 7 the traceless part is
  -5.9e-04 while `R_s12` is -6.5e-03.  (`R_s12` does not appear to propagate
  strongly into `m1`: a seed change moved corrected `m1` by only 7.4e-04.);
* **`--window-draws 16777216` (2^24) is the standing default for anything
  quoted.**  Measured 2026-09-04: it puts the selection MC error at 7.4e-04 on
  `m1`, i.e. below `tau = 1e-3`, and the `R_s` traceless monitor at 5.9e-04.
  The terms depend only on the flow, `C_M`, `Sigma_X` and the window -- never
  on the targets -- so they cost no target work and re-solve `ghat` offline
  (`dev/selection_convergence.py`, minutes; `ghat` is affine in them).  There
  is no reason to ever quote a 2^20 number again.
* the estimator does NOT degrade with draws, unlike the pathwise one: the top
  draw's share of `R_s11` falls 3.8% -> 0.5-0.9% from 2^20 to 2^24.

Rule: **a paired difference is only as good as its unpaired terms.**  Pairing
(same galaxies, same noise field) correctly cancels the population scatter and
that is why the configs share a seed -- but it cancels nothing that differs
between the two runs, and the MC draws differ whenever the targets' moments do.

### 3.2 Estimator error and model error separated

Two different claims, two different tests, and conflating them has repeatedly
cost time.

* **Estimator**: does BFD-with-a-flow-prior solve the right equation?  Tested
  against `gauss2`, the analytic population where `P(m|g)`, `Q` and `R` are
  known in closed form.  Currently in good shape -- the exact-split control is
  clean to 1.7e-4.
* **Model**: does the trained flow reproduce the population's density AND its
  shear response?  This is where every residual bias has actually lived.

**A bulgedisc number is not interpretable until the same measurement passes on
gauss2 at the same `tau`.**  gauss2 currently has its own ~5e-3 floor,
attributed to the centroid spin-2 undershoot.  That closes first.

### 3.3 The known open channels closed or BOUNDED

Each of these is at or above `tau`.  "Not currently visible" is not a bound; a
bound is a number with an error.

* the conditional-support leak at the size edge
* the flux-gradient response error
* the centroid ellipticity response ratio: 1.0431, where symmetry gives 1.0
* the gauss2 ~5e-3 floor (3.2)

### 3.4 Window stability -- LOCAL, not general

`m1` depends on the window it is quoted in, which is why no `m1` is quotable
without one.  Removing that dependence is the entire job of the selection
correction, presently validated only to ~0.005 on gauss2 (5x too loose, and
predating the 2^24 draw count).

**The requirement is local stability, not general window-independence, and the
difference matters.**  The window is not a free choice.  Its boundaries exist
to exclude regions governed by systematics this estimator does not model:
PSF-model error near the resolution floor, background subtraction and S/N at
the faint end, saturation and non-linearity at the bright end.  A correction
that failed on some far-away window would not be refuted by that failure --
that window is unavailable in a real analysis for independent reasons.
Demanding generality here would also collide with the known size-edge/support
leak (3.3), which is a separate channel and must not be misattributed to the
selection machinery.

What is genuinely required: **the answer must not depend on where exactly a
boundary sits, given that the systematics motivating it fix its location only
to some tolerance.**  Test by perturbing each boundary by ~5-10% about its
nominal value, one at a time, and confirming corrected `m1` is flat within the
selection MC floor (7.4e-04 at 2^24).

Quote the SLOPE `d m1 / d(boundary)`, not just the scatter.  A trend rather
than scatter is the signature of a correction that is slightly wrong, and the
slope is what converts the analysis's own uncertainty in the cut location into
a systematic on `m1` -- which is the number a real analysis actually needs.

Cheap: no target work, only the selection terms per window -- and not even one
pass per window, because the flow's `Q`, `R` per draw do not depend on the
window at all (`selection_terms_score(windows=[...])`, `dev/window_scan.py`).

**MEASURED 2026-09-05, and the pass condition above cannot be met at 20k
targets.**  No slope exceeds 1.0 sigma, but every bar is 1.3x to 5.4x `tau`:
`size_lo +6.5e-04 +/- 3.5e-03`, `flux_lo +4.2e-03 +/- 5.3e-03`,
`flux_hi -2.7e-04 +/- 1.7e-03` per +/-10% of the boundary.  By section 1 that
BOUNDS the window systematic at ~+/-5e-03; it does not show it below `tau`.
The seed-7 null settles where the limit is: the slopes move by ~1e-04 between
draw seeds while nominal `m1` moves 7.4e-04, so the slope carries no
selection-term MC error and its bar is pure galaxy sample variance.  **This
test finishes at 1M targets, not at more draws** -- which reorders section 6:
the window scan is no longer cheap-and-pending, it is a rider on 6.4.

Two riders on how to run it.  Pair the windows -- one bootstrap index shared
across all of them -- or a boundary move looks like a defect when it is only
the few hundred galaxies it swapped in or out.  And do not read the loosened
size floor as a window result: at `size_lo -5%`/`-10%` the `R_s12` monitor
jumps 10x, i.e. the selection terms themselves stop converging there, which is
the size-edge channel of 3.3 and not the selection machinery.

---

## 4. The free exactness tests -- use them first, always

These follow from symmetry, cost no known truth and no extra rendering, and
each one has caught a real bug.  Prefer them over any measurement that needs a
reference value.

| test | why it is exact | current |
|---|---|---|
| `c1 = c2 = 0` | isotropic population + circular PSF is rotation-symmetric, so `ghat` at `g = 0` vanishes in expectation | -1.9e-03, -5.1e-03 (+/- 3.5e-03), 1.4 sigma |
| `R_s11 = R_s22`, `R_s12 = 0` | the window cuts only spin-0 quantities, so `P_s` depends on `\|g\|^2` alone | at 2^24: traceless +3.1e-04, `R_s12` +1.1e-03 on `R_s11 ~ -0.168`.  `R_s12` is the larger channel -- read both |
| `Q_s = 0` | same isotropy | `Q_s2 = -4.4e-03` vs err 2.0e-03; shown to be chance by Rao-Blackwellising over rotations |
| Fisher identity `sum q^2 / sum(-r) = 1` | score identity | 0.94-0.965 IN-WINDOW on every v11 config.  The 1.24 at the faintest quintile is entirely below the `Mf > 2500` cut and is not quoted |
| centroid-layer rotation equivariance | the transport is built from spin contractions | at the float32 floor (8.6e-07) since the `chart()` fix |
| latent moments independent of PSF ellipticity | BFD deconvolves what it convolved | bit-identical, gate G0 |

`c = 0` deserves emphasis: it is the cheapest absolute test available, it needs
no model of anything, and its only floor is the finite sample's own quadrupole.
Drive `c` to `tau` alongside `m`.

---

## 5. Conditions, and how they widen

Any claim today reads: *fixed circular PSF, one population, isolated galaxies,
one shared `C_M` and `Sigma_X` across targets.*  Widen deliberately, one axis
at a time, each with its own measurement:

1. **fixed elliptical PSF** -- CHANNEL IDENTIFIED, bound below `tau`
   (2026-09-05).  Structurally: the moments are PSF-corrected, so the
   ellipticity reaches the estimator only through `C_M` and `Sigma_X` (latent
   moments bit-identical at `e_psf = 0.2`, gate G0), and no retraining or chart
   change is needed.  Three things that were open here are now closed, and none
   of them the way they were expected:

   * **the `C01` hypothesis is falsified.**  Rotating the prior draws AND
     `Sigma_X` together (`Sigma_X -> R Sigma_X R^T`, spin-1) through the full
     chained flow at the `e2p` off-diagonal is equivariant to 7.1e-15 (p50) on
     `log P` -- the float64 floor -- while the control shows the path is
     genuinely exercised.
   * **"the signal does not rotate" was an artifact of comparing two
     correlated numbers by their separate bars.**  The PAIRED gap is
     `-4.8e-04 +/- 2.7e-04`, 1.8 sigma.  The signal rotates.
   * **the channel is `Sigma_X`, measured by the amplitude scan.**  `dc2` runs
     as `e_psf^1.07` (1-sigma 0.85-1.34) over four amplitudes; at fixed `p`,
     chi2 over 3 dof is 23.8 (constant), **6.24 (linear)**, 11.4 (quadratic).
     So an `e_psf`-independent artifact is excluded at 4.2 sigma, `C_M` at 2.3.
     The no-centroid bisect is not needed.

   The bound at DES-like ellipticity, measured directly rather than
   extrapolated:

       dc2(e_psf = 0.05) = -5.1e-04 +/- 1.6e-04    (3.2 sigma, below tau)
       dm1               consistent with zero at every amplitude

   **This axis is a DETECTION below `tau`, not a null.**  `chi2/dof = 2.08` for
   the linear fit, so a sub-leading second component is not excluded.  What
   would close it fully: the same scan on `gauss2`, where the leak's size can
   be compared against a known answer rather than only against zero.

   One methodological result from this, worth more than the number: **the
   0.05/0.10/0.20 scan ALONE gives `p = 0.29` and reads as an artifact.**  A
   power law needs an anchor near zero -- three points over one decade at 30%
   errors do not constrain an exponent.  Drop the `e = 0.02` point and the
   fitted exponent inverts; drop any other and it moves by 0.4.  The anchor is
   also the CHEAPEST point, because the paired bar shrinks with the
   perturbation (4.6e-05 at 0.02 against 2.2e-04 at 0.20) as the two catalogs
   converge to identical.

2. **varying PSF across the survey** -- the real-data case.  Requires per-target
   `C_M`/`Sigma_X` plumbing that does not exist, and probably `Sigma_X`-sampled
   training.  This is the long pole.
3. **varying depth / heteroscedastic `C_M`** -- same plumbing.
4. **other populations** -- the flow must not be tuned to bulgedisc.

Out of scope by the stated restriction: blending, neighbours, deblending.

---

## 6. Order of work

Deliberate, and the reverse order wastes the most time.

1. **Noise floor (3.1).**  Cheap, and it tells you whether anything else is
   measurable.  Chasing a bias under the noise is the most expensive mistake
   available here.  DONE for the selection term (2^24, confirmed twice); the
   remaining term is galaxy sample variance, i.e. item 4.
2. **gauss2 to `tau` (3.2).**  Separates estimator from model.
3. **The open channels (3.3).**
4. **bulgedisc at 1M+ targets.**  This now also carries 3.4: the window scan's
   slopes are galaxy-limited, not draw-limited, so they finish here and
   nowhere else.
5. **Widen conditions (5).**

---

## 7. Rules of evidence

Adopted because each one is a mistake already made and paid for.

* **Quote no `m1` without its window.**
* **Difference against a baseline sharing the seed.**  Absolute `m1` at 20k has
  a +/-0.008 draw-to-draw scatter -- twice the effects being chased.  Two
  independent 20k draws of the same population gave +0.018 and -0.004.
* **Percentiles, never the max.**  Tails carry a handful of ill-conditioned
  draws that carry no weight.
* **Check concentration before believing a number.**  One target holding 7.9%
  of `sum|R11|` produced an `m1` of -0.11 in this very session.  Report the
  top-k share; if the answer moves when the top target is dropped, it is not a
  measurement.
* **A null run is part of the experiment, not an optional extra.**  Same data,
  different Monte Carlo seed, differenced the same way as the real comparison.
* **An identical rebuild moves noisy `m1` by sd 0.0008.**  Do not read a 0.001
  difference as a result.
* **A negative result is a result** and gets recorded with the same care as a
  positive one -- see the NEGATIVE/NULL entries in the memory index, several of
  which stopped a rebuild of something that would not have worked.
* **Provenance is a correctness property.**  Assert it (`dev/check_provenance.py`)
  rather than trusting file names; the silent-staleness class of bug has cost
  more here than any modelling error.

---

## 8. What would make the claim, in one paragraph

A bounded `|m1|` and `|c|` below `tau`, on 1M+ isolated galaxies, with an error
budget whose three terms are each shown to integrate down, reproduced across at
least two windows and confirmed on `gauss2` where the truth is known, with the
free symmetry tests of section 4 passing at the same precision, under conditions
stated explicitly and widened one axis at a time.  We are roughly an order of
magnitude short in precision and have three or four known channels at or above
`tau` still open.  Neither of those is evidence that the method is biased --
the estimator-level controls are clean -- but both are reasons the claim cannot
be made today.
