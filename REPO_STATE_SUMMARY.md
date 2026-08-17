# Repo state summary — bfd_cnf, 2026-08-17

Branch `clean-start` (branched from `main`, ground-up rebuild — the old
`working` branch's machinery like `ShearTaylorLast` is intentionally absent).
5 commits on top of the initial import.

## What this project is

Fitting a normalizing flow to the population density of BFD (Bayesian Fourier
Domain) even moments `[Mf, Mr, M1, M2, Mc]`, then using it as the prior in a
BFD shear estimator (`bias.py`) to measure multiplicative/additive shear bias
(`m1`/`m2`/`c1`/`c2`) against known injected shear, on simulated galaxy
populations (`gauss2`, analytically tractable; `bulgedisc`, realistic
morphology). The whole project is chasing residual bias down to the
sub-percent level.

## Current uncommitted working-tree changes

One coherent unit of work: **a second hard chart ceiling, on `Mc/Mr`**, plus
the two real bugs found while deploying it. Not yet committed.

- `models/bijections.py` — adds `POINT_SOURCE_MC = 6.662089` (the point-source
  limit of `Mc/Mr`, exact analogue of the existing `POINT_SOURCE` limit of
  `Mr/Mf`). Chart slot 2 changes from a bare ratio `Mc/Mr` to
  `logit(Mc/(POINT_SOURCE_MC·Mr))`, closing an unmodelled interior cliff that
  bulgedisc piles against (skew -2.02). `in_domain` now also requires `Mc > 0`
  (was missing — a non-positive `Mc` produces NaN, not -inf, so no downstream
  mask survived it). `safe_point` (moved here from `bias.py`) now clamps `Mc`
  **after** `Mr`, because clamping `Mr` down raises `Mc/Mr` and a row that only
  violated slot 1 came out of the fix violating slot 2 — this exact bug cost
  18.9% of noisy targets before being caught.
- `bias.py` — imports `safe_point`/`in_domain` from `models/bijections.py`
  instead of defining a local (now-stale) copy; new `--noise-scale` flag to
  rescale `C_M` in moment space as a cheap stand-in for a deeper catalog
  (raises `SystemExit` if combined with an image-noise catalog, since that
  would rescale only the kernel, not the target's own noise realization).
- `bulk.py`, `truth.py` — mirror the slot-2 logit in their standalone
  `to_coords` (plotting / ground-truth paths must not drift from the flow's
  own transform).
- `centroid.py` — the centroid-layer training loss now masks the same way
  `bias.py`'s noisy path does: substitutes an in-domain dummy point (rather
  than masking the *result*) for any copy the layer maps off-chart, because
  `jnp.where` still differentiates the discarded branch and a NaN there
  poisoned an entire 8000-step run at step 144. Logs the off-chart fraction
  per step.
- `tests/test_bounded_chart.py`, `tests/test_conv.py` — new tests for the
  slot-2 ceiling and the `Mc>0` guard; `test_conv.py`'s fixture point moved
  because the old one (`Mc/Mr=6.818`) sat above the new 6.662 ceiling, which
  had been silently turning "unmasked integral" comparisons into masked ones.

**LANDING HAZARD called out in HANDOFF.md**: this chart change invalidates
every flow in `flows/` (old flows deserialize their old mean/std, then get run
through the new transform → silently wrong, no error). All flows need
retraining; `retrain_mcbound.sh` (untracked, new) does this.

Net effect measured before commit (bulgedisc noiseless 1M): error bars on m1
shrink 34× overall, 79× in the diffuse window, by removing a handful of
targets whose R was blowing up to O(1e10) near the old ratio's skewed tail.

### Untracked files
- `NOISY_BINNING_BIAS.md` — a finished, standalone writeup (not a scratch
  note) of a major finding: **binning a noisy m1 profile on the target's true
  (latent) `Mr/Mf` biases it even with a perfect density**, because the
  estimator only sees noisy `M`. Comes with a validated control (draw targets
  from the flow itself, so the prior is exact by construction) and the
  conclusion that essentially all of the project's "resolution ramp" /
  "broad hump" / edge-collapse structure is this artifact, not a flow error.
  Not yet linked from anywhere else in the tree.
- `dev/check_*.py` (8 scripts) — diagnostic tools built during the Mc/Mr-bound
  and centroid-bias investigation: dead-target attribution, ESS/log-det
  honesty, float32 precision audit, morphology attribution, centroid-layer
  fold detection, safe_point mask-seam check, self-consistency control, noisy
  size-profile scoring.
- `dev/refine_prior.py` — fits the flow's bulk density directly against noisy
  targets through the noise kernel (no truth/simulation needed), validated to
  remove an injected density error on gauss2.
- `retrain_mcbound.sh` — retrains all 8 canonical flows under the new chart.
- Two PNGs (`dev/bias_flux_size_mcb.png`, `dev/hexbin_corrected.png`) and
  their generating scripts.

## Recent commit history (5 commits since initial import)

1. `9d7e527` — m/c estimation on targets, noiseless or C_M-integrated
2. `50db52b` — Phase 3: centroid marginalisation as a data-adjacent layer
3. `e4f3ff4` — shared proposal flow so alpha<1 runs stay paired
4. `8b8d5cf` — jit the proposal, force full-precision matmuls (TF32 was
   silently injecting ~1 nat of error into log-densities)
5. `2f817cd` — "wip: before Mc/Mr ceiling handoff" (current HEAD)

## What HANDOFF.md documents (68KB, the real project log)

A long, dated investigation log (2026-08-11 → 2026-08-15) tracking down a
resolution-axis (`Mr/Mf`) structure in `m1` after fixing the window function.
Headline findings, roughly in the order they were nailed down:

- The window-function fix flattened the *slope* of m1 vs Mr/Mf but that was a
  **cancellation**: a real multi-lobe oscillation survives fine binning and a
  flux-split phase-lock test, ruling out spline-knot ripple and
  coordinate-warping as causes.
- Both `gauss2` and `bulgedisc` show it; the boundary that matters is each
  population's own support edge in the **concentration** direction (bulge
  ratio / point-source degeneracy), not the chart's `Mr/Mf` ceiling.
- The Mc/Mr chart bound (the change staged in the working tree above) fixes
  a real numerical pathology (R blowing up near the old ratio's skew) but is
  null on the oscillation itself.
- **Three real bugs found and fixed**, each the same shape (a mask applied at
  the wrong point in the pipeline): `safe_point` not knowing about the second
  ceiling (cost 18.9% of noisy targets), `in_domain` missing `Mc>0` (NaN, not
  -inf), and `_mixture_chunk` masking the *raw* draw when the flow it
  evaluates includes the centroid layer (found harmless in practice — the
  affected draws carry ~0 weight — but fixed on principle).
- **`NOISY_BINNING_BIAS.md`'s finding** (see above) explains most of what
  remained: after subtracting a self-consistency control, essentially every
  noisy-profile bin lands within ~2σ of zero.
- A separate deep-noise (`Sigma_X` ladder) investigation found a persistent
  **+0.006–0.007 multiplicative bias specific to the centroid-marginalization
  path**, non-monotone in `Sigma_X` (zero at both `Sigma_X=0` and `3×`, peaks
  near the operating point), converged in draw count, flat in `g`, and
  survived elimination of every plumbing suspect (log-det, mask seams, merge
  arithmetic, layer fold, float32, PQR truncation). **Unresolved** — this is
  the most concrete open thread in the log.
- General project rule reaffirmed several times: **draw count / ESS
  convergence is depth-dependent** — 8192 draws is fine at shallow noise but
  under-converged at `Sigma_X≈2.73×`; several earlier "residual bias" numbers
  turned out to be half or entirely an unconverged-integral artifact.

## Environment issue found while checking this out

`python -m pytest` currently **fails at collection/runtime** with:
```
jax.errors.JaxRuntimeError: NOT_FOUND: No FFI handler registered for
cusolver_getrf_ffi on a platform CUDA (canonical cuda)
```
Cause: the installed `jax_cuda13_plugin` (0.11.0) is incompatible with the
installed `jaxlib` (0.10.2) and gets silently disabled, then a CUDA linear-solve
op has no CPU fallback registered. This breaks `tests/test_truth.py`,
`tests/test_shear_perturbative.py`, and 9/15 tests in
`test_bounded_chart.py`/`test_conv.py` (the 6 that pass don't touch a linear
solve). **This is an environment/dependency-version problem, not a code
regression from the staged diff** — but it currently blocks running the test
suite to validate the Mc/Mr-bound changes before committing. Fixing the
jaxlib/jax-cuda-plugin version pin is a prerequisite for the next real step
(retrain and validate under the new chart).

## Suggested next steps (from the trail already laid down)

1. Fix the jaxlib/CUDA-plugin mismatch so the full test suite can run.
2. Run `retrain_mcbound.sh`, then the full test suite, then commit the staged
   Mc/Mr-ceiling change (HANDOFF.md's own "before Mc/Mr ceiling handoff" wip
   commit is exactly waiting on this).
3. Decide what to do with `NOISY_BINNING_BIAS.md` and the `dev/check_*.py`
   tools — they're finished, validated work sitting untracked.
4. Pick back up the unresolved centroid +0.006–0.007 response-scale bias
   (HANDOFF.md's last open thread) — every plumbing explanation is now ruled
   out; what's left points at the model/density itself.
