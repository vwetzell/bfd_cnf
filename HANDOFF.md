# Handoff: the resolution-axis structure in m1, post window-fix

Written 2026-08-12. The previous handoff (regenerate everything under the
corrected `KBlackmanHarris`/`POINT_SOURCE` weight) is **done** — catalogs
regenerated, flows retrained, tests passing. This one picks up the
measurement that motivated it: what does m1 look like along the resolution
axis now, and it turns out the answer is more interesting than "the ramp is
gone."

---

## Where things stand

**1. The window fix worked, but didn't finish the job.** Two independent
checks against the retrained `flows/shear.eqx` / `flows/shear_gauss2.eqx`:

- `dev/sweep_ramp_cut.py --eval-only flows/shear.eqx`: within Mr/Mf<3.5, the
  linear-fit slope of m1 vs Mr/Mf went from a consistent **+0.009 to +0.014**
  pre-fix to **-0.0005** post-fix. RMS(m1) and peak-to-peak (0.012, 0.032)
  are essentially unchanged from before. **Do not read "slope ~ 0" as "bias
  gone"** — see next point.
- `dev.check_flow_vs_truth --flow flows/shear_gauss2.eqx --pop gauss2`: flow
  m1 roughly halved (-1.77e-2 -> -9.2e-3), density scatter and response error
  both dropped 2-3x in the bulk. The point-source-edge failure (density and
  response both wrong near the ceiling) is smaller but **did not go away** —
  it just moved from the old stale ceiling (3.976) to the new correct one
  (3.6926).

**2. The "flattened slope" was a cancellation, not an improvement.** A 2D
hexbin of m1 in the (log Mf, Mr/Mf) plane (`dev/plot_bias_flux_size.py`)
showed a coherent flux-independent band structure, not noise. Following up
at 1D resolution (`dev/check_size_oscillation.py`) confirmed it's real:

- 60-bin quantile profile with bootstrap errors (~5e-4/bin, 1M targets): a
  smooth, low-frequency curve, only 4/59 bin-to-bin transitions are
  noise-level sign flips.
- **Phase-lock test**: split the population at the flux median, profile each
  half independently. The two curves overlay almost exactly in Mr/Mf — same
  peak, trough, and bump locations. A binning artifact or flux-size
  interaction would not reproduce the same shape at the same location in two
  disjoint flux samples.
- Equal-**width** bins (as opposed to equal-count quantile bins, which are
  coordinate-invariant and can't test this) reveal the full-range shape:
  m1 = +0.02 near the most compact galaxies (Mr/Mf~1.2-1.5), declines
  smoothly through zero around Mr/Mf 2.0-2.7, rises to a peak (+0.023 at
  Mr/Mf~3.15), dips to a trough (-0.017 to -0.038 at ~3.37), a small bump
  (+0.005 at ~3.43), then joins the known collapse toward the ceiling. This
  runs across the **whole** resolution range, not just the point-source edge.

**3. Two mechanisms ruled out:**

- **Spline-knot ripple** — the classic cause of a coherent multi-lobe
  residual. Ruled out: `models/bijections.py`'s flow layers
  (`EquivariantAutoregressiveLayer` = `Spin0AutoregressiveLayer` +
  `Spin2CouplingLayer` + `ExplicitPolyLast`) are affine/polynomial
  autoregressive transforms. No rational-quadratic-spline transformer
  anywhere in this codebase, so nothing imprints a discrete period.
- **Coordinate-warping illusion** (the shape is smooth in the flow's own
  chart, only looks wiggly in raw Mr/Mf) — ruled out by re-running the
  equal-width profile against `logit(Mr/(POINT_SOURCE*Mf))`, the actual
  coordinate `RawMomentStandardize` hands the flow. The multi-lobe shape
  survives with comparable complexity in both coordinates; logit just
  stretches the near-edge lobe over a wider span (it diverges as
  Mr/Mf -> ceiling) instead of tidying it up.

**Conclusion so far**: this is a genuine multi-mode error in what the flow
learned along the resolution direction, present across the full range, not a
plumbing or binning artifact. Corrected memory
[[resolution-edge-bias-is-real]], which had (pre-fix) diagnosed a similar
mid-range wiggle as pure noisy-binning artifact — that conclusion doesn't
hold post-fix; this profile already uses the noise-free control that old
diagnosis itself required.

---

## Tooling built this session (all in `dev/`, reusable)

- **`dev/plot_bias_flux_size.py`** — hexbin of m1 (and m1/bootstrap-sd) in
  the (log Mf, Mr/Mf) plane. Noiseless path: exact `dm_dg`/`d2m_dg2` lensing
  of `targets_g0_1M.fits` to g1=+/-0.02, so pairing is exact and bin
  assignment uses the **unsheared** moments — a hexagon boundary can't leak
  into a selection term (same reasoning as `dev/check_bias_by_selection.py`).
  Caches per-target PQR to `dev/bias_flux_size_cache.npz` (~63 MB, gitignore
  it or delete when done) so follow-up analysis doesn't re-run the flow.
  Colorbars currently clamped to m1 in [-0.02, 0.02], significance in [-3,3].
- **`dev/check_size_oscillation.py`** — 1D profile of m1 vs a size
  coordinate, marginalised over flux, reading the cache above (fast, no flow
  eval). `--coord raw|logit` picks the binning variable; `--width` switches
  from equal-count quantile bins (coordinate-invariant, mostly useless for
  testing reparametrizations) to equal-width bins (the real test). Also runs
  the flux-split phase-lock check and prints a bin-to-bin sign-flip count.
  Drops bins with bootstrap sd > 0.01 from the plot (a handful of low-Mr/Mf
  bins have near-singular R and blow up to O(1) — not signal, just a few
  outlier targets breaking the 2x2 solve; harmless in the printed table,
  filtered from the figure).

Both scripts default to `flows/shear.eqx` / bulgedisc. Re-running end to end
(no cache) takes ~5-8 min, dominated by two `bias.pqr` passes (grad+hessian)
over 1M targets.

---

## 2026-08-13: open question 3 answered, and it relocated the boundary

**gauss2 is not clean.** `dev/check_size_oscillation.py --cache
dev/bias_flux_size_cache_gauss2.npz --coord raw --width` gives a smooth
structure (0/59 significant sign flips): a −0.011 lobe at Mr/Mf~2.25, a +0.004
hump at ~2.75, then a monotone collapse to **−0.41** by 3.49. The bulk
wiggle is ~5x milder than bulgedisc's ±0.02, but with 2e-4 bootstrap errors
it is 20σ — and gauss2 has `Var[Q|m] = 0` by construction, so the bulk
structure is unambiguously the **flow's density**, not population response
scatter. The edge collapse is ~10x *worse* than bulgedisc's.

**The boundary is each population's own support edge, not the chart ceiling.**
`dev/check_edge_distance.py` (written last session, never run — now run):
gauss2 is at m1 = −0.66 where z0 = +3.05, and bulgedisc is at −0.025 at the
same z0. The two curves are anti-aligned in z0 and share sign + monotone
deepening only in (z0_edge − z0). `POINT_SOURCE = 3.6926` is **not** what
binds.

**What actually leaks: the concentration direction.** New,
`dev/check_support_leak.py`. gauss2's realised population is P0 *truncated* by
`sim.sample_population_gauss2`'s acceptance box (~70% of P0's mass rejected on
sigma/rho/|e|), so the support edge is a genuine cliff. Sampling
`flows/shear_gauss2.eqx` at g=0 through that same box:

- The Mr/Mf **marginal is near-perfect** — flow q999 3.5132 vs train 3.5150,
  mass above the train q999 0.09% vs 0.10%. The flow does *not* smear across
  the size edge, and a right marginal turned out to say nothing about the
  conditional.
- 3.25% of mass is outside the support regardless, and the rejection rate
  climbs monotonically with Mr/Mf **over exactly the range m1 collapses**:
  0.30% at 2.60–2.74 (m1 +0.003) -> 1.71% at 3.09–3.16 (−0.010) -> 7.36% at
  3.23–3.32 (−0.05) -> **15.23%** at 3.32–3.57 (−0.13 to −0.41).
- The failing constraint is `rho` (bulge_ratio) and `solve` at near-identical
  rates, both sides but mostly rho < 0.02 (1.54%) over rho > 0.98 (0.73%).

Reading: the physical support **pinches** in the concentration direction as
Mr/Mf approaches its edge (rho -> 0 and near-point-source are the same
degeneracy corner). The flow, being smooth, puts a growing share of its
conditional mass in that empty region; Q and R there are extrapolation, and m1
collapses in proportion.

This also explains the null in memory `bound-the-latent-not-the-observed`:
that experiment closed the *chart* ceiling on slot 1 and measured 0.0000
weight above it. Correct, and irrelevant — the flow never leaked there. It
leaks across the *concentration* boundary, which the chart does not bound.

---

## Open questions for the next session

1. **What is actually causing the multi-lobe shape?** Both mechanisms tried
   so far were about the *coordinate* (spline structure, logit
   reparametrization) and both came back negative. The next lead is the
   *population*, not moment space: bin/color the same profile by the
   bulgedisc structural parameters (bulge/disk flux share, sigma ratio,
   ellipticity) instead of anything derived from `m`. bulgedisc doesn't save
   `theta` on disk the way gauss2 does (see `truth.py` /
   `imsims/analytic.py`) — check whether `imsims.sim --pop bulgedisc` can be
   made to, or whether the gauss2 population (which does have exact `theta`)
   shows an analogous structure that's easier to attribute mechanistically.
2. **Does this reproduce in the noisy pipeline?** Everything above is the
   noiseless path (exact per-galaxy derivatives, no kernel MC, no centroid
   layer). Confirm the same peak/trough/bump survives once
   `bias.mixture_draws`/`pqr_streamed` and the centroid layer are in the
   loop — the earlier point-source tail finding (`ess-starvation-is-the-
   resolution-edge`) says the noisy path can behave differently right at the
   edge.
3. ~~**Does gauss2 show it too?**~~ **Answered 2026-08-13** — see the section
   above. Yes, and it relocated the boundary from the chart ceiling to the
   population's own support edge in the concentration direction.

4. **Is the leak the cause, or a co-symptom?** The correlation between
   rejection rate and m1 is monotone over ten bins, but both could be driven
   by "the flow is bad near the edge" without the leaked mass being the
   mechanism. The cheap discriminator: penalise or bound the Mc/Mr direction
   in training and see whether m1's edge collapse moves. Note gauss2's box is
   an artefact of its rejection sampler — check the mechanism on bulgedisc
   against an *empirical* conditional support before treating it as general.

---

## Prompt for the next session

```
Read HANDOFF.md. We just found that the multiplicative-bias ramp along the
Mr/Mf axis didn't go away with the window-function fix -- it reshaped into a
real, reproducible multi-lobe oscillation that survives fine binning, a
flux-split phase-lock test, and two ruled-out coordinate explanations
(spline knots, logit reparametrization). The remaining lead is that it's a
population effect, not a moment-space coordinate effect.

Start with open question 3 (gauss2): rerun dev/plot_bias_flux_size.py and
dev/check_size_oscillation.py against flows/shear_gauss2.eqx and the gauss2
catalogs instead of bulgedisc. gauss2 has Var[Q|m]=0 by construction and
exact theta on disk (truth.py / imsims/analytic.py), so if the same
peak/trough/bump shape shows up there, we can directly regress it against
theta (bulge/disk flux share, sigma ratio, e1/e2) instead of guessing from
moments -- and any residual is unambiguously the flow's fault, no population
response-scatter to argue about first.

If gauss2 is clean (no oscillation), that's equally informative: it would
mean the effect is specific to the bulgedisc population's own morphology
distribution, and open question 1 (getting theta saved for bulgedisc, or
finding a good proxy) becomes the priority instead.

Don't skip open question 2 (does it survive the noisy pipeline) before
calling this understood well enough to act on -- the point-source tail is
already known to behave differently once kernel MC and the centroid layer
are in the loop (memory: ess-starvation-is-the-resolution-edge).
```
