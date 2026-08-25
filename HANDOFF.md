> **Read the 2026-08-24 (evening) section at the bottom first.** The centroid
> layer's g-conditioning had a real sign bug (fixed, committed dc41c8c) that
> makes every `flows/centroid_*.eqx` checkpoint from earlier today stale, and
> a still-open, now doubly-confirmed (optimizer-independent AND
> seed-independent) instability in the shear response near Mr/Mf ~ 3.1 whose
> exact mechanism is unresolved.

# Handoff: the resolution-axis structure in m1, post window-fix

Written 2026-08-12. The previous handoff (regenerate everything under the
corrected `KBlackmanHarris`/`POINT_SOURCE` weight) is **done** — catalogs
regenerated, flows retrained, tests passing. This one picks up the
measurement that motivated it: what does m1 look like along the resolution
axis now, and it turns out the answer is more interesting than "the ramp is
gone."

> **Read the 2026-08-19 section at the bottom first.**  Two findings there
> supersede large parts of what follows: every m1 below was measured on an
> **undertrained bulk**, and every noisy number taken through `pqr_streamed`
> without `--centroid` was measured through a **NaN gradient** that silently
> dropped 16% of targets.  The "broad hump" and "resolution-edge bias" threads
> are the second of those, not physics.

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

## 2026-08-13 (later): open question 2 answered, and a slot-2 chart bound

**Open question 2: the noiseless oscillation does NOT survive C_M.** Run
`bias.py --flow flows/shear.eqx --pop bulgedisc --samples 8192 --chunk 2048
--alpha 0.5 --n-targets 50000 --save-pqr`, scored by the new
`dev/check_noisy_size_profile.py`. Overall m1 = +0.0002 +/- 0.0009. Binning is
on the TRUE Mr/Mf (`--n-targets` is a prefix slice, so npz row i = catalog row
i), which is why this used the noiseless 1M catalog rather than the image-noise
ones: no regression proxy, and no binning on a noisy quantity.

The noisy profile is a single broad POSITIVE hump peaking near Mr/Mf ~3.3, then
the collapse — not the multi-lobe shape:

    <2.655 +0.0079  <3.019 +0.0172  <3.209 +0.0312  <3.321 +0.0356
    <3.396 +0.0303  <3.465 +0.0087  <3.536 -0.0326  <3.676 -0.1886

Pre-registered peak-minus-trough (windows fixed from the noiseless profile
first): **+0.0058 +/- 0.0024** against a noiseless **+0.02831 +/- 0.00038** —
21%, and only 2.4 sigma from zero. **This is not just blurring**:
sigma(Mr/Mf) from C_M is 0.052 against a 0.22 lobe spacing, and Gaussian
smoothing at that ratio predicts 76% survival, not 21%. The convolution
replaces the structure rather than smoothing it. Consequence: **chasing the
noiseless oscillation is not the route to the noisy bias.**

**The Mc/Mr point-source ceiling is now in the chart.** `Mc/Mr` has an exact
ceiling by the same argument as `POINT_SOURCE` — a point source has
Itilde/T = 1, so `sum(W k^4)/sum(W k^2) = 6.662089`. Slot 2 was a bare ratio,
leaving that as an unmodelled interior cliff; bulgedisc piles against it (max
6.6365 = 99.6% of it, **skew -2.02**, which the logit takes to -0.22, matching
slot 1's -0.28). `models/bijections.POINT_SOURCE_MC`, mirrored in
`bulk.to_coords` / `truth.to_coords` / `in_domain`, tested in
`tests/test_bounded_chart.py`.

What it bought, bulgedisc noiseless 1M, `flows/shear_mcb.eqx` vs
`flows/shear.eqx` (`retrain_mcbound.sh`):

| window | control | bounded |
|---|---|---|
| diffuse [1.5,2.5) | +0.0794 +/- 0.0395 | +0.00218 +/- 0.00050 |
| ALL | +0.00342 +/- 0.00342 | +0.00102 +/- 0.00010 |
| edge [3.62,3.70) | -0.0419 | +0.0064 (54 sigma) |
| edge [3.55,3.62) | -0.0360 | -0.0251 (25 sigma) |
| bump [3.40,3.46) | +0.0021 | +0.0102 (28 sigma, WORSE) |
| peak - trough | +0.02831 | +0.02926 (unchanged) |

The headline is conditioning: the control had ONE target holding **61.5%** of
the ensemble sum|R11| at |R11| = 2.9e10 against a median of 164, and `bias.py`'s
own guard drops 7 targets on it and **0** on the bounded flow. Errors shrink 34x
overall and 79x in the diffuse window.

**Mechanism correction:** those blown-up targets were NOT near the ceiling —
they sat at v = Mc/(rc* Mr) ~ 0.67-0.76 (4th-10th percentile) and low Mr/Mf, the
DIFFUSE end. The old ratio's -2.02 skew pushed that tail to ~4 sigma in
standardised units where the flow extrapolated and R exploded. The win is the
**reparametrisation**, not the closure.

**The bulgedisc leak check came back negative**, which is why the bound was
justified on the skew/ceiling argument instead. `dev/check_support_leak.py`
grew an empirical A/B support test (quantile-box and k-NN) since bulgedisc has
no analytic acceptance box. Validated against gauss2's exact box first, and
both empirical variants **under-report badly** (+2.0% and +1.2% where the box
says 15.2%) and misattribute the coordinate — unavoidable, since across a hard
truncation the leaked points sit adjacent to dense real data. As detectors of
the pattern: gauss2's leak grows monotonically to its edge and tracks m1;
bulgedisc's is flat at ~0 over the whole range its lobes live in, rising only in
the last bin (+0.38%). And the bound is a **null on gauss2's leak** (3.25% ->
3.25%) — correctly, since gauss2 maxes at Mc/Mr 6.5110, below the ceiling.

**LANDING HAZARD:** the chart change silently invalidates every flow in
`flows/`. `mean`/`std` are serialised leaves, so an old flow restores its old
standardisation and then gets the new transform — wrong numbers, no error.
Everything must be retrained; baselines above were run from a worktree pinned
before the change. `in_domain` also masks more noisy targets now (deep 4.02% ->
6.02%); they are not dropped, only barred from direct `log_prob`.

---

## 2026-08-13 (later still): flows updated, and the hump run to ground

**All eight canonical flows retrained under the new chart.** Old-chart copies
are archived in `flows/pre_mcbound/` (same convention as `flows/pre_isotropy`).
val NLL is marginally better everywhere (it is chart-independent — `log_prob`
is the density in raw moment space — so it is a fair comparison):
bulgedisc bulk 40.4567 -> 40.4447, sersic shear 42.3620 -> 42.3594, gauss2 bulk
42.4230 -> 42.4223.

**The "broad hump" IS the long-standing noisy ramp** (+0.031 in the old
sweeps). Scalar for it: `HUMP = m1[3.15,3.40) - m1[2.00,2.60)`.

**It is not importance-sampling bias.** Same 20k targets, same flow, only the
estimator moved:

| run | median ESS | HUMP |
|---|---|---|
| old chart, a=0.5, S=8192 | 541 | +0.0254 +/- 0.0028 |
| new chart, a=0.5, S=8192 | 541 | +0.0236 +/- 0.0029 |
| new chart, a=1.0, S=8192 | 889 | +0.0240 +/- 0.0033 |
| new chart, a=1.0, S=32768 | 3556 | +0.0241 +/- 0.0032 |

A **4x rise in ESS moves it by 0.0001 +/- 0.0033**. This also retires
`sweep_noisy.py`'s "the draw count is not negotiable" caution in the upward
direction: 8192 is converged for the ramp; the 2048 pathology was a low-S
effect that does not continue.

**It is not domain truncation.** Kernel draws leaving the chart run 0.00-0.02%
across the hump's entire range and only reach 0.92% / 7.24% in the top two
bins. Truncation drives the EDGE COLLAPSE; the hump is a separate thing.

**gauss2 does not have it, and tilts the other way.** Same settings,
`flows/shear_gauss2.eqx`: overall m1 = **+0.00096 +/- 0.00064**, profile a
monotone decline +0.0187 -> -0.0635, HUMP = **-0.0336** against bulgedisc's
+0.0240. So the C_M machinery is not intrinsically biased — it reaches
m1 ~ 0.001 on a population the flow models well. **The hump is bulgedisc's
flow density.**

**The Mc/Mr bound halves the noisy edge collapse too.** Per-bin, old vs new
chart, the change is null everywhere except the top two bins: -0.0342 ->
-0.0220 and **-0.1683 -> -0.0949**. Note the trap this exposes: overall noisy
m1 goes +0.0028 -> +0.0128 under the bound, which looks worse but is not — the
old chart's near-zero total was a **cancellation** between the positive hump
and the negative edge collapse. Removing the collapse unmasks the ramp. Same
lesson as [[response-lever-is-exhausted]]'s "+0.0006 is cancellation"; do not
score this pipeline on the population total alone.

---

## 2026-08-13 (leads): morphology, and a control that changes the scoreboard

**bulgedisc DOES save its structural parameters.** Ext `POPULATION`, columns
`flux/sigma/e1/e2/bulge_frac/bulge_ratio/bulge_e1/bulge_e2`, on `moments.fits`
and every target catalog. Earlier text in this file saying otherwise was wrong,
and open question 1 was never actually blocked. `dev/check_hump_morphology.py`
does the attribution.

The structural contrast with gauss2 is real and suggestive: bulgedisc gives the
bulge its OWN ellipticity and a free `bulge_frac`, so five even moments do not
determine the galaxy (Var[Q|m] > 0); gauss2 is co-elliptical with a fixed flux
share, so they do (Var[Q|m] = 0).

**But the morphology attribution is confounded, and so is every binned noisy
profile in this project.** Binning by morphology — or, in the noisy path, by
the TRUE Mr/Mf — selects a subsample on a variable the estimator cannot see.
That subsample's moment density differs from the population prior, and BFD is
only unbiased under the right prior, so per-bin m1 is biased **even with a
perfect density**. This is NOT the objection `dev/check_bias_by_selection.py`
answers (that one is about cutting on a SHEARED quantity, eq. 45-46, cured by
cutting on the g=0 catalog); cutting on g=0 does not help here.

**The control: `dev/check_selfconsistency_noisy.py`.** Draw targets FROM THE
FLOW at +/-g (a shared base draw `z` across conditions supplies the pairing),
add noise, run the identical pipeline and binning. The prior is exact by
construction, so anything nonzero is method.

|                | old chart | new chart | real run |
|----------------|-----------|-----------|----------|
| dead targets   | **0**     | 3136 (15.7%) | 2815-3131 |
| overall m1     | -0.0061 +/- 0.0031 | +0.0131 +/- 0.0015 | +0.0128 +/- 0.0014 |
| HUMP           | **+0.0114 +/- 0.0026** | **+0.0123 +/- 0.0030** | +0.0240 +/- 0.0033 |
| top-bin edge   | -0.2253   | -0.0459   | — |

1. **About half the hump is not real.** The control reproduces +0.011 to
   +0.012 of the real +0.0240 on both charts. The old-chart control has zero
   dead targets, converged draws and an exact prior, so what is left is the
   hidden-variable binning. Genuine flow-density part: **+0.0126 +/- 0.0042**.
2. **The old chart's edge collapse was also method** — -0.2253 in the top bin
   with a perfect prior. The bound cutting it to -0.0459 fixes a NUMERICAL
   pathology (cf. the |R11| = 2.9e10 spikes), not a density error.
3. **REGRESSION: the Mc/Mr bound kills 15.7% of noisy targets** ("no draw with
   any weight"), at alpha 1.0 and 0.5 alike; the old chart killed none. The
   extra ceiling puts the entire kernel off-domain for them.

**This corrects the "unmasking" reading above:** the real run's +0.0028 ->
+0.0128 shift under the bound is NOT mostly the edge fix revealing the ramp —
the control shows the same shift with a perfect prior, so it is substantially
the dead targets.

---

## 2026-08-13 (followups): a real bug, and the ramp mostly evaporates

**BUG, found and fixed: `bias.safe_point` never learned about the second
ceiling.** It clamps `Mf` and `Mr` for rows `jnp.where` will discard — and
**clamping Mr DOWN raises Mc/Mr**, so a row violating only slot 1 came out of
the slot-1 clamp violating slot 2. The dummy's own `dlogP/dg` then overflowed,
`where` multiplied it by zero, `0 * inf = NaN`, and `pqr_streamed`'s `good`
mask threw away the WHOLE target. Fix: clamp `Mc` after `Mr`.

    targets with no weighted draw   18.9%   ->  1.07%
    control overall m1             +0.0131  -> -0.0044 +/- 0.0016
    real run overall m1            +0.0128  -> -0.0016 +/- 0.0017

**That bug accounted for the entire apparent noisy multiplicative bias** —
bulgedisc's overall noisy m1 is consistent with zero without it. It also
invalidates two claims made earlier in this file: the "+0.0028 -> +0.0128 is
unmasking" reading, and "the bound halves the noisy edge collapse" (the
control's top bin went -0.0459 -> -0.2043 once fixed, so that halving WAS the
dead targets). `dev/check_dead_targets.py` is the diagnostic that found it: the
density was finite for 100% of targets while the DERIVATIVES failed for 18.9%,
which is what pointed at the dummy rather than the draws.

Fixed alongside: `tests/test_conv.py`'s fixture sat at Mc/Mr = 6.818, above the
new 6.662 ceiling, so six tests comparing against an "unmasked" integral were
silently comparing against a masked one. **Run the whole suite after a chart
change, not just the chart's own tests.**

**2026-08-13 (revised): the control is a NULL TEST, not a calibration.**  The
"corrected" columns throughout this file are ATTRIBUTION -- they say how much of
a residual is the estimator versus the flow's density, which is what decides
where a fix belongs.  They are NOT what a survey would suffer: the control draws
its targets from the flow, so subtracting it is exact only to the extent the
flow already equals the truth, and in the field there is nothing to subtract.
**Report the raw number as the result; use the control to locate the cause.**

**Superseded reporting rule (kept for context):**
`check_selfconsistency_noisy.py --save-pqr` feeds
`check_noisy_size_profile.py --control`, which prints real / control /
corrected on shared fixed bin edges (the two runs cannot share bin membership,
only edges). The corrected profile:

    Mr/Mf bin        real       control     CORRECTED
    [2.00,2.60)   +0.0082      +0.0134    -0.0052 +/- 0.0030
    [2.60,3.00)   +0.0172      +0.0151    +0.0021 +/- 0.0024
    [3.00,3.15)   +0.0291      +0.0234    +0.0057 +/- 0.0032
    [3.15,3.40)   +0.0311      +0.0258    +0.0053 +/- 0.0024
    [3.40,3.55)   -0.0147      -0.0156    +0.0009 +/- 0.0047
    [3.55,3.70)   -0.2542      -0.2422    -0.0119 +/- 0.0274

**Every bin lands within ~2 sigma of zero**, the -0.25 edge collapse included.
HUMP corrected = **+0.0105 +/- 0.0039** is the only surviving structure. So the
resolution ramp is, at this flow, almost entirely an artefact of binning noisy
targets by a variable the estimator cannot see.

Caveat: the control's population is the FLOW's distribution, not the real one,
so this is a leading-order correction — exact only insofar as the flow is close
to the population.

---

## 2026-08-13 (last): the image-noise + centroid path, checked

Open question 2's remaining half -- the fix was verified on the moment-space
`add_noise` + C_M path, never on the **image-noise + centroid-layer** one.  It
holds there.  All three runs: `bulgedisc_noisy`, 20000 targets, 8192 draws,
alpha 1.0, chunk 2048.

| run | overall m1 | no-weight targets |
|---|---|---|
| `flows/centroid.eqx`, layer on | -0.0074 +/- 0.0024 | 139 (0.70%) |
| `flows/shear.eqx --no-centroid`, same targets | -0.0088 +/- 0.0023 | 139 (0.70%) |
| self-consistency control, layer on | -0.0046 +/- 0.0016 | 144 (0.72%) |

1. **The `safe_point` fix carries over.** 0.70% dead against the 18.9% the bug
   cost, and the control agrees at 0.72% -- same residual diffuse-end
   population, not a centroid-path effect.
2. **Corrected overall m1 = -0.0028 +/- 0.0029**, consistent with zero.  The
   control is -0.0046, statistically identical to the non-centroid control's
   -0.0044: the method floor does not care about the layer.
3. **The layer is worth +0.0014 +/- 0.0013** (`bias.py --compare`, paired over
   the same targets and draws), positive in four of five flux quintiles and
   monotone in q3-q5.  Small, as expected at this depth -- the layer's regime is
   `bulgedisc_deep` (noise_sigma 2.73), where it was previously measured at
   +2e-3 to +1.4e-2.
4. ESS is unchanged by the layer (median 872 vs 879, frac<10 0.019 vs 0.018).

**The deep depth is done too — see the next section.**

`dev/check_selfconsistency_noisy.py` grew `--centroid`: it builds the flow with
the layer and draws/evaluates at Sigma_X from `--cov-from`.  Note what that
control does and does not test -- its model is centroid-marginalised moments
plus INDEPENDENT C_M noise, exactly what `bias.py`'s image-noise path assumes.
In the real catalog the centroid shift and the moment noise come from the same
pixel realisation, so the control measures method bias UNDER that model, not
whether the model itself is right.

---

## 2026-08-13 (deep): a second chart bug, and a residual the control does NOT explain

Retraining `flows/centroid_deep.eqx` under the new chart NaN'd at step 144 of
8000 and wrote a flow of NaNs.  Two things came out of chasing it.

**BUG: `in_domain` never tested `Mc > 0`.**  Slot 2 of the chart is
`logit(Mc / (rc* Mr))`, so `Mc <= 0` is the log of a negative -- a NaN, which
unlike a -inf no downstream mask survives.  It is reachable: **1.5e-4 of the
shallow copy catalog and 4.1e-4 of the deep one** have `Mc <= 0` (a badly
off-centre copy loses its high-k moment first), and so do kernel draws in the
noisy path.  The deep training hit it because `CopySampler` weights flatten with
Sigma_X, so the far-off-centre copies actually get drawn.

Fixed in `models/bijections.in_domain`, pinned by
`tests/test_bounded_chart.test_in_domain_rejects_non_positive_mc`.  What it was
costing the MEASUREMENT, shallow `bulgedisc_noisy`, 20k targets:

    targets with no weighted draw   139 (0.70%)  ->  0
    overall m1                      -0.0074      ->  -0.0083 +/- 0.0023

paired (`--compare`) the shift is **-0.0009 +/- 0.0004, all of it in the faintest
flux quintile** (+0.0128 -> +0.0076): exactly the targets whose kernel reaches
`Mc <= 0`.  So the residual 1.07% "dead targets" left over from the `safe_point`
fix were this, and they are now gone.

**`safe_point` moved to `models/bijections.py`,** beside `in_domain`, because the
centroid training loss needs the identical guard and the last bug here was one
copy of it not learning about a new ceiling.  `bias.safe_point` still resolves.

**The centroid training loss now guards the same way** (`centroid.train`):
masking the RESULT of `log_prob` is not enough, since `jnp.where` still
differentiates the -inf branch and hands back NaN.  The input is substituted
instead, and the loop prints `off-chart`, the fraction of copies the layer maps
out of the chart -- it ran 0 to 1e-3 per batch on the deep retrain.

**The retrained deep flow is good.**  `centroid.py check`: layer shift vs
catalog shift -4.157e-3/-4.124e-3 (Mf), -7.929e-3/-8.020e-3 (Mr),
-1.139e-2/-1.156e-2 (Mc), and the ellipticity response -- the thing the layer
exists for -- **+1.7809e-2 against the catalog's +1.7774e-2, 0.2%**, where the
old pre-chart flow was 1.1% off.

### The integration at both depths

`bulgedisc_noisy` at alpha 1.0 / `bulgedisc_deep` at alpha 0.5 (median ESS 395
against alpha 1.0's 24 -- the mixture is not optional at this depth), 8192
draws, 20k targets, 0 dead targets everywhere:

| | shallow (sigma 1) | deep (sigma 2.73) |
|---|---|---|
| real, centroid layer on | -0.0083 +/- 0.0023 | -0.0120 +/- 0.0024 |
| real, `--no-centroid` | -0.0088 +/- 0.0023 | -0.0196 +/- 0.0021 |
| self-consistency control | -0.0048 +/- 0.0016 | **+0.0007 +/- 0.0023** |
| **corrected (real - control)** | **-0.0035 +/- 0.0028** | **-0.0127 +/- 0.0033** |
| layer worth, paired | +0.0014 +/- 0.0013 | **+0.0076 +/- 0.0021** |

1. **The layer works, and its size is the predicted one.**  +0.0076 at depth
   against the +2e-3 to +1.4e-2 the docstring predicts from
   `<e_w e0*>/<|e0|^2> - 1`, and **+0.048 +/- 0.009 in the faintest deep
   quintile** -- the 1/(S/N)^2 scaling, measured.
2. **The method is clean at depth**: the control returns +0.0007 +/- 0.0023 with
   an exact prior, i.e. none of the machinery (kernel MC, the peel, the mixture
   proposal, the centroid condition) carries bias of its own there.  The
   shallow control's -0.0048 is the one that is NOT zero, and it is unchanged by
   this fix (-0.0046 before).
3. **So the deep residual is real.**  -0.0127 +/- 0.0033 survives the control at
   3.8 sigma.  Shallow is -0.0035 +/- 0.0028, consistent with zero.  The bias
   that "went away" at shallow depth is present at 2.73x the noise, in the
   density rather than the method, and the centroid layer removes only a third
   of it.

---

## 2026-08-13 (gauss2, deep): a truth-free prior refinement, validated

The self-consistency control is blind to the flow being wrong -- it draws its
targets FROM the flow.  `dev/refine_prior.py` is the attack on that blind spot:
you never observe p(m), but you DO observe M, and the model's prediction for it
is the convolution the estimator already computes, so the target catalog is
itself a likelihood for the prior.  Fitting it needs no truth and no simulation.

**Shear is not assumed.**  Each target's (M1, M2) is rotated by its own random
angle before the fit.  Shear is a spin-2 distortion with a coherent orientation,
so that destroys it while leaving every spin-0 quantity -- where this project's
residual bias lives -- bit for bit.  The surviving |e| distribution moves with
shear only at O(g^2) ~ 4e-4.  The assumption is intrinsic-orientation isotropy,
the standard one, not knowledge of g.  Only the bulk is fit; the shear layer and
the standardisation are frozen.  `bias.py --noise-scale` (new) puts the
moment-space catalogs at the deep depth.

**gauss2 at deep noise is CLEAN, which is why it needed an injected error.**
`flows/shear_gauss2.eqx`, noise x2.73, alpha 0.5, 20k targets:
**m1 = +0.00012 +/- 0.00148**.  So the whole machinery -- kernel MC, the
mixture proposal, the convolution -- reaches 1e-4 at 2.73x noise on a population
the flow models well, and there is nothing there to correct.  This also sharpens
the bulgedisc deep result above: that -0.0127 is the density, not the depth.

The fixture: resample the gauss2 training population by exp(0.3 * standardised
size coordinate), which shifts the size marginal by +0.31 sigma, train a bulk on
it, and graft it under the validated shear layer.  (A bulk starved on 2k
galaxies was tried first and rejected: it injects a badly CONDITIONED density,
m1 = +0.025 +/- 0.123, not a usefully wrong one.)

| prior | overall m1 | q1 (faintest) |
|---|---|---|
| correct, 1M-trained | +0.00012 +/- 0.00148 | +0.074 +/- 0.023 |
| **tilted (+0.31 sigma)** | **+0.01871 +/- 0.00160** | +0.156 +/- 0.027 |
| tilted -> refined, 300 steps | **+0.00126 +/- 0.00591** | +0.104 +/- 0.147 |
| tilted -> refined, 3000 steps | +0.473 +/- 0.176 | **-1.219 +/- 0.100** |
| correct -> refined, 300 steps | +0.00296 +/- 0.00444 | +0.166 +/- 0.099 |

1. **It works.**  A +0.0187 injected bias comes back to +0.0013 +/- 0.0059,
   from sheared targets only, with no truth anywhere in the loop.  Per quintile
   q2 +0.0398 -> -0.0002 and q3 +0.0229 -> -0.0006.  Independent corroboration
   that the density moved towards the data: median ESS 608 -> 921.
2. **It is a fixed point, not a nudge.**  Started from the correct prior instead,
   it lands at +0.0030 +/- 0.0044 -- the same place, within errors, as starting
   from the tilted one.  And it does not break a prior that was already right.
3. **Over-training is catastrophic, and early stopping is mandatory.**  At 3000
   steps the faintest quintile goes to -1.2 and 84 targets die.  The damage is
   where the convolved likelihood barely constrains the density (lowest S/N) and
   BFD's m is most sensitive.  The held-out objective plateaus by step ~100 and
   is a usable stopping rule.
4. **But the objective cannot RANK priors.**  Tilted and correct start 0.008
   nats/target apart (45.087 vs 45.094) while their m1 differ by 0.019.  Over
   56000 targets that is 450 nats, so the gradient still points the right way --
   but do not expect val NLL to tell you which flow has the smaller bias.  Same
   lesson as [[capacity-does-not-fix-the-ramp]], now for the deconvolving
   objective.
5. **Cost: the m1 error inflates 3x** (+/-0.0015 -> +/-0.0044), all of it the
   faint quintile.  Regularising towards the pre-refinement density, or
   restricting the correction to a smooth low-dimensional family instead of all
   bulk weights, is the obvious next move.

NOT done: the density was scored only through m1.  gauss2 has exact truth, so
`dev/check_flow_vs_truth.py` can say whether the refinement moved the density
towards the true one or merely somewhere that cancels in m -- run it next.

---

## 2026-08-13 (deep, refined): the residual is the DENSITY, not Var[Q|m]

> **SUPERSEDED IN PART, later the same day.**  Everything below was measured at
> 8192 draws, and the depth scan's convergence test shows that setting is NOT
> converged at sigma 2.73: the raw deep bias falls from -0.0094 to -0.0021 at
> 32768 draws.  So most of the -0.0127 this section attributes to the density is
> an under-converged integral, and the refinement that "removed 80% of it" was
> largely compensating for an integration artefact by distorting the density --
> which is the more likely explanation for why it was so easy to overshoot.  The
> Var[Q|m] conclusion STANDS (the residual was never the response-scatter floor);
> the density conclusion does not.  Repeat the image-noise + centroid deep run at
> 32768 draws before acting on any number here.

Question asked: is bulgedisc's -0.0127 just `Var[Q|m] > 0`, given gauss2 (which
has `Var[Q|m] = 0` by construction) is perfect?  Two answers, one from the
existing measurement and one new.

**From what was already measured:** `Var[Q|m]` at fixed LATENT m is 1.42%, worth
**m ~ 4e-4** ([[floor-is-at-fixed-measured-moments]]) -- 3% of the residual --
and `varq.py`'s correction for exactly that term measured NULL, which is what a
4e-4 term should do.  Do not reach for the 22% figure in that same table: it is
the scatter at fixed MEASURED moments, and it is not a floor on this estimator,
because if the LATENT density is right the convolution is right automatically.
gauss2 also differs from bulgedisc in two other ways that could each explain a
null: its density is Gaussian in the flow's own chart BY CONSTRUCTION, and its
shear response is exactly representable (the co-elliptical family is closed
under shear).

**The decisive test.** `dev/refine_prior.py` fits the BULK only -- the shear
layer is frozen -- so it can repair `p(m|0)` and cannot touch the deterministic
transport that drops `Var[Q|m]`.  Whatever it removes was density error.  On
`bulgedisc_deep`, 60 steps at lr 1e-4 on symmetrised sheared targets:

| | before | after 60 steps |
|---|---|---|
| held-out convolved NLL | 44.918 | 44.592 |
| median ESS | 395 | 465 |
| real run | -0.01200 +/- 0.00242 | **+0.00353 +/- 0.00190** |
| control | +0.0007 +/- 0.0023 | +0.0059 +/- 0.0018 |
| **corrected** | **-0.0127 +/- 0.0033 (3.8 sigma)** | **-0.0024 +/- 0.0026 (0.9 sigma)** |

**A density-only fit removes ~80% of the residual.**  It cannot have touched
`Var[Q|m]`, so the residual was the density -- consistent with the 4e-4 estimate
and against the Var[Q|m] hypothesis.

**bulgedisc is far more fragile than gauss2 under this fit.**  At 200 steps /
lr 3e-4 -- gentler than the setting gauss2 tolerated -- the same fit improved the
objective by 0.37 nats and made m1 UNMEASURABLE: +0.069 +/- 0.215, per-quintile
errors to +/-6, 141 targets dropped.  The likelihood is happy to sharpen the
density near bulgedisc's hard support edges, and R is exquisitely sensitive to
exactly that curvature.  Note what this means for the stopping rule: the
held-out objective improved in BOTH runs and does not distinguish them.  Watch
the |R| guard's drop count and the per-quintile error bars, not the NLL.

Two bugs in the refiner found on the way, both the same shape as this project's
recurring one -- a mask in the wrong place:
- `in_domain` must be tested on the draws AFTER the centroid layer, as
  `pqr_streamed` does.  Applied to the raw draws it admits 0.27% that the layer
  maps off the chart; their `log_prob` is -inf, which the VALUE survives and the
  gradient does not (170 of 172 gradient leaves NaN).
- a draw whose layer log-det is non-finite has to be moved to a point
  `in_domain` REJECTS, so the -inf in the logsumexp is a constant rather than a
  computed one. A computed -inf differentiates to NaN; a masked one does not.

---

## 2026-08-13 (depth scan): the bias is LINEAR in the noise sigma, no threshold

`bias.py --pop bulgedisc --flow flows/shear.eqx --alpha 0.5 --samples 8192
--n-targets 20000` at four `--noise-scale` values.  Moment-space noise, no
centroid layer, so this is the population's density error alone, scanned in
depth.  RAW m1 -- what a survey would suffer, no control subtracted:

| noise scale | median ESS | m1 |
|---|---|---|
| 1.0  | 546 | +0.00056 +/- 0.00106 |
| 1.5  | 382 | -0.00309 +/- 0.00136 |
| 2.0  | 357 | -0.00551 +/- 0.00171 |
| 2.73 | 381 | -0.00938 +/- 0.00220 |

A least-squares line gives **m1 = 0.0058 - 0.0056 sigma**, every point within
~5e-4 of it.  So it is linear in the noise SIGMA (not the variance), with no
knee and no safe plateau -- the bias at any depth can be read off, and only the
shallow end is clean.  Two corroborations: the 2.73 point agrees with the
image-noise + centroid path's -0.0120 (different catalogs, different machinery,
same effect), and at the base depth +0.0006 +/- 0.0011 is inside a 0.003 budget
with no correction of any kind.

**This is the number that decides whether any correction is needed.**  At the
depth these sims were built for, nothing is required.  The question for a real
application is only where on this line the survey sits.

**Attribution (null test at the two ends, NOT a subtraction):** the same flow's
self-consistency control returns **-0.0005 +/- 0.0010 at sigma 1.0** and
**-0.0048 +/- 0.0021 at sigma 2.73**.  So of the raw -0.0094 at depth, roughly
HALF is the estimator with a perfect prior and half is the flow's density.  That
is a different split from the image-noise + centroid path, whose control was
+0.0007 -- the control is flow-specific, so the two are not in conflict, but do
not carry either split across paths.

**CONFIRMED: the machinery half is the paper's own 1/ESS bias.**  Sec. 2.5 puts
the induced m at ~1/ESS, and at 8192 draws the median ESS is 381, i.e. ~0.0026 --
the order of the -0.0048 measured.  Repeating the sigma = 2.73 control at 32768
draws:

    control, sigma 2.73,  8192 draws:  -0.0048 +/- 0.0021
    control, sigma 2.73, 32768 draws:  -0.0002 +/- 0.0015

It is gone.  And the REAL run at the same setting says the same, more strongly:

    real, sigma 2.73,  8192 draws:  -0.00938 +/- 0.00220   (median ESS  381)
    real, sigma 2.73, 32768 draws:  -0.00206 +/- 0.00160   (median ESS 1525)

**So the deep bias is almost ENTIRELY the draw count, not the density.**  With a
converged integral the raw deep number is -0.0021 +/- 0.0016 (1.3 sigma from
zero) and the control is -0.0002, leaving a density contribution of
-0.0019 +/- 0.0022 -- consistent with zero.  The measured effect is ~4x larger
than a naive 1/ESS = 1/381 = 0.0026 would predict, so do not trust that formula
quantitatively; trust the convergence test.

This also corrects the earlier "4x ESS moves it by 0.0001" null
([[broad-hump-is-real-and-bulgedisc-specific]]): that was measured at the
SHALLOW depth, where the control is already -0.0005 and there was nothing for
more draws to remove.  **The draw count is depth-dependent, and 8192 is NOT
converged at sigma 2.73.**  Any deep number in this file measured at 8192 --
including the -0.0120 image-noise + centroid run and the whole depth scan --
carries an ESS component of this size.

---

## 2026-08-13 (last): the centroid path has a +0.007 method bias that draws do NOT fix

Repeating the image-noise + centroid deep measurement at 32768 draws, as the
depth scan's convergence test demanded.  It does NOT behave like the
moment-space path:

| sigma = 2.73, alpha 0.5 | 8192 draws | 32768 draws |
|---|---|---|
| moment-space real | -0.0094 +/- 0.0022 | **-0.0021 +/- 0.0016** |
| moment-space control | -0.0048 +/- 0.0021 | **-0.0002 +/- 0.0015** |
| centroid real | -0.0120 +/- 0.0024 | **-0.0076 +/- 0.0018** |
| centroid control | +0.0007 +/- 0.0023 | **+0.0070 +/- 0.0017** |

**The centroid control gets WORSE with more draws** -- +0.0007 to +0.0070, 4.1
sigma from zero with an exact prior and a converged integral.  Read with the
moment-space column, the 8192 value was two errors cancelling: an ESS bias of
~-0.006 against a genuine +0.007 that draws do not touch.  So the earlier
"the machinery is clean at depth" claim in this file was wrong for that reason.

**Consequence: no centroid-path residual can be attributed to the flow's density
until this is found.**  The clean deep measurement is the moment-space one,
-0.0021 +/- 0.0016.  Also note the centroid layer's measured VALUE (+0.0076
paired) was taken at 8192 and deserves re-checking at 32768.

**Ruled out: the layer's fixed-point inverse.**  `sample` uses the 3-pass
`marginalize` while `log_prob` uses the exact closed-form `unmarginalize`, so at
alpha < 1 the proposal is drawn from one distribution and weighted by another --
and the round trip is 6x worse at the deep Sigma_X (p99 4.8e-4 at n_iter 3
against 7.7e-5 at 6; 81 of 4000 rows above 1e-4, plateauing at ~36 by n_iter 12,
so part of it is a genuine fold rather than iteration count).  A/B at 8192
draws, new `--n-iter` on `check_selfconsistency_noisy.py`:

    n_iter =  3:  +0.0007 +/- 0.0023
    n_iter = 12:  +0.0019 +/- 0.0023      (shift +0.0012 +/- 0.0033)

Null.  A proposal misspecification biases the estimator's TARGET, not its
variance, so it is draw-count independent and this transfers to the 32768 case.

**Suspects left**, in the order worth trying: ~~the peel's log-det folded into the
importance weight~~ (**dead, 2026-08-14, see below**); the `safe_point`
substitution interacting with the layer's output; and the streaming merge at low
per-chunk ESS.  ~~A cheap first cut: run the centroid control at alpha = 1.0 (no
flow sampling at all) with enough draws to keep the ESS up, and see whether
+0.007 survives.~~  **Tried 2026-08-14 -- it is neither cheap nor decisive, see
below.  Do not repeat it.**

---

## 2026-08-14: alpha = 1 cannot answer the +0.007, and the log-det is innocent

Two results, both negative, both worth not re-deriving.

**1. The alpha = 1 route is closed.**  `dev/check_selfconsistency_noisy.py` at
alpha 1.0, S = 131072, n = 20000, against yesterday's alpha 0.5 / S = 32768:

| control | alpha 0.5, S 32768 | alpha 1, S 131072 | shift |
|---|---|---|---|
| centroid deep | +0.0070 +/- 0.0017 | -0.0065 +/- 0.0108 | -0.0135 |
| moment-space, sigma 2.73 | -0.0002 +/- 0.0015 | -0.0157 +/- 0.0102 | -0.0155 |

The second row is the one that decides it: that control is KNOWN to converge to
zero (-0.0048 at 8192 -> -0.0002 at 32768, alpha 0.5).  At alpha 1 with 16x the
draws of that reference it reads -0.0157 +/- 0.0102 -- 1.5 sigma BELOW zero and
6.8x noisier.  A proposal that cannot reproduce a known zero cannot adjudicate a
+0.007.  Both paths shift by the same amount and the same (negative) sign, which
is the signature of residual 1/ESS bias in an under-converged integral, not of
anything centroid-specific.  Ladder at n = 1000 for the record (only the OVERALL
number is meant to be zero; per-bin is hidden-variable binning by design):
S = 32768 -> -0.066 +/- 0.025, 131072 -> +0.040 +/- 0.049, 524288 -> +0.022 +/-
0.015, i.e. tail-dominated until at least 131072.

A by-product worth keeping: the control's per-BIN profile is identical across
the proposal change and the 4x draw change -- all 8 bins within errors, centroid
HUMP -0.0444 +/- 0.0065 (alpha 0.5) against -0.0452 +/- 0.0092 (alpha 1), and
the moment-space path gives the same 8-bin shape.  **The hidden-variable binning
artefact is invariant to proposal and draw count.**  No binned noisy profile is
rescuable by moving alpha.

Logs: `logs_control_alpha1_S131k.txt`.  Each n = 20000 arm is ~68 min on the
4080S at chunk 2048 (chunk 8192 OOMs at 18.5 GB; do not raise it).

**2. Why it is closed, quantitatively -- and the log-det is exonerated.**
`dev/check_ess_logdet.py` (new) measures ESS directly at each S instead of
scaling one chunk, with and without the peel's log-det in the weight.  200 deep
targets, `flows/centroid_deep.eqx`, median ESS / S:

| S | alpha 0.5 with ld | alpha 0.5 without | alpha 1 with ld | alpha 1 without |
|---|---|---|---|---|
| 2048   | 0.0469 | 0.0470 | 0.0144 | 0.0144 |
| 8192   | 0.0468 | 0.0469 | 0.0095 | 0.0092 |
| 32768  | 0.0470 | 0.0470 | 0.0091 | 0.0091 |
| 131072 | 0.0470 | 0.0471 | 0.0062 | 0.0062 |

- **The log-det does not concentrate the weights.**  It moves ESS by ~0.1% at
  every S and both alphas (96.1 vs 96.2 at S = 2048; 6165.7 vs 6169.8 at
  131072), despite varying ~18 nats across a chunk.  It is also mathematically
  the right thing to fold in -- draws live in raw moment space, the peeled
  flow's density in z, so `ld` IS the change of variables.  Suspect 1 is dead.
- **`bias.main`'s scaled ESS is honest at alpha 0.5 and a fiction at alpha 1.**
  ESS/S is flat to 4 decimals over 64x in S at alpha 0.5 -- the defensive
  mixture's bounded weights give finite variance, so the one-chunk measurement
  times S/chunk is real.  At alpha 1 it FALLS 2.3x over the same range: kernel
  weights are heavy-tailed, so the reported number overstates convergence and
  4x the draws buys only 2.7x the ESS.
- **The cost of the closed route, in numbers.**  At alpha 1 / S = 131072 the
  median ESS is 814 with a 5th percentile of 16.7 -- i.e. our "16x the draws"
  run was LESS converged than yesterday's alpha 0.5 / S = 32768 run (median
  1539, 5th pct 252).  Matching that alpha-0.5 ESS at alpha 1 needs S ~ 2.3e6 on
  the measured ESS ~ S^0.72 scaling.  That is why the shift is -0.014 rather
  than 0, and why no affordable draw count fixes it.

**Where that leaves the +0.007**: unexplained, and now with one fewer suspect.
Next is `safe_point` interacting with the layer's output, then the streaming
merge at low per-chunk ESS -- both testable at alpha 0.5, where the ESS
diagnostic can be trusted and the error bars are 6x tighter.

---

## 2026-08-14 (cont.): suspects 2 and 3 are dead too -- the +0.006 is not plumbing

**Suspect 2, `safe_point` and its mask: a REAL seam that carries no weight.**
`bias._mixture_chunk` (bias.py:449) masks with `in_domain(x)` on the RAW draw,
but the `flow` it evaluates there is the FULL flow, centroid layer included --
so the correct test is `in_domain(centroid(x, Sigma_X))`.  This is the same bug
shape found in the refiner last session, still live in the proposal.
`pqr_streamed` masks on the TRANSFORMED draw downstream, so the two disagree
both ways, and one direction is dangerous: a draw off-chart in raw space but
on-chart in z gets `log_f = -inf`, which drops the flow component out of `log_q`
and inflates `log_wt` -- yet the draw still COUNTS downstream.  That would be a
target bias: draw-count independent, alpha < 1 only, centroid-path only, i.e.
all three fingerprints of the +0.007.

Measured (`dev/check_safe_point_seam.py`, deep targets, alpha 0.5, S = 8192):

| | 200 targets | 2000 targets |
|---|---|---|
| in_domain(raw), not in_domain(z) [harmless] | 9.8e-4 | 1.2e-3 |
| in_domain(z), not in_domain(raw) [COUNTED] | 2.4e-3 | 1.5e-3 |
| the latter's share of Phat | max 2.6e-48 | max 3.2e-10 |
| their over-weighting factor | 1.000x | 1.000x |

The affected draws sit far enough into the tail that the flow component of the
proposal was worth nothing there anyway, so `log_q_true` = `log_q_used` to
float precision and correcting the seam moves `log Phat` by identically zero.
**Fix it on principle** -- it is a latent trap the moment the proposal widens or
alpha drops -- but it is not this bias.

**Suspect 3, the streaming merge: null with the merge REMOVED.**  S = 32768 at
`chunk = 32768` is one chunk, i.e. the exact single-logsumexp path with no merge
at all.  Run at `batch 4` so `batch x chunk` (the memory) matches the production
`64 x 2048`; `--batch` added to `check_selfconsistency_noisy.py` for this.

| centroid deep control, S = 32768, alpha 0.5 | m1 |
|---|---|
| chunk 2048, batch 64 (production, 16 merges) | +0.0070 +/- 0.0017 |
| chunk 2048, batch 4 (16 merges) | +0.0055 +/- 0.0018 |
| **chunk 32768, batch 4 (NO merge, exact)** | **+0.0060 +/- 0.0018** |

Unchanged.  The second row also re-measures it on a different random stream
(`batch` is part of the seed), so it is not a draw realization either.
Logs: `logs_control_merge_test.txt`.

**So the +0.006 is in the model, not the plumbing.**  Every mechanism on the
suspect list is now eliminated, and the estimator computes what it claims to
compute.  What is left is the control's own construction: its targets are made
by `flow.bijection.transform`, which runs the layer's 3-pass fixed-point
`marginalize`, while `log_prob` scores them through the exact `unmarginalize`.
If the fixed point has not converged, the control's targets are NOT drawn from
the density the estimator assumes -- a model mismatch in the TARGET generation,
which is draw-count independent and centroid-only.  The `--n-iter` A/B ruled
this out at 8192 draws (+0.0007 vs +0.0019, shift +0.0012 +/- 0.0033) but that
error bar cannot exclude 0.006, and it was measured where a -0.006 ESS bias was
cancelling.  Re-running it at S = 32768 is the next number.  Note HANDOFF's
earlier finding that ~36 of 4000 rows never converge at any `n_iter`, i.e. part
of it is a genuine fold -- if the A/B is null again, the layer is not a bijection
on those rows and that is a deeper problem than iteration count.

**Done, and null: `n_iter 12` at S = 32768 gives +0.0053 +/- 0.0018** against
+0.0070 +/- 0.0017 at `n_iter 3`, same seed and same draws -- shift
-0.0017 +/- 0.0025, which now excludes 0.006 at ~3 sigma where the 8192 test
only reached 1.4.  Target generation is not it.  `logs_control_niter12_S32k.txt`.

**Which empties the list, so the untested assumption is now the ladder itself.**
"The centroid control gets WORSE with draws" rests on TWO points, +0.0007 at
8192 and +0.0070 at 32768, read as "an ESS bias of -0.006 cancelling a genuine
+0.007".  The competing reading -- it is still moving and has not converged to
anything -- fits the same two points, and the two differ at the third.  Running
S = 131072 at alpha 0.5 (`logs_control_a05_S131k.txt`): a plateau near +0.006
means a genuine target bias in the model and the -0.006-cancellation story
stands; continued climbing means the centroid path's estimator is not converging
in S at all, and every centroid number at every draw count is provisional.

**It plateaus.**  8192 -> +0.0007 +/- 0.0023, 32768 -> +0.0070 +/- 0.0017,
131072 -> +0.0058 +/- 0.0016.  The estimator converges, to +0.006, and the
"-0.006 ESS bias cancelling a genuine +0.006 at 8192" reading is confirmed on
three points.

---

## 2026-08-14 (last): the +0.006 is FLAT IN g, and everything mechanical is ruled out

The remaining non-bug hypothesis was the PQR expansion's own truncation: `bias`
takes m1 from a +/-g pair, so even orders cancel but an O(g^3) error in `ghat`
survives as m1 proportional to g^2, and the centroid layer's extra curvature
would amplify it where the flat moment-space path stays clean.  That predicts
numbers.  `--g` added to `check_selfconsistency_noisy.py`; S = 32768, alpha 0.5,
20000 targets:

| g | O(g^2) predicts | measured |
|---|---|---|
| 0.01 | +0.0015 | +0.0052 +/- 0.0021 |
| 0.02 | +0.0070 (anchor) | +0.0070 +/- 0.0017 |
| 0.04 | +0.024 | **+0.0067 +/- 0.0017** |

**Flat**, and the g = 0.04 point kills the quadratic at ~10 sigma.  Also checked
and clean: the peel really is exact -- `CentroidMarginalize.unmarginalize` slices
`condition[-3:]` (centroid.py:280) and its coefficient net takes only the moment
invariants `(r, k, q)`, so the layer never sees g and `centroid_transform`'s
zeroing of the shear slots costs nothing.

**What the centroid control now is, as a measured object:** a +0.006
multiplicative bias that is converged in S, flat in g, invariant to the merge
(exact single chunk), invariant to the layer's inverse iteration count,
unaffected by the mask seam and by the log-det, with a prior that is exactly
right by construction.  Flat in g means it is a pure RESPONSE-SCALE error --
`ghat` is uniformly 0.6% too large, i.e. R too small or Q too large by that
factor, at first order.  Nothing about it is a plumbing artefact.

**The next test, and there is already a hint at the answer.**  Every number
above is at the DEEP Sigma_X.  [[centroid-path-bias-clean]] records the SHALLOW
image-noise + centroid path as clean (corrected m1 -0.0035 +/- 0.0028), and the
layer's own marginalisation is only FIRST order in T (models/centroid.py:57),
with T proportional to Sigma_X.  So run the control's Sigma_X ladder --
shallow, deep, and a point or two between, everything else fixed.  If the +0.006
scales with Sigma_X it is the layer's first-order truncation showing up as a
response-scale error, which is a modelling limit with a known fix (carry the
next order in T) rather than a bug.  If it is flat in Sigma_X too, the layer's
strength is not what sets it and the response scale has to be attacked directly.

**Run (`--sigma-x-scale`, which dials Sigma_X alone and leaves C_M untouched, so
the control stays self-consistent at whatever Sigma_X it is told).  It is
NON-MONOTONIC:**

| Sigma_X | m1 | linear-in-T would give |
|---|---|---|
| x0.1 | +0.0048 +/- 0.0017 | +0.0007 |
| x0.3 | +0.0075 +/- 0.0017 | +0.0021 |
| x1.0 | +0.0070 +/- 0.0017 | +0.0070 (anchor) |
| x3.0 | **+0.0011 +/- 0.0018** | +0.0210 |

A bump peaking near the operating point and consistent with ZERO at 3x, with 0
dropped targets at every point -- so not a linear or quadratic truncation in T,
and not degeneracy at the extrapolated end either.  `logs_control_sxscan.txt`.

**And the depth confound is closed**: `load_cov` on the deep catalog equals
`load_cov` on the shallow one times 2.73^2 to 3e-8 per component.  The
moment-space control (clean) and the centroid control (+0.006) run at *exactly*
the same C_M, target count and estimator settings.  The only differences are the
layer and the trained flow itself.

Next: Sigma_X = 0, where `response` is exactly the identity ("U is the identity
at Sigma_X = 0", centroid.py:287) and the layer provably drops out.  Bias
persists there -> it is `centroid_deep.eqx`'s, not the layer's.  Bias vanishes ->
the layer is required for it, in a way that is not monotone in its strength.

**It vanishes.  The full ladder, six points, all with 0 dropped targets:**

| Sigma_X | m1 |
|---|---|
| x0.00 | **+0.0001 +/- 0.0016** |
| x0.03 | +0.0032 +/- 0.0017 |
| x0.1 | +0.0048 +/- 0.0017 |
| x0.3 | +0.0075 +/- 0.0017 |
| x1.0 | +0.0070 +/- 0.0017 |
| x3.0 | **+0.0011 +/- 0.0018** |

A clean bump over two decades: zero when the layer is exactly the identity,
rising to +0.007 by x0.3, gone again by x3.  **The layer is NECESSARY** (which
also clears the trained flow -- same flow, same targets, same C_M at every point)
**but the bias is not monotone in its strength**, so it is not a truncation order
in T and not the layer's magnitude as such.  `logs_control_sxzero.txt`.

Zero at both ends is the shape of a PRODUCT of two factors moving oppositely:
something that needs the layer to be active, times something that dies as the
layer smooths the density.  That is the shape to explain, and it is what made
the float32 hypothesis fit -- the layer supplies curvature and an 18-nat log-det
spread, while Sigma_X smooths the density and reconditions the cancellation.

**float32 is NOT it, by five and a half orders of magnitude.**
`dev/check_float32_bias.py` (new) runs the same 1000 targets and the SAME draws
through the identical estimator twice, once with float32 parameters and inputs
and once with float64, in one chunk so no merge is involved.  The draw values
are shared, so the difference is precision alone and needs no bootstrap:

    |dQ|/|Q| median 1.2e-07, p99 1.7e-05
    |dR|/|R| median 2.4e-07, p99 1.0e-05
    m1 (float32) = -0.01258511
    m1 (float64) = -0.01258513
    m1 shift     = +1.3e-08          against the 6e-3 to explain

`logs_precision_audit.txt`.  Note for whoever reads `pqr_streamed`'s docstring
next: its "1.7% on R in float32" is about the MERGE arithmetic across chunks at
low ESS, not the per-chunk flow evaluation -- and the merge was already cleared
by the single-chunk run above.  Watch the trap that cost a first attempt here:
casting only the INPUTS to float32 leaves float64 parameters to promote every op
back up, and the "float32" arm silently agrees with float64 to 3e-8.  Both flows
have to be cast.

---

## Where the centroid +0.006 stands, 2026-08-14 end of session

Eliminated, each with a number: the peel's log-det; the raw-vs-z mask seam and
`safe_point`; the streaming merge (exact single chunk); the layer's fixed-point
inverse at the converged draw count; the draw count itself (plateau on three
points); the PQR expansion's truncation (flat in g, 10 sigma); float32 (1.3e-8);
the trained flow (Sigma_X = 0 is clean with the same flow); and depth as a
confound (the two controls' C_M agree to 3e-8).

What survives, as measured properties the answer must reproduce:

1. **It needs the layer.**  Exactly zero at Sigma_X = 0, where the layer is
   provably the identity.
2. **It is not monotone in the layer's strength.**  Zero at x0, +0.007 at
   x0.3-1, zero again at x3 -- a product of two opposing factors.
3. **It is a pure response-scale error.**  Flat in g means `ghat` is uniformly
   0.6% too large: R too small or Q too large by that factor at first order.
4. **It is not a variance effect.**  Converged in S, immune to the merge, and
   the prior is exactly right by construction.

**The candidate that fits 1 and 2 and has NOT been tested: the layer's fold.**
(**Tested 2026-08-15 and DEAD -- see the next section.**)
`marginalize` is a fixed-point inverse of `unmarginalize`, and ~36 of 4000 rows
never converge at ANY `n_iter` -- which is the signature of a genuine
non-injectivity, not slow convergence.  The `--n-iter` A/B tests the iteration
count and would be null against a fold no matter how big the fold was, so that
result does not cover this.  A fold needs the layer (dead at Sigma_X = 0, point
1) and its effect on the estimator can turn over as Sigma_X grows if the density
smooths faster than the fold region grows (point 2).  The test is direct and
cheap: measure the round-trip |unmarginalize(marginalize(y)) - y| on the
control's own targets at each Sigma_X scale, and ask whether the failing FRACTION
traces the same bump.  If it does, the layer is not a bijection at these
Sigma_X and the fix is in `models/centroid.py`, not in `bias.py`.

**A candidate the shape fits, worth testing next either way: float32.**  The
per-chunk `f, df, d2f` come out of the jitted flow in float32 and only the MERGE
is float64 -- and `pqr_streamed`'s own docstring measures the float32 merge as
costing **1.7% on R**, against the 0.6% response-scale error we are chasing.  A
float32 cancellation error would be draw-count independent, g-independent,
sensitive to how sharp the density is (hence non-monotone in Sigma_X, which
smooths it), worse with the layer's extra curvature and log-det spread, and
invisible to every test run today.  Test it paired and small: a few hundred
targets, identical draws, `one` evaluated in float64 against float32, comparing
per-target Q and R and the ensemble m1 shift -- the difference needs no
bootstrap because the draws are common.

---

## 2026-08-15: the fold is REAL, and it is not the +0.006

`dev/check_layer_fold.py` (new), the test the previous session specified.
`flows/centroid_deep.eqx`, 20000 control targets per point, float64 (x64 on
after the load, as in `check_float32_bias.py`), round trip
`unmarginalize(marginalize(y)) - y` at `n_iter` 3 and 30, plus the sign of
`det d(unmarginalize)/dx` -- which is the only detector that separates a fold
from a lazy iteration, since an orientation flip is a PROOF of non-injectivity.
Run on the targets and on the targets plus one C_M draw, because the estimator
also unmarginalises every kernel draw and those reach much further out.

| Sigma_X | targets frac>1e-4 @it30 | det<=0 | +C_M frac>1e-4 @it30 | +C_M det<=0 | control m1 |
|---|---|---|---|---|---|
| x0    | 0       | 0      | 0      | 0      | +0.0001 |
| x0.03 | 0       | 0      | 0.0021 | 1.0e-4 | +0.0032 |
| x0.1  | 0       | 0      | 0.0035 | 1.0e-4 | +0.0048 |
| x0.3  | 0       | 0      | 0.0052 | 3.0e-4 | +0.0075 |
| x1    | 4.5e-4  | 0      | 0.0074 | 7.5e-4 | +0.0070 |
| x3    | 0.0035  | 2.0e-4 | 0.0141 | 1.4e-3 | +0.0011 |

1. **The fold exists.**  Rows with `det <= 0` are there, and `n_iter = 30`
   removes ~97% of the it3 residual everywhere *except* them -- which is exactly
   why the `--n-iter` A/B came back null and could never have covered this.
2. **It is MONOTONE in Sigma_X; the bias is a bump.**  At x3 the fold is 4-8x
   worse than at the operating point while m1 is +0.0011, and at x0.03 the bias
   is already half its peak with a fold ~30x smaller.  Over the top decade the
   two are ANTI-correlated.  The failing fraction does not trace the bump, which
   was the pre-registered discriminator.
3. **At the operating point the control's own targets are fold-free**: 0 rows
   with `det <= 0`, 4.5e-4 unconverged at it30.  Target generation is clean, an
   independent confirmation of the `--n-iter` null by a different instrument.

**The one channel that is NOT inert: the kernel draws.**  Same script, weight
share of Phat (real deep targets, alpha 0.5, 8192 draws, the
`check_safe_point_seam` construction).  "off-branch" = the fixed-point inverse
of a draw's image does not return the draw:

| Sigma_X | folded draws | targets with >1% of Phat folded | worst target |
|---|---|---|---|
| x0.03 | 1.2e-3 | 0.002 | 0.070 |
| x0.1  | 1.7e-3 | 0.004 | 0.055 |
| x0.3  | 2.6e-3 | 0.009 | 0.056 |
| x1    | 4.7e-3 | 0.020 | 0.248 |
| x3    | 9.6e-3 | 0.033 | 0.853 |

Unlike the mask seam (3e-10 of Phat), this is real weight: at the operating
point 2% of targets take more than 1% of their Phat from draws where the layer
is not injective, and the worst takes 25%.  On a folded branch the pushforward
is missing its other preimages, so `unmarginalize`'s log-det is not the whole
change of variables and the density there is simply wrong.  **Worth fixing on
its own** -- and note the estimator never INVERTS the layer on a draw, so the
fix is in the forward map's parametrisation (keep `response` a diffeomorphism at
large T), not in `marginalize`.  But this share is monotone in Sigma_X too, 2%
-> 3.3% where the bias goes +0.007 -> +0.001, and a few percent of targets
cannot produce a bias that is UNIFORM in g and flat across the population.

Logs: `logs_layer_fold.txt`.  **So candidate list for the +0.006 is empty again**,
and every "product of two opposing factors" story now has to name a second
factor that dies between x1 and x3 while the first grows monotonically.

---

## 2026-08-17: the control can only be biased four ways, and two more are now dead

No new simulation in this section: everything below is analysis plus arithmetic
on PQR already saved in `dev/`.

**The pruning argument, which should have been written down much earlier.**  In
the self-consistency control the targets are drawn from the flow and scored by
the same flow, and the noise is added with the same `C_M` the estimator
convolves with.  The prior and the likelihood are therefore both exact *by
construction*, so **no model misspecification of any kind can make m1 nonzero
there**.  That disposes, with no run at all, of the entire physics list: the
layer's first-order-in-T truncation, the `Var[.|m]` transport floor, centroid
noise bias, template incompleteness, selection, and the `|J(u)|`-inside versus
`J(M)`-outside question against the paper's eq. (35)-(36) -- which was checked
this session and is *consistent*: the paper's `J(M)` is a per-sky-area detection
density, the layer's `|J(u)|` is the per-detection recentring density, and
`INT |J(u)| N(X^G(u); Sigma_X) d2u = 1` under the same convexity assumption the
paper makes in section 5.3.  Two different bookkeepings of the same physics.

What survives is only four terms: `O(g^2)`, `O(1/N)` in TARGETS, `O(1/S)` in
draws, and numerics.  Numerics was already dead (1.3e-8) and `O(g^2)` was
already dead (flat in g at 10 sigma).  So:

**`O(1/N)` is dead too, and this is new.**  `ghat = -(sum r)^-1 (sum q)` is a
ratio of two finite sums and is biased at O(1/N) even with perfect per-target
`q` and `r` -- the target-side analogue of the paper's Fig. 4 `1/N_template`
effect, which nobody had checked.  `dev/check_finite_targets.py`, on the saved
deep centroid control: the jackknife O(1/N) bias is **|b/N| < 3e-6**, and the
subsample curve is FLAT over 32x in N (+0.00066 at N = 20000, +0.00132 at
N = 625, where a 1/N term would have grown 32-fold).  Same on the shallow
control.

**Which leaves `O(1/S)` as the only survivor -- and it has a named mechanism
that no ESS diagnostic can see.**  `pqr_streamed` forms `qhat = B/A` and
`jhat = -(C/A - (B/A)^2)`.  That `(B/A)^2` is the SQUARE of a Monte-Carlo
estimate, so

    E[jhat] = j - Cov_MC(qhat)

i.e. the observed information comes out systematically LOW, with no matching
term in `qhat`, and since `m1 = sum qhat / (g sum jhat) - 1` the estimator reads
systematically HIGH by

    dm1 = sum_i Cov_MC(qhat_i)_11 / sum_i j_i,11

**positive, exactly flat in g, and a pure response-scale error** -- every
fingerprint of the +0.006.  `dev/toy_is_bias.py` validates it against exact
quadrature on a 1-D problem: the predicted deficit tracks the measured one,
scales as 1/S (0.0504, 0.0132, 0.0034, 0.0009 at S = 512..32768), is dead flat
in g (0.01325 / 0.01324 / 0.01324 at g = 0.01/0.02/0.04), and grows steeply as
the density sharpens (6e-4 at spike width 1.0 to 0.12 at 0.05, fixed S).

Critically it is **invisible to `check_ess_logdet.py`**: the defensive mixture
bounds the importance weights, so ESS/S is flat in S exactly as that script
measured -- but what carries this bias is the variance of the RATIO `B/A`, not
the concentration of `w`.  The flat ESS/S at alpha = 0.5 is therefore *not*
evidence that this term has converged, and the earlier reading of it as such is
the gap in the elimination chain.

**Measured on the real runs, from two saved S values** (`dev/check_var_mc.py`,
which needs no new compute -- two runs differing only in S bracket `V/S` within
a factor <2 whether or not the draws are nested):

| path | predicted dm1 at S = 32768 |
|---|---|
| moment-space, deep | +0.00176 to +0.00294 |
| image-noise + centroid, deep | +0.00178 to +0.00301 |

So the term is **real and not small -- but it is the SAME in both paths**, and
therefore is not by itself the centroid path's differential +0.006.

**Two consequences that outlive that null.**

1. **"Converged in S" is a CANCELLATION, not a convergence.**  This one term is
   ~+0.0025 at S = 32768 and would be ~+0.010 at 8192, yet the measured net
   moves the other way (moment-space control -0.0048 -> -0.0002).  Competing
   O(1/S) terms are cancelling at the ~0.003 level, which is larger than several
   conclusions in this file rest on.
2. **Every error bar in this file is missing the draw realisation.**  The
   bootstrap resamples TARGETS and leaves each target's draws untouched, so it
   is structurally blind to the Monte-Carlo error of the integral.  The merge
   test already hinted at it (+0.0070 vs +0.0055 at the same S on a different
   stream) but it was read as a null rather than as a scatter measurement.

**A new free diagnostic worth keeping: the information identity.**
`dev/check_info_identity.py`.  Any normalised density obeys
`E_0[s s^T] = -E_0[h]`, i.e. `sum q^2 = sum j`, and arm-averaging leaves

    sum q^2 / sum j - 1  =  (g^2/2) Var_0(s_1^2 + h_11) / F  +  2 Var_MC(qhat)/sum j

| run | 11 | 22 |
|---|---|---|
| centroid control, deep | **+0.0566** | +0.0048 |
| centroid control, shallow | +0.0514 | -0.0005 |
| **moment-space CONTROL** | **+0.0372** | -0.0003 |
| moment-space REAL, S = 32768 | +0.0067 | -0.0003 |
| centroid real, S = 32768 | +0.0264 | -0.0046 |

**Corrected 2026-08-17:** an earlier version of this table put the moment-space
REAL run (`dev/pqr_depth_2.73_S32k.npz`, +0.0067) on the "moment-space control"
row and concluded the centroid path was **8.5x** more non-Gaussian in g.  Wrong
comparison -- control against real.  The moment-space CONTROL is +0.0372, so the
true ratio is **1.4x**, not 8.5x.  The qualitative statement survives; the
factor does not, and nothing should be built on the larger number.

The 22 component is ~0 everywhere, as it must be with shear in g1 only (there it
is a cross-covariance, not a variance) -- that split is what identifies the term
rather than leaving it as "the estimator is inconsistent", which is the trap.
Read as the first line, the centroid path's per-target `log P` is **~1.4x more
non-Gaussian in g** than the moment-space path's.  That is exactly the regime
the paper's eq. (61) warns about ("alpha is expected to be of order unity UNLESS
d log P/dg becomes large for some targets").  It is NOT the +0.006 itself --
alpha g^2 is excluded at ~10 sigma by the g-scan -- but it is the first
quantitative statement of what the centroid layer does to the estimator's
conditioning, and it costs nothing to compute on any saved PQR.

**The measurement this points at -- DONE, and it takes half the bias.**
`--draw-seed` added to `check_selfconsistency_noisy.py` (until now `--seed` moved
the targets too, so two runs could never differ in draws alone).  Two deep
centroid controls at S = 32768, alpha 0.5, 20000 targets, seeds 107 and 500,
identical in everything else (`logs_control_drawseed.txt`,
`dev/check_draw_seed.py`):

| | |
|---|---|
| m1, seed A | +0.00522 |
| m1, seed B | +0.00450 |
| **draw-realisation sigma** | **0.00051** (1 dof) |
| `sum Cov_MC(qhat)_11 / sum j_11` | **+0.00263** |
| m1, averaged, uncorrected | +0.00486 |
| **m1, `Cov_MC` SUBTRACTED from R** | **+0.00751 +/- 0.00158** |

**SIGN, settled in closed form -- an earlier version of this section had it
backwards.**  `j = qhat qhat^T - C/A` and `E[qhat qhat^T] = q q^T + Cov_MC`, so
`E[jhat] = j + Cov_MC`: **R comes out too LARGE**, ghat too small, and m1 too
NEGATIVE.  Correcting therefore moves m1 UP.  Verified on a Gaussian where
`P(M|g) = N(M; g, 1+s^2)` is exact, so `<jhat>` can be compared with `j`
directly -- it tracks `j + Var_MC` at every M and every S from 64 to 4096, to
3-4 digits.  The toy's own `Rdef` column was positive all along, i.e. a surplus,
and was mislabelled a deficit.

**So this term does NOT explain the centroid control's bias -- it deepens it**,
from +0.0049 to +0.0075.  It is real, it is worth removing, and the thread it
was supposed to close stays open.

Three things to keep straight before quoting that.

1. **The runs reproduce, and the anchor everyone has been quoting was a high
   draw.**  +0.0052 and +0.0045 land inside the historical cluster at this
   operating point -- +0.0053 (`niter12_S32k`), +0.0055 and +0.0060
   (`merge_test`), +0.0058 (`a05_S131k`), +0.0070 -- and the HUMP matches too
   (-0.0363, -0.0420 against -0.0364..-0.0407).  The **+0.0070 used as THE
   number throughout this file is the high member of a seven-run cluster whose
   mean is ~+0.0056.**  Quote +0.0056.
2. **The draw-realisation sigma is small**, 0.0005 against the target
   bootstrap's 0.0017.  That worry is closed -- the bootstrap does dominate --
   but note it could never have shown that by itself.
3. **The correction is not a universal fix and must not be applied blind.**  It
   removes only the `(B/A)^2` term.  Applied to the moment-space control it
   takes -0.0002 -> +0.0024 at S = 32768 and -0.0048 -> +0.0057 at 8192, i.e.
   away from the zero that path is known to converge to, and not to a constant.
   So other O(1/S) terms of comparable size and opposite sign are present, and
   the honest budget is SEVERAL terms of order 0.003.

**A tension that the sign fix mostly resolves.**  The information identity moved
only +0.0566 -> +0.0524 between S = 8192 and 32768, which looked incompatible
with a `Var_MC` changing 4x.  With the correct sign it is not: the identity is
`sum qhat^2 / sum jhat - 1`, and `Var_MC` inflates the NUMERATOR and the
DENOMINATOR alike (`sum qhat^2 = sum q^2 + V`, `sum jhat = sum j + V`), so with
`sum q^2 ~ sum j` at small g the two largely cancel and the identity is nearly
blind to this term.  It is therefore close to a pure measurement of the
`(g^2/2) Var_0(s_1^2 + h_11)/F` fourth-moment term, which strengthens rather
than weakens the "more non-Gaussian in g" reading above.  A ~7% residual
drop against a predicted ~0.8% remains unexplained.

**So the honest status of the centroid +0.006: still open, and slightly worse.**
It is ~+0.0056 uncorrected at the operating point and **+0.0075 +/- 0.0016** once
this artifact is removed.  The artifact is nevertheless real and worth removing
on its own account -- it is ~+0.0026 at S = 32768, comparable to the whole
effect being chased, and it rides on a term the ESS diagnostic cannot see.
**The consequence for the rest of this file is larger than the consequence for
this thread: at S = 32768 the machinery cannot resolve a multiplicative bias
below ~0.003, so every conclusion here resting on a difference smaller than that
needs re-reading.**  The way forward is to reduce the estimator's O(1/S) error
-- more draws, or a debiased `R` (now implemented, see the next section) -- not
to look for an eleventh mechanism.

---

## 2026-08-17 (cont.): R is debiased by a delete-one jackknife, not a cross-fit

`bias.pqr_streamed(jackknife=True)`, now the default.  `jackknife=False`
reproduces the old estimator.

**The obvious fix is a trap, and this is the useful part.**  The first attempt
was the direct one: `R = C/A - (B/A)(B/A)^T` has exactly one squared
Monte-Carlo estimate in it, so replace `(B/A)(B/A)^T` with a weighted
U-statistic over the chunks (which are independent draw sets),
`(bhat bhat^T - sum_c a_c^2 b_c b_c^T) / (1 - sum_c a_c^2)`, keeping only cross
terms between DIFFERENT chunks.  That works exactly as designed -- on the
closed-form Gaussian at M = 0, where the true `q` is 0 so `E[(B/A)^2]` is pure
Monte-Carlo variance, it takes the term from +0.00080 to +0.00001.

**And the total gets no better**, because `C/A` is a self-normalised ratio
carrying its OWN O(1/S) bias, measured at -0.00055 in the same configuration --
opposite sign, same order.  The plain estimator's net +0.00025 is those two
partly cancelling.  Kill one and the other stands uncancelled; which of the two
estimators then wins depends on the configuration.  This is the same "converged
in S is a cancellation" fact as above, now visible inside a single target.

**So debias the estimator as a whole.**  With `theta_hat` the full-sample value
and `theta_(e)` the value recomputed dropping chunk `e`,

    theta_jack = k theta_hat - (k-1) mean_e theta_(e)

removes the entire leading O(1/S) bias of any smooth function of the chunk sums,
whatever its source.  It costs nothing: `_merge_init` now keeps each chunk's
`(log A_c, B_c/A_c, C_c/A_c)` instead of only a running total (28k floats at 64
chunks and batch 64), and the leave-one-out sums are
`(bhat - a_e b_e) / (1 - a_e)`.  Both Q and R are corrected -- Q's own O(1/S)
bias is odd in g by isotropy, so it acts multiplicatively on m1 exactly as R's
does.  Targets where one chunk holds nearly all the weight keep the plain
estimator (the jackknife would divide by a vanishing `1 - a_e`) and are counted
in the run log.

**Measured end to end**, deep centroid control, n = 2000, S = 8192, alpha 0.5,
same targets and same draws in both arms:

| | m1 |
|---|---|
| `jackknife=False` (old estimator) | +0.00697 |
| `jackknife=True` | **+0.01635** |

a paired shift of +0.0094 against the +0.0105 predicted from `Var_MC/j` at
S = 8192 -- right sign, right size.  It moves m1 UP, so it makes the centroid
control's residual larger, exactly as the sign fix above says it must.

`tests/test_pqr_crossfit.py` (9 tests) pins the algebra (the plain path is still
the A-weighted ratio; the jackknife matches an explicit delete-one computation;
single-chunk and dominant-chunk fall back; dead chunks are dropped) and the
statistics on the closed-form Gaussian, where `j = 1/(1+s^2)` is known: the
jackknife more than halves the R bias at two values of M and beats the plain
estimator at two draw counts.  One test exists purely to guard the reasoning
above -- that the two O(1/S) biases are opposite in sign and within a factor of
4 of each other -- so nobody re-tries the cross-fit alone.

---

## 2026-08-17 (last): the estimator now PASSES its null, and the centroid bias survives

Three runs at S = 32768, n = 20000, alpha 0.5, draw seed 107
(`logs_control_fork.txt`), 0 dropped and 0 jackknife fallbacks throughout:

| control | jackknife OFF | jackknife ON |
|---|---|---|
| moment-space (`shear.eqx`, `--noise-scale 2.73`) | -0.0018 +/- 0.0017 | **+0.0001 +/- 0.0017** |
| centroid (`centroid_deep.eqx`) | +0.0052 / +0.0045 | **+0.0070 +/- 0.0018** |

Row 1 column 1 reproduces the historical -0.0002 +/- 0.0015 anchor within 0.7
sigma, so the configuration is right.

**The debiased estimator lands on zero where the answer is known.**  That is the
result: with the jackknife the moment-space control reads +0.0001 +/- 0.0017 --
not by a cancellation of competing O(1/S) terms, but because the leading one is
gone.  The prediction in the previous section that it would move AWAY from zero,
to about +0.0025, was wrong; extrapolating one term while ignoring the rest is
what made it wrong.

**And the same estimator gives +0.0070 +/- 0.0018 on the centroid path** -- same
target count, same C_M, same S, same alpha, same seed.  A 3.9 sigma difference
that can no longer be charged to the estimator's O(1/S), because the estimator
just passed its own null on the sibling path.  So the centroid control's bias is
REAL and is now isolated as cleanly as this machinery can isolate it.

**Which means the pruning argument has a hole, and it is findable.**  That
argument said prior and likelihood are exact by construction, leaving only
O(g^2) / O(1/N) / O(1/S) / numerics.  It is exact only if the centroid layer is
a BIJECTION -- and it is not.  This file's own fold measurement:
4.7e-3 of KERNEL DRAWS at Sigma_X x1 land on a folded branch, and **2% of
targets take more than 1% of their Phat from such draws, the worst 25%**.  On a
folded branch `unmarginalize`'s log-det is not the whole change of variables, so
`log_prob` there is simply the wrong density -- the estimator is not evaluating
the distribution that generated the targets.  Centroid-only, draw-count
independent, and untouched by everything eliminated so far.

It was dismissed earlier because the folded fraction is MONOTONE in Sigma_X
while m1 bumps.  **But that ladder was measured with the biased estimator**, and
the correction is Sigma_X-dependent (it scales with how sharp the integrand is),
so the bump shape itself may be an artifact of the old R.

**Next, and it is the highest-value run available:** the Sigma_X ladder again
with `--jackknife`, six points x0 .. x3.  If the bump flattens into something
monotone, the fold is the prime suspect again and the fix is in
`models/centroid.py`'s forward parametrisation (keep `response` a diffeomorphism
at large T), not in `bias.py`.  x0 comes free as a null, since the layer is
exactly the identity there.  ~2.5 h.

---

## 2026-08-17 (last): the bump is REAL, and the fold is still not it

The ladder, rerun with `--jackknife` at settings otherwise identical to the
2026-08-14 one, so those numbers are the paired jackknife-OFF arm.  S = 32768,
n = 20000, alpha 0.5, draw seed 107, 0 dropped and 0 fallbacks throughout
(`logs_control_sxjk.txt`, PQR in `dev/pqr_sxjk_*.npz`):

| Sigma_X | jackknife OFF | jackknife ON | shift |
|---|---|---|---|
| x0    | +0.0001 | +0.0019 +/- 0.0016 | +0.0018 |
| x0.03 | +0.0032 | +0.0050 +/- 0.0017 | +0.0018 |
| x0.1  | +0.0048 | +0.0066 +/- 0.0017 | +0.0018 |
| x0.3  | +0.0075 | **+0.0095 +/- 0.0017** | +0.0020 |
| x1    | +0.0070 | +0.0070 +/- 0.0018 | +0.0000 |
| x3    | +0.0011 | +0.0031 +/- 0.0019 | +0.0020 |

**The correction is FLAT in Sigma_X: +0.00188 over five points with a spread of
0.0002.**  The prediction that made this run worth doing -- that the correction
would be Sigma_X-dependent, so the bump might be an artifact of the biased R --
is therefore WRONG.  The whole ladder moves up rigidly and **the bump shape is
preserved exactly**.

The x1 row's apparent zero shift is not an exception, it is the anchor: the
historical x1 = +0.0070 is the high member of the seven-run cluster noted above.
Against this session's own jackknife-OFF measurement at x1 (+0.0052 / +0.0045,
mean +0.0049) the shift is **+0.0021**, in family with the other five.  So the
ladder independently reconfirms both that the correction is uniform and that
+0.0070 was an unlucky draw.

**Consequences.**

1. **The bump is real**, not an estimator artifact.  It survives an estimator
   that has been shown to read zero on a control where the answer is known.
2. **The fold is still ruled out by the pre-registered discriminator.**  Folded
   draw fraction is MONOTONE in Sigma_X (1.2e-3 -> 9.6e-3 over x0.03 -> x3);
   the bias is a bump peaking at x0.3 and falling by a factor 3 by x3.  Reviving
   it was the point of this run and it did not survive.
3. **The peak has sharpened and moved to x0.3** (+0.0095), with x1 at +0.0070 --
   the old "plateau over x0.3-x1" was the anomalous x1 anchor.
4. **"Zero at both ends" is now weaker.**  x0 reads +0.0019 +/- 0.0016 and x3
   +0.0031 +/- 0.0019, each 1.2-1.6 sigma from zero rather than on it.  x0 is
   still consistent with the moment-space control's +0.0001 (0.8 sigma apart),
   which it must be -- the layer is exactly the identity there, so that point is
   the underlying shear flow and nothing else.

So the object to explain is now: a **real** bump of amplitude ~+0.008 over a
~+0.002 pedestal, peaking at Sigma_X ~ 0.3x the deep value, measured with an
estimator that passes its own null.  Every monotone-in-Sigma_X mechanism --
which is all of them tested so far, the fold included -- is the wrong shape.

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
2. ~~**Does this reproduce in the noisy pipeline?**~~ **Answered 2026-08-13**
   — no, it survives at 21% and only 2.4 sigma; see the section above. Still
   open in one respect: that run was the moment-space `add_noise` + C_M path,
   chosen so the true Mr/Mf is exact. The full **image-noise + centroid-layer**
   path was not run, and would need `dev/check_bias_flatness.noise_free` for
   binning. Given the C_M result, its value is now confirmatory rather than
   decisive.
3. ~~**Does gauss2 show it too?**~~ **Answered 2026-08-13** — see the section
   above. Yes, and it relocated the boundary from the chart ceiling to the
   population's own support edge in the concentration direction.

4. **Is the leak the cause, or a co-symptom?** Partly settled: the Mc/Mr bound
   moved gauss2's leak not at all, but it was inactive there, so this is not
   yet a real test. A live test needs a bound on the direction gauss2 actually
   leaks across — the sampler's interior `rho` boundary — which no chart
   ceiling reaches.

5. **Why did the bump window get worse?** [3.40,3.46) went +0.0021 -> +0.0102
   under the bound, 28 sigma, while its neighbours improved. Worth one look
   before the bound is called a clean win.

6. ~~**Is the broad noisy hump the thing to chase now?**~~ **Chased
   2026-08-13** — it is the old ramp, it is real (immovable under 4x ESS,
   proposal and chart), and gauss2 does not have it. What is left is question
   1, now the top thread: **it is bulgedisc's flow density in the resolution
   direction**, so attribute it against the population's structural
   parameters. bulgedisc does not save `theta` on disk the way gauss2 does —
   getting `imsims.sim --pop bulgedisc` to save it (or finding a proxy) is now
   the blocking step for the highest-value lead.

8. ~~**Fix the 15.7% dead targets**~~ **DONE** — it was the `safe_point` bug,
   see above. A residual 1.07% remains, and it is a DIFFERENT population: the
   diffuse end (median true Mr/Mf 1.54 against 3.33 for the survivors), i.e.
   the same corner that produced the |R| spikes. Worth a look, but it is small
   and pre-existing rather than chart-induced.

9. ~~**Every binned noisy profile needs the control subtracted**~~ **DECIDED
   and implemented** — `--control` on `check_noisy_size_profile.py`. What is
   left is to apply it to the older results: `dev/sweep_noisy.py`'s arms,
   `check_bias_by_selection.py`, and anything else quoting a binned noisy m1,
   were all scored WITHOUT it and are inflated by roughly the control's
   contribution. Several past conclusions may need revisiting on that basis.

10. **Is there anything left to chase?** After the correction, the only
    surviving noisy structure is HUMP = +0.0105 +/- 0.0039 and per-bin
    residuals <= 0.006. That is close to the point where the noiseless
    diagnostics (which need no control, since binning on m is a function of
    what the estimator sees) are the more sensitive instrument. Consider
    redirecting effort there, and re-deriving what accuracy target actually
    matters.

10. **Why is gauss2's noisy tilt the opposite sign?** Its profile declines
   monotonically (+0.0187 -> -0.0635) where bulgedisc rises then falls. Both
   are flow-density errors, but a sign difference is a clue about what the
   convolution does to a density error, and gauss2 is the population where the
   exact answer is computable (`truth.py`). A noisy run scored against exact
   Q, R rather than the flow's would separate "flow error" from "what C_M does
   to any error" — feasible at alpha = 1.0, where the proposal is the kernel
   and no flow sampling is needed, but costly: `truth.pqr` runs a Newton solve
   per draw, so it needs far fewer targets and draws than 20k x 8192.

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

---

## 2026-08-19: two findings that supersede much of the log above

Written at the end of the 08-18/08-19 sessions, whose detail lives in the git
log (`159918b` back to `16ece5c`) rather than here.  Both of these are large
enough to invert earlier conclusions, so they go at the front of anyone's
reading order.

### 1. The bulk was undertrained, and that was setting the answer

`bulk.py train`'s 4000-step default — used by `retrain.sh` and by every flow
ever measured in this file — leaves the bulk visibly unconverged:

    bulk steps   val nll   window mass vs catalog   noiseless m1
          4000   40.4405                   +3.70%        -0.0126
         20000   40.4199                   +1.70%        -0.0227
         60000   40.4070                   +0.83%        -0.0428
        150000         -                   +0.61%        -0.0600

Both columns monotone.  The shear layer is *already* converged at its own
default (6000/20000/60000 steps on a converged bulk: -0.064/-0.056/-0.067), so
this is the bulk alone.  An undertrained bulk was partially cancelling the
shear layer's spin-0 response error against its own smoothness; converging it
removes the cancellation and the response error reads out in full.

**So `m1 = -0.06 +/- 0.01` (noiseless, bulgedisc) is this pipeline's honest
number, and every m1 above in this file is an undertrained-model number.**
Rebuild scatter across five converged rebuilds is sd 0.010 — do not read a
difference below ~0.02 between two chains.

By elimination the residual is **the spin-0 shear response**: shear steps,
template count (25k/50k/90k), derivative weight (1e4/1e5/1e6) and likelihood
overfitting are all flat, while `dm/dg` against bfd's exact derivatives is 17%
(Mf), 28% (Mr), 46% (Mc) against 2.6% for spin-2, and 100x more supervision
weight moves them by 0.07 percentage points.  `dev/check_a0_representable.py`
fits the same coefficient standalone — same net, inputs, bound — to 0.002
against +0.38 in flow, so capacity, conditioning and inputs are all cleared and
what is left is the joint optimisation.  That is the open architectural thread.

### 2. The noisy "hump" was a NaN gradient, not the flow's density

`pqr_streamed` passed `ok_raw` into `log_conv_is` unconditionally.  With no
centroid peel the draws are still raw moments, but that branch stands masked
rows in at `jnp.zeros` — a valid standardised coordinate, an invalid raw moment
(`Mf = 0` -> `log10(0)`).  The `where` hides it from the value; `0 * inf` NaNs
the **gradient**.  Every chunk of any target with even one off-chart draw was
marked bad, so 15.7% of bulgedisc silently got `Q = R = 0` — and those targets
sit at the chart edge, making the survivors a latent-selected sub-ensemble.

    control    +0.0159 +/- 0.0015 (3196 dead)  ->  +0.0003 +/- 0.0178 (0 dead)
    bulgedisc  +0.0131 +/- 0.0015              ->  +0.0008 +/- 0.0020

One line (`ok_raw = ... if layer is not None else None`).  Only the
**no-centroid streamed** path was affected (`--pop bulgedisc`, `gauss2`,
controls without `--centroid`); the image-noise path peels first and the
non-streamed `pqr` path was never affected — which is why the single-chunk run
always read -0.0024 while the streamed one read +0.013.

**This supersedes the "broad hump", "resolution bias is the bulk density" and
"resolution-edge bias is real" threads above.**  The tell is the
`N targets had no draw with any weight` line reading above ~0.  Any noisy
number in this file taken through `pqr_streamed` without `--centroid` needs
re-measuring before it is reasoned from.

Related and equally superseding, for the *binned* profiles: the m1-vs-M1/M2
parabola is not a bias at all.  BFD's estimator is consistent only for the
ensemble whose prior it uses, so binning on a latent quantity produces per-bin
structure under a perfect prior — verified identical in the exact-prior control
(+0.314), in gauss2 whose population m1 is +0.003 (+0.197), and unchanged by
the fix above.  See `NOISY_BINNING_BIAS.md`.

### And the thing that is now measurable: selection

Eq. (40)/(45)-(46) are implemented (`window_prob`, `selection_terms`,
`--window-size`/`--window-flux`), so a target window can finally be applied to
a bias measurement.  Numbers, caveats and the "own observed M only" rule are in
README.md's phase-4 section rather than here, since they are the current state
rather than an investigation.

### Open, in the order I would take them

1. **Why do noiseless (-0.06) and noisy-windowed (~0.00) disagree on the same
   flows?**  Plausibly the `C_M` integral averaging over fine density structure
   a point evaluation reads directly — but undemonstrated, and until it is
   demonstrated the method's accuracy is bracketed by the two rather than equal
   to the smaller.  Cheapest probe: a `--noise-scale` ladder on fixed targets,
   watching m1 walk from -0.06 toward 0.
2. **The spin-0 response, jointly.**  Finding 1 says this is the architecture,
   and `check_a0_representable.py` says the representation is not the limit —
   so the target is the joint optimisation (trunk sharing was cleared in
   `4d5a9a1`; only the joint fit itself is left).
3. **The flow's +4.1% window mass** against the templates' — one coherent
   normalisation error at the window edge, currently worth 0.002 in m1 and the
   only place `--window-terms flow` and `templates` disagree.
4. Question 1 of the old list (attribute the multi-lobe shape to bulgedisc's
   structural parameters) is **still open but much less urgent**: most of what
   it was chasing turned out to be findings 1 and 2 above.

---

## 2026-08-20: the 200k noisy measurement, and the guard that was set for 40k

The point of the run was precision: 199,969 targets (the whole
`bulgedisc_noisy` catalog, 12 dropped for non-converged recentring) against the
40k the selection-terms result was measured on, `S = 8192`, `--chunk 4096`,
window `2.2 < Mr/Mf < 3.2` and `2500 < Mf < 50000`, `flows/centroid.eqx`.
Median ESS 816, 2.6% of targets below ESS 10.  It produced a precision result
and a systematic one, and the systematic one comes first.

### `sane_targets`' 1000x threshold is calibrated for a 40k sample

At 200k the population's own tail reaches the guard: **max/median `|Q1|` is
1028** and `|R11|` is 1040, so the worst target sits exactly at the threshold
that was meant to be far above anything real.  (The earlier note that
"max/median is 13 for Q and 70 for R, so 1000x has never fired" was measured on
40k, where that tail is simply not sampled.)  The consequence is that the
shipped guard admits ~14 targets that then dominate the ensemble sums:

    guard      kept      unwindowed m1        windowed + eq.(45)/(46)
     1000x   199969   -0.01939 +/- 0.00554     +0.01900 +/- 0.00645
      300x   199963   -0.01392 +/- 0.00208     +0.01280 +/- 0.00182
      100x   199955   -0.01003 +/- 0.00120     +0.01280 +/- 0.00171
       30x   199939   -0.00975 +/- 0.00089     +0.01281 +/- 0.00176
       10x   199861   -0.00997 +/- 0.00069     +0.01285 +/- 0.00164
        5x   199566   -0.00915 +/- 0.00066            --

Everything from 100x down is one number; 1000x is a different number with a 5x
wider error bar, both of which are 14 targets out of 200000 (7e-5).  The 14 are
what the paper's sec. 2.5 describes -- 13 of them sit at `Mr/Mf` 3.6-3.8, i.e.
at and past the point-source ceiling where a noisy moment can land but a latent
one cannot, and one is a single 1.7e6 flux outlier against a population median
of ~5e3.

**The default is NOT changed in this commit.**  Dropping targets is a selection
and the threshold is a science judgment, not a bug fix; what is established
here is only that 1000x is not a safe place to leave it, and that the answer is
flat over two decades below it.

### The numbers, at the 100x guard

    unwindowed                 m1 = -0.0100 +/- 0.0012
    windowed, uncorrected      m1 = +0.0523 +/- 0.0005
    windowed + eq.(45)/(46)    m1 = +0.0128 +/- 0.0017

and the prefix walk is stable, so this is converged in target count rather than
still moving:

    first  20000   m1 = -0.01302 +/- 0.00377
    first  40000   m1 = -0.01560 +/- 0.00353
    first  80000   m1 = -0.01130 +/- 0.00211
    first 120000   m1 = -0.01098 +/- 0.00168
    first 160000   m1 = -0.01017 +/- 0.00140
    all   199955   m1 = -0.01003 +/- 0.00120

`P_s = 0.3033`, `R_s = 0.763 I` isotropic to 0.3% (off-diagonal 2.3e-3 against
a diagonal of 0.763), `Q_s` consistent with zero as isotropy requires.  The
selection correction is doing real work -- it moves the windowed number by
-0.040 -- and +0.0128 is what is left after it.

### This does not reproduce the 40k result, and the difference is the flow

The selection-terms session measured `-0.0021 +/- 0.0034` unwindowed and
`+0.00007`/`+0.00214` windowed-corrected on two chains.  Those were on the
**same targets** (row 0 of both saved PQR files is the same galaxy) but a
**different flow**: `dev/e2e/pqr_noisy_40k.npz` was written at 08:30 on
2026-08-19 and `flows/centroid.eqx` was retrained at 15:34 that afternoon, and
the per-target `Q1` between the two differs by a median of **50%**.  So the
comparison is not like for like, and the earlier "windowed corrected is
consistent with zero" is not evidence about the flow now on disk.

**So the honest current statement is `m1 ~ -1%` unwindowed and `+1.3%` windowed
on the converged chain, both many sigma from zero, not the `|m1| <~ 4e-3` this
repo has been quoting.**  The two disagree in SIGN, which is itself a lead: the
window keeps 30% of targets and the correction moves m1 by 0.04, so whatever is
wrong is not uniform across the population.

### What that changes about the noiseless/noisy gap

The gap is now smaller and differently shaped than the open question above
assumed: noiseless -0.06, noisy unwindowed -0.010, noisy windowed +0.013.  A
`--noise-scale` ladder is still the right probe, but it is now interpolating
between -0.06 and -0.010 rather than between -0.06 and 0.

---

## 2026-08-21/22: derivative supervision back on, the Mr/Mf 3.0-3.2 floor, and a gauss2 run that found two new bugs

**Uncommitted at the end of this session** — `bias.py`, `centroid.py`, `shear.py`
all have local changes; nothing below is on a branch yet.

### NLL alone cannot learn the spin-0 response; `deriv_weight` is back

Re-derived the self-consistency argument from scratch (target exactly
representable, bulk exactly correct, `Var[Q|m] = 0` by construction): NLL alone
recovers spin-2 (`alpha` 0.96-0.97) but not spin-0, because spin-0 response is
odd in `e` and cancels at `O(g)` on an isotropic population, appearing only at
`O(g^2)`, ~400x weaker than spin-2 at `g_max = 0.02`.  Every convergence knob
(steps, batch, `g_max`) makes it worse or does nothing.  Restored `shear.py`'s
`--deriv-weight`, regressing the layer's own `(dm/dg, d2m/dg2)` onto bfd's exact
per-template derivatives — never itself fit to anything, so this is supervision
against ground truth, not a second loss term to trade off.  `deriv_weight = 1e4`
recovers alpha = 1.00 in the self-consistency control; not a value to search.

### A real bug in `centroid.py`, found along the way

Training on `copies_bulgedisc_deep.fits` blew up (nll -> 1e19-1e25 by step
250-500).  Not a gradient explosion (per-row gradients all finite, max 124.8) —
the FORWARD `log_prob` on the untrained flow was already -4.4e10 for one
ordinary-looking copy, and walking it bijection-by-bijection found `|z|`
cascading 5 -> 195 -> 296954 across the bulk's autoregressive stack (layer 15
of 16), pre-existing and unrelated to centroid or derivative supervision.
Fixed with a second `in_domain`-style gate in `step()`: evaluate `log_prob`,
flag rows with `lp <= -1e4` (`stop_gradient`'d), substitute `safe_point`, and
re-evaluate before computing the loss. Stable through the full 16000-step run.

### The bulgedisc-deep measurement, corrected

`flows/centroid_derivsup_deep.eqx` (deriv-supervised shear + the centroid fix),
200k targets at `noise_sigma = 2.73`, `alpha = 0.5`, window `(2.2,3.2)` x
`(2500,50000)`:

    unwindowed              m1 = -0.02407 +/- 0.00482
    windowed, uncorrected   m1 = +0.05817 +/- 0.00090
    windowed, corrected     m1 = -0.00913 +/- 0.00205

Residual m1 is not primarily `Var[Q|m]`: score-contracted first-order error is
89% reachable, not floor. It is not under-convergence either (6000 -> 36000
steps moved it negligibly). It concentrates entirely above Mr/Mf ~3.0, flat
below — see next section.

### The Mr/Mf 3.0-3.2 response-fit floor: real, and it is bulge/disc misalignment

Isolated to **second-order (curvature, R)** spin-2 response specifically — Q
(first order) is actually *better* in 3.0-3.2 than the 2.4-3.0 control.
Solved `models/shear.py`'s own `(mu, nu, rho)` ansatz **exactly per galaxy**
(no population averaging, no training) against bfd's true Hessian:

    per-galaxy exact-fit residual:    control 16.28%    target 27.13%

Sanity check on the same method applied to the (nearly-exact) first-order
`(A, B)` case gave 1.23% — small, consistent with float32 noise, validating the
method. Correlated the residual against the population table's hidden
variables (not visible to the model, which only sees combined moments):

    corr(residual, bulge_frac*(1-bulge_frac))  control 0.50   target 0.43
    corr(residual, |bulge_e - disc_e|)         control 0.15   target 0.15

Same magnitude in both bins, so it is not that harder galaxies are more common
in the target bin — the *same* hidden structure produces a bigger curvature
ambiguity as resolution degrades. Mechanism: a less-resolved galaxy's weight
function samples fewer independent k-modes before the point-source ceiling, so
the same bulge/disc asymmetry leaves a bigger blind spot in what the moments
can see.

**The trained network is already at its achievable ceiling, not 20 points
short of it.** First read (network 36-47% vs the 27% per-galaxy floor) looked
like a large reachable gap. It was comparing against the wrong bound: no
function of the *measured* invariants — however flexible — can beat what a
single **pooled constant** `(mu, nu, rho)` for the whole bin already achieves,
because the per-galaxy floor assumes information (which hidden bulge/disc
split) that is not in the moments. Pooled-constant fit: 32.98%/36.36%
(control/target) — matching the trained network's 31.5%/36.1% almost exactly.
Three confirmations that there is no reachable gap left:

- 6000 -> 36000 steps: 31.50/36.08 -> 31.46/35.58 (flat)
- Flattening the training density across Mr/Mf (equal gradient mass per bin
  instead of following the population): 31.50/36.08 -> 30.95/37.45 (no
  improvement, target got slightly worse)
- Three independent training seeds: 31.50/36.08, 31.91/36.57, 31.40/36.31 —
  spread under 0.5 points, well inside the pooled-fit floor's own range

### gauss2 control: the mechanism confirmed, cleanly

gauss2 is co-elliptical by construction (bulge and disc share one `e`) — no
hidden misalignment to leak. The per-galaxy ansatz-fit residual, computed the
same way, is **0.000% at every Mr/Mf from 2.0 to 3.6**, exact analytically
(gauss2 carries bfd's *exact* autodiff derivatives, not the DG*DG* columns —
verified those are fine now too, <0.05% error, contra a stale docstring in
`imsims/analytic.py`: the Nuttall-coefficient fix already killed the
moving-boundary bug it describes). Trained a full fresh gauss2 stack
(`bulk_gauss2_derivsup.eqx` 150k steps, `shear_gauss2_derivsup.eqx`
`deriv_weight=1e4`, `centroid_gauss2_deep.eqx` 16k steps on a freshly-rendered
`copies_gauss2_deep.fits`/`targets_gauss2_deep_g*_200k.fits` at the same
`noise_sigma=2.73`). The trained shear layer's own `dm/dg2` check: M1/M2
residual 2.64%/2.73% aggregate, against bulgedisc's 31-36% in the same metric.
Mechanism confirmed.

### But the full noisy gauss2 run surfaced two new, unrelated bugs

**Bug 1 — `bias.py --train-data` silently defaults to bulgedisc's
`moments.fits`.** `train_data = a.train_data or f"{data_dir}/moments{'_sersic'
if pop=='sersic' else ''}.fits"` special-cases literally the string "sersic"
and nothing else. Running `--pop gauss2_deep` without `--train-data` gave a
selection-term correction byte-identical to the bulgedisc_deep run (P_s, Q_s,
Q_s_err all matched to displayed precision) because `selection_terms`'
`--window-terms templates` path lenses whatever `train_data` resolves to.
Flow evaluation itself is unaffected (`eqx.tree_deserialise_leaves` overwrites
every array leaf regardless of what dummy data built the tree), but the
selection correction is silently wrong for any non-bulgedisc, non-sersic
population. Not fixed in code yet — worked around by passing `--train-data`
explicitly. **Should be fixed**: derive the default from `a.pop`, or require
`--train-data` outright when `--pop` isn't "bulgedisc".

**Bug 2 — the trained centroid layer is broken for gauss2 at low flux.**
Corrected gauss2_deep run, `P_s = 0.6324` (now matches the 65% window fraction
measured directly on the noiseless catalog), windowed corrected `m1 = +0.06585
+/- 0.00097` — an order of magnitude worse than bulgedisc_deep. Flux-quintile
breakdown:

    q1  +5.6715 +/- 0.3284   q2  +0.3153   q3  +0.0909   q4  -0.0136   q5  -0.0172

Not outlier-driven — dropping the 50 worst `|R|` targets in q1 only moves it
5.67 -> 5.30. Diagnosed by ablation:

- Noiseless, no centroid (plain `shear_gauss2_derivsup.eqx` on the 1M noiseless
  gauss2 catalogs): all quintiles sane, -0.0094 to -0.0248. The trained
  bulk/shear flow is fine.
- Noisy, `--no-centroid` (same noisy `gauss2_deep` targets, `alpha=0.5`,
  `samples=8192`, but the pre-centroid flow): q1 = +0.0318 +/- 0.0057, all
  quintiles sane. ESS was equally healthy with and without centroid (~630-670
  median either way), so it is not an integration problem.

So it is the **trained centroid layer specifically**. Likely cause, not yet
verified: gauss2's population is built by rejection sampling in moment space
(~30% acceptance, `sample_population_gauss2`), which plausibly leaves the copy
catalog thin at low flux, giving the centroid layer a poorly-conditioned
marginalisation there — matches `centroid.py check()`'s own diagnostic, which
(before the low-flux bug was known) already showed a 20x mismatch between the
layer's predicted ellipticity response and the catalog's own (+1.70e-1 vs
+8.4e-3).

**Bug/finding 3 — a generic bulk-density blowup at each population's OWN
support edge, unrelated to bulge/disc misalignment.** Binning the *noiseless*,
no-centroid `bias()` (not the ansatz-residual metric) by Mr/Mf on the bright
half of each population:

    gauss2     (max Mr/Mf ~3.588):  2.4-3.0 -0.001   3.0-3.2 -0.017   3.2-3.6 -0.127
    bulgedisc  (max Mr/Mf ~3.68):   3.0-3.2 -0.002   3.2-3.4 -0.045   3.4-3.6 -0.113   3.6-3.8 -0.408

Same shape in both populations, each one blowing up sharply right at its own
support ceiling, present with *no* noise, *no* centroid layer, and (for
gauss2) *no* misalignment. This is a different, larger-scale effect than the
misalignment-driven second-order floor above — it dominates the raw `bias()`
number at the extreme tail and looks like the bulk density's known difficulty
near the `POINT_SOURCE`/`POINT_SOURCE_MC` chart ceiling (`[[mc-mr-ceiling-bound]]`,
`[[resolution-bias-is-bulk-density]]`), not the response layer at all.

### Open, in the order I would take them

1. **Fix `bias.py --train-data`'s default** — cheap, and it will silently
   corrupt the next non-bulgedisc selection-term measurement otherwise.
2. **Find the actual code path behind the centroid layer's low-flux failure**
   on gauss2 — the rejection-sampling/thin-copies hypothesis is plausible but
   unverified; check copy density vs flux directly, or retrain centroid with
   flux-stratified sampling (same trick already tried and shown NOT to help
   the Mr/Mf floor — worth checking whether it behaves differently here).
3. **The point-source-ceiling bulk-density blowup** — present in both
   populations, independent of everything investigated this session. Probably
   the same open mechanism as `[[resolution-bias-is-bulk-density]]`, just
   re-surfaced with cleaner evidence (matched shape across two populations).

---

## 2026-08-22 continued: all three open items closed or re-diagnosed

### Item 1 — `bias.py --train-data` is now population-aware

Replaced the `"sersic"`-only special case with an explicit `TRAIN_DATA` dict,
keyed by every `CATALOGS` population, mapping each to the catalog its flows
were actually standardised on (`bulgedisc`/`bulgedisc_noisy`/`bulgedisc_deep`
all share `moments.fits`, since noisy/deep are the same underlying population
just measured with image noise; `sersic` keeps `moments_sersic.fits`;
`gauss2`/`gauss2_2k`/`gauss2_deep` all get `gauss2_g0_1M.fits`/`gauss2_g0_2k.fits`
— there is no `moments_gauss2.fits` on disk, and the zero-shear 1M/2k catalog is
what a `shear.py train --data ...` for gauss2 would actually have used). All
four files verified present. No more silent bulgedisc fallback.

### Item 2 — the gauss2 centroid low-flux blowup: NOT thin copies, an old
tanh gradient-trap left in `models/centroid.py`

**The rejection-sampling/thin-copies hypothesis is refuted.** Measured copy
count and per-galaxy weight-sum (`CopySampler.total`, the detection
probability) vs flux directly on `copies_gauss2_deep.fits`: copies are
**denser** at low flux (median 302/galaxy in the faintest quintile vs 196 in
the brightest), and the weight-sum/ESS profile is close to identical to
`copies_bulgedisc_deep.fits`'s (q1 `frac(total<0.9)` 2.8% gauss2 vs 1.8%
bulgedisc; both populations have the same `min=7` copies at all — a 0.22%
tail, not a systematic thinness). Whatever breaks gauss2's centroid layer is
not the copy catalog.

**What it actually is**: re-ran `centroid.check`-style diagnostics per flux
quintile on the trained `centroid_gauss2_deep.eqx`. The ellipticity-response
mismatch (layer vs catalog) is 24x at q1 (0.716 vs 0.029) falling to 11x at
q5 — bad everywhere, worst at low flux, consistent with the HANDOFF q1
m1 = +5.67. The predicted `|de|` shift **saturates at a hard ceiling** (~0.078
physical, ~90th-100th percentile all pinned there) — the signature of a
bounded activation, not a smooth extrapolation error. Evaluating
`models/centroid.py`'s `_Coeffs` net directly on the trained flow: coefficient
`D` is saturated (`|D| > 11` of `_COEFF_MAX = 12`) in **79-100% of galaxies at
every flux quintile**, and `A`/`B` saturate 20-33% at the flux extremes — vs
bulgedisc's trained layer, where `A`/`B` never saturate and `D` only reaches
43% in the brightest quintile.

Re-tracing training from a fresh warm start (`dev/spin0_selfconsistent_levers.py`-
style, snapshotting `|coeff| > 11` fraction every training checkpoint):
saturation is **0% at step 0 and grows monotonically to step 16000**
(D: 0%→33%→92%; A: 0%→9%; B: 0%→14%). This is not an initialisation or
data-density artifact, it is a **training-time runaway**.

`models/centroid.py`'s `_Coeffs.__call__` was still bounding with
`_COEFF_MAX * jnp.tanh(net(u) / _COEFF_MAX)` — the exact form
`[[spin0-failure-is-gradient-death]]` diagnosed and replaced in
`models/shear.py`'s `_Coeffs` on 2026-08-20: `tanh`'s gradient `sech^2(x/C)`
decays exponentially, so once a coefficient saturates it gets zero gradient
and can never recover — "a one-way ratchet." `models/shear.py` fixed this with
the rational bound `x / sqrt(1 + (x/C)^2)` (same asymptote and unit slope at
the origin, decays only as `(C/x)^3`). **`models/centroid.py` never got the
same fix.** Applied it (same one-line change, `models/centroid.py` `_Coeffs`).
Full `tests/test_centroid.py` (11 tests) still passes.

Re-traced training with the fix: saturation growth is sharply reduced and
plateaus rather than ratcheting away — at step 16000, `D` 39% (vs 92%
unfixed, still climbing), `A` 6.5% (vs 9%), `B` 5.6% (vs 14%), and D's growth
visibly flattens after step ~4000 rather than continuing to climb. **Not a
full fix** (D still saturates in ~40% of galaxies) — plausibly `_COEFF_MAX = 12`
is itself undersized for gauss2's D coefficient, the same kind of
under-fitting `[[coeff-bound-per-template-trap]]` found for the shear layer's
bound. Have not yet done the full `centroid.py train --steps 16000` retrain +
`bias.py --pop gauss2_deep` remeasurement — that is the natural next step to
quantify how much of the +5.67 q1 m1 this recovers.

### Item 3 — the support-edge blowup: not a mass leak, the score degrades in
the sparse tail (same mechanism at a softer edge)

Checked whether the bulk flow's Mr/Mf marginal **leaks mass past each
population's own empirical max** (the mechanism `[[resolution-bias-is-bulk-density]]`
found at the hard `POINT_SOURCE` ceiling, before that ceiling was logit-bounded).
Drew 200k-1M samples from `shear_derivsup.eqx`/`shear_gauss2_derivsup.eqx` at
g=0: **no leak** — sample max sits at or below the true population max for
both populations (bulgedisc samples reach 3.6863 vs true 3.6750, 0.0095% of
mass past it; gauss2 samples don't even reach the true 3.5883 max). Binning
the marginal density in narrow bins across the top 15% of each population's
range: ratio of flow density to true density is within a few percent
everywhere except the last 1-2 bins, where sample counts are single digits.
**The chart's Mr/Mf and Mc/Mr ceilings are doing their job — this is not the
old unmodelled-boundary bug.**

What IS wrong: the shear response layer's own **first-order score** (`A =
d(M1+iM2)/dg1 / Mr`, compared to bfd's exact per-galaxy `dm_dg` column, at
g=0, no noise, no centroid — same method as `dev/check_shear_response_vs_truth.py`,
which is stale against the current `dm_dg(layer, m, chart)` signature and
needed patching to run). Binned in narrow strips approaching each
population's true Mr/Mf max: fractional error in `A` grows from ~1-2% in the
mid-range to **-54% in the last 8 galaxies before bulgedisc's true max**
(monotone: -0.6%, -2.2%, -12%, -18%, -54% over the last five strips, n = 5954,
81, 68, 43, 25, 8) and to +6.4% by gauss2's much thinner tail (n drops to
single digits past Mr/Mf 3.48, too few to bin further). **This is not a
density-mass problem, it is the trained response network extrapolating badly
over the handful of training galaxies that exist in the last ~1% of each
population's own support** — `deriv_weight=1e4` derivative supervision is
active and still leaves this residual, meaning even exact per-galaxy targets
aren't enough training signal when there are only 8-40 galaxies to fit against.
Same shape in both populations because it's a generic small-N-near-a-hard-edge
effect, not specific to bulge/disc misalignment or to gauss2's construction.

**Not yet tried**: oversampling/reweighting the training batch near the edge
(the equivalent trick was tried for the Mr/Mf 3.0-3.2 misalignment floor and
did NOT help there — a different mechanism, so worth retesting here
specifically for the response layer rather than the density).

### The gauss2_deep retrain + remeasurement (done same session)

Backed up the old checkpoint to `flows/pre_tanh_fix/centroid_gauss2_deep.eqx`,
retrained `centroid.py train --copies copies_gauss2_deep.fits --flow
flows/centroid_gauss2_deep.eqx --init flows/shear_gauss2_derivsup.eqx --steps
16000` (matching `retrain.sh`'s bulgedisc convention) with the rational-bound
fix in place, then re-ran the full 200k-target `bias.py --pop gauss2_deep
--samples 8192 --alpha 0.5 --chunk 4096 --window-size 2.2 3.2 --window-flux
2500 50000` (now with the fixed `--train-data` default too, so both bugs are
addressed in this one number):

    unwindowed              m1 = -0.02133 +/- 0.00050
    windowed, uncorrected   m1 = +0.00052 +/- 0.00040
    windowed, corrected     m1 = -0.01598 +/- 0.00079

Against the old, doubly-contaminated +0.06585 +/- 0.00097: sign flips, and the
faintest-flux catastrophe is gone entirely —

    Mf quintile     m1              (was: q1 +5.67, q2 +0.32)
       q1        +0.0003 +/- 0.0053
       q2        -0.0303 +/- 0.0014
       q3        -0.0270 +/- 0.0009
       q4        -0.0237 +/- 0.0009
       q5        -0.0163 +/- 0.0005

q1 (the exact bin that blew up to +5.67) is now consistent with zero. What's
left is a fairly uniform -0.016 to -0.030 across q2-q5 — a DIFFERENT shape
than bulgedisc's residual (which concentrates at the high-Mr/Mf tail, not
spread across flux), consistent with the centroid layer's D coefficient still
being ~40% saturated even under the rational bound (measured in the training
re-trace above). Overall magnitude (-0.016 windowed-corrected) is now the same
order as bulgedisc_deep's -0.00913, not 7x larger and opposite-signed.

**Runtime note**: the first attempt at this `bias.py` run was killed by the
harness partway through (right at the tail end of the first arm, ~30 min in,
no traceback, GPU/CPU both idle at the instant checked) — looked like an
idle-output watchdog on an auto-backgrounded command rather than a crash.
Confirmed by re-running fully detached (`nohup ... & disown`, polled by PID
and log file directly rather than the tool's own background-task tracking),
which completed cleanly end to end (~2 hours for three 200k-target arms at
`samples=8192`). If a long `bias.py` run needs to survive unattended, detach
it this way rather than relying on the harness's auto-background promotion.

### The uniform q2-q5 residual: NOT centroid, NOT new — it's `rho` still
saturating `_COEFF_MAX` in both populations' shear layers

Checked whether the centroid layer's remaining ~40% D-saturation causes the
residual, by comparing the shear response evaluated at the layer's
(imperfect) unmarginalized point vs. the catalog's true one: the mismatch is
**largest in q1 (-4.3%) and smallest in q5 (-0.07%)** — backwards from what
would be needed to explain a residual that's ~zero in q1 and worst in
q2-q5. Wrong mechanism.

Ran `bias.py --pop gauss2 --flow flows/shear_gauss2_derivsup.eqx` directly —
the full 1M-galaxy **noiseless** population, no image noise, no centroid, no
MC integration at all:

    q1  -0.0094   q2  -0.0120   q3  -0.0142   q4  -0.0155   q5  -0.0248

Same sign, same order of magnitude as the full noisy/windowed/corrected run
(-0.0163 to -0.0303). **The residual is not introduced by noise integration or
the centroid layer — it's already there in the plain trained shear response.**

Checked the shear layer's own trained coefficients (`f,a,b,q -> layer.coeffs`)
for saturation against `_COEFF_MAX = 12`: **`rho` (the second-order spin-2
curvature coefficient, `coeffs[13]`, the same `[mu, nu, rho]` ansatz from the
misalignment-floor investigation) is pinned at the bound for ~73% of gauss2
galaxies, essentially flat across all five flux quintiles (72-74%)** — a much
better shape match to the flat residual than anything flux-dependent. Checked
`flows/shear_derivsup.eqx` (bulgedisc) the same way: **its `rho` saturates
even harder, 78% at q2-q5** (`c6`, a spin-0 coefficient, also saturates
54-59% there, vs <2% in gauss2). So this is not gauss2-specific — it is a
**shared defect in the trained shear response**, the same category of issue
`[[coeff-bound-per-template-trap]]` diagnosed for `a_Mr` on 2026-08-20 (that
fix, or its supersession by the chart-Jacobian reparameterisation, evidently
never touched `rho`). It was invisible in bulgedisc's aggregate because the
larger, already-diagnosed misalignment floor dominates there; gauss2 has zero
misalignment, so this shared floor shows up nakedly instead of being buried
under something bigger.

### Why `rho` keeps saturating: confirmed the same un-patched bug as `a_Mr`
for bulgedisc, but gauss2 has something else on top

Binned `rho` saturation by `a` (=z1, the Mr/Mf-related chart coordinate with
the divergent logit Jacobian — the same one that broke `a_Mr`):

    bulgedisc  Mr/Mf  0.80-2.92  2.92-3.24  3.24-3.38  3.38-3.49  3.49-3.68
               sat      2.4%       50.2%      94.8%      98.7%      99.9%

A clean climb from near-zero to total saturation approaching the point-source
ceiling — the identical signature `dev/flexibility_audit.py` found for
`a_Mr` before its fix. **`rho` (and by construction `A, B, mu, nu`, which
share the same unprotected path) has the same chart-divergence bug, just
never patched** — `_chart_spin0_jac`'s analytic-Jacobian treatment was only
ever applied to the spin-0 block.

Gauss2 is messier — NOT a pure ceiling effect. Same binning:

    gauss2     Mr/Mf  1.95-2.74  2.74-2.93  2.93-3.08  3.08-3.23  3.23-3.59
               sat      87.7%      68.1%      63.7%      63.6%      82.6%

U-shaped, and elevated (>60%) even far from any ceiling, unlike bulgedisc's
near-zero baseline there. Binning by `|e|` instead: bulgedisc's saturation
tracks it cleanly (94.8% at the roundest quintile down to 22.5% at the most
elongated — the "dividing a residual by a small e^3" mechanism the `a_Mr` fix
comment already warned raising the bound would unleash), but gauss2's is
flat at ~73% regardless of `|e|`. So gauss2 has the same underlying
vulnerability (no protective Jacobian) PLUS something else keeping it
elevated basically everywhere in its own (narrower, more homogeneous)
population — not yet isolated.

### Open, in the order I would take them

1. **Extend the spin-0 chart-Jacobian fix (`_chart_spin0_jac` /
   `[[chart-jacobian-reparam]]`) to the spin-2 block (`A, B, mu, nu, rho`)** —
   confirmed root cause for bulgedisc's `rho`, and the natural next step given
   how directly this mirrors the already-fixed `a_Mr` case. Derive the raw
   physical (mu, nu, rho) via the same kind of per-galaxy exact solve already
   done for the misalignment-floor ansatz fit, work out what divergent factor
   the current z-space parameterisation implicitly carries, and apply it
   analytically instead of asking the network for the compound quantity.
2. **Gauss2's extra, non-ceiling elevation in `rho` saturation** — separate
   from the chart-divergence mechanism above; check whether it's a
   conditioning/whitening issue (the four `_Coeffs` inputs are whitened as a
   group, but `rho`'s own natural scale relative to `mu`/`nu`'s isn't treated
   specially) or a genuine property of gauss2's narrower population.
2. **`_COEFF_MAX` for `models/centroid.py`'s `_Coeffs`** — D is still ~40%
   saturated post rational-bound-fix; a smaller, separate lever from the
   above, for the centroid layer's own residual overshoot (still measurable,
   just not the driver of the q2-q5 shape).
3. **The support-edge score degradation** — try training-batch reweighting
   toward the sparse tail for the shear response layer specifically (not the
   bulk density, where the equivalent trick already failed for a different
   floor); if that doesn't move it, this may just be an intrinsic small-N
   limit worth documenting rather than chasing further.
3. `dev/check_shear_response_vs_truth.py` is stale (`dm_dg` gained a required
   `chart` argument on 2026-08-20) — worth a real patch if it's going to keep
   getting reached for.

---

## 2026-08-22 correction: `rho`'s "chart-divergence bug" diagnosis above is
wrong; the real mechanism, and a negative experimental result

Item 1 above ("extend the spin-0 chart-Jacobian fix to spin-2, confirmed
root cause") does not survive scrutiny. Tried it, in a separate clone
(`../bfd_cnf_rho_polar`, not merged — kept there with its own `EXPERIMENT.md`)
rather than in this repo.

**The premise was wrong.** z3, z4 (the spin-2 slots `rho` acts through)
carry no logit and no divergent Jacobian, unlike z1/z2 — so `rho` cannot
have `a_Mr`'s exact bug. The earlier claim rested on `rho`'s saturation
tracking Mr/Mf-ceiling proximity in bulgedisc (2.4% → 99.9%), but that
is a POPULATION correlation (bulgedisc's near-ceiling galaxies happen to
have smaller `|e|`), not evidence of the same divergent-Jacobian mechanism.

**Tried the fix anyway**: replace `e` with a bounded direction
`e_dir = e / sqrt(|e|^2 + EPS^2)` in `rho`'s term. A first version applied
the same substitution to `B, mu, nu` too and was catastrophic (bulgedisc
noiseless m1 -0.0884 → -0.44) — `B*e^2*gcc` is a real, population-wide
effect (the reduced-shear term), and forcing its `e` to saturate at unit
magnitude destroyed its natural `|e|`-proportional scaling; training
responded by crushing `B` toward zero rather than fit an impossible target.
The `rho`-only version is clean (saturation 69% → 0%, `A`/`B` unchanged in
scale) but still makes the actual bias WORSE: m1 -0.0884 → -0.1826 (6000
steps) → -0.1651 (24000 steps) — not a convergence artifact, since the
second-order (`d2m/dg2`) M1/M2 residual stayed pinned at ~24.8% regardless
of step count while the first-order residual kept improving.

**The real mechanism**, found by solving the exact 3x3 `(mu, nu, rho)`
system per galaxy directly from bfd's own Hessian (no network at all):
`mu, nu` stay bounded (median 0.4-0.9) at every `|e|`, matching their 0%
saturation — but **`rho` genuinely needs to be enormous for most of the
population** (median 64 at `|e|` < 0.01, p99 in the tens of thousands).
Not an artifact. But that exact solve assumes real coefficients, which is
only valid as a population average over an isotropic ensemble (the
equivariance argument); solved per individual galaxy, `rho` comes out with
a large IMAGINARY part (up to 65% of magnitude at small `|e|`) — meaning a
single galaxy's true curvature has structure the real-coefficient ansatz
cannot represent at all, and this non-equivariant residual correlates
(r ~ 0.17-0.22) with the same bulge/disc misalignment variables
(`bulge_frac*(1-bulge_frac)`, `|bulge_e - disc_e|`) the Mr/Mf floor
investigation already used, on top of the dominant effect that `|e|` is
simply small, amplifying whatever's in the numerator regardless of origin.

So `rho`'s regression target is a population mean of a partly
non-equivariant, `1/e^3`-amplified per-galaxy quantity — genuinely noisy,
not a clean signal merely lacking room to be represented. The saturating
bound functions as an accidental shrinkage estimator against that noise;
removing it lets training fit the noise more precisely, which is why the
bias got worse rather than better.

**Revised open items, replacing item 1 above:**

1. Don't re-attempt an `e`-substitution/floor fix for `rho` expecting a
   better epsilon to solve this — the underlying problem (a noisy, partly
   non-equivariant target) doesn't depend on where the floor sits.
2. If `rho`'s noise is genuinely misalignment-correlated, the honest fix
   may be the same one item 0 already closed out for the Mr/Mf floor: not
   fixable by architecture or training changes, just a property of the
   population's hidden structure leaking through the moments. Worth
   checking whether `rho`'s saturated (bound-pinned) value is actually
   CLOSE to the population-mean-optimal shrinkage target, or whether there
   is still headroom a smarter (not merely unbounded) estimator could
   recover — e.g. an explicit shrinkage/prior on `rho` rather than a hard
   cap, which is a different structural change than the one tried here.
3. `A, B` have not been checked for the `a_Mr`-style ceiling-driven
   divergence the way `rho` was (incorrectly) suspected of — worth a
   quick pass before assuming they're clean, since they share `rho`'s
   unprotected (no `_chart_spin0_jac`-equivalent) path.

---

## 2026-08-24: the 6000-step default was the bias; `rho` is retired; the
centroid layer is now conditioned on shear

Started from `flows/shear_gauss2.eqx` as retrained on 2026-08-23 against the
`project_to_physics` refactor of `models/shear.py`.

### The refactor is a no-op, and the retrained flow was broken

`project_to_physics` computes Q and R by `jacfwd`/`hessian` of the shift at
g = 0 and re-forms `z + Q.g + g.R.g/2`.  The shift is exactly quadratic in g, so
this is the same map: verified over 100 random draws, value identical to 5e-7
and **Q and R bit-identical** (rel. diff 0.0).  It cannot have changed a result.

What had changed was the training.  That checkpoint measured **m1 = -2.146 +/-
0.014** on 1M noiseless gauss2 targets -- a sign-flipped estimator -- because
`shear.py --deriv-weight` defaults to `0.0` and the shear response is invisible
to the NLL:

| | as retrained | + `--deriv-weight 1e4` | ref. `shear_gauss2_derivsup.eqx` |
|---|---|---|---|
| val nll | 42.3999 | 42.4115 | 42.4110 |
| dm/dg Mf/Mr/M1/M2/Mc | 75/92/9.7/9.8/101% | 4.9/4.5/1.9/1.9/8.5% | 5.0/4.1/1.5/1.5/7.8% |
| d2m/dg2 | 1622/534/985/1011/272% | 1.0/1.7/2.8/2.9/1.9% | 0.9/1.4/2.6/2.7/1.7% |
| noiseless m1 | **-2.146** | -0.0185 | -0.0146 |

Val NLL is *lower* on the broken flow.  Old checkpoint kept at
`flows/nll_only/shear_gauss2.eqx`.

### The residual bias was the 6000-step default

Ladder on gauss2, 1M noiseless targets, `--deriv-weight 1e4`, same
`bulk_gauss2.eqx` warm start, `--seed 0`:

| steps | val nll | dm/dg Mf/Mr/M1/M2/Mc | d2m/dg2 | noiseless m1 |
|---|---|---|---|---|
| 6000 | 42.4115 | 4.90/4.54/1.92/1.94/8.47% | 1.02/1.73/2.80/2.89/1.85% | -0.01846 +/- 0.00075 |
| 20000 | 42.4114 | 3.30/4.21/1.43/1.44/5.78% | 0.57/0.69/1.91/1.93/0.97% | -0.00789 +/- 0.00016 |
| 60000 | 42.4112 | 2.70/4.28/1.29/1.31/5.12% | 0.49/0.54/1.42/1.50/0.70% | -0.00138 +/- 0.00015 |
| 180000 | 42.4112 | 2.49/4.49/1.30/1.33/4.84% | 0.34/0.42/1.35/1.40/0.43% | -0.00419 +/- 0.00014 |

**The response fit and the bias converge at different rates.**  `d2m/dg2` keeps
halving to 180k while `m1` gets worse -- past ~60k, fitting `dm/dg` better stops
buying bias.  Three seeds at 60k give -0.00138 / -0.00428 / -0.00277, **mean
-0.0028, sd 0.0015**, and 180k's -0.00419 is 0.95 sd from that mean.  So m1
plateaus at 60k and 180k costs 3x for nothing.

Quote gauss2 noiseless as **m1 = -0.0028 +/- 0.0015** (seed sd).  The per-run
bootstrap (1.5e-4) is 10x too small, and the ladder looked monotone until the
fourth point arrived.  `flows/shear_gauss2.eqx` is now the 60k seed-0 flow;
the 6k one is at `flows/nll_only/shear_gauss2_6k.eqx`, and seeds 1/2 and the
20k/180k rungs are at `flows/shear_gauss2_{60k_s1,60k_s2,20k,180k}.eqx`.

README's "the shear layer converges at its own default (6000/20000/60000 give
-0.064/-0.056/-0.067)" is a **pure-NLL-era** claim and should be struck.

### `rho` is not the bias, and the 2026-08-22 diagnosis above is wrong

Exact per-galaxy solve of (mu, nu, rho) from bfd's own dm_dg/d2m_dg2
(`dev/exact_response.py`, self-checked against the catalog's own lensed shape):

* True |rho| is median **3.37** on gauss2 (flat across |eps| quintiles,
  3.24-3.43) and 5.93 on bulgedisc, against `_COEFF_MAX = 12`.
* Clipping at 12 fires on 13% / 37% of galaxies and costs **0.48% / 0.44%** of
  the second-order response, population-weighted.  bulgedisc's median 64.6 at
  the roundest quintile is real but occurs where `eps^3 ~ 0`, so the term
  contributes 0.1% there -- 1/|eps|^3 amplification of a numerator already going
  to zero, not physical amplification.
* Ablating rho in the trained layer moves d2m/dg2 M1/M2 from 2.93%/3.22% to
  3.10%/3.39%.  It earns its place and cannot carry a 0.018 bias.
* **Decisive**: 6k -> 60k took m1 from -0.0185 to -0.0014 while rho saturation
  went **UP**, 74% -> 96.3% (c_Mc 7.6% -> 84.8%, c_Mr 0% -> 37%,
  `dev/coeff_saturation.py`).

Saturation is a symptom of weak identifiability -- rho's design column carries
|eps|^3 where mu/nu carry |eps|, so at |eps| ~ 0.05 its gradient is ~3 orders
down and Adam walks a nearly-flat direction to the bound at full step size.
**Do not read a saturation fraction as a bias diagnosis.**  Non-equivariance
differs by population and both earlier numbers were right: gauss2
|Im(rho)|/|rho| ~ 1e-7, bulgedisc ~0.36 -- so the "partly non-equivariant
target" of `rho-polar-fix-tried-and-failed` is a bulgedisc/misalignment
statement, not a general one.

### Nothing blocks the spin-0 response either

On gauss2 every candidate obstruction is eliminated: the response is parallel
to eps to **0.0%** per galaxy (bulgedisc 0.4/2.9/5.2%); `shear.py scatter` puts
the floor at 0.09/0.58/1.07%; required c_X is 1.64/2.29/2.50 against a bound of
12 with 0.0% saturating; the even log-det is **-0.0012** nats against a physical
+0.0003 (it was 0.509 -- `chart-jacobian-reparam` fixed it, so the NLL is no
longer buying density with the spin-0 block).

Still open: **`dm/dg` Mr is flat at 4.2-4.5% across the whole 30x step ladder**
while Mf and Mc keep falling, against a 0.58% floor.  Training time is not what
limits it.  Response residuals are seed-stable (Mr 4.28/4.36/4.38 over three
seeds) while m1 scatters by 0.0015, so m1's remaining scatter is the
density/bulk interaction, not the response layer.

### A layout bug that cost a round of wrong conclusions

`dm_dg` is `(n, 2, 5)` and `d2m_dg2` is `(n, 3, 5)` -- **G INDEX FIRST**.
`reshape(-1, 5, 2)` SUCCEEDS (2*5 == 5*2) and silently returns transposed data
with no error, and shapes do not catch it.  This produced a completely wrong
first version of the rho analysis.  Pinned by
`tests/test_shear.py::test_catalog_derivative_layout_is_g_index_first`, which
checks the physical signature (`dM1/dg1` is 37x `dM1/dg2`) rather than shape.

Audited every reader: the live code is clean -- `shear.load`/`shear.lens`,
`partial_norms`, `check`, `train`, and `bias.py`'s eq. (40) template-lensing all
use the native layout.  The 13 `dev/` scripts that index `[:, :4]` with
`(n,4,2)` comments import `bfd_cnf.config`, a package that no longer exists, and
cannot run.

### Centroid layer: now conditioned on shear

The marginalisation is a property of the LENSED galaxy.  Most of that already
arrived through the layer's input (it sits downstream of `ShearResponse`
generatively): measured, the layer's `M1` shift already moved **26.8%** across
+/-0.02, `Mf` 6.7%.  What the input path cannot carry is the dependence at FIXED
m -- the layer models `E[marg | m]`, and shear changes which profiles map to a
given m.

* `models/centroid.py`: `split_condition` returns `(g, Sigma_X)` deciding on the
  vector's LENGTH (`condition[:2]` on a standalone `(3,)` would have handed
  `C00, C01` to the layer as a shear -- `dm_dsigma` passes exactly that);
  `g_invariants` builds `p1 = Re(ebar g)`, `p2 = |g|^2` on the physical shape,
  normalised by `G_MAX`; `_Coeffs` takes them as two extra net inputs (4 -> 6).
  Coefficients are scalars, so expanding e.g. `A(p1,p2).t2` generates the
  g-bearing spin-2 structures with the right symmetry automatically --
  `N_COEFFS` stays 9.
* The two g columns of the first layer are **zero-initialised**.  A g = 0 run
  gives them exactly zero gradient (`dL/dw = delta * input = 0`), so at random
  init they would stay random and `bias.py` would read an arbitrary untrained
  g-dependence straight into Q and R.  Zeroed, the layer is exactly
  g-independent until something trains it in g.
* `bias.py`: `split_centroid` now REFUSES to peel a `(5,)`-conditioned layer --
  the peel's justification was that the log-det is g-independent, which is now
  false, and peeling would evaluate the layer at g = 0 and discard the
  dependence.  A standalone `(3,)` layer is still peeled.  **This costs what the
  peel bought**: 5 extra JVPs of the coefficient net per draw inside the
  forward-over-reverse Hessian.  Unmeasured -- drop `--chunk` if a deep run OOMs.
* `G_MAX` moved from `shear.py` to `models/shear.py` (re-exported), so a layer
  does not import the training script.

### The copies DO carry their own derivatives -- `copies.py` was discarding them

`makeTemplates` returns a `bfd.Template` per copy whose `even` is
`(NE=5, NPQR=10)`: moments AND exact shear derivatives.  `copies.py` kept only
slot `D0`.  A copy is the parent measured about a shifted origin, so the
parent's derivatives do not transfer -- but the copy's own were there all along,
at no extra render cost.

* `imsims/copies.py`: `COPY_DTYPE` gains `dm_dg (2,5)` and `d2m_dg2 (3,5)`,
  extracted exactly as `sim.fill_row` does.  Also gained `--shear G1 G2`, which
  lenses templates before gridding (unused by the tasks below, but it is the
  other way to build a sheared copy catalog).
* Validated on a 60-galaxy render: the copy nearest u = 0 matches its parent's
  derivatives to 5.9e-6 at |u| = 0.0005 and 1.3e-2 at |u| = 0.018 -- agreement
  scaling with |u| is the signature of genuine per-copy derivatives.
* `centroid.py`: `CopySampler.draw` returns them; `train` gains `--g-max`
  (default `G_MAX`), samples g with antithetic +/-g pairing and lenses each
  drawn copy by its own derivatives before the NLL, conditioning on that g.  It
  raises a clear error on a catalog without the columns.
* Smoke-tested end to end: the g columns went `0.000e+00 -> 3.951e-02` in 60
  steps and the layer's shift now moves with g.

Full suite 64 passed throughout.

### WHAT IS STALE

* **Every copies catalog on disk** -- no derivative columns.  Must be
  re-rendered before the centroid layer can be trained with `--g-max > 0`.
* **Every centroid checkpoint** -- twice over: the coefficient net is 6-input
  now (`eqx.tree_deserialise_leaves` will raise on the old 4-input weight), and
  the training objective changed.
* The `gauss2_deep` numbers below were measured on the **6k** shear flow and on
  a g = 0 centroid layer; expect them to move a lot.  Last full run, rebuilt
  chain: unwindowed -0.02661 +/- 0.00058, windowed uncorrected -0.00681,
  windowed corrected **-0.02311 +/- 0.00079**, P_s = 0.6324.
* The centroid layer's own defect is untouched by any of the above: its
  ellipticity response is **4.3x** the catalog's (layer +3.56e-2 vs catalog
  +8.36e-3) with `Mf`/`Mr` shifts of the WRONG SIGN.  With the shear layer now
  contributing only ~0.003 to m1, this is likely the dominant term in the deep
  number rather than a secondary one.

### NEXT SESSION, in order

1. **Re-render the gauss2 deep copies catalog** with the derivative columns.
   ~3.5 GB (was ~1 GB); 193 GB free at time of writing.

       cd ../bfd_cnf_imsims
       python -m imsims.copies --n 100000 --seed 0 --pop gauss2 \
           --noise-sigma 2.73 --out data/copies_gauss2_deep.fits

   Check afterwards that `dm_dg`/`d2m_dg2` are present, finite and nonzero, and
   that the copy nearest u = 0 matches its `GALAXIES` row.

2. **Retrain the centroid layer with g sampling**, off the converged shear flow.

       python centroid.py train --copies ../bfd_cnf_imsims/data/copies_gauss2_deep.fits \
           --flow flows/centroid_gauss2_deep.eqx --init flows/shear_gauss2.eqx \
           --steps 16000

   Confirm the g columns of `coeffs.net.layers[0].weight[:, 4:]` are nonzero
   afterwards -- if they are still 0 the g path never got gradient and the rest
   is meaningless.  Watch the `ellipticity response` line in the check: 4.3x is
   the number to beat.

3. **Measure `gauss2_deep`** on the rebuilt chain.  The peel is now declined, so
   this is slower and heavier than the ~110 min the last run took; start with
   `--n-targets 20000` to size it before committing.

       python -u bias.py --flow flows/centroid_gauss2_deep.eqx --pop gauss2_deep \
           --samples 8192 --alpha 0.5 --chunk 4096 \
           --window-size 2.2 3.2 --window-flux 2500 50000

4. **Then the open physics**: `dm/dg` Mr flat at 4.2-4.5% across the whole step
   ladder against a 0.58% floor, and the centroid layer's 4.3x ellipticity
   response.

Do NOT spend effort on `rho`'s bound or parameterisation, and do not re-run the
`e_dir` substitution -- see above and `rho-polar-fix-tried-and-failed`.

### One measurement was started and CANCELLED -- worth redoing first

`bias.py --pop gauss2_deep` on the **60k shear flow with the OLD g = 0 centroid
layer** (`flows/centroid_gauss2_deep.eqx` as retrained off
`flows/shear_gauss2.eqx`, before the `models/centroid.py` change landed).  It
was killed part-way through the `minus` arm to free the machine, so there is NO
result -- do not go looking for one.

It is worth running again as step 0, because it isolates a variable nothing else
does: what the converged shear layer alone buys at depth, with the centroid
layer held at its old behaviour, against the 6k chain's windowed-corrected
-0.02311 +/- 0.00079.  If that alone lands near zero, the centroid layer's 4.3x
ellipticity-response error is doing less damage than feared and steps 1-3 are
less urgent than they look.

To run it you need the pre-change layer back, since the checkpoint on disk is a
4-input net that the current `models/centroid.py` cannot deserialise:

    git stash                     # or check out this commit's parent
    python -u bias.py --flow flows/centroid_gauss2_deep.eqx --pop gauss2_deep \
        --samples 8192 --alpha 0.5 --chunk 4096 \
        --window-size 2.2 3.2 --window-flux 2500 50000

Or skip it and go straight to step 1, accepting that the shear and centroid
contributions stay entangled.

---

# 2026-08-24 (evening): the centroid sign bug, fixed and committed; the shear
# response's Mr/Mf ~ 3.1 instability, confirmed real and still unexplained

Continuation of the same day's earlier session (g-conditioning added,
slope-instability found and fixed with `_SLOPE_MAX`). This picks up
immediately after that: g-conditioning was confirmed mandatory (real target
galaxies have unknown centroids; the marginalisation must depend on the
lensed galaxy), so the fix had to make the sign correct, not remove the
feature.

## The sign bug, and getting it wrong twice before getting it right

Displacing the origin costs Mf, Mr/Mf and Mc/Mr for **every copy of every
galaxy**, unconditionally -- verified directly from the copies catalog,
binned by `|u|`, on five separate galaxies with no exceptions. This is not a
population tendency NLL might reasonably fail to recover exactly; it is a
deterministic geometric fact the layer's coefficients should not need
training to discover the sign of.

Mind the Fourier-space convention the user had to correct me on: Mr/Mf and
Mc/Mr are **inverse** size and concentration, so "falls" means "larger and
less concentrated" -- `models/centroid.py`'s own header already had this
right (`imsims/copies.py`'s copy had drifted to the opposite claim; fixed,
commit 411a06a in bfd_cnf_imsims).

Scanning NLL directly against the trained (unconstrained) Mf/t0 coefficient
proved this isn't a training-convergence problem: NLL is monotonically
**minimised** at the physically wrong sign, and gets worse moving toward the
correct one. Density-matching a deterministic transport against a genuinely
stochastic marginalisation (a mixture over centroid offsets, not a point
map) simply rewards a different answer than the one bias.py needs.

The fix, in `models/centroid.py`'s `_Coeffs.__call__`: both columns of `s0`
(the T0-column, indices 0/2/4, and the `(ec*t2).real`-column, indices 1/3/5)
are sign-constrained by construction via `_rational_bound(softplus(...))`
rather than left free. **Getting the direction of these two constraints
right took three attempts.** `marginalize = unmarginalize^-1`, and to
leading order `unmarginalize(x) ~= x + shift(x)` inverts to
`marginalize(y) ~= y - shift(y)` -- one sign flip relative to the physical
requirement on `marginalize`'s output, easy to miss and easy to verify
against (compute `marginalize(x) - x` directly and check its sign, on the
UNTRAINED layer, before spending a retrain). Both wrong attempts were caught
by retraining and checking `centroid.py check()`'s per-moment table before
committing to anything; the fix that shipped was verified this way first
and is `models/centroid.py`/`tests/test_centroid.py` in commit dc41c8c
(bfd_cnf) -- read that commit message for the full mechanism.

## Validated at full depth

| | windowed-corrected m1 (200k targets) |
|---|---|
| no centroid at all | -0.0103 +/- 0.0027 (20k only) / **-0.00998 +/- 0.00110** (200k) |
| old g=0 centroid (this morning's baseline) | -0.0285 +/- 0.0029 |
| broken-sign g-conditioned (this session, superseded) | -0.0317 +/- 0.0031 |
| **sign-corrected g-conditioned (committed)** | **-0.01005 +/- 0.00170** |

The sign-corrected, slope-stable centroid layer matches the no-centroid
number almost exactly at the aggregate level, while reshaping the bias
across Mf quintiles the way a real centroid-uncertainty correction should
(concentrated at faint flux, vanishing at bright -- q1..q5 with centroid:
-0.213, -0.086, +0.016, +0.0099, +0.0021; without: +0.038, -0.002, -0.023,
-0.019, -0.011). The centroid layer is doing real, physically meaningful
work; the residual ~1% bias lives elsewhere.

`flows/centroid_gauss2_deep.eqx` on disk is this final, correct checkpoint
(g-conditioned, `_SLOPE_MAX=0.1`, sign-constrained). Every OTHER
`flows/centroid_*.eqx` on disk (the g=0 baseline, the broken-sign attempts)
is stale relative to today's `models/centroid.py`.

## The residual ~1% bias: isolated to the shear/bulk response, not fixed

Confirmed (not selection, not centroid) three ways: the unwindowed number
(no selection cut, `P(s|g)=1` exactly) matches the windowed-corrected one;
the no-centroid run matches the with-centroid run at the aggregate level;
both point to the same pre-existing residual in the shear/bulk layers that
predates this session.

Traced it to a specific, reproducible defect: `shear.py check`'s dm/dg
residual for Mr, binned by `r = Mr/Mf`, **peaks at r ~ 3.08-3.23 (6.76-6.98%)
and is smaller on both sides** (0.68% at low r, 4.46-4.61% at high r) --
NOT a monotonic edge effect, against a genuinely flat, achievable floor of
~0.5-0.58% everywhere (a flexible offline regression on the exact per-galaxy
response hits that floor uniformly, including in the bad bin). This
reproduces across three independently-trained checkpoints at 60k steps
(`flows/shear_gauss2.eqx`, `_60k_s1`, `_60k_s2`: bin-3 frac_resid 6.76%,
6.77%, 6.98%) -- **confirmed NOT seed noise** -- and is flat across
20k/60k/180k steps with identical val NLL throughout -- **confirmed
converged, not undertrained.**

A physical mechanism was found and is well-supported: `bulge_kwidth/kmax`
(the compact bulge's characteristic k-space width, divided by the BFD
weight function's cutoff) crosses 1.0 **exactly** between the same two bins
that bracket the error peak (0.803 -> 1.072). Since `rho` (bulge/disc size
ratio) is always < 1 for this population, the bulge is always the first
component to have its k-space power clipped by the weight function's finite
support as the galaxy shrinks -- a real regime transition in the moment
integral, not an artifact.

**What is NOT yet established: the exact computational mechanism by which
this produces a locally unstable (not just biased) coefficient in the
trained net.** Five toy reproductions were attempted and NONE of them
reproduced the specific "peak at r~3.1, fine on both sides" pattern:

  1. A synthetic 1D kink (the "spectral bias" hypothesis) -- refuted, fits
     everywhere.
  2. Real 4D inputs (f,a,b,q), real exact `a_Mr` target, direct MSE
     regression -- refuted, fits everywhere (0.05-0.13%).
  3. Same, but the actual z-space target the reparameterised net has to
     learn (`(a_Mr - a_Mf)/(scale*(1-u))`, which itself has a genuine,
     verified slope explosion from ~2 to >100 across the same r-range) --
     STILL refuted, fits everywhere (0.15-0.48%).
  4. Same, with a shared 13-output head (mimicking the real net's 14
     coefficients competing for hidden-layer capacity) -- refuted, fits
     everywhere (0.12-0.50%).
  5. The one remaining structural difference -- fitting the net's output
     through the REAL `response()`/`project_to_physics` composition,
     differentiated via `jax.jacfwd`/`jax.hessian` to get Q and R (rather
     than regressing a precomputed target directly) -- did NOT cleanly
     reproduce it either; instead it produced a uniform ~95% failure across
     ALL bins that does not respond to a 10x learning-rate change at all
     (identical loss trajectory), which is very likely a bug in the toy
     script rather than a real finding. NOT debugged before the session
     ended -- see `/tmp/.../scratchpad/toy_nested_autodiff.py` (scratch,
     not preserved) for the last state; single-galaxy gradient checks there
     were NOT zero (Q-only grad norm 1.45, Q+R grad norm 662), so the bug is
     something about the batched/vmapped/jit training loop, not dead
     gradients outright.

So: the phenomenon is doubly confirmed real (optimiser-independent via a
from-scratch AND a warm-started full-batch L-BFGS test that could not
improve on the SGD optimum at all; seed-independent via three trained
checkpoints), physically grounded (the bulge/kmax crossing), but the actual
mechanism connecting "the true response has a real transition here" to "the
trained net is locally unstable, not just imprecise" is not pinned down.
Every hypothesis tested in isolation (kink-fitting difficulty, target
steepness, shared-capacity competition) has been directly refuted by a
controlled toy. Whatever it is, it requires something about the ACTUAL
training loop's mechanics beyond what a static, direct, single-composition
regression captures.

## What's stale / what to check first

- `flows/shear_gauss2_purederiv.eqx`, `flows/shear_gauss2_lbfgs.eqx` (never
  written -- that run was killed), `flows/shear_gauss2_lbfgs_warm.eqx`:
  scratch checkpoints from this investigation, not meant to replace
  `flows/shear_gauss2.eqx`. Safe to delete once no longer needed for
  cross-checking the toy investigation.
- No code changes came out of the shear-response investigation -- it's
  diagnosis only, nothing to revert. `models/shear.py` is untouched.
- The `bulge_kwidth/kmax` check used `kmax ~ 5.2` read off a code comment in
  `imsims/analytic.py`, not computed exactly from `bfd.KBlackmanHarris`
  itself -- worth verifying precisely if this thread continues.

# 2026-08-24 (late): the Mr/Mf ~ 3.1 "instability" was a stale chart, and it is
# fixed

The open thread from the previous section is CLOSED, and it was not a physical
regime transition, not an optimisation pathology, and not Var[Q|m].  It was a
bug, in a place nothing had looked: the shear layer's frozen copies of the
chart's statistics.

## The bug

`RawMomentStandardize.mean` and `.std` are plain `jax.Array` fields, so they
are TRAINABLE, and `bulk.train` moves them a long way:

| catalog | chart std, fresh from the data | after 150k bulk steps |
|---|---|---|
| gauss2 | [0.296, 0.499, 0.475, **0.0402, 0.0402**] | [0.591, 0.509, 0.396, **0.1565, 0.1567**] |
| bulgedisc | [0.304, 0.946, 0.914, **0.0403, 0.0408**] | [0.463, 0.883, 0.799, **0.1482, 0.1517**] |
| sersic | [0.332, 0.619, 0.631, **0.0709, 0.0708**] | [0.406, 0.579, 0.549, **0.1651, 0.1648**] |

`ShearResponse` holds its own frozen `chart_loc`, `chart_scale` and `e_scale`,
set by `bulk.build_flow` from the RAW SAMPLE.  `shear.py --init` then grafts
every non-shear bijection out of the bulk checkpoint -- **including
`bijections[0]`, the chart** -- leaving the layer's copies describing a chart
that is no longer underneath it.  On gauss2 `e_scale` was 3.9x too small.

That is not cosmetic.  `_chart_spin0_jac` reconstructs
`logit(u) = chart_scale*z1 + chart_loc` to build the 1/(1-u) size Jacobian, and
`response` reconstructs the physical `e = e_scale*(z3 + i z4)`.  Both come out
wrong, and the logit one comes out wrong **in a way that varies along z1**,
which IS Mr/Mf.  Hence a residual peaked on the resolution axis.

`centroid.py` already knew about this failure mode and fixed it -- for its own
layer only, five lines, with a comment saying exactly why.  The shear layer
grafted three lines above it was left stale, and `shear.py` never did it at all.

## The measurement

One script, one knob, everything else identical (real flow, real warm start,
real loss, 20k steps, gauss2), `frac_resid` of dm/dg for Mr by Mr/Mf octile:

| octile | r | as shipped | resynced |
|---|---|---|---|
| 0 | 2.04-2.64 | 1.50% | 0.64% |
| 3 | 2.90-3.01 | 2.71% | 0.71% |
| 4 | 3.01-3.10 | 4.30% | 0.88% |
| **5** | **3.10-3.19** | **6.36%** | **0.90%** |
| 6 | 3.19-3.30 | 4.90% | 0.85% |
| 7 | 3.30-3.56 | 4.51% | 1.10% |

The peak is gone, the profile is flat, and it sits at the ~0.5-1% floor the
offline regression said was achievable.  Derivative MSE at 20k steps:
8.9e-3 -> 4.1e-4, a factor 22.  The as-shipped column reproduces
`flows/shear_gauss2.eqx`'s own numbers (bin 5: 6.36% here, 6.81% on the 60k
checkpoint), so the reproduction is faithful.

## What was eliminated, and how

Every one of these was a controlled run in the SAME script, not a proxy toy --
which is why they succeeded where last session's five toys could not.

1. **The denominator.** `frac_resid`'s denominator (RMS of the true response)
   rises MONOTONICALLY 0.074 -> 0.116 across the bins; the numerator -- the
   absolute, loss-relevant error -- is what peaks, 0.00045 -> 0.00696 -> 0.0049.
   Not a metric artifact.
2. **NLL competing with the derivative term.** Refuted decisively: the same
   loop at `nll_weight = 0` gives bin 5 = 6.34% against 6.36% with it on.  The
   NLL is irrelevant to this residual.  (An earlier read of
   `flows/shear_gauss2_purederiv.eqx` had already hinted at this, but that
   checkpoint's provenance was unknown -- the controlled run is what settles it.)
3. **The inputs.** A plain 3x256 MLP on the same `z` the layer sees, plain MSE,
   fits the exact normalised dMr/dg to 0.0002 FLAT across all eight octiles --
   35x better than the trained layer at its peak.  Nothing about moment space
   is hard.
4. **The parameterisation.** The real `_Coeffs` + `project_to_physics` +
   `dm_dg` composition, trained on Mr's Q alone, fits FLAT at 0.02-0.11%.  The
   14-coefficient bounded equivariant form is not the problem.
5. **Q vs R competition.** Adding R to that same run costs a factor 3
   (0.0003 -> 0.0010) and stays MONOTONE.  No hump.
6. **Var[Q|m].** Measured model-free: local least squares in 120-neighbour
   patches of z, projecting out the local z-gradient, gives the irreducible
   response scatter as |dy/drho|_m * sigma(rho|m) = 0.0010 and FLAT.  The
   trained layer is 7-10x above it in the bad bins.  gauss2's population has
   `bulge_frac` constant and `bulge_e == disc_e` exactly, so `bulge_ratio` is
   its only hidden dof; it is 5 free parameters onto 5 moments.
7. **The `bulge_kwidth/kmax` crossing.** A coincidence.  Nothing needed it.

An HGB (tree) regression from moments DID show a hump at bin 5 while the same
model from galaxy parameters did not -- that was a red herring: gradient-boosted
trees approximate this smooth ridge function badly, and the MLP in (3) shows
the target is perfectly learnable from moments.

## The fix

`bulk.sync_chart_constants(flow)` -- re-points every `ShearResponse` and
`CentroidMarginalize` frozen copy at `bijections[0]`'s actual mean/std, with
`e_scale` recomputed as the symmetrised spin-2 std.  Called from `shear.py`
after its graft and from `centroid.py` in place of its old two-field version.
Pinned by `tests/test_shear.py::test_sync_chart_constants_follows_a_grafted_chart`.

It meets the bar the user set for a fix: physically interpretable (a layer must
be told which chart it is behind), not a tunable hyperparameter, nothing to
search, and population-independent -- the drift is present on all three
catalogs.

## WHAT IS STALE

**Every `flows/shear_*.eqx` and `flows/centroid_*.eqx` on disk.** All of them
were trained behind a mis-described chart.  So is every m1 ever measured from
one, including this morning's centroid validation table (`-0.01005 +/- 0.00170`
etc.) -- that comparison was internally consistent, so its CONCLUSION about the
centroid sign fix stands, but the numbers are void.  The bulk checkpoints are
unaffected (the chart is theirs; it is the graft that goes wrong).

## NEXT SESSION, in order

1. Rerun the chain: `bulk.py train --steps 150000` is unaffected and can be
   reused; `shear.py train --steps 60000 --deriv-weight 1e4` and
   `centroid.py train --steps 16000` must be redone for gauss2, bulgedisc and
   sersic.
2. Re-measure `shear.py check` and the binned profile -- expect flat.
3. Re-measure the noisy 200k-target m1 on gauss2.  The residual ~1% was
   attributed to the shear/bulk response; this was the shear response's defect,
   so this is the number that says how much of the 1% it was.
4. Only then reconsider the bulk-density threads, which have not been retested
   against a correctly-charted shear layer.

## Open design question, deliberately not acted on

Should `RawMomentStandardize.mean/std` be trainable at all?  Everything else in
the codebase treats them as data-determined constants (`_Coeffs`' whitening
docstring says so in as many words), the flow has downstream scale freedom
anyway, and freezing them would make this whole class of bug impossible rather
than merely fixed.  It would change `bulk.train`'s optimum, so it is a real
experiment, not a cleanup -- left for the user to call.

# 2026-08-24 (overnight): the fix is committed and the chain is rebuilt; the
# response fit improved 6x, the noisy bias did NOT

Continuation of the section above, which found and fixed the stale-chart bug.
This is what happened when the gauss2 deep chain was rebuilt on top of it.

## Committed

`03e3551` -- `bulk.sync_chart_constants`, called from `shear.py` and
`centroid.py` after their warm-start grafts, plus
`tests/test_shear.py::test_sync_chart_constants_follows_a_grafted_chart`.
Full suite: **65 passed** (6m11s, confirmed twice on a free GPU -- an earlier
"collection error" was three orphaned pytest processes fighting over the card,
not a real failure).

The pre-fix checkpoints are preserved at `flows/pre_chartsync/{shear_gauss2,
centroid_gauss2_deep}.eqx` -- keep them, every paired comparison below needs
them.

## The rebuild, and what it bought

Same `flows/bulk_gauss2.eqx` warm start as before, so the ONLY difference is
the chart sync.  `shear.py train --steps 60000 --deriv-weight 1e4 --seed 0`,
then `centroid.py train --steps 16000`.

`shear.py check`, dm/dg RMS residual / RMS truth:

| | Mf | Mr | M1 | M2 | Mc |
|---|---|---|---|---|---|
| dm/dg before | 2.70% | 4.28% | 1.29% | 1.31% | 5.12% |
| **dm/dg after** | **0.29%** | **0.66%** | **0.32%** | **0.32%** | **0.87%** |
| d2m/dg2 before | 0.49% | 0.54% | 1.42% | 1.50% | 0.70% |
| **d2m/dg2 after** | **0.12%** | **0.20%** | **0.25%** | **0.25%** | **0.27%** |

val nll 42.4112 both, to four decimals -- as expected, and one more datum for
"val NLL cannot rank flows for response-fit quality".

Binned by Mr/Mf octile, Mr's first-order residual: the peak is gone.  Bin 5
(r 3.10-3.19) 6.81% -> 0.81%, and the whole profile is flat at 0.29-0.87%.
**The long-open "dm/dg Mr flat at 4.2-4.5% against a 0.58% achievable floor"
item is closed.**

Centroid layer retrained fine: all three spin-0 shifts negative and 72-86% of
the catalog's magnitude (Mf -5.09e-3 vs -5.92e-3, Mr -8.24e-3 vs -1.13e-2,
Mc -1.12e-2 vs -1.59e-2).  Ellipticity response still 4.36x the catalog
(3.65e-2 vs 8.36e-3) -- unchanged by this fix, still its own open item.

## But the noisy bias got WORSE, and the two estimators stopped agreeing

`bias.py --pop gauss2_deep --samples 8192 --alpha 0.5 --chunk 4096
--batch-budget 65536 --n-targets 20000 --window-size 2.2 3.2 --window-flux
2500 50000`:

| | before (200k) | after (20k) |
|---|---|---|
| windowed, corrected m1 | -0.01005 +/- 0.00170 | **+0.01530 +/- 0.00284** |
| unwindowed m1 | ~ -0.010 (matched the windowed) | **-0.02184 +/- 0.00243** |

Two separate problems there.  The magnitude went up, and -- more diagnostic --
**the windowed-corrected and unwindowed numbers now differ by 0.037**, where
their agreement used to be one of the three checks that localised the residual
to the shear/bulk response in the first place.  That divergence is new
information and is probably the more useful thread.

ESS is healthy and not the cause: 638 median, 110 at the 5th percentile,
`frac<10 = 0.000`.  Mf quintiles: -0.297 / -0.037 / +0.024 / +0.015 / -0.000
(q1 +/- 0.0145), so the faint end dominates as usual.

## The shear layer alone is FINE -- better, in fact

Paired noiseless run, 1M targets, `--no-centroid --pop gauss2`, nothing but the
shear flow swapped:

| shear flow | noiseless m1 |
|---|---|
| pre-fix (`flows/pre_chartsync/shear_gauss2.eqx`) | -0.00138 +/- 0.00015 |
| **post-fix (`flows/shear_gauss2.eqx`)** | **+0.00047 +/- 0.00017** |

Both are inside the 0.0015 seed sd (the 60k three-seed mean is -0.0028 +/-
0.0015), the new one marginally closer to zero.  So the fix did not damage the
shear layer -- it improved the response fit 6x and left the noiseless bias
where it was.  **Whatever produces +0.0153 at depth enters through the noisy
path**: the centroid layer, or the C_M integration, or the selection terms.

## RUNNING WHEN THIS WAS WRITTEN -- read these first

A paired 2x2 completion: deep, 20k targets, `--no-centroid`, on the post-fix
and pre-fix shear flows, launched with

    for f in shear_gauss2 pre_chartsync/shear_gauss2; do
      o=$(echo $f | tr "/" "_")
      python -u bias.py --flow flows/$f.eqx --no-centroid --pop gauss2_deep \
        --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536 \
        --n-targets 20000 --window-size 2.2 3.2 --window-flux 2500 50000 \
        > logs_deep_nocent_$o.txt 2>&1
    done

Results land in `logs_deep_nocent_shear_gauss2.txt` and
`logs_deep_nocent_pre_chartsync_shear_gauss2.txt`.  Together with the two
with-centroid numbers this is the full 2x2 (old/new shear) x (with/without
centroid), all at 20k targets, all on the same targets -- which says whether
the +0.0153 is the shear layer at depth or the centroid layer retrained on top
of it.  The reference point for the no-centroid column is **-0.0103 +/- 0.0027**
(20k, pre-fix, from the evening session's table).

## Practical notes

* `--batch-budget 65536` is now REQUIRED for this run.  At the default
  (131072/chunk = 32 targets x 4096 draws) the first `pqr_streamed` batch asks
  for 11.72 GiB and OOMs the 16 GB card.  It is pure device chunking -- the
  streamed accumulation is unchanged, so it does not touch the arithmetic.
  This is not `--n-targets`-dependent; the 200k run would have hit it too.
* A 200k-target run was started and killed to get the 20k number sooner.  There
  is no 200k result on the rebuilt chain yet.  Once the 2x2 is understood, the
  200k rerun is what turns +0.0153 +/- 0.0028 into a number worth quoting.
* Recall the working rule: quote gauss2 m1 with the SEED spread (~0.0015-0.002
  at 60k), not a single run's bootstrap.  +0.0153 vs -0.0100 is a real shift
  even at that tolerance, but a single seed is a single seed.

## What is stale

All `flows/shear_*.eqx` and `flows/centroid_*.eqx` OTHER than the two gauss2
ones rebuilt tonight -- bulgedisc and sersic have not been retrained against
the fix, and their chart drift is the same size (spin-2 std 0.040 -> 0.148 and
0.071 -> 0.165).  Bulk checkpoints are unaffected.

## Open design question, still not acted on

Should `RawMomentStandardize.mean/std` be trainable at all?  Freezing them
makes this whole bug class impossible rather than merely fixed, and everything
else in the codebase already treats them as data-determined constants.  It
would change `bulk.train`'s optimum, so it is an experiment, not a cleanup.
