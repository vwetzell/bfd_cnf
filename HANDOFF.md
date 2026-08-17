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
| moment-space control | **+0.0067** | -0.0003 |
| centroid real, S = 32768 | +0.0264 | -0.0046 |

The 22 component is ~0 everywhere, as it must be with shear in g1 only (there it
is a cross-covariance, not a variance) -- that split is what identifies the term
rather than leaving it as "the estimator is inconsistent", which is the trap.
Read as the first line, the centroid path's per-target `log P` is **8.5x more
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
than weakens the "8.5x more non-Gaussian in g" reading above.  A ~7% residual
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
