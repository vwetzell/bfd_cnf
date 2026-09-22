# Session 2026-09-18 — PSFE additive-leak investigation, full findings

Standalone record of everything found in this session, in the order it was
found. This session picked up Thread 1 (the PSF-ellipticity leak) from
`NEXT_SESSION_PSFE_FORMALISM_AUDIT.md`'s Step 2 (does `C_M`/`Sigma_X`
anisotropy reach the likelihood correctly) and ended up identifying the
actual root cause, reconciling it with the paper's own validation, and
testing (not yet fully landing) a fix. Everything below is also written
into `NEXT_SESSION_PSFE_FORMALISM_AUDIT.md` as sessions 5-11, in more
granular form with exact recipes; this file is the consolidated narrative.

**Ground rule this session enforced on itself (and had to self-correct once
for violating)**: cite only the paper, `~/gitrepos/bfd`, and this repo's own
code as literally read/re-run this session — not a docstring, not a prior
session's memory file, not an assumption from a filename.

## 1. Step 2 closed: no wiring/packing bug found

Traced `C_M`/`Sigma_X` from disk to the likelihood, end to end:
- `bias.load_cov` reads each psfe config's own catalog (not a shared
  isotropic stand-in).
- `sigma_x_all = rows["zero"]["cov_odd"]` carries the full anisotropic
  `[C00,C01,C11]` triple per target.
- `bias.condition()` forwards `[g1,g2,C00,C01,C11]` verbatim, no truncation.
- `SigmaXBlockLayer._unpack`/`cx_to_sx_cond` reconstruct the full 2x2
  `C_X` and convert to `[log_scale,e1,e2]` with the standard convention —
  verified against `bfd/moment.py:278-281`'s own even-to-odd covariance
  formula (`odd[X,X]-odd[Y,Y]=Cov(M0,M1)`, `odd[X,Y]=0.5*Cov(M0,M2)`), not
  just asserted by inspection.
- `bias.py:1599`, `cinv = np.linalg.inv(cov)` is a genuine full-matrix
  inverse; `log_conv_is_kernel` uses it as a full quadratic form
  (`einsum("si,ij,sj->s", r0, cinv, delta)`), not a diagonal approximation.

**No bug found anywhere in this path.** This closes Step 2 and (together
with session 4's already-existing finding) rules out flow/estimator wiring
as the leak's mechanism.

## 2. Self-correction: a docstring's claim was repeated without verifying it

Initially reported that an uncommitted fix to `SigmaXBlockLayer._ellipticity`
(replacing an `(I+cE)`-matrix form with `x+D*e+c*proj*e`) was "believed to be
the mechanism" per that method's own docstring, and that it didn't close the
leak. The belief-attribution was correctly not-yet-verified; the correction
made:
- Built both formulas and numerically tested rotation covariance directly
  (rotate `(x3,x4)`,`(e1,e2)` as spin-2 quantities, recompute with `D,c` held
  fixed since they depend only on invariants): **old form residual 0.90 (NOT
  covariant), new form residual 4e-16 (covariant to machine precision).**
  The docstring's math claim held up under an actual test, but had not
  actually been tested before it was repeated.
- Confirmed via file mtimes that the flow tested (`sigmaxfix`) really did
  postdate this uncommitted change, so the empirical "fix doesn't close the
  leak" comparison was valid.
- Flagged clearly that "believed to be the mechanism" was one session's
  hypothesis, not an established fact, and that this session's own data
  already argued against it.

## 3. Higher-order noise-rectification effect: quantified

Built a script computing, per galaxy: (a) the exact-Gaussian-`u` prediction
(analytic `J0` via `bfd.MomentCalculator.xyJacobian`, real `Sigma_X`, exact
`getMoment` evaluation — no Taylor truncation), and (b) the REAL bias from
actual pixel noise + actual `recenter()` Newton solves (paired antithetic
draws). n=150, `psf_e1=0.05`:

| | dM1 |
|---|---|
| exact-Gaussian-u prediction | -0.145 +/- 0.102 |
| real noisy recenter() bias | +1.530 +/- 0.452 (3.4 sigma) |

**Real effect ~11x the Gaussian-u prediction, opposite sign** — the
higher-order (non-Gaussian third/fourth `u`-moment) term is not a small
correction, it IS essentially the whole measured effect. Consistent with
session 4's earlier, larger-n finding at the same amplitude.

## 4. Copy-grid density/extent: REFUTES a prior memory finding

Built two real copies pools (n=800 `gauss2_fwd` galaxies, seed 0): narrow
(`--sigma-range 1.4`, production, 246 copies/galaxy) and wide (`--sigma-range
3.0`, 3157 copies/galaxy). Naive comparison (each grid's own ESS>=1000 cut)
looked like a huge grid-density effect (`c1` -0.0046 vs -0.0476) — but the
two cuts admit almost entirely DIFFERENT targets (786 vs 2185 of 20000)
because a denser grid trivially raises every target's ESS. Redone with the
SAME matched target set fed to both grids:

| ESSMIN | n (matched) | narrow c1 | wide c1 | diff |
|---|---|---|---|---|
| 0 | 6242 | -0.09528 | -0.09526 | +0.00003 |
| 500 | 1363 | -0.03228 | -0.03228 | -0.00000 |
| 1000 | 786 | -0.00462 | -0.00462 | -0.00000 |

**Identical to 5 decimal places.** The copy grid's extent/density has NO
measurable effect on the per-target computation once the admitted target set
is held fixed. The prior memory finding (`~7-16%`, "real, second-order")
almost certainly measured the same admission-selection artifact via a
different route (a paired-bootstrap slope across a 10-config grid, which
also doesn't hold the admitted set fixed) — **treat that memory entry as
superseded/null.**

## 5. Why does this population expose a leak the paper's own test doesn't?

Fetched `bfd_paper.pdf` (arXiv:1508.05655) fresh and read §4.2 directly. The
paper's own PSF-anisotropy test (eq. 59-60, `c=(-1.3+/-0.9)e-5`, ">3000x
suppression") uses a population that **decenters the bulge relative to the
disc** ("which might otherwise be canceling some systematic error in the
method", verbatim), S/N 5-25, weight sigma=3.5px vs PSF r50=1.5px. Checked
every population this codebase can render (`imsims/sim.py:695-709`) — none
decenters bulge/disc; `draw_bulge_disc`'s own docstring says "two
co-centred Gaussians." **This is not the same experiment as the paper's.**

### 5a. Ellipticity-orientation confound (real, but doesn't explain the gap)

Per-galaxy diagnostic (n=150) found `resid1` (higher-order leak) strongly
coupled to the galaxy's own ellipticity aligned with the PSF:
`b (linear e1_gal coeff) = -24.7 +/- 4.0` (6.1 sigma). Since gauss2 forces
bulge/disc to share exactly one ellipticity, every galaxy has one large,
undiluted spin-2 shape to couple through. But this term's population mean
is exactly zero by construction (isotropic orientation) — at n=150 the
sample's own `mean(e1_gal)=-0.018+/-0.0075` inflated the raw mean(resid1)
by ~27% (a real confound in the SMALL-n test, +1.67 raw -> +1.22 debiased).

**Checked directly against production (n~20000, already-saved
`pqr/*.npz`, no new sims)**: `mean(e1_gal)` is within 1 sigma of zero at
every psfe config already (CLT at this N). Debiasing (zeroing the linear
term's contribution) does NOT shrink the production `dc1/d(psf_e1)` slope —
it makes it slightly LARGER (-0.058 -> -0.075, unwindowed). **This confound
does not explain the production-scale leak.**

### 5b. Ellipticity-WIDTH hypothesis: REFUTED

Paper's Table 1: `sigma_e=0.2` for BOTH tests, eq. 58's
`P(e)~e(1-e^2)^2 exp(-e^2/2 sigma_e^2)` (`imsims.sim._ellipticity`).
`gauss2_fwd` instead uses `_ellipticity_wide` (no `(1-e^2)^2` suppression),
`sigma_e=0.87`. Sampled both: paper's gives mean(e^2)=0.068, ours gives
0.44 — 6.6x larger. Reran the n=150 diagnostic with the galaxy ellipticity
swapped to the paper's EXACT eq. 58 form and sigma_e=0.2: measured `e_gal`
shrank 15x but the debiased leak was UNCHANGED within error
(**+1.60+/-0.29 narrow-e vs +1.22+/-0.43 wide-e**). **Galaxy shape is not
the driver.**

### 5c. Noise-regime hypothesis (RMS(u)/galaxy-size): CONFIRMED, decisively

Noted `median sqrt(Sigma_X)/sqrt(Mr) = 1.378` at `noise_sigma=0.93` — the
RMS recentring shift is LARGER than the galaxy's own size. Reran the
diagnostic (paper's sigma_e=0.2 held fixed) at `noise_sigma=0.93/4=0.2325`
(~4x higher S/N, ratio drops to 0.349): **the orientation-independent
intercept collapsed 16x, from +1.75+/-0.25 to +0.107+/-0.016**; the linear
ellipticity coupling shrank ~18x too (-24.7 -> -1.35). Both terms scale down
together with noise level — consistent with a leak driven by `u`'s
higher (non-Gaussian) moments, which grow with `Sigma_u`/size, not with
galaxy shape.

**Mechanism, complete**: this repo's PSFE test suite runs targets at a
noise depth where centroid-recentring uncertainty is comparable to or
larger than the galaxy's own size. At that scale `recenter()`'s Newton
solve routinely lands far enough from the true centre that `M(u)`'s
higher-order nonlinearity dominates, and BFD's eq. 35/36 Gaussian-kernel
marginalisation (an implicit small-`u` approximation) is not valid. The
paper's own §4.2 test explicitly avoided this via `8<S/N<20` selection and
a generous weight/PSF ratio. **Not a formalism bug, not a population-shape
artifact — a depth/selection choice this codebase's PSFE grid never
applied.**

## 6. Confirmed on production catalogs (no new sims)

Read `cov_odd`/`moments` directly from the real `targets_v3psfe0010_...`/
`targets_v3psfe1p10_...` zero-shear catalogs: **median
`sqrt(Sigma_X)/sqrt(Mr) = 2.96`** in both psfe00 and psfe1p10 (near-identical
— confirms it's not PSF-ellipticity-driven itself). Only 7.2% of targets
have ratio<1.0; 75.7% have ratio>2. Production targets sit even deeper in
the problem regime than the smaller test above.

## 7. A ratio-cut suppresses the leak (point estimate, not yet validated-null)

Matched each `pqr/g2v3d_full2_jac_psfe*.npz`'s zero-arm moments to the raw
catalog's own `cov_odd` (nearest-neighbor, same technique as
`dev/psfe_paired_diff.py`), computed the ratio per target, recomputed raw
`B.bias()` (no selection-term correction) at several thresholds:

| selection | n (per config) | dc1/d(psf_e1) |
|---|---|---|
| unwindowed | ~19940 | -0.058 |
| ratio<2.0 | ~4830 | +0.009 |
| ratio<1.5 | ~2890 | -0.020 |
| ratio<1.0 | ~1430 | +0.022 |
| ratio<0.7 | ~785 | -0.017 |

Every ratio-cut slope is 3-6x smaller and sign-flips — consistent with the
true effect being ~0 once the noise-dominated tail is excluded (no bootstrap
errors computed here — point estimates, not a validated claim).

`corr(ratio, log10(Mf))=-0.898` — the ratio is almost entirely a flux/S-N
proxy. Thread 1's EXISTING validated window (`flux_lo=1500`) barely touches
it (median ratio only drops 2.96->2.26 inside that window) — **the
window's own flux floor is nowhere near aggressive enough.**

## 8. Proper (selection-term-corrected) flux_lo scan: a real trade-off found

A naive raw `flux_lo` scan (no selection correction) was uninformative —
`m1` blew up as the window tightened, an expected artifact of windowing
without the eq. 45-46 non-detection correction, not new physics.

Redone properly via `dev/window_scan.py --no-scan` (real selection terms,
2^24 draws, `flows/centroid_g2v3d_full2_jac_sigmaxfix_naniso100_chunked.eqx`
— confirmed via a frozen, bit-identical chart vs its pre-jac parent that
this flow's lineage WAS trained with `--jac-weight`, i.e. Thread 2's fix is
already baked in):

| flux_lo | psf_e1 | n_in | c1 | m1 corrected |
|---|---|---|---|---|
| 1500 | 0.00/0.02/0.05/0.10 | ~6300 | -0.0086/-0.0093/-0.0101/-0.0103 | +0.0065/+0.0025/-0.0074/-0.0164 |
| 5000 | 0.00/0.02/0.05/0.10 | ~2056 | -0.0079/-0.0082/-0.0085/-0.0084 | **-0.0261/-0.0260/-0.0338/-0.0274** |

`dc1/d(psf_e1)`: 1500 -> -0.020, 5000 -> -0.006 (real ~3x drop). **But**
`m1` jumped to a large, consistent -0.026 to -0.034 at flux_lo=5000 —
because that window triggered a NEW leader-concentration failure in the
selection-term estimator (`WARNING: one draw carries 10% of R_s11`, vs 1.6%
at flux_lo=1500) — the same failure mode Thread 2 fixed elsewhere, now
recurring for a different, narrower window. **flux_lo=5000 trades the
additive-leak problem for a new multiplicative-bias problem; not a clean
validated fix as tested.**

## 9. Two candidate levers for the trade-off in §8, both checked, neither a free lunch

### 9a. More prior draws (2^24 -> higher)

Not run this session, but reasoned from evidence already in hand: the
leader flagged at `flux_lo=5000` sits at `Mf=1.896e4, Mr/Mf=3.079` — right
at the window's own edges (`flux_hi=20000`, `size_hi=3.2`). This repo's own
memory (`[[rs-tail-is-unphysical-draws]]`, `[[size-edge-splits-in-two]]`)
has repeatedly found boundary-adjacent leaders don't fully resolve with more
draws alone — they need a guard/support-density fix. Worth trying (cheap),
not expected to be a complete fix on its own.

### 9b. Wider weight function (closer to the paper's own weight/PSF ratio)

This repo: `WEIGHT_SIGMA=0.65`, `PSF_SIGMA=0.4` -> ratio 1.62. Paper:
weight sigma=3.5px, PSF r50=1.5px -> equivalent ratio ~2.75. Measured
`sqrt(Sigma_X)/sqrt(Mr)` directly as a function of weight_sigma (n=300,
paper's sigma_e=0.2, noise_sigma held fixed):

| weight_sigma | ratio to PSF | median sqrt(Sigma_X)/sqrt(Mr) |
|---|---|---|
| 0.40 | 1.00 | 4.64 |
| 0.65 (current) | 1.62 | 1.57 |
| 1.00 | 2.50 | 1.13 |
| 1.50 | 3.75 | 0.99 |
| 2.00 | 5.00 | 0.94 |

**Real, measurable effect** (1.57 -> ~1.0 moving toward the paper's ratio)
but a much smaller lever than fixing noise depth directly (session 5c's 4x
noise change gave 16x leak reduction; this gives ~37%). Also not free:
widening the weight function shrinks measured `Mf`/`Mr` substantially
(median `Mf` 1645 -> 368 going from sigma 0.65 to 1.5) — a compensated
(KBlackmanHarris) weight changes the whole population's effective depth,
so this would mean rebuilding the entire pipeline (chart, flow, copies
grid) at a new weight scale, not a quick patch.

### 9c. More simulated template galaxies (bigger training/copies catalog)

Checked training-population density directly (`moments_gauss2_fwd_g2v3d.
fits`, 100k galaxies, the only one that exists — no bigger catalog on
disk):

| flux sub-band (size 2.2-3.2) | n (of 100k) | density (per flux-unit) |
|---|---|---|
| [5000,8000) | 4737 | 1.58 |
| [8000,12000) | 2937 | 0.73 |
| [12000,16000) | 1623 | 0.41 |
| [16000,20000) | 1051 | 0.26 |

The `flux_lo=5000` window has 10,348 real galaxies overall (looks fine in
aggregate) but density falls ~6x across the window's own range, and the
flagged leader (`Mf=1.896e4`) sits exactly in the thinnest sub-band — the
SAME density-gradient pattern Thread 3 already found at the bright tail of
the wider window, recurring at smaller scale on this window's own edge.
**A denser render (more simulated galaxies, concentrated near flux~16-20k)
would plausibly help for a specific, mechanistic reason**: better-
constrained flow density there, and less Monte Carlo noise in the flow-free
template-sum tools that draw directly from the copies catalog. Not run
this session — this is the most expensive option discussed (a new render +
partial retrain), not a quick check.

## Bottom line / state at end of session

- **Root cause identified and confirmed at production scale**: the PSFE
  additive leak is driven by targets whose centroid-recentring uncertainty
  is comparable to or larger than their own size (`sqrt(Sigma_X)/sqrt(Mr)`,
  median ~2.96 in production) — not a formalism bug, not galaxy shape, not
  copy-grid density, not a flow/wiring defect (all directly tested and
  ruled out or reconciled this session).
- **Reconciled with the paper**: the paper's own >3000x suppression comes
  from a population/selection choice (decentered bulge+disc, `8<S/N<20`,
  wider weight/PSF ratio) that keeps this same ratio small; this codebase's
  PSFE grid never applied an equivalent selection.
- **A fix is in progress, not landed**: a size/flux window (or explicit
  ratio cut) suppresses the leak in point estimates, but the one concrete
  window tried (`flux_lo=5000`) destabilizes the selection-term estimator
  (`R_s` leader concentration) and introduces a new `m1` bias. Three
  candidate mitigations were checked/scoped (more draws, wider weight
  function, denser training population) — each real but partial, none run
  to completion as an actual fix.

## Concrete next steps, in order of cost

1. Try an intermediate `flux_lo` (2500 or 3000) with proper selection-term
   correction — may raise effective S/N enough to help `c1` without
   starving `R_s` the way 5000 did. Cheapest, not yet run.
2. Rerun `flux_lo=5000` (or the intermediate above) at `--log2-draws 25/26`
   to see if more prior draws rescues the leader-concentration on its own.
3. Add a guard/support-density fix targeted at the boundary leader
   specifically (matching Thread 2's own precedent), rather than only
   varying the window.
4. Longer-term: a denser training render (§9c) and/or a wider weight
   function (§9b), each requiring a partial or full pipeline rebuild —
   scope compute cost before committing.
