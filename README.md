# bfd_cnf

A conditional normalizing flow for the BFD prior of Bernstein et al. 2016
(MNRAS 459, 4467):

    P(m | g, Sigma_X)

where `m = [Mf, Mr, M1, M2]` are the four even moments, `g` the shear, and
`Sigma_X` the covariance of the odd/centroid moments `X` that BFD marginalises
the unknown source position over (paper eq. 36).  The architecture stages that
as a generative stack

    base -> bulk -> shear(g) -> Sigma_X -> data

(`models/bijections.py:new_masked_autoregressive_flow`), and this branch builds
it one arrow at a time against the simulated catalogs from
[`bfd_cnf_imsims`](../bfd_cnf_imsims).

## Phase 1 — the bulk (here now)

`bulk.py` trains only the unconditional part: the `RawMomentStandardize`
coordinate change into `[log10 Mf, logit(Mr/Mf/r*), logit(Mc/Mr/rc*), M1/Mr,
M2/Mr]` — both size-like ratios carry a hard point-source ceiling — followed by
eight `EquivariantAutoregressiveLayer` steps.  The three spin-0 coordinates come
first and the spin-2 pair last; the permutation between layers cycles through
all six orderings of the spin-0 slots so none is permanently the head of the
autoregression, while the spin-2 pair is never permuted — swapping `M1`/`M2`
would rotate the shape by 45 degrees and break the equivariance.  No `g`, no
`Sigma_X` — in the phase-1 sims `Sigma_X` is the same for every galaxy, so
conditioning on it would be learning a constant.

```
python -m imsims.sim --n 100000 --out data/moments.fits   # in ../bfd_cnf_imsims
python bulk.py train  --data ../bfd_cnf_imsims/data/moments.fits
python bulk.py corner --data ../bfd_cnf_imsims/data/moments.fits
```

`plots/bulk_corner.png` overlays the flow on the sims in the four flow
coordinates, with the point-source (`Mr/Mf = 3.692575`) reference marked — above
that line a source is unresolved and carries no shape information.

## Phase 2 — shear conditioning (here now)

`models/shear.py` adds the `ShearResponse` layer, derived from scratch against
the paper's appendix C and checked against `bfd`.  Its functional form is fixed
by three symmetries rather than chosen:

* **spin** — `Mf`, `Mr` are spin 0 and `e = (M1 + iM2)/Mr` is spin 2, as is `g`,
  which leaves exactly three spin-0 and five spin-2 structures through order
  `g^2`;
* **parity** — every coefficient is real;
* **flux** — moments are linear in the image, so the eleven coefficients depend
  only on `Mr/Mf` and `|e|^2`.  Verified exact to six digits over a 1000x flux
  range, which is why the coefficient network has two inputs.

`shear.py` trains it on the density, not on a fitted response map: templates are
lensed by their own exact `Q`, `R` and the flow maximises `log P(m(g)|g)`.  The
analytic derivatives also enter as a low-variance estimator of the same
transport velocity — see that module's docstring for why that costs no
generality at first order, and where it does at second.

```
python shear.py train   --data ../bfd_cnf_imsims/data/moments.fits
python shear.py derivs  --data ../bfd_cnf_imsims/data/moments.fits
python shear.py scatter --data ../bfd_cnf_imsims/data/moments.fits
```

`plots/shear_derivs.png` shows `P` and its first and second shear derivatives
over the `(M1/Mr, M2/Mr)` plane at fixed flux and size.

`scatter` needs no trained flow: it inverts the exact response coefficients out
of the catalog's own `dm_dg` and asks how much of them a regression on the
flux-blind invariants cannot explain.  That residual is `Var[Q|m]`, the piece no
deterministic transport can carry and the floor on the multiplicative bias — so
it says up front how well any flow *can* do on a given population, and whether
modelling `Mc` was enough for it.  Run with and without `Mc` in the regression,
and split by galaxy type where the catalog has one.

## Measured on the two imsims populations

100k templates each, identical settings (6000 steps, bulk frozen and warm
started).  `dm/dg` residuals are RMS over the moment vector, against bfd's exact
appendix-C derivatives; the floor is what `scatter` says the moments cannot
determine even in principle.

| | bulge + disc | Sersic red/blue |
|---|---|---|
| spin-2 `dm/dg`, layer | 2.36% | 0.46% (0.22% at 20k steps) |
| spin-2 floor, `Var[Q\|m]` | 1.42% | 0.12% |
| spin-0 `dMr/dg`, layer | 26.6% | 1.16% |
| spin-0 `dMr/dg` floor | 21.4% | 1.12% |

The layer sits at or near the floor in every case, so the gap between the two
columns is a property of the populations, not of the implementation: eight
parameters of galaxy behind five moments versus five behind five.  Dropping `Mc`
from the regression raises the Sersic spin-2 floor from 0.12% to 9.2%, which is
the sharpest statement yet of why `Mc` is modelled.

What does *not* improve with training is the second derivative: 3.3x more steps
moves `d2m/dg2` for `Mc` from 3.48% to 3.46%.  That is the `Var[Q|m]` term the
module docstring predicts a deterministic transport cannot supply, and it is the
next thing to fix — a stochastic layer, not a bigger network.  (It *did* improve
with cleaner supervision: these catalogs zero-pad the stamps before the FFTs,
which halved that residual from the 7.33% measured without padding.  More steps
still do nothing, which is the point.)

## Sheared targets, and the bias measured on them

`bfd_cnf_imsims` now also produces targets lensed at the **image level**, for
measuring a multiplicative bias from the flow's point estimates against ground
truth rather than against bfd's own derivatives:

```
../bfd_cnf_imsims/data/targets_g1p02_1M.fits   # g1 = +0.02, 1M
../bfd_cnf_imsims/data/targets_g1m02_1M.fits   # g1 = -0.02, same galaxies
../bfd_cnf_imsims/data/targets_g0_1M.fits      # unsheared twin, for binning
```

These used to carry a floor: the image response and bfd's analytic `dm_dg` —
what this flow is trained on — differed coherently by 0.1–1%, growing towards
the resolution limit, worth `m1 ~ -6e-3` on the bulge + disc population.  That
was a moving-boundary term in bfd's appendix-C derivatives, traced to the
Blackman-Harris weight not reaching zero at `kmax`, and it is **fixed** — see
that repo's README.  Population-weighted `sum(FD)/sum(Q) - 1` on `M1` is now
`+0.006%` against `-0.61%` before, and flat across resolution.  Every catalog
and flow here was regenerated after it (`../bfd_cnf_imsims/regen.sh`, then
`retrain.sh`).

On noiseless targets, which is where this measurement is currently sharpest:

```
python bias.py --flow flows/shear.eqx        # 1M bulge + disc, g1 = +/-0.02

  m1 = -0.060 +/- 0.010
```

**That number is the converged one, and it used to read `+0.00128`.**  The
difference is not a regression: `bulk.py train`'s old 4000-step default left the
bulk flow visibly unconverged, and the amount it was unconverged by was setting
the answer —

| bulk steps | val nll | window mass vs catalog | noiseless `m1` |
|---|---|---|---|
| 4000 | 40.4405 | +3.70% | -0.0126 |
| 20000 | 40.4199 | +1.70% | -0.0227 |
| 60000 | 40.4070 | +0.83% | -0.0428 |
| 150000 | — | +0.61% | -0.0600 |

Both columns are monotone, so any intermediate step count is a choice about
which error to hide; only the converged end is a property of the *method*.  The
shear layer, by contrast, is already converged at its own default (6000 / 20000
/ 60000 steps on a converged bulk give -0.064 / -0.056 / -0.067).  `retrain.sh`
now trains the bulk for 150000 steps (~21 min a flow against ~40 s) and the old
flows are kept in `flows/pre_convergence/`.  **Every `m1` quoted in this repo or
in `HANDOFF.md` from before 2026-08-19 is an undertrained-model number.**

Rebuild scatter on the converged chain is `sd ~ 0.010` across five independent
rebuilds, far above any single run's bootstrap error — quote `-0.07 +/- 0.01`
and do not read a difference below ~0.02 between two chains.

What is left at -0.06 is **the spin-0 shear response**, by elimination: shear
steps, template count, derivative weight and likelihood overfitting are all flat
(see `retrain.sh` and the memory trail), while `dm/dg` against bfd's exact
derivatives is 17% (Mf), 28% (Mr) and 46% (Mc) against 2.6% for spin-2, and 100x
more supervision weight moves those by 0.07 percentage points.
`dev/check_a0_representable.py` then removes the flow entirely and fits the same
coefficient with the same net, inputs and bound to 0.002 — against +0.38 in
flow — so it is the **joint optimisation**, not capacity, conditioning or the
choice of inputs.  That is the open architectural thread.

Note the contrast with the noisy, windowed measurement below, which is
consistent with zero on the *same flows*: a noiseless run is a point evaluation
and reads the density's fine structure directly, where the `C_M` integral
averages over it.  Which of those two is the honest statement of the method's
accuracy is not settled — see "What is not measured" at the end.

An older fix worth keeping in view, from when this number was small enough for
it to matter: `RawMomentStandardize` was standardising
`M1/Mr` and `M2/Mr` with independent per-coordinate mean and scale, which is the
one place in the stack that could give the prior a preferred direction on the
sky, and a prior with a preferred direction reads out as additive shear.
Symmetrising it (`_effective()`, zero spin-2 mean and one shared scale, applied
on use so the optimiser cannot undo it) took `c2` from `+3.78e-4` to `+3.2e-5`
and `m1` from `+2.05e-3` to the number above.
`tests/test_shear.py::test_flow_isotropy` guards it.

## Noisy targets, and the integral under `C_M`

A real target is `M = M^G + M^n` with `M^n ~ N(0, C_M)` (paper eq. 5), and the
probability the ensemble estimator needs is the prior *convolved* with that
noise — the continuum limit of the paper's template sum, eq. (38):

    P(M | g) = INT dm P(m | g) L(M - m),   L = N(., C_M)

`bias.py --samples S` does that integral by drawing `S` points from the noise
kernel itself and averaging the flow over them (`bias.log_conv`).  Because the
draws do not depend on `g`, `Q` and `R` (eq. 12–13) are just that one estimator
differentiated — common random numbers throughout, never three noisy quantities
divided by each other.  The catalogs are still noiseless on disk; the
realization is added at load time, in moment space.  That is exact rather than a
shortcut: at a **fixed** centre the moments are a linear functional of the image,
so Gaussian pixel noise gives exactly `N(0, C_M)` in moment space.
`bfd_cnf_imsims`'s `test_moment_noise_is_the_stored_covariance` checks it against
actual noisy images, which is also the only test of `C_M` itself through the
padded FFT.

```
python bias.py --flow flows/shear.eqx --samples 1024 --n-targets 20000
python -m tests.test_conv        # the integral against an analytic Gaussian
```

The cost is `S` flow evaluations per target per catalog: ~33 ms/target at
`S = 1024`, so 20k targets in each of the three catalogs is about half an hour.

**Kernel sampling is fine on the median target and poor on the faint tail.**
The ESS printed at startup is the paper's own diagnostic — sec. 2.5 warns that
dividing `Q` and `R` by a noisy `P` biases the answer inversely with the number
of templates contributing to `P`, and here that number is the ESS, so the
induced `m` is of order `1/ESS`:

| draws/target | median ESS | 5th pct | frac. below 10 |
|---|---|---|---|
| 1024 | 106 | 5 | 8.2% |
| 4096 | 383 | 21 | 2.3% |
| 16384 | 1545 | 56 | 2.0% |

The median scales linearly with `S`, as it must, but ~2% of targets sit below
ESS 10 no matter how many draws they get.  Those are the faint, barely resolved
ones, where the kernel is far wider than the thin `Mc` sheet the prior lives on,
so blind draws almost never land on it.  Buying that tail with more draws does
not work; it needs a proposal that knows where the prior is — which is what the
paper's k-d tree does when it keeps only templates within `chi^2 < sigma_max^2`
of the target (sec. 3.2).

That tail is what dominated the first measurements at `S = 1024`, which could
say nothing sharper than `+/-1%`.  `S = 8192` is now the working setting and the
per-target error is no longer what limits the answer; see the end-to-end numbers
below.

**One bug lived in this path for a long time and deserves to be findable.**
`pqr_streamed` passed its domain mask `ok_raw` into `log_conv_is`
unconditionally.  With no centroid peel the draws are still *raw* moments, but
that branch stands masked rows in at `jnp.zeros` — a valid standardised
coordinate, an invalid raw moment (`Mf = 0` → `log10(0)`).  The `where` hides it
from the value and `0 * inf` NaNs the **gradient**, so every chunk of any target
with even one off-chart draw was marked bad: 15.7% of bulgedisc silently got
`Q = R = 0`.  Those targets sit at the chart edge, so the survivors were a
latent-selected sub-ensemble, and the "+0.013 noisy hump" that several
`HANDOFF.md` sections chase was that selection, not the flow's density:

```
  control    +0.0159 +/- 0.0015 (3196 dead)  ->  +0.0003 +/- 0.0178 (0 dead)
  bulgedisc  +0.0131 +/- 0.0015              ->  +0.0008 +/- 0.0020
```

Only the **no-centroid streamed** path was affected (`--pop bulgedisc`,
`gauss2`, controls without `--centroid`); the image-noise path peels first, so
zeros is a legitimate stand-in there, and the non-streamed `pqr` path was never
affected.  The tell is the `N targets had no draw with any weight` line reading
above ~0.  **Any noisy number in `HANDOFF.md` taken through `pqr_streamed`
without `--centroid` is suspect — re-measure before reasoning from it.**

## Phase 3 — centroid marginalisation (here now)

A real target's centroid is not known: it is *found*, by solving `X = 0` on a
noisy image, so the moments a target reports are those of the galaxy seen from a
slightly wrong origin.  BFD never assumes the template's origin either — each
template is replicated over a grid of origins `u` and the prior is the weighted
sum over that grid (eq. 36) — and `models/centroid.py` is that marginalisation as
a layer.  It is data-adjacent, so the stack is now complete:

    base -> bulk -> shear(g) -> centroid(Sigma_X) -> data

Centroid comes last because it happens last: a galaxy is lensed on the sky and
only then measured about a centroid somebody had to guess.

```
python -m imsims.copies --n 100000 --out data/copies_bulgedisc.fits   # in ../bfd_cnf_imsims
python centroid.py train --copies ../bfd_cnf_imsims/data/copies_bulgedisc.fits
python -m tests.test_centroid
```

**What it buys.**  The marginalisation coherently *amplifies* apparent
ellipticity (the copies' `|e|` grows faster than their position-angle scatter
kills it), by `+3.5e-4` at `S/N > 40` rising to `+1.4e-2` below 20 — a
multiplicative shear bias of the same size if left out.  Against the 100k copy
catalog, 8000 steps:

```
         layer shift  catalog shift      resid
  Mf      -5.146e-04     -5.795e-04  2.398e-04
  Mr      -1.233e-03     -1.164e-03  3.096e-04
  M1      -8.382e-07     -8.581e-07  1.756e-04
  M2      -3.232e-06     -9.136e-07  1.220e-04
  Mc      -1.677e-03     -1.734e-03  2.582e-04
  ellipticity response  catalog +2.6925e-03   layer +3.1858e-03
```

So the spin-0 shifts land within 3–11% and the ellipticity response within 18%,
i.e. the layer removes about 5.5× of the bias and leaves `~5e-4`.  The `resid`
column — per-galaxy RMS, larger than the mean shifts themselves — is the
`Var[.|m]` floor again: two galaxies with identical moments marginalise
differently, and a deterministic transport can only carry the conditional mean.
Same diagnosis and same eventual fix as the shear layer's `d2m/dg2`.

**The form is fixed by symmetry, not chosen.**  `Sigma_X` is a symmetric
2-tensor, so it splits into a spin-0 trace and a spin-2 traceless part exactly as
`(|g|², g)` does for shear, and the same three symmetries then fix the map.  The
variable is not `Sigma_X` but the displacement covariance it induces,
`Sigma_u = J^-1 Sigma_X J^-T` with `J = dX/du` bfd's own `xyJacobian`, made
dimensionless as `T = (Mr/Mf) Sigma_u`.  The marginalisation is second order in
`u` and hence first order in `T`, leaving two spin-0 structures and three spin-2
ones — nine real coefficients, functions of the same three flux-blind invariants
the shear layer uses.

One subtlety worth stating because it is easy to get backwards: with `Sigma_X`
isotropic `t2` does **not** vanish, because `J` is elliptical and an elliptical
galaxy's centroid error is anisotropic — larger along the major axis, where the
flux gradient is shallower.  What holds is that `J` depends only on the galaxy's
own `e`, so `t2` comes out *parallel* to `e`: the marginalisation rescales
ellipticity without rotating it and picks out no axis on the sky.  Only a
genuinely anisotropic `Sigma_X` — an elliptical PSF — can turn a galaxy, and then
it should.  `tests/test_centroid.py` pins both halves, in float64 (see below).

**Trained by supervision, not by likelihood alone.**  The marginalisation moves
the moments by `~1e-3` fractionally, which is worth far less likelihood than the
batch noise on a 1024-galaxy NLL — on the NLL alone the `nll` does not move at
all.  So `centroid.py` adds a supervised MSE against the catalog's weighted copy
mean, exactly as `shear.py` does against bfd's `dm/dg`, and here the supervision
is *exact* rather than approximate: the weighted copy mean **is** eq. (36)'s
first moment, computed from the catalog with no model in between.  It took the
shift MSE from `6.1e-6` to `5.3e-8`.

**Sampling needs no importance weights.**  Copy weights span `e^-8` across a
galaxy's grid, so a flat batch of copies has ESS ~13% of its length — the reason
this needed SNIS before.  It does not need it here: `CopySampler` draws a galaxy
in proportion to the *sum* of its copy weights, then one copy within it in
proportion to `w`, which is exact stratification at full ESS.

That per-galaxy sum is the galaxy's detection probability.  On the shallow
catalog it is 1.00000 for every galaxy, but it is not an identity: at
`noise_sigma = 2.73` the faintest ~1% fall below 0.99 and the worst to 0.69,
because their centroid distribution reaches the `det J <= 0` boundary
`makeTemplates` cuts on — the paper's positive-Jacobian assumption (§5.3)
failing at low S/N, not a grid that is too small.  Those galaxies genuinely are
less likely to be detected, so drawing galaxies by that sum rather than
uniformly is the correct weighting, and it is what makes the sampler right at
both depths.  A flux cut would move the same quantity further from 1 without
changing anything else here.

**A term no eq.-(36) prior can absorb.**  See the imsims README: recentring lands
on a maximum of the *noisy* flux surface, so the moment noise at the found origin
is correlated with the first-moment noise that moved the origin there — which
eq. (26) factorises away.  Measured at `+4 sigma_XY²/Mr` on `Mf`, i.e. `+0.18σ`
at `S/N 14`, isotropic and so spin-0 only.  It is a floor on this whole approach,
not something the layer is failing to learn.

## Noisy targets, end to end

`bfd_cnf_imsims` produces targets with pixel noise in the *image*, measured with
`recenter()`, so the centroid is found rather than assumed
(`data/targets_noisy_g{0,1p02,1m02}_200k.fits`), and `bias.py` runs on them:

```
python bias.py --flow flows/centroid.eqx --pop bulgedisc_noisy --samples 2048
python bias.py --flow flows/shear.eqx --no-centroid --pop bulgedisc_noisy --samples 2048
```

The `IMGNOISE` header decides the mode.  For such a catalog `bias.py` does *not*
apply `add_noise` — adding `N(0, C_M)` at load time is exact only at a fixed
centre, and these already carry their noise in the image, where recentring can
respond to it — and it switches the centroid layer on.  `--samples 0` is refused
there: a point evaluation of the prior at a noisy `M` lands in its tails.
Targets whose recentring did not converge (`badcenter`) are dropped from all
three catalogs together, so the `+g`/`−g` pairing stays aligned galaxy for
galaxy.

**The `jacfwd` never enters the shear derivative.**  The centroid layer is
data-adjacent, so in data → base order it is applied first, to the raw moments,
conditioned only on `Σ_X` — both its input and its condition are independent of
`g`.  Hence

    log p(x | g, Σ) = log p_rest(centroid(x, Σ) | g) + log|det ∂centroid/∂x|

with the second term exactly `g`-independent, contributing nothing to `Q` or
`R`.  `split_centroid` peels the layer off and `centroid_transform` applies it
once, folding the log-det into the importance weight; the forward-over-reverse
Hessian then never traverses the layer's 5×5 `jacfwd`, which is 5 extra JVPs of
the coefficient network *per draw* and is what made the Hessian run out of
memory at the batch sizes the shear-only flow used.  This is exact, and
`tests/test_centroid.py::test_peeling_the_layer_off_is_exact_and_g_independent`
checks both the identity and that the offset carries no shear signal.

### Cost, and the two numbers that set it

`S` flow evaluations per target per catalog, so the integral is the whole cost:
`~33 ms/target at S = 1024` on the shear-only flow, scaling linearly in `S`.  The
noiseless path (`--samples 0`) is a *point* evaluation of the prior and ~1000x
cheaper per target; that is why the 1M-target noiseless runs above are quick and
these are not.

Three fixes took the deep configuration (`S = 32768`) from 60 to **1384
targets/min**, all of them plumbing rather than physics:

* `centroid_transform` built its `eqx.filter_jit` wrapper *inside* the function,
  so a fresh wrapper was made per call and JAX recompiled on **every chunk** —
  1181 ms against 11.7 ms once hoisted to module scope, and it had been 91% of
  the per-chunk time.  Worth remembering as a class of bug: a jit wrapper
  created in a function body is a recompile, not a cache hit.
* `pqr_streamed`'s per-target kernel called `f`, `jax.grad(f)` and
  `jax.hessian(f)` separately — three evaluations of the flow, when `hessian` is
  `jacfwd(grad)` and had already computed the other two.  One `jax.jvp` of
  `value_and_grad` per `g` direction returns the value, the gradient and a
  Hessian column together.  Verified bit-identical, per target.
* `mixture_draws` at `alpha = 1` formed `log_k` and then returned
  `log_k - log_k`.  The kernel is its own proposal there, so the weights are
  identically zero and the triangular solve over every draw is pure waste.

What is *not* removable: the centroid layer's log-det.  It looks like
`log(1 + O(T))` with `T ~ 4e-3` and therefore negligible, but measured it varies
by **18 nats** across a chunk — kernel draws at depth reach the poorly resolved
region where `T` is large — and dropping it moves `R` by 7.9%.

## Phase 4 — selection (here now)

A real analysis does not use every detection: it cuts, on flux and on size, and
the paper's eq. (40) and (45)–(46) are what makes such a cut unbiased.  Those
terms are now implemented — `window_prob`, `selection_terms`, and an `ns=`
argument threaded through `ghat`/`bias`/`bootstrap`:

```
python bias.py --flow flows/centroid.eqx --pop bulgedisc_noisy --samples 8192 \
    --chunk 4096 --window-size 2.2 3.2 --window-flux 2500 50000
python -m tests.test_selection      # synthetic, no flow, no FITS, seconds
```

`window_prob` is eq. (30)'s `INT_{M in S} dM L(M - M^G)`: the probability a
galaxy's *noisy* moments land in the window, which is what the estimator needs
because it never sees the non-selected galaxies' own `M`.  The ratio cut
`s0 < Mr'/Mf' < s1` is recast as two linear constraints and integrated by a 1-D
Gauss–Legendre quadrature over the flux direction — exact, not approximate,
whenever the flux window has a non-negative floor (0 disagreements against the
literal ratio cut in 2e6 noise draws; guarded, since it genuinely fails for an
unbounded flux window where `Mf'` can go negative).  `selection_terms` then
differentiates `P(s|g) = E_{m ~ P(.|g)}[F(m)]` twice at `g = 0`.

Two factors of eq. (30)/(38)/(40) are deliberately absent: `|J(M)|` is a
function of `M` alone, so it cancels exactly out of every `Q_i`, `R_i` (those
are `g`-derivatives at fixed `M_i`), and `L(X^G)`'s grid sum is what the
centroid layer already carries.  That was checked, not asserted: pushing the
true template population through `window_prob` gives `P_s = 0.3033` against the
noisy catalog's own measured selection fraction of `0.3016`.

Measured on `bulgedisc_noisy`, 40k targets, `S = 8192`, window
`2.2 < Mr/Mf < 3.2` and `2500 < Mf < 50000`:

```
  no window                                 -0.0021 +/- 0.0034
  cut on each arm's own M, uncorrected      +0.0354 +/- 0.0009
  cut on each arm's own M, + eq.(45)/(46)   -0.0054 +/- 0.0036
  cut on the g = 0 twin, uncorrected        +0.0171 +/- 0.0009
  cut on the g = 0 twin, + eq.(45)/(46)     -0.0230 +/- 0.0009   <- WORSE
```

**The correction is only valid for a cut on the target's own observed
moments.**  There, eq. (29) leaves `P_i` untouched and the whole error is the
missing non-selection term.  A cut on the `g = 0` twin restricts which
*galaxies* are in the sample, so the prior itself is wrong, and applying the
boundary correction over-corrects by about as much as it helps.

The correction is carried almost entirely by `R_s`: `Q_s` is exactly zero by
isotropy for a spin-0 window (`P_s` can only depend on `|g|^2`), measured
`+3e-4 +/- 6e-4`, and forcing it to zero moves `m1` by less than `1e-5`.
`R_s = 0.79·I`, isotropic to 0.9% — a free correctness check the code does not
enforce, so watch it.  `Q_s_err` says whether the draw count was enough.

`--window-terms` chooses where those terms come from: `templates` (default)
lenses the training catalog by its own exact `dm/dg`, which is what eq. (40)
literally is; `flow` integrates the fitted prior instead, which is what you
would have to do on real data.  The gap between them is a direct measure of the
flow's density error at the window edge, and it is currently **+4.1%** — the
flow puts `P_s = 0.3157` and `R_s = 0.794·I` against truth's `0.3033` and
`0.763·I`, one coherent normalisation error.  It costs `0.002` in `m1`.

Reproduced on the converged chain with two independent training seeds, same
targets and noise realisation, whole chain rebuilt: corrected windowed `m1` of
`+0.00007 +/- 0.00360` and `+0.00214 +/- 0.00347`.  Two chains differing by
0.0021 against a combined error of ~0.005 makes that a property of the
**method**, not of one training run.

## What is not measured

* **The Poisson / sky branch**, eq. (53)–(55).  Everything here is the
  postage-stamp branch, with `N_ns` counted; one galaxy per stamp, already
  detected.
* **Varying `C_M` or `Sigma_X` across the catalog.**  The formalism allows it
  and `bias.py` reads `Sigma_X` per row, but `selection_terms` assumes one
  `C_M` for the whole measurement, as these catalogs have.
* **Magnification** (paper sec. 6.4) — the packed output is `PqrNoMu`.
* **Multi-exposure / multi-band** (sec. 6.2).
* **Why the noiseless and noisy numbers disagree by 0.06 on the same flows.**
  Plausibly the `C_M` integral averaging over fine density structure that a
  point evaluation reads directly, but nobody has demonstrated it.  Until
  someone does, the honest statement is that the method's accuracy is bracketed
  by the two, not that it is the smaller.  The cheap experiment is a
  `--noise-scale` ladder on fixed targets, watching `m1` walk from -0.06 to ~0.

## Layout

```
bulk.py               phase-1 build / train / corner plot
shear.py              phase-2 train / check / scatter / shear-derivative plot
centroid.py           phase-3 train / check, and the CopySampler
bias.py               m and c on the targets, noiseless or integrated under
                      C_M, with eq. (40)/(45)-(46)'s selection terms
models/shear.py       the ShearResponse layer
models/centroid.py    the CentroidMarginalize layer
models/bijections.py  the bulk layers
dev/                  one-off diagnostics; each script's docstring says what it
                      answered and what the answer was
tests/                pytest; the whole suite is ~7 min on one GPU
HANDOFF.md            the dated investigation log
NOISY_BINNING_BIAS.md why binning a noisy m1 profile on the target's LATENT
                      Mr/Mf biases it even under a perfect density -- read this
                      before believing any binned noisy profile in HANDOFF.md
```

Two standing warnings about the older half of that log, both established
2026-08-19 and both large enough to invert conclusions: every `m1` from before
that date is an **undertrained-bulk** number (see above), and every noisy number
taken through `pqr_streamed` without `--centroid` is a **NaN-gradient** number
(see above).  `NOISY_BINNING_BIAS.md` is the third: binning on a latent quantity
biases the per-bin `m1` even under a perfect prior.

## Note on precision

The flow runs in float32.  `models/centroid.py`'s spin-2 part `t2` is ~1% of
`t0`, so forming it as a difference of two nearly-equal `Sigma_u` diagonals loses
most of its float32 significance: the spin-2 response picks up a component
perpendicular to `e` of up to 1.5%, against `5e-11` in float64.  It is structured
by the coordinate axes (period `pi/2`, alternating sign) and averages to zero
over an isotropic population, so it does not read out as additive shear — but it
is why `tests/test_centroid.py` runs its symmetry checks in float64, and the
analytic fix is noted in `_tensor` should it ever need to be exact.

## Notes

The layer stack is kept whole even though phase 1 uses a slice of it, so the
conditional layers slot in later without rebuilding the bulk.  `models/flows.py`
(the ELBO/shear-supervision losses) and the training/inference drivers are *not*
on this branch: they were wired to a `config.py`/`data.py` pair built around the
old DES-derived catalogs, and will be rewritten against the sims as phases 2-3
land.  They remain on `main` and `working`.

## Requires

`jax`, `flowjax`, `equinox`, `optax`, `paramax`, `numpy`, `fitsio`, `corner`,
`matplotlib`.
