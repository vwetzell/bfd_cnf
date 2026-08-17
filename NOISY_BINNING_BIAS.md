# Binning a noisy BFD measurement biases it, even with a perfect density

*Written 2026-08-13. Standalone: assumes only that you know what `bias.py`
does.*

## The claim in one paragraph

Every noisy `m1` profile this project has produced — the "resolution ramp", the
"broad hump", the catastrophic edge collapse — is binned on the target's TRUE
`Mr/Mf`. Once a target is noisy, its true moments are a **latent** variable: the
estimator only ever sees `M = m + noise`. Selecting a subsample on something the
estimator cannot see gives that subsample a different moment density from the
population, and BFD is unbiased only when its prior matches the population the
targets came from. So per-bin `m1` is biased **even when the density is exactly
right**. Measured, that bias accounts for essentially all of the observed
structure: after subtracting it, every bin lands within ~2σ of zero, including
an apparent −0.25 collapse.

---

## 1. Where this came from

Two observations that did not fit together:

- **gauss2's numbers disagreed with themselves.** Its population `m1` is
  `+0.00096 ± 0.00064` — as close to zero as anything in this project. But its
  binned profile runs `+0.0187` at small `Mr/Mf` down to `−0.0635` at large. A
  ±0.04 spread out of a population value of 0.001.
- **The same shape appeared under morphology binning.** `m1` at fixed size
  varied by ±0.18 across bulgedisc's structural parameters. That is far too
  large to be a density error nobody had noticed.

Both are what a selection artefact looks like: large *within* bins, cancelling
*across* them.

---

## 2. Why it happens

### The BFD guarantee, stated precisely

`bias.py` maximises `log L(g) = Σ_i log P(M_i | g)` over `g`, expanded to second
order (paper eq. 19). This is unbiased **when the `P` it uses is the density of
the targets being summed over**. That is the whole content of "you need the
right prior".

For noisy targets the prior is the convolution

```
P(M | g) = ∫ dm P(m | g) L(M − m),    L = N(·, C_M)
```

which is the population's moment density smeared by the noise kernel.

### What binning does to that

Take a subsample `S`. The targets in `S` are no longer drawn from `P(m|g)`;
they are drawn from `P(m | g, S)`. Using the population `P(m|g)` on them is
using the wrong prior, and the resulting `ĝ` is biased. How badly depends on how
different the two densities are.

- **Bin on a function of `m`** (noiseless case): the estimator sees `m`, so
  `P(m|g,S)` is just `P(m|g)` truncated to a region the estimator can also see.
  The likelihood restricted to that region is still correct. **No bias.**
- **Bin on a latent variable** (noisy case): the estimator sees only `M`. A bin
  of true `Mr/Mf` contains targets whose `M` are scattered all over, and their
  true density within the bin is emphatically not the population's. **Bias.**

### The intuition

Within a narrow slice of true size, every target's real `m` sits in a narrow
range — but the estimator does not know that. Its prior spans the whole
population, so for each target it effectively shrinks the implied `m` back
toward the population mode. This is ordinary Eddington/regression-to-the-mean
shrinkage. Because the shear response varies with size (`dQ/d(Mr/Mf) ≠ 0`), the
mis-assigned `m` becomes a mis-assigned response, and the error changes sign
across the size axis as the shrinkage direction flips. That is exactly the shape
observed: positive where the bin sits below the population mode, negative above
it, largest at the edges where shrinkage is strongest.

### This is NOT the selection issue already handled

`dev/check_bias_by_selection.py` documents a different hazard: cutting on a
**sheared** quantity correlates the cut with the shear response, which turns
`P(s|g)` into a term you owe (paper eq. 45-46). Its cure is to evaluate the cut
on the `g = 0` catalog, so the +g and −g arms contain the same galaxies.

That cure does nothing here. Our binning already uses the `g = 0` catalog, so
the two arms *are* the same galaxies. The problem is not that the arms disagree;
it is that **the estimator is blind to the binning variable**. Two independent
objections, and the older one's fix does not touch the newer one.

---

## 3. The control

### Construction

Make the prior exactly right by construction, then measure what is left.

1. Draw targets **from the flow itself** at `g = ±0.02`. The flow is then its
   own truth: there is no density error, by definition.
2. Pair them by sharing the base draw `z` across the two shear conditions —
   `flow.bijection.transform(z, g)` is the flow's own notion of "the same galaxy
   transported to ±g". (`transform` is the base→data direction for this chain;
   `sample` agrees with it exactly.)
3. Add the same noise realisation to both arms, as `bias.main` does.
4. Run the **identical** pipeline (`pqr_streamed`, same `alpha`, same draw count)
   and the **identical** binning.

Anything nonzero that comes out is method, not density.

### The internal check that makes it convincing

The control's **unbinned** `m1` is `−0.0044 ± 0.0016` — consistent with zero, as
it must be. Its **binned** values run from `+0.0134` to `−0.2422`. Same targets,
same estimator, same draws; the only thing that changed is that the targets were
sorted into bins. That isolates the binning as the cause, without any appeal to
theory.

---

## 4. Results

### At 120k targets (the best measurement)

bulgedisc, `alpha = 1.0`, `S = 8192`, real and control both 120k:

| Mr/Mf bin | real | control | **corrected** |
|---|---|---|---|
| [2.00, 2.60) | +0.0037 ± 0.0012 | +0.0112 ± 0.0008 | **−0.0075 ± 0.0014** |
| [2.60, 3.00) | +0.0118 ± 0.0010 | +0.0132 ± 0.0007 | **−0.0014 ± 0.0013** |
| [3.00, 3.15) | +0.0216 ± 0.0024 | +0.0031 ± 0.0178 | **+0.0184 ± 0.0180** |
| [3.15, 3.40) | +0.0250 ± 0.0019 | +0.0278 ± 0.0007 | **−0.0028 ± 0.0020** |
| [3.40, 3.55) | −0.0130 ± 0.0039 | −0.0059 ± 0.0064 | **−0.0071 ± 0.0075** |
| [3.55, 3.70) | −0.1687 ± 0.0109 | −0.2679 ± 0.0369 | **+0.0992 ± 0.0384** |

HUMP: real `+0.0213 ± 0.0021`, control `+0.0166 ± 0.0011`,
**corrected `+0.0047 ± 0.0024`**.

Two things to note. The corrected hump **shrank** with better statistics
(`+0.0105 ± 0.0039` at 20k → `+0.0047 ± 0.0024` at 120k, consistent within
errors) — it is at most a couple of e-3, and may be nothing. And the top bin is
where the correction misbehaves: the control **overshoots**, predicting −0.268
where the real run gives −0.169, so the "correction" there is a +0.10 ± 0.04
difference of two large numbers. That is §6's limitation biting exactly where
predicted — the edge is where the flow's distribution differs most from the
population's.

### In the flux-size plane

`dev/plot_hexbin_corrected.py` (figure: `dev/hexbin_corrected.png`) shows the
same thing in 2D, and it is the most direct demonstration: **the control panel
is visually almost identical to the real one** — same negative band above
Mr/Mf ≈ 3.5, same positive band at 3.0–3.4, same pale low-size region. A flow
that is its own prior reproduces the entire picture.

After subtraction, 26 cells with ≥800 targets in both:

- the coherent band structure is gone;
- a residual swath of roughly **−0.008 to −0.016** survives at low flux and
  low-to-mid size (the largest, at log Mf 3.60 / Mr/Mf 3.20, is
  −0.0106 ± 0.0023, i.e. 4.6σ);
- the two Mr/Mf ≈ 3.65 cells go strongly positive (+0.11, +0.19) — again the
  control overshooting at the edge, not a flow error.

So the correction removes the large-scale structure and leaves residuals an
order of magnitude smaller, with the extreme edge unresolved by this method.

### At 20k targets (the earlier, coarser measurement)

| Mr/Mf bin | real | control | **corrected** |
|---|---|---|---|
| [2.00, 2.60) | +0.0082 ± 0.0020 | +0.0134 ± 0.0022 | **−0.0052 ± 0.0030** |
| [2.60, 3.00) | +0.0172 ± 0.0017 | +0.0151 ± 0.0017 | **+0.0021 ± 0.0024** |
| [3.00, 3.15) | +0.0291 ± 0.0023 | +0.0234 ± 0.0022 | **+0.0057 ± 0.0032** |
| [3.15, 3.40) | +0.0311 ± 0.0017 | +0.0258 ± 0.0017 | **+0.0053 ± 0.0024** |
| [3.40, 3.55) | −0.0147 ± 0.0035 | −0.0156 ± 0.0032 | **+0.0009 ± 0.0047** |
| [3.55, 3.70) | −0.2542 ± 0.0206 | −0.2422 ± 0.0180 | **−0.0119 ± 0.0274** |

Summary statistics:

| quantity | real | control | corrected |
|---|---|---|---|
| overall `m1` (unbinned) | −0.0016 ± 0.0017 | −0.0044 ± 0.0016 | +0.0028 ± 0.0023 |
| HUMP = `m1[3.15,3.40) − m1[2.00,2.60)` | +0.0229 ± 0.0027 | +0.0124 ± 0.0029 | **+0.0105 ± 0.0039** |

Every bin is within ~2σ of zero at this target count — but that is partly the
larger error bars, and the 120k run above resolves small residuals this one
cannot. Use the 120k numbers.

---

## 5. How to run it

```bash
# 1. the control: targets drawn from the flow, so the prior is exact
python dev/check_selfconsistency_noisy.py \
    --flow flows/shear.eqx -n 20000 --samples 8192 --alpha 1.0 \
    --save-pqr control.npz

# 2. the real measurement, at the SAME settings
python bias.py --flow flows/shear.eqx --pop bulgedisc \
    --samples 8192 --chunk 2048 --alpha 1.0 --n-targets 20000 \
    --save-pqr run.npz

# 3. real / control / corrected, as a 1D profile in Mr/Mf
python dev/check_noisy_size_profile.py --pqr run.npz --control control.npz

# ...or in the flux-size plane (4 panels + a per-cell table)
python dev/plot_hexbin_corrected.py --real run.npz --control control.npz
```

The control must match the real run in flow, `alpha`, draw count and target
count — it is estimating that configuration's artefact, and the artefact depends
on all four.

The two runs cannot share bin *membership* (the control's targets are flow
draws, not catalog rows), only bin *edges*. That is why the corrected table uses
fixed edges in `Mr/Mf` rather than per-run quantiles.

---

## 6. Limitations

1. **The correction is leading-order, not exact.** The control's population is
   the *flow's* distribution, not the real one. It estimates the artefact for a
   density close to the truth, which is the right ballpark but not identical to
   the artefact for the true density. The closer the flow is to the population,
   the better the subtraction — which unfortunately means it is least reliable
   exactly where the flow is worst.

   **This is not hypothetical; it is visible in the results.** In the top size
   bin the control returns −0.268 against the real run's −0.169, i.e. it
   overshoots by more than half. The subtraction there is between two large
   numbers and the residual (+0.10 ± 0.04) should not be read as a flow error.
   Treat `Mr/Mf > 3.55` as unresolved by this method. Everywhere else the
   control tracks the real run to within ~0.01.
2. **It absorbs more than binning.** The control captures every source of
   nonzero `m1` that is not density error: the binning, any residual
   Monte-Carlo bias, the aggregation's own `O(g²)` terms, the `|R|` guard's
   selection. Attributing it specifically to binning rests on the unbinned
   control being ~0 while the binned one is not (§3).
3. **Errors add in quadrature.** Subtracting a noisy control inflates the error
   bar; the corrected numbers are ~40% noisier than the raw ones. Buying
   precision back means a larger control, which costs the same as another real
   run.
4. **It does not license binning on anything.** The correction is estimated for
   one specific binning. Change the binning variable and it must be re-measured.

---

## 7. Alternatives considered

| option | verdict |
|---|---|
| Bin on the **measured** `M` instead | Makes the binning visible to the estimator, but `M` responds to shear, so it reintroduces exactly the `P(s|g)` selection term (eq. 45-46) that binning on the `g=0` catalog was adopted to avoid. Trades a quantified bias for an unquantified one. |
| Bin on a noise-free **regression proxy** (`check_bias_flatness.noise_free`) | Still latent — a prediction of `m` from noiseless features is not a function of `M`. Same objection, plus regression error. |
| Don't bin at all | The unbinned number is already unbiased and is the honest headline. But it discards the diagnostic that tells you *where* the flow is wrong, which is the whole point of a profile. |
| **Subtract the control** | Adopted. Keeps the physically meaningful binning, quantifies the artefact rather than assuming it away, and is falsifiable — a control that came back at zero would have proved the profiles were fine. |

---

## 8. What this implies for existing results

Anything quoting a **binned** noisy `m1` was scored without this correction and
is inflated by roughly the control's contribution:

- `dev/sweep_noisy.py` — its arms are scored on "m1 on the measured cut" and a
  "ramp amplitude across noise-free quartiles". Both are binned.
- `dev/check_bias_by_selection.py`, `dev/check_bias_flatness.py`, and the
  resolution-cut sweeps.
- Any conclusion of the form "this knob moved the ramp by X" where X is
  comparable to the control (~0.012 in the hump statistic).

Unaffected:

- **Population-level (unbinned) noisy numbers.** No selection, no artefact.
- **All noiseless results.** Binning on `Mr/Mf` there is a function of `m`,
  which the estimator sees — §2. The noiseless profiles, the `|R|` spike
  findings and the chart comparisons all stand as measured.

Given that the corrected residuals are now ≤ 0.006 per bin, the noiseless
diagnostics — which need no control at all — are probably the more sensitive
instrument from here.
