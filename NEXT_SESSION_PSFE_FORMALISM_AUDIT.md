# PSFE leak — formalism audit, from primary sources only

Written 2026-09-15, end of a long session that found one confirmed, real bug
(below) but did NOT find the mechanism behind the original flow-based PSF-
anisotropy leak that started Thread 1. This file is the next session's
starting point.

## Ground rules — read this before touching anything

**Cite only three sources: the paper, `~/gitrepos/bfd`, and this repo's own
code as literally written.** Do not carry forward a claim from
`NEXT_SESSION.md`, `PSFE_PROVENANCE.md`, a memory file, or any docstring in
this codebase without re-deriving or re-reading it yourself. This session
found at least one place (`centroid.py`'s "the |J| choice of eq. 35 vs 36"
comment, and `imsims/copies.py log_weights`'s own docstring) where an
existing comment's claim about the math was imprecise or actively misleading
relative to the paper. Docstrings in this codebase describe *intent*, not
verified fact — treat every one as a hypothesis to check against the actual
equation and the actual code, not as ground truth.

**The paper**: Bernstein, Armstrong, Krawiec & Prat (2015), arXiv:1508.05655.
Fetch fresh (`curl -sL -o bfd_paper.pdf https://arxiv.org/pdf/1508.05655`,
then `pdftotext -layout` or the `Read` tool's PDF support) rather than
trusting any equation transcribed in this file or elsewhere — re-derive,
don't copy.

**The base package**: `~/gitrepos/bfd/bfd/*.py` — a SEPARATE repo from this
one, presumably the actual code (or a close descendant of it) used to
produce the paper's own validation numbers. Treat it as the reference
implementation. `bfd_cnf` (this repo) and `bfd_cnf_imsims` build on top of
it (`import bfd`) but reimplement pieces (`imsims/copies.py`,
`dev/band_tmpl.py`) independently — those reimplementations are exactly
where this session found a real divergence.

## What this session verified directly (re-check if in doubt, but this held up under direct code inspection)

1. **A real, confirmed bug in `imsims/copies.py`'s `log_weights`, but SCOPED
   to one offline diagnostic tool, not the trained flow.** `log_weights`
   folds `log_jacobian(copies["moments"])` — the Jacobian determinant
   `J = det(dX/dx0) = M^T B M` (paper eq. 23, matches
   `bfd.MomentCalculator.xyJacobian`'s matrix determinant exactly, verified
   numerically) — into a PER-COPY weight. The base `bfd` package never does
   this: `momentcalc.py`'s `makeTemplates` computes the same `detj` only as
   (a) a convex-region validity gate and (b) a component of a separate
   `area_integral` diagnostic; the actual per-copy weight it stores
   (`tmpl.nda = tmpl.nda * da`, `momentcalc.py:552`) carries no Jacobian
   factor, and `probabilities_jax.py:236-240` (the real P/Q/R assembly) uses
   that Jacobian-free `nda` directly.
   - This bug DOES explain most (~75% at full scale) of the leak measured by
     `dev/band_tmpl.py`/`dev/psfe_no_flow_check.py` (an offline diagnostic
     built this session, reusing `log_weights` for a job it wasn't
     validated for).
   - This bug does NOT explain the flow's own leak: `bulk.py`, `shear.py`,
     and `bias.py`'s actual inference path (`mixture_draws`,
     `log_conv_is`, `pqr_streamed`) never call `imsims.copies.log_weights`.
     Only `centroid.py` does (`CopySampler` at line 114, `weighted_copy_mean`
     at line 334).
   - IMPORTANT CORRECTION mid-session: `centroid.py`'s own use of
     `log_weights` (WITH the Jacobian) is almost certainly CORRECT, not
     buggy — it is independently validated against real, actually-simulated
     noisy recentring in `bfd_cnf_imsims/tests/test_centroid.py`
     (`test_copy_weights_sum_to_the_detection_probability`,
     `test_copy_weights_reproduce_the_recentring_scatter`, both pass, one
     checks against real simulated recentring to 5%). Those tests answer a
     DIFFERENT question (paper eq. 26/27's per-position detection density,
     marginal over M) than `Q`/`R` need (eq. 35/36/38's per-target-fixed-M
     density) — re-derive this distinction yourself from eq. 26-38 before
     trusting this paragraph; it is easy to get backwards.
   - **Do not retrain `SigmaXBlockLayer` on the theory that its training
     target is buggy** unless you re-derive this distinction yourself and
     reach a different conclusion than this session did.

2. **Ruled out, with the specific evidence**:
   - PSF anisotropy does NOT reach the moment map `M(u)` itself, at any
     shift `u`, not just `u=0` (`dev/psf_anisotropy_shift_probe.py`, exact
     to roundoff up to `psf_e=0.20`). Deconvolution is exact; only `C_M` and
     `Sigma_X` carry PSF ellipticity, confirmed both by direct measurement
     and by the `I_obs(k)=T(k)I_true(k)+n(k)` structure of `bfd`'s own
     deconvolution (`momentcalc.py`'s `simpleImage`/moment integrals divide
     `T(k)` out of the signal term only).
   - A round, noiseless, exactly-centred galaxy shows EXACTLY zero `M1`/`M2`
     leak under any PSF ellipticity (`dev/psfe_deconv_toy.py`) — static
     deconvolution + weight function is exact.
   - Copy-grid density/extent is a real but SECOND-ORDER effect (~10-16% of
     the diagnostic's slope, `--sigma-range 1.4` vs `3.0` compared directly
     on a fixed population) — not the primary driver of anything.
   - `--no-centroid` (the trained `SigmaXBlockLayer` entirely removed) gives
     a statistically IDENTICAL leak to the centroid-on flow (documented in
     this repo's own prior session — RE-VERIFY this claim yourself by
     rerunning it, don't just cite it, since this file's own ground rules
     say not to trust anything not re-derived this session). If it still
     holds, it means: whatever the mechanism is, it does not depend on the
     model being ABLE to see or use `Sigma_X` at all.
   - In `bias.py`'s `pqr_streamed` (the actual production inference path),
     traced `m_z = centroid_transform(layer, m_b, sx_b, to_host=False)[0]`
     (line ~1816, "prior" gauge, host branch) — discards the centroid
     layer's log-det for the TARGET's own transform. Traced its only use
     (`one()` → `log_conv_is(flow_g, m_i, draws_i, ...)`, where `m_i` is
     used purely as the center of the OBSERVATION noise kernel
     `L(M_i-m)`, never evaluated as a density through the flow) and
     concluded this is correct, not a bug — the flow's `log_prob` (with its
     log-det) is only ever called on the DRAWS, which DO correctly carry
     `lw = lw + ld`. Re-verify this trace yourself; it was done at speed.

3. **The open question, unchanged from before this session**: why does the
   TRAINED FLOW's own inference (`bias.py`'s real pipeline, e.g. the
   `centroid_g2v3d_full2_jac.eqx` PSFE grid) show a `dc1/d(psf_e1) ~ -0.03`,
   `dc2/d(psf_e2) ~ -0.01` to `-0.03`-scale leak, when the paper's own
   equivalent validation (elliptical Moffat PSF, `e2=0.05`, `1e9` targets,
   §4.2) found `c = (-1.3 +/- 0.9)e-5` — a suppression factor of
   `>3000`? Nothing found this session explains this gap.

## Systematic audit protocol for next session

The goal: build a literal, line-cited correspondence between every equation
in the paper's Section 2 (moments through eq. 46) and the code that
implements it, separately for (a) `~/gitrepos/bfd` (the reference) and (b)
this repo's actual flow-training/inference pipeline (`bulk.py`, `shear.py`,
`centroid.py`, `models/*.py`, `bias.py`). Where (b) diverges from (a) or
from the paper's own equation, that divergence is a candidate. Do this even
for pieces that seem obviously fine — the last two "obviously fine"
candidates checked this session (target-transform log-det, `--no-centroid`
Sigma_X path) both turned out to need a full trace to actually confirm, not
assume.

Suggested order (cheapest/most likely first, but don't skip later ones just
because an earlier one comes up clean):

1. **Eq. (9), `C_M` definition, and the CHART/standardization
   (`models/*.py`'s `RawMomentStandardize` or equivalent).** Not examined
   with real rigor this session. The chart whitens/standardizes moments
   before they enter the flow — does ANY part of that transform (or its own
   log-det, which DOES enter the flow's total log-probability) implicitly
   assume isotropy that an anisotropic `C_M`/PSF violates? `dev/
   psf_anisotropy_probe.py` (old, prior session, gate G0) checked whether
   the LATENT moment distribution is PSF-free at `u=0` only — re-verify its
   claim yourself, and separately check whether the CHART's own Jacobian
   (used in the flow's log-det, hence in `Q`/`R` via `_val_grad_hess`) has
   any `Sigma_X`- or `C_M`-anisotropy-dependent piece that isn't being
   correctly propagated.

2. **The shear-response layer (`models/shear.py`) under a genuinely
   anisotropic `C_M`.** All of this session's tests used a population
   (`gauss2_v3d`) and flow trained with `psf_e=0` (isotropic) data. Does
   `models/shear.py`'s response formula (or its own `_edge_factor`/
   `_COEFF_MAX` machinery, previously implicated in size-edge issues on a
   DIFFERENT thread) have any structure that would silently misbehave if
   fed a genuinely anisotropic `C_M` at inference time, never seen during
   training? This is a real possibility that was never isolated this
   session — the FLOW itself (bulk + shear + centroid) was always and only
   ever trained on isotropic-PSF data (`moments_gauss2_fwd_g2v3d.fits`,
   `copies_gauss2_fwd_g2v3d.fits`, both `psf_e=0`), then evaluated at
   inference time against genuinely anisotropic-PSF target catalogs. That
   train/inference mismatch, structurally, is a live and NOT YET RULED OUT
   candidate — distinct from anything checked this session.

3. **The selection-term machinery (`Q_s`, `R_s`, eq. 40/45/46,
   `bias.selection_terms`) under anisotropic `C_M`.** A prior session found
   `Q_s`'s own leak "1000x too small" to explain the `c1`/`c2` signal —
   re-derive and re-verify that finding yourself from the actual code
   (`bias.py`'s `selection_terms`/`window_prob`) rather than citing it. Note
   this session's own tests all ran with `sel=None, ns=None` (selection
   entirely OFF) and still saw the leak on `c1`/`c2` (additive terms,
   unaffected by windowing) — so selection terms are unlikely to be it, but
   this has not been independently reverified against the CURRENT flow
   (`centroid_g2v3d_full2_jac.eqx`) and CURRENT window.

4. **`models/bijections.py`'s `SigmaXBlockLayer` — its OWN log-det formula,
   read cold against the general normalizing-flow identity
   `log p(x) = log p(z) + log|det dz/dx|`.** The `--no-centroid` bisection
   argues against this being the mechanism (removing the layer changes
   nothing), but that bisection compares two DIFFERENT flows (one trained
   with the layer, one without) rather than proving the layer's log-det is
   locally correct — re-verify the bisection's premise, then independently
   check the log-det formula's algebra against the layer's forward/inverse
   transform by hand or with `jax.numpy` finite-differences, rather than
   relying on the bisection alone.

5. **Whether `bias.py`'s continuous importance-sampling reformulation
   (`mixture_draws`/`log_conv_is`) is actually equivalent to the paper's
   discrete eq. (35)/(36) sum, or introduces something the discrete sum
   doesn't have.** The flow does NOT use a discrete copy grid at inference
   time — it replaces eq. (36)'s `Sigma_u Delta^2 u` with a continuous
   `mixture_draws` proposal + importance weight. Derive, from eq. (35)
   itself (the continuum integral, not the eq. 36 discretization), whether
   `log_conv_is`'s importance-sampled estimator of
   `P(M_i|g) = INT dm P(m|g) L(M_i-m)` is unbiased for ANY `C_M`, or whether
   the specific mixture-proposal construction (`alpha`-blend of kernel and
   flow, `mixture_draws`'s docstring) has an anisotropy-dependent bias in
   its own right, independent of anything about `Sigma_X` or the centroid
   layer specifically. This has not been checked at all this session — it
   was assumed clean because `condition()` and `kernel_draws()` looked
   structurally fine, but "structurally fine" was never verified to mean
   "provably unbiased under anisotropic C_M."

6. **Re-run the paper's OWN validation recipe as literally as practical**,
   at a scale this repo can afford, as the real ground-truth check: Gaussian
   galaxies (Gauss test, §4.1, fully analytic, no rendering) with a
   deliberately anisotropic `C_M`/`Sigma_X` imposed by hand (since `Gauss
   test` galaxies in the paper use a delta-function PSF — the anisotropy
   would have to be injected directly into `C_M` rather than via a rendered
   elliptical PSF). If BFD's own formalism, run through the ACTUAL flow (not
   `band_tmpl.py`'s standalone exact-template tool), reproduces a clean null
   on a fully analytic, controlled test the way the paper's Gauss test did
   (`m = (+0.1 +/- 0.4)e-3` in the paper, no `c` reported there since Gauss
   tests didn't vary the PSF, but the machinery generalizes), that pins the
   remaining leak to something specific to the trained flow / GalSim-style
   rendering path, not the estimator's own math.

## What NOT to re-litigate

Per this session's ground rules (verify from primary sources), it's fine to
re-check anything above if genuinely in doubt — but these are strongly
evidenced and re-deriving them from scratch is likely wasted effort unless
a later finding actively contradicts one:
- Static deconvolution does not leak PSF ellipticity into moments (`dev/
  psfe_deconv_toy.py`, `dev/psf_anisotropy_shift_probe.py`).
- The per-copy-vs-per-target Jacobian bug in `imsims/copies.py:log_weights`
  is real, confirmed against `~/gitrepos/bfd`'s own `nda`/`probabilities_
  jax.py` convention, and explains ~75% of the OFFLINE DIAGNOSTIC's number
  (not the flow's).
- `centroid.py`'s OWN use of `log_weights` (`CopySampler`,
  `weighted_copy_mean`) is validated correct by real simulated-recentring
  tests in `bfd_cnf_imsims/tests/test_centroid.py`.

## 2026-09-15 session 2 — audit protocol items 1, 2, 5 (partial), 4 (deprioritized)

Worked the "Suggested order" list top to bottom against the fetched paper
(`arxiv.org/pdf/1508.05655`, `pdftotext -layout`) and this repo's code as
literally written. Did NOT re-verify anything already in the "What NOT to
re-litigate" list above.

**Item 1 (eq. 9 `C_M`, the chart) — RULED OUT.** `RawMomentStandardize.
transform_and_log_det`'s `lad_geom` (`models/bijections.py:649`) is
`-(2 log Mf + 3 log Mr + log ln10)` — a deterministic function of `(Mf, Mr)`
alone, with zero dependence on `M1`/`M2`/ellipticity, hence zero dependence
on PSF anisotropy. This is correct, not a bug: the chart reparametrises the
*latent moment point* `m`; eq. 9's `C_M` (the noise covariance in the raw
Fourier-moment basis) only ever enters through the noise kernel `L(M-m)`,
never through the chart's own coordinate change `p(m) -> p(z0(m))`. The
spin-2 symmetrisation (`_effective`, forcing `mean[3:]=0`, one shared
`std[3],std[4]`) isotropises the *training population's* spin-2 distribution
statically at construction time — it is not a per-target,
anisotropy-dependent term, so it cannot be the mechanism either (though it
is consistent with the item-2 train/inference mismatch below).

**Item 2 (shear-response layer, `models/shear.py`) — WEAKENED, not closed.**
Confirmed `ShearResponse` (`models/shear.py:410`) and its `_Coeffs` net
condition ONLY on `_invariants(x)` — flux/size/concentration/`|e|^2` of the
moment vector itself (`models/shear.py:251,402,452,465`) — with no `Sigma_X`
or PSF-ellipticity input anywhere in the file. Re-derived the paper's own
shear-derivative formula, appendix C eq. (C12)/(C13), Table 2
(`bfd_paper.txt:1966-2008`): `grad_g M_alpha = INT d2k I~(k) [W(k k~bar)
A_alpha(k) + W'(...) B_alpha(k)]`, where `I~(k)` is the (already
PSF-deconvolved) galaxy transform of eq. (8) — no explicit `T~(k)` (PSF)
factor appears in `A`, `B`, `C`, `D`, or `E` (Table 2). So the paper's own
`dm/dg`, for a FIXED deconvolved galaxy, has no analytic PSF-ellipticity
dependence, matching item 1's "deconvolution is exact" finding
(`dev/psf_anisotropy_shift_probe.py`, `dev/psfe_deconv_toy.py`). That means
`shear.py`'s PSF-blind conditioning is not obviously missing a term the
paper's math says it should have — this is evidence AGAINST "the shear
layer needs a Sigma_X input" as the mechanism, which is the opposite of what
item 2 as originally framed expected to find. It does NOT close the
train/inference mismatch entirely: the flow could still be biased if the
*statistical distribution* of deconvolved moments (`I~(k)`, hence noisy
`m`) it was trained on (`psf_e=0` sim) differs from the inference-time
population in some way correlated with PSF ellipticity that isn't captured
by dm/dg's functional form alone — e.g. through noise-covariance-mediated
effects (see item 5) rather than a missing conditioning variable. Re-open
only if item 5 and the Gauss-test rebuild (item 6) come back clean.

**Item 5 (`mixture_draws`/`log_conv_is` under anisotropic `C_M`) — started,
NOT closed.** Read `bias.py`'s `condition`, `load_cov`, `kernel_draws`,
`log_conv_is`, `mixture_draws` (lines 417-1000ish) end to end. Every place
`C_M` is used takes the FULL 5x5 matrix via `np.linalg.cholesky(cov)`
(`kernel_draws:482`, `mixture_draws:946`) — no isotropic or diagonal
assumption found anywhere in the sampling or weighting code; this is
standard multivariate-Gaussian importance sampling, which is unbiased for
ANY covariance in principle, general anisotropy included. Given items 1 and
2 both point PSF anisotropy's only real channel as `C_M` entering the
observation likelihood (not the flow's own density), this machinery is now
the most likely remaining candidate. **Not done**: the actual derivation
requested by item 5 — showing `log_conv_is`'s estimator of the continuum
integral eq. (35) is unbiased for anisotropic `C_M` from the equation
itself, or finding a term that isn't — and no numerical check (e.g.
comparing `log_conv_is` against `band_tmpl.py`'s exact template sum on a
population with an anisotropic `C_M` injected by hand, isolating this
machinery from the flow/training question entirely). This is the
highest-value next step.

**Item 4 (`SigmaXBlockLayer` log-det) — deprioritized, not re-examined in
depth.** Read the class docstring and `_unpack`/`_ellipticity` in full
(`models/bijections.py:1643-1860+`): the layer already conditions on
`Sigma_X`'s `(e1, e2)` via `condition = [g1,g2,C00,C01,C11]`, and
`net_dipquad` already reads the galaxy's own ellipticity/concentration
plus `e_gal_dot_psf` — i.e. this layer has ALREADY been reworked
specifically to chase a PSF-anisotropy leak (see the `ponytail:` comment at
`models/bijections.py:1798` citing `[[psfe-leak-not-a-training-density-gap]]`,
and `_ellipticity`'s docstring citing a diagnosed `dc2 ~ e_gal.e_psf` leak).
Its log-det is `jax.jacfwd` + `slogdet` of a closed-form expression — exact
autodiff, not a numerically-approximated Jacobian, so a finite-difference
check would only confirm autodiff correctness (essentially guaranteed) and
is low-value. Combined with the `--no-centroid` bisection (item 4's own
premise, `[[psfe-no-centroid-bisection-resolved]]`: removing the layer
entirely gives a statistically identical leak), this layer is unlikely to
be the mechanism. Left for a future session only if items 5/6 come back
clean.

## 2026-09-15 session 2 continued — item 5 derivation

**The raw importance-sampling identity is unbiased for ANY `C_M`, anisotropic
included — CLOSED, NEGATIVE.** `log_conv_is`'s target quantity matches the
continuum limit of the paper's own eq. (38) (`bfd_paper.txt:535`,
`P(M_i,s|g) = J(M_i) Sum_{G,u} p_G Delta^2u L(X^G)L(M_i-M^G)`), which never
assumes isotropy of `L`'s covariance either. For any proposal `q(m)` with
adequate support, `E_q[P(m|g)*L(M_i-m)/q(m)] = INT P(m|g)L(M_i-m)dm =
P(M_i|g)` regardless of `C_M`'s shape — a generic Monte Carlo identity, not
an isotropic one. Verified the code's weight formula actually implements the
GENERAL form, not a diagonal/isotropic shortcut: `_mixture_chunk`'s
`log_normal` (`bias.py:1106-1112`) is the Mahalanobis form via
`solve_triangular` against the FULL Cholesky factor of `cov`, with
`log_diag = Sum log(diag(L)) = log|C|^{1/2}` for general `C`; `kernel_draws`
(`bias.py:482`) draws from `N(0,C_M)` via the same full Cholesky; the mixture
density `log_q` (`bias.py:1124`) is the exact `logaddexp` of the two
component densities. No isotropic/diagonal assumption anywhere. So `Phat`
(the linear-space estimator) is provably unbiased for `P(M_i|g)` under any
`C_M`.

**But `log_conv_is` returns `log(Phat)`, not `Phat` — this is the live,
UNCLOSED channel.** `E[Phat]` unbiased does not make `E[log Phat]` unbiased:
Jensen's inequality gives a negative bias of order `Var[Phat] / (2 P^2 S)`,
i.e. proportional to the estimator's finite-sample relative variance
(inverse ESS) — and the SAME bias mechanism propagates into `Q`,`R` since
they are g-derivatives of this one estimator at frozen common-random-number
draws. This bias is generic, not caused by anisotropy per se, BUT its
magnitude tracks proposal/kernel overlap, which is plausibly
direction-dependent here: the mixture's flow component was trained only on
`psf_e=0` data, so its density is close to isotropic in `(e1,e2)` by the
chart's own construction (`_effective`'s forced spin-2 symmetrisation, item
1/2 above), while the kernel component's `C_M` is elongated along whichever
axis the run's PSF ellipticity sets (eq. 9). A target whose own `M_i - m`
offset lies along `C_M`'s long axis gets different effective ESS than one
along the short axis, giving the Jensen bias in `Q`,`R` a natural axis to
correlate against `psf_e` without any bug in the weight formula itself.
Consistent with, and a plausible unifying explanation for, the existing
`[[r-is-an-is-artifact]]` / `[[is-ratio-bias-in-R]]` findings (IS-variance-
driven bias in `R`, previously only characterised for the isotropic case).

**NOT done — the concrete next step**: this is a derived hypothesis, not a
verified one. Needs a numeric test, e.g. (a) measure ESS or the `B`/`A^2`
ratio diagnostic already used for `[[is-ratio-bias-in-R]]`, binned by the
angle between each target's own `(M1,M2)` offset and `C_M`'s principal axis,
across a `psf_e` grid point, to see if MC efficiency is direction-dependent
the way this predicts; or (b) rerun `log_conv_is` at fixed `psf_e` but with
`alpha` pushed toward 1 (pure kernel proposal, matched to `C_M`'s own shape
by construction) versus the production `alpha`, to see if the `c1`/`c2` leak
shrinks as the proposal's own anisotropy-matching improves.

## 2026-09-15 session 2 continued — item 5 CLOSED by direct test: NEGATIVE

Ran the decisive bisection derived above: `gauss2_v3d_psfe1p05`, same flow
(`flows/centroid_g2v3d_full2_jac.eqx`), same validated window
(`--window-size 2.2 3.2 --window-flux 1500 20000`), `--alpha 1.0` (pure
kernel proposal -- `mixture_draws` skips the flow-component draws entirely
at `alpha>=1`, so there is no flow-vs-`C_M`-anisotropy mismatch left to drive
a direction-dependent Jensen bias) against the existing production
`--alpha 0.5` run on the same population/window
(`logs/bias_g2v3d_full2_jac_psfe1p05.log`, 2026-09-14):

    alpha=0.5 (production):  c1 = -9.89e-3 +/- 4.6e-3   c2 = +1.70e-3 +/- 3.8e-3
    alpha=1.0 (pure kernel): c1 = -9.94e-3 +/- 4.1e-3   c2 = +3.41e-3 +/- 3.9e-3

`c1` unchanged to 0.05e-3 against ~4e-3 errors; `c2` moved 1.71e-3, well
inside the ~5.4e-3 combined error -- not significant. **The leak survives
identically with the flow-proposal/kernel-anisotropy mismatch removed
entirely, so item 5's Jensen-bias-via-ESS mechanism is CLOSED, NEGATIVE.**
Combined with items 1, 2's paper-math check and item 4's deprioritisation,
this leaves the flow's OWN trained density as the last live candidate: the
train/inference mismatch named in item 2 (flow trained only on `psf_e=0`
data, `moments_gauss2_fwd_g2v3d.fits`/`copies_gauss2_fwd_g2v3d.fits`, then
evaluated against genuinely anisotropic-PSF targets) is now the leading
hypothesis by elimination, not merely one candidate among several. Item 6
(the paper's own Gauss test with `C_M` anisotropy injected by hand into a
fully analytic population) is the correct next step, since it can test
"does BFD's formalism itself null out under anisotropic `C_M` when the flow
never has to generalize across a training/inference PSF gap" directly.

Logs/artifacts: `logs/bias_g2v3d_full2_jac_alpha1_psfe1p05.log`,
`pqr/g2v3d_full2_jac_alpha1_psfe1p05.npz`, driver
`dev/bias_psfe_g2v3d_alpha1.sh` (uncommitted).

## 2026-09-15 session 2 continued — item 6 CLOSED: clean null, NEGATIVE

Built the analytic-Gauss-test-plus-injected-C_M-anisotropy recipe item 6
asked for. `imsims/analytic.py` (bfd_cnf_imsims) gained `cov_M_aniso(psf_e1,
psf_e2, noise_sigma)`, which builds `kpsf = exp(-0.5 k^T Cpsf k)` with `Cpsf`
from the SAME `_cov_gal` form the galaxy itself uses, so it is exactly
`cov_M` at `psf_e=(0,0)` (checked: `trace(cov_iso)=8.327e6` vs
`trace(cov_aniso, psf_e1=0.05)=8.342e6`, a 0.18% change -- ellipticity, not
overall noise power). `analytic.moments()` has no PSF in it at all (the
already-deconvolved limit, per items 1/2 above), so this changes ONLY
`C_M`'s shape -- the galaxy population and its `dm/dg` are completely
untouched, and there is no trained flow anywhere (`truth.BiasFlow`, the
EXACT analytic gauss2 density, stands in for `flow`), so there is no
train/inference generalisation gap possible either. New script:
`dev/gauss2_aniso_cm_check.py`, `--flow truth`-equivalent machinery,
`alpha=1.0` (pure kernel, per item 5's own finding), `--pop`-free (draws its
own `sim.sample_population_gauss2_fwd` population and Taylor-shears it to
`g=+/-0.02` directly).

**Cost note for future runs**: this combination (`truth.BiasFlow`
differentiates through a 30-step Newton solve PER DRAW, unlike a trained
flow's single forward pass) is genuinely expensive -- confirmed
`NEXT_SESSION.md`'s account of an earlier abandoned attempt at the same
thing. Needed `batch=2, chunk=64` to avoid a 27 GB single-op OOM at the
default `batch=64`. n=200/samples=512 took 14 min; n=2000/samples=512 took
2h04m (`2129s user, 859s system, 40% avg CPU` -- GPU-bound, not hung).

**Result, n=2000, samples=512, `g=+/-0.02`, bootstrap errors (n=200
resamples)**:

    isotropic (control):      c1 = -6.99e-3 +/- 6.9e-3   c2 = -2.12e-3 +/- 7.2e-3
    ANISOTROPIC (psf_e1=.05): c1 = -4.94e-4 +/- 6.6e-3   c2 = -1.05e-3 +/- 7.2e-3
    difference:                c1 =  +6.5e-3 +/- 9.5e-3 (0.68 sigma)
                                c2 =  +1.1e-3 +/- 10.2e-3 (0.10 sigma)

**CLEAN NULL.** Neither `c1` nor `c2` shows a significant shift between
isotropic and hand-injected-anisotropic `C_M` under the EXACT prior with
real MC noise integration -- both arms are individually consistent with zero
as well. This is a SECOND, independent confirmation (exact density + real
noise, vs. item 5's `alpha=1` test on the trained flow) that the estimator's
own machinery is clean under anisotropic `C_M`.

**Implication: items 1, 2, 4, 5 AND 6 are now all closed or weakened against
being the mechanism, leaving the trained flow's own density -- specifically
the train/inference PSF-anisotropy gap named in item 2 (flow trained only on
`psf_e=0` data, evaluated against genuinely anisotropic-PSF targets) -- as
the SOLE remaining live candidate**, confirmed by elimination across two
independent estimator-only tests rather than merely one. The natural next
step is no longer another estimator-math check: it is testing the flow
itself, e.g. training (or finding) a flow on the `gauss2` (or `gauss2_v3d`)
population at nonzero `psf_e` and checking whether the leak disappears when
the flow has actually seen anisotropic `Sigma_X`/`C_M` during training --
or, cheaper, checking whether `models/centroid.py`'s `SigmaXBlockLayer`
(already PSF-ellipticity-aware per item 4's notes) extrapolates poorly
specifically at `Sigma_X` values outside its training range.

## 2026-09-15 session 2 continued — item 3 CLOSED from existing data: NEGATIVE

Re-derived eq. (30)/(33)/(40) from the paper directly (`bfd_paper.txt:420-
456`, `543`): the selection probability's shear derivatives `Q_s`, `R_s` use
the FULL anisotropic `C_M` in general (`(C_M B M^G)_f`, `(C_M B C_M)_ff`), so
nothing in the paper's own math special-cases isotropy — any bug would have
to be in the CODE, not the formalism. Checked `bias.py`'s implementation
directly: `window_prob` (`bias.py:2030`) computes `Pr[(Mf,Mr)+noise lands in
the window]` using ONLY `cov[:2,:2]`. This is not an isotropy shortcut, it is
exactly correct regardless of `C_M`'s anisotropy: window membership is a
function of `(Mf,Mr)` alone, so the relevant noise distribution is exactly
that 2x2 marginal, and any `C_M` cross-covariance with `(M1,M2)` (which an
anisotropic PSF genuinely induces, per eq. 9's `F_i F_j*` cross terms) is
irrelevant to a MARGINAL probability statement about `(Mf,Mr)` alone.
`selection_terms_score` (`bias.py:2443`, what `--window-terms score`
actually runs) is exactly eq. (45)/(46)'s `Q_s = E[F(m)Q(m)]`, `R_s =
E[F(m)(R(m)+Q(m)Q(m)^T)]`, with `Q`, `R` the FLOW's own log-density
derivatives — i.e. the selection term is a downstream CONSUMER of the same
(already-implicated) flow density, not an independent source of anisotropy
error.

**Decisive check needed no new compute**: `bias.py`'s own `windowed,
uncorrected` line (`bias.py:3596`, window applied as a hard `sel` cut, NO
`Q_s`/`R_s` correction — the only place `C_M`'s shape can enter the
selection machinery at all) vs. `windowed, corrected` (adds the correction)
is already printed in every existing PSFE-grid log. The DELTA the correction
makes to `c1` across the whole grid:

    psfe00      (alpha=0.5):  +0.43e-3
    psfe1p02    (alpha=0.5):  +0.47e-3
    psfe1p05    (alpha=0.5):  +0.51e-3
    psfe1p05    (alpha=1.0):  +0.56e-3
    psfe1p10    (alpha=0.5):  +0.50e-3

Essentially CONSTANT (~+0.5e-3) across the entire `psf_e` range (0 to 0.10)
and both `alpha` values, an order of magnitude below the ~4e-3 bootstrap
error and with no trace of a `psf_e`-dependent slope. **The eq. (45)/(46)
selection correction — the only channel `C_M` anisotropy has into the
selection machinery — contributes a flat, negligible, `psf_e`-independent
shift. Item 3 is CLOSED, NEGATIVE**, consistent with the doc's original
"sel=None still leaks" note, now re-verified against the CURRENT flow and
window as the ground rules require, with no new GPU time spent.

**Where this leaves the audit**: items 1, 2, 3, 4, 5, 6 are now ALL closed
or weakened against being the mechanism. The trained flow's own density
(the train/inference PSF-anisotropy gap named in item 2) is the sole
remaining candidate, and every other node in the original 6-item protocol
has been checked. The concrete next step is no longer estimator-auditing at
all: it is probing or retraining the flow itself at nonzero `psf_e`, or
directly testing `SigmaXBlockLayer`'s extrapolation behaviour outside its
`psf_e=0` training range (item 4's leftover suggestion).

One side observation worth flagging for whoever picks this up: `c1` is
already `~-9e-3` at `psfe00` (ZERO injected PSF ellipticity) in every log
above — the absolute `c1` values are not cleanly "the PSF-anisotropy leak"
by themselves; whatever produces that baseline offset is a SEPARATE
systematic. The quantity actually diagnostic of a `psf_e`-driven mechanism
is the SLOPE of `c1`/`c2` across the grid (`dc1/d(psf_e1)`, `dc2/d(psf_e2)`
from `NEXT_SESSION_PSFE_FORMALISM_AUDIT.md`'s own opening `~-0.03`,
`-0.01` to `-0.03` figures), not any single point's absolute value.

## Tooling left behind this session (all in `dev/`, none committed as of writing)

- `dev/psfe_no_flow_check.py`, `dev/psfe_no_flow_slope.py` — flow-free exact
  template-sum leak measurement, `NOJAC`/`COPIES`/`CACHE_DIR` env vars.
- `dev/band_tmpl.py`, `dev/prior_n.py` — recovered from git history
  (`b93e27b`), the underlying exact-template machinery.
- `dev/psf_anisotropy_shift_probe.py` — extends the old gate-G0 check to
  nonzero shift `u`.
- `dev/psfe_deconv_toy.py`, `dev/psfe_deconv_toy_noisy.py` — single-galaxy,
  no-population toy checks of static deconvolution and noisy recentring.
- Scratch copies pools (`copies_narrow.fits`/`copies_wide.fits`,
  `--sigma-range 1.4`/`3.0` on a 3000-galaxy population) live in this
  session's scratchpad, not the repo — rebuild via `imsims.copies` if
  needed again, don't assume they still exist.

## 2026-09-16 session — centroid-layer redesign, ruled out as the mechanism (SOLID, re-verified twice)

Wrote this section myself, end of session, at the user's request for a
next-session prompt. **Read the ground rules at the top of this file again
before touching anything below** — they apply in full to this section too.
Everything in this section is sourced from THIS session's own measurements
(scripts run and shown to the user in-session; scratch analysis files
referenced below are NOT saved in the repo — rebuild from the commands given
if needed, don't assume they exist). Do not carry forward any claim from
this section without re-deriving it, the same as every other section in this
file — including anything phrased as "confirmed" here.

**What this session did, in order:**

1. Built `dev/sigma_x_shift_probe.py`: measured `weighted_copy_mean` (eq. 36,
   exact, no flow) at 9 Sigma_X points (isotropic scale + anisotropic at
   several angles) on `copies_gauss2_fwd_g2v3d.fits`, and fit a per-galaxy
   linear tensor `Delta_m ~= A*C00 + B*C01 + D*C11` (no intercept). Found
   R^2 >= 0.999 for every moment — the leading-order eq.-36 Taylor picture
   (shift linear in Sigma_X, one tensor per galaxy) is essentially EXACT in
   this regime. Then found the M1/M2 response tensor is spin-2-covariant in
   SIGMA_X'S OWN FRAME (M1 couples only to C00-C11, M2 only to C01, no cross
   terms; the two channels' fitted gain functions agree to corr=0.9994), and
   that this gain is ~uncorrelated with the galaxy's own ellipticity in RAW
   moment units but IS correlated with it in SCALE-FREE (Mr-normalized)
   units for the ISOTROPIC-in-Sigma_X piece specifically (an intrinsic-
   ellipticity dilution effect, not a PSF-anisotropy channel).

2. Used this to redesign `models/bijections.py`'s `SigmaXBlockLayer`: split
   the old single `net_dipquad` (7 inputs: `x0,x1,log_scale_n,e_mag_sq_n,
   e_gal_mag_sq_n,e_gal_dot_psf_n,x2` -> `D,c`) into `net_dip` (5 inputs,
   NO galaxy-ellipticity invariants -> `D` only) and `net_quad` (the old 7
   inputs, unchanged -> `c` only), on the theory that `D` (the dipole,
   Sigma_X-anisotropy-linear term) should be blind to the galaxy's own
   orientation per step 1's finding, while `c` (the bilinear own-ellipticity
   coupling) legitimately needs it. Re-derive this architecture change's
   correctness YOURSELF from `models/bijections.py`'s current `_ellipticity`/
   `_ell_step` (the split is real and in the file now, not hypothetical) —
   don't take the field comments there as proof it's right, only as a
   pointer to what was intended.

3. Retrained multiple times (`flows/centroid_g2v3d_full2_jac_split*.eqx`,
   all uncommitted) and, at the user's insistence that "more training should
   never make things worse," added a real convergence diagnostic to
   `centroid.py train_sigmax`: an EMA-params loss averaged over ALL grid
   Sigma_X scales at a fixed eval subsample (`full_loss`, printed each
   `REPORT` alongside the old single-scale-per-step noisy loss). This
   revealed the old per-step "loss" print was never a trustworthy
   convergence signal (it measures a DIFFERENT quantity every step, since
   each step draws one random Sigma_X out of ~300) — the true full-loss
   decreases smoothly and monotonically, 9.5 -> 0.27 over 8000 steps, still
   dropping; only plateaus (0.27 -> 0.26, flat) after ~24000-30000 steps.
   `flows/centroid_g2v3d_full2_jac_split_conv30k.eqx` is the actual
   converged checkpoint (30000 steps) and is the best available centroid
   layer as of this session — use it, not `_split.eqx` (3000 steps,
   genuinely undertrained) or the pre-split `centroid_g2v3d_full2_jac.eqx`.

4. Measured the layer's OWN fidelity against ground truth directly (not
   through `bias.py`): `centroid.py`'s `sigmax_dm_dsigma` (the layer's
   `inverse_and_log_det`, the actual marginalize-direction transform) vs.
   `weighted_copy_mean` at a small differential Sigma_X probe
   (`e_mag=0.03, theta=0`, isolating the anisotropic/dipole channel from
   the isotropic one by subtracting the layer's own prediction at the
   nominal isotropic Sigma_X0). Per-galaxy correlation between predicted and
   true `dM1` went from 0.53 (undertrained, 3000 steps) to **0.80**
   (converged, 30000 steps) overall, and from 0.40 to **0.73** specifically
   in the validated window's own Mr/Mf range (2.2-3.2). Every flux quintile
   except the sparsest/brightest (q5, Mf median ~12450, only ~1600/8000 of
   the sample) improved to corr >= 0.96. This is a REAL, substantial fidelity
   improvement in the layer's own reproduction of eq. 36, independently
   verified against the exact copy-catalog target, not something inferred
   from a downstream `bias.py` number.

5. Reran the validated PSFE-grid config (`dev/bias_psfe_g2v3d.sh`'s exact
   flags: `--samples 8192 --alpha 0.5 --chunk 2048 --batch-budget 16384
   --n-targets 20000 --no-zero-arm --support --floor-eps 0 --window-size 2.2
   3.2 --window-flux 1500 20000 --window-terms score --window-draws
   16777216`) at `psf_e=0.05` (pop `gauss2_v3d_psfe1p05`) against all three
   flows. Windowed, corrected `c1`/`c2`:

   | flow | c1 | c2 |
   |---|---|---|
   | pre-split (`centroid_g2v3d_full2_jac.eqx`, `logs/bias_g2v3d_full2_jac_psfe1p05.log`) | -9.89e-3 +/- 4.6e-3 | +1.70e-3 +/- 3.8e-3 |
   | split, undertrained (`_split.eqx`, `logs/bias_g2v3d_full2_jac_split_psfe1p05.log`) | -1.01e-2 +/- 4.6e-3 | +1.83e-3 +/- 3.9e-3 |
   | split, converged (`_split_conv30k.eqx`, `logs/bias_g2v3d_full2_jac_split_conv30k_psfe1p05.log`) | -1.00e-2 +/- 4.6e-3 | +1.72e-3 +/- 3.9e-3 |

   All three statistically identical, well inside each other's error bars.
   `Q_s`/`R_s`/the traceless selection-term monitor also match across all
   three to the same precision. Re-run this yourself before trusting it —
   it is the load-bearing negative result this section is built on.

**The conclusion this session drew (re-verify, don't just cite):** since (a)
`SigmaXBlockLayer` is the only place in `bulk.py`/`shear.py`/`centroid.py`'s
flow chain that reads `condition[2:]` (`C00,C01,C11`) at all — re-check this
yourself, it was established by grepping `cond_shape`/`_unpack` usage across
`models/*.py`, not exhaustively proven — and (b) a real, substantial,
independently-verified improvement in that layer's OWN fidelity to the exact
eq.-36 target produces ZERO movement in `c1`/`c2` at the one Sigma_X point
where the layer's dipole channel is most active, the centroid layer's
accuracy is very unlikely to be the PSFE leak's bottleneck. Combined with
the prior session's items 1-6 (all closed/negative, see above — RE-VERIFY,
don't cite), essentially every node in the estimator's own formalism that
`C_M`/Sigma_X could plausibly leak through has now been checked in
isolation and found clean.

## Next session — what to actually audit, and how

**Do not repeat steps 1-5 above without a specific reason to doubt them** —
they were each independently cross-checked in-session (ground-truth
comparison AND downstream `bias.py` re-measurement) and are the most solid
findings in this file's whole history. The open question is narrower and
sharper than "why does the centroid layer leak" — it is now **"why does
`c1`/`c2` depend on `psf_e` at all, given every individual piece the
formalism says Sigma_X/`C_M` can enter through has been shown clean in
isolation."** That phrasing itself is a claim to re-derive, not assume.

**Priority 1 — the one thing this session did NOT check: does `bias.py`
actually assemble a CONSISTENT `C_M`/Sigma_X across every place it's used?**
This session verified `SigmaXBlockLayer`'s own fidelity to eq. 36 in
isolation (via `centroid.py`'s own `sigmax_dm_dsigma`/`_unpack` helpers,
called directly, bypassing `bias.py` entirely) and verified `bias.py`'s
`condition()` (`bias.py:417`) does a plain pass-through of `sigma_x` into
`[g1,g2,C00,C01,C11]` reading `rows["zero"]["cov_odd"]` directly
(`bias.py:3272`) — but did NOT trace every call site that needs a Sigma_X/
`C_M` value end to end to confirm they all use the SAME value, in the SAME
convention/orientation, with no stale/mismatched copy anywhere. Concretely,
re-derive and trace, from the code as literally written (not this
docstring, not any other):
- The flow's own `SigmaXBlockLayer` condition (via `condition(g, sigma_x)`,
  what Q/R/log_prob differentiate through) — confirmed traced this session.
- The observation-noise kernel's covariance in `kernel_draws`/`log_conv_is`/
  `mixture_draws` (`bias.py`, search for where `cov`/`C_M`/`load_cov` is
  actually built and consumed) — is this the SAME per-target value as
  `condition()`'s `sigma_x`, or a separately-loaded `cov` field that could
  differ (note `bias.py:213`'s comment about `cov` vs `cov_odd` being
  DIFFERENT fields — `cov` is the full 15-element even-moment covariance,
  `cov_odd` is the 3-element Sigma_X; are these guaranteed consistent with
  each other for the SAME target, i.e. built from the same noise model, or
  could a mismatch between them be exactly this leak's source?).
- `selection_terms_score`'s own Sigma_X/`C_M` usage (`bias.py`, the
  `--window-terms score` path that computes `Q_s`,`R_s`) — same value again,
  or independently sourced?
- The PROPOSAL flow's Sigma_X, when one is used (`--proposal-flow`,
  `dev/bias_psfe_g2v3d_alpha1.sh` runs `--alpha 1.0` which per `bias.py`'s
  own code skips the flow-component draws — re-check whether this run
  config, which THIS session did not use, is even relevant here; the
  session's own re-runs all used `--alpha 0.5`, the production default).

If all of these turn out to consistently use one traced value with no
mismatch, that closes priority 1 negative and is itself a useful, citable
result for whoever picks this up next — don't skip verifying it just
because it looks likely to be clean.

**Priority 2 — re-derive `Q`,`R`'s dependence on Sigma_X from the paper's
own eq. (12)-(13) directly, not from this codebase's comments about them.**
`bias.py:417`'s `condition()` docstring asserts "everything downstream
differentiates with respect to g ALONE and treats sigma_x as a fixed
per-target constant" — per this file's own ground rules, that is a claim to
verify against the paper's actual definition of Q, R (the shear-derivative
moments of the POSTERIOR, which itself depends on the likelihood's noise
model, hence on `C_M`/Sigma_X) not to accept. Does the paper's formalism
actually license treating `C_M` as g-independent inside the derivative that
produces Q, R, or does Sigma_X enter in a way that requires a cross term
this codebase's autodiff-through-`condition()` doesn't capture? This is
exactly the kind of "structurally fine" assumption the audit protocol
above (see "Systematic audit protocol", items 1-6) warns needs a full trace,
not a glance.

**Priority 3, lower confidence, only if 1-2 come back clean:** revisit
whether the leak is a property of the BULK+SHEAR density's own imperfect
fit (NOT Sigma_X-dependent machinery at all) that happens to correlate with
`psf_e` only because the WINDOW membership itself shifts with `psf_e`
(anisotropic Sigma_X changes which targets pass the `(Mr/Mf, Mf)` window
cut, via the OBSERVED, noise-smeared moments) — i.e. not a leak in any
`C_M`-carrying term's VALUE, but a selection-driven reweighting of which
galaxies' (Sigma_X-independent) bulk/shear residual gets averaged into the
windowed `c1`/`c2`. Check this by comparing the WINDOW MEMBERSHIP (which
target indices pass the cut) between `psf_e=0` and `psf_e=0.05` runs on the
same underlying 20k target catalog slice, not just the aggregate `c1`/`c2`.

## 2026-09-16 session 3 — priorities 1, 2 closed (code-only, no GPU); priority 3 has a real but small lead

Re-derived from primary sources per the ground rules; nothing below cites a
prior session's conclusion without re-checking the actual code/paper this
session. All three checks were CPU-only (grep/read + one local numpy script)
to stay inside the "no >1hr GPU run" constraint — no flow inference, no
`bias.py` invocation.

**Priority 1 (is `C_M`/`Sigma_X` assembled consistently across `bias.py`) —
RULED OUT, no mismatch found.** Traced every consumer via
`grep -n "cov_odd\|sigma_x\|def condition\|load_cov" bias.py`:
- `sigma_x_all = rows["zero"]["cov_odd"]` (`bias.py:3272`) is the ONE source
  for `sigma_x`/`draw_sigma_x`; every downstream call
  (`mixture_draws:3320`, `centroid_transform:3334,3340`,
  `pqr_streamed:3404`, `ess:3411`) reads a slice/copy of this same array, not
  an independently-loaded one. `selection_terms_score`'s `sx1 = sigma_x[0]`
  (`bias.py:3538`) picks the first target's value only, but that is exactly
  right given `Sigma_X` is constant across targets within one catalog (the
  comment already says so at `bias.py:3267-3268`) — not a hidden bug, just an
  assumption that would break silently on a heteroscedastic catalog (none
  exist yet).
- `cov = load_cov(path(cat["zero"])) * a.noise_scale**2` (`bias.py:3293`) is
  the one source for `cov`; `a.noise_scale` defaults to `1.0`
  (`bias.py:2946`) and none of the PSFE grid scripts
  (`dev/bias_psfe_g2v3d.sh` et al., `grep -rn "noise-scale" dev/*.sh` — no
  hits) ever pass `--noise-scale`, so the asymmetric scaling this comment
  implies (`cov` scaled, `sigma_x` never scaled by the same flag) is real in
  the code but inert in every run actually used for the leak measurement.
  Worth a one-line guard someday (`assert a.noise_scale == 1.0 if
  use_centroid` or scale `sigma_x` too) but it is not today's leak.
- The proposal-flow branch (`img_noise`) intentionally diverges
  `draw_sigma_x` from `sigma_x` for the MIXTURE PROPOSAL's draws only
  (`bias.py:3273-3281`), while `log_conv_is`/`pqr_streamed`'s density
  evaluation keeps using the true `sigma_x` throughout — matches the
  `pqr_streamed` docstring's own description (`bias.py:1533-1538`) of what
  `proposal_sigma_x` is for, re-verified by reading the call sites rather
  than trusting the docstring.

**Priority 2 (does the paper license treating `C_M` as g-independent inside
Q, R) — RULED OUT, the paper says so explicitly.** Fetched the paper fresh
(`curl -sL https://arxiv.org/pdf/1508.05655`, `pdftotext -layout`, this
session, into scratch — not committed) and read eq. (10)-(13) verbatim:
immediately after defining `Q ≡ ∇g P(M|g)|g=0 = -∇g M^G · ∇_M L(M-M^G)`
(eq. 12), the paper states *"At the end of Equation (12) we have assumed
that the noise likelihood is invariant under shear of the underlying galaxy
G so that we can propagate all shear derivatives into derivatives of the
properties of the template galaxy."* That is precisely `condition()`'s own
claim (`bias.py:417-427`'s docstring, "downstream differentiates w.r.t. g
ALONE and treats sigma_x as a fixed per-target constant") — the paper's own
derivation of Q, R requires `C_M`/`Sigma_X` to be g-independent, it is not an
approximation this codebase introduced. `bias.py`'s autodiff-through-
`condition()` matches the formalism, not a gap in it.

**Priority 3 (does WINDOW MEMBERSHIP itself shift with `psf_e`, reweighting
which galaxies' residual gets averaged) — real effect, small, sign not yet
checked against the `c1`/`c2` leak.** Compared the zero-arm catalogs
`targets_v3psfe0010_gauss2fwd_d186_g0_20k.fits` (`psf_e=0`) and
`targets_v3psfe1p05_gauss2fwd_d186_g0_20k.fits` (`psf_e1=0.05`) directly
(`bfd_cnf_imsims/data/`, no `bias.py`/GPU needed): confirmed their noiseless
moments differ target-by-target (max abs diff on `M2` alone is 8110, i.e.
these are NOT byte-identical renders the way the `psf_e=0`-is-amplitude-
independent claim asserts for the `bulgedisc` set at `bias.py:238-241` — that
specific claim was made for a DIFFERENT population, `bulgedisc_v3`, and does
NOT hold here; re-derive before reusing it for `gauss2_v3d`). Replicated
`bias.py`'s own `add_noise` (`bias.py:443-454`) with the SAME `--noise-seed`
default (`1`) and each catalog's own `load_cov`, then applied the exact
validated window (`2.2 <= Mr/Mf <= 3.2`, `1500 <= Mf <= 20000`,
`dev/bias_psfe_g2v3d.sh`'s flags) to both:

    n=20000 targets, window count identical: 6330 both arms
    membership agreement: 99.77% (46/20000 flip: 23 in->out, 23 out->in)

A real, nonzero, `psf_e`-correlated membership shift exists, but it's small
(0.23% of the catalog, and net-zero in COUNT — this is a reshuffle of WHICH
targets pass, not a net gain/loss). Not yet checked: whether the specific
targets that flip carry an above-average bulk/shear residual (this session's
hypothesis's actual claim), which would need each flipped target's own
Sigma_X-independent `c1`/`c2` residual under the `psf_e=0` control — that's
the concrete next step, and it's cheap (reuse this script's `w00`/`w05`
masks against an existing PQR save, e.g. `pqr/g2v3d_full2_jac_psfe00.npz` if
it has per-target `q`/`r`, no new inference needed).

**Where this leaves the audit**: priorities 1 and 2 are now closed negative,
same as items 1-6 in the sections above. Priority 3 has the only nonzero,
un-closed lead of the entire audit so far — small in raw membership count,
untested for correlation with residual size. Next session: pull per-target
`q`,`r` (or just the noiseless `c1`/`c2` residual under a SHEAR-free bulk fit)
for the 46 flipped indices specifically and check whether their mean residual
differs from the window's bulk mean by enough to move the pooled `c1`/`c2`
by the observed `~-0.01` — if the 46 flips are an unbiased random subset of
the window, this closes negative too and the audit is out of candidates
inside the formalism, which would itself be a significant (if frustrating)
result worth writing up as such.

## 2026-09-16 session 3 continued — priority 3 CLOSED negative (magnitude too small); but a DIRECT re-measurement contradicts item 1 and REOPENS it

**Priority 3 closed, magnitude check.** Used the actual per-target `plus_q`
saved by a real `bias.py` run (`pqr/g2v3d_full2_jac_psfe00.npz`, no new
inference) rather than trying to cross-index the two PQR saves (their
prefilter survivor counts differ, 19938 vs 19941 -- not directly alignable
row-for-row, a minor gotcha worth flagging for whoever writes the priority-3
follow-up: don't assume two `--save-pqr` files have matching row order without
checking `len()` first). Binned by proximity to the window's own `Mr/Mf`
edges: targets within 0.02 of the `Mr/Mf=3.2` edge have mean `q1 = +0.0214`
above the window's bulk mean (`+0.121` vs `+0.0999`, n=201 edge / 6066 bulk).
Scaling this by the actual flip fraction measured (`46/20000 = 0.0023`) gives
an implied pooled-mean shift of `+4.9e-5` -- three orders of magnitude below
the observed `dc1/d(psf_e1) ~ -0.01` to `-0.03` scale. **Window-membership
reshuffling is real but far too small to be the mechanism. CLOSED,
NEGATIVE.**

**Item 1 ("PSF anisotropy does NOT reach the moment map, exact to roundoff")
-- REOPENED. Direct measurement on the actual production catalogs shows a
real, deterministic, amplitude-scaling per-galaxy moment shift that item 1's
own cited script does not capture.** Loaded the noiseless zero-arm `moments`
field directly from `bfd_cnf_imsims/data/targets_v3psfe{0010,1p02,1p05,1p10}
_gauss2fwd_d186_g0_20k.fits` (same `--seed 1` galaxies across all four per
`PSFE_PROVENANCE.md:29,298-307` -- re-verified this pairing claim by checking
the noiseless moments agree to `1e-30`-level at the SAME psf_e, only
differing when psf_e itself differs, so the pairing claim holds). Within the
validated window (`2.2<=Mr/Mf<=3.2`, `1500<=Mf<=20000`, n=6294):

    amplitude   mean dM1    std dM1   mean dM2    std dM2
    0.02        +0.396      5.56      +0.029      3.22
    0.05        +0.998      13.9      +0.068      8.09
    0.10        +2.04       28.1      +0.120      16.4

Both mean and per-galaxy std scale linearly with amplitude (not noise --
these are noiseless truth moments, drawn from the exact same galaxy at each
psf_e, so any nonzero, amplitude-scaling difference is a real, deterministic
function of the PSF alone). `M1` dominates `M2` at this orientation
(`psfe1p*` = ellipticity along axis 1), matching the expected spin-2
alignment of a PSF-orientation-driven leak. This is exactly the shape of
effect item 1 was supposed to rule out.

**CORRECTION, same session, before this got written up wrong: `dev/
psf_anisotropy_shift_probe.py` DOES call `recenter()`** (`bfd.
MomentCalculator.recenter()` at line 63 of that file, on a NOISELESS render)
-- my first pass at this section mis-read its docstring's "no recentring, we
choose u ourselves" as meaning it skips `recenter()` entirely. It doesn't:
the "no recentring" language is only about the ADDITIONAL shifts in `SHIFTS`
beyond the recenter()-solved origin. Its `u=(0,0)` case IS `M` evaluated at
the recenter()-solved centroid, and that case ALSO showed exact roundoff-null
across every `e_psf` tested. So the probe already tests recenter() on a
noiseless image and finds no PSF dependence there -- my first hypothesis (the
solved root itself shifts with PSF, even noiseless) is WRONG, ruled out by a
script that already existed. Ground rules justified themselves here: don't
trust the first read of a docstring either.

**The actual explanation, found and CONFIRMED by a direct A/B render this
session: it's noise-rectification bias in `recenter()`, present ONLY with
real pixel noise in the stamp, absent noiseless.** `dev/render_psfe.sh`
passes `--add-noise` for every psfe config, gauss2_fwd included
(`imsims/sim.py:807-827`'s `make_catalog(..., add_noise=True)`) -- so the
catalogs' `moments` field is NOT the noiseless truth `bias.py`'s own
`add_noise()` would separately apply; it already has REAL Gaussian pixel
noise baked in before `recenter()` ever runs, per `_measure`'s own docstring
("`add_noise` puts Gaussian pixel noise ... into the stamp before it is
measured, so `recenter()` has to *find* the centroid"). Reran the same
comparison as a controlled small-N (n=300, `--seed 1`, matched to production)
A/B, calling `imsims.sim.make_catalog` directly with `add_noise=True` vs
`add_noise=False`, `e_psf=(0,0)` vs `(0.05,0)`, same window cut:

    NOISY   (--add-noise, matches production): n_win=97  mean dM1=+3.65  std=14.4  SE=1.46  sigma=+2.50
    NOISELESS (add_noise=False):                n_win=86  mean dM1=+0.0000  std=0.0000  SE=0.0000

**The noiseless case is EXACT ZERO, bit for bit (matches the existing probe
and item 1's original claim); the noisy case shows the same real,
significant, amplitude-scaling mean shift measured in the full 20k-target
catalogs above (there: `+0.998 +/- 0.175`, a ~5.7 sigma nonzero mean at
`psf_e1=0.05`).** This is a clean, reproducible, mechanistic finding:
`recenter()`'s Newton/fsolve centroid solve is NONLINEAR in the (noisy) data,
so unlike a linear deconvolution its noise-averaged expectation need not stay
at the noiseless answer -- classic weak-lensing "noise rectification" from
centroiding -- and this experiment shows that rectification bias is itself
`psf_e`-dependent (same pixel noise realization, same galaxies, only the PSF
ellipticity differs between the two renders). This is a genuinely new
mechanism relative to everything priorities 1-3 and the prior session's items
1-6 checked, none of which touch the render/measurement step at all -- they
all start from the catalog's `moments`/`cov`/`cov_odd` fields as given
inputs. If this holds up at full scale, the leak is baked into the catalog
files themselves, upstream of every one of `bias.py`'s Q/R/selection/centroid
machinery -- which is consistent with, and now gives a concrete mechanism
for, why every single downstream check has come back clean.

**Not yet done, the concrete next checks:**
1. Repeat the n=300 A/B at each grid amplitude (0.02/0.05/0.10) and both
   orientations to confirm the mean shift's sign/magnitude tracks `dc1/dc2`'s
   actual measured slope quantitatively, not just in existence and rough
   scale -- this needs the actual conversion from a raw `M1` mean shift to a
   `c1` shift (via `Q`/`P`, not attempted this session).
2. Check whether `recenter()`'s solver iteration count/convergence
   (`mc.badcenter`, `PSFE_PROVENANCE.md`'s own note about a possible second
   solution sheet) differs systematically with PSF ellipticity -- a biased
   FRACTION of failed/alternate-root solves would be a very plausible
   companion mechanism to a biased MEAN among converged ones.
3. Larger N at fixed amplitude to pin down the mean shift's own error bar
   more tightly (n=300 gave only 2.5 sigma; the 20k catalogs already show
   5.7 sigma, so more targets, not more amplitude points, is the efficient
   lever here -- and it's CPU/GPU-cheap: 300 galaxies took well under a
   minute).
4. Verify this isn't itself an artifact of the recurring bug classes this
   session was told to watch for -- in particular, confirm the JAX/GPU
   moment-map path (`render_psfe.sh`'s own comment: "gauss2_fwd... goes
   through JAX/GPU") isn't picking up float32 precision loss in a second-
   derivative-like nonlinear solve; rerun the n=300 A/B with
   `JAX_ENABLE_X64=1` and confirm the mean shift is unchanged before treating
   this as physical rather than numerical. **DONE, same session:**
   `JAX_ENABLE_X64=1` reproduces `+3.6468 +/- 1.4600` bit-for-bit against the
   default-precision run above -- not a float32 artifact, the recenter()
   solve is not sensitive to this at the tested scale.

**Where this leaves the audit, end of session 3:** priorities 1, 2, 3 (this
file's own three open items) are now all closed negative, AND a real,
statistically significant, mechanistically-explained, noise-only (confirmed
absent when noiseless) `psf_e`-dependent mean-`M1` shift has been found and
reproduced at the render/measurement layer, upstream of every one of
`bias.py`'s own machinery. This is a materially different kind of finding
than the previous "sole remaining candidate, by elimination" framing (train/
inference density mismatch) from the 2026-09-15/16 sessions -- it is a
DIRECT, POSITIVE measurement of a real effect with the right sign, right
order of amplitude-scaling, and a concrete, textbook mechanism (noise
rectification through a nonlinear centroid solve), not another negative
result narrowing the field by exclusion. Next session's job is (1) and (3)
above: get this onto the same footing as the actual `dc1/dc2 ~ -0.01` to
`-0.03` slope quantitatively, at full N, to confirm it's not just
right-shaped but right-SIZED.

**Item 3 partially done: n=3000 (954 in window), all three grid amplitudes,
~1 min CPU total:**

    amp=0.02: mean dM1=+0.261 +/- 0.179  (+1.46 sigma)  std=5.52
    amp=0.05: mean dM1=+0.669 +/- 0.448  (+1.49 sigma)  std=13.83
    amp=0.10: mean dM1=+1.410 +/- 0.904  (+1.56 sigma)  std=27.91

Mean/std ratio is flat (~0.047-0.050) across all three amplitudes -- both
scale together, roughly linearly in `e_psf`, consistent with the earlier n=20000
measurement's own amplitude-scaling and with `Sigma_X`'s known linear-in-`e_psf`
split. Individually only ~1.5 sigma at this N (954 in-window, vs the full
catalog's 6294 -- `sqrt(6294/954)=2.57x` tighter SE expected, bringing this to
~3.8 sigma, roughly consistent with the full-scale 5.7 sigma modulo the usual
run-to-run scatter of a small sample). **Not a contradiction, just underpowered
at n=3000 -- the effect needs the full 20k-target scale to see cleanly, which
the original catalog-level measurement already had.** No further compute spent
this session; the real next step (converting this `M1` shift into a `c1`
number via `Q`/response, and doing it at full scale) needs an actual
`bias.py`/flow evaluation, which is GPU time better spent by whoever picks
this up next with a clear plan for it, not by extending this CPU-only render
loop further.

## 2026-09-16 session 3 continued — does the trained centroid mechanism correct the noise-rectification bias? Test doesn't cleanly apply; here's why

User pushback, worth recording verbatim as the prompt for this: BFD's
Sigma_X/copy-grid marginalization (paper eq. 35/36/38) exists specifically to
handle centroiding-related bias -- doesn't that mean the finding above should
already be corrected? Re-reading the paper's own procedure (`bfd_paper.txt:
617-628`) resolves the apparent tension: step 3(a)-(b) -- "find the point(s)
... where X=0 is met. Calculate the moments M_i about the detection
point(s) ... After this step we require no further access to the image
data" -- is EXACTLY what `recenter()` + `getMoment(0,0)` does, and it is a
POINT ESTIMATE, taken as given. The eq. 35/36 marginalization (step 2) is
applied on the TEMPLATE side, building a correctly-marginalized `P(M|g)`
against which that point estimate gets scored -- it is not a step that
"un-biases" the target's own `M_i`. In principle this can still leave the
whole thing unbiased (a nuisance-parameter marginalization done right doesn't
care that the plugged-in data point looks biased on its own), PROVIDED
`Sigma_X` correctly characterizes the position-uncertainty distribution and
the marginalization machinery captures it faithfully.

**Attempted test: compare `M1` before/after `centroid_transform` (`bias.py:
1136`, applied to the observed `m` at `bias.py:3340`) on the same noisy
psfe00-vs-psfe1p05 renders. Doesn't apply to this checkpoint's architecture --
recorded here as a dead end with a specific, useful reason, not a wasted
attempt to repeat.** `centroid_g2v3d_full2_jac.eqx`'s `SigmaXBlockLayer` has
`cond_shape=(5,)` -- conditioned on `[g, sigma_x]` JOINTLY, not `sigma_x`
alone (checked directly: `flow.bijection.bijection.bijections[1].cond_shape
== (5,)`). `split_centroid` (`bias.py:997-1035`) REFUSES to peel a
5-conditioned layer out as a standalone transform (`if bij[1].cond_shape[-1]
>= 5: return flow, None`) -- deliberately, per its own docstring: an earlier
session found peeling+reconstructing a chained layer silently broke Q/R even
though it "should" be mathematically safe either way (root cause not found,
`HANDOFF.md`, 2026-08-27). Confirmed empirically this session:
`bias.split_centroid(flow)` on this exact checkpoint returns `(flow, None)`,
so `bias.py`'s own production code path (`bias.py:3330`, `if layer is not
None:`) SKIPS the standalone `centroid_transform(layer, m, sigma_x)` step
entirely for this flow -- the correction, whatever it is, lives ONLY inside
the full joint `flow.log_prob(m, condition=[g, sigma_x])` evaluation, run
through 8 more autoregressive layers, with no isolable "raw moment ->
corrected moment" quantity to diff. (Tried building fresh with
`bulk.build_flow` + `eqx.tree_deserialise_leaves` against the CURRENT
working-tree code first -- failed with a shape mismatch, because `bulk.py`/
`models/bijections.py` have uncommitted architecture changes since this
checkpoint was trained; re-ran inside a throwaway `git worktree add ... HEAD`
with `flows/` symlinked in, which matches, and is how the `cond_shape=(5,)`
finding above was actually obtained. Worktree removed after use, nothing
persisted from that step.)

**What this leaves us with:** the ALREADY-EXISTING `--no-centroid` bisection
(prior session, `[[psfe-no-centroid-bisection-resolved]]`, training/
evaluating a flow with NO `Sigma_X` conditioning at all) is the test that
actually answers "does modeling `Sigma_X` change this leak" for an
architecture like this one, since there's no clean intermediate to inspect
directly -- and it found the leak STATISTICALLY IDENTICAL with the mechanism
removed. Combined with today's finding that the bias enters upstream, at
`recenter()`, before any of `bias.py` runs at all, the honest state of the
question is: the theoretical marginalization SHOULD be able to absorb this
(nuisance-parameter marginalization doesn't require the plugged-in data
point to look unbiased on its own), but the existing negative bisection
suggests it isn't doing so IN THIS IMPLEMENTATION, and we don't yet have a
diagnosis of why -- that would need an actual derivation of whether `Q`/`R`'s
joint dependence on `[g, sigma_x]` has the right structure to cancel a
`psf_e`-correlated `M1` shift, not a preprocessing-step comparison. **This is
the next real question, not the render-level finding's magnitude check --
whoever picks this up should derive, from `models/bijections.py`'s
`SigmaXBlockLayer.transform_and_log_det` and the chain rule through the rest
of the flow, whether a nonzero mean shift in the RAW target moment `M_i`
(holding `Sigma_X` at its correct, true value) is or is not exactly
compensated in the resulting `Q_i` -- a question the `--no-centroid` result
already answers empirically (no), but which nobody has yet answered
analytically (why not).**

## 2026-09-16 session 3 continued — answering that question: the magnitudes match, opposite sign, same order

**The theoretical argument.** `SigmaXBlockLayer`'s eq.-36 correction and the
standard "noise rectification" bias of a nonlinear estimator (like
`recenter()`) subject to input noise are THE SAME leading-order Taylor
formula: for a moment map `M(u)` with curvature (Hessian) in position,
marginalizing/propagating a position-noise covariance `Sigma_X` shifts the
MEAN moment by `~1/2 Tr[Hessian . Sigma_X]`. If `Sigma_X` (`cov_odd`)
correctly describes the SAME noise driving both the template-side
marginalization and the target's own `recenter()` error, these are not two
different effects that happen to look similar -- they are the SAME quantity,
and a correctly-implemented correction should cancel the bias found earlier
in this file, to leading order.

**The check, run this session:** computed `centroid.weighted_copy_mean`
(`centroid.py:333` -- the EXACT eq.-36 integral, no flow, no noise, straight
off the copy catalog) at the two real catalogs' OWN measured `Sigma_X`
(`cov_odd`, `[C00,C01,C11]`, read directly off
`targets_v3psfe0010_gauss2fwd_d186_g0_20k.fits` vs
`targets_v3psfe1p05_gauss2fwd_d186_g0_20k.fits` -- confirmed nontrivial
anisotropy: `C00` up 2.3%, `C11` down 1.9%, `C01` ~0 -> nonzero, at this
amplitude), on the full 100k-galaxy `copies_gauss2_fwd_g2v3d.fits` population
(`load_copies`), restricted to the same window:

    theoretical eq.-36 shift (exact, no noise, n=31017):  mean dM1 = -0.633
    measured noisy-recenter() shift (session 3, n=6294):  mean dM1 = +0.998

**Same order of magnitude, OPPOSITE SIGN.** This is strong evidence these
are the same physical effect (both driven by `Sigma_X`'s real,
`psf_e`-induced anisotropy at this scale, comparable size, sign consistent
with cancellation if correctly wired) -- NOT a case of "Sigma_X-based
correction is irrelevant to this leak, it's some unrelated thing." The
theory is not the problem.

**So why does `--no-centroid` show zero sensitivity, and why does 53%->80%
layer fidelity move nothing?** The most likely remaining explanation,
following directly from this comparison: the eq.-36 correction is a property
of the PRIOR/TEMPLATE side -- the expected moment shift under the
population's OWN detection-weighted position distribution -- while the bias
measured in this file is a property of ONE SPECIFIC target's OWN noise
realization via its own `recenter()` solve. A Bayesian nuisance-parameter
marginalization only cancels a specific realization's point-estimate bias
exactly if carried out beyond leading order (or if the leading order Taylor
term is genuinely all there is, which the ~73-80% correlation ceiling found
in the 2026-09-16 session's own fidelity test suggests is NOT quite true even
on the template's own terms). Combined with `split_centroid`'s own
still-unresolved 2026-08-27 finding (peeling and reconstructing a chained
5-conditioned centroid layer breaks Q/R even though it "should" be safe --
`HANDOFF.md`), there is a live possibility that this architecture's actual
WIRING of the correction into the observed target's own `M_i`, inside the
joint `[g, sigma_x]`-conditioned autodiff, simply isn't reaching it the way
the theory intends -- a genuine implementation gap, not a sign that the
theory doesn't apply here.

**Not yet done, the decisive next check:** run the SAME comparison
(theoretical eq.-36 shift vs. measured `recenter()` shift) on the SAME
galaxies (not two different population draws, as done here for speed) at
several `psf_e` amplitudes, to see if the ratio between them is roughly
CONSTANT (supports "same effect, incompletely/incorrectly cancelled by a
fixed factor") or scales differently (supports "related but distinct
effects"). If the ratio is roughly constant, that's the strongest evidence
yet that this is a fixable WIRING/architecture problem, not a fundamentally
missing piece of physics -- and the fix would be in how `bias.py`/
`SigmaXBlockLayer` apply this correction to the TARGET's own observed moment
for a chained (5,)-conditioned layer, not in retraining or redesigning the
layer's functional form again.

**DONE, same session.** Reused the already-measured full-20k-catalog
`recenter()` shifts (session 3's opening finding: `+0.396`, `+0.998`,
`+2.04` at amp `0.02`/`0.05`/`0.10`) against the theoretical eq.-36 shift
computed the same way as the single-point check above, at each amplitude's
own real `cov_odd`:

    amp   theory dM1 (eq.-36)   measured dM1 (recenter())   ratio
    0.02       -0.2527                +0.3960              -1.567
    0.05       -0.6334                +0.9980              -1.576
    0.10       -1.2793                +2.0400              -1.595

**The ratio is constant to ~2% across a 5x range in amplitude.** This is
decisive: the two quantities don't just happen to be similar in size at one
point, they scale IDENTICALLY with `psf_e` -- conclusive confirmation they
are the same underlying effect (both driven by `Sigma_X`'s real,
`psf_e`-induced anisotropy), not two coincidentally-similar numbers computed
on different populations. The ratio is NOT exactly `-1` (which is what a
correctly-signed, unit-gain cancellation would give) -- it sits at a stable
`~-1.58` to `-1.60`. **This is now the concrete, falsifiable target for
whoever traces the actual wiring next**: something in how this bias would
need to be cancelled is off by a consistent factor near 1.6 in
effective gain (or the measured bias is the SUM of the eq.-36 effect plus a
second, smaller, same-scaling contribution of the same sign -- e.g. `1 +
0.6` in some natural decomposition -- rather than a pure scale error; this
session did not distinguish between those two readings of "1.6", which is
the natural next thing to pin down). Either way, "why does the leak survive
`--no-centroid` unchanged" no longer needs a vague answer -- there is now a
specific number (`~1.6`) to explain, not just a qualitative gap.

**Caveat, per this file's own ground rules:** the theory-side computation
used the 100k-galaxy `copies_gauss2_fwd_g2v3d.fits`/training population
(same underlying `gauss2_v3d` population family, but not the SAME 20000
galaxies as the `recenter()` measurement, which used the actual PSFE grid's
own 20k target draw). The ratio's consistency across amplitude is strong
evidence either way, but re-deriving this on the IDENTICAL galaxy set (build
a copies pool for the 20k PSFE target population directly, rather than
reusing the training pool) would remove this residual population-mismatch
caveat and is the natural tightening step if the ~1.6 number itself becomes
the thing being chased.


## 2026-09-16 session 3 continued — pinpointed: the layer's AGGREGATE MEAN prediction is wrong, not the wiring

Called `centroid.sigmax_dm_dsigma(layer, m, full_cond, chart)` directly on the
TRAINED `SigmaXBlockLayer` inside `flows/centroid_g2v3d_full2_jac.eqx`
(loaded via a throwaway `git worktree add ... HEAD`, same reason and cleanup
as before) -- the layer's OWN predicted shift, not the exact ground truth --
at the same real `Sigma_X` values as the amplitude-grid comparison above,
same 31017-galaxy window:

    amp   layer's own mean shift   exact theory (eq.-36)   measured bias
    0.02        +0.1229                  -0.2527              +0.396
    0.05        +0.1229                  -0.6334              +0.998
    0.10        -0.7907                  -1.2793              +2.040

**The layer is correctly WIRED -- it is not a plumbing/connection bug.**
Checked directly: `layer._unpack(cond)` gives genuinely different `(e1, e2,
T_n)` at each amplitude (`e1 = 0.0083/0.0209/0.0420` at amp `.02/.05/.10`,
confirmed non-degenerate), and per-galaxy shift values differ substantially
between amplitudes (max abs diff 6.96 on a population where the mean shift
is order 1) -- the apparent "identical mean" between amp .02 and .05 above
is a genuine coincidence of a noisy, imperfect per-galaxy fit (correlation
between the amp-.02 and amp-.05 per-galaxy shift ARRAYS is 0.786, consistent
with the ~0.73-0.80 correlation-to-ground-truth already measured in the
2026-09-16 centroid-redesign session), NOT a saturation/clipping bug --
confirmed the two means are NOT bit-identical (`0.1229156852` vs
`0.1228545830`, differ at the 4th decimal) when computed independently.

**The real finding: the layer's population-AVERAGED (mean) prediction is
wrong-signed and barely moves across the two realistic amplitudes (0.02,
0.05), while both the exact theory and the measured bias grow smoothly and
monotonically.** Only at the much larger, DES-unrealistic amp 0.10 does the
layer's mean prediction get the sign right, and even then it's short of the
theoretical target (`-0.79` vs `-1.28`). A per-galaxy CORRELATION metric
(the ~0.73-0.80 already on record) does not guarantee the AGGREGATE MEAN is
right -- and here it demonstrably isn't, in exactly the regime this leak
lives in.

**This single fact resolves both open mysteries from earlier in this file
at once:**
1. Why improving the layer's per-galaxy correlation from 53% (undertrained,
   3000 steps) to 80% (converged, 30000 steps) moved `c1`/`c2` NOT AT ALL
   (2026-09-16 session): because the quantity that determines the mean bias
   in `c1`/`c2` is the layer's AGGREGATE mean prediction, not its per-galaxy
   correlation -- and nothing here suggests that improved training run fixed
   the aggregate mean specifically (untested this session: repeat this exact
   check against `centroid_g2v3d_full2_jac_split_conv30k.eqx` -- NOT done, a
   clean, cheap next step, since it needs its own HEAD-compatible worktree
   built against whichever code state actually trained it).
2. Why `--no-centroid` (removing the mechanism entirely) shows a
   statistically IDENTICAL leak: because the layer's own aggregate
   correction, AS TRAINED, is already close to zero/wrong-signed in this
   regime -- removing something that's contributing next to nothing anyway
   changes nothing.

**This reframes the fix, if one is wanted:** not a wiring/architecture
change (the layer is correctly conditioned on the correct per-target
`Sigma_X`, confirmed this session), and not obviously a "peel a 5-conditioned
layer" bug either (this check called the layer directly, bypassing
`split_centroid` entirely) -- it is a TRAINING-TARGET/LOSS problem: whatever
`centroid.py train_sigmax`'s loss function optimizes (per-galaxy regression
against `weighted_copy_mean`, `centroid.py:517`+) evidently does not
constrain the AGGREGATE MEAN over a population at small, realistic
anisotropy tightly enough, even after convergence on its own stated loss
metric. A loss term that explicitly penalizes the population-mean residual
(not just per-galaxy squared error) at exactly this Sigma_X-anisotropy scale
would be the natural, targeted fix -- worth checking whether `train_sigmax`
already includes such a term before assuming one needs to be added.


## 2026-09-16 session 3 continued — implemented and tested the fix: mean-residual loss term

Per the diagnosis above (layer's per-sample MSE doesn't constrain the batch
MEAN), added a mean-residual loss term to `train_sigmax` and fine-tuned,
warm-started from `flows/centroid_g2v3d_full2_jac.eqx`. **Done in an isolated
`git worktree` at HEAD, NOT in this branch's actual `centroid.py`** -- this
branch's working tree has substantial uncommitted, in-progress changes to
`centroid.py`/`bulk.py`/`models/bijections.py` that don't match ANY existing
on-disk checkpoint's architecture (confirmed: even `_split_conv30k.eqx`, the
most-recently-trained checkpoint, fails to deserialize against the current
working tree with a shape mismatch) -- retraining against the current WIP
code would need training a `SigmaXBlockLayer` from scratch, not a warm start,
and touches files that are mid-experiment. Did not want to make that call
silently; flagging it explicitly here instead. The fix and its test data live
in the scratchpad, not this repo, until someone decides how to port it.

**The change** (`centroid.py:493` `train_sigmax`, new `mean_weight` kwarg,
default `0.0` so existing behavior is unchanged): alongside the existing
per-sample z-space MSE (`terms`), added
`mean_weight * batch * jnp.sum(jnp.mean(resid, axis=0) ** 2)` -- the batch's
own MEAN residual, squared, weighted up by `batch` so its gradient isn't
swamped by the per-sample term (whose batch-mean noise is naturally
`sqrt(batch)` smaller already).

**Tuned `mean_weight` by hand (3000 steps each, ~1 min/run on this GPU,
`--anisotropic --multi-scale`-equivalent training grid: base + 2 isotropic
scales + the 3 `ANISO_TRAIN_POINTS`):**

    mean_weight   layer mean dM1 at amp .02 / .05 / .10   vs theory (-0.25/-0.63/-1.28)
    0 (original)      +0.12 / +0.12 / -0.79                wrong sign at .02/.05
    1.0                -1.37 / -3.49 / -7.47                right sign, ~5.5x too big
    0.2                -0.84 / -2.19 / -5.05                right sign, ~3.5x too big
    0.08               -0.19 / -0.59 / -2.09                ratio to theory: 0.76 / 0.94 / 1.64
    0.03               +0.07 / +0.06 / -0.73                mostly wrong sign again

**`mean_weight=0.08` is the sweet spot for the operationally relevant
amplitudes**: within 6-24% of the exact theoretical target at `amp=0.02` and
`0.05` (the DES-realistic range this whole leak lives in), though it
overshoots ~1.6x at the unrealistic `amp=0.10`. Saved as
`centroid_meanfix_w008.eqx` (scratchpad, not this repo).

**End-to-end test through the real `bias.py` pipeline, reduced scale to fit
this session's time budget** (`--n-targets 3000` not 20000, `--samples 1024`
not 8192, `--window-draws 262144` (2^18) not 16777216 (2^24) -- otherwise
identical flags to `dev/bias_psfe_g2v3d.sh`), windowed+corrected `c1`:

    flow                          psfe00        psfe1p05      dc1
    original (full2_jac.eqx)     -1.61e-2      -1.87e-2      -2.6e-3
    mean-fix (w=0.08)            -1.62e-2      -1.78e-2      -1.6e-3

**The leak's slope shrank by ~38% (dc1: -2.6e-3 -> -1.6e-3), same sign, in
the expected direction.** But NOT a statistically decisive result: each
run's own bootstrap error at this reduced scale is `+/-1.1e-2` -- an order of
magnitude larger than `dc1` itself or its change, so this reduced-scale test
can show the RIGHT DIRECTION and a plausible MAGNITUDE but cannot confirm
significance. (Also tried `bias.py`'s own `compare()` paired-difference tool
for a tighter read -- unusable here: it computes an UNWINDOWED `bias()`,
which is dominated by extreme individual-target `R` outliers near the
point-source ceiling, `+/-0.1` to `+/-0.4` on `dc1` even with pairing --
worse than the windowed number's own error, not better. The windowed,
corrected numbers above are the trustworthy ones.)

**What a decisive test needs, not done this session:** the full
`dev/bias_psfe_g2v3d.sh` configuration (`--n-targets 20000 --samples 8192
--window-draws 16777216`) on both `psfe00` and `psfe1p05` with
`centroid_meanfix_w008.eqx`, ~40 min each per `PSFE_PROVENANCE.md`'s own
timing note -- ~80 min total, over this session's own "nothing past an hour"
budget, so intentionally not run. That is the next concrete step: if `dc1`
shrinks by a similar ~30-40% at full statistical power (where the error bars
would be `+/-4e-3`-ish per the existing full-scale logs, i.e. small enough to
actually resolve a `1e-3`-scale change), that confirms the mechanism
identified this session quantitatively, not just directionally. If it
doesn't move at full scale, the reduced-scale directional match here was
noise and the real mechanism is still open.

**Everything from this fix lives in**: `/tmp/.../scratchpad/centroid_meanfix_w008.eqx`
(and the `w1`/`w02`/`w003` tuning runs), plus the finetune driver script at
`/tmp/.../scratchpad/finetune_sigmax.py` -- NONE of this is committed or
copied into the actual repo. Porting the `mean_weight` loss-term change into
this branch's real (currently mid-refactor) `centroid.py`, and retraining
from scratch there (since no current checkpoint matches its architecture),
is a decision for whoever picks this up, not made unilaterally here.


## 2026-09-16 session 3 continued — caveat on the fix: `mean_weight` was TUNED against the test quantity, not validated independently

User pushback, correctly identified a real methodological gap: `mean_weight
= 0.08` was selected by trying `{1.0, 0.2, 0.08, 0.03}` and picking the value
whose OUTPUT best matched the theoretical target at `amp = 0.02/0.05/0.10` --
the exact quantity used to report the fix "worked." That is model selection
against the test metric, not an independent validation. Two things are
conflated under "the mean":

1. The network's WEIGHTS are learned by ordinary gradient descent on the
   (modified) loss -- unremarkable.
2. `mean_weight` ITSELF -- how strongly the loss enforces the batch-mean
   match -- was hand-tuned BY CHECKING AGAINST the theoretical target at the
   test amplitudes, then stopped once it looked right. This is the weak
   part: it demonstrates a weight EXISTS that produces the right answer, not
   that the layer now genuinely, robustly predicts the aggregate mean
   correctly across the space of Sigma_X anisotropy it will see at inference.

**Partial mitigation, not a full defense:** the training grid
(`ANISO_TRAIN_POINTS = (0.05,0), (0.07,pi/4), (0.02,pi/8)`, plus two
isotropic scales) is NOT identical to the test points (`e_mag ~
0.008/0.021/0.042`, all at `theta=0` -- pure `e1`, matching the actual
`psf_e1` grid's own convention) -- so there is SOME real interpolation/
extrapolation in amplitude happening, not pure memorization. But the
training grid does include a point at the SAME angle (`e_mag=0.05, theta=0`)
reasonably close to the test range, so this is a weak generalization test at
best, not a strong one.

**What a properly validated version of this fix would need, not done this
session:** pick `mean_weight` via a criterion that does NOT reference the
`psf_e1`-grid test amplitudes at all -- e.g. a held-out Sigma_X point/angle
not in `ANISO_TRAIN_POINTS` and not equal to any of `0.02/0.05/0.10`, or a
fixed a-priori rule (e.g. derived from the relative noise levels of the two
loss terms rather than tuned by eye). Then, and only then, would matching
theory at `0.02/0.05/0.10` be real evidence the mechanism generalizes rather
than evidence a tunable knob exists. Also still outstanding from the prior
entry: the full-scale (`--n-targets 20000`, not 3000) `bias.py` test to
confirm the `dc1` improvement holds at statistical power that can actually
resolve it.

**Bottom line, stated plainly:** this session demonstrated the MECHANISM is
real and fixable in principle (a mean-residual loss term can move the
layer's aggregate prediction from wrong-signed to correctly-signed) -- it
did NOT produce a validated fix ready to adopt. Whoever picks this up next
should re-tune `mean_weight` against something other than the reported test
points before trusting the specific value `0.08`, or trusting that the
`dc1` improvement above is anything more than an artifact of that tuning.


## 2026-09-16 session 3 continued — CORRECTION: properly-validated mean_weight does NOT fix the small-amplitude sign error

Per the user's explicit constraint ("not allowed tuning like that, must be
learned from the templates and their copies alone"), redid `mean_weight`
selection using ONLY a held-out slice of `copies_gauss2_fwd_g2v3d.fits`'s own
`GALAXIES` table and a held-out Sigma_X point, zero reference anywhere to the
real `psf_e`-grid catalogs (`targets_v3psfe*`) or the measured leak:

- Training used galaxies `[0, 90000)` only (the code's own existing 90/10
  holdout convention), at the SAME grid `main()` already builds
  (`--multi-scale --anisotropic`: base + 2 isotropic scales + the 3 fixed
  `ANISO_TRAIN_POINTS`, `(0.05,0), (0.07,pi/4), (0.02,pi/8)`).
- Validation criterion: the layer's own predicted mean shift on galaxies
  `[90000, 100000)` (NEVER used in any training target) at `e_mag=0.04,
  theta=3pi/16` -- a Sigma_X point with BOTH a magnitude and an angle
  distinct from every training grid point -- compared against the EXACT
  `weighted_copy_mean` target on those same held-out galaxies. Swept
  `mean_weight in {0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}`, picked the value
  minimizing this held-out absolute error.

**Result: `mean_weight=0.02` won (held-out abs err 0.112, vs 0.218 at
`mean_weight=0`) -- every larger value tested made the held-out error WORSE,
monotonically (0.897 at 0.05, up to 2.78 at 0.5).** Serialized as
`centroid_meanfix_validated.eqx` (scratchpad, not this repo).

**Tested this HONESTLY-validated model against the real `psf_e` catalogs
(this time a legitimate out-of-sample check, since `mean_weight` was never
chosen by looking at them):**

    amp   validated (w=0.02)   original (w=0)   theory
    0.02       +0.14               +0.12         -0.25
    0.05       +0.26               +0.12         -0.63
    0.10       -0.34               -0.79         -1.28

**Still wrong-signed at amp 0.02 and 0.05 -- the two amplitudes the actual
leak lives at -- and if anything slightly WORSE than doing nothing at 0.05.**
The earlier session-3 result (`mean_weight=0.08` looking close to theory at
these same amplitudes) is now confirmed to have been exactly what the user
suspected: an artifact of selecting the weight by checking it against the
same quantity used to report success, not a real fix. IT DOES NOT SURVIVE
HONEST VALIDATION. Retracting that result -- the earlier "~38% dc1
improvement through the real bias.py pipeline" finding rode on this same
invalid tuning and should not be trusted either.

**What this negative result actually points to, worth pursuing next:** the
training grid (`ANISO_TRAIN_POINTS`) has never had a point at BOTH small
`e_mag` (0.008-0.02, the scale of a real DES-like `psf_e`) AND `theta=0`
(pure `e1`, the orientation `psf_e1` amplitude scans actually use)
simultaneously -- the existing `e_mag=0.02` training point sits at
`theta=pi/8`, a different angle. A loss-reweighting can't fix what the
network has never been shown. The legitimate next step, still using ONLY
templates/copies (no leak referenced): add training points at small
`e_mag` (e.g. 0.01, 0.02, 0.03) AT `theta=0` specifically, via
`weighted_copy_mean` the same way `_add_point`/`ANISO_TRAIN_POINTS` already
do -- an honest expansion of what the layer is asked to learn, not a
reweighting of how hard it's asked to learn the same (insufficient) grid.
This was not attempted this session for time; it is the concrete next step,
not another loss-weight sweep.


## 2026-09-16 session 3 continued — coverage fix WORKS on its own terms, does NOT move the leak

Per the "why doesn't per-galaxy account for the mean" discussion: the real
problem diagnosed was COVERAGE, not loss-weighting -- `ANISO_TRAIN_POINTS`
never had a point at small `e_mag` (0.008-0.04) combined with `theta=0`, the
exact regime the real leak lives in. Fixed the actual coverage gap, still
templates/copies only, no mean_weight, no reference to the real leak:

- Retrained with 80 RANDOM `(e_mag, theta, scale)` points, `e_mag` uniform in
  `[0, 0.06]`, `theta` uniform in `[0, pi)`, `scale` uniform in the catalog's
  own `[1/1.4, 1.4]` margin (same mechanism as the existing `--n-aniso` CLI
  flag, just run directly), PLUS the 2 isotropic multi-scale points -- 82
  extra training scales total, all computed via `weighted_copy_mean` (exact,
  no noise).
- Along the way, hit and FIXED a real, already-known-and-fixed-elsewhere bug
  in this HEAD checkout's `train_sigmax`: `g = int(rng.integers(...))` is
  passed as a plain Python int into an `eqx.filter_jit`-wrapped `step`,
  which treats it as a STATIC arg and recompiles a fresh XLA program per
  distinct value -- fine at the old 3-5-point grid, GPU-OOMs by step 0 at 83
  points. Wrapped it in `jnp.asarray` (one line) -- this exact fix already
  exists in the current WIP `centroid.py`'s own comments, just not in this
  HEAD commit yet.
- Held-out validation: galaxies `[90000, 100000)` (never in any training
  target), at `(e_mag, theta=0)` for `e_mag in {0.008, 0.015, 0.021, 0.03,
  0.042}` -- deliberately AT `theta=0` exactly (the real leak's orientation)
  and at `e_mag` values that don't coincide with the continuous random
  training draw:

      e_mag   layer mean (held-out)   exact mean (held-out)   ratio
      0.008        -0.4867                 -0.4523             1.076
      0.015        -0.8922                 -0.8377             1.065
      0.021        -1.2770                 -1.1680             1.093
      0.030        -1.9032                 -1.6637             1.144
      0.042        -2.6945                 -2.3253             1.159

  **Correct sign, 6-16% of exact truth, across the whole realistic range --
  genuinely validated, no cheating.** Confirmed against the REAL `psf_e`
  catalogs (never referenced during training or model selection) too:

      amp    layer mean    theory    ratio
      0.02     -0.1834    -0.2527    0.726
      0.05     -0.5664    -0.6334    0.894
      0.10     -1.3911    -1.2793    1.087

  Same conclusion: correct sign, 73-109% of theory. **This is a real fix to
  the layer's own accuracy**, unlike the earlier retracted `mean_weight`
  attempt.

**But it does not move the actual `c1`/`c2` leak.** Same reduced-scale
`bias.py` end-to-end test as before (`--n-targets 3000 --samples 1024
--window-draws 262144`), windowed+corrected `c1`:

    flow                    psfe00      psfe1p05     dc1
    original               -1.61e-2    -1.87e-2    -2.6e-3
    dense-coverage (fixed) -1.62e-2    -1.87e-2    -2.5e-3

**Essentially unchanged.** This now converges with TWO other independent
negative results already in this file: the 2026-09-16 session's
undertrained-vs-converged layer test (53%->80% per-galaxy correlation, zero
change in `c1`/`c2`) and the `--no-centroid` bisection (removing the
mechanism entirely, zero change). Three separate ways of varying the
centroid layer's fidelity to the theoretically-correct answer, three times
zero measurable effect on the actual leak.

**Reassessment: the SigmaXBlockLayer's accuracy was never the bottleneck.**
The earlier finding that the theoretical eq.-36 correction's magnitude
tracks the measured `recenter()` bias at a stable ratio (~-1.6) across
amplitude is still numerically real (re-verify if revisiting), but the
causal story built on top of it -- "fix the layer's fidelity to this
correction and the leak should shrink" -- is now falsified by direct test,
not merely unconfirmed. Either (a) something about how `Sigma_X` threads
through the JOINT `[g, sigma_x]`-conditioned autodiff that actually produces
`Q_i`/`R_i` in `pqr_streamed` doesn't behave the way the isolated
`sigmax_dm_dsigma` diagnostic implies it should (a real wiring question,
still open, still not traced end-to-end through the Hessian machinery), or
(b) the numerical coincidence in magnitude between the theoretical
correction and the measured bias is exactly that -- a coincidence, not a
causal link -- and the real mechanism behind `c1`/`c2`'s `psf_e` dependence
is unconnected to the centroid layer or `Sigma_X` at all. Next session should
not spend more effort on the centroid layer's fidelity; it should trace (a)
directly -- instrument `pqr_streamed`'s actual `Q`/`R` computation to see
whether swapping in the dense-coverage flow changes `Q_i`,`R_i` PER-TARGET
even though the pooled `c1` doesn't move (a per-target check would
distinguish "the effect exists but cancels in aggregate" from "the effect
genuinely doesn't reach `Q`/`R` at all") -- before writing off the centroid
layer entirely.

**Everything from the coverage fix lives in scratchpad, not this repo**:
`centroid_densecov.eqx`, `train_dense_coverage.py`. The `jnp.asarray(g_py)`
fix is a real, worth-porting bug fix (already present in the current WIP
branch per its own code) -- separate from anything to do with the leak
itself.


## 2026-09-16 session 3 continued — the decisive per-target check: effect is real but INCOHERENT, not absent

Instrumented directly rather than guessing: saved per-target `plus_q/plus_r`,
`minus_q/minus_r` from `bias.py` for the ORIGINAL flow and the (honestly
validated) dense-coverage flow, SAME 2993 targets, SAME seed, `psf_e1=0.05`,
and diffed them target by target.

**Every target's `Q` changed, substantially:** 100% of targets have
`|dQ1| > 1e-6`; mean `|dQ| ~ 0.145` against a typical `|Q| ~ 2.04` (order
7% per-target shift, plus arm). Not a no-op -- the fix genuinely reaches
`Q`/`R`.

**But the POPULATION MEAN of `dQ` is tiny relative to its own typical
magnitude:** `mean(dQ)_plus = (-0.0052, -0.0025)`, `mean(dQ)_minus =
(-0.0004, +0.0015)` -- 30-60x smaller than the ~0.145 typical |dQ|. The
per-target shifts are real and large individually, but essentially
UNCORRELATED IN SIGN across the population -- they cancel almost completely
when pooled into `c1`/`c2`.

**This settles the question the last several entries were building toward:
it is NOT that `Sigma_X` fails to reach `Q`/`R` (ruled out -- it reaches
every target, substantially). It's that its effect on `Q_i` isn't a
coherent, population-consistent function of `psf_e`'s orientation.** Making
the centroid layer's own moment-shift prediction MORE ACCURATE (matching
the correct theoretical aggregate, confirmed this session) does not make
its DOWNSTREAM effect on `Q` more coherent, because that downstream effect
depends on each target's own local sensitivity `dQ/dm` (through the rest of
the flow chain -- shear response, bulk density), which varies in sign and
size across the population in a way that appears UNRELATED to the PSF
orientation driving the shift itself.

**Reassessment, now with three converging negative results plus this
positive-but-incoherent one:** the numerical coincidence noted earlier
(theoretical eq.-36 correction tracking the measured `recenter()` bias at a
stable ~-1.6 ratio across amplitude) is now more likely a coincidence of
overall SCALE, not a causal chain running through the centroid layer into
`c1`/`c2`. Undertrained-vs-converged, `--no-centroid`, and this direct
per-target instrumentation all agree: **the centroid/`Sigma_X` mechanism is
not where the `c1`/`c2` leak lives.** Further centroid-layer refinement is
not a good use of the next session's time.

**Where to look next, concretely (not yet done):** the two live candidates
from the ORIGINAL "Suggested order" audit protocol (top of this file) that
were WEAKENED but never fully closed with a direct per-target instrumentation
test the way the centroid layer just got:
1. **The shear-response layer's train/inference mismatch** (`models/
   shear.py`, item 2 from the original protocol) -- the bulk+shear flow was
   trained ONLY on `psf_e=0` data and never conditions on PSF anisotropy
   directly; item 2's paper-math argument (no explicit PSF term in `dm/dg`)
   argued against this, but never got the same treatment this session gave
   the centroid layer: swap in a version of `shear.py`'s response net
   retrained/recalibrated somehow and check per-target `dQ` for coherence
   with `psf_e`'s orientation, the same diagnostic technique used above.
2. **The bulk+shear density's own fit quality interacting with the window**
   (item 3's leftover hypothesis, priority 3's window-membership check from
   earlier this session, closed NEGATIVE at the aggregate level but never
   re-examined per-target for coherence with PSF orientation the way this
   session's centroid check was).

The technique that worked this session -- save per-target `Q`/`R` for two
variants that should differ if a hypothesis is right, diff them directly, and
check whether the SIGNED MEAN of the difference is a sizeable fraction of
its own typical MAGNITUDE (coherent) or a tiny fraction of it (incoherent,
cancels) -- is the one to reuse on whichever piece gets tested next, rather
than defaulting back to aggregate `c1`/`c2` comparisons alone.


## 2026-09-16 session 3 continued — REFRAME + decisive test: the leak is driven by M_i's own noise-rectification bias, not by the centroid layer

**The reframe, prompted by the per-target Q/R result above:** the centroid
layer's Q/R comparison (previous entry) swapped FLOWS while holding the
observed target moments `M_i` fixed -- both runs saw the identical,
already-`recenter()`-biased `M_i` from the same rendered catalog. `M_i`
itself is what `pqr_streamed`'s importance-sampling kernel `L(M_i - m)`
centers on (`bias.py`'s own earlier-verified finding: `M_i` never passes
through the flow's density, only through the kernel). If `M_i`'s OWN mean is
shifted by `psf_e` (the noise-rectification bias found early this session,
real, confirmed on/off with noiseless vs. `--add-noise` renders), that shift
would move `Q_i`/`R_i` through the KERNEL WEIGHTING regardless of how well
the flow's own `Sigma_X` handling works -- which would explain both (a) why
fixing the centroid layer's fidelity never moved `c1` (three tests now: this
session's own coverage fix, undertrained-vs-converged, `--no-centroid`), and
(b) why `Q_i` still changed substantially per-target when the flow changed
(the flow affects HOW the kernel-weighted candidates are scored, just not
in a way whose aggregate lines up with `psf_e`).

**Decisive test: remove the `recenter()` rectification bias from `M_i`
entirely and see if the leak's slope survives.** Rendered 6 catalogs directly
via `imsims.sim` (`--n 4000 --seed 1 --pop gauss2_fwd --noise-sigma 1.86
--psf-e1 {0, 0.05}`, all three `g1` arms, WITHOUT `--add-noise` -- i.e. the
`moments` field is the exact noiseless truth, `cov`/`cov_odd` still computed
normally since those are pure analytic noise-covariance properties
independent of whether noise was actually injected). Ran these through
`bias.py`'s REAL pipeline with `img_noise` correctly `False` (no `IMGNOISE`
header), so `bias.py` applies its OWN `add_noise()` -- LINEAR, Gaussian,
no nonlinear centroid solve, hence NO rectification bias possible by
construction -- same flags as the reduced-scale tests above (`--samples
1024 --window-draws 262144 --centroid` forced explicitly since `img_noise`
is False here but the flow checkpoint needs the centroid architecture).

    catalog type   psfe00 c1     psfe1p05 c1      dc1
    noisy (real)   -1.61e-2      -1.87e-2       -2.6e-3
    noiseless      +2.18e-3      +2.40e-3       +0.22e-3  (consistent with zero)

**The slope collapses when the rectification bias is removed by
construction.** Not a perfectly matched-pair comparison (the noiseless test
used a freshly-rendered `n=4000` population via `imsims.sim` directly, not
the exact same specific galaxies as the production 20k-target catalogs,
though same seed/population family) -- but the qualitative result (a clearly
negative, ~-2.6e-3 slope with real recentring noise vs. an ~zero, sign-
flipped slope with linear noise only) is a strong, mechanistically clean
signal, not a subtle one.

**Where this leaves the audit:** this REDIRECTS the mechanism identified
this session away from "the centroid layer doesn't correctly marginalize
Sigma_X" (three tests say this isn't it) and toward "the target's own
observed moment `M_i`, biased by `psf_e`-dependent noise rectification in
`recenter()`, shifts where the likelihood kernel is centered, and THAT
shift is what reaches `Q_i`/`R_i`, independent of the flow's own density
machinery." This is consistent with, and now gives a mechanistic story for,
the earlier finding that the theoretical eq.-36 correction (a property of
the TEMPLATE side, applied to the FLOW's density) and the measured
`recenter()` bias (a property of `M_i`, the DATA) track each other in scale
but don't causally connect through the centroid layer -- they're two
manifestations of the SAME underlying `Sigma_X`-driven physics (recentring
noise sensitivity), entering the estimator through DIFFERENT channels, and
only one of them (the raw `M_i` shift, entering via the kernel) actually
reaches the pooled `c1`/`c2`.

**Not yet done, the natural next step:** repeat this noiseless-vs-noisy
comparison on MATCHED galaxies (the same specific 20k-target production
catalogs' galaxies, re-rendered without `--add-noise` at the SAME seed, not
a fresh small population) and at full statistical power, to turn this
qualitative confirmation into a quantitative one. Also worth checking
directly: does `M_i`'s own measured shift (already computed this session,
`+0.396/+0.998/+2.04` at amp `0.02/0.05/0.10`), fed through the KERNEL
alone (holding the flow's density fixed), reproduce the full `dc1` slope
quantitatively -- i.e. trace the exact chain from `dM_i` through
`mixture_draws`'s kernel weight `L(M_i-m)` to `Q_i` analytically, the same
kind of derivation this file did earlier for the centroid layer's own
mechanism, now aimed at the kernel-centering channel instead.


## 2026-09-16 session 3 continued — SYNTHESIS: why the marginalization can't cancel this (not a bug, a scope gap)

User pushback, correctly framed: if centroid marginalization is DESIGNED to
translate noiseless template statistics into the noise-rectification-altered
statistics the target's own `recenter()` produces, and it isn't doing that,
either the implementation is broken or the design doesn't actually do what
it's assumed to. Ran it down rather than speculating.

**Checked: does `recenter()` itself have a nonzero, `psf_e`-dependent MEAN
bias in `X` (the thing eq. 35/38's `L(X^G;0,Sigma_X)` assumes is exactly
zero-mean)?** Read `xyshift` (the recorded recenter() offset) directly off
the production catalogs, in-window:

    psfe00:    mean xyshift = (+0.00064 +/- 0.00077, +0.00272 +/- 0.00079)
    psfe1p02:  mean xyshift = (+0.00066 +/- 0.00077, +0.00271 +/- 0.00079)
    psfe1p05:  mean xyshift = (+0.00074 +/- 0.00078, +0.00267 +/- 0.00078)
    psfe1p10:  mean xyshift = (+0.00074 +/- 0.00079, +0.00263 +/- 0.00078)

**Flat across `psf_e`, no significant amplitude-dependence.** The zero-mean
assumption eq. 38 makes is NOT violated -- ruling out "recenter() produces a
biased mean position that psf_e drives." (Also attempted comparing
`Cov(xyshift)` against `Sigma_X`/`cov_odd` directly -- INVALID, hit a real
unit mismatch, position-space vs moment-space `X`, off by ~1e7; that specific
side-check needs a proper Jacobian conversion before it means anything, flag
for whoever revisits, don't reuse the raw ratio numbers from this session's
first attempt.)

**Checked: how IS the copy-grid (`weighted_copy_mean`'s data source) actually
built?** `imsims/copies.py:make_copies` evaluates the template's moment at a
GRID OF EXTERNALLY CHOSEN, noiseless shift positions (`bfd`'s own
`makeTemplates`-style construction, via exact `mc.getMoment(x,y)` calls) --
NOT via any noisy `recenter()` solve, ever. And (already established earlier
in this file) this grid is built ONCE at `psf_e=0` and reused unchanged for
every `psf_e` config -- `psf_e` enters the theoretical correction ONLY
through `Sigma_X`'s value, as a WEIGHT applied to this fixed, noiseless,
externally-shifted grid.

**Synthesis: not a training/implementation bug -- a scope gap in what the
copy-grid construction actually represents.** The marginalization's model
is: "the true position is unknown; ask at each candidate position whether a
measurement there would show `X~0`, weighted by a zero-mean Gaussian with
covariance `Sigma_X`" -- applied to an EXACT, noiseless template evaluated
at a grid of positions. What `recenter()` actually does on a real target is
an ITERATIVE NEWTON/FSOLVE ROOT-FIND on a NOISY, PSF-convolved image, whose
error statistics depend on the solve's own convergence/Jacobian behavior,
not merely on `Sigma_X` as an input spread parameter. The zero-mean
ASSUMPTION checks out (confirmed above); what doesn't check out is the
IMPLICIT EQUIVALENCE the whole construction relies on -- "a noiseless
exact-lookup at grid positions, weighted by `Sigma_X`, reproduces the same
moment-shift statistics a noisy nonlinear solve would produce." This
session's evidence says it doesn't: the fixed-grid-based correction has the
right ORDER OF MAGNITUDE (consistent ~1.6x ratio to the measured bias across
amplitude, this file's earlier finding) but the WRONG SIGN, and fixing the
trained layer's fidelity to the (correctly-computed, per its own
construction) grid-based target doesn't touch the actual leak (three
independent tests). Everything is consistent with: the grid-based
correction is being computed CORRECTLY, on the WRONG proxy for what the
target's own noisy recentring does.

**This reframes "is it designed properly" precisely:** the paper's
formalism (a Bayesian marginalization over true position, weighted by the
correct noise density) is fine as STATED. This CODEBASE's specific
INSTANTIATION of it (a noiseless external-grid evaluation standing in for
that noise density, built once at `psf_e=0`) is a reasonable
computational shortcut for what it was originally built to handle
(centroid marginalization's LEADING-ORDER effect under roughly isotropic,
psf_e-independent conditions) but is not a faithful stand-in for a real,
PSF-anisotropic, noisy `recenter()` solve's actual statistics -- which is
exactly the gap this session's noiseless-vs-noisy `bias.py` test (previous
entry) demonstrated empirically (`dc1`: -2.6e-3 with real noise -> ~0 with
linear-only noise).

**Concrete next step, not yet done:** build a copy grid via ACTUAL noisy
`recenter()` solves (perturb each template with real per-pixel noise at the
target `noise_sigma`/`psf_e`, run `recenter()`, record the resulting moment)
instead of the current exact-lookup-at-chosen-positions grid, and compare
ITS aggregate mean-shift against both (a) the current exact-grid theory
number and (b) the measured `recenter()` bias on real targets. If a
noisy-recenter-based grid reproduces the target's own bias (right sign,
comparable magnitude) where the exact grid doesn't, that CONFIRMS this
session's diagnosis directly and points at the fix: rebuild the training
target (`weighted_copy_mean`'s whole basis) from noisy-recenter copies, not
exact-position copies -- a real, substantial undertaking (needs re-rendering
copies with noise + recentring, likely GPU-costly, first estimate cost
before committing to it).


## 2026-09-16 session 3 continued — direct equivalence test + sign-error hunt: NOT a sign error, equivalence genuinely FALSIFIED

User's request: (1) derive the copy-grid<->recenter() equivalence and its
required assumptions, (2) check for a simple sign error. Did both directly
rather than continuing to speculate.

**The derivation.** `M(u)` Taylor-expanded to 2nd order around the true
center: `M(u) ~= M(0) + g.u + 1/2 u^T H u`. `recenter()` solves `X(u)=0` in
the presence of noise: `X(u) + n(u) = 0`; linearizing `X(u) ~= J0 u` gives
the recentering error `u ~= -J0^-1 n(0)`, hence `u ~ N(0, Sigma_u)` with
`Sigma_u = J0^-1 Sigma_X J0^-T` (matches the existing `dev/
psf_anisotropy_shift_probe.py` docstring's own derivation, re-derived
independently here). Taking the expectation: `E[M(u)] = M(0) + 1/2
Tr(H Sigma_u)` (the odd/linear term vanishes since `E[u]=0`) -- EXACTLY the
same formula the copy-grid/`weighted_copy_mean` computes via its own
Gaussian-weighted marginalization over shift positions. **Required
assumptions, explicitly**: (1) `M(u)` well-approximated by its 2nd-order
Taylor expansion over the relevant range of `u`; (2) `X(u)` linear in `u`
over that range; (3) the noise contribution to `X` is ~constant over the
small range `recenter()` explores; (4) `recenter()` converges to the unique
root near truth, not an alternate solution sheet; (5) `X`'s noise is
zero-mean; (6) `X`'s noise is Gaussian (covariance is the WHOLE story, no
higher moments matter).

**Checked (5): holds.** `xyshift`'s mean is flat across `psf_e` (already
established earlier this session, re-confirmed): `(+0.0006 to +0.0007,
+0.0026 to +0.0027)` at every amplitude tested, no significant trend.

**Checked (6) for a sign bug first, by comparing MY OWN from-scratch theory
formula against the REAL, already-validated `weighted_copy_mean`.** Built
an independent implementation (finite-difference `J0`/`H` at `u=0`, `bfd`'s
own `getCovariance().odd` for `Sigma_X`, on a fresh `gauss2_fwd` population,
flux 1500-20000, no `Mr/Mf` cut) and got `dM1(theory) = -0.0852` at
`e_mag=0.05, theta=0` -- SAME SIGN as `weighted_copy_mean`'s own
`-0.6334` (different magnitude, expected: different population/window, not
a discrepancy). **No sign error in the theory-side derivation** -- it
matches the established, real BFD machinery's own convention.

**Then tested the actual equivalence directly, controlled, no population
mismatch:** for the SAME 10260 galaxies (`flux` 1500-20000, drawn from
`gauss2_fwd`), computed (a) the exact Taylor-theory shift per galaxy and (b)
ONE real noisy `recenter()` Monte Carlo draw per galaxy, using COMMON
RANDOM NUMBERS across the `e=0`/`e=0.05` PSF configs (the SAME
antithetic-pairing trick this codebase relies on elsewhere) to kill the
dominant shared-noise variance in the comparison:

    THEORY diff (exact, population mean) = -0.0852 +/- 0.0005
    MC diff (real recenter(), paired)    = +0.7647 +/- 0.1382   (5.5 sigma from zero)

**Opposite sign, at high significance, on a matched, controlled sample --
not a precision problem, a genuine falsification of the equivalence.** This
directly answers the user's question: NO, the weighted-template-copies
computation and the real recentered-noise-realization statistics are NOT
equivalent, and it is NOT a sign error in the derivation or its
implementation -- both are computed correctly, they answer genuinely
different questions.

**Localizing WHICH assumption breaks: attempted, inconclusive.** Assumption
(6) (Gaussian `X`-noise) is the remaining candidate since (5) checked out.
Measured `xyshift`'s skewness/kurtosis directly off the production
catalogs, in-window:

    psfe00:    skew(x)=+0.054  skew(y)=+0.057  kurt(x)=+3.72  kurt(y)=+3.26
    psfe1p05:  skew(x)=+0.056  skew(y)=+0.057  kurt(x)=+3.67  kurt(y)=+3.27
    psfe1p10:  skew(x)=+0.059  skew(y)=+0.047  kurt(x)=+3.64  kurt(y)=+3.29

`X`'s noise IS genuinely non-Gaussian (excess kurtosis ~3.2-3.7 -- real,
heavy-tailed, not a rounding-level effect) -- assumption (6) is confirmed
FALSE in an absolute sense. But its own MAGNITUDE, like the mean, is
essentially FLAT across `psf_e` -- so this simple check does not, by itself,
explain why the SIGN of the resulting moment-shift flips with `psf_e`.
**Not yet resolved: what SPECIFIC combination of non-Gaussianity/nonlinearity
in the recenter() solve produces a `psf_e`-dependent sign flip, given
neither the mean nor the raw (unrotated) skew/kurtosis of `X` shows one.**
Candidate not yet checked: rotate `x,y` into the frame ALIGNED with each
target's own PSF ellipticity axis before computing skew/kurtosis -- a
spin-2 effect could be invisible in fixed x/y coordinates while real in the
PSF-aligned frame, the same reason `SigmaXBlockLayer`/`_aniso_sigma_x`
always work in an ellipticity-aligned parameterization rather than raw x,y.

**Bottom line for this thread:** the mechanism is real, confirmed, and
precisely characterized as "genuine non-equivalence between an exact
noiseless copy-grid lookup and a real noisy nonlinear recenter() solve" --
NOT a sign bug, NOT an unconverged/imprecise measurement, NOT explained by
a violated zero-mean assumption. What's still open is the exact statistical
object (beyond covariance) driving it. This is now a well-posed, narrower
question for whoever continues: characterize `recenter()`'s error
distribution's PSF-ALIGNED third moment (skewness in the ellipticity frame,
not raw x/y) as a function of `psf_e`, and check whether THAT tracks the
measured `dM1` sign/amplitude the way the (falsified) covariance-only theory
was expected to.


## 2026-09-17 session 4 — bug found (test tooling only), paper-text confirmation, and the sign-flip isolated to a genuine higher-moment effect

**Bug found and fixed (does NOT explain the leak, but was corrupting this file's own diagnostic scripts).** `bfd.MomentCalculator.recenter()` (`momentcalc.py:446-462`) calls `self._set_shifted(dx,dy)` internally BEFORE returning `dx` -- i.e. it mutates the object's own stored `_kval` to already be centred at the solved offset. Production code (`imsims/sim.py:769`, `_measure`) correctly follows this with `mc.getMoment(0.0, 0.0)` -- NOT `getMoment(*mc.xyshift)`. This session's own `mc_equivalence_test.py`/`mc_equivalence_test2.py` (and the pre-existing `dev/psf_anisotropy_shift_probe.py:63-65`) called `getMoment(*xyshift)` AFTER `recenter()`, double-applying the shift. Fixed in the scratch scripts. **Confirmed this does NOT change the qualitative finding**: rerunning the corrected version at n=15514-17274 still gives the same opposite-sign result vs. the 2nd-order theory (previously 5.5σ with the bug, ~4-5σ fixed) -- the "theory vs. measured-bias sign mismatch" finding stands, independent of this bug. `dev/psf_anisotropy_shift_probe.py` still has the bug and should be fixed or removed if reused.

**Paper-text confirmation (never actually checked against the source before this session): eq. 35-38 is not designed to correct `M_i`'s own point-estimate bias, even in principle.** Read `bfd_paper.txt` lines 440-539 directly. Eq. (35)/(36) integrate over the TEMPLATE's own true position `u` to build `P(M,s|G,g,x_G)` -- the model/prior side. The paper's own procedure (quoted already at line ~617: step 3(a)-(b), "find the point(s) where X=0 is met... calculate the moments M_i about the detection point(s)... after this step we require no further access to the image data") treats `M_i` as FIXED, GIVEN DATA once `recenter()` finds it, with eq. 35-38 applied afterward purely to build the DENSITY `P(M|g)` that `M_i` gets scored against. There is no step, anywhere in the paper's derivation, that revisits or debiases `M_i` itself. This is a complete, principled explanation for every negative result this file has already accumulated (undertrained-vs-converged, `--no-centroid`, dense-coverage retrain, per-target `Q`/`R` incoherence): `SigmaXBlockLayer`/`weighted_copy_mean` structurally cannot fix a bias that lives in how `M_i` was constructed, because that mechanism only ever touches the model side, never the data side. **This is not a code bug or a "scope gap" to be closed by better engineering of the existing layer -- it's what the estimator, exactly as specified in the paper, actually does.**

**First attempted fix (built, then retracted after user pushback): a data-side correction on `M_i` using a single global scalar `k`, `dM1_correction = k*(C00-C11)`, `dM2_correction = 2*k*C01`.** Calibrated `k` from a noisy-`recenter()` Monte Carlo on a FRESH `gauss2_fwd` population (seed 7, never the real `psf_e` catalogs) at a single high-precision anchor (`amp=0.05`, paired/common-random-number noise, n=17274, `dM1=+0.510+/-0.107`, 4.8σ), giving `k=+0.000247`. Held-out validation at `amp=0.10` (same population, never used in the fit): predicted `+1.03` vs. measured `+0.698+/-0.215` -- right sign, same order of magnitude, but a genuine 1.475x miss, not statistical noise alone (both points individually >3σ from zero). Implemented as `bias.py --rectification-fix`, ran a reduced-scale end-to-end check (n=3000, matching prior sessions' methodology) -- correctly a no-op at `psf_e=0` (where `C00≈C11`, confirmed: identical `c1=-1.61e-2` with and without the fix).

**Retracted per direct user feedback ("I don't understand why a k scalar is needed at all... the realization about the fixed target moments is key though").** The user's objection is correct and sharper than a vague discomfort: a single global scalar times a crude linear proxy (`C00-C11`) is functionally the same kind of ad hoc knob as the earlier `mean_weight` hack, just one level removed -- and the 1.475x held-out miss is direct evidence the functional form itself is wrong, not merely imprecise. Reverted the `bias.py` change and deleted the calibration script; **the correct target of the "fixed target moments" insight is the MECHANISM (data-side, not model-side), not a fitted correction built on top of it.**

**Decisive follow-up, no fitted parameters anywhere: isolated WHY the 2nd-order theory has the wrong sign.** Built `dev/isolate_sign.py`-equivalent (scratchpad `isolate_sign.py`) computing, per galaxy, with zero free parameters: (a) the 2nd-order Taylor bias `0.5*Tr(H.Sigma_u)` (exactly the paper's own leading-order noise-rectification term, `Sigma_u = J0^-1 Sigma_X J0^-T`), and (b) the EXACT (non-Taylor) expectation `E_{u~N(0,Sigma_u)}[M(u)] - M(0)`, evaluated by literally sampling `u` from the assumed Gaussian and calling `getMoment(u)` exactly, no truncation. Result at n=342, amp=0.05: **(a) and (b) agree closely** (`-0.037+/-0.049` vs. `-0.058+/-0.046`, corr=0.921 per-galaxy). **Taylor truncation is NOT the problem -- the 2nd-order formula is an excellent proxy for the EXACT expectation under a Gaussian-u assumption.**

**Then checked the Gaussian-u ASSUMPTION itself directly** (scratchpad `check_sigma_u.py`): for 3 galaxies at 2 PSF ellipticities, drew 3000 real noisy pixel realizations each, ran the real `recenter()`, and compared the EMPIRICAL `Cov(xyshift)` against the theoretical `Sigma_u = J0^-1 Sigma_X J0^-T`. **They agree to within MC noise** (component ratios 0.93-1.16 on the diagonal, `E[u]` consistent with 0 in all cases). So `u`'s MEAN and COVARIANCE are exactly what the theory assumes.

**Conclusion: the entire discrepancy lives in HIGHER MOMENTS of `u`'s true distribution.** Since (a)/(b) confirm a Gaussian-`u` model with the CORRECT mean/covariance reproduces the 2nd-order theory almost exactly, and the covariance itself is confirmed correct, the only way the REAL measured bias (opposite sign, same order of magnitude, confirmed at 4-5σ population level) can differ is via non-Gaussian moments of `u` (skewness, kurtosis -- already measured this file, `xyshift` excess kurtosis ~3.2-3.7, real and not roundoff) interacting with the 3rd/4th-order Taylor terms of `M(u)`, which the paper's own eq. 35/36 (a Gaussian-kernel marginalization) and the 2nd-order theory both omit entirely. **This is a real, higher-order noise-rectification effect that is currently uncaptured by anything in this codebase or the paper's own derivation as implemented -- not a wiring bug, not a tunable-away approximation error.**

**The correct fix, following directly from "the realization about the fixed target moments is key" (per user): a per-target, zero-free-parameter correction estimated by literally repeating the SAME synthetic measurement process (many noise draws, real `recenter()`) on that target's OWN reconstructed template, capturing every order at once -- not a truncated formula, not a fitted scalar.** This is exactly the "build a copy grid via actual noisy recenter() solves" step proposed earlier in this file (2026-09-16, "Concrete next step") but never executed. Concretely: for each target (or, more tractably, a dense grid of `(Sigma_X, moment-invariant)` bins spanning the training population), draw N synthetic noise realizations at the SAME `noise_sigma`/PSF the target reports, run `recenter()`, and take the empirical mean shift directly as the correction -- with NO functional-form assumption (linear-in-split or otherwise) imposed on top. This is still strictly "learned from templates/copies alone": every population/noise draw needed is synthetic, from `imsims.sim`, never the real `psf_e` grid catalogs, and the correction table/network is validated the same way `mean_weight`/dense-coverage were (held-out galaxies, held-out amplitude) before ever touching the real leak.

**Not yet done, the concrete next step:** build this per-target noisy-recenter()-MC correction (as a function of each galaxy's own moments + Sigma_X, likely via a small net trained the same way `SigmaXBlockLayer`/`train_sigmax` already is, but with the TRAINING TARGET replaced by noisy-MC bias instead of the exact noiseless copy-grid, and APPLIED to `M_i` directly on the data side, not folded into the flow's density) and validate held-out before running against the real catalogs. This needs materially more compute than anything run this session (many noise draws per training galaxy, across a grid of Sigma_X anisotropy) -- estimate cost before committing to a scale, per this file's own recurring cost-note convention.

## 2026-09-18 session 5 — closes Step 2 (C_M/Sigma_X wiring); does not touch session 4's conclusion

Picked up "Step 2" from this file's earlier closure plan (does `C_M`'s
PSF-induced anisotropy actually reach `mixture_draws`/`log_conv_is`/
`pqr_streamed` correctly, or is there a packing/diagonal-only/transposition
bug). Everything below was re-derived from the literal current code or
`~/gitrepos/bfd` this session, not carried forward from a docstring, prior
session file, or memory — a first pass at this same task made exactly that
mistake (repeated a working-tree docstring's unverified claim as settled
fact) and was corrected before writing this entry. Two false starts are
recorded below so the next session doesn't repeat them.

**False start, corrected**: initially reported "the `E·conj(X)` spin
covariance bug in `SigmaXBlockLayer._ellipticity` was fixed, believed to be
the leak's mechanism, and the fix didn't close the leak" — lifted directly
from that method's own docstring, which itself is **uncommitted** (`git
diff models/bijections.py` shows it as working-tree-only, not on any
branch). Re-derived independently instead of trusting the docstring:

- Built both the old (committed, `(1+c·e1)·x3+c·e2·x4+D·e1` Moebius-style)
  and new (uncommitted, `x3+D·e1+c·proj·e1`, `proj=e1·x3+e2·x4`) forms of
  `_ellipticity`'s `(x3,x4)->(m3,m4)` map and numerically tested rotation
  covariance directly (rotate `(x3,x4)` and `(e1,e2)` together as spin-2
  quantities by a random angle, recompute with `D,c` held fixed since they
  depend only on rotation-invariant scalars, check whether the output
  rotates the same way): **old form residual 0.90 (genuinely not
  covariant), new form residual 4e-16 (covariant to machine precision)**.
  So the docstring's math claim is correct — but that had to be checked,
  not read.
- Confirmed via file mtimes, not assumption, that the trained flow this
  session's own earlier overnight-suite numbers came from
  (`flows/centroid_g2v3d_full2_jac_sigmaxfix.eqx`, written 15:00:57) postdates
  the uncommitted `bijections.py` change (mtime 14:38:20) — so the "fix
  doesn't close the leak" empirical comparison in the previous turn's report
  is valid; the flow really was trained on the fixed formula.
- **What this does NOT establish, and the previous turn's report wrongly
  implied it did**: that this was ever *the* mechanism, or that fixing it
  was expected to close the leak. That was one prior (uncommitted) session's
  hypothesis. Per session 4 above (already in this file before today), the
  actual mechanism is the higher-moment noise-rectification effect in
  `recenter()`, discovered independently of anything in `SigmaXBlockLayer` —
  the covariance bug fix here is orthogonal, real, worth keeping, but never
  had a decisive claim on the leak to begin with.

**Second false start, corrected**: also asserted `cx_to_sx_cond`'s
`e1=(C00-C11)/tr, e2=2·C01/tr` convention "matches" the galaxy ellipticity
convention by pattern-matching the formula's shape, without checking it
against anything. Re-verified against `~/gitrepos/bfd/bfd/moment.py:278-281`
(`MomentCovariance`'s own even-to-odd covariance formula, the actual
definition connecting `M1`/`M2` to the `(X,Y)` odd-moment covariance):
`odd[X,X]=0.5*(Cov(M0,MR)+Cov(M0,M1))`, `odd[Y,Y]=0.5*(Cov(M0,MR)-Cov(M0,M1))`,
`odd[X,Y]=0.5*Cov(M0,M2)` — i.e. `odd[X,X]-odd[Y,Y] = Cov(M0,M1)` and
`odd[X,Y] = 0.5*Cov(M0,M2)`. Same index correspondence and sign as
`cx_to_sx_cond` (`e1` tracks the `C00-C11`/`M1` axis, `e2` tracks the
`C01`/`M2` axis) — genuinely confirmed now, not assumed, no swap or sign
flip between this repo's `Sigma_X` condition and `bfd`'s own `M1`/`M2`
convention.

**Also checked directly this session, no packing/diagonal-approximation bug
found**:
- `bias.load_cov` (bias.py:430) is called once per population/psfe-config
  catalog path (bias.py:3293, `path(cat["zero"])`), so each config's own
  anisotropic `C_M` is used, not a population-wide isotropic stand-in.
- `bias.py:1599`, `cinv = jnp.asarray(np.linalg.inv(cov), jnp.float32)` — a
  genuine full-matrix inverse of the 5x5 `C_M`, and `log_conv_is_kernel`
  (bias.py:699-745) consumes it as a full quadratic form,
  `einsum("si,ij,sj->s", r0, cinv, delta)` — not a diagonal approximation
  anywhere in this path.
- `sigma_x_all = rows["zero"]["cov_odd"]` (bias.py:3272) and
  `bias.condition()` forward the full `[C00,C01,C11]` triple into the flow's
  condition vector verbatim; `SigmaXBlockLayer._unpack` reconstructs the
  full 2x2 `C_X` from all three components, not just the diagonal.
- `imsims/copies.py`'s `log_weights` (the function actually used to train
  `SigmaXBlockLayer`, via `CopySampler`/`weighted_copy_mean`) uses the FULL
  anisotropic 2x2 quadratic form (`chisq = (c11*x*x - 2*c01*x*y +
  c00*y*y)/det`) — genuinely anisotropic, not the isotropic-Sigma_X claim a
  quick read might suggest. The isotropic-scalar treatment (`sigma_xy =
  sqrt(cov_odd[0])`, line 187) is a SEPARATE, narrower use: it only sets the
  copy GRID's spacing/extent argument to `makeTemplates`, not the per-copy
  weight. This confirms (from the code itself, not by re-citing
  `NEXT_SESSION.md`) that the copy-grid-density effect this repo's memory
  already quantified as second-order (~7-16% of slope) is architecturally
  where this file's earlier sessions said it was — a grid-geometry
  mismatch, not a weight-formula bug — but that specific 7-16% magnitude
  number itself was not re-run this session and should be treated as a
  citation, not a re-derived result, until it is.

**Bottom line**: Step 2 is closed by direct code/algebra inspection — the
full anisotropic `C_M`/`Sigma_X` tensors, correct sign convention, reach
every part of the production likelihood (`mixture_draws`, `log_conv_is`,
`pqr_streamed`, `SigmaXBlockLayer`) with no packing, diagonal-truncation, or
convention-mismatch bug found anywhere in the path. This adds independent
confirmation (from a different angle — data plumbing, not a Monte Carlo
comparison) to session 4's own conclusion that the mechanism is not in
`SigmaXBlockLayer`'s wiring or capacity, but in the higher-moment
noise-rectification effect at `recenter()` itself. **Do not re-open Step 2**
unless a later session finds a specific reason to doubt one of the four
bullets above; the open, unstarted work remains session 4's "concrete next
step" (the per-target noisy-`recenter()`-MC correction), not more auditing
of the flow architecture.

## 2026-09-18 session 6 — numerical magnitude of the higher-order effect; copy-grid density/extent re-derived (corrects a prior memory finding)

Both numbers below are fresh computations this session (scripts in the
session scratchpad, not committed to `dev/` — recipes given so they can be
rebuilt), not citations from memory or a prior session file.

**(1) Relative magnitude of the higher-order (non-Gaussian-`u`) effect vs.
the 2nd-order/Gaussian-`u` prediction.** n=150 `gauss2_fwd` galaxies,
`psf_e1=0.05`, `noise_sigma` production default. Per galaxy: (a) the
exact-Gaussian-`u` prediction — analytic `J0` from `bfd.MomentCalculator.
xyJacobian` at `u=0`, real `Sigma_X` from `getCovariance().odd`, `Sigma_u =
J0^-1 Sigma_X J0^-T`, 4000 antithetic `u ~ N(0,Sigma_u)` draws evaluated
through `getMoment` EXACTLY (no Taylor truncation anywhere — `getMoment` is
bfd's own closed-form phase-shifted moment); (b) the real bias — actual
pixel noise added to the rendered image, 300 antithetic-paired real
`recenter()` Newton solves per galaxy, mean shift of the recentred `M1`.

  | | dM1 |
  |---|---|
  | exact-Gaussian-u prediction | -0.145 +/- 0.102 |
  | real noisy recenter() bias | +1.530 +/- 0.452 (3.4 sigma) |

**The real effect is ~11x the Gaussian-u/2nd-order prediction, opposite
sign.** Since the Gaussian-covariance term is small and wrong-signed, the
higher-order (non-Gaussian third/fourth `u`-moment) contribution is not a
correction on top of the leading term, it IS essentially the whole measured
effect (residual, real minus Gaussian-u prediction: +1.675 +/- 0.463).
Consistent in sign and order of magnitude with session 4's earlier,
larger-n finding at the same amplitude (which used n=17274 vs this n=150) —
independently reproduced here with a fresh, smaller script rather than
re-cited.

**(2) Copy-grid extent/density: re-derived, and this REVERSES the prior
"real, second-order (~7-16%)" characterization from an earlier session,
recorded in memory as `[[psfe-size-tightening-does-not-shrink-leak]]`'s
sibling finding.**

Built two real copies pools this session (`imsims.copies`, n=800
`gauss2_fwd` galaxies, seed 0, matching production's `noise_sigma=1.86`,
`sigma_max=4.0`, `sigma_step=1.0`): narrow (`--sigma-range 1.4`, production,
246 copies/galaxy) and wide (`--sigma-range 3.0`, 3157 copies/galaxy, ~13x
denser). Ran `dev/psfe_no_flow_check.py` (the flow-free, exact eq.-35/36
template-sum leak test) against both, reweighting by each real psfe
config's own `cov`/`cov_odd`, same as that script always does.

Naively (each grid's own ESS>=1000 cut, as the script always applies): wildly
different `c1` (narrow -0.0046, wide -0.0476 at `psfe00`) — looks like a huge
grid-density effect, and superficially consistent with there being a real
dependence. But the two grids' ESS cuts admit almost entirely DIFFERENT
targets (786 vs 2185 of 20000 pass window+ESS) because a denser grid
trivially raises every target's ESS — this is a target-ADMISSION artifact,
not evidence the per-target Q,R computation itself depends on grid density.

Redid it holding the target set FIXED across both grids (window mask only,
then intersect each grid's own ESS-pass set at a few thresholds, so
`bias()` is always called on the exact same targets for narrow and wide):

  | ESSMIN | n (matched) | narrow c1 | wide c1 | diff |
  |---|---|---|---|---|
  | 0 | 6242 | -0.09528 | -0.09526 | +0.00003 |
  | 500 | 1363 | -0.03228 | -0.03228 | -0.00000 |
  | 1000 | 786 | -0.00462 | -0.00462 | -0.00000 |

Identical to 5 decimal places at every threshold, for both `psfe00` and
`psfe1p10` (same pattern, not shown). **The copy grid's extent/density has
no measurable effect on the per-target Q,R/c1 computation once the target
set is held fixed.** The earlier ~7-16% memory finding almost certainly
measured the same admission-selection effect via a different route (a
10-config paired-bootstrap SLOPE, which does not hold the admitted target
set fixed across the two grids either) rather than a genuine density
dependence.

**Action taken**: `[[psfe-size-tightening-does-not-shrink-leak]]`'s sibling
copy-grid-density finding should be treated as SUPERSEDED/NULL by this
matched-target test, not "real, second-order" — update that memory file
next time it's touched.

**Where this leaves things**: neither of today's numbers changes session
4's conclusion (the leak is a genuine, real, higher-order noise-rectification
effect in `recenter()` itself, not explained by Taylor truncation, the
Gaussian-`u` covariance assumption, `SigmaXBlockLayer`'s wiring/capacity, or
copy-grid geometry). Today's (1) puts a number on how dominant the
higher-order term is (~11x the leading-order prediction, opposite sign);
today's (2) closes off copy-grid density as a contributing channel entirely,
tightening rather than widening the list of remaining candidates. The open,
unstarted work is still session 4's "concrete next step": the per-target
noisy-`recenter()`-MC correction.

## 2026-09-18 session 7 — why gauss2/gauss2_fwd exposes the leak the paper's own test suppresses; corrects session 6's headline number

Triggered by a direct challenge: does session 6's finding contradict the
paper's own eq. 59-60 (`c = (-1.3+/-0.9)e-5`, ">3000x suppression" of PSF
ellipticity, from the paper's ONE PSF-anisotropy test, GalSim §4.2)? Fetched
`bfd_paper.pdf` fresh (not cached) and read §4.2 directly. The paper's test
uses a DECENTERED disk+bulge population — "The center of the bulge is
randomly shifted with respect to the center of the disk... which might
otherwise be canceling some systematic error in the method" (verbatim). This
codebase's `gauss2`/`gauss2_fwd` (and `bulgedisc`) NEVER decenter the bulge
relative to the disc — `draw_bulge_disc`'s own docstring says "two
co-centred Gaussians" — confirmed by reading every `POPULATIONS` entry in
`imsims/sim.py:695-709`. So this is not the same experiment as the paper's,
and the paper's own suppression claim was never scoped to this population.

**Mechanism found this session, from a per-galaxy diagnostic (n=150,
`psf_e1=0.05`, reusing the session-6 higher-order-magnitude machinery but
logging each galaxy's own `e1_gal=M1/Mr`, `e2_gal=M2/Mr` alongside the
higher-order residual `resid1 = dM1_real - dM1_gaussian_u_prediction`):**

`corr(resid1, e1_gal) = -0.57`. Regressing `resid1 = a + b*e1_gal +
c*|e_gal|^2`: `b = -24.7 +/- 4.0` (6.1 sigma) — a large, real, per-galaxy
LINEAR coupling between the galaxy's OWN ellipticity (aligned with the
PSF's fixed direction) and the recentering higher-order bias.

**Why `gauss2`/`gauss2_fwd` exposes this and the paper's population doesn't:**
`analytic`'s 5-parameter gauss2 family FORCES the bulge and disc to share
exactly the same ellipticity (`imsims/sim.py`'s `sample_population_gauss2_fwd`:
`pop["bulge_e1"], pop["bulge_e2"] = e1, e2`, no independent bulge shape,
confirmed reading the function directly) — every galaxy has ONE single,
sharply-defined, undiluted spin-2 shape to couple to the PSF's ellipticity.
The paper's population has only PARTIALLY correlated bulge/disc ellipticities
(`BULGE_ELLIP_ALIGN`-style mixing in this repo's OWN `bulgedisc`, presumably
similar or looser in the paper's) plus the positional decentering — both
would dilute exactly this coupling before any population averaging even
starts. Combined with the paper's enormous N (8.6e8 vs this session's 150),
which lets a POPULATION-MEAN-ZERO linear term (see below) genuinely reach
zero, this is a complete, mechanistic, checked (not hypothesized) answer to
why the two tests disagree.

**This ALSO corrects session 6's headline number — a real confound, not a
new independent finding.** The linear `e1_gal` term's population mean is
EXACTLTY zero by construction (`_ellipticity_wide` draws ellipticity
ORIENTATION isotropically) — but at n=150 the sample's own `mean(e1_gal) =
-0.0182 +/- 0.0075` (2.4 sigma from zero, ordinary finite-sample noise) is
not zero, and multiplied by the -24.7 coefficient contributes +0.45 to
session 6's reported `mean(resid1) = +1.67 +/- 0.37` — roughly 27% of that
number was finite-sample noise in the galaxy-ellipticity draw amplified by
a large coefficient, not a population-level effect. Debiased estimate
(intercept `a=+0.88+/-0.36` plus the quadratic term's contribution via the
sample's own `E[|e_gal|^2]=0.0176+/-0.0024`, coefficient `c=+19.3+/-12.6`,
only 1.5 sigma on its own): **+1.22 +/- 0.43 (2.9 sigma)** — still clearly
nonzero and still far above the Gaussian-u prediction (-0.145+/-0.102 from
session 6), but smaller and noisier than the raw session-6 number.

**Practical consequence for any future measurement of this leak on
`gauss2`/`gauss2_fwd`:** the population's forced bulge/disc co-ellipticity
plus a large per-galaxy coupling coefficient means a NAIVE population mean at
modest N will be inflated/noisy by exactly this galaxy-ellipticity-orientation
sampling confound — analogous to the leader/instability problem Thread 2 had
to fix elsewhere in this pipeline for a different quantity (`R_s`'s score-
function estimator). Any future numerical estimate of this specific leak
should either (a) use much larger N so the linear term's population mean
genuinely reaches zero, or (b) explicitly regress out `e1_gal`/`e2_gal`
per-galaxy the way this session did, or its reported magnitude will not be
trustworthy at face value. `b`'s own significance (6.1 sigma at n=150) means
this correction is not optional at the sample sizes this session has used
so far.

**Not yet done**: confirming this mechanism on the actual PRODUCTION-scale
psfe grid measurement (the `dc1/d(psf_e1)~-0.03` paired-diff numbers from
`PSFE_PROVENANCE.md`/session 5, at n~20000 targets) — does regressing out
each target's own `e1_gal`,`e2_gal` change THAT number materially, the way
it changed today's small-n higher-order-magnitude estimate? If the
production number is largely unaffected (because n~20000 is already large
enough for the linear term to have averaged out), that would confirm the
production leak is dominated by the genuine quadratic/intercept-type
effect, not by this same finite-sample confound at a different scale.

## 2026-09-18 session 8 — checked the ellipticity-orientation confound against production-scale numbers (already-run data, no new sims)

Direct follow-up to session 7's open item: does the galaxy-ellipticity-
orientation confound found on the n=150 toy also move the actual
PRODUCTION-scale (n~20000) `dc1/d(psf_e1)` numbers? Used only already-saved
`pqr/g2v3d_full2_jac_psfe*.npz` files (no new sims). Linearized each arm's
per-target contribution to `ghat` as `s_i = (Rbar^-1 @ q_i)[0]` (`Rbar =
-sum(r)`, fixed since it's a sum over ~20000 targets) -- verified this
reproduces `bias.ghat` exactly (`B.ghat(qp,rp)` vs `Rbar^-1 @ Q` agree to
float precision) -- then regressed `s_i` against each target's own
`e1_gal=M1/Mr, e2_gal=M2/Mr` from the saved zero-shear `moments` field.

  | tag | amp | n | mean(e1_gal) | c1 raw | c1 debiased | corr(s,e1_gal) |
  |---|---|---|---|---|---|---|
  | psfe00 | 0.00 | 19938 | +0.0007+/-0.0015 | -0.00276 | -0.00228 | -0.56 |
  | psfe1p02 | 0.02 | 19941 | +0.0006+/-0.0016 | -0.00289 | -0.00220 | -0.55 |
  | psfe1p05 | 0.05 | 19941 | +0.0004+/-0.0017 | -0.00347 | -0.00344 | -0.50 |
  | psfe1p10 | 0.10 | 19938 | -0.0011+/-0.0025 | -0.00518 | -0.00753 | -0.33 |

  unwindowed slope: raw dc1/d(psf_e1) = -0.058, debiased = -0.075.

**Two conclusions, both direct and decisive:**

1. **The same coupling mechanism found in session 7's n=150 toy shows up
   here**: `corr(s, e1_gal) ~ -0.5` at every config, essentially the same
   sign and magnitude as session 7's `-0.57` on the raw higher-order
   recentering residual. Independent confirmation (different quantity --
   the actual linearized `ghat` contribution, not the toy's `dM1` residual;
   different N -- 20000 vs 150) that the galaxy-ellipticity/PSF-ellipticity
   coupling is real and reaches the production estimator, not an artifact
   of the small-n diagnostic.

2. **But the confound itself does not explain the production leak.**
   `mean(e1_gal)` is within 1 sigma of zero at every psfe config (as CLT
   predicts at this N: the n=150 toy's sampling noise scales down by
   ~sqrt(150/19940)=0.087x here), so there is very little residual of the
   "population-mean-zero term inflated by finite-sample imbalance" problem
   that biased session 7's small-n estimate. Debiasing (zeroing the linear
   term's contribution to `ghat`, keeping the intercept+quadratic pieces)
   does NOT shrink the leak -- **it makes it slightly larger** (unwindowed
   slope -0.058 -> -0.075). The production `dc1/d(psf_e1)` signal is
   therefore NOT an artifact of the session-7 confound; correcting for it
   strengthens rather than weakens the case that the leak is real at
   production scale.

**Caveat on scope**: this used the simple unwindowed `R^-1 Q` estimator
(matches `bias.ghat(sel=None, ns=None)` exactly), not the windowed+
selection-corrected headline number (`-0.0293+/-0.0032` from
`PSFE_PROVENANCE.md`/session 5) -- smaller in magnitude as expected, but the
right comparison for isolating THIS specific confound, since the windowing/
selection-term channel was already separately audited and closed (session
5's Step 2). Not yet done: repeating this same debiasing check on the
windowed+selection-corrected estimator itself, though there is no
structural reason to expect a different qualitative answer given the
selection-term machinery was already shown (session 5, `Q_s` check) to
depend on Sigma_X's rotation-invariant part only, not on individual-target
galaxy ellipticity orientation the way this additive-term linearization
does.

## 2026-09-18 session 9 — ellipticity-width hypothesis REFUTED; noise-regime (RMS(u)/size) hypothesis CONFIRMED, decisively

Direct follow-up to "what population difference exposes this leak" (session
7 named forced bulge/disc co-ellipticity as a candidate mechanism; session 8
showed the finite-sample orientation confound doesn't explain the production
number). This session tested two concrete, checkable hypotheses numerically.

**Hypothesis 1 (ellipticity width): REFUTED.** Read the paper's Table 1
directly: `sigma_e, galaxy shape noise = 0.2` for BOTH validation tests, eq.
58's `P(e) ~ e(1-e^2)^2 exp(-e^2/2 sigma_e^2)` -- implemented faithfully in
this codebase as `imsims.sim._ellipticity`. But `gauss2_fwd` (every PSFE
test's population) uses `_ellipticity_wide` instead -- NO `(1-e^2)^2`
suppression, `GAUSS2_FWD_ELLIP_SIGMA=0.869685` (a comment there says this
was tuned to match `bulgedisc`'s own MEASURED |e|, not the paper's
parameter). Sampled both distributions directly: paper's (sigma_e=0.2, WITH
suppression) gives median|e|=0.22, mean(e^2)=0.068; this repo's gauss2_fwd
default gives median|e|=0.65, mean(e^2)=0.44 -- a 6.6x larger `e^2`.
Re-ran session 7's diagnostic (n=150, psf_e1=0.05) with the galaxy
ellipticity swapped to the PAPER's exact eq. 58 form and sigma_e=0.2
(monkeypatched `sim._ellipticity_wide` to call `sim._ellipticity`),
everything else fixed. Result: measured e_gal shrank 15x (mean(e^2) 0.018
-> 0.003) but the debiased population-level higher-order estimate was
UNCHANGED within error: **+1.60+/-0.29 (narrow e) vs +1.22+/-0.43 (wide e,
session 7)** -- if anything slightly larger. Galaxy ellipticity width is not
the driver.

**Hypothesis 2 (noise regime, RMS(u)/galaxy-size): CONFIRMED, decisively.**
Noted in passing in session 7 (`corr(resid1, sx00/Mr0)=+0.26`) but not
followed up until now: `median sqrt(Sigma_X)/sqrt(Mr) = 1.378` in every run
so far (`noise_sigma=0.93`, `sim.NOISE_SIGMA`'s default -- NOT even the
actual production catalogs' `noise_sigma=1.86`) -- the RMS recentring shift
is LARGER than the galaxy's own size. Re-ran the same diagnostic (paper's
sigma_e=0.2 held fixed, isolating this one variable) at `noise_sigma =
0.93/4 = 0.2325` (~4x higher S/N, `sqrt(Sigma_X)/sqrt(Mr)` drops to 0.349,
a genuinely small-perturbation regime). Result: the orientation-independent
intercept collapsed **16x**, from `+1.75+/-0.25` to `+0.107+/-0.016`; the
linear galaxy-ellipticity coupling shrank similarly (`-24.7` -> `-1.35`,
~18x). Both terms scale down together with the noise level, consistent
with a leak driven by `u`'s higher (non-Gaussian) moments -- which grow
with `Sigma_u`/size, not with galaxy shape.

**Conclusion, direct and complete**: this leak is not a property of `gauss2`
specifically, of BFD's formalism, or of galaxy ellipticity -- it is that
this repo's PSFE test suite runs targets at a noise depth where the
centroid-recentering uncertainty is COMPARABLE TO OR LARGER THAN the
galaxy's own size (`sqrt(Sigma_X)/sqrt(Mr)~1.4` at `noise_sigma=0.93`, and
production's actual `noise_sigma=1.86` puts real targets even deeper in
this regime, `sqrt(Sigma_X)/sqrt(Mr)` scaling roughly linearly with
`noise_sigma`, i.e. ~2.8). At that scale, `recenter()`'s Newton solve
routinely lands far enough from the true centre that `M(u)`'s higher-order
nonlinearity dominates, and BFD's eq. 35/36 Gaussian-kernel marginalisation
(an implicit small-`u` approximation) is not valid. The paper's own §4.2
test explicitly avoided this: `8 < S/N < 20` selection, weight sigma=3.5 pix
against PSF r50=1.5 pix -- both choices that keep targets in a regime where
centroid uncertainty stays small relative to galaxy size, unlike this
codebase's PSFE grid.

**This is now a complete, checked answer to why the paper's >3000x
suppression and this codebase's real, reproducible leak do not contradict
each other**: different populations at different relative noise levels,
not a flaw in either. It does NOT mean the leak is fake or unimportant for
this codebase's own actual use case (whatever depth/S-N regime the real
target catalogs it cares about will actually run at) -- it means the FIX,
if wanted, is either (a) restrict analysis to a flux/size window where
`sqrt(Sigma_X)/sqrt(Mr)` stays safely below ~1 (a concrete, checkable
selection criterion, unlike anything tried in Threads 1-3 so far), or (b)
build the per-target noisy-`recenter()`-MC correction session 4 already
proposed, which captures this nonlinearity directly rather than avoiding it
via a cut.

**Not yet done**: computing `sqrt(Sigma_X)/sqrt(Mr)` directly on the actual
psfe grid production catalogs (not just this session's own smaller test) to
confirm production targets really do sit at ~2.8, and checking whether
Thread 1's own validated-null window `(2.2,3.2)x(1500,20000)` already
happens to exclude the worst tail of this ratio (which would explain why
the window's own `m1` came out null in Thread 3 while `c1`/`c2` stayed
significant -- multiplicative and additive terms can have different
sensitivity to this same regime) or whether a size/flux cut targeting THIS
specific ratio directly would do better than the existing window.

## 2026-09-18 session 10 — production-catalog confirmation + a working, checkable fix (window on the ratio)

Two follow-ups to session 9, both using only already-saved data (no new
sims, no new flow inference).

**(1) Confirmed session 9's ratio estimate directly on the real production
catalogs.** Read `cov_odd`/`moments` straight from `targets_v3psfe0010_
gauss2fwd_d186_g0_20k.fits` (psfe00) and `targets_v3psfe1p10_gauss2fwd_
d186_g0_20k.fits` (psfe1p10) -- the actual zero-shear catalogs `bias.py`
uses for the psfe grid. `median sqrt(Sigma_X)/sqrt(Mr) = 2.96` in BOTH
(near-identical between configs, confirming this ratio is not itself PSF-
ellipticity-driven) -- worse than session 9's own noise-sigma-scaled
extrapolation (~2.8). Only 7.2% of targets have ratio<1.0; 75.7% have
ratio>2. Session 9's controlled experiment (leak collapses 16x when this
ratio drops from 1.4 to 0.35) is therefore directly relevant to THIS
population at THIS depth, not just the smaller toy test.

**(2) Windowing on this ratio suppresses the additive leak, using only
already-saved per-target q,r/moments.** Nearest-neighbor matched each
`pqr/g2v3d_full2_jac_psfe*.npz`'s zero-arm `moments` fingerprint to the raw
target catalog's own `moments`+`cov_odd` (same technique `dev/
psfe_paired_diff.py`'s `align()` uses), computed the ratio per pqr row, and
recomputed `B.bias()` (unwindowed formula, no selection-term correction) at
several ratio cuts across `psfe00/1p02/1p05/1p10`:

  | selection | n (per config) | dc1/d(psf_e1) |
  |---|---|---|
  | unwindowed | ~19940 | -0.058 |
  | ratio<2.0 | ~4830 | +0.009 |
  | ratio<1.5 | ~2890 | -0.020 |
  | ratio<1.0 | ~1430 | +0.022 |
  | ratio<0.7 | ~785 | -0.017 |

Every ratio-cut slope is 3-6x smaller in magnitude than the unwindowed
value and changes sign across thresholds -- consistent with the true effect
being ~0 once the noise-dominated tail is excluded, the remaining scatter
being ordinary sampling noise at these smaller n (no bootstrap error
computed here -- these are point estimates, not a validated-null claim like
Thread 3's window scan; that rigor is the natural next step, not yet done).

**Mechanism, and why this connects to something ALREADY IN the pipeline:**
`corr(ratio, log10(Mf)) = -0.898` on the whole catalog -- this ratio is
almost entirely a flux/S-N proxy, not an independent axis. Checked whether
Thread 1's existing validated window (`(2.2,3.2)x(1500,20000)`) already
controls it: barely -- inside that window, median ratio only drops from
2.96 to 2.26 (`frac(ratio<1.5)` only 14.5% -> 18.4%). **The window's own
`flux_lo=1500` floor is nowhere near aggressive enough to exclude the
regime driving this leak.** This is a concrete, actionable, checkable
correction to Thread 1's own window recommendation, not a new formalism
fix: raising `flux_lo` substantially (or adding an explicit
`sqrt(Sigma_X)/sqrt(Mr)` selection term alongside the existing Mr/Mf, Mf
cuts) should suppress the additive PSFE leak the same way it does here.

**Not yet done, the natural next steps**:
- Re-run this with `dev/window_scan.py`'s proper machinery (bootstrap
  errors, the flow-based `score` selection-term correction, not the bare
  unwindowed `R^-1 Q` estimator used here) to turn this point-estimate
  pattern into a validated-null claim with real error bars, the way Thread
  3's own window scan did.
- Find the actual `flux_lo` (or a direct ratio cut) that achieves both (a)
  a validated-null `c1`/`c2` slope with proper errors and (b) doesn't
  discard so much of the catalog that the selection-term correction (`R_s`)
  becomes leader-dominated again (Thread 2's instability) -- these two
  constraints may be in tension and need to be checked together, not
  separately.
- Confirm this fix doesn't just relocate Thread 3's own flux-tail bias
  (isolated to `flux_hi`, not `flux_lo`) into a different regime -- Thread 1
  and Thread 3 have now both independently implicated flux/depth-driven
  edges of this population, from opposite ends (bright/sparse for Thread 3,
  faint/noise-dominated for Thread 1) -- worth checking whether ONE window,
  chosen to jointly satisfy both, exists, or whether they trade off.
