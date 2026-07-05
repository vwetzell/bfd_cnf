---
name: bfd-shear-flow-integration
description: >-
  Implement and debug the BFD shear-recovery integral for a conditional normalizing-flow
  prior — estimating P(g)=∫ p_theta(x|g) N(x; M, Sigma) dx and its shear derivatives
  Q=∇_g P and R=∇²_g P from a target's measured moments M and covariance Sigma. Use this
  whenever the task involves: recovering shear from a trained flow at inference;
  integrating a flow prior against a Gaussian noise kernel; importance sampling over a
  cNF; mode-finding or Laplace proposals in the flow base (z) space; RQMC/QMC integration
  of P/Q/R; Pareto-k (PSIS) or ESS diagnostics on importance weights; skewed or
  heavy-tailed posteriors where the integrand peak differs from the measured moments; or
  keeping flow calls low (~1000/target) while staying unbiased. Trigger for "recover
  shear from the flow", "integrate the prior against the target covariance", "estimate P
  Q R", "importance sampling the flow", "k-hat / effective sample size", or "the peak
  isn't at the measured moments", even when a JAX/Equinox/FlowJax stack is only implied.
---

# BFD cNF shear-recovery integration

This skill implements the inference-time integral that turns a trained conditional
normalizing-flow prior into BFD shear statistics. The training device `q_phi` is gone
by now; the only job is to integrate the **prior** `p_theta(x | g)` against each
target's Gaussian **noise kernel** and differentiate the result in the shear `g`.

There is a **verified reference implementation** in `scripts/reference_estimator.py`
and a self-test in `scripts/test_estimator.py` that checks it against an analytic
Gaussian ground truth (P, Q, R, the z-mode, and the k-hat diagnostic). **Read and run
the test first** — it pins down every sign and normalization convention before you
touch the real flow.

## The integral and why naive sampling fails

For a target with measured moments `M ∈ R^4` and covariance `Sigma`:

```
P(g) = ∫_{R^4} p_theta(x | g) · N(x; M, Sigma) dx          x ≡ M^G (noiseless moments)
Q(g) = ∇_g P(g)        R(g) = ∇²_g P(g)                     (the BFD shear response)
```

The integrand is a **shifted, skewed, heavy-tailed** bump, not a Gaussian centered on
`M`. Its mode `x*` solves `∇_x log p_theta(x*|g) = Sigma^{-1}(x* − M)`, so the prior
slope pushes the peak off `M` (the "peak ≠ measured moments" symptom), and the
local-vs-tail curvature mismatch is the asymmetric tail. Any method must respect this.

Two dual unbiased Monte Carlo forms exist and **fail in opposite regimes**:

- **Sample prior, weight by kernel** `P = E_{x~p_theta}[N(x;M,Sigma)]` — dies for
  bright/narrow-kernel targets (almost every draw lands where the kernel ≈ 0).
- **Sample kernel, evaluate density** `P = E_{x~N(M,Sigma)}[p_theta(x|g)]` — dies for
  faint/broad-kernel targets where the prior slopes hard across the kernel.

Neither is robust across the S/N range. The fix is one importance-sampling proposal
that interpolates, built in the flow's **base (z) space**.

## The recipe (the spine of every implementation)

Work entirely in z-space using the change-of-variables identity (no Jacobian needed):

```
P(g) = E_{z ~ N(0,I)} [ N(T_g(z); M, Sigma) ]        T_g = flow base→data forward map
```

1. **Mode-find in z.** Maximize `log φ(z) + log k(T_g(z))` with Adam + gradient
   clipping (`φ=N(0,I)`, `k=N(·;M,Sigma)`). The flow has absorbed most of the prior's
   non-Gaussianity, so the pushed-forward integrand is far closer to isotropic in z —
   the skew/tails you fight in x-space are largely gone here. Get `z*`.
2. **Laplace curvature.** `H = −∇²_z [log φ + log k(T_g(z))]` at `z*`; proposal
   covariance `Σ_prop = inflate · H^{-1}` (inflate ≈ 1.2–1.5). Symmetrize + jitter
   before Cholesky.
3. **Defensive Student-t proposal.** `q(z) = α · t_ν(z*, Σ_prop) + (1−α) · N(0,I)`
   with `ν ∈ [3,7]`, `α ≈ 0.8`. The t tails are the cure for the heavy-tailed target:
   a Gaussian Laplace proposal has *lighter* tails than the target, so the weights
   `φ·k/q` are unbiased but **infinite-variance** in practice. The base-normal
   defensive component is free to sample (it *is* ordinary prior sampling) and
   provably bounds the weights in the prior tail.
4. **Draw with RQMC + antithetics.** Scrambled Sobol in `[0,1]^{d+1}` mapped through
   inverse-CDFs (Gaussian dims + one chi-square mixing dim for the t). 4D is in
   RQMC's sweet spot. Antithetic reflection of the seed about `z*` cancels the leading
   first-order slope variance. Use a **deterministic component split** (round(αS) from
   the t, the rest from the base) and evaluate the **full mixture density** in the
   denominator for every point — this keeps it unbiased.
5. **IS estimate of P, autodiff for Q/R.** `log P = logsumexp_s(log φ_s + log k_s(g) −
   log q_s) − log S`. Because the proposal `q` and the samples `z_s` are **fixed and
   g-independent** (built once at a reference g), `Q` and `R` come from autodiff of
   `log P(g)` over the *same* draws (common random numbers) — low variance and mutually
   consistent. Convert: `Q = P·∇_g logP`, `R = P·(∇²_g logP + (∇_g logP)(∇_g logP)^T)`.
6. **Gate every target on PSIS k-hat.** Fit a generalized Pareto to the upper weight
   tail. `k-hat < 0.5` → trust it; `0.5–0.7` → marginal; `> 0.7` → proposal inadequate.
   This is the per-target check that tells you which targets need a wider proposal or
   more draws, so you can allocate the ~1000-call budget adaptively.
7. **Escalate flagged targets.** For high k-hat: broaden (raise `ν`, inflate `Σ_prop`,
   shift `α` toward the base) and/or run another round, pooling all draws with
   deterministic-mixture (balance-heuristic) denominator weights. Borrow budget from
   the cheap bright targets so the ensemble mean stays near 1000 flow calls/target.

The committed default for a 1000-call budget: mode-find in z → defensive Student-t →
scrambled-Sobol RQMC + antithetics → standard (normalized-weight) IS for P with
autodiff Q/R → k-hat gate → re-run heavy-k-hat targets. Unbiased up to negligible
PSIS smoothing, robust across S/N, self-diagnosing.

## When to use which fallback

| Symptom | Diagnostic | Action |
|---|---|---|
| Bright galaxy, kernel narrow | low ESS from prior sampling | the t-component dominates; `z*` far from 0 — expected, fine |
| Faint galaxy, kernel broad | k-hat fine, big `(1−α)` contribution | the base component carries it; fine |
| k-hat in 0.5–0.7 | marginal | raise `inflate` to 1.5–2, lower `ν` to 3, one extra round |
| k-hat > 0.7 | proposal inadequate | broaden + 2nd AMIS round with DM weights (see references) |
| P fine but Q/R noisy | curvature undersampled | R is 2nd-order; add samples (error ~1/√S) before suspecting a bug |
| heavy skew a single t can't fit | k-hat marginal across rounds | 2–3 round AMIS fitting a small t-mixture (references/derivations.md) |

## Correctness invariants — check these before trusting any number

These are the failure modes that produce plausible-but-wrong shear. They are spelled
out with derivations in `references/derivations.md`; the short version:

- **Normalized weights, never self-normalized.** `w = φ·k/q` with `q` a proper density.
  Self-normalization silently re-introduces a partition-function bias.
- **`q` and the samples must not depend on `g`.** Build them once at the reference `g`
  (usually 0), then differentiate `k(T_g(z))` only. If the proposal tracks `g`, Q/R
  pick up spurious score terms.
- **No flow Jacobian in the z-space estimator.** The change of variables cancels it
  exactly. If a `log_det` appears in your weight, you have double-counted — recheck.
- **Round-trip the forward map.** Confirm `inverse(transform(z)) ≈ z` and that
  `transform` is base→data, not the reverse. The reference uses `lad_fwd + lad_inv ≈ 0`
  in the same spirit; a flipped direction makes everything subtly wrong.
- **Mode is a real optimum.** Check `‖∇_z(logφ + logk)‖ ≈ 0` at `z*` and that `H ≻ 0`.
- **Differentiate in g via reparameterization, not through the sampler.** g enters only
  inside `T_g(z)`; z is fixed. `jax.value_and_grad` / `jax.hessian` of `log P(g)` is
  exact.

## How to validate (do this before wiring the real flow)

`scripts/test_estimator.py` replaces the flow with an affine map `x = A z + b0 + B g`,
making the prior exactly Gaussian so P, Q, R, and the z-mode are **closed form**. It
asserts the estimator matches to Monte Carlo tolerance, shows R's error shrinking with
sample count (proving unbiasedness, not bias), and shows k-hat correctly spiking for a
deliberately too-narrow proposal. Run it, read it, then swap in the real forward map:

```python
forward_map = lambda z, g: flow.bijection.transform(z, condition=g)   # confirm direction!
```

Keep this affine test as a regression guard — it will catch sign flips and
normalization slips in any future refactor.

## Implementation guidance and the real-flow wiring

`references/jax_implementation.md` covers: the FlowJax/Equinox interface (forward map,
round-trip check, condition vector layout `[g1, g2, log_scale, e1, e2]`), batching with
`vmap`/`jit` and when NOT to jit (Sobol generation is host-side), float64, numerically
stable logsumexp Q/R, `jax.debug.print` in-trace diagnostics, the AMIS/deterministic-
mixture multi-round bookkeeping, the k-hat-gated budget allocator across an ensemble,
the analytic-core control-variate accelerator (Gaussian × Gaussian closed form + MC on
the residual), and the GPJax Bayesian-quadrature cross-check (sample-efficient but
model-based, so a check on the IS estimate rather than the primary).

`references/derivations.md` has the full algebra: the z-space change of variables, the
dual forms and their variance, the mode/Hessian equations, multivariate-t sampling and
density, the antithetic first-order cancellation, the PSIS GPD estimator, and the
AMIS/DM-weight consistency argument.

## Reference papers

Bernstein et al. 2016 (MNRAS, stw879) for the BFD P/Q/R formalism and the ensemble
shear estimator `g_hat = −(Σ R)^{-1}(Σ Q)`; Dockhorn et al. 2020 (arXiv:2006.09396)
for the flow-integration context. Both PDFs are in the project knowledge.
