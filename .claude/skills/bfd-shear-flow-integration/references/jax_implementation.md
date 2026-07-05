# JAX / FlowJax implementation notes

Companion to `SKILL.md` and `references/derivations.md`. The verified core lives in
`scripts/reference_estimator.py`; this file is about wiring it to the real flow and
scaling it to an ensemble.

## 1. Wiring the real forward map

The reference uses an affine stub. In production, the only thing that changes is
`forward_map(z, g) -> x`, the flow's **base→data** map at condition `g`:

```python
import equinox as eqx

# Build the full condition vector the flow was trained on:
#   cond = [g1, g2, log_scale, e1, e2]   (shear + PSF centering-bias params)
def make_forward(flow, log_scale, e1, e2):
    def forward_map(z, g):                       # g = [g1, g2]
        cond = jnp.concatenate([g, jnp.array([log_scale, e1, e2])])
        return flow.bijection.transform(z, condition=cond)   # CONFIRM direction
    return forward_map
```

**Confirm the direction before anything else.** FlowJax's `Transformed(base, bijection)`
convention is that `bijection.transform` maps base→data and `log_prob` uses the inverse.
If your custom bijection stack is the other way, you will integrate the wrong density.
Use the round-trip check the project already relies on:

```python
z = jr.normal(key, (d,))
x = forward_map(z, g)
z_back, lad_fwd = flow.bijection.transform_and_log_det(z, condition=cond)
z_inv,  lad_inv = flow.bijection.inverse_and_log_det(x, condition=cond)
assert jnp.allclose(z_inv, z, atol=1e-5)         # base recovered
assert jnp.allclose(lad_fwd + lad_inv, 0.0, atol=1e-5)   # log-dets cancel
```

Differentiability in `g` requires `transform` to be differentiable in `condition`,
which Equinox/JAX give for free. Nothing else in the estimator touches the flow.

## 2. JAX structure: what to jit, what not to

- **Host-side (NumPy/SciPy):** Sobol point generation (`scipy.stats.qmc`) and the
  inverse χ² CDF. These produce the fixed `z_samples` array. Do this once per target,
  outside any traced function.
- **Traced (jit-able):** mode-finding (`lax.scan` Adam), the Laplace Hessian, and the
  `log P(g)` closure with its `value_and_grad` / `hessian`. `vmap` the forward map over
  the sample axis: `jax.vmap(lambda z: forward_map(z, g))(z_samples)`.
- **float64.** `jax.config.update("jax_enable_x64", True)`. Moment integrals span many
  orders of magnitude in the weights; float32 underflows the tails and corrupts k-hat.
- **Numerically stable P/Q/R.** Always go through `logsumexp` for `log P`, then
  `jax.value_and_grad(logP)` and `jax.hessian(logP)`, and convert to P/Q/R with the
  identities in §7 of the derivations. Do not exponentiate weights before summing.

## 3. Vectorizing across the target ensemble

Two layers of `vmap`: inner over the `S` samples (already in the reference), outer over
targets. The outer `vmap` is only clean if every target uses the **same** S and proposal
family; since the budget allocator (below) gives different targets different S and round
counts, prefer a Python loop (or `lax.map` with padding) over the ensemble and `vmap`
only the inner sample axis. Mode-finding and the Hessian are cheap relative to the S
forward calls, so the loop overhead is negligible.

## 4. In-trace diagnostics

Mirror the project's existing RQMC debugging with `jax.debug.print` inside the mode
finder and the weight computation:

```python
jax.debug.print("z* grad norm {g:.2e}", g=jnp.linalg.norm(grad_at_zstar))
jax.debug.print("logP {p:.4f}  max ℓ {m:.2f}  min ℓ {n:.2f}", p=logP, m=lw.max(), n=lw.min())
```

Per target, log at least: `‖∇_z L(z*)‖` (should be ~0), the smallest eigenvalue of `H`
(should be > 0), k-hat, and ESS. A target with `‖∇L‖` not near zero means mode-finding
did not converge — raise `n_steps` or lower `lr`.

## 5. AMIS / multi-round escalation (the k-hat gate)

```python
def estimate_with_escalation(forward_map, g0, M, Sigma, base_S=512,
                             khat_thresh=0.7, max_rounds=3):
    proposals, samples = [], []          # accumulate (q params, z block) per round
    nu, alpha, inflate = 4.0, 0.8, 1.3
    for r in range(max_rounds):
        # build proposal r (broaden each round), draw base_S points, accumulate
        ...
        # DM weights over the POOLED set: denominator = mean over rounds of q_r(z)
        #   D_s = (1/R) Σ_r q_r(z_s);  w_s = φ(z_s) k(T_g(z_s)) / D_s
        khat = psis_khat(pooled_log_weights)
        if khat < khat_thresh:
            break
        nu, inflate, alpha = max(3.0, nu-1), inflate*1.5, max(0.5, alpha-0.1)
    return P, Q, R, khat, total_calls
```

The crucial detail (see derivations §9): the denominator for **every** pooled sample is
the *average of all round proposals* evaluated at that sample, not the proposal that
generated it. Get this wrong and the multi-round estimate is inconsistent.

## 6. k-hat-gated budget allocator across the ensemble

To hit ~1000 flow calls/target on average while spending where it matters:

1. First pass: every target at `base_S` (e.g. 512). Record k-hat and ESS.
2. Bright/narrow targets typically land at k-hat ≪ 0.5 with ESS already high — leave
   them; they "bank" budget.
3. Targets with k-hat > 0.5 get a second round (DM-pooled), escalating to 1024–2048.
4. Track the running ensemble mean of calls; stop escalating once it approaches the
   1000 cap, prioritizing the highest-k-hat targets first.

This keeps the *ensemble* near budget even though individual targets vary by 4×.

## 7. Analytic-core control variate (accelerator)

Once the plain estimator is verified, optionally subtract the closed-form Gaussian core
(derivations §10). Build a Gaussian `p̃ = N(μ_p, Σ_p)` from the Laplace fit pushed to
x-space (`μ_p = T_g(z*)`, `Σ_p = J Σ_prop J^T`), compute `∫ p̃ k = N(M; μ_p, Σ_p+Sigma)`
in closed form, and MC only the residual. Verify it returns the *same* P as the plain
estimator on the affine test (where the residual is exactly zero) before trusting it on
the real flow.

## 8. GPJax Bayesian-quadrature cross-check

Given the project's GPJax thread, BQ is a good **independent check** on the IS estimate,
not a replacement. Model the non-negative integrand `f(z) = k(T_g(z))` (or its log,
warped) with a GP under the Gaussian measure `φ`; the kernel mean embedding gives a
closed-form posterior mean and variance for `P = ∫ f φ dz`. It can reach low error in
well under 1000 calls and hands you an error bar. Caveats that keep it secondary:
its estimate is **model-based, not frequentist-unbiased**, and active point selection is
sequential (hard to vectorize). Use it to validate the IS P on a handful of targets
spanning the S/N range; if BQ and IS agree within their error bars, trust the fast IS
path for the full ensemble.

## 9. Serialization

Persist the inference config alongside the flow with the project's existing pattern:
`eqx.tree_serialise_leaves` for the flow, JSON header for hyperparameters. Record the
estimator settings used (`S`, `nu`, `alpha`, `inflate`, antithetic on/off, RQMC seed
policy, k-hat threshold) in that header so a shear catalog is reproducible.

## 10. Minimal-diff integration

The reference is structured so you can lift individual functions into an existing
notebook cell rather than adopting the whole module: `find_mode`, `laplace_chol`,
`sample_mixture_rqmc`, `make_logP`, and `psis_khat` are independent. Keep your current
RQMC-inference cell's mode-finding (Adam + clipping) and Hessian-proposal scaffolding;
the new pieces are (a) moving them into z-space if they were in x-space, (b) the
defensive-t mixture density in the weight denominator, and (c) the autodiff Q/R from the
fixed-sample `log P(g)`.
