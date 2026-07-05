# Derivations

All algebra for the z-space defensive-IS estimator. Notation: `x ∈ R^d` are the
noiseless moments `M^G` (d=4); `M`, `Sigma` are the target's measured moments and
covariance; `k(x) = N(x; M, Sigma)` is the noise kernel; `φ(z) = N(z; 0, I)` is the
flow base; `T_g` is the flow's base→data forward map at shear `g`; `p_theta(x|g)` is
the prior density. We want `P(g)=∫ p_theta(x|g) k(x) dx` and its g-derivatives.

## 1. The z-space identity (no Jacobian)

The flow is a bijection `x = T_g(z)` with `z ~ φ`, so the density is
`p_theta(x|g) = φ(z) / |det J|`, where `J = ∂T_g/∂z`. Change variables in the integral,
`dx = |det J| dz`:

```
P(g) = ∫ p_theta(x|g) k(x) dx
     = ∫ [φ(z)/|det J|] · k(T_g(z)) · |det J| dz
     = ∫ φ(z) k(T_g(z)) dz
     = E_{z~φ}[ k(T_g(z)) ].
```

The Jacobian cancels exactly. **The estimator never needs a flow log-det.** If one
appears in your importance weight, it is double-counted. This identity is also why
mode-finding and the proposal live in z-space: the integrand-as-density is
`π(z) ∝ φ(z) k(T_g(z))`, which is much closer to isotropic Gaussian than `p_theta · k`
in x-space because `T_g` has already removed most of the prior's non-Gaussian shape.

## 2. The two dual MC forms and their variance

**Form A (sample prior, weight by kernel):** the identity above with `z ~ φ`,
`P̂ = (1/S) Σ k(T_g(z_s))`. Variance `= Var_φ[k]/S`. When the kernel is narrow (bright
galaxy) almost all prior draws land in `k ≈ 0`; the few nonzero terms dominate and the
variance is huge.

**Form B (sample kernel, evaluate density):** `P = E_{x~N(M,Sigma)}[p_theta(x|g)]`,
`P̂ = (1/S) Σ p_theta(x_s|g)`, `x_s ~ N(M,Sigma)`. Variance `= Var_N[p_theta]/S`. Tiny
when the kernel is narrow (the prior is ~constant across it) but blows up when the
kernel is broad and the prior slopes/has tails across it (faint galaxy). Form B also
needs a flow `log_prob` (density) evaluation, i.e. the inverse map + log-det.

They fail in opposite regimes, motivating a single interpolating proposal.

## 3. Importance sampling in z

Pick a proposal density `q(z)`. Then

```
P(g) = ∫ q(z) · [φ(z) k(T_g(z)) / q(z)] dz = E_{z~q}[ w(z) ],   w = φ k / q.
```

`P̂ = (1/S) Σ w(z_s)` is **unbiased iff `q` is a normalized density** and `w` uses
those true densities (standard weights). Self-normalized weights `w_s/Σ w_s` are biased
at finite S and re-introduce exactly the partition-function bias removed at training —
do not use them for P. In log space, with `w_s = exp(ℓ_s)`,
`ℓ_s = log φ(z_s) + log k(T_g(z_s)) − log q(z_s)` and
`log P̂ = logsumexp_s(ℓ_s) − log S`.

## 4. Mode and Laplace curvature in z

Let `L(z) = log φ(z) + log k(T_g(z)) = −½‖z‖² − ½(T_g(z)−M)^T Σ^{-1}(T_g(z)−M) + c`.

Stationarity `∇_z L = 0`:

```
−z + J^T Σ^{-1}(M − T_g(z)) = 0   ⇒   z = J^T Σ^{-1}(M − T_g(z)),   J = ∂T_g/∂z.
```

`z=0` is generally not a solution: the prior slope `J^T Σ^{-1}(M − b)` displaces the
mode (the "peak ≠ measured moments" effect, in z-coordinates). Solve by Adam ascent
with gradient clipping from `z0 = 0`.

Curvature: `H = −∇²_z L(z*)`. Expanding,

```
−∇²_z L = I + J^T Σ^{-1} J − Σ_a [Σ^{-1}(T_g − M)]_a · ∂²(T_g)_a/∂z∂z.
```

The first two terms are the Gauss–Newton part (always PSD); the third is the
flow-curvature correction and can make `H` indefinite far from the mode. Use the full
`jax.hessian` at `z*`, then **symmetrize and add jitter** before inverting:
`Σ_prop = inflate · H^{-1}`, `inflate ∈ [1.2, 1.5]`. If `H` is not PD, fall back to the
Gauss–Newton approximation `I + J^T Σ^{-1} J` (always PD).

**Affine sanity check** (used by the self-test): for `T_g(z)=Az+b`, `J=A`, and the mode
solves `(I + A^T Σ^{-1} A) z* = A^T Σ^{-1}(M − b)`, with `H = I + A^T Σ^{-1} A` exactly.

## 5. Defensive Student-t proposal

```
q(z) = α · t_ν(z; z*, Σ_prop) + (1−α) · φ(z),     ν ∈ [3,7],  α ≈ 0.8.
```

Why a t and not a Gaussian Laplace proposal: the target `π(z) ∝ φ k(T_g(z))` has
heavier tails than a Gaussian in the directions the prior is heavy-tailed. A Gaussian
`q` with lighter tails makes `w = φk/q → ∞` along those directions: the estimator is
unbiased but has **infinite variance**, which shows up as a few enormous weights and a
high k-hat. The t with low `ν` has polynomial tails that dominate, bounding `w`.

Why the defensive base component: `(1−α)φ(z)` is free to sample (those draws pushed
through `T_g` are ordinary prior samples) and guarantees `q(z) ≥ (1−α)φ(z)` everywhere,
so `w ≤ φk/((1−α)φ) = k/(1−α) ≤ max(k)/(1−α) < ∞`. The weights are provably bounded —
the single most important robustness property.

**Multivariate-t density** (`Σ_prop = LL^T`, `y = L^{-1}(z−z*)`, `maha = y^T y`):

```
log t_ν(z) = lnΓ((ν+d)/2) − lnΓ(ν/2) − (d/2)ln(νπ) − ½ ln|Σ_prop|
             − ((ν+d)/2) · ln(1 + maha/ν).
```

**Sampling** `t_ν(z*, Σ_prop)`: draw `ζ ~ N(0,I)`, `u ~ χ²_ν`, set
`z = z* + L ζ · sqrt(ν/u)`. (Equivalently a Gaussian scale mixture with inverse-gamma
mixing.) For QMC, map `[0,1]^d → ζ` via inverse normal CDF and one extra coordinate
`[0,1] → u` via the inverse χ²_ν CDF.

**Mixture sampling = unbiased component split.** Sampling from `q = α t + (1−α)φ` is
equivalent to drawing a `round(αS)`-sized batch from `t` and the rest from `φ`
(deterministic stratified allocation), provided the **full mixture density `q`** is used
in every weight's denominator regardless of which component produced the point. This
plays well with RQMC (each component gets its own low-discrepancy block).

## 6. RQMC and the antithetic first-order cancellation

Map scrambled-Sobol points in the unit cube through the inverse-Rosenblatt / inverse-CDF
of each component. In d=4 (+1 for the t mixing variable) RQMC gives close to
`O(S^{-1})` error vs `O(S^{-1/2})` for plain MC on smooth integrands. Request
power-of-2 blocks (`Sobol.random_base2`) for the balance property, then slice.

Antithetics: reflect the Gaussian seed about the mode, `ζ → −ζ`, i.e. include both
`z* + δ` and `z* − δ`. Write the integrand as `f(z* + δ)` and Taylor-expand in `δ`:

```
½[f(z*+δ) + f(z*−δ)] = f(z*) + ½ δ^T ∇²f δ + O(δ⁴).
```

The first-order term `∇f·δ` — exactly the leading variation the strong prior slope
induces across the kernel — cancels in the pair average, removing the dominant variance
contribution. (This is variance reduction, not a bias fix; IS is already unbiased.)
For Gaussian QMC coordinates this is the cube reflection `p → 1−p` (inverse normal CDF
is antisymmetric about ½).

## 7. Q and R by autodiff under common random numbers

Hold `q` and the samples `{z_s}` **fixed** (built once at a reference `g`, typically 0).
Then g enters only through `k(T_g(z_s))`:

```
P(g) = (1/S) Σ_s φ(z_s) k(T_g(z_s)) / q(z_s).
```

Differentiating the sample mean is exact (no sampler reparameterization needed because
z is already g-independent):

```
Q_i = ∂P/∂g_i = (1/S) Σ_s [φ_s/q_s] · ∂_{g_i} k(T_g(z_s)),
R_ij = ∂²P/∂g_i∂g_j = (1/S) Σ_s [φ_s/q_s] · ∂²_{g_i g_j} k(T_g(z_s)),
```

with `∂_{g_i} k = ∇_x k · ∂_{g_i} T_g` etc. — all delivered by `jax.grad`/`jax.hessian`
of `log P(g)`. Using the **same draws** for P, Q, R (common random numbers) makes them
mutually consistent and cancels most of the sampling noise in the *ratios* that BFD
ultimately uses. Compute the stable `log P` first, then convert:

```
Q = P · ∇_g logP,     R = P · ( ∇²_g logP + (∇_g logP)(∇_g logP)^T ).
```

`∂P = ∂(unbiased estimator)` is itself an unbiased estimator of `∂(mean)`, so Q and R
are unbiased; R is just higher-variance (error `~1/√S`), which the self-test confirms by
shrinking it with sample count.

These feed the BFD ensemble estimator. To leading order `g_hat = −(Σ_n R_n)^{-1}(Σ_n
Q_n)` with the sums over the target ensemble (selection terms aside) — see Bernstein
et al. 2016.

## 8. PSIS k-hat

Take the importance weights at the reference g. Set the tail length
`Mt = min(0.2S, 3√S)`, take the `Mt` largest weights, subtract the threshold (the next
weight), and fit a generalized Pareto to the exceedances by the Zhang–Stephens profile
estimator. The shape parameter `k-hat` characterizes the weight-tail heaviness:

- `k-hat < 0.5`: weight variance finite, estimator reliable.
- `0.5 ≤ k-hat < 0.7`: marginal; usable with caution, expect slower convergence.
- `k-hat ≥ 0.7`: the proposal does not cover the target; the estimate (and especially
  its variance) is untrustworthy — broaden the proposal or add an AMIS round.

Optional Pareto smoothing replaces the largest weights with the fitted GPD order
statistics, which trades an `O(1/S)` bias for finite variance — the one deliberate
(tiny) bias in the pipeline, far preferable to a nominally-unbiased estimator with
unbounded weights. The reference returns k-hat as a gate; smoothing is optional.

## 9. AMIS / deterministic-mixture multi-round weights

When one round's k-hat is too high, run `R` rounds with proposals `q_1, …, q_R`
(broadening each time: raise `ν`, inflate `Σ_prop`, shift `α` toward the base, or refit
to the weighted samples). Pool **all** draws from all rounds and weight each with the
**deterministic-mixture / balance-heuristic** denominator (Owen & Zhou 2000; Cornuet et
al. 2012 AMIS):

```
w_s = φ(z_s) k(T_g(z_s)) / [ (1/R) Σ_{r=1}^R q_r(z_s) ],   summed over the pooled set,
P̂ = (Σ_{r,s} φ k / D)  /  N_total,   D = (1/R) Σ_r q_r(z_s).
```

The DM denominator (the *average* of all proposals evaluated at every point, not the
proposal that generated it) is what keeps the multi-round estimator consistent and
low-variance; using each point's own proposal in the denominator instead breaks it. The
recycling means earlier draws are never wasted, so escalation stays cheap.

## 10. Optional: analytic-core control variate

Moment-match a Gaussian `p̃(x) = N(x; μ_p, Σ_p)` to the prior near `x* = T_g(z*)`. Then
`∫ p̃ k = N(M; μ_p, Σ_p + Sigma)` is closed form (Gaussian × Gaussian), and you MC only
the residual `∫ (p_theta − p̃) k`, which is small and bounded when `p̃` is a good local
fit:

```
P = N(M; μ_p, Σ_p + Sigma) + E_{z~q}[ φ k/q − (p̃(T_g(z))/p_theta_eval) · (...) ].
```

In practice the cleanest form keeps the residual in z-space and subtracts the analytic
Gaussian contribution computed from the Laplace fit. This can cut S substantially
because the bulk is analytic and MC only cleans up the non-Gaussian remainder. It stays
unbiased. Use it as an accelerator once the plain estimator is verified.
