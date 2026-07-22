# The BFD Flux-Limit Selection Correction, End to End

*A derivation and implementation walkthrough for readers already comfortable with
the Bayesian Fourier Domain (BFD) shear estimator, normalizing flows, and
reparameterized Monte Carlo. The goal is that you could re-derive the code from
this note, or re-derive this note from the code.*

Code entry points referenced throughout:

| Symbol | Location |
|---|---|
| `selection_pqr` | `bfd_cnf/inference.py:139` |
| `selection_pqr_binned` | `bfd_cnf/inference.py:187` |
| `qr_log_totals` | `bfd_cnf/statistics.py:63` |
| `net_qr_totals` / `apply_selection` | `bfd_cnf/statistics.py:107` / `:128` |
| `bootstrap_independent_mult_bias` (selection-aware) | `bfd_cnf/statistics.py:233` |
| `_bins_from_sigma_f` / `_selection_terms` | `apply_selection_correction.py:46` / `:62` |

---

## 0. Notation and conventions

We work with BFD **even moments**. The moment vector the flow models is

$$
x = (M_f,\; M_r,\; M_1,\; M_2) \in \mathbb{R}^4,
$$

flux monopole $M_f$, a size/second-radial moment $M_r$, and the two spin-2
quadrupoles $M_1, M_2$. Only $M_f$ (dimension 0) enters the selection, so the
identities of the other three never matter below — but it is worth keeping in
mind that $M_f$ is **spin-0** while shear $g=(g_1,g_2)$ is **spin-2**. That single
fact drives the whole qualitative structure of the result.

Two "spaces" for the PQR arrays:

- **Probability space**: $P,\;Q=\nabla_g P,\;R=\nabla_g^2 P$ — plain derivatives of
  a probability $P$. This is what `selection_pqr` returns, in flow column order
  $[P,\,Q_1,\,Q_2,\,R_{11},\,R_{22},\,R_{12}]$.
- **Log space** (the accumulation quantities the estimator actually sums):

$$
Q_{\mathrm{tot}} \equiv \nabla_g \log P = \frac{Q}{P},
\qquad
R_{\mathrm{tot}} \equiv -\nabla_g^2 \log P = \frac{Q\otimes Q}{P^2} - \frac{R}{P}.
$$

`qr_log_totals` performs exactly this map (`statistics.py:63`, via `bfd.logPqr`).

The BFD maximum-likelihood shear from a catalogue is

$$
\boxed{\;\hat g = \Big(\textstyle\sum_i R_{\mathrm{tot},i}\Big)^{-1}\Big(\textstyle\sum_i Q_{\mathrm{tot},i}\Big)\;}
\tag{0.1}
$$

with **equal target weight**. A key invariance: any per-object constant prefactor
on $[P,Q,R]$ cancels in $Q_{\mathrm{tot}}, R_{\mathrm{tot}}$, so normalization of the flow prior
is irrelevant to $\hat g$. (This is why we can be cavalier about overall constants
throughout.)

Throughout, $\Phi$ is the standard normal CDF and $\varphi=\Phi'$ its pdf; recall
$\varphi'(u) = -u\,\varphi(u)$.

---

## 1. Why there is a correction at all

### 1.1 Selection multiplies each likelihood by a truncation factor

Suppose we only keep objects whose **measured** flux moment falls in a band
$[f_{\min}, f_{\max}]$. For a single object with data $M_i$, the likelihood of the
data *given that it was selected* is the truncated, renormalized density

$$
p(M_i \mid g,\ \text{selected}) = \frac{p(M_i \mid g)}{P_{\mathrm{sel},i}(g)},
$$

where $P_{\mathrm{sel},i}(g)$ is the probability that an object drawn from the
sheared population passes the cut, evaluated with object $i$'s own noise level.
The denominator is exactly the normalization of the truncated distribution: it is
$g$-dependent because shear reshapes the population and hence changes the fraction
that survives the cut.

Sum the log-likelihood over the selected sample:

$$
\log \mathcal{L}(g) = \sum_i \Big[\log p(M_i\mid g) \;-\; \log P_{\mathrm{sel},i}(g)\Big].
\tag{1.1}
$$

### 1.2 The correction is the selection's own PQR, subtracted per object

Everything downstream follows from differentiating (1.1). Each object contributes
its *measured* log-derivatives **minus** the log-derivatives of its selection
probability:

$$
\nabla_g \log\mathcal{L}
= \sum_i \Big[\,Q_{\mathrm{tot},i} - Q^{\mathrm{sel}}_{\mathrm{tot},i}\,\Big],
\qquad
-\nabla_g^2 \log\mathcal{L}
= \sum_i \Big[\,R_{\mathrm{tot},i} - R^{\mathrm{sel}}_{\mathrm{tot},i}\,\Big],
$$

where $Q^{\mathrm{sel}}_{\mathrm{tot}} = Q^{\mathrm{sel}}/P_{\mathrm{sel}}$ and
$R^{\mathrm{sel}}_{\mathrm{tot}} = Q^{\mathrm{sel}}\!\otimes Q^{\mathrm{sel}}/P_{\mathrm{sel}}^2 - R^{\mathrm{sel}}/P_{\mathrm{sel}}$
are the log-totals of the *selection* probability, computed with the identical
$P,Q,R\!\to\!Q_{\mathrm{tot}},R_{\mathrm{tot}}$ recipe. Setting the gradient to zero and
linearizing gives the corrected estimator

$$
\boxed{\;
\hat g_{\mathrm{sel}}
= \Big(\textstyle\sum_i R_{\mathrm{tot},i} - \sum_i R^{\mathrm{sel}}_{\mathrm{tot},i}\Big)^{-1}
  \Big(\textstyle\sum_i Q_{\mathrm{tot},i} - \sum_i Q^{\mathrm{sel}}_{\mathrm{tot},i}\Big).
\;}
\tag{1.2}
$$

This is precisely `apply_selection` (`statistics.py:128`): compute the arm's
log-totals, subtract the per-object selection log-totals, then do the usual linear
solve. With no selection terms it collapses to (0.1) / `pqr2g`.

**Read (1.2) carefully — three structural consequences you will see mirrored in code:**

1. There is **one selection term per selected object**, not one global term. The
   $\log P_{\mathrm{sel},i}$ in (1.1) depends on object $i$'s noise $\sigma_{f,i}$, so
   the subtraction is per object and carries that object's noise.
2. The correction enters through the **same log-space machinery** as the
   measurement, so it is literally another PQR passed through `qr_log_totals`.
3. Because it is attached per object, a bootstrap resample moves the measured and
   selection sums **together**, preserving their correlation (§5.3).

Everything else in this note is about computing one ingredient: the selection
probability $P_{\mathrm{sel}}(g)$ and its two $g$-derivatives.

---

## 2. The selection probability $P_{\mathrm{sel}}(g)$

### 2.1 Passing probability at fixed true moments

The cut acts on the **measured** flux, which is the true flux moment $M_f(x)$ plus
Gaussian measurement noise of standard deviation $\sigma_f$ (the square root of the
$[0,0]$ entry of the moment covariance, $\sigma_{f,i}=\sqrt{\Sigma_i[0,0]}$).
Conditional on the true moment vector $x$, the probability of landing in the band
is a difference of Gaussian CDFs — the hard band, smeared by noise into a smooth
"box":

$$
s(x) \;=\; \Pr\!\big(f_{\min} \le M_f(x)+\varepsilon \le f_{\max}\big),\quad \varepsilon\sim\mathcal N(0,\sigma_f^2)
$$
$$
\;=\; \Phi\!\Big(\frac{f_{\max}-M_f(x)}{\sigma_f}\Big) - \Phi\!\Big(\frac{f_{\min}-M_f(x)}{\sigma_f}\Big)
\;\in [0,1].
\tag{2.1}
$$

This is the `s` at `inference.py:177`. Note $s$ is bounded and infinitely smooth in
$M_f$ — remember this; it is why §6 can use plain Monte Carlo.

### 2.2 Marginalize over the sheared prior

$P_{\mathrm{sel}}$ is the population-average passing probability under the flow prior
conditioned on shear $g$:

$$
P_{\mathrm{sel}}(g) \;=\; \int p_\theta(x\mid g)\, s(x)\, dx
\;=\; \mathbb{E}_{x\sim p_\theta(\cdot\mid g)}\big[\,s(x)\,\big].
\tag{2.2}
$$

$p_\theta(\cdot\mid g)$ is the trained conditional normalizing-flow prior. The
subtlety that makes this tractable is *how* we take the expectation, which is the
whole content of the next section.

---

## 3. Reparameterized (pathwise) Monte Carlo

### 3.1 Push the $g$-dependence into the integrand

The flow is a bijection $x = T(z; g, s_x)$ with base variable $z\sim\mathcal N(0, I_4)$,
and by construction the pushforward of the base normal *is* $p_\theta(\cdot\mid g)$.
Change variables $x\to z$ in (2.2):

$$
P_{\mathrm{sel}}(g)
= \int \mathcal N(z;0,I)\, s\big(T(z;g,s_x)\big)\, dz
= \mathbb{E}_{z\sim\mathcal N(0,I)}\Big[\, s\big(T(z;g)\big)\,\Big].
\tag{3.1}
$$

The integrating **measure no longer depends on $g$**. All shear dependence now
lives *inside* the integrand, through the map $T$. This is the reparameterization
(a.k.a. pathwise-gradient) trick, and it is what lets us freeze the randomness.

### 3.2 Freeze $z$, then differentiate the estimator directly

Draw a single fixed batch $z_1,\dots,z_N \sim \mathcal N(0,I)$ (`inference.py:171`)
and form

$$
\widehat P_{\mathrm{sel}}(g) = \frac1N \sum_{i=1}^{N} s\big(T(z_i; g)\big),
\qquad z_i \text{ frozen.}
\tag{3.2}
$$

Because the sample points do not move with $g$, we may differentiate under the sum:

$$
\nabla_g \widehat P_{\mathrm{sel}}(g) = \frac1N\sum_i \nabla_g\, s\big(T(z_i;g)\big),
\qquad
\nabla_g^2 \widehat P_{\mathrm{sel}}(g) = \frac1N\sum_i \nabla_g^2\, s\big(T(z_i;g)\big).
\tag{3.3}
$$

These are what `jax.grad(P_sel)` and `jax.hessian(P_sel)` compute at
`inference.py:182-183`.

### 3.3 Why not sample $x\sim p_\theta(\cdot\mid g)$ directly?

If you sampled in data space, $g$ would sit in the *density*, and the gradient would
be the score-function (REINFORCE) estimator

$$
\nabla_g P_{\mathrm{sel}} = \mathbb{E}_{x\sim p_\theta(\cdot\mid g)}\big[\, s(x)\,\nabla_g \log p_\theta(x\mid g)\,\big],
$$

whose variance is typically orders of magnitude larger, and which does not share
random numbers across nearby $g$. The pathwise estimator (3.3) reuses the *same*
$z_i$ for every $g$ (common random numbers), so the finite-sample $P_{\mathrm{sel}}(g)$
is a smooth surface in $g$ and its curvature is not swamped by MC noise. This is the
technical reason the correction can be estimated to the precision that a
second-derivative ($R$) requires.

---

## 4. The chain rule, by hand

`jax` does this automatically, but for an advanced reader the closed form is the
point: it shows *which* flow derivative carries the signal and *why the first-order
term is essentially zero*.

### 4.1 The three composed maps

For a single frozen draw $z_i$, shear enters $s$ through a chain of three maps:

$$
g \;\xrightarrow{\;T\;}\; x_0 = \big[T(z_i;g)\big]_0
\;\xrightarrow{\text{destandardize}}\; M_f = 10^{\,\mathrm{std}_0\, x_0 + \mathrm{mean}_0}
\;\xrightarrow{(2.1)}\; s.
\tag{4.1}
$$

Only dimension 0 (flux) is involved. The middle step inverts the flow's dimension-0
standardizer, which stored $\log_{10} M_f$; hence
$M_f = 10^{\mathrm{std}_0 x_0 + \mathrm{mean}_0}$ at `inference.py:176`. Write
$\lambda \equiv \ln 10$ for brevity.

Define the two standardized band edges
$u_e = (f_e - M_f)/\sigma_f$ for $e\in\{\min,\max\}$, and note that **both edges
depend on $g$ only through $M_f$**, so

$$
\frac{\partial u_e}{\partial g_a} = -\frac1{\sigma_f}\frac{\partial M_f}{\partial g_a}
\quad\text{(same for both edges).}
\tag{4.2}
$$

The flux derivatives, from $M_f = e^{\lambda(\mathrm{std}_0 x_0 + \mathrm{mean}_0)}$:

$$
\frac{\partial M_f}{\partial g_a} = M_f\,\lambda\,\mathrm{std}_0\,\frac{\partial x_0}{\partial g_a},
\tag{4.3}
$$
$$
\frac{\partial^2 M_f}{\partial g_a \partial g_b}
= M_f\,\lambda\,\mathrm{std}_0\Big[\,\lambda\,\mathrm{std}_0\,\frac{\partial x_0}{\partial g_a}\frac{\partial x_0}{\partial g_b}
   + \frac{\partial^2 x_0}{\partial g_a\partial g_b}\,\Big].
\tag{4.4}
$$

Here $\partial x_0/\partial g$ is the flow's Jacobian of the flux output w.r.t. the
shear condition, and $\partial^2 x_0/\partial g^2$ its Hessian — these are the only
places the trained network enters.

### 4.2 First derivative $Q^{\mathrm{sel}}$

Differentiate (2.1) and use (4.2). With the "edge weight"
$D \equiv \varphi(u_{\max}) - \varphi(u_{\min})$:

$$
\frac{\partial s}{\partial g_a}
= \varphi(u_{\max})\frac{\partial u_{\max}}{\partial g_a} - \varphi(u_{\min})\frac{\partial u_{\min}}{\partial g_a}
= -\frac{D}{\sigma_f}\,\frac{\partial M_f}{\partial g_a}
= -\frac{\lambda\,\mathrm{std}_0}{\sigma_f}\, M_f\, D\, \frac{\partial x_0}{\partial g_a}.
\tag{4.5}
$$

Then $Q^{\mathrm{sel}}_a = \frac1N\sum_i \partial s_i/\partial g_a$.

### 4.3 Why $Q^{\mathrm{sel}}\approx 0$: a spin selection rule

Everything in (4.5) is generic except $\partial x_0/\partial g_a$ — the linear
response of the **flux monopole** to shear. Flux is spin-0; shear is spin-2. To
first order a shear reshapes the profile against the fixed isotropic weight, and the
spin-2 perturbation integrates to zero against a spin-0 moment by angular
orthogonality (the area Jacobian of a pure shear is $1 - |g|^2 = 1 - O(g^2)$, so
even the magnification enters only at second order). Hence

$$
\left.\frac{\partial x_0}{\partial g_a}\right|_{g=0} \approx 0
\quad\Longrightarrow\quad
Q^{\mathrm{sel}} = \nabla_g P_{\mathrm{sel}}\big|_{g=0} \approx 0.
$$

The first-order selection term is *spin-forbidden*. This is not a numerical
accident; it is why the correction cannot be a first-order additive shift and must
appear in $R$ (the calibration / multiplicative channel).

### 4.4 Second derivative $R^{\mathrm{sel}}$

Differentiate (4.5) again. Using $\varphi'(u)=-u\varphi(u)$ and defining the
"edge-curvature" $E \equiv u_{\max}\varphi(u_{\max}) - u_{\min}\varphi(u_{\min})$:

$$
\frac{\partial^2 s}{\partial g_a\partial g_b}
= \underbrace{-\frac{E}{\sigma_f^2}\,\frac{\partial M_f}{\partial g_a}\frac{\partial M_f}{\partial g_b}}_{\text{(i) box curvature}}
\;\;\underbrace{-\,\frac{D}{\sigma_f}\,\frac{\partial^2 M_f}{\partial g_a\partial g_b}}_{\text{(ii) 2nd-order flux response}}.
\tag{4.6}
$$

Term (i) is $\propto (\partial M_f/\partial g)^2 \propto (\partial x_0/\partial g)^2$,
which is negligible for the same spin-0 reason that killed $Q$. The surviving signal
is term (ii), and within (4.4) its leading part is the flow **Hessian**
$\partial^2 x_0/\partial g^2$ — the genuine second-order (spin-2 $\times$ spin-2
$\to$ spin-0) response of the flux moment:

$$
R^{\mathrm{sel}}_{ab} \;\approx\;
-\frac{\lambda\,\mathrm{std}_0}{\sigma_f}\,\frac1N\sum_i M_{f,i}\, D_i\,
\frac{\partial^2 x_{0,i}}{\partial g_a \partial g_b}.
\tag{4.7}
$$

This is the nonzero content of the selection correction, and it is exactly what
`jax.hessian` extracts at `inference.py:183`. **Summary of §4:** $Q^{\mathrm{sel}}$ is
spin-forbidden and $\approx 0$; $R^{\mathrm{sel}}$ is allowed and carries the whole
effect — consistent with the empirical finding that the correction lives in $R$, not
in the first-order multiplicative bias.

---

## 5. From one $P_{\mathrm{sel}}(g)$ to a per-object catalogue correction

### 5.1 Binning over $\sigma_f$ (`selection_pqr_binned`, `inference.py:187`)

$P_{\mathrm{sel},i}$ depends on object $i$ through a single scalar, its noise
$\sigma_{f,i}$, which enters **only** the CDF weight (2.1) and *not* the flow
pushforward $T$. Recomputing a full flow pushforward per object would be wasteful, so
we bin: histogram the catalogue's $\sigma_{f,i}$ into a handful of bins, take one
representative $\sigma_f$ per bin, and evaluate `selection_pqr` once per bin
(`inference.py:221-227`). Two design points:

- **Common random numbers across bins.** Every bin reuses the same base draws $z_i$
  (same `key`), so the PQR varies smoothly across the $\sigma_f$ sweep — differences
  between bins reflect the CDF weight, not MC jitter.
- Each bin's PQR is converted to log-totals by `qr_log_totals`, giving
  `q_tot_b (n_bins, 2)` and `r_tot_b (n_bins, 2, 2)` — the per-bin
  $Q^{\mathrm{sel}}_{\mathrm{tot}}, R^{\mathrm{sel}}_{\mathrm{tot}}$.

### 5.2 Attaching a selection term to each object (`_selection_terms`, `apply_selection_correction.py:62`)

`_bins_from_sigma_f` (`:46`) histograms $\sigma_{f,i}=\sqrt{\Sigma_i[0,0]}$ and
returns, row-aligned to the catalogue, each object's bin index `bin_idx`. Each object
then **gathers** its selection log-totals from its own bin:

```python
sel_qtot = q_tot_b[bin_idx]   # (N, 2)
sel_rtot = r_tot_b[bin_idx]   # (N, 2, 2)
```

This realizes the per-object $Q^{\mathrm{sel}}_{\mathrm{tot},i}, R^{\mathrm{sel}}_{\mathrm{tot},i}$ of
(1.2), with object $i$ carrying the selection response evaluated at *its* noise level
(approximated by its bin's representative $\sigma_f$).

### 5.3 The subtraction, and why it is per object (`net_qr_totals` / bootstrap)

`net_qr_totals` (`statistics.py:107`) subtracts the gathered selection log-totals
row-by-row before the sum-and-solve of (1.2). Attaching the term *per object* rather
than as a single precomputed grand sum is what makes the uncertainty correct:
`bootstrap_independent_mult_bias` (`statistics.py:233`, the `_sub_sel` closure at
`:319`) resamples objects, and because each resampled object drags its own selection
term along, the measured sum $\sum Q_{\mathrm{tot},i}$ and the selection sum
$\sum Q^{\mathrm{sel}}_{\mathrm{tot},i}$ co-vary correctly under the resample. A fixed
global selection sum would sever that correlation and misstate the error bar.

### 5.4 The multiplicative bias

Apply (1.2) **per shear arm** (the $\pm$ ring-test arms), then

$$
m = \frac{\hat g_{+}[0] - \hat g_{-}[0]}{\Delta g} - 1,
$$

with and without the selection subtraction to isolate its impact; the driver prints
exactly this $\Delta$ (`apply_selection_correction.py:238-240`).

---

## 6. Numerical choices and their justification

- **Plain MC, no importance sampling / no Pareto-$\hat k$.** The measurement
  integral $P(g)=\int p_\theta(x\mid g)\,\mathcal N(x;M,\Sigma)\,dx$ needs importance
  sampling and PSIS diagnostics because its integrand (a narrow Gaussian kernel) is
  peaked and heavy-tailed in weight. The **selection** integrand $s\in[0,1]$ is
  bounded and smooth (2.1), so the estimator variance is a benign $O(1/N)$ with no
  tail — a raw mean over $N=2^{17}$ prior draws suffices (`inference.py:147`,
  `:165`).
- **Evaluate at $g_0=0$** (`inference.py:180`). The BFD estimator is a linearization
  about the unsheared fiducial; $P_{\mathrm{sel}}, Q^{\mathrm{sel}}, R^{\mathrm{sel}}$ are the
  Taylor coefficients there.
- **$s_x$ held fixed** at $[\,\texttt{prior\_sigmax\_log\_scale\_mean},0,0\,] \approx [11.98, 0, 0]$ (`inference.py:168`, `config.py:48`): the non-shear conditioning
  (galaxy size scale and the two centroid-covariance ellipticity components) is pinned
  to one representative value rather than marginalized over its full distribution. An
  approximation, and a natural place to check sensitivity if the correction ever
  matters at the $10^{-3}$ level.
- **One representative $\sigma_f$ per bin.** With enough bins this is accurate because
  $P_{\mathrm{sel}}$ is smooth in $\sigma_f$; the `ponytail:` note at `inference.py:212`
  records the obvious speedup (hoist the $g$-Jacobian/Hessian of $M_f(z,g)$ out of the
  loop, since $\sigma_f$ only enters the CDF weight) if the bin count ever grows.

---

## 7. Sanity checks worth running

1. **$Q^{\mathrm{sel}}\to 0$.** Print $Q^{\mathrm{sel}}$ from `selection_pqr`; both
   components should be statistically consistent with zero at reasonable $N$. A
   nonzero, $N$-stable $Q^{\mathrm{sel}}$ would signal either a spin leak in the flow
   or a bug in the destandardization (4.1).
2. **Band limits.** As $[f_{\min}, f_{\max}]$ widens to cover the whole flux support,
   $s\to 1$, so $P_{\mathrm{sel}}\to 1$ and $Q^{\mathrm{sel}}, R^{\mathrm{sel}}\to 0$ — the
   correction must vanish when nothing is cut.
3. **Common-random-number smoothness.** Sweep $\sigma_f$ and confirm the per-bin PQR
   is smooth (no MC hash); if it is jagged, the shared `key` was lost.
4. **Reduction to `pqr2g`.** `apply_selection(pqr, None, None)` must equal
   `pqr2g(pqr)` bit-for-bit (`statistics.py:136`).
5. **Finite-difference the flow response.** Cross-check `jax.hessian`'s
   $\partial^2 x_0/\partial g^2$ against a central difference of the flow output — the
   $R^{\mathrm{sel}}$ signal (4.7) rides entirely on this term.

Automated coverage lives in `tests/test_selection_pqr.py` and
`tests/test_pqr_bfd.py`.

---

## 8. One-paragraph summary

Selection by a measured-flux band divides each object's likelihood by a
$g$-dependent survival probability $P_{\mathrm{sel},i}(g)$ (1.1); differentiating turns
that division into a per-object subtraction of the selection's own log-space PQR from
the measurement's (1.2). $P_{\mathrm{sel}}(g)$ is the prior-averaged
difference-of-Gaussian-CDFs (2.1)–(2.2), estimated by reparameterized MC with frozen
base draws so that $g$-derivatives are exact pathwise gradients (§3). The chain rule
(§4) shows the first derivative is spin-forbidden ($Q^{\mathrm{sel}}\approx 0$, flux is
spin-0) and the whole effect rides on the flow's second-order flux response
$\partial^2 x_0/\partial g^2$ inside $R^{\mathrm{sel}}$. The correction is binned over
$\sigma_f$, gathered onto each object by its noise bin, and subtracted per object so
the bootstrap stays correct (§5).
