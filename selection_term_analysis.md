# Selection Bias in BFD and the `SigmaXCouplingLayer`

A pedagogical analysis of how the BFD paper's selection correction maps (or fails to map)
onto the current `bfd_cnf` implementation.

---

## 1. Why This Document Exists

The `SigmaXCouplingLayer` in `models/bijections.py` is described as accounting for the
"selection term" in BFD. After reading both the paper (Bernstein et al. 2016, MNRAS 459,
4467; hereafter **B16**) and the code carefully, there is a mismatch worth understanding.
The layer is a defensible architectural choice, but it is **not** what B16 calls the selection
term — and the correction B16 actually requires is absent from the codebase entirely.

This document builds up the BFD selection formalism from scratch, then traces exactly where
the code diverges from it, with explicit equation references throughout.

---

## 2. The BFD Framework — A Quick Orientation

BFD does not fit galaxy shapes with a model. Instead, it compresses each pixel image **D**
into a vector of Fourier-domain moments **M** = [M_f, M_r, M_+, M_×] and works entirely in
moment space. The central object is the probability of observing moments **M** given lensing
shear **g**:

```
P(M | g) = L(M^n) = L[M − M^G(g, x_G)]                             (B16 Eq. 6)
```

where M^G(g, x_G) are the noiseless template moments of galaxy G shifted by lensing g, and
L denotes the measurement noise likelihood (a Gaussian with covariance C_M). Because this
is only second-order in **g**, BFD Taylor-expands it (B16 Eq. 10):

```
P(M | g) ≈ P + Q · g + ½ g · R · g
```

with scalar P, vector Q = ∇_g P|_{g=0}, and matrix R = ∇_g∇_g P|_{g=0}.
Summing over many detected galaxies yields the shear maximum-likelihood estimate
(B16 Eqs. 16–19):

```
Q_tot = Σ_i  Q_i / P_i
R_tot = Σ_i (Q_i Q_i^T / P_i^2 − R_i / P_i)
ĝ     = R_tot^{−1} Q_tot
```

**This is the no-selection case.** It assumes every galaxy in the survey is always detected.
The rest of B16 §2.2 derives what changes when detection is probabilistic.

---

## 3. The Selection Problem — Built Up Carefully

### 3.1 The Detection Mechanism

BFD adds a second moment vector **X** (the "centroid moments," B16 Eq. 20):

```
X(x_0) = ∫ d²k  T̃^o(k; x_0)  W(|k|²)  [ik_x, ik_y]^T
```

**X** measures the Fourier-weighted displacement of the galaxy's flux centroid from the
trial position x_0. The detection criterion is: find the position x_0 near the galaxy
centre where **X(x_0) = 0** (B16 §2.2). The utility of **X** comes from two properties:

1. **dM_f / dx_0 = X** — the gradient of the flux moment with respect to trial position
   equals **X** (B16 Eq. 21).
2. **Cov(M, X) = 0** for stationary noise (B16 Eq. 22) — the moment and centroid noise
   are uncorrelated.

From these, the Jacobian of the position-to-detection mapping is:

```
J = |dX/dx_0| = (M_r² − M_+² − M_×²) / 4 = M^T B M          (B16 Eqs. 23–24)
```

with **B** = diag(0, 1/4, −1/4, −1/4). This J appears in every selection probability
expression going forward. Notice that **J depends on the moments M themselves**, not just
the template moments — this matters.

The code has `_moment_jacobian_det` in `models/flows.py:246`:
```python
def _moment_jacobian_det(m_raw):
    Mr, Mp, Mx = m_raw[1], m_raw[2], m_raw[3]
    return jnp.maximum((Mr**2 - Mp**2 - Mx**2) / 4.0, 1e-10)
```
This matches B16 Eq. (23). The `B_RAW_JNP` matrix in `config.py:82` also matches B16 Eq. (24).

### 3.2 The Selection Criterion

B16 Eq. (28) defines selection as membership in the set S of moment space:

```
S :  f_min < f < f_max                                               (B16 Eq. 28)
```

where f = M_f is the flux moment. This is a flux-only cut (at fixed PSF; more general cuts
are discussed but not implemented). The selection criterion picks galaxies bright enough to
be detected above noise but below the saturation/blending limit.

### 3.3 The Per-Detection Probability

For a known galaxy G at position u = x_G − x_0 relative to the detection point, the
joint probability of **observing moments M and making a detection** is (B16 Eq. 29):

```
P(M, s | G, g, x_G, x_0) = L(M − M^G) · L(X^G) · |J| · Δ²x     if M ∈ S
                           = 0                                       if M ∉ S
```

Three factors:
- **L(M − M^G)**: the moment likelihood — how well the observed moments match template G
- **L(X^G) = N(X^G; 0, C_X)**: the centroid likelihood — how probable is this template's
  centroid offset under PSF noise covariance C_X (= **Σ_X** in the code)
- **|J|**: the Jacobian of the detection mapping

**This is the first and primary place where Σ_X enters the BFD formalism.** The term
L(X^G) = N(X^G; 0, C_X) directly depends on the PSF noise covariance C_X. Larger or more
elliptical PSFs make X^G larger in variance, changing how templates are weighted.

Marginalising over the detection position u gives the **selection probability for one
galaxy** (B16 Eq. 30):

```
P(s | G, g, u) = L(X^G) Δ²x  ∫_{M∈S}  dM  L(M − M^G)  |J(M)|    (B16 Eq. 30)
```

### 3.4 The Y Integral and the Convex-Galaxy Approximation

The integral in Eq. (30) is hard because |J| depends on M through the noisy moments.
B16 uses the "convex galaxy" approximation (valid when f_min ≳ 5σ_f): J is positive
everywhere in S, so the integration can be performed analytically (B16 §5.3).

Under this approximation, J(M) is expanded around M^G to give (B16 Eq. 31):

```
J = J^G + 2(M^G)^T B M^n + M^n^T B M^n > 0
```

where M^n is the noise contribution. Substituting into Eq. (30) and integrating, the
result factorises (B16 Eq. 33):

```
P(s | G, g, u) = L(X^G) Δ²x × [J^G Y + 2(C_M B M^G)_f ∂Y/∂f_G + (C_M B C_M)_ff ∂²Y/∂f_G²]
```

where Y is the flux-selection integral (B16 Eq. 34):

```
Y = (2π)^{−1/2} ∫_{(f_min−f_G)/σ_i}^{(f_max−f_G)/σ_i}  dv  e^{−v²/2}
  = Φ((f_max − f_G)/σ_i) − Φ((f_min − f_G)/σ_i)
```

where Φ is the normal CDF and σ_i² = (C_M)_{ff} is the flux moment variance.

**Key point**: Y depends on the actual selection thresholds f_min, f_max and on σ_i
from the measurement noise, but **not directly on Σ_X**. The Σ_X dependence of
the full expression comes entirely through the prefactor **L(X^G) = N(X^G; 0, C_X)**.

### 3.5 Summing Over the Galaxy Population

Summing over all template galaxies G with prior weight p_G gives the full-population
detection and moment probabilities (B16 Eqs. 38 and 40):

```
P(M_i, s | g) = J(M_i) Σ_{G,u} p_G Δ²u  L(X^G)  L(M_i − M^G)    (B16 Eq. 38)
```

```
P(s | g)      = Σ_{G,u} p_G Δ²u  L(X^G)
               × [J^G Y + 2(C_M B M^G)_f ∂Y/∂f_G + (C_M B C_M)_ff ∂²Y/∂f_G²]
                                                                     (B16 Eq. 40)
```

Notice that in Eq. (38), `P(M_i, s | g)` is the probability of *both* detecting a
galaxy *and* observing moments M_i. This is the P_i that appears in the BFD estimator.

Then both are Taylor-expanded in g around g = 0 (B16 Eqs. 41–42):

```
P(M_i, s | g) ≈ P_i + Q_i · g + ½ g · R_i · g                     (B16 Eq. 41)
P(s | g)       ≈ P_s + Q_s · g + ½ g · R_s · g                     (B16 Eq. 42)
```

where Q_s = ∇_g P(s | g)|_{g=0} is the shear derivative of the *total* detection
probability, and R_s its second derivative.

### 3.6 The Full Shear Estimator With Selection Correction

B16 shows that the likelihood for the full data, including both selected and non-selected
stamps, leads to the **corrected** shear estimator (B16 Eqs. 45–46):

```
Q_tot = Σ_i  Q_i/P_i  −  N_ns × Q_s / (1 − P_s)                  (B16 Eq. 45)

R_tot = Σ_i (Q_i Q_i^T / P_i^2 − R_i / P_i)
      + N_ns × (Q_s Q_s^T / (1−P_s)^2 + R_s / (1−P_s))            (B16 Eq. 46)
```

where **N_ns** is the number of non-selected (rejected) stamps.

Compare these to the no-selection case (B16 Eqs. 16–17):
```
Q_tot = Σ_i  Q_i / P_i                             ← no N_ns correction
R_tot = Σ_i (Q_i Q_i^T / P_i^2 − R_i / P_i)       ← no N_ns correction
```

The correction terms subtract out the contribution that *would* be expected from the
N_ns galaxies that were not selected. Intuitively: if a galaxy is not selected, it was
probably faint, and faint galaxies have different shapes on average (because they are
further away, more intrinsically compact, etc.). Ignoring them biases the shape estimate.

**The paper explicitly quantifies this bias.** B16 §4.2 (bottom of page 4475) reports:

> "If we omit the selection terms in equations (45) and (46), we obtain
> m = −0.0122 ± 0.0004."

Compared to the corrected result m = (+2.1 ± 0.4) × 10⁻³. That is a ~1.2% bias from
omitting the selection correction — roughly 20× larger than the target accuracy.
**This is the selection term.** It lives in the statistics layer, not in the flow architecture.

---

## 4. What `SigmaXCouplingLayer` Actually Does

### 4.1 The Architecture

`SigmaXCouplingLayer` (`models/bijections.py:875`) adds a final bijection to the prior
flow that conditions on `[log_scale, e1, e2]` — a compact parameterisation of the PSF
noise covariance Σ_X:

- `log_scale = 0.5 × log det(Σ_X)` — controls the overall PSF noise level
- `e1, e2` — PSF ellipticity components

The layer transforms galaxy moments in two blocks:

**Spin-0 block** (M_f, M_r — flux and size):
The two components are scaled by networks conditioned on `log_scale_n`:
```python
ls0 = _bounded_log_scale(self.net_s0(jnp.array([log_scale_n]))[0])
ls1 = _bounded_log_scale(self.net_s1(jnp.array([x0, log_scale_n]))[0])
y0 = x[0] * exp(-ls0)
y1 = x[1] * exp(-ls1)
```

**Spin-2 block** (M_+, M_× — ellipticity moments):
Transformed by the linear map A = s·I + c·E where E = [[e1,e2],[e2,−e1]] — a matrix
that specifically encodes the *orientation* of the PSF ellipticity:
```python
y2 = s * x[2] + c * (e1 * x[2] + e2 * x[3])
y3 = s * x[3] + c * (e2 * x[2] - e1 * x[3])
```

The comment in the code explains the design rationale for A (the `FIX:` note at
`bijections.py:965–975`): the coupling must be first-order in ellipticity, not
second-order, so that PSF asymmetries actually affect the learned distribution.

### 4.2 What the ELBO Trains the Prior to Learn

The ELBO in `models/flows.py` (the `_elbo_for_sx` branch, lines 603–617) trains the
prior by:

1. Drawing N_sx = 8 different Σ_X conditions per gradient step
2. For each Σ_X condition, computing the centroid likelihood weight
   `log_w_b = log N(X^G; 0, C_X)` — exactly L(X^G) from B16 Eq. (29)
3. Evaluating the prior log-probability at each condition
4. Averaging the ELBO across conditions, weighted by `log_w_b`

The averaging step (lines 626–628) performs a self-normalised importance-sampling
marginalisation over Σ_X:

```python
marginal_lse_b = logsumexp(log_w_all + lse_all, axis=0) - logsumexp(log_w_all, axis=0)
```

This corresponds to estimating:
```
log E_{Σ_X}[L(X^G | C_X) × exp(ELBO at Σ_X)] / E_{Σ_X}[L(X^G | C_X)]
```

So the prior is being trained to maximise the centroid-likelihood-weighted ELBO across
a distribution of PSF conditions. This means the flow is learning an **effective prior**:

```
p_eff(M^G | g, Σ_X) ∝ p_true(M^G) × L(X^G | C_X) × P(sel | M^G, Σ_X)
```

rather than the true galaxy population prior p_true(M^G), which in the BFD formalism
is **Σ_X-independent**. This is a reasonable choice for RQMC variance reduction —
sampling from a distribution that down-weights templates that would have negligible
centroid probability under the actual PSF — but it is an approximation.

Also included in the ELBO (lines 592–593):
```python
log_p_y_minus_sel = log_p_y - log_p_sel
```

where `log_p_sel` is:
```python
def _log_p_select_given_x_raw(mf_true, var_mf, threshold):
    sigma = jnp.sqrt(jnp.maximum(var_mf, 0.0) + 1e-12)
    return jstats.norm.logsf((threshold - mf_true) / sigma)
```
called with `threshold = 1000` (line 588). This is log(1 − Φ((f_min − M_f)/σ_f)) —
the log probability that the true flux exceeds the threshold, which corresponds to
the Y integral from B16 Eq. (34) in the limit f_max → ∞.

---

## 5. Comparing Code to Paper — Issue by Issue

### Issue 1 (Critical): The N_ns Correction Is Absent

**What the paper requires:** B16 Eqs. (45–46) add correction terms proportional to
N_ns (the number of non-selected stamps) that subtract out the expected contribution
from galaxies that *could have been* detected but were not. Without these terms, the
shear estimator is biased at the ~1.2% level.

**What the code does:** `statistics.py:pqr2g` (lines 51–67):

```python
Q_tot = jnp.nansum(pqr_arr[:, 1:3] / pqr_arr[:, 0][..., None], axis=0)
R_tot = jnp.nansum(
    pqr_arr[:, 1:3][..., :, None] * pqr_arr[:, 1:3][..., None, :]
    / pqr_arr[:, 0][..., None, None] ** 2
    - R_arr / pqr_arr[:, 0][..., None, None],
    axis=0,
)
```

This is exactly B16 Eqs. (16–17) — the **no-selection** case. There are no N_ns,
Q_s, or R_s terms anywhere in `statistics.py`. The `bootstrap_total_mult_bias`
function (lines 218–248) also uses the same uncorrected formula.

**The consequence:** The code is currently estimating the shear as if no selection
had occurred, using a prior that was trained to partially account for selection via
the ELBO, but the statistical estimator itself does not include the non-selection
correction. The ~1.2% bias from B16 §4.2 should therefore be expected to appear in
any multiplicative bias measurement from this code.

To apply the correction, you would need:
- N_ns: the count of stamps that failed the selection cut
- Q_s and R_s: the shear derivatives of the detection probability P(s|g=0),
  computed from B16 Eq. (40) marginalised over the template set

### Issue 2 (Critical): The Flux Threshold Is Hardcoded

**What the paper requires:** The Y integral (B16 Eq. 34) depends explicitly on
f_min and f_max — the actual flux selection boundaries used in the catalogue.
These must match the cuts applied when generating the target galaxy catalogue.

**What the code does:** In `models/flows.py:588`:
```python
log_p_sel = _log_p_select_given_x_raw(mf_true, var_mf[None, :], 1000)
```

The threshold `1000` is hardcoded. There is no `f_max` (the upper cut is effectively
∞, which is reasonable if there is no saturation cut). But `f_min = 1000` in
whatever units M_f is measured in must match the selection cut applied in the
template catalogue generation. If the catalogues use a different flux threshold —
for example, a threshold in units of SNR rather than raw flux, or a different
absolute value — then the selection correction in the ELBO is wrong.

**Where to check:** The FITS template catalogue path is `config.py:72`:
```python
FITS_PATH = "/home/vwetzell/Documents/BFD_cNF/tmpl_t04_joined.fits"
```
Verify what flux units M_f is stored in and what f_min was applied when detecting
sources in this catalogue.

### Issue 3 (Architectural): The Prior Is Σ_X-Dependent

**What the paper says:** In B16's formalism, the prior p(M^G) is the distribution
of *true, noiseless, unlensed* galaxy moments. This quantity describes the galaxy
population and does not depend on the PSF or observational conditions (Σ_X).
The Σ_X dependence enters only through L(X^G | C_X) — the centroid likelihood
weight — which is a separate factor in B16 Eqs. (35–38).

**What the code does:** The prior flow takes condition `[g1, g2, log_scale, e1, e2]`,
making p(z | g, Σ_X) explicitly Σ_X-dependent. As described in §4.2 above, the
flow is learning the centroid-likelihood-reweighted effective prior rather than the
true galaxy prior.

**Is this a problem?** It is a well-motivated approximation. The effective prior
p_eff(M^G | Σ_X) ∝ p(M^G) × L(X^G | C_X) correctly downweights templates with
small centroid probability, which reduces RQMC variance. However:

1. The flow's shear derivatives ∂ log p/∂g encode both the galaxy population's
   response to shear **and** the Σ_X-dependent centroid weight's implicit shear
   dependence. If L(X^G | C_X) has a nonzero shear gradient (e.g., because shear
   changes galaxy sizes and hence centroid localisability), this bleeds an extra
   term into Q_i = ∇_g log P_i that is not part of the true shear signal.
   Whether this effect is significant requires numerical investigation.

2. At inference time (see Issue 4), the prior is evaluated at a *fixed* reference
   Σ_X, not at the actual Σ_X for each target. This means the Σ_X-dependence that
   was trained into the prior via the ELBO is not used consistently.

**A note on what this means for the selection term:** The BFD selection correction
(B16 Eqs. 45–46) requires Q_s and R_s — the derivatives of the *detection probability*
P(s|g=0) with respect to shear. These should be computed from B16 Eq. (40), which
involves the full template integral with J(M^G), Y, and centroid weights. The prior
flow as currently designed does not output P(s|g) directly; it outputs a reweighted
density that conflates detection probability with galaxy population density. To
compute Q_s and R_s correctly, a separate integration over the template set would
be needed using B16 Eq. (40).

### Issue 4 (Moderate): Inference Uses a Fixed Reference Σ_X

**What the paper says:** B16 Eq. (38) shows that P(M_i, s | g) requires summing
L(X^G | C_X) over templates, where C_X is the actual PSF noise covariance for each
target observation. If different targets have different PSF conditions (different
exposures, different sky positions), their C_X values differ, and the integral should
use the target-specific C_X.

**What the code does:** In `inference.py:75–76`:
```python
if sx_cond is None:
    sx_cond = jnp.array([prior_sigmax_log_scale_mean, 0.0, 0.0])
```

`prior_sigmax_log_scale_mean = 2.0 × log(400.0) ≈ 11.98` from `config.py:46`.
This is a circular PSF at a fixed noise scale. Every target galaxy is evaluated
against the same reference PSF, regardless of its actual observing conditions.

**When this is acceptable:** If all targets share the same PSF (single-epoch,
uniform observing conditions), this is exact. For multi-epoch data or surveys with
spatially varying PSF, this is a systematic approximation.

**What would be needed:** Each target's actual `[log_scale, e1, e2]` should be
computed from its measurement covariance C_M and passed to `make_flow_prob_and_derivs`.
Alternatively, the inference step could marginalise over Σ_X using the same
importance-sampling scheme used in the ELBO.

### Issue 5 (Subtle): The ELBO's Selection Correction and Σ_X Are Not Fully Coupled

**The BFD factorisation:** In B16 Eq. (29), the per-detection probability is:
```
P(M, s | G, g, x_G, x_0) = L(M − M^G) × L(X^G) × |J| × [M ∈ S indicator]
```

The Σ_X dependence is in L(X^G) = N(X^G; 0, C_X). The selection indicator [M ∈ S]
depends on M_f compared to f_min, which depends on σ_i = sqrt((C_M)_{ff}) — the
measurement noise, not Σ_X directly. So in the BFD formalism, the selection
probability and the centroid weight are **coupled**: when Σ_X is large (bad PSF),
L(X^G) is wider, and the effective C_M for detected galaxies also changes (because
worse PSF means larger moment noise). The full P(s | g, Σ_X) from B16 Eq. (40)
accounts for this coupling through the J^G Y term.

**What the code does:** The ELBO separates these into two independent terms:
```python
log_p_y_minus_sel = log_p_y - log_p_sel          # selection: fixed threshold, no Σ_X
...
log_w_b = _batch_log_L_X(X_b, sx_cond)           # centroid: Σ_X-dependent
```

`log_p_sel` uses `var_mf = S_b[:, 0, 0]` — the (0,0) element of the measurement
covariance — but this covariance is the *training template* covariance, not the
Σ_X-dependent C_M that would appear in a real observation. The Σ_X conditioning
of the covariance is not propagated into the selection probability.

**How large is this error?** The correction terms in B16 Eq. (33) — the Y derivatives
— are proportional to σ_i, which scales with PSF size. For large variations in Σ_X
(the training range spans log-scale 11–13, a factor of ~e² ≈ 7 in noise variance),
this cross-coupling could produce a percent-level effect on Q_s. This is speculative
without numerical experiments, but worth being aware of.

---

## 6. What Would Need to Change to Match the Paper

Listed in order of estimated impact:

### Step 1: Implement the N_ns correction in `statistics.py`

This is the most impactful missing piece. To implement B16 Eqs. (45–46), you need:

**a) Count N_ns per catalogue:**
N_ns is the number of stamps that failed the selection cut (flux below f_min or above
f_max). This must be tracked during the target catalogue generation and stored
alongside the PQR arrays.

**b) Compute Q_s and R_s:**
These are the shear gradient and curvature of the detection probability P(s|g=0)
from B16 Eq. (40), marginalised over the template population:
```
Q_s = ∇_g P(s|g)|_{g=0} = Σ_{G,u} p_G Δ²u L(X^G) × [J^G ∂Y/∂g + ...]
```
This requires evaluating derivatives of Y with respect to the shear through the
dependence of M^G on g (the template moments shift with shear). The `dm_dg` arrays
in the training data already contain ∂M^G/∂g, so the derivatives of f_G and hence
Y can be computed.

**c) Modify `pqr2g` to accept N_ns, Q_s, R_s and apply the correction:**
```python
# B16 Eq. (45)
Q_tot = Σ Q_i/P_i  −  N_ns * Q_s / (1 − P_s)
# B16 Eq. (46)
R_tot = Σ(Q_iQ_i^T/P_i^2 − R_i/P_i) + N_ns * (Q_sQ_s^T/(1−P_s)^2 + R_s/(1−P_s))
```

### Step 2: Verify or fix the flux threshold

Check that `threshold = 1000` in `flows.py:588` matches the actual f_min used
in the BFD catalogue generation. If not, update it (or better, read it from a
config parameter rather than hardcoding it).

### Step 3: Pass actual Σ_X per target at inference time

If the data has per-target PSF information, compute `[log_scale, e1, e2]` for
each target and pass it to `make_flow_prob_and_derivs` instead of using the default
reference condition.

### Step 4 (Optional): Decouple the true prior from the Σ_X reweighting

If the goal is to have the prior represent the true galaxy population p(M^G) rather
than the observation-weighted effective prior, the Σ_X conditioning should be moved
outside the flow and computed analytically using L(X^G | C_X). This would allow the
flow to learn p(M^G | g) (Σ_X-independent) and the RQMC integral to use explicit
L(X^G) importance weights — matching the B16 Eq. (38) factorisation exactly.
Whether this is worth the complexity increase depends on how much the Σ_X variation
matters for the specific survey.

---

## 7. What to Check First

Before changing anything, run two diagnostics:

1. **Compute multiplicative bias with the current code and compare to B16 §4.2.**
   If you get m ≈ −1.2% (as the paper predicts for the no-selection-correction case),
   that confirms Issue 1 is the dominant problem.

2. **Compare the flux threshold to the catalogue f_min.**
   Open the FITS catalogue and check the flux distribution near the selection boundary.
   If the minimum flux is ~1000 in M_f units, the threshold is correct. If the
   selection was applied in SNR or different units, the threshold needs updating.

---

## 8. Summary Table

| Requirement from B16 | Code status | Relevant code | Paper equation |
|---|---|---|---|
| L(X^G\|C_X) centroid weight in ELBO | ✓ Present | `flows.py:606` | Eq. (29) |
| Flux selection P(s\|M_f) in ELBO | ✓ Present (but threshold hardcoded) | `flows.py:588` | Eq. (34) |
| Σ_X-conditioned prior flow | ✓ Present (but conflates prior with centroid weight) | `bijections.py:875` | Eq. (38) |
| **N_ns × Q_s/(1−P_s) correction in Q_tot** | **✗ Absent** | `statistics.py:57` | **Eq. (45)** |
| **N_ns correction in R_tot** | **✗ Absent** | `statistics.py:58–64` | **Eq. (46)** |
| Per-target Σ_X at inference | ✗ Fixed reference PSF | `inference.py:76` | Eq. (38) |
| f_min matches catalogue | ❓ Hardcoded 1000, verify | `flows.py:588` | Eq. (28) |

The highest-priority fix is implementing the N_ns correction in `statistics.py`.
That single change addresses the dominant selection bias identified by B16.
