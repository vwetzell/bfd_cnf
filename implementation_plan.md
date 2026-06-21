# Implementation Plan: PSF Conditioning and Selection Corrections

This document translates the findings in `selection_term_analysis.md` into a concrete
set of changes, given the priorities and design choices discussed. It covers three active
items and documents the choices made for the remaining issues.

---

## Context and Design Choices Already Made

| Item | Decision |
|---|---|
| Prior is Σ_X-conditioned | Accepted. The prior learns p_eff(M^G \| g, Σ_X) rather than the strictly Σ_X-independent BFD prior. The size–centroiding cross-term is negligible for weak lensing. |
| Fixed reference Σ_X in `make_flow_prob_and_derivs` | Intentional for flow visualisation. Production inference will use per-target PSF. |
| N_ns selection correction absent | Deferred. To be addressed after PSF incorporation is validated. |
| Issue 5 (Σ_X/selection coupling in ELBO) | Accepted for now; expected to be small given survey PSF uniformity. |

---

## Item 1: Clarify and Fix the Flux Threshold

### The three flux values in the code

| Location | Value | Physical meaning |
|---|---|---|
| `data.py:295` — `bfd.TemplateTable(sampleSpecs={"fluxMin": 1500.0})` | 1500 | Minimum M_f for a galaxy to be **loaded as a template** |
| `data.py:360` — `good_moments &= moments[:, 0] > 1000.0` | 1000 | Secondary quality cut (always passes given fluxMin=1500) |
| `flows.py:588` — `threshold = 1000` in `_log_p_select_given_x_raw` | 1000 | **f_min for the target selection function** used in the ELBO |

### Fix 1a — Make the threshold a named config parameter

The value 1000 in the ELBO is the target f_min — the minimum M_f at which a galaxy
would be detected and included in the target catalogue. It must match the actual
selection applied to the targets. Make it explicit:

```python
# config.py
target_flux_min: float = 1500.0   # f_min for the target selection function (raw M_f units)
                                   # Must match the selection applied to the target catalogue.
                                   # Also used as the template quality cut lower bound.
```

```python
# flows.py — replace the hardcoded 1000
from ..config import target_flux_min
...
log_p_sel = _log_p_select_given_x_raw(mf_true, var_mf[None, :], target_flux_min)
```

```python
# data.py:360 — replace hardcoded 1000 with the same symbol
from .config import target_flux_min
good_moments &= moments[:, 0] > target_flux_min
```

Also check whether `bfd.TemplateTable(fluxMin=1500)` and `target_flux_min` should
agree. If `target_flux_min` is lowered below 1500, the template set needs to extend
there too; if they agree the quality cut in `data.py:360` becomes redundant
(everything already has M_f > fluxMin) but is harmless.

### Fix 1b — Understand and document `var_mf = S_b[:, 0, 0]`

The selection probability in the ELBO is:
```python
# flows.py:585-590
z0_0 = z.reshape(S * BG, -1)[:, 0] * raw2standard.std[0] + raw2standard.mean[0]
mf_true = jnp.power(10.0, z0_0).reshape(S, BG)        # (S, BG)  flux of flow sample
var_mf  = jnp.broadcast_to(
    S_b[:, 0, 0][:, None], (batch_size, G)
).reshape(BG)                                           # (BG,)  template flux variance
log_p_sel = _log_p_select_given_x_raw(
    mf_true, var_mf[None, :], target_flux_min
).reshape(S, batch_size, G)
```

Breaking this down:

- **`mf_true`**: the flux M_f of the **flow sample** z — a hypothetical true galaxy
  flux drawn from the current prior estimate. This is what the Y integral integrates
  over.
- **`var_mf = S_b[:, 0, 0]`**: the (0,0) element of the **template's** measurement
  covariance C_M — the flux variance σ²_f of the template galaxy selected for this
  batch element. This is *not* a fixed global value; it varies across the batch
  because different templates have different noise levels.
- **`_log_p_select_given_x_raw(mf_true, var_mf, f_min)`** computes:
  `log P(M_f_observed > f_min | M_f_true = mf_true, σ²_f = var_mf)`
  = `log(1 − Φ((f_min − mf_true) / σ_f_template))`

**The design intent (training convergence):** By using the per-template noise
rather than a fixed target noise, the selection correction varies continuously
across the batch in proportion to each template's own S/N. For a bright, low-noise
template (σ_f_template small), `P_sel ≈ 1` for any flow sample above the threshold,
so the correction is negligible and the bright template contributes its full gradient
signal. For a template near the flux boundary (σ_f_template comparable to
`mf_true - f_min`), the correction is significant and provides a smooth transition
that prevents the loss surface from having a hard edge. This adaptive softening
can help sparse high-S/N regions of template space converge without the gradient
noise that a hard step function would introduce.

**The physical concern:** Strictly, the Y integral in B16 Eq. (34) uses σ_i for the
**target** observation, not the template. Template galaxies are from deep fields and
have systematically smaller σ_f than targets at the same flux. Using σ_f_template
makes the selection transition sharper than it would be for a real target, which
means the correction is more aggressive (closer to a hard cut) for bright templates
and less conservative near the threshold.

**Recommendation:** The current approach is a reasonable pragmatic choice for
training stability and does not need to change unless you find evidence it is
biasing the learned prior. Document it with a comment in the code:

```python
# var_mf uses the template's own flux variance (S_b[:, 0, 0]) rather than a
# fixed target noise. This makes the selection correction per-template: bright
# low-noise templates get P_sel ≈ 1 (negligible correction); templates near
# the flux threshold get a smooth, gradient-friendly correction. The physical
# target noise would be larger (shallower survey), making the true transition
# wider. If you need to match B16 Eq. (34) exactly, replace with a fixed
# representative target noise (e.g., config.target_flux_var = (f_min/SNR_min)^2).
```

---

## Item 2: Per-Target PSF at Production Inference

### Overview of what needs to change

The prior flow takes a 5-dimensional condition vector:
```
condition = [g1, g2, log_scale, e1, e2]
```

During training (ELBO), `[log_scale, e1, e2]` is randomly sampled from the
training distribution at each gradient step and weighted by L(X^G | C_X).
During inference, the prior currently uses a fixed reference:
```python
# inference.py:75-76
sx_cond = jnp.array([prior_sigmax_log_scale_mean, 0.0, 0.0])
```

For production inference, each target galaxy should use its own `[log_scale, e1, e2]`
derived from its measurement covariance. The chain of quantities is:

```
target's CM_raw  (4×4, from grid catalogue)
        ↓  slice [2:4, 2:4]
      C_X      (2×2, centroid noise covariance)
        ↓  cx_to_sx_cond()
  [log_scale, e1, e2]  (3,)
        ↓  concatenate with [g1=0, g2=0]
   condition    (5,)  fed to prior_flow.log_prob()
        ↓
  log p(z | g=0, Σ_X)   used in RQMC integrand
```

### Step 2a — Verify which covariance block is C_X

**What C_X is:** The 2×2 centroid noise covariance for the centroid moments
X^G = [M_1, M_2] (the last two components of the raw moment vector, stored as
`odd_moments_jnp` in the template data). The covariance of [M_1, M_2] noise under
the PSF gives C_X.

**What `CM_raw` is:** The full 4×4 measurement noise covariance for
[M_f, M_r, M_1, M_2] in raw moment space. The lower-right 2×2 block,
`CM_raw[2:4, 2:4]`, is the noise covariance of [M_1, M_2] — which is C_X.

Before relying on this, verify the moment ordering is [M_f, M_r, M_1, M_2] in
the numpy arrays you have on disk. A quick check:

```python
import numpy as np
grid_gal = np.load(GRID_P_PATH)

# Check one entry — the diagonal of C_X should reflect PSF noise properties.
# For a circular PSF the [0,0] and [1,1] elements should be equal.
# The off-diagonal [0,1] encodes PSF ellipticity (should be ~0 for circular PSF).
sample_cov = grid_gal["covariance"][0].reshape(5, 5)[:4, :4]  # adjust reshape as needed
print("C_X sample:\n", sample_cov[2:4, 2:4])
```

Adjust the slice if the grid data uses a different column structure.

### Step 2b — Add `cx_to_sx_cond` to `models/flows.py`

The existing `_sx_cond_to_CX` (flows.py:305) converts [log_scale, e1, e2] → 2×2 C_X.
Add its inverse next to it:

```python
def cx_to_sx_cond(CX: jax.Array) -> jax.Array:
    """Convert a 2×2 centroid noise covariance C_X to [log_scale, e1, e2].

    Inverse of _sx_cond_to_CX. The parameterisation is:
      C_X = (T/2) * [[1+e1, e2], [e2, 1-e1]]
    where T/2 = exp(log_scale) / sqrt(1 - |e|^2), so:
      log_scale = 0.5 * log det(C_X)
      e1 = (C_X[0,0] - C_X[1,1]) / (C_X[0,0] + C_X[1,1])
      e2 = 2 * C_X[0,1]           / (C_X[0,0] + C_X[1,1])

    Parameters
    ----------
    CX : jax.Array, shape (2, 2)
        Centroid noise covariance (positive definite, symmetric).

    Returns
    -------
    jax.Array, shape (3,)
        [log_scale, e1, e2] condition vector.
    """
    _, logdet = jnp.linalg.slogdet(CX)
    log_scale = 0.5 * logdet
    trace = CX[0, 0] + CX[1, 1]
    e1 = (CX[0, 0] - CX[1, 1]) / jnp.maximum(trace, 1e-30)
    e2 = 2.0 * CX[0, 1]        / jnp.maximum(trace, 1e-30)
    return jnp.array([log_scale, e1, e2])
```

Unit-test the round-trip before using it in production:
```python
# Should recover the original CX to floating-point precision
from models.flows import _sx_cond_to_CX, cx_to_sx_cond
CX_test = jnp.array([[500.0**2, 0.1 * 500.0**2],
                      [0.1 * 500.0**2, 480.0**2]])
sx = cx_to_sx_cond(CX_test)
CX_recovered = _sx_cond_to_CX(sx)
assert jnp.allclose(CX_test, CX_recovered, rtol=1e-4), f"Round-trip failed: {CX_test} vs {CX_recovered}"
```

### Step 2c — Compute sx_conds for all targets before calling `rqmc_pqr_grid`

In `run.py`, after loading the grid data:

```python
# run.py
from models.flows import cx_to_sx_cond

joined_grid = load_grid_data()

# +shear catalogue
CM_raw_all_p = jnp.array([...])  # (N, 4, 4) — per-target raw covariance, +shear
CX_all_p     = CM_raw_all_p[:, 2:4, 2:4]   # (N, 2, 2) — centroid noise block
sx_conds_p   = jax.vmap(cx_to_sx_cond)(CX_all_p)  # (N, 3) = [log_scale, e1, e2]

# -shear catalogue (same procedure)
CM_raw_all_m = jnp.array([...])
CX_all_m     = CM_raw_all_m[:, 2:4, 2:4]
sx_conds_m   = jax.vmap(cx_to_sx_cond)(CX_all_m)

# Diagnostic: check PSF coverage against training range
log_scales_p = sx_conds_p[:, 0]
e_mags_p     = jnp.hypot(sx_conds_p[:, 1], sx_conds_p[:, 2])
print(f"Target log_scale range: [{log_scales_p.min():.2f}, {log_scales_p.max():.2f}]")
print(f"  Training range in config: {log_scale_range}")   # should bracket the above
print(f"Target |e| range:      [{e_mags_p.min():.4f}, {e_mags_p.max():.4f}]")
print(f"  Training e_max in config: {e_max}")             # should be >= the above max
```

If any target log_scale falls outside `config.log_scale_range`, the prior flow is
extrapolating. This needs to be fixed by expanding the training range and retraining
before running production inference.

### Step 2d — Refactor `rqmc_pqr_grid` to accept per-target sx_cond

**The key issue with the current interface:** `flow_prob_and_derivs` and
`log_flow_fn` are currently passed as **static** JIT arguments (they appear in
`static_argnames`). This means JAX treats them as compile-time constants and traces
through the entire grid with a single fixed Σ_X baked into the closures. To use
per-target Σ_X, the flow itself must be a *traced* (non-static) argument so that
`sx_cond` can vary per object inside the vmapped `_single` function.

**Proposed interface change:**

```python
# inference.py
@partial(jax.jit, static_argnames=("n_points", "n_replicates", "batch_size"))
def rqmc_pqr_grid(
    mu_std_all,       # (N, D)  standardised observed moments
    cov_std_all,      # (N, D, D)  standardised covariances
    mu_raw_all,       # (N, D)  raw observed moments (for noise augmentation)
    CM_raw_all,       # (N, D, D)  raw covariances (for noise augmentation)
    sx_conds_all,     # (N, 3)  per-target [log_scale, e1, e2]  ← NEW
    n_points=2**10,
    n_replicates=16,
    batch_size=128,
    *,
    raw2standard,
    prior_flow,       # equinox pytree — passed as traced arg, not static  ← CHANGED
):
```

`prior_flow` is an equinox `Transformed` object (a JAX pytree with frozen
parameters). Passing it as a non-static argument lets JAX trace through it with
variable `sx_cond` while still JIT-compiling the full grid loop.

**Inside `_single`:** build the flow evaluation functions from `prior_flow` and the
per-target `sx_cond` inline, so that `jax.grad` differentiates through the correct
condition for each galaxy:

```python
def _single(mu_std, cov_std, mu_raw, CM_raw, sx_cond):
    # ... existing noise augmentation code (unchanged) ...

    # ── Per-target flow functions ────────────────────────────────────────
    # The 5-dim condition is [g1, g2, log_scale, e1, e2].
    # At g=0 the prior gives log p(z | g=0, Σ_X_target).
    # jax.grad differentiates the log-prob w.r.t. g1 and g2.

    def _log_p_g(x_single, g1, g2):
        cond = jnp.concatenate([jnp.array([g1, g2]), sx_cond])  # (5,)
        return prior_flow.log_prob(x_single, condition=cond)

    _dlp_dg1     = jax.grad(_log_p_g, argnums=1)
    _dlp_dg2     = jax.grad(_log_p_g, argnums=2)
    _d2lp_dg1dg1 = jax.grad(_dlp_dg1, argnums=1)
    _d2lp_dg2dg2 = jax.grad(_dlp_dg2, argnums=2)
    _d2lp_dg1dg2 = jax.grad(_dlp_dg1, argnums=2)

    def flow_prob_and_derivs(x_batch):
        """Returns (log_p, d/dg1, d/dg2, d²/dg1², d²/dg2², d²/dg1dg2) for each row."""
        def _one(xi):
            return (
                _log_p_g(xi, 0.0, 0.0),
                _dlp_dg1(xi, 0.0, 0.0),
                _dlp_dg2(xi, 0.0, 0.0),
                _d2lp_dg1dg1(xi, 0.0, 0.0),
                _d2lp_dg2dg2(xi, 0.0, 0.0),
                _d2lp_dg1dg2(xi, 0.0, 0.0),
            )
        return jax.vmap(_one)(x_batch)  # (N_points, 6)

    def log_flow_fn(x):
        """log p(x | g=0, Σ_X_target) for a single point x, shape (1, D) → (1,)."""
        cond = jnp.concatenate([jnp.zeros(2), sx_cond])
        return prior_flow.log_prob(x[0], condition=cond)[None]

    # ... rest of the RQMC loop — identical to current, using the above closures ...
```

**The vmap over the batch:** `_single` is currently vmapped via:
```python
_single_batch = jax.vmap(_single, in_axes=(0, 0, 0, 0))
```
Add `sx_conds_all` as a batched input:
```python
_single_batch = jax.vmap(_single, in_axes=(0, 0, 0, 0, 0))  # add axis for sx_cond
```

And pass it through the `lax.map` over batches:
```python
batched = jax.lax.map(
    lambda args: _single_batch(*args),
    (mu_std_b, cov_std_b, mu_raw_b, CM_raw_b, sx_conds_b),  # add sx_conds_b
)
```

where `sx_conds_b` is `sx_conds_all` reshaped to `(n_batches, batch_size, 3)`.

**Backward-compatible default (visualisation path):** When `sx_conds_all = None`,
fall back to the reference Σ_X condition as before:

```python
if sx_conds_all is None:
    sx_conds_all = jnp.broadcast_to(
        jnp.array([prior_sigmax_log_scale_mean, 0.0, 0.0]),
        (N, 3)
    )
```

This keeps the existing visualisation behaviour unchanged.

### Step 2e — Verify the full end-to-end condition matches training

A quick sanity check: pick one target, compute its sx_cond, and call the prior
flow directly to confirm the condition vector is in the expected range:

```python
# In a notebook or run.py
sx_one = sx_conds_p[0]   # [log_scale, e1, e2] for first target
print("sx_cond:", sx_one)
print("  log_scale (training range {config.log_scale_range}):", sx_one[0])
print("  e1, e2:", sx_one[1], sx_one[2])

# Evaluate at a representative z point
z_test = jnp.zeros(4)
cond_test = jnp.concatenate([jnp.zeros(2), sx_one])
lp = prior_flow.log_prob(z_test, condition=cond_test)
print("log p(z=0 | g=0, Σ_X_target):", lp)   # should be a finite, reasonable number
```

---

## Item 3: Verify the `odd_moments` / C_X Moment Ordering

The ELBO uses `data_X = odd_moments_jnp` in `_batch_log_L_X(X_b, sx_cond)`,
computing `log N(X^G; 0, C_X)`. The docstring in `load_data` calls these
"first-order Fourier moments (centroid moments)". This needs explicit verification
because the quality cuts in `data.py` treat the same columns as spin-2 shape
moments:

```python
# data.py:362-365  — used as ellipticity: M_+/M_r and M_×/M_r
good_moments &= (
    np.hypot(moments[:, 2] / moments[:, 1], moments[:, 3] / moments[:, 1]) < 0.99
)
```

Dividing by M_r (index 1) and requiring the result to be small (< 0.99) only makes
sense if columns 2 and 3 are the spin-2 shape moments M_+, M_×, not the centroid
moments X. Verify which convention the `bfd` library uses:

```python
print(template_tbl.getMoments().colnames)   # or equivalent for your bfd version
# Expected if centroid moments are separate: ['Mf', 'Mr', 'M+', 'Mx', 'Xx', 'Xy']
# Expected if centroid moments are embedded: check what .getMoments() actually returns
```

If columns 2 and 3 are [M_+, M_×] (shape moments), then:
- The quality cut is correct (ellipticity < 0.99)
- The `odd_moments_jnp` used in the ELBO is **not** the centroid X^G — it is the
  shape ellipticity, and the ELBO centroid weight `log N(X^G; 0, C_X)` is being
  evaluated with the wrong quantity
- Fix: retrieve the true centroid moments (first-order, index 4 and 5 if they exist
  in the BFD output) and use those as `data_X`

If columns 2 and 3 **are** the centroid moments (the BFD library stores them there):
- The quality cut is computing something else but may still be a valid proxy cut
- Confirm that `C_X = CM_raw[:, 2:4, 2:4]` is the noise covariance of these columns

This is the most important low-level check to do before running any of the Item 2
changes, because an incorrect C_X slice would silently produce wrong sx_cond values.

---

## Summary of Changes by File

### `config.py`
- [ ] Add `target_flux_min: float` (check against target catalogue minimum flux)
- [ ] Add comment to `log_scale_range` and `e_max` noting they must bracket target PSF range

### `data.py`
- [ ] Replace hardcoded `1000.0` quality cut with `config.target_flux_min`
- [ ] Add comment explaining the `odd_moments` slice and the intended physical quantity
- [ ] Verify `bfd.TemplateTable(fluxMin=...)` matches `target_flux_min`

### `models/flows.py`
- [ ] Replace `threshold = 1000` with `config.target_flux_min`
- [ ] Add comment on `var_mf = S_b[:, 0, 0]` explaining the per-template noise
      design choice (training convergence, not target noise proxy)
- [ ] Add `cx_to_sx_cond(CX)` function after `_sx_cond_to_CX`
- [ ] Unit-test the round-trip `_sx_cond_to_CX(cx_to_sx_cond(CX)) ≈ CX`

### `inference.py`
- [ ] Add `sx_conds_all` (N, 3) argument to `rqmc_pqr_grid` (default `None` for reference)
- [ ] Remove `flow_prob_and_derivs` and `log_flow_fn` from `static_argnames`
- [ ] Add `prior_flow` as a non-static (traced pytree) argument
- [ ] Build per-target `_log_p_g`, `flow_prob_and_derivs`, `log_flow_fn` inside `_single`
- [ ] Add `sx_conds_b` to the vmap and lax.map argument lists

### `run.py`
- [ ] Verify the moment ordering in the grid data (Step 2a / Item 3)
- [ ] Compute `sx_conds_p` and `sx_conds_m` using `cx_to_sx_cond`
- [ ] Add PSF coverage diagnostic printout before integration
- [ ] Pass `sx_conds_all` to `rqmc_pqr_grid` in production calls

---

## Order of Operations

1. **Verify moment ordering** (Item 3): Confirm what `odd_moments_jnp[:, 2:]` physically
   is in the BFD library output. This underpins everything in Item 2.

2. **Fix the threshold** (Item 1a): Check the minimum flux in the target grid files
   against `target_flux_min`. Update `config.py` and propagate to `data.py` and `flows.py`.

3. **Document `var_mf`** (Item 1b): Add the explanatory comment. No code change required
   unless you decide to switch to target noise for physical accuracy.

4. **Implement `cx_to_sx_cond`** (Step 2b): Add to `models/flows.py`, run the round-trip
   test.

5. **Compute and inspect `sx_conds`** (Step 2c): Run the diagnostic before touching
   inference. If PSF coverage is outside training range, expand and retrain first.

6. **Refactor `rqmc_pqr_grid`** (Step 2d): Thread `prior_flow` and `sx_conds_all` through.
   Confirm `sx_conds_all = None` gives identical output to the current code.

7. **End-to-end sanity check** (Step 2e): Run a single target through the full chain and
   confirm the log-prob is finite and in a reasonable range.
