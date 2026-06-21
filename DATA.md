# BFD cNF Data Files Reference

## Scientific Background

**BFD** (Bayesian Fourier Domain; Bernstein & Armstrong 2014, arXiv:1304.1843; Bernstein et al.
2016, arXiv:1508.05655) estimates weak-lensing shear by compressing galaxy pixel data into a
Fourier-domain moment vector. The full 5-component basis is **M** = [Mf, Mr, M+, Mx, Mc]:

| Moment | Description |
|--------|-------------|
| **Mf** | Flux moment (zeroth order) |
| **Mr** | Radial / size moment |
| **M+, Mx** | Ellipticity moments (spin-2 components) |
| **Mc** | 4th-order concentricity moment, needed to estimate magnification (μ) |

Shear and magnification response is encoded via the **PQR** formalism. The full basis includes
magnification terms: P, Q1, Q2, Q_μ (3 first-derivative terms) and R11, R12, R22, R1μ, R2μ, Rμμ
(6 second-derivative terms) — 10 elements total.

**This project drops magnification.** Mc did not improve shear recovery on this dataset, so the
cNF pipeline works with the 4-moment basis [Mf, Mr, M+, Mx] and the reduced 6-element PQR
[P, Q1, Q2, R11, R12, R22]. The 5th moment and magnification PQR/derivative terms are present in
the raw data files (computed during the original BFD run) but are discarded at load time.

Shear is estimated as **g = R⁻¹ · ΣQ / ΣP** summed over galaxies.

**cNF** replaces the original analytic BFD moment-space priors with trained conditional normalizing
flows, improving multiplicative shear bias. DES (Dark Energy Survey) Deep Field and COSMOS galaxies
serve as the template population representing the true galaxy distribution.

All data files live in `/home/vwetzell/Documents/BFD_cNF/`.

---

## 1. Shear Simulation Grids

### `merged_masked_bfd_grid_p.npy` (~1.6 GB) · `merged_masked_bfd_grid_m.npy` (~1.6 GB)

NumPy structured arrays (record arrays) holding DES galaxy catalogues from shear simulations:
- `_p` — +shear simulation (5,432,002 galaxies)
- `_m` — −shear simulation (5,364,752 galaxies)

These are used together: joined on `id` for shear calibration via the `±g` pair.

**Fields:**

| Field | dtype | shape | Description |
|-------|-------|-------|-------------|
| `id` | int64 | scalar | Galaxy identifier |
| `ra` | float64 | scalar | Right ascension (degrees) |
| `dec` | float64 | scalar | Declination (degrees) |
| `psf_moments` | float32 | (4,) | PSF Fourier-domain moments |
| `psf_hsm_moments` | float32 | (3,) | PSF moments from the GalSim HSM algorithm |
| `Mf_per_band` | float32 | (4,) | Per-band flux moments (4 DES bands: g,r,i,z) |
| `cov_Mf_per_band` | float32 | (4,) | Variance of Mf in each band |
| `mfrac_per_band` | float32 | (4,) | Masked pixel fraction per band |
| `bkg` | float64 | scalar | Background level |
| `tilename` | string | 12-char | DES coadd tile name |
| `dered_corr` | float64 | (4,) | Dereddening correction per band |
| `moments` | float32 | (5,) | BFD moments [Mf, Mr, M+, Mx, Mc]; Mc is dropped in the cNF pipeline |
| `covariance` | float32 | (15,) | Lower-triangular packing of 5×5 moment covariance (Mc row/col dropped in pipeline) |
| `pqr` | float32 | (10,) | PQR statistics [P, Q1, Q2, Q_μ, R11, R12, R22, R1μ, R2μ, Rμμ]; magnification terms dropped in pipeline |
| `select` | int8 | scalar | Selection flag (1 = passes cuts) |
| `tierNumber` | int16 | scalar | BFD template tier used (matches `TIER_NUM` in template headers) |
| `mAdded` | float32 | (5,) | Noise-added moment perturbation |
| `minChisq` | float32 | scalar | Minimum χ² from template matching |
| `logDetInvCov` | float32 | scalar | log\|Σ⁻¹\| for normalization |

**Loading:**
```python
import numpy as np
from numpy.lib.recfunctions import join_by

grid_p = np.load("merged_masked_bfd_grid_p.npy")
grid_m = np.load("merged_masked_bfd_grid_m.npy")

# Join +/- shear pairs for calibration
grid = join_by("id", grid_p, grid_m, usemask=False, r1postfix="_p", r2postfix="_m")

# Moments and covariance (drop 5th moment for cNF)
moments = grid["moments_p"][:, :4]   # (N, 4): [Mf, Mr, M+, Mx]
cov_pkd = grid["covariance_p"]       # (N, 15): unpack with bfd.MomentCovariance.bulkUnpack()
pqr     = grid["pqr_p"][:, :6]      # (N, 6): [P, Q1, Q2, R11, R12, R22]
```

**Selection cuts applied in notebooks:**
- 1500 < Mf < 90000
- 2.2 < Mr/Mf < 3.5

---

## 2. Template Catalogues (FITS)

### `summary_templates_new.fits` (~1.1 GB)

FITS binary table (HDU 1) holding 1,369,189 DES Deep Field template galaxies.
Loaded via the `bfd` library's `TemplateTable` API.

Header: `WT_SIGMA = 0.65` (Blackman-Harris weight function σ), `FLUX_MIN = 1500.0`

**Columns:**

| Column | Format | Description |
|--------|--------|-------------|
| `moments` | 5×float64 | BFD moments [Mf, Mr, M+, Mx, Mc] |
| `covariance` | 15×float64 | Packed lower-triangular 5×5 moment covariance |
| `bkg` | float64 | Background level |
| `meds_index` | int64 | Index into MEDS (Multi-Epoch Data Structure) file |
| `mfrac_band` | 4×float64 | Masked pixel fraction per band |
| `Mf_band` | 4×float64 | Per-band flux moments |
| `cov_Mf_band` | 4×float64 | Per-band flux moment variances |
| `tile` | 10-char string | DES tile name |
| `coadd_ID` | int64 | DES coadd object ID |
| `dM_dg1` | 7×float64 | ∂M/∂g₁ — first shear derivative |
| `dM_dg2` | 7×float64 | ∂M/∂g₂ — first shear derivative |
| `dM_dmu` | 7×float64 | ∂M/∂μ — magnification derivative *(not used in cNF pipeline)* |
| `dM_dg1_dg1` | 7×float64 | ∂²M/∂g₁² |
| `dM_dg1_dg2` | 7×float64 | ∂²M/∂g₁∂g₂ |
| `dM_dg2_dg2` | 7×float64 | ∂²M/∂g₂² |
| `dM_dmu_dg1` | 7×float64 | ∂²M/∂μ∂g₁ *(not used in cNF pipeline)* |
| `dM_dmu_dg2` | 7×float64 | ∂²M/∂μ∂g₂ *(not used in cNF pipeline)* |
| `dM_dmu_dmu` | 7×float64 | ∂²M/∂μ² *(not used in cNF pipeline)* |

**Loading:**
```python
import bfd

template_tbl = bfd.TemplateTable(
    weightSpecs={"weightSigma": 0.65},
    sampleSpecs={"fluxMin": 1500.0}
).readOldFITS("summary_templates_new.fits")

moments          = template_tbl.getMoments()               # (N, 5)
cov_pkgd         = template_tbl.tab["covariance"]          # (N, 15) packed
cov              = bfd.MomentCovariance.bulkUnpack(cov_pkgd)[:, :5, :5]  # (N, 5, 5)
m, dm_dg, d2m_dg2 = template_tbl.getDerivs()
```

**Quality cuts applied in code:**
- SNR > 5: Mf / √Cov[0,0] > 5
- Mf > 600
- |ellipticity| = √((M+/Mr)² + (Mx/Mr)²) < 0.99

---

### `tmpl_t04.fits` (~4.0 GB)

FITS binary table (HDU 1) holding 43,767,115 template galaxies — a larger-area complement to
`summary_templates_new.fits`.

Header: `TIER_NUM=4`, `WT_TYPE=BlackmanHarris`, `WT_SIGMA=0.65`, `SIG_MAX=5.5`,
`SIG_STEP=1.2`, `SIG_FLUX=270.71`, `SIG_XY=391.41`, `FLUX_MIN=1500.0`

**Columns:**

| Column | Format | Description |
|--------|--------|-------------|
| `id` | int64 | Template galaxy identifier |
| `moments` | 7×float32 | BFD moments (7-component extended basis) |
| `moments_dg1` | 7×float32 | ∂M/∂g₁ |
| `moments_dg2` | 7×float32 | ∂M/∂g₂ |
| `moments_dmu` | 7×float32 | ∂M/∂μ *(not used in cNF pipeline)* |
| `moments_dg1_dg1` | 7×float32 | ∂²M/∂g₁² |
| `moments_dg1_dg2` | 7×float32 | ∂²M/∂g₁∂g₂ |
| `moments_dg2_dg2` | 7×float32 | ∂²M/∂g₂² |
| `moments_dmu_dg1` | 7×float32 | ∂²M/∂μ∂g₁ *(not used in cNF pipeline)* |
| `moments_dmu_dg2` | 7×float32 | ∂²M/∂μ∂g₂ *(not used in cNF pipeline)* |
| `moments_dmu_dmu` | 7×float32 | ∂²M/∂μ² *(not used in cNF pipeline)* |
| `weight` | float32 | Template sampling weight |
| `jSuppress` | float32 | j-suppression factor (Bernstein & Armstrong 2014): downweights templates whose noise contribution is large relative to the signal, stabilizing the BFD likelihood integral |

**Loading:**
```python
import fitsio

data    = fitsio.read("tmpl_t04.fits")
moments = data["moments"]    # (N, 7)
jsup    = data["jSuppress"]  # (N,)
```

---

### `tmpl_t04_joined.fits` (~4.0 GB)

FITS binary table (HDU 1) with 14,905,052 rows — a filtered subset of `tmpl_t04.fits` with
pre-attached covariance matrices. Same header keywords as `tmpl_t04.fits`. Used primarily for
selection bias cross-checks (`selection_term_analysis.md`).

Identical columns to `tmpl_t04.fits` plus:

| Column | Format | Description |
|--------|--------|-------------|
| `covariance` | 15×float64 | Packed lower-triangular 5×5 moment covariance |

---

## 3. COSMOS Template Container

### `template_moments_container_COSMOS_C24_r3764p01.npy` (~1.6 GB)

NumPy object array wrapping a pickled Python dict of `bfd.MomentTable` objects, one per template
galaxy. Source: COSMOS field (Cosmic Evolution Survey), DES Deep Field run C24, realization
r3764p01. COSMOS provides high-S/N images used as ground-truth templates; DES Deep Fields overlap
COSMOS and enable S/N ≫ wide-survey template measurements.

Requires a `RemappingUnpickler` because `bfd.weightfunction.KBlackmanHarris` may have moved
between `bfd` library versions.

**Loading:**
```python
import numpy as np
import pickle
import numpy.lib.format as nf

class RemappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "KBlackmanHarris":
            from bfd.weightfunction import KBlackmanHarris
            return KBlackmanHarris
        return super().find_class(module, name)

with open("template_moments_container_COSMOS_C24_r3764p01.npy", "rb") as f:
    version = nf.read_magic(f)
    if version == (2, 0):
        nf.read_array_header_2_0(f)
    else:
        nf.read_array_header_1_0(f)
    data = RemappingUnpickler(f).load()

# Access a specific galaxy
galaxy_id     = 712265208
moment_table  = data.item()[galaxy_id]["moments"]          # bfd.MomentTable
moment_val    = moment_table.getTemplate(0, 0).getMoment() # moment value
# .even / .odd: even = spin-0 and spin-2 components; odd = spin-1 and spin-3 components
```

---

## 4. Trained Flow Models (Equinox / JAX)

All `.eqx` files are Equinox PyTree checkpoints storing trained JAX neural network parameters.

**Weight function used during training**: Blackman-Harris (`KBlackmanHarris`), σ = 0.65 — a
smooth radial window applied in Fourier space to minimize sidelobe contamination from neighbours.

**Architecture**: `Spin0AutoregressiveLayer` bijections with `CoeffNet` neural networks (ReLU
MLPs) generating autoregressive transformation coefficients conditioned on shear [g1, g2] and
noise covariance Σ.

**Load / save:**
```python
import equinox as eqx

# Load — requires a template model instance with the correct architecture
prior_flow = eqx.tree_deserialise_leaves("prior_flow_xy.eqx", prior_flow_template)

# Save
eqx.tree_serialise_leaves("prior_flow_xy.eqx", trained_prior_flow)
```

### Prior flows — p(z | g, Σ_X)

| File | Size | Notes |
|------|------|-------|
| `prior_flow_xy.eqx` | ~1.3 MB | **Current primary model** — XY shear-conditioned prior |
| `prior_flow.eqx` | ~1.3 MB | Earlier baseline prior |
| `prior_flow_min.eqx` | ~1.4 MB | Minimal/optimized architecture variant |
| `prior_flow_12.eqx` | ~844 KB | 12-layer deeper prior experiment |
| `prior_flow_v2.eqx` | ~633 KB | Version 2 architecture experiment |
| `prior_flow_v3.eqx` | ~145 KB | Version 3 (smallest; likely simplified) |

### Q-flows (variational posterior) — q(z | y, Σ_y, g)

| File | Size | Notes |
|------|------|-------|
| `q_flow_xy.eqx` | ~180 KB | **Current primary model** — XY-conditioned posterior |
| `q_flow.eqx` | ~180 KB | Earlier baseline posterior |
| `q_flow_min.eqx` | ~500 KB | Minimal variant |
| `q_flow_12.eqx` | ~2.1 MB | 12-layer deeper experiment |
| `q_flow_v2.eqx` | ~889 KB | Version 2 |
| `q_flow_v3.eqx` | ~564 KB | Version 3 |

### Older single-model cNF checkpoints

| File | Size | Notes |
|------|------|-------|
| `cNF_quadratic.eqx` | ~2.6 MB | Quadratic moment parametrization |
| `cNF_quadratic_noise.eqx` | ~218 KB | Quadratic + noise covariance modeling |
| `cNF_nonlinear.eqx` | ~20 KB | Minimal nonlinear test |

---

## 5. Template Lookup Index

### `flow_index.pkl` (~325 MB)

Python pickle containing a dict mapping template galaxy IDs to pre-computed moment and derivative
data. Enables fast lookup during RQMC integration without recomputing template shear derivatives
at inference time.

```python
import pickle

with open("flow_index.pkl", "rb") as f:
    flow_index = pickle.load(f)

# flow_index[template_id] → {moments, derivatives, ...}
```

---

## 6. Diagnostic Plot

### `corner_nominalCov.png` (~346 KB)

Corner plot of the four transformed moment variables used in flow training:
log₁₀(Mf), Mr/Mf, M+/Mr, Mx/Mr. Diagonal panels are marginal 1D histograms; off-diagonal panels
are 2D scatter/contour plots. "Nominal covariance" refers to the adopted measurement noise Σ_y
distribution. Used to validate that the prior moment distributions from template catalogues match
the observed galaxy population.

---

## Summary Table

| File | Format | Size | Rows | Purpose |
|------|--------|------|------|---------|
| `merged_masked_bfd_grid_p.npy` | NumPy structured array | 1.6 GB | 5,432,002 | +shear simulation catalogue |
| `merged_masked_bfd_grid_m.npy` | NumPy structured array | 1.6 GB | 5,364,752 | −shear simulation catalogue |
| `summary_templates_new.fits` | FITS binary table | 1.1 GB | 1,369,189 | Deep-field template library |
| `tmpl_t04.fits` | FITS binary table | 4.0 GB | 43,767,115 | Large-area template set (tier 4) |
| `tmpl_t04_joined.fits` | FITS binary table | 4.0 GB | 14,905,052 | Filtered templates + covariance |
| `template_moments_container_COSMOS_C24_r3764p01.npy` | Pickled object array | 1.6 GB | dict | COSMOS template library |
| `prior_flow_xy.eqx` | Equinox JAX pytree | 1.3 MB | — | Prior p(z\|g, Σ_X) — current |
| `q_flow_xy.eqx` | Equinox JAX pytree | 180 KB | — | Posterior q(z\|y, Σ_y, g) — current |
| `flow_index.pkl` | Python pickle | 325 MB | dict | Template lookup index |
| `corner_nominalCov.png` | PNG | 346 KB | — | Diagnostic corner plot |

---

## Coordinate Transform Pipeline

The 5th moment (Mc) and magnification PQR terms are dropped first. The remaining 4-moment basis
is then transformed for flow training:

1. **Raw** [Mf, Mr, M+, Mx] → **Transformed**: [log₁₀(Mf), Mr/Mf, M+/Mr, Mx/Mr]
2. **Transformed** → **Standardized**: subtract mean, divide by std (per component)
3. **Covariance propagation**: Σ_transformed = J Σ_raw Jᵀ where J is the Jacobian of the
   log/ratio transforms, plus a second-order Hessian correction for the nonlinearities
