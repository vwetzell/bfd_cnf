# SigmaXCouplingLayer redesign — spec for review

A physically-constrained replacement for the spin-0/spin-2 transform in
`SigmaXCouplingLayer`, derived from the leading-order effect of marginalizing
over the target centroid. The goal: a transform that **only allows the changes
the centroid-marginalization actually produces** (à la `ExplicitPolyLast` for
shear), replacing the 5 free fields `{ls0, ls1, s, c, d}` with a small set of
physically-locked ones. **No code yet** — this is for sign-off.

---

## 1. Physical basis (recap)

A centroid offset `δ=(u,v)` injects moments **quadratic** in `δ`:
`ΔMr=κ(u²+v²)`, `ΔM1=κ(u²−v²)`, `ΔM2=κ(2uv)`, `ΔMf≈0`. Marginalizing
`δ∼N(0,C_X)` gives, **to leading order in C_X**, a pure mean shift
`ΔM=(0, a_s, a_e e1, a_e e2)` with `a_s,a_e ∝ tr C_X` — a deterministic
**transport** (the broadening is the O(C_X²) remainder). This is the part a
single bijection can represent exactly.

**Key caveat (assumption A1):** the ELBO target is the *reweighting*
`P(M|C_X) ∝ P_0(M)·g(M;C_X)`, not literally the convolution. But the *form* of
the leading mean shift — spin-0 size shift + spin-2 dipole, with the
size/ellipticity scalings locked — is fixed by spin symmetry and holds for the
reweighting tilt as well; only the **sign and magnitude** differ, and those are
**learned** (exactly as `ExplicitPolyLast` learns its shear coefficients). The
form is physics; the constants are data.

---

## 2. Coordinates and constants

Flow ratio coordinates (output of `RawMomentStandardize._forward_transform`):
`w = (w0,w1,w2,w3) = (log10 Mf, Mr/Mf, M1/Mr, M2/Mr)`.
Standardized (what the layer sees): `z = (w − μ)/σ`, with per-coordinate
`μ=raw2standard.mean`, `σ=raw2standard.std`.

Define the fixed constant `c1 ≡ μ1/σ1` (size). Note `μ2≈μ3≈0` (ellipticity is
centered by symmetry), `σ2=σ3` (rotational symmetry) — used below.

Condition vector `[g1,g2,λ,e1,e2]`; the layer uses `(λ,e1,e2)` only
(`λ=½log det C_X`; `g` handled upstream by `ExplicitPolyLast`). Normalized
inputs: `λ_n=(λ−λ̄)/s_λ`, `ehat²=(e1²+e2²)/e_ref²`, `|e|²=e1²+e2²`.

---

## 3. Core transform (v1)

Two scalar fields from the conditioner, inputs `f=[z0, λ_n, ehat²]`:

```
g_s = net_size(f)        # log size-inflation;     κ ≡ exp(g_s) > 0
D   = net_dip(f)         # dipole amplitude (z-units), unbounded (sign learned)
```

**Forward** `T : z → z'` (data→base direction):

```
z0' = z0
z1' = κ·z1 + c1·(κ − 1)          # size: scale  +  LOCKED loc-shift
z2' = (z2 + D·e1) / κ            # ellipticity: dipole + RECIPROCAL shrink
z3' = (z3 + D·e2) / κ
```

**log-abs-det** (diagonal Jacobian in (z1,z2,z3); z0 untouched):

```
log|det J| = (+g_s) + (−g_s) + (−g_s) = −g_s
```

**Inverse** `T⁻¹ : z' → z` (closed form; `g_s,D` recomputed from `z0=z0'`):

```
z0 = z0'
z1 = (z1' − c1·(κ − 1)) / κ
z2 = κ·z2' − D·e1
z3 = κ·z3' − D·e2
log|det J⁻¹| = +g_s
```

Three things this encodes that the current layer does not:

- **The loc-shift `c1·(κ−1)`** on `z1` is exactly the missing mean-shift term
  (the current layer scales the zero-mean `z1`, which can't move the size mean;
  this scales the offset-`μ1`-mean `w1`, which does). Closes the monopole gap.
- **The ellipticity scaling is `1/κ`, locked to the size inflation `κ`** — same
  physical `M_r`-denominator knob. The current layer has these as independent
  free fields (`ls1`, `s`).
- **Dropped:** no `c2,c3` ellipticity loc terms (they are `∝μ2,μ3≈0` and would
  break spin-2 equivariance — see §6).

**Identity init:** initialize `net_size`, `net_dip` to output `≈0` ⇒ `κ≈1`,
`D≈0` ⇒ `T≈identity`. Respects "small perturbation" and starts training from the
unmodified base shape.

---

## 4. Equivariance & invertibility

- **Spin-2 equivariance:** under rotation by α, `(z2,z3)` and `(e1,e2)` both
  rotate by 2α; `g_s,D,κ` are scalars (functions of invariants `z0,λ,|e|²`), so
  `D·(e1,e2)` is spin-2 (matches `(z2,z3)`) and `1/κ` is spin-0. Exactly
  equivariant **provided** the dropped `c2,c3` terms stay dropped. ✓
- **Parity:** no antisymmetric/rotation coupling — correct (marginalization has
  no handedness). ✓
- **Bijectivity:** `κ=exp(g_s)>0` always ⇒ invertible everywhere; inverse is
  closed-form (no root-finding) because `g_s,D` depend only on `z0` (unchanged
  by the map). ✓

---

## 5. Parametrization & normalization fixes

- `net_size, net_dip`: `CoeffNet(in=3, out=1, …)` each — **or one
  `CoeffNet(in=3, out=2)`** sharing features (recommended). Inputs `[z0, λ_n, ehat²]`.
- Fix the conditioning normalization while here:
  - `s_λ`: set so `λ_n` spans ≈[−1,1] over `log_scale_range=(11,15)` (i.e.
    `λ̄≈13`, `s_λ≈2`), instead of the current `std=1` (inputs ran to +3).
  - `e_ref`: use the **actual** `e_max=0.2` (current code normalizes by `0.1²`).
- `c1=μ1/σ1` stored as a static (non-trainable) field, taken from
  `raw2standard.mean/std` at build time (passed through `build_flows`).

Parameter-field count: **2** (`g_s`, `D`) vs the current **5**
(`ls0,ls1,s,c,d`).

---

## 6. Optional extensions (each a higher-order "change this process produces")

**(O1) Quadrupole — anisotropic broadening, O(C_X²).** Replace the isotropic
`1/κ` on `(z2,z3)` with `(1/κ)(I + c·E)`, `E=[[e1,e2],[e2,−e1]]`,
`c = tanh(net_q(f))` (so `|c|<1`):

```
(z2',z3') = (1/κ)·[ (I + c·E)(z2,z3) + D·(e1,e2) ]
log|det J| = −g_s + log(1 − c²|e|²)
```
Captures the leading genuinely-non-transport (broadening) anisotropy.

**(O2) Flux response.** `z0' = κ0·z0 + c0·(κ0−1)`, `κ0=exp(g0)`,
`g0=net_flux([λ_n, ehat²])`, `c0=μ0/σ0`; `log|det J| += g0`. Motivated not by the
`ΔMf≈0` mean-shift but by the reweighting tilting the *flux distribution*
(brighter ⇒ tighter centroid). Likely small.

**(O3) Size-conditioning of `ρ`.** Let `g_s,D` depend on `z1` too (physically
`ρ=a_s/M_r` depends on size). Keeps a lower-triangular Jacobian (order
z0→z1→{z2,z3}) but the inverse needs care: `z1'=κ(z1)·z1+c1(κ(z1)−1)` is a 1-D
map, invertible if monotone — adds a `log(1 + z1·∂g_s/∂z1)`-type term and a 1-D
solve. Recommend **deferring** to v2; v1 uses `z0`-only conditioning (clean
closed-form inverse, log-det `−g_s`).

---

## 7. Comparison to the current layer

| effect | current | proposed v1 |
|---|---|---|
| flux scale | `ls0` (free) | identity (O2 optional) |
| size scale | `ls1` (free, **no loc**) | `κ·z1 + c1(κ−1)` (scale **+ loc**) |
| ellipticity isotropic scale | `s` (free, **independent**) | `1/κ` (**locked to size**) |
| ellipticity dipole | `d·e` | `D·e` (kept) |
| ellipticity quadrupole | `c·E` (always on) | optional O1 |
| free scalar fields | 5 | 2 (+1 per option) |
| log-det | `−(ls0+ls1)+log(s²−c²|e|²)` | `−g_s` (`+log(1−c²|e|²)` w/ O1) |

---

## 8. Assumptions & risks

- **A1 (form ⇔ reweighting):** §1 — the locked form is symmetry-justified for
  the reweighting tilt; if the true tilt is strongly non-Gaussian the affine
  leading-order form under-fits (mitigated by O1, or later a spline).
- **Locking risk:** if the data genuinely wants *independent* size and
  ellipticity scalings, locking them underfits. Physically they are the same
  knob, so this is the intended inductive bias — but it is a testable bet.
- **Sign:** the convolution picture gives `+e`, the data reweighting gave `−e`;
  `D` (and `g_s`) are learned, so the sign is data-driven. Form fixed, constants
  free.
- **Requires retraining from scratch** (architecture change). Old checkpoints
  won't load.

---

## 9. Validation plan (post-implementation)

1. **Unit:** round-trip `T∘T⁻¹` to ~1e-6; finite `log_prob`; identity at init.
2. **Dipole:** `plot_centering_dipole_diagnostic` — diagonal, symmetric
   `d⟨M2⟩/de2 → −0.049` target (as the current fixed layer reached ~95%).
3. **Monopole (the new fix):** `⟨Mr/Mf⟩` vs `log_scale` should now track the
   data target (3.31→3.04), not the ~20% the current layer manages. This is the
   decisive test that the locked loc-shift works.
4. **Corner / shear-derivative** plots unchanged or improved.

---

## 10. Open choices for sign-off

1. **Scope:** v1 core only `{g_s, D}`, or core + O1 (quadrupole)?
2. **O2 flux term:** include or defer?
3. **Conditioning:** `z0`-only (clean, recommended) or add size `z1` now (O3)?
4. **Normalization:** apply the `s_λ`/`e_ref` fixes in §5 in the same change?

Default recommendation: **v1 core + O1 quadrupole**, `z0`-only conditioning,
with the §5 normalization fixes. Smallest physically-complete-to-the-modeled-order
layer; closes both the dipole and the monopole gaps.
