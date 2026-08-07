# Why Σ_X does NOT drop out of the Sobolev shear-derivative (current order)

*A walk-through of the "matched inverse pair" argument, and why the current
architecture (Σ_X data-adjacent) breaks the pairing that used to make it exact.*

## The question

The Sobolev loss supervises the flow's **moment shear-response** — how a decoded
moment `m` moves as the flow is conditioned on shear `g`:

```
A = dm/dg,   B = d²m/dg²   (at g = 0)
```

The prior flow is conditioned on *both* the shear `g` **and** the centroid-noise
covariance `Σ_X` (packed as `sx = [log_scale, e1, e2]`). Whether the shear response
depends on `Σ_X` turns out to hinge entirely on layer ORDER — see below.

## The three layers (current order)

The generative order is `base -> bulk -> shear -> Σ_X -> data`
(``models/bijections.py::new_masked_autoregressive_flow``, `Chain([sigmax, shear])`
composed so `shear` runs right after the bulk and `Σ_X` is data-adjacent — see
`EarlyChain`). The condition still splits cleanly across layers:

| layer | symbol | conditioned on | ignores |
|-------|--------|----------------|---------|
| bulk (unconditional autoregressive) | `E`      | — (nothing)      | `g`, `sx` |
| shear layer (`ShearTaylorLast`)      | `H_g`    | `g` only         | `sx`      |
| SigmaX coupling layer                | `S_sx`   | `sx` only        | `g`       |

Two facts still hold:

1. **`sx` enters the flow *only* through `S_sx`.** Nothing else sees it.
2. **`S_sx` is `g`-independent** (the SigmaX layer ignores `g1, g2`), and **`H_g`
   is `sx`-independent**.

But the ORDER of `H_g` and `S_sx` relative to each other has flipped relative to the
architecture this note originally analysed (`Σ_X` used to sit between bulk and shear;
now `Σ_X` is data-adjacent, outside the shear step). That flip is exactly what breaks
the cancellation below.

## How the response is measured: two passes

The response is computed by `_flow_shear_derivs`, which runs the flow **twice**.
There is only *one* physical SigmaX layer — but it is invoked once forward (encode)
and once backward (decode).

**Pass 1 — encode `m0` at `g = 0`, `sx`.** Just to obtain the latent `z`:

```
m0 ──[S_sx]──▶──[H₀]──▶──[E]──▶ z
     SigmaX      shear     bulk
                (=id at
                 g=0)
```

`H₀` is the identity because `ShearTaylorLast` is the identity at `g = 0`.

**Pass 2 — decode that `z` at varying `g`, same `sx`.** Reverse order, inverse layers:

```
z ──[E⁻¹]──▶──[H_g⁻¹]──▶──[S_sx⁻¹]──▶ m(g)
    bulk        shear        SigmaX
```

The response is `dm(g)/dg` at `g = 0`, holding `z` fixed.

## Gluing the passes: the telescope (and where it now STOPS)

```
m0 ─[S_sx]─▶─[H₀]─▶─[E]─▶ z ─[E⁻¹]─▶─[H_g⁻¹]─▶─[S_sx⁻¹]─▶ m(g)
             └id┘   └── these telescope ──┘
```

Cancel the bulk pair (still exact — `E` is `g`-independent) and drop the identity:

```
[E] ─▶ z ─[E⁻¹]  →  E⁻¹(E(·)) = id     (bulk cancels, as before)
[H₀]             →  id                 (shear is identity at g = 0, as before)
```

What survives this time is **not** just the shear layer — `S_sx` and `S_sx⁻¹` are no
longer adjacent to each other, because `H_g⁻¹` (a genuinely `g`-dependent step) now
sits BETWEEN them:

```
m0 ──[S_sx]──▶ v ──[H_g⁻¹]──▶ v(g) ──[S_sx⁻¹]──▶ m(g)
```

So

```
m(g) = S_sx⁻¹( H_g⁻¹( S_sx(m0) ) )
```

The `S_sx … S_sx⁻¹` pair is no longer "matched" in the sense that made it cancel
before (see next section) — `Σ_X` genuinely affects `dm/dg` now, through `S_sx`'s own
local Jacobian evaluated at `v = S_sx(m0)`.

## Why the pair no longer cancels

The old cancellation needed two things simultaneously:

- **Same layer, same `sx`, directly adjacent, forming an exact `S_sx⁻¹ ∘ S_sx = id`
  round-trip.** Still true here in isolation.
- **Nothing `g`-dependent sits between them.** This is the condition that now FAILS:
  `H_g⁻¹` sits directly between `S_sx` and `S_sx⁻¹` in the composition above, so the
  pair brackets the `g`-sweep instead of bracketing each other. `Σ_X` is no longer
  invisible to `dm/dg`.

## Practical consequences (current order)

`models/flows.py::_shear_response` implements `m(g) = S_sx⁻¹(H_g⁻¹(S_sx(m0)))`'s
derivatives directly, rather than assuming Σ_X-independence:

1. **Averaging over `Σ_X` would now be the "more correct" thing to do** (mirroring
   the density loss's own per-`Σ_X` stencil average), since the response genuinely
   depends on `sx`. `_sobolev_loss` still evaluates at a single reference `sx_ref`
   for cost reasons — a deliberate cheap approximation, not an exact simplification
   (see its docstring).

2. **Still no whole-flow autodiff needed.** `_shear_response` recovers `v = S_sx(m0)`
   via `SigmaXCouplingLayer.transform_and_log_det` (the closed-form inverse of its
   own `.inverse_and_log_det`), gets the shear layer's own local response `A, B` from
   `ShearTaylorLast.shear_derivs_generative(v)` (unchanged — a property of that layer
   alone), and composes through `S_sx`'s own *local* Jacobian/Hessian
   (`∂S_sx/∂v`, `∂²S_sx/∂v²`, both cheap small-coeff-net derivatives):

   ```
   A_data        = J_sx · A
   B_data[j,a,b] = J_sx · B[.,a,b]  +  H_sx[j,.,.] : A[.,a] A[.,b]
   ```

   — the standard 2nd-order chain rule, since `S_sx` doesn't depend on `g` explicitly.

## One-line summary

> `Σ_X` still enters the flow only through `S_sx`, but with `Σ_X` data-adjacent, the
> shear-derivative measurement's matched `S_sx … S_sx⁻¹` pair brackets the `g`-sweep
> instead of each other, so `Σ_X` is no longer invisible to `dm/dg` — the response is
> the shear layer's own coefficients pushed through `S_sx`'s local Jacobian/Hessian.
