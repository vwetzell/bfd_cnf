# Why Σ_X drops out of the Sobolev shear-derivative

*A walk-through of the "matched inverse pair" that makes the flow's shear response
independent of the centroid-noise condition `Σ_X`.*

## The question

The Sobolev loss supervises the flow's **moment shear-response** — how a decoded
moment `m` moves as the flow is conditioned on shear `g`:

```
A = dm/dg,   B = d²m/dg²   (at g = 0)
```

The prior flow is conditioned on *both* the shear `g` **and** the centroid-noise
covariance `Σ_X` (packed as `sx = [log_scale, e1, e2]`). It is tempting to think the
shear response depends on `Σ_X` — the old `_sobolev_loss` averaged it over many `Σ_X`
draws for exactly that reason. It does **not**. This note shows why, with pictures.

## The three layers

The condition splits cleanly across the flow's layers:

| layer | symbol | conditioned on | ignores |
|-------|--------|----------------|---------|
| bulk (unconditional autoregressive) | `E`      | — (nothing)      | `g`, `sx` |
| SigmaX coupling layer                | `S_sx`   | `sx` only        | `g`       |
| shear layer (`ShearTaylorLast`)      | `H_g`    | `g` only         | `sx`      |

Two facts do all the work:

1. **`sx` enters the flow *only* through `S_sx`.** Nothing else sees it.
2. **`S_sx` is `g`-independent** (the SigmaX layer ignores `g1, g2`), and **`H_g`
   is `sx`-independent**.

## How the response is measured: two passes

The response is computed by `_flow_shear_derivs`, which runs the flow **twice**.
There is only *one* physical SigmaX layer — but it is invoked once forward (encode)
and once backward (decode). Keep that in mind: the "pair" we cancel later is this one
layer used in both passes.

**Pass 1 — encode `m0` at `g = 0`, `sx`.** Just to obtain the latent `z`:

```
m0 ──[H₀]──▶──[S_sx]──▶──[E]──▶ z
     shear     SigmaX     bulk
    (=id at
     g=0)
```

`H₀` is the identity because `ShearTaylorLast` is the identity at `g = 0`.

**Pass 2 — decode that `z` at varying `g`, same `sx`.** Reverse order, inverse layers:

```
z ──[E⁻¹]──▶──[S_sx⁻¹]──▶──[H_g⁻¹]──▶ m(g)
    bulk        SigmaX       shear
```

The response is `dm(g)/dg` at `g = 0`, holding `z` fixed.

## Gluing the passes: the telescope

Because Pass 2 decodes exactly the `z` that Pass 1 produced, we can write the whole
thing as one chain from `m0` to `m(g)`:

```
m0 ─[H₀]─▶─[S_sx]─▶─[E]─▶ z ─[E⁻¹]─▶─[S_sx⁻¹]─▶─[H_g⁻¹]─▶ m(g)
    └id┘   └────────── these telescope ──────────┘   └ only g-dependent step
```

Cancel adjacent inverse pairs, working outward from the middle:

```
[E] ─▶ z ─[E⁻¹]      →   E⁻¹(E(·))     = id     (bulk cancels)
[S_sx] ── [S_sx⁻¹]   →   S_sx⁻¹(S_sx(·)) = id   (SigmaX cancels — the matched pair)
[H₀]                 →   id                     (shear is identity at g = 0)
```

What survives:

```
m0 ──────────────────▶──[H_g⁻¹]──▶ m(g)
```

So

```
m(g) = H_g⁻¹(m0)
```

Every `sx`-carrying box was undone by its own inverse. The shear response `dm/dg` is
**just the shear layer evaluated at `m0`** — no `Σ_X` appears at all. This is why the
numerical check gives *exactly* zero dependence on `Σ_X`.

## Why the pair is "matched"

`S_sx` (Pass 1) and `S_sx⁻¹` (Pass 2) cancel because of two things:

- **Same layer, same `sx`.** Pass 2 inverts the very `z` that Pass 1 built, at the
  same `Σ_X`, so `S_sx⁻¹ ∘ S_sx = id`. If you encoded at `sx` and supervised the
  decode at a *different* `sx′`, you would get `S_sx⁻¹ ∘ S_{sx′} ≠ id`, and `Σ_X`
  **would** survive. The cancellation is a property of the *matched round-trip*, not
  of the layer alone.
- **Nothing `g`-dependent sits between them.** `S_sx` does not move when you vary `g`
  (it ignores `g`), so the pair stays perfectly matched across the whole `g`-sweep and
  contributes exactly zero to `dm/dg`.

## The other layer order, for contrast

With the shear layer **base-adjacent** (`conditional_first = False`), the shear step
sits *inside* the bulk bracket instead of outside it:

```
m0 ─[E]─▶─[H₀]─▶─[S_sx]─▶ z ─[S_sx⁻¹]─▶─[H_g⁻¹]─▶─[E⁻¹]─▶ m(g)
                 └───── matched pair cancels ─────┘
```

The SigmaX pair still cancels, so the response is **still `sx`-independent**. But now
the bulk does *not* telescope away — it brackets the shear step:

```
m(g) = E⁻¹( H_g⁻¹( E(m0) ) )
```

The shear response threads through `E⁻¹` (the bulk decode Jacobian). That is the
*entanglement* that makes the base-adjacent response differ from the shear layer's own
analytic coefficient — and why we prefer the data-adjacent order, where
`m(g) = H_g⁻¹(m0)` gives the layer coefficient directly.

## Practical consequences

Both simplifications in the current `_sobolev_loss` follow directly:

1. **No averaging over `Σ_X`.** The response is `Σ_X`-independent in *either* layer
   order, so the old per-`Σ_X` `lax.map` was evaluating the identical quantity `M`
   times. One evaluation at a single reference `sx_ref` suffices.

2. **Analytic derivatives (data-adjacent only).** Because `m(g) = H_g⁻¹(m0)` with
   nothing after the shear layer, the whole-flow response equals the shear layer's own
   closed-form generative coefficients:

   ```
   A_gen = −A
   B_gen = −B + (∂A/∂x)·A + (symmetric)
   ```

   computed from the layer's `A`, `B` and the *local* Jacobian `∂A/∂x` of its coeff
   nets — no whole-flow autodiff. (Base-adjacent, the extra `E⁻¹` bracket breaks this
   identity, so that path falls back to autodiff.)

## One-line summary

> `Σ_X` enters the flow only through `S_sx`, and the shear-derivative measurement always
> brackets the `g`-sweep with a matched `S_sx … S_sx⁻¹`, so `Σ_X` is invisible to
> `dm/dg`.
