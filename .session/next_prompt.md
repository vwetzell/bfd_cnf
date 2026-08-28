Read HANDOFF.md's last section in full before doing anything else --
"2026-08-28 (cont.): the centroid approximation is NOT the cause -- the
ensemble R has flipped sign at the faint end". Branch:
`feat/centroid-shear-conditioning`.

## Where things stand

The `m1 ~ -1` catastrophe on `bulgedisc_deep_v2` is diagnosed down to a
mechanism, but not fixed.

**The previous session's centroid-approximation hypothesis is FALSIFIED and
the user's disagreement was correct.** Measured directly: the analytic
transport's `trace(P)`, its round-trip residual, its pre-cap `|dz|` and its
log-det spread over kernel draws are all comparable to -- in the tail, milder
than -- `gauss2_deep`'s. Do not revisit it.

**What is actually wrong:** the ensemble `R` has the wrong SIGN. The Fisher
identity `sum q^2 / sum(-r)` is `+0.958` for `gauss2_deep` and `-0.272` for
`bulgedisc_deep_v2`. `ghat = -R^-1 Q` off a wrong-signed R is exactly the
"tight `m1` pinned near -1" symptom. Use this ratio as the health check from
now on: it needs one arm rather than three, it is far cheaper than `m1`, and
it localises the failure.

**Where:** entirely in the two faintest flux quintiles (`Mf < ~1450`, S/N <
~16). q3/q4/q5 give +1.02/+1.15/+1.19, as healthy as gauss2's.

**Why:** `R` contains `Var_w[score_g]` over the importance-weighted draws, and
`score_g` scales with `|grad_z log p_bulk|`. That score is 6.4 (median, flat in
flux) for gauss2 and 18.2 for `bulgedisc_v2`, rising to 33.6 in the faintest
quintile with a p99 of 483. The retune INVERTED the flux dependence: the old
bulgedisc's score rose with flux (steepest where the kernel is narrowest,
harmless); the retuned one's falls with flux, so the density is steepest
exactly where the noise kernel is widest -- and the retune also moved the
median flux down 3x, putting most of the catalog there.

## Ruled out this session, with numbers (do not re-run)

- **ESS / Monte-Carlo bias.** `sum R11` flat over S = 2048/8192/32768
  (+18477/+21901/+19589). Kernel ESS matches gauss2's (median 153 vs 149).
- **Ceiling-violating targets alone.** Dropping every `Mr/Mf > POINT_SOURCE`
  or `Mc/Mr > POINT_SOURCE_MC` target plus a 3% margin still leaves
  `sum R11 = +4795`. (This also CORRECTS the previous entry's item 4: the top
  20 by |R| carry 8.9% of `sum |R|`, but the top 20 by SIGNED R11 carry 58%.)
- **The second-order shear response.** Full ablation at 4000 targets x 3 arms:
  full -1.28915, no-2nd-spin0 -1.34367, no-2nd-spin2 -1.29118, no-2nd-at-all
  -1.34654. Every one slightly worse. A clean negative. (A single-point grid
  scan at `M1 = M2 = 0` looks like it confirms this hypothesis -- it does not
  survive the ensemble. Do not be fooled by it.)
- **The shear layer's fit.** `dm/dg` / `d2m/dg2` vs bfd truth in the faint
  quintiles is 0.4%/1.9%/3.3%/3.1%/4.1% -- the response is fit well exactly
  where R is wrong.
- **Training-data volume.** `bulgedisc_v2_1M`'s score profile is identical to
  the 100k flow's (18.9 vs 18.2 median).
- **Setup/config differences.** Headers, `cov` vs `noise_sigma^2` scaling,
  chart constants across all three layers of every checkpoint, and the
  bulk+shear warm-start graft are all clean -- see the HANDOFF entry.

## Next steps, in rough priority order

1. **Decide whether the density really is that sharp or the flow is
   manufacturing it.** Sample `flows/bulk_bulgedisc_v2.eqx` at faint flux and
   compare its marginals against `moments_bulgedisc_v2.fits`. Look at the
   flux floor specifically: the catalog has 22% of its galaxies between
   `Mf = 562` and `1103` above a hard cut near 300 -- a near-discontinuity in
   `log10 Mf` the flow has to represent as a cliff. The pre-retune population
   had no such pile-up.
2. **If the density is genuinely that sharp, the convolution is the problem,
   not the fit.** The noise kernel at `Mf ~ 900` is `sqrt(cov00) = 89.6`,
   i.e. +/-10% in flux, straddling that cliff. The retune's flux floor is a
   rendering choice, not real-sky physics -- ask whether it should be softened,
   or the depth reconsidered so the kernel does not span it.
3. **`sane_targets`' `factor = 1000` guard is far too loose for this
   population** (threshold 22197 against a worst target of 22735, which it
   therefore keeps). Symptom-catcher, not the fix, but worth revisiting.
4. **`split_centroid` never peels.** It returns `(flow, None)` for every flow
   `bulk.build_flow(shear=True, centroid=True)` builds, because those give the
   layer `cond_dim = 5`. So `bias.py` differentiates through the centroid layer
   on BOTH populations and the `centroid_transform`/log-det-in-the-weight path
   is dead code. Not a difference between the populations, but re-check
   anything whose reasoning assumed the peel was live.

## Working rules (carried over, still true)

- `shear.py train` on `bulgedisc` needs `--deriv-weight 1e4`.
- `centroid.py train` has nothing to train (zero free parameters) -- warm
  starts + runs `check` only.
- Always use `--alpha 0.5` for any `_deep` population at `noise_sigma` in the
  0.9-2.73 range (`--alpha 1.0` is ESS-starved and unreliable).
- `bias.py` on a `_deep` population needs `--batch-budget 65536` or it OOMs
  the 16GB card; don't run two GPU jobs concurrently.
- `noise_sigma = 0.9` is the correct depth for `bulgedisc_deep_v2`. It matches
  the two populations almost exactly in `Sigma_X / (Mf Mr)`, the dimensionless
  combination that drives the centroid layer (1.2e-3 vs 1.15e-3).
- `python -u <script>` from the relevant repo root; background long jobs with
  `nohup ... & disown` and poll the PID directly. `pgrep -f <script>` SELF-
  MATCHES the polling shell's own command line -- wait on the PID or on the
  output file, never on `pgrep -f`.
- Any real-data numeric target pulled from `summary_templates_new.fits` for a
  NEW comparison must be re-verified against the exact filtering path of
  whatever it's being compared to -- bit this investigation three times.
