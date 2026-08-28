Read HANDOFF.md's last TWO sections before doing anything else (2026-08-28 and
2026-08-28 cont.). Branch: `feat/centroid-shear-conditioning`.

## Where things stand

`bulgedisc_v2` is the realistic population and stays; the METHOD has to adapt
to it. The `m1 ~ -1` catastrophe is fully diagnosed and the cause is located,
but no fix is built yet.

**The flow's fitted density is the defect. The population is fine.** BFD's own
template-sum prior -- `P(M|g) = (1/N) SUM_G N(M - m_G(g); C)`, exact autodiff,
no flow and no importance sampling -- gives `sum -R11 = +1971`, Fisher ratio
+1.26 on the faintest 20% of the same targets, where the flow gives -9.3e4 and
-0.03. So neither the population nor the noise depth is the problem.

**Why:** the template sum's bandwidth IS the noise covariance C, so its score is
bounded by construction. The fitted flow is free to put structure at any scale
and does: bulk score p99 ~ 1e5 in z units at the faint end, identical on the 1M
retrain, so structural rather than overfitting.

**Where:** the score climbs with `v = Mc / (POINT_SOURCE_MC * Mr)`, the slot-2
concentration coordinate -- p50 of 8.9 below v = 0.80 (gauss2-like) rising to
355 above v = 0.99. `bulgedisc_v2` puts 15.5% of its population above v = 0.95
and reaches 0.998; gauss2 puts 1.0% there and stops at 0.977. The flux
dependence reported earlier is a CORRELATE of this -- the flux-floor reading
was the wrong suspect.

## Two things that are settled, do not redo them

- **Selection is not an escape.** `--window-size 2.2 3.5 --window-flux 1345 1e9`
  (S/N >= 15; the committed `FLUX_WINDOW = (2500, 50000)` is stale, calibrated
  when median flux was 5090 and it is now 1770) gives `m1 = -0.01649 +/-
  0.00486` uncorrected -- but the eq. (40)/(45)-(46) correction takes it to
  `+0.87631 +/- 0.03566`. That is not a bug: `P_s = 0.5486`, `Q_s ~ 0`, terms
  taken from the templates. The correction's job is to add back what the CUT
  galaxies would have contributed, which is exactly the population the flow
  gets wrong. A cut cannot dodge a density error.
- **The sign constraint is not a hard one, and nothing mechanical is broken.**
  `sum r` is forced negative only via `E[q^2] = E[-r]`, which needs
  `Z(g) = INT P(M|g) dM = 1` AND the targets to be drawn from the model.
  Measured `Z(g) = 1` to 5e-4 with bounded weights (`q = (p_0+p_g)/2`; the naive
  `E_{p0}[p_g/p_0]` diverges, do not use it), `Z(0) = 1` exact as a control.
  The shear map DOES fold (`det <= 0` for 0.07-0.11% of noisy targets at
  `g = +/-0.02`, min det -81.6) but far too rarely to move Z. So the zero
  crossing is genuine misspecification.

## Directions for the fix, none built

1. **Make the fitted prior no sharper than the template sum.** "No sharper than
   C" is not a tuned bandwidth -- it is the bandwidth BFD's own prior already
   has. The noise-split (train on `m + N(0, eps C)`, evaluate with kernel
   `(1-eps)C`) is EXACT in `P(M|g)` for any eps, and in a pre-test cut the
   faint-end score p99 from 282878 to 154 at eps = 0.02 while barely moving
   gauss2. **The user has ruled this out as tuning** because eps is free. Any
   version needs eps pinned by an argument rather than chosen -- e.g. derived
   from the template sum's own effective bandwidth.
2. **Check the chart's z2 tail before redesigning it.** A smooth population
   under a logit link gives a BOUNDED score as `v -> 1` (`grad_z log p -> -1`),
   so the measured 355 says the flow is not fitting that tail, not that the
   chart is wrong in principle. Look at the z2 marginal, fitted vs catalog,
   above `v = 0.95` (15.5% of the population, `z2 > ~3`) before changing the
   parameterisation.
3. **Supervise the density against the template sum**, the way `--deriv-weight`
   already supervises the derivatives against bfd's exact `dm_dg`/`d2m_dg2`.
   The faint-end `log P` from the template sum is cheap and correct.

Note the template sum is NOT a drop-in replacement: on `bulgedisc_v2`'s
BRIGHTEST 20% it blows up (ratio 2.3e5) because 100000 templates cannot cover a
flux tail reaching 2.7e6. That is the failure mode the flow exists to fix.

## Working rules (carried over, still true)

- `shear.py train` on `bulgedisc` needs `--deriv-weight 1e4`.
- `centroid.py train` has nothing to train (zero free parameters) -- warm
  starts + runs `check` only.
- Use `--alpha 0.5` for any `_deep` population. Verified again this session:
  alpha 0.5 is converged by S = 32768 and reproducible over 4 independent
  seeds; alpha 1.0 does not converge at S = 8192/32768/131072.
- **`pqr_streamed` seeds chunk `c` as `seed + 7919*c`, so raising S at a fixed
  seed NESTS the smaller draw set.** An S scan at one seed is not a convergence
  test -- vary the seed too.
- `bias.py` on a `_deep` population needs `--batch-budget 65536` or it OOMs the
  16GB card; don't run two GPU jobs concurrently.
- `noise_sigma = 0.9` is the correct depth for `bulgedisc_deep_v2`.
- Diagnose with the Fisher ratio `sum q^2 / sum(-r)` on the g=0 arm, binned by
  flux -- one arm, far cheaper than `m1`, and it localises the failure. gauss2
  gives 0.95-1.01 in every bin and under every window.
- `python -u <script>` from the repo root; background long jobs with
  `nohup ... & disown` and poll the PID or the output file. `pgrep -f <script>`
  SELF-MATCHES the polling shell's own command line.
