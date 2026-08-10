# bfd_cnf

A conditional normalizing flow for the BFD prior of Bernstein et al. 2016
(MNRAS 459, 4467):

    P(m | g, Sigma_X)

where `m = [Mf, Mr, M1, M2]` are the four even moments, `g` the shear, and
`Sigma_X` the covariance of the odd/centroid moments `X` that BFD marginalises
the unknown source position over (paper eq. 36).  The architecture stages that
as a generative stack

    base -> bulk -> shear(g) -> Sigma_X -> data

(`models/bijections.py:new_masked_autoregressive_flow`), and this branch builds
it one arrow at a time against the simulated catalogs from
[`bfd_cnf_imsims`](../bfd_cnf_imsims).

## Phase 1 — the bulk (here now)

`bulk.py` trains only the unconditional part: the `RawMomentStandardize`
coordinate change into `[log10 Mf, Mr/Mf, Mc/Mr, M1/Mr, M2/Mr]` followed by
eight `EquivariantAutoregressiveLayer` steps.  The three spin-0 coordinates come
first and the spin-2 pair last; the permutation between layers cycles through
all six orderings of the spin-0 slots so none is permanently the head of the
autoregression, while the spin-2 pair is never permuted — swapping `M1`/`M2`
would rotate the shape by 45 degrees and break the equivariance.  No `g`, no
`Sigma_X` — in the phase-1 sims `Sigma_X` is the same for every galaxy, so
conditioning on it would be learning a constant.

```
python -m imsims.sim --n 100000 --out data/moments.fits   # in ../bfd_cnf_imsims
python bulk.py train  --data ../bfd_cnf_imsims/data/moments.fits
python bulk.py corner --data ../bfd_cnf_imsims/data/moments.fits
```

`plots/bulk_corner.png` overlays the flow on the sims in the four flow
coordinates, with the point-source (`Mr/Mf = 3.976`) reference marked — above
that line a source is unresolved and carries no shape information.

## Phase 2 — shear conditioning (here now)

`models/shear.py` adds the `ShearResponse` layer, derived from scratch against
the paper's appendix C and checked against `bfd`.  Its functional form is fixed
by three symmetries rather than chosen:

* **spin** — `Mf`, `Mr` are spin 0 and `e = (M1 + iM2)/Mr` is spin 2, as is `g`,
  which leaves exactly three spin-0 and five spin-2 structures through order
  `g^2`;
* **parity** — every coefficient is real;
* **flux** — moments are linear in the image, so the eleven coefficients depend
  only on `Mr/Mf` and `|e|^2`.  Verified exact to six digits over a 1000x flux
  range, which is why the coefficient network has two inputs.

`shear.py` trains it on the density, not on a fitted response map: templates are
lensed by their own exact `Q`, `R` and the flow maximises `log P(m(g)|g)`.  The
analytic derivatives also enter as a low-variance estimator of the same
transport velocity — see that module's docstring for why that costs no
generality at first order, and where it does at second.

```
python shear.py train   --data ../bfd_cnf_imsims/data/moments.fits
python shear.py derivs  --data ../bfd_cnf_imsims/data/moments.fits
python shear.py scatter --data ../bfd_cnf_imsims/data/moments.fits
```

`plots/shear_derivs.png` shows `P` and its first and second shear derivatives
over the `(M1/Mr, M2/Mr)` plane at fixed flux and size.

`scatter` needs no trained flow: it inverts the exact response coefficients out
of the catalog's own `dm_dg` and asks how much of them a regression on the
flux-blind invariants cannot explain.  That residual is `Var[Q|m]`, the piece no
deterministic transport can carry and the floor on the multiplicative bias — so
it says up front how well any flow *can* do on a given population, and whether
modelling `Mc` was enough for it.  Run with and without `Mc` in the regression,
and split by galaxy type where the catalog has one.

## Measured on the two imsims populations

100k templates each, identical settings (6000 steps, bulk frozen and warm
started).  `dm/dg` residuals are RMS over the moment vector, against bfd's exact
appendix-C derivatives; the floor is what `scatter` says the moments cannot
determine even in principle.

| | bulge + disc | Sersic red/blue |
|---|---|---|
| spin-2 `dm/dg`, layer | 2.36% | 0.46% (0.22% at 20k steps) |
| spin-2 floor, `Var[Q\|m]` | 1.42% | 0.12% |
| spin-0 `dMr/dg`, layer | 26.6% | 1.16% |
| spin-0 `dMr/dg` floor | 21.4% | 1.12% |

The layer sits at or near the floor in every case, so the gap between the two
columns is a property of the populations, not of the implementation: eight
parameters of galaxy behind five moments versus five behind five.  Dropping `Mc`
from the regression raises the Sersic spin-2 floor from 0.12% to 9.2%, which is
the sharpest statement yet of why `Mc` is modelled.

What does *not* improve with training is the second derivative: 3.3x more steps
moves `d2m/dg2` for `Mc` from 3.48% to 3.46%.  That is the `Var[Q|m]` term the
module docstring predicts a deterministic transport cannot supply, and it is the
next thing to fix — a stochastic layer, not a bigger network.  (It *did* improve
with cleaner supervision: these catalogs zero-pad the stamps before the FFTs,
which halved that residual from the 7.33% measured without padding.  More steps
still do nothing, which is the point.)

## Sheared targets, and the bias measured on them

`bfd_cnf_imsims` now also produces targets lensed at the **image level**, for
measuring a multiplicative bias from the flow's point estimates against ground
truth rather than against bfd's own derivatives:

```
../bfd_cnf_imsims/data/targets_g1p02_1M.fits   # g1 = +0.02, 1M
../bfd_cnf_imsims/data/targets_g1m02_1M.fits   # g1 = -0.02, same galaxies
../bfd_cnf_imsims/data/targets_g0_1M.fits      # unsheared twin, for binning
```

These used to carry a floor: the image response and bfd's analytic `dm_dg` —
what this flow is trained on — differed coherently by 0.1–1%, growing towards
the resolution limit, worth `m1 ~ -6e-3` on the bulge + disc population.  That
was a moving-boundary term in bfd's appendix-C derivatives, traced to the
Blackman-Harris weight not reaching zero at `kmax`, and it is **fixed** — see
that repo's README.  Population-weighted `sum(FD)/sum(Q) - 1` on `M1` is now
`+0.006%` against `-0.61%` before, and flat across resolution.  Every catalog
and flow here was regenerated after it (`../bfd_cnf_imsims/regen.sh`, then
`retrain.sh`).

On noiseless targets, which is where this measurement is currently sharpest:

```
python bias.py --flow flows/shear.eqx        # 1M bulge + disc, g1 = +/-0.02

  m1 = +0.00128 +/- 0.00016
  c1 = +4.26e-05 +/- 1.1e-04   c2 = +3.16e-05 +/- 1.1e-04
```

The residual `m1` is about what the two known systematics predict together: the
`Var[Q|m]` floor of the deterministic transport (~4e-4, the table above) and the
estimator's own truncation at `O(g^2)` (~4e-4 at `g = 0.02`).  Getting there
took one real fix on the flow's side — `RawMomentStandardize` was standardising
`M1/Mr` and `M2/Mr` with independent per-coordinate mean and scale, which is the
one place in the stack that could give the prior a preferred direction on the
sky, and a prior with a preferred direction reads out as additive shear.
Symmetrising it (`_effective()`, zero spin-2 mean and one shared scale, applied
on use so the optimiser cannot undo it) took `c2` from `+3.78e-4` to `+3.2e-5`
and `m1` from `+2.05e-3` to the number above.
`tests/test_shear.py::test_flow_isotropy` guards it.

## Noisy targets, and the integral under `C_M`

A real target is `M = M^G + M^n` with `M^n ~ N(0, C_M)` (paper eq. 5), and the
probability the ensemble estimator needs is the prior *convolved* with that
noise — the continuum limit of the paper's template sum, eq. (38):

    P(M | g) = INT dm P(m | g) L(M - m),   L = N(., C_M)

`bias.py --samples S` does that integral by drawing `S` points from the noise
kernel itself and averaging the flow over them (`bias.log_conv`).  Because the
draws do not depend on `g`, `Q` and `R` (eq. 12–13) are just that one estimator
differentiated — common random numbers throughout, never three noisy quantities
divided by each other.  The catalogs are still noiseless on disk; the
realization is added at load time, in moment space.  That is exact rather than a
shortcut: at a **fixed** centre the moments are a linear functional of the image,
so Gaussian pixel noise gives exactly `N(0, C_M)` in moment space.
`bfd_cnf_imsims`'s `test_moment_noise_is_the_stored_covariance` checks it against
actual noisy images, which is also the only test of `C_M` itself through the
padded FFT.

```
python bias.py --flow flows/shear.eqx --samples 1024 --n-targets 20000
python -m tests.test_conv        # the integral against an analytic Gaussian
```

The cost is `S` flow evaluations per target per catalog: ~33 ms/target at
`S = 1024`, so 20k targets in each of the three catalogs is about half an hour.

**Kernel sampling is fine on the median target and poor on the faint tail.**
The ESS printed at startup is the paper's own diagnostic — sec. 2.5 warns that
dividing `Q` and `R` by a noisy `P` biases the answer inversely with the number
of templates contributing to `P`, and here that number is the ESS, so the
induced `m` is of order `1/ESS`:

| draws/target | median ESS | 5th pct | frac. below 10 |
|---|---|---|---|
| 1024 | 106 | 5 | 8.2% |
| 4096 | 383 | 21 | 2.3% |
| 16384 | 1545 | 56 | 2.0% |

The median scales linearly with `S`, as it must, but ~2% of targets sit below
ESS 10 no matter how many draws they get.  Those are the faint, barely resolved
ones, where the kernel is far wider than the thin `Mc` sheet the prior lives on,
so blind draws almost never land on it.  Buying that tail with more draws does
not work; it needs a proposal that knows where the prior is — which is what the
paper's k-d tree does when it keeps only templates within `chi^2 < sigma_max^2`
of the target (sec. 3.2).

That tail is what dominates the first measurement (`bias_bulgedisc_noisy.txt`,
20k targets, `S = 1024`, median ESS 134):

```
  m1 = -0.00504 +/- 0.00883        # noiseless, same flow: +0.00128 +/- 0.00016
  c1 = +1.09e-03 +/- 1.2e-03   c2 = +5.43e-04 +/- 1.3e-03
```

Per target the error bar is ~8x the noiseless one, and it is not spread evenly —
the middle flux quintiles carry `+/- 0.03` to `+/- 0.04` while the faintest
carries `+/- 0.002`.  That is not shape noise (the +/-g pairing removes it, and
both catalogs share a noise realization and share their kernel draws); it is a
handful of targets whose `Phat` rests on one or two draws, so their `Q/P`
is arbitrary and their weight in eq. (45) is not.  The paper describes the same
failure for template sums in sec. 2.5 — "a target ... dominated by a single
template that is many sigma away ... giving spuriously large influence in the
final lensing estimator" — and its answer is the same prior-aware proposal.
So `m1` here is consistent with zero, but only because it cannot yet say
anything sharper than +/-1%.

Not yet handled: centroid marginalisation (targets and templates are both
measured at their known centre, so the `L(X^G)|J|` factors of eq. (27)/(38) are
absent), and selection — `bias.py` cuts on nothing, so `P(s|g) = 1` and the
non-detection terms of eq. (45)–(46) are legitimately absent.  Bin or cut on a
*noisy* flux and they stop being.

## Layout

```
bulk.py               phase-1 build / train / corner plot
shear.py              phase-2 train / check / scatter / shear-derivative plot
bias.py               m and c on the targets, noiseless or integrated under C_M
models/shear.py       the ShearResponse layer
models/bijections.py  the bulk layers, and the Sigma_X layer phase 3 needs
```

## Notes

The layer stack is kept whole even though phase 1 uses a slice of it, so the
conditional layers slot in later without rebuilding the bulk.  `models/flows.py`
(the ELBO/shear-supervision losses) and the training/inference drivers are *not*
on this branch: they were wired to a `config.py`/`data.py` pair built around the
old DES-derived catalogs, and will be rewritten against the sims as phases 2-3
land.  They remain on `main` and `working`.

## Requires

`jax`, `flowjax`, `equinox`, `optax`, `paramax`, `numpy`, `fitsio`, `corner`,
`matplotlib`.
