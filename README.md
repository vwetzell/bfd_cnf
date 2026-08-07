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
coordinate change into `[log10 Mf, Mr/Mf, M1/Mr, M2/Mr]` followed by eight
`EquivariantAutoregressiveLayer` steps.  No `g`, no `Sigma_X` — in the phase-1
sims `Sigma_X` is the same for every galaxy, so conditioning on it would be
learning a constant.

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
python shear.py train  --data ../bfd_cnf_imsims/data/moments.fits
python shear.py derivs --data ../bfd_cnf_imsims/data/moments.fits
```

`plots/shear_derivs.png` shows `P` and its first and second shear derivatives
over the `(M1/Mr, M2/Mr)` plane at fixed flux and size.

## Layout

```
bulk.py               phase-1 build / train / corner plot
shear.py              phase-2 train / check / shear-derivative plot
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
