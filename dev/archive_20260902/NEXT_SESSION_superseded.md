# Next session prompt

Paste the block below to start.  Everything it refers to is in `HANDOFF.md`
under the `## 2026-09-02 (night, cont. 4)` through `(cont. 7)` sections and in
the auto-memory index.

---

Branch `feat/centroid-shear-conditioning`, repo `bfd_cnf` (sim repo
`../bfd_cnf_imsims`).  Read `HANDOFF.md`'s `## 2026-09-02 (night, cont. 5)`,
`(cont. 6)` and `(cont. 7)` first -- cont. 6 is a RETRACTION and shows the
failure mode this task is about.

**Task: a provenance audit of every `bulgedisc_v2` artifact.**  Confirm that
the training catalog, the 1M catalog, the three target arms, the copies, and
the flow checkpoints were all built from the same, current code and the same
population, and that nothing stale is silently in use.  This is not
speculative housekeeping: the last month has produced at least five bugs of
exactly this shape, each of which invalidated numbers that had already been
acted on --

* [[stale-chart-constants-were-the-peak]] -- `chart_loc`/`chart_scale`/
  `e_scale` stale after a graft, spin-2 std off 3.9x; voided every checkpoint.
* [[coeff-whitening-was-stale-too]] -- `sync_chart_constants` never resynced
  `coeffs.u_mean`/`u_white`.
* [[gauss2-truth-chart-is-stale]] -- `to_coords` slot 2 wrong for nine days.
* [[gauss2-deep-run-two-new-bugs]] -- `bias.py --train-data` silently defaulted
  to bulgedisc's `moments.fits` for non-`sersic` populations.
* [[bulgedisc-real-match-ellipticity-limit]] -- the `|e|` target itself was
  stale, "the 3rd such bug".

**Do NOT re-derive the inventory below; it is already checked.**

### What is already established

`../bfd_cnf_imsims` and `bfd_cnf` are both git-clean.  The population was
retuned in imsims `03a2fe3` (2026-08-27 15:29, "Retune bulgedisc's
flux/size/ellipticity marginals"), and there has been NO imsims commit since,
so on timestamps alone every `_v2` artifact post-dates the current code:

| artifact | built |
|---|---|
| `moments_bulgedisc_v2.fits` (100k, trains the flow) | 08-27 15:42 |
| `targets_deep_g1p02_200k_v2.fits` | 08-27 22:27 |
| `targets_deep_g1m02_200k_v2.fits` | 08-27 22:40 |
| `targets_deep_g0_200k_v2.fits` | 08-27 22:52 |
| `copies_bulgedisc_v2.fits` (22.7M) | 08-27 23:13 |
| `moments_bulgedisc_v2_1M.fits` | 08-28 01:42 |
| `flows/bulk_bulgedisc_v2.eqx` | 08-27 15:49 |
| `flows/centroid_bulgedisc_v2.eqx` (the production flow) | 08-27 23:13 |
| `flows/bulk_bulgedisc_v2_1M.eqx` | 08-28 01:45 |
| `flows/centroid_bulgedisc_v2_1M.eqx` | 08-28 01:48 |

FITS headers, `ext=1`:

| file | PIXSCALE | WTSIGMA | PSFSIGMA | **NOISESIG** | SEED | IMGNOISE | **SIG_XY** |
|---|---|---|---|---|---|---|---|
| `moments_bulgedisc_v2` | 0.2 | 0.65 | 0.4 | **1.0** | 0 | False | **119.4** |
| `moments_bulgedisc_v2_1M` | 0.2 | 0.65 | 0.4 | **1.0** | 0 | False | **119.4** |
| `targets_deep_*_200k_v2` | 0.2 | 0.65 | 0.4 | **0.9** | 0 | True | **107.5** |
| `copies_bulgedisc_v2` | 0.2 | 0.65 | 0.4 | **0.9** | 0 | -- | **107.5** |

The two moments catalogs' `Mf`, `Mr/Mf`, `|e|` and `Mc/Mr` marginals agree to
the percentile, so those two are the same population as each other.

### Lead 1 -- the depth mismatch, already found and NOT yet resolved

**The prior catalogs were rendered at `--noise-sigma 1.0`, the targets and
copies at 0.9.**  The moments themselves are noiseless so this cannot move
them, but it moves two stored columns, and `Sigma_X` by 11%:

* `centroid.py:486` trains the centroid layer at
  `sigma_x = galaxies["cov_odd"][0] * sigma_scale**2` -- from the TRAINING
  catalog, i.e. 119.4.
* `bias.py:2096` evaluates at `rows["zero"]["cov_odd"]` -- from the TARGET
  catalog, i.e. 107.5.

The layer is CONDITIONED on `Sigma_X`, so this is what conditioning is for and
it may be entirely fine -- but the layer's accuracy in `Sigma_X` is a live
topic ([[differential-centroid-layer]]: the copy grid is good over sigma
[0.7, 1.4] but the analytic route is pinned over [0.9, 1.1] only;
[[centroid-gain-is-population-calibrated]]).  Settle it:

1. Was 1.0 vs 0.9 deliberate or a slip?  Check the `regen`/render commands and
   `sim.NOISE_SIGMA`'s default.
2. Does `--sigma-scale`'s training range actually cover 107.5/119.4 = 0.90?
3. Measure it rather than argue it: re-run the centroid check at both
   `Sigma_X` values and see whether the layer's transfer ratio moves.  If it
   does, the production flow is being evaluated off its training point and
   every centroid number since 08-27 carries that.

### Lead 2 -- the check the artifacts make easy

`copies_bulgedisc_v2.fits` carries `GALAXIES` and `POPULATION` extensions, and
so do the moments and target catalogs.  **`POPULATION` holds the noiseless
drawn parameters** (`flux`, `sigma`, `e1`, `e2`, `bulge_frac`, `bulge_ratio`),
so it compares across artifacts with no noise confound -- unlike `moments`,
which is noisy in every target catalog (see below).  Verify:

* the three target arms are the same 200000 galaxies (they must be, row for
  row, or the `+/-` pairing is broken -- [[antithetic-pairing-was-broken]]);
* the copies' `POPULATION` is the same 100000 galaxies as
  `moments_bulgedisc_v2` (both `--n 100000 --seed 0`), so the copy sum and the
  flow share a prior;
* `moments_bulgedisc_v2_1M`'s first 100000 rows either do or do not match the
  100k catalog, and which it is matters for [[rs-flow-vs-templates-is-draw-count]].

Already confirmed: `targets_deep_g0_200k_v2` and a fresh `--seed 0 --n 200000`
render are the same galaxies, with and without `--add-noise`.

### Lead 3 -- the flow checkpoints

`flows/` holds ~100 `.eqx` files and the production one is
`centroid_bulgedisc_v2.eqx`.  Nothing in the file records what it was trained
on.  For each of `bulk_bulgedisc_v2`, `centroid_bulgedisc_v2` and their `_1M`
twins, establish from `logs/`, `retrain*.sh` and the shell history which
catalog and which arguments produced it, and whether the chart constants inside
match the catalog it is used with -- that is the [[stale-chart-constants-were-the-peak]]
failure, and `shear.py check` / `sync_chart_constants` are the tools.  A
checkpoint whose provenance cannot be established should be treated as
suspect, not as fine.

### Traps

* **`moments` in a TARGET catalog is NOISY.**  `bias.py`'s `truth = m["zero"]`
  is a third noise realization ([[saved-truth-is-the-zero-arm]]); a
  target catalog's `Mr/Mf` reaches 4.845 against the population's true ceiling
  3.683, and `|e|` reaches 6.17.  Comparing a target catalog's `moments` to a
  training catalog's is comparing noisy to noiseless and will manufacture a
  population "mismatch" that is not there.  Compare `POPULATION` instead.
* **A target catalog's `dm_dg` column is CONTAMINATED** -- off by -10% in `M1`
  at the compact end ([[response-diagnostics-need-a-clean-reference]]).  Only
  the noiseless catalogs' derivative columns are trustworthy, and those are
  exact to ~1e-4.
* **All arms share `--seed 0`**, so they share the noise FIELD as well as the
  galaxies.  Useful (noise cancels in `m_+ - m_-`) and dangerous (binning on
  one arm correlates with a residual measured on the others).
* **`imsims/analytic.py` is not ground truth for bulgedisc** -- it hard-codes
  `BULGE_FRAC = 0.5` while `sim.py:319` draws `U(0, 0.4)`; only gauss2 sets
  `pop["bulge_frac"] = analytic.BULGE_FRAC` (`sim.py:510`).
* One JAX GPU job at a time.  `selection_terms` accumulates `eqx.filter_jit`
  CUBINs and OOMs at about the fifth call at 1M rows -- one route and one
  window per process (`dev/rs_copies.py` shows the pattern).
* A render is ~17 min for 200k at `OMP_NUM_THREADS=1`; the Bash tool caps at
  10 min, so launch detached and wait on the PID, not `pgrep -f`
  ([[background-hang-is-pgrep-selfmatch]]).

### Why this is worth a session now

The headline result it protects: **the ~2% residual is not the flow.**  BFD's
own template sum reproduces it on identical targets, paired,
`templates - flow = +0.0040 +/- 0.0041` ([[residual-is-not-the-flow]]).  That
conclusion rests on the flow, the copies and the targets all describing ONE
population.  If they do not, the paired agreement means something else
entirely -- and the same is true of [[per-band-m1-is-not-a-bias]] and of every
`R_s` comparison in cont. 7.

Tooling written this session and ready to reuse: `dev/edge_share.py`,
`dev/prior_n.py`, `dev/response_g3.py`, `dev/rs_copies.py`.
