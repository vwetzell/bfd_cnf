# Next session prompt

Paste the block below to start.  `GUIDING_PRINCIPLES.md` is the standing
reference for what counts as evidence and in what order to spend effort -- read
it first; everything here is an instance of it.

---

Branch `feat/centroid-shear-conditioning`, repo `bfd_cnf` (sim repo
`../bfd_cnf_imsims`).

**Task: the gauss2_v3d 500k run.**

    setsid bash dev/g2v3d_500k.sh

~3.8 h, then it re-solves the selection terms offline by itself.  Everything
below was measured 2026-09-05/06; do not re-derive it.  Nothing is committed --
the working tree carries a session's worth of changes on top of pre-existing
uncommitted work.

## HEADLINE: the size window was wrong, and it mattered

**The size window is (2.2, 3.2).**  Every script hardcoded `--window-size 2.2
1e9` -- no ceiling -- and `dev/window_scan.py` even carried a comment saying
there was nothing there to perturb.  All 12 files are fixed.

Without the ceiling the window admits objects whose measured `Mr/Mf` is at or
past `POINT_SOURCE` = 3.692575, i.e. **smaller than a point source** --
noise-scattered and physically impossible.  They are 8.05% of the no-ceiling
window, `in_domain` rejects most of their kernel cloud, and they carried
essentially all of the deep arm's draw noise.

| | 2.2 - 1e9 (void) | 2.2 - 3.2 |
|---|---|---|
| deep seed-to-seed spread, windowed uncorrected m1 | 0.01815 | **0.00179** |
| uncorrected m1, shallow / deep | -0.04698 / -0.05122 | **+0.03196 / +0.02753** |

Re-solved at 2^24 off the saved PQR, **gauss2 still passes and the arms now
agree**:

| arm | corrected m1 | from zero |
|---|---|---|
| `gauss2_v3` shallow | -0.00344 +/- 0.01398 | 0.25 sigma |
| `gauss2_v3d` deep | -0.00228 +/- 0.01624 | 0.14 sigma |

against the void -0.00128 +/- 0.01196 and -0.01738 +/- 0.01251.  **The deep
arm's old 1.4-sigma tension was the missing ceiling.**

Same cut for bulgedisc: the two priors match at p50 `Mr/Mf` 3.149 vs 3.150 and
both are 0.00% past POINT_SOURCE (only the NOISY targets scatter past, 4.8% /
4.3%).  Inside the flux window the ceiling removes 27% of targets -- a real
selection, which is why it moved m1 by 0.08.

## 2^24 IS NOT ENOUGH AT THIS WINDOW -- but it converges

`R_s12 = 0` and `R_s11 = R_s22` are forced by isotropy, so their size IS the
selection MC error.  Shallow ladder:

| draws | seed 0 | seed 7 | m1 spread | R_s12 (s0 / s7) |
|---|---|---|---|---|
| 2^24 | -0.00344 | +0.00557 | 9.0e-03 | +1.22e-02 / -5.50e-03 |
| 2^26 | +0.00385 | +0.00409 | 2.4e-04 | +2.81e-03 / +7.70e-05 |

`R_s12` improves ~4.8x for 16x draws -- the 1/sqrt(N) rate.  **The Hill < 2
tail is not binding at these sample sizes**, contrary to what the 2^24 pair
alone suggests.  **2^26 minimum to believe a number, 2^28 to quote against
tau** (~2.7 h, offline, no target integration).  GUIDING_PRINCIPLES 3.1's
"7.4e-04 at 2^24" was measured at the NO-ceiling window and does not transfer.

## What makes 3.8 h possible

All verified bit-identical on m1/c1/c2/ghat and the zero-weight/fallback counts
unless noted.  `bias.py` is 30 -> 27.1 ms per target per arm, then two arms
instead of three, then half the targets integrated.

| change | worth |
|---|---|
| `make_psi_ld` -- psi's log-det by chain rule, not `slogdet(jacfwd(psi))` | 2.6% |
| `mixture_draws(to_host=False)` -- draws stay on device | 3% |
| `shear()` reads Q, R from `response_tensors` instead of nested jacfwd | 5% |
| `--no-zero-arm` | 30% |
| `--prefilter-pad 0.2 --prefilter-sample 0.15` | ~2x at 1M |

Ruled out, do not retry: `jax.linearize` for the Hessian (0% -- XLA already
CSEs the shared primal), bigger/smaller batches (<2%), TF32 (7% and breaks
pairing), `--lambda-stride 8` (6% but moves m1 half a bar).  **The remaining
~88% is `flow.log_prob` differentiated twice in g over 8192 draws** -- that is
the estimator, not the plumbing.  `--gauge auto` is 2.4x `--gauge prior`, which
is why older code felt fast: it was computing an infinite-variance estimator.

`BFD_TIMING=1` splits `pqr_streamed`'s loop, but **its buckets mis-attribute**
(async dispatch: the first barrier absorbs whatever is in flight).  Trust
ablations, not those timers.

## The two pre-filter traps -- both cost a wrong run

Out-of-window targets reach eq. (45)-(46) ONLY as the count `N_ns`; `ghat`
masks their Q and R away.  So skipping their integration is free -- but:

1. `n_ns` must be counted on the FULL catalog before the pre-filter.  From the
   integrated subset it counts only the padding ring plus the sample (~1100
   against a true ~4050).
2. `bootstrap`'s corrected branch RECOMPUTES `N_ns` per replicate, so fixing
   the point estimate alone is not enough.  Fixed with per-target `weights`
   (1 inside the pad, 1/sample outside).

Verified on the SHALLOW arm -- `n_ns` identical (4612/4616), bars matching to
0.3%, residual 9.2e-04 shared identically by the uncorrected and corrected
numbers, which is re-batching draw noise and not an `N_ns` error.  **Verify
pre-filter changes on the shallow arm only**: at depth, draw noise swamps the
comparison.

The 15% out-of-window sample is not decoration -- the zero-weight population is
**100% out of window** (deep arm: 4338 targets, 0.00% in-window against 32.4%
out), and it is an open IS-proposal failure.  A hard cut would hide it.

## Still open

1. **Re-solve the two 20k arms at 2^26+.**  The headline table above is 2^24,
   now known under-converged.  Cheap, offline.
2. **`--prefilter-pad` with `--no-zero-arm` has never been run together** --
   this run is the first.  Watch the "excluded (drives the correction)" line
   and the in/out-of-window zero-weight split.
3. **`CentroidMarginalize.inverse_and_log_det` is wrong for this layer.**  It
   returns the inverse-function-theorem value, but `marginalize` and
   `unmarginalize` are two CLOSED FORMS and not exact inverses, so it leaves
   5.6e-04 in the log-det where Psi is the identity.  `make_psi_ld` avoids it;
   every other caller is UNAUDITED.
4. `project_to_physics`/`response` carry a dead `g` parameter -- a trap for
   anyone assuming Q, R depend on g.
5. The 22% zero-weight deep targets: still an unexplained IS-proposal failure.
6. `[[rs-tail-is-unphysical-draws]]`'s CONCLUSION survives the window fix (86%
   of leaders invert to no galaxy, against 91% before) but its MECHANISM does
   not -- the leaders moved off the ceiling to a flux-independent locus at
   `Mr/Mf ~ 2.94, Mc/Mr ~ 5.94` that is not yet identified.
7. The older channels: conditional-support leak at the size edge, flux-gradient
   response error, `Sigma_X` PSF leak (below tau at DES-like ellipticity).

## Traps -- all paid for already

* **`--floor-eps 1e-3 --window-guard 100` on every gauss2 run.**  Without them
  the deep arm returns `m1 = -5.75 +/- 116.64`.
* **`--batch-budget` bounds MEMORY independently of `--samples`.**
  `batch = (batch_budget or 131072) // chunk`, so OMITTING it gives a LARGER
  batch and the card OOMs at 13 GiB.
* **The Bash tool runs zsh, which does NOT word-split `$var`.**  `COMMON="..."`
  passed as `$COMMON`, and `set -- $spec`, both fail silently in a direct tool
  call.  Cost three runs this session.  Fine inside `bash foo.sh`.
* **`pgrep -f "python -u bias.py"` matches the waiter's own command line** and
  hangs forever.  Check `nvidia-smi --query-compute-apps` instead.
* **ONE JAX GPU JOB AT A TIME.**  Launching a second while the first finishes
  is an instant OOM -- cost three runs this session.
* Pipe run output through `tee`, not a bare `grep` (an unbuffered grep hides
  progress until exit).
* **Quote no `m1` without its window** -- and now, without its draw count too.
* Never compare two paired differences using their separate bars.
* An identical rebuild moves noisy `m1` by sd 0.0008.
* **Launch long runs with `setsid`.**

## State

Catalogs (`../bfd_cnf_imsims/data`): `moments_gauss2_fwd_{g2v3,g2v3d}.fits`,
`targets_{g2v3,g2v3d}_{g0,g1p02,g1m02}_500k.fits`,
`copies_gauss2_fwd_{g2v3,g2v3d}.fits`.  bulgedisc has only 200k targets -- a 1M
bulgedisc run needs a re-render first (~1.5-2 h CPU).

Flows: `flows/{bulk,shear,centroid}_{g2v3,g2v3d}.eqx`; bulgedisc's are
`{bulk,shear}_v10` + `centroid_v11`.

PQR: `pqr/{g2v3,g2v3d}_20k_matched.npz` are the 20k runs (re-solvable at the
correct window offline).  This run writes `pqr/g2v3d_500k.npz`.

New this session: `bias.py --prefilter-pad/--prefilter-sample/--no-zero-arm/
--lambda-stride`, `BFD_TIMING=1`, `make_psi_ld`, `ShearResponse.response_tensors`,
`dev/g2v3d_500k.sh`, `dev/window_scan.py --size/--flux` + a `size_hi` scan
boundary, `dev/{bench_pqr,prefilter_check,draw_seed_null,draw_noise_scaling}`,
`tests/test_psi_logdet.py`, `tests/test_response_tensors.py`.
Suite: 121 passed, 1 skipped.
