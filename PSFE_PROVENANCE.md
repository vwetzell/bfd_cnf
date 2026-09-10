# PSF-anisotropy catalogs and the SigmaXBlockLayer flows — provenance record

Written 2026-09-10. Reconstructed almost entirely from Claude Code session
transcripts (`~/.claude/projects/-home-vwetzell-gitrepos-bfd-cnf/*.jsonl`)
and file mtimes, because **none of this survived in git, shell history, or
a training log** — see "How this was recovered" below. If you touch any of
these flows again, log the command to `logs/` with `tee`, the way
`rebuild_v3.sh` does, so the next session doesn't have to redo this.

---

## 1. The psfe target catalogs

Location: `../bfd_cnf_imsims/data/targets_v3psfe<axis><sign><mag>_g<arm>_20k.fits`

Built by `dev/render_psfe.sh` (amplitude `$1`, default 0.2):

```
python -u -m imsims.sim --n 20000 --seed 1 --pop bulgedisc --noise-sigma 0.93 \
    --psf-e1 <e1> --psf-e2 <e2> --g1 <g> --add-noise \
    --out data/targets_v3psf<name><TAG>_<arm>_20k.fits
```

- **Population: `bulgedisc`**, not gauss2.
- **`--noise-sigma 0.93`** — hardcoded, this is the DES/COSMOS-matched
  "deep" depth used by `bulgedisc_v3`/`gauss2_v3` (median flux S/N 19.0,
  `bias.py:192-194`).
- **Matched/paired sims**: every config (`e00`, `e1p`, `e2p`, `e1m`) and
  every arm (`g0`, `g1p02`, `g1m02`) shares `--seed 1` — same galaxies,
  same noise field, across the whole grid. This antithetic pairing is what
  lets a PSF-rotation leak as small as the target signal be resolved; see
  the script's own header comment for the amplitude-0.2 exaggeration
  rationale (`Sigma_X` split +17.1% linear, `C_M` split +1.68% quadratic,
  both at `psf_e = 0.2`).
- **No new prior catalog per config**: latent moments are PSF-independent
  bit-for-bit (`dev/psf_anisotropy_probe.py`, gate G0), so
  `moments_bulgedisc_v3.fits` is correct for every `psf_e`.
- Existing on-disk amplitudes: `0002`/`0005`/`0010`/`0020` (small sweep,
  unsigned) and `02`/`05`/`10`/`20` (the `e1p`/`e1m`/`e2p` orientation
  triplet), all at `noise-sigma 0.93`.

## 2. The centroid layer: SigmaXBlockLayer replaces CentroidMarginalize

On branch `feat/centroid-shear-conditioning` (committed `b93e27b`),
`models/bijections.py` gained `SigmaXBlockLayer`, wired into
`bulk.build_flow` (`bulk.py:265-281`) in place of the old
`CentroidMarginalize`. It has five independent coefficient nets and a
condition vector `[g1, g2, C00, C01, C11]` — a genuinely anisotropic
Sigma_X, not just a trace/isotropic one, calibrated to `e_max = 0.1`
(`bulk.py:282`).

`centroid.py`'s old `train()` (the k⁴ spin-2 bracket regression) does not
apply to it. The session that ported it in also added
`train-sigmax`/`check-sigmax` CLI modes (`centroid.py`, ~line 384 on):
plain supervised regression of `relative_channels` (dMf/Mf, dMr/Mr, dMc/Mc,
spin-2 projected on the galaxy's own e) against `weighted_copy_mean`'s
eq.-36 target — an NLL/log_prob attempt was tried first and measured to
fail (loss stuck above the "predict zero" floor until a `size_loc` bug in
`bulk.build_flow`'s `SigmaXBlockLayer(...)` call was fixed).

Two flags extend the training grid, **both uncommitted-until-now and never
run before this branch**:

- `--multi-scale`: also trains against `Sigma_X * s²` for
  `s ∈ {1/1.4, 1.4}` (i.e. `noise_sigma * s`, since `--sigma-scale`'s own
  help text says it "rescale[s] Sigma_X by this factor squared"). Without
  it, the layer was measured to NOT generalise across this margin
  (ellipticity-response ratio 0.79 → 1.63 → −0.09 at scale 1.0 → 0.8 →
  1.2).
- `--anisotropic`: also trains against `ANISO_TRAIN_POINTS = ((0.05, 0.0),
  (0.07, π/4))` — two off-axis Sigma_X ellipticity points (different
  angles, so the nets can't learn a direction-specific shortcut), same
  trace as the base point.

## 3. Which flow file has which training — recovered by session forensics

No shell-history hit, no committed script, and no training log survived
for these three files anywhere in the repo. They were recovered by
matching `flows/*.eqx` mtimes (local time) against Claude Code session
transcript timestamps (UTC = local + 4h) in
`~/.claude/projects/-home-vwetzell-gitrepos-bfd-cnf/`.

| file | mtime (local) | session | UTC timestamp | command | notes |
|---|---|---|---|---|---|
| `flows/centroid_g2v3d_sigmaxblock.eqx` | Sep 9 13:59 | `b36f2ac9…` subagent `aa077e52f24fc4b66` | `2026-09-09T17:58:57Z` | `train-sigmax --steps 6000 --batch 8192 --holdout 0.05 --multi-scale` | **5th and last** overwrite of this path in that subagent; earlier 4 runs (16:01/16:25/17:31/17:40 UTC) were debugging passes (NLL→relative-channel loss, z-space, slot-mapping bug, `size_loc` bug) and are superseded. Held-out resid: Mf 1.7e-2, Mr 1.8e-2, Mc 1.8e-2, ellipticity ratio 0.80. |
| `flows/centroid_g2v3d_sigmaxblock_v2.eqx` | Sep 9 18:21 | `d1c891d4…` | `2026-09-09T22:20:13Z` | `train-sigmax --steps 6000 --batch 8192` (no `--holdout`, no `--multi-scale`, no `--anisotropic`) | Full-catalog fit, isotropic base Sigma_X only. Resid: Mf 2.5e-3, Mr 6.4e-3, Mc 7.5e-3, ellipticity ratio 0.815. |
| `flows/centroid_g2v3d_sigmaxblock_multiscale.eqx` | Sep 9 18:25 | `d1c891d4…` | `2026-09-09T22:24:44Z` | `train-sigmax --steps 6000 --batch 8192 --multi-scale --anisotropic` | **Only one of the three trained with `--anisotropic`.** Resid: Mf 2.4e-3, Mr 6.3e-3, Mc 7.6e-3, ellipticity ratio 0.820. **This is the flow used for last night's 500k bias run** — confirmed by `logs/bias_g2v3d_sigmaxblock_multiscale_500k.log:1`. |

All three were trained on `../bfd_cnf_imsims/data/copies_gauss2_fwd_g2v3d.fits`
warm-started from `flows/shear_g2v3d.eqx` — **population `gauss2_v3d`
(gauss2_fwd), not bulgedisc.**

## 4. What `gauss2_v3d` actually is, and the training's real ranges

`gauss2_v3d` is "the SAME galaxies [as `gauss2_v3`] at 2x the depth"
(`bias.py:290-299`): **`noise_sigma = 1.86`** (median flux S/N 9.6), vs.
`gauss2_v3`'s `noise_sigma = 0.93` (S/N 19.0). Confirmed from the training
log's `Sigma_X = [49339.7, 0, 49339.7]`: `sqrt(49339.7) = 222.1`, matching
the comment's quoted `SIG_XY = 222.12547` for v3d, exactly 2x v3's
`111.06273`.

So the trained ranges for `centroid_g2v3d_sigmaxblock_multiscale.eqx` are:

- **Noise**: base `noise_sigma = 1.86`, `--multi-scale` extends to
  `noise_sigma ∈ [1.86/1.4, 1.86*1.4] ≈ [1.33, 2.60]`.
- **Sigma_X ellipticity**: `e_Sigma_X ∈ {0, 0.05, 0.07}` (two angles for
  the nonzero points), hard bound `e_max = 0.1`.

## 5. Compatibility with the existing psfe catalogs — NOT compatible, two ways

**(a) Noise level.** psfe catalogs render at `noise_sigma = 0.93`
(`dev/render_psfe.sh`). The trained range is `[1.33, 2.60]`. **0.93 is
below the trained floor** — not an interpolation, an extrapolation past the
edge of the grid.

**(b) Population.** psfe catalogs are `--pop bulgedisc`. The only
SigmaXBlockLayer flow with anisotropic training is on `gauss2_v3d`
(gauss2_fwd). `[[no-cross-population-transfer]]`: a flow uses ONLY its own
population's data. **There is currently no bulgedisc-population
SigmaXBlockLayer flow at all** — the old `CentroidMarginalize` bulgedisc
flows (`centroid_g2v3d.eqx` is gauss2 too; the actual bulgedisc ones are
`centroid_v3.eqx` etc.) predate `SigmaXBlockLayer` entirely.

**(c) PSF-e vs. Sigma_X-e mapping** (informational, not a blocker by
itself): PSF ellipticity reaches Sigma_X at roughly linear order —
measured `+17.1%` split `(C00-C11)/mean` at `psf_e = 0.2`
(`[[psf-leak-is-the-sigma-x-channel]]`). In `centroid._aniso_sigma_x`'s
convention (`split = 2*e1`), that's `e_Sigma_X ≈ 0.086` at `psf_e = 0.2`,
i.e. **`e_Sigma_X ≈ 0.43 * psf_e`**:

| catalog `psf_e` | implied `e_Sigma_X` | vs. `e_max=0.1` / training anchors 0.05, 0.07 |
|---|---|---|
| 0.02 | ~0.009 | well inside |
| 0.05 | ~0.021 | well inside |
| 0.10 | ~0.043 | inside, below both anchors |
| 0.20 | ~0.086 | inside `e_max` but past the 0.07 anchor — extrapolating |

This axis is fine for `psf_e <= 0.10`; it's noise level and population that
actually block using the existing catalogs as-is.

## 6. What "fixes" this

To test the trained `SigmaXBlockLayer` flow against a PSF-anisotropy
signal with no population or noise mismatch, the catalogs need to be
**`gauss2_fwd`, at `noise_sigma` inside `[1.33, 2.60]`** — `1.86` matches
the 500k baseline and the flow's own training center exactly, so that's
the natural choice, not the geometric-mean or extremes of the range.

`dev/render_psfe.sh` renders `--pop bulgedisc`; it needs a `--pop
gauss2_fwd --noise-sigma 1.86` variant (new script or parameterize the
existing one — see `NEXT_SESSION_psfe_rerender.md`).

## 7. The gauss2_v3d rerender (2026-09-10, this session) — done

Ran `NEXT_SESSION_psfe_rerender.md`'s task: `--pop gauss2_fwd --noise-sigma
1.86`, amplitudes 0.02/0.05/0.10 (0.20 skipped, per that file's reasoning),
`--seed 1` unchanged.

- `dev/render_psfe.sh` now takes `[amplitude] [pop] [noise_sigma]` (both
  optional, default `bulgedisc 0.93` — the default invocation's filenames are
  byte-for-byte what they were before, so nothing already on disk collided).
  A non-default pop/sigma gets a `_<pop>_d<sigma>` filename suffix, e.g.
  `_gauss2fwd_d186_`.
- **Deviation from the plan, found the hard way**: `bulgedisc`'s render is a
  CPU pixel loop, so the script's original 12-way `&`-backgrounded parallelism
  (all 4 configs x 3 arms at once) is free. `gauss2_fwd` goes through
  JAX/GPU for the moment-map inversion — 12 concurrent processes each
  preallocating their own ~75% GPU slab blew out the 16GB card
  (`CUDA_ERROR_OUT_OF_MEMORY`, first attempt at amplitude 0.02). Fully serial
  (one job at a time) fixed the OOM but left the GPU idle between each short
  job's Python/JAX startup — 8 files in ~5 min, worse than the original
  35-min/amplitude estimate implied. Settled on 4-way concurrency for
  non-bulgedisc pops, each process capped via
  `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.225`
  (`0.9/4`) — no OOM, GPU stayed busy. All 36 files rendered clean.
- Rendered files, `../bfd_cnf_imsims/data/targets_v3psf<name><tag>_gauss2fwd_d186_<arm>_20k.fits`,
  `tag` in `{0002, 0005, 0010}` (E=0.02/0.05/0.10), `name` in
  `{e00, e1p, e2p, e1m}`, `arm` in `{g0, g1p02, g1m02}` — 36 files total, all
  present and verified by directory listing.
- `bias.py` `CATALOGS`/`TRAIN_DATA` (bias.py:~264-303, ~372-382): added
  `gauss2_v3d_psfe00` (baseline, taken from the E=0.10 render's own e00 —
  e_psf=0 is amplitude-independent, same reasoning as the bulgedisc entries)
  and `gauss2_v3d_psfe{1p,2p,1m}{02,05,10}`, all pointing at
  `moments_gauss2_fwd_g2v3d.fits` (g2v3d's own prior, PSF-independent latent
  moments, same as the bulgedisc_v3_psfe* entries).
- Sanity check (plan step 5): `python bias.py --flow
  flows/centroid_g2v3d_sigmaxblock_multiscale.eqx --pop gauss2_v3d_psfe1p05
  --n-targets 500 --samples 256 --chunk 256 --batch-budget 4096` ran to
  completion, no population-mismatch guard error, no crash. `--batch-budget`
  had to be capped (default OOM'd the g-Hessian at this catalog's target
  count on this GPU — unrelated to the psfe rerender itself). The printed
  m1/c1/c2 from this run are NOT a real measurement (500 targets, 256
  draws/target, median ESS in the single digits at 4 of 5 flux quintiles) —
  it only confirms the pipeline runs; a real bias measurement needs the
  500k-scale run this section was gating.
- Not done, deliberately out of scope per the plan: the actual 500k-scale
  bias run on these catalogs, and anything involving a bulgedisc-population
  SigmaXBlockLayer flow (§5b — none exists).

## 8. The centroid-layer anisotropic-PSF bias test (2026-09-10, same session as §7)

Goal: with the gauss2_v3d/gauss2_fwd catalogs from §7 in hand, measure how well
`flows/centroid_g2v3d_sigmaxblock_multiscale.eqx` (the only anisotropically
trained `SigmaXBlockLayer`) suppresses a PSF-anisotropy leak into shear.

### 8a. The bias run

`dev/bias_psfe_g2v3d.sh` runs all 10 configs (`psfe00` baseline +
`psfe{1p,2p,1m}{02,05,10}`) through `bias.py`, mirroring `dev/bias_psfe.sh`'s
bulgedisc recipe crossed with `dev/g2v3d_500k.sh`'s gauss2-only flags:
`--samples 8192 --alpha 0.5 --window-size 2.2 3.2 --window-flux 2500 50000
--window-terms score --gauge auto --floor-eps 1e-3 --window-guard 100`.

**Two OOM incidents, both fixed by cutting batch size, not by changing the
science:**
- The render script's original 12-way parallelism (fine for `bulgedisc`'s
  CPU-only pixel loop) OOM'd a 16GB card on `gauss2_fwd`, which goes through
  JAX/GPU for the moment map. Fixed in `dev/render_psfe.sh`: bulgedisc keeps
  12-way `&` parallelism; every other population runs at 4-way concurrency
  with `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.225`
  per process (`GPU_PAR=4`, fraction `0.9/GPU_PAR`).
- `bias.py`'s own g-Hessian OOM'd at the historical `--chunk 4096
  --batch-budget 65536` (16 targets/chunk) that the OLD `CentroidMarginalize`
  flow tolerated — the new `SigmaXBlockLayer` (5 independent coefficient
  nets, richer condition path) needs more memory per batch element. Fixed by
  cutting to `--chunk 2048 --batch-budget 16384` (8 targets/chunk); this
  made each config run noticeably slower than `bias_psfe.sh`'s historical
  ~40 min estimate (closer to ~35-45 min/config here too once past the
  initial mistuning, but a fully-serial 4-way-GPU-render mixup mid-session
  cost real wall time — see the transcript, not repeated here).
- **A duplicate-process incident**: an early `setsid`-launched
  `bias_psfe_g2v3d.sh` was believed to have failed (`Exit code 1` from a
  `disown`/backgrounding quirk) but had actually started; launching a SECOND
  copy on top of it raced both on the GPU and on the shared output
  files/logs, corrupting `logs/bias_g2v3d_psfe00.log` and crashing both.
  **Lesson, now enforced procedurally in every later step of this session:**
  always `ps aux | grep "bias.py --pop"` before launching, and NEVER trust a
  "the launch failed" read without checking for an orphaned process first.

`pqr/g2v3d_psfe*.npz` and `logs/bias_g2v3d_psfe*.log` hold the 10 runs.
`bias.py`'s default `--window-draws` was bumped from `1<<20` to `1<<24` for
all FUTURE runs (see `bias.py`'s own comment there) — these 10 runs still
used the pre-bump in-run default (`2^20`), which is why §8b's afterburn
exists.

### 8b. The 2^24 afterburn

`dev/psfe_afterburn.sh` re-solves each config's selection terms at `2^24`
draws via `dev/window_scan.py --no-scan --no-prefilter-weights
--window-guard 100` (no target re-integration needed — `ghat` is affine in
`P_s/Q_s/R_s`). It polls for the GPU to be idle before each resolve so it
never overlaps `bias.py` (needed: `dev/g2v3d_500k.sh`'s own note that two
JAX processes contest a 16GB card). Selection-term caches land in
`logs/phase_a/terms_gauss2_v3d_psfe*_24_s0_1w_g100.npz`; resolved results in
`logs/resolve_g2v3d_psfe*.log`.

**2^20 in-run vs 2^24 afterburn, windowed-corrected (m1 / c1 / c2):**

| config | 2^20 m1/c1/c2 | 2^24 m1/c1/c2 |
|---|---|---|
| psfe00 | +0.02554±0.01489 / −3.43e-04±4.4e-03 / +2.19e-03±4.1e-03 | +0.02749±0.01495 / +3.40e-04±4.4e-03 / +1.97e-03±4.1e-03 |
| psfe1p02 | +0.02191±0.01432 / −4.26e-04±4.2e-03 / +2.08e-03±4.5e-03 | +0.02386±0.01438 / +2.56e-04±4.2e-03 / +1.86e-03±4.5e-03 |
| psfe2p02 | +0.02641±0.01522 / −7.97e-05±4.2e-03 / +1.92e-03±4.5e-03 | +0.02840±0.01528 / +6.06e-04±4.2e-03 / +1.70e-03±4.5e-03 |
| psfe1m02 | +0.02701±0.01530 / +7.37e-05±4.2e-03 / +2.24e-03±4.4e-03 | +0.02902±0.01536 / +7.58e-04±4.2e-03 / +2.03e-03±4.4e-03 |
| psfe1p05 | +0.02322±0.01674 / −8.63e-04±4.2e-03 / +1.86e-03±4.7e-03 | +0.02516±0.01682 / −1.82e-04±4.3e-03 / +1.65e-03±4.7e-03 |
| psfe2p05 | +0.03463±0.01482 / +1.36e-04±4.3e-03 / +1.42e-03±4.7e-03 | +0.03666±0.01488 / +8.22e-04±4.4e-03 / +1.20e-03±4.7e-03 |
| psfe1m05 | +0.02329±0.01423 / +2.48e-04±4.5e-03 / +2.17e-03±4.1e-03 | +0.02525±0.01427 / +9.31e-04±4.5e-03 / +1.96e-03±4.1e-03 |
| psfe1p10 | +0.01842±0.01496 / −1.08e-03±4.6e-03 / +1.77e-03±4.0e-03 | +0.02043±0.01503 / −4.00e-04±4.7e-03 / +1.55e-03±4.0e-03 |
| psfe2p10 | +0.04361±0.01370 / +1.13e-04±4.2e-03 / +1.44e-03±4.5e-03 | +0.04559±0.01376 / +7.96e-04±4.2e-03 / +1.23e-03±4.5e-03 |
| psfe1m10 | +0.02578±0.01408 / +7.50e-04±4.1e-03 / +2.21e-03±4.7e-03 | +0.02778±0.01414 / +1.44e-03±4.1e-03 / +1.99e-03±4.7e-03 |

**Naive read (independent errors, WRONG for this design):** every one of
these differs from `psfe00` by well under 1σ — looks like a clean null. This
reading is retracted in §8d.

### 8c. `psfe00` was a statistical fluctuation, not a bug

`psfe00`'s own m1 (+0.0275 at 2^24) sits ~1.8-2σ above three independent 20k
`psf_e=0` draws from a prior session (`pqr/g2v3d_sigmaxblock_multiscale_20k{,_b,_c}.npz`,
2^24 m1 = −0.00067, −0.01062, +0.00622). Checked and ruled out as a cause:
population/moment distributions (Mf, Mr, Mr/Mf, window occupancy all match
across all 4 draws to <1%), R11 concentration (median/max |R11| in-window
match, `psfe00` is *less* concentrated than one of the three prior draws),
N_ns reconstruction (prefiltered vs exact count agree to <0.2%), and render
parameters (identical `SEED=1 NOISESIG=1.86 G1=G2=0 IMGNOISE=True` headers,
only `NAXIS2` differs). **Confirmed by direct replication**:
`dev/psfe00_replicate.sh` renders a fresh seed-2 `psf_e=0` 20k draw
(`targets_v3psfe00_gauss2fwd_d186_s2_*_20k.fits`, `CATALOGS` key
`gauss2_v3d_psfe00_s2`), runs it through the identical `bias.py` flags,
lands at 2^24 m1 = **+0.00130 ± 0.01296** — squarely back in the −0.011 to
+0.006 cluster. Conclusion: n=20k galaxy-sample scatter on this estimator is
wide enough that any single draw can land ~2σ off; don't trust any one
config's absolute m1 as a bias measurement, but see §8d for why the
orientation/amplitude *comparison* survives this fine.

### 8d. The real finding: a small, real c1/c2 leak, invisible until you pair correctly

`dev/psfe_paired_diff.py <tag_a> <tag_b>` re-derives what the independent
per-config error bars miss: every psfe config shares `psfe00`'s exact
seed-1 galaxies and pixel noise (only a tiny PSF-ellipticity difference in
the render), so ~99% of each config's own bootstrap error is common-mode
and cancels in a properly-paired difference. Since each config's own
recentring-convergence/outlier drops remove a *different* handful of rows
(so `bias.py --compare` itself refuses these files — `plus_q` lengths range
19937-19941), rows are realigned by nearest-neighbor match on the saved
`moments` field (the zero-arm's raw moment vector — a near-unique per-galaxy
fingerprint since pixel noise is seeded by `(seed, row index)` alone,
independent of PSF ellipticity). Match threshold: `0.5 *` the population's
own median nearest-neighbor gap (tuned once against the actual cross- vs
self-config distance distributions, see the script's docstring for why
`0.01x` — the first attempt — was catastrophically too strict).

This drops the c1/c2 error bars from ~4.2-4.7e-3 (independent/naive) to
~1.6-3.7e-4 (paired) — a **25-30x reduction**, stable across bootstrap
seeds. It reveals a clean, physically-sane signature the naive analysis
completely missed: c1 flips sign between `e1p` (Δc1 negative, growing in
magnitude with amplitude: −0.07e-3 → −0.73e-3 → −0.59e-3 at 0.02/0.05/0.10)
and `e1m` (Δc1 positive and growing: +0.42e-3 → +0.87e-3 → +1.08e-3),
exactly as required since `psf_e1` flips sign between them; `e2p` pulls c2
consistently negative and growing (−0.18e-3 → −0.59e-3 → −0.88e-3). m1 shows
no comparable pattern (all points <1σ, slope consistent with zero) — the
right contrast, since a PSF leak should be additive (c-term), not
multiplicative.

**Pooling for a rigorous combined slope** (`dev/psfe_joint_slope.py`): a
single joint bootstrap (one shared galaxy resample per draw, applied via
each config's own pairwise match map — NOT a 10-way simultaneous
intersection, which collapses the usable sample from ~19700 to ~2300 and is
a dead end) gives:

  - all 9 configs: dc1/d(psf_e1) = −0.0102 ± 0.0023 (4.3σ), dc2/d(psf_e2) = −0.0096 ± 0.0032 (3.0σ)
  - dropping the degraded-match 0.10-amplitude configs: dc1/d(psf_e1) = **−0.0156 ± 0.0030 (5.3σ)**, dc2/d(psf_e2) = **−0.0115 ± 0.0036 (3.2σ)**
  - 0.02-amplitude alone (cleanest matches, least leverage): dc1/d(psf_e1) = −0.0123 ± 0.0050 (2.5σ), dc2/d(psf_e2) = −0.0091 ± 0.0057 (1.6σ)

Point estimate is stable (−0.010 to −0.016 for c1, −0.009 to −0.012 for c2)
across every subset choice — the strongest evidence this is real rather than
a fit artifact.

**Known limitation, unresolved this session:** the nearest-neighbor match
rate degrades sharply with amplitude (0.02: ~99%, 0.05: ~70%, 0.10: ~40%) as
the PSF-induced moment shift starts rivaling the population's own
nearest-neighbor spacing — the 0.10-amplitude points are built from a
smaller, possibly-biased subsample (whichever galaxies happen to shift
least), which is consistent with why including them *attenuates* the fitted
slope toward zero rather than sharpening it. The real fix is having
`bias.py`/`save_pqr` record the original population row index directly
(exact identity, no approximate matching needed) — not done this session,
would need a rerun.

**Cross-check against prior work, and what it implies about the training:**
[[psf-leak-is-the-sigma-x-channel]] found essentially the SAME leak
(`dc2 ~ e_psf^1.07`, e.g. `dc2 = -5.1e-04` at `e_psf=0.05`) on `bulgedisc_v3`
with the OLD `CentroidMarginalize` layer (no anisotropic Sigma_X training at
all). This session's `dc2` at the same amplitude (~-0.05*0.0115 ≈ -5.8e-04,
from the pooled slope) is the **same sign and same order of magnitude** as
the pre-anisotropic-training result on a different population. That is
mildly concerning for "how well did the centroid layer perform": it
suggests the anisotropic `SigmaXBlockLayer` training may not have
meaningfully suppressed this leak relative to a flow with no anisotropic
Sigma_X handling at all. **Not yet confirmed** — the rigorous test is the
`--no-centroid` bisection (`--flow flows/shear_g2v3d.eqx --no-centroid` on
the same 10 configs, mirroring `dev/bias_psfe.sh`'s own bisect mode), which
would show whether the leak is *larger* without the layer (proving partial
suppression) or the *same* (no suppression at all). Proposed but not run
this session — recommended next step.

### 8e. Estimator-behavior side findings, general (not psfe-specific)

- **The ±g antithetic pairing helps m1 enormously, and mildly HURTS c1/c2.**
  `dev/plus_minus_pairing_check.py` compares the standard paired bootstrap
  (same resampled galaxy index for both `+g`/`-g` arms) against an
  "unpaired" variant (independent index per arm) on
  `pqr/g2v3d_sigmaxblock_multiscale_500k.npz`: unpaired m1 error is
  10-53x LARGER than paired (this pairing is *why* m1 is measurable at all
  — it comes from a difference, and correlating the two arms cancels their
  shared shape/pixel noise in that difference). But c1/c2 come from the
  *sum* `(gp+gm)/2`, where the same positive correlation that helps a
  difference INFLATES a sum's variance — paired c1/c2 bootstrap error is
  ~30-40% LARGER than the unpaired version would be. Not a bug; a genuine
  property of antithetic variates applied to a sum vs. a difference.
- **A naive delta-method ("formal") error matches bootstrap well
  unwindowed, but badly for windowed m1.** `dev/formal_vs_bootstrap.py`
  derives the standard M-estimator sandwich variance (`Var(g) = R^-1
  Var(Q) R^-T`, treating `R = -SUM r` as a fixed plug-in Hessian) directly
  from the per-galaxy `q` vectors, no resampling. On the unwindowed full
  population it agrees with the bootstrap to 1-2% (a good self-consistency
  check on the formula). Under the size/flux window it's fine for c1/c2
  (~1.05x) but **wildly overestimates windowed m1's error**: 7.4x too large
  uncorrected, 1.4x too large corrected. Cause: the window shrinks the
  Hessian-estimating subset to ~46% of the catalog, so `R` itself carries
  real sampling noise correlated with `Q`'s — the bootstrap captures that
  (it resamples `R` and `Q` together every draw); the fixed-`R`
  delta-method formula can't. **Trust the bootstrap for windowed m1**; the
  formal number isn't catching a real extra effect, it's missing one.

### 8f. Files created this session (all under this repo unless noted)

- `dev/render_psfe.sh` — parameterized (`[amplitude] [pop] [noise_sigma]`),
  4-way-concurrent GPU path added for non-bulgedisc pops.
- `dev/bias_psfe_g2v3d.sh` — the 10-config gauss2_v3d psfe bias driver.
- `dev/psfe_afterburn.sh` — 2^24 selection-term re-solve, GPU-idle-gated.
- `dev/psfe00_replicate.sh` — the seed-2 independent psf_e=0 replicate
  (render + bias.py + afterburn in one script).
- `dev/psfe_paired_diff.py` — one-pair matched-pairs paired-bootstrap diff.
- `dev/psfe_joint_slope.py` — pooled joint-bootstrap slope fit across all 9
  configs (`dev/psfe_joint_slope_no10.py`/`_02only.py` are throwaway
  amplitude-subset variants made ad hoc during this session, not meant to
  be kept as permanent tools).
- `dev/plus_minus_pairing_check.py` — paired vs. unpaired ±g bootstrap.
- `dev/formal_vs_bootstrap.py` — delta-method vs. bootstrap error.
- `bias.py`: `CATALOGS`/`TRAIN_DATA` entries for `gauss2_v3d_psfe00`,
  `gauss2_v3d_psfe{1p,2p,1m}{02,05,10}`, `gauss2_v3d_psfe00_s2`; default
  `--window-draws` bumped `1<<20 -> 1<<24`.
- `../bfd_cnf_imsims/data/targets_v3psfe*_gauss2fwd_d186[_s2]_*_20k.fits` —
  36 (main grid) + 3 (seed-2 replicate) catalogs.
- `pqr/g2v3d_psfe*.npz`, `pqr/g2v3d_psfe00_s2.npz` — saved PQR for every run
  above; `logs/bias_g2v3d_psfe*.log`, `logs/resolve_g2v3d_psfe*.log`,
  `logs/phase_a/terms_gauss2_v3d_psfe*_24_*.npz` — the corresponding logs
  and selection-term caches.

## How this was recovered

`git log` and `dev/psfe_compare.py`/`dev/render_psfe.sh`'s own comments
only describe the **catalogs** (§1), which are old (2026-09-04-ish) and
well documented. The **flow training** (§§2-4) has zero trace in git,
`~/.bash_history`, `~/.zsh_history`, or `~/.local/share/fish/fish_history`
(`grep -c "centroid.py"` → 0 hits) — the work was done entirely through
Claude Code's own `Bash` tool in two sessions
(`b36f2ac9-b920-4af3-9d82-b8e2f0ec31ac` and
`d1c891d4-5728-483a-8b31-9259bc83b314`), whose tool calls don't land in
the user's interactive shell history. Recovered by:

1. `find`-ing `/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/*/scratchpad/train_sigmax*.log` for the actual training curves and `check` output.
2. Matching those scratchpad paths back to their owning session directory to identify which session ran them.
3. `grep`-ing each session's (and subagent's) `.jsonl` for `Bash` `tool_use` blocks containing `train-sigmax`, and cross-checking each command's timestamp (UTC) against the target `.eqx` file's mtime (local, UTC-4) to confirm which command produced which file — and, for the path that was overwritten five times in one subagent, which was the *last* write.
