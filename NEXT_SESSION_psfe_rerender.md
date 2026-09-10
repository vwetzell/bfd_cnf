# NEXT SESSION — rerender the psf_e = 0.02/0.05/0.10 catalogs at a depth and population the trained flow can actually use

Written 2026-09-10. Read `PSFE_PROVENANCE.md` first — it has the full
reasoning; this file is just the task.

## Why

`dev/render_psfe.sh` renders `--pop bulgedisc --noise-sigma 0.93`. The only
SigmaXBlockLayer flow trained with anisotropic Sigma_X coverage,
`flows/centroid_g2v3d_sigmaxblock_multiscale.eqx`, was trained on
`gauss2_fwd` population copies at `noise_sigma = 1.86`, with `--multi-scale`
extending coverage to `noise_sigma ∈ [1.33, 2.60]`. The existing psfe
catalogs are the wrong population AND outside the trained noise range (0.93
< 1.33) — see `PSFE_PROVENANCE.md` §5 for the full compatibility table.

So: new catalogs, `--pop gauss2_fwd --noise-sigma 1.86` (the flow's own
training center, and exactly the 500k baseline's depth), for amplitudes
0.02, 0.05, 0.10 — skip 0.20 for now, it's the one config that extrapolates
past the anisotropic training anchors (`e_Sigma_X ≈ 0.086` vs. the 0.07
anchor).

## Task

1. **Parameterize `dev/render_psfe.sh`** (or copy it — your call, but don't
   silently overwrite the bulgedisc/0.93 version, other things may still
   reference it) so `--pop` and `--noise-sigma` aren't hardcoded. Minimal
   version: add `POP=${2:-bulgedisc}` and `SIGMA=${3:-0.93}` (script
   currently takes amplitude as `$1`), and change the render call's
   `--pop bulgedisc` → `--pop $POP` / `--noise-sigma 0.93` → `--noise-sigma
   $SIGMA`.

2. **Filenames must not collide** with the existing bulgedisc/0.93 catalogs
   (`targets_v3psfe<name><TAG>_<arm>_20k.fits`). Add a population+depth tag,
   e.g. `targets_v3psfe<name><TAG>_gauss2fwd_d186_<arm>_20k.fits`, or put
   them in a separate subdirectory. Check `../bfd_cnf_imsims/data/` for
   what's already there before choosing — don't guess, look.

3. **Render** for `E in 0.02 0.05 0.10`, `POP=gauss2_fwd`, `SIGMA=1.86`:
   ```
   for E in 0.02 0.05 0.10; do
       bash dev/render_psfe.sh $E gauss2_fwd 1.86
   done
   ```
   Each call renders 4 configs (`e00`, `e1p`, `e2p`, `e1m`) x 3 arms
   (`g0`, `g1p02`, `g1m02`) = 12 files, ~35 min total per the script's own
   cost comment (scale for 3 amplitudes). Keep `--seed 1` as-is — do not
   change it, the pairing is what makes the signal resolvable, and `e00`
   at each amplitude is redundant work (same seed, same population, same
   noise → identical catalog) but cheap enough not to bother deduplicating
   unless you want to.

4. **`--n 20000`, not 200000`** — matches the existing psfe convention and
   `dev/psfe_compare.py`'s expectations. Don't reuse `targets_g2v3d_g0_500k`
   rows for the `e00` baseline; `sample_population` isn't a prefix-stable
   sequence across `--n` (same reasoning as the script's own header comment
   about the bulgedisc `e00` re-render).

5. **Before spending the 500k-scale GPU time on a bias run**, sanity-check
   one config end to end first (e.g. `psfe1p05`) — run `bias.py` at 20k on
   just that config against `flows/centroid_g2v3d_sigmaxblock_multiscale.eqx`
   and confirm it loads/runs without the population-mismatch guard bias.py
   might have (`_CENTROID_LAYER_TYPES`/`TRAIN_DATA` keying) tripping. If
   `bias.py` doesn't already have a `CATALOGS`/`TRAIN_DATA` entry for this
   new population+depth+psf_e combination, you'll need to add one following
   the `gauss2_v3d` pattern (`bias.py:298-301`).

6. **Update `PSFE_PROVENANCE.md`** with what actually got rendered (exact
   filenames, any deviations from this plan) once done — don't let this
   round trip into the same "nobody recorded the command" hole as the
   SigmaXBlockLayer flows did.

## Not in scope for this session

- No bulgedisc-population SigmaXBlockLayer flow exists yet
  (`PSFE_PROVENANCE.md` §5b). If the eventual goal is testing PSF
  anisotropy against the bulgedisc population specifically (matching the
  *existing* 0.93-depth catalogs instead of re-rendering), that's a
  separate, larger task: train a new `SigmaXBlockLayer` on
  `copies_bulgedisc_v3.fits`-equivalent data first. Don't start that here
  unless asked — this session is just the rerender.
- Don't touch `flows/centroid_g2v3d_sigmaxblock.eqx` or `_v2.eqx` — only
  `_multiscale.eqx` has anisotropic coverage; the other two are dead ends
  for this purpose (see `PSFE_PROVENANCE.md` §3).
