Read HANDOFF.md's last several sections in full before doing anything else
-- everything from "2026-08-27: the centroid layer is now an exact,
zero-parameter analytic transport" through the final entry, "'still not
good enough' / 'keep trying the bulge disc' -- a third stale target, then
a log-normal sigma closes most of the remaining gap". Branch:
`feat/centroid-shear-conditioning`.

## Where things stand

Last session did two things, in order:

1. Landed the analytic (zero-free-parameter) centroid marginalisation
   layer in `models/centroid.py` and confirmed it generalises from
   `gauss2` to `bulgedisc` (see the two 2026-08-27 entries on that).
2. Spent the rest of the session retuning `../bfd_cnf_imsims/imsims/
   sim.py`'s `bulgedisc` population against real DES/COSMOS templates
   (`~/Documents/BFD_cNF/summary_templates_new.fits`), via the corner
   plot in `dev/check_bulgedisc_vs_real.py`. This was NOT asked to reach
   a specific number -- the user's calls throughout were "still not good
   enough" / "keep trying" / "close enough", so read the last several
   HANDOFF entries to understand what "close enough" actually landed on:
   `Mr/Mf` median 3.14 (target 3.50), std 0.43 (target 0.47), median
   `|e|` 0.090 (target 0.089, near-exact). NOT a perfect match, but the
   user was satisfied enough to move on to this task.

**`../bfd_cnf_imsims` has UNCOMMITTED changes** (`fit_gauss2_p0.py`,
`imsims/sim.py`, `tests/test_sim.py`) -- `git status`/`git diff` there
before doing anything else so you know exactly what changed. Consider
whether to commit them (ask the user if unsure; they were never asked
for this session, only edited).

**Every existing `bulgedisc`-population artifact on disk is now STALE**
against the retuned population: `../bfd_cnf_imsims/data/moments.fits`
(if it's the bulgedisc training set), `targets_deep_g0_200k.fits`/
`targets_deep_g1p02_200k.fits`/`targets_deep_g1m02_200k.fits`,
`copies_bulgedisc_deep.fits`, and every `flows/*bulgedisc*.eqx`
checkpoint (`bulk_bulgedisc.eqx`, `shear_bulgedisc_fresh.eqx`,
`centroid_bulgedisc_deep_analytic.eqx`). They were trained/rendered
against the population BEFORE this session's flux/size/ellipticity
retune. Don't reuse them -- render and train fresh, with new filenames
so the stale ones stay around for comparison rather than being silently
overwritten (e.g. `_v2` or a date suffix -- your call).

## Task

Pick up training a new flow on the retuned `bulgedisc` image sims, and
measure the multiplicative bias on deep (noisy) targets from the SAME
retuned population. Concretely, in `../bfd_cnf_imsims` then back in
`bfd_cnf`:

1. Render a noiseless training catalog: `python -u -m imsims.sim --n
   <N> --pop bulgedisc --out data/<new-name>.fits` (the last session used
   `--n 40000` for corner-plot checks; the prior full training run used
   a much larger N -- check `HANDOFF.md`'s "the analytic transport
   generalises to bulgedisc" entry, which used
   `bulk.py train --data moments.fits --steps 20000` at whatever N
   `moments.fits` was rendered at, and pick something comparable or
   larger for a real training run, not the 40k smoke-test size).
2. `bulk.py train --data <new-moments>.fits --steps 20000` (or more --
   `--steps 20000` was "clean, val nll 40.42" last time this population
   was trained, but that was the PRE-retune population).
3. `shear.py train --data <new-moments>.fits --deriv-weight 1e4 --steps
   60000` -- **`--deriv-weight 1e4` is not optional for this
   population**, confirmed twice now (`shear.py --help` explains why:
   NLL alone cannot identify the spin-0 response). Do not skip it or
   treat it as a tuning knob to search over.
4. Render deep (noisy) targets at the established depth (`noise_sigma`
   2.73, image-level noise + `recenter()`, matched +g/-g/g0 triplet) --
   check `imsims/sim.py`'s `CATALOGS`/`--add-noise`/`--noise-sigma`
   plumbing and the `bulgedisc_deep` comment block in `bias.py`'s
   `CATALOGS` dict for the exact recipe (200k targets each of
   `g1p02`/`g1m02`/`g0` last time).
5. Render `copies_bulgedisc.fits` fresh (`python -u -m imsims.copies
   --n <N> --pop bulgedisc --out data/<new-copies-name>.fits`) and run
   `centroid.py train --copies <...> --flow <...> --init <shear
   checkpoint>` -- this is a warm-start + `check` only, there is nothing
   left to train in the centroid layer (zero free parameters); don't
   wait on a training loop.
6. Back in `bfd_cnf`: `bias.py --flow <new centroid checkpoint> --pop
   bulgedisc_deep --samples 8192 --alpha 0.5 --chunk 4096
   --batch-budget 65536 --n-targets 20000`. **Use `--alpha 0.5`, not the
   default 1.0** -- confirmed this session and the one before that
   `--alpha 1.0` is ESS-starved at this noise depth and gives an
   unreliable, differently-signed `m1` (see
   `[[bulgedisc-deep-centroid-q2q3-blowup]]` memory / the "q2/q3
   instability" HANDOFF entry).

## What to actually check once you have a number

- Report `m1` (and the by-quintile breakdown `bias.py` prints)
  plainly, with whatever precision the run supports.
- The PRIOR `bulgedisc_deep` run (against the OLD, pre-retune
  population) found `m1 = +0.1627 +/- 0.0735`, dominated by q2/q3's
  near-uninformative variance while q1/q4/q5 were all within ~1sigma of
  zero -- an unresolved, real anomaly (see
  `[[bulgedisc-deep-centroid-q2q3-blowup]]`), NOT explained by ESS,
  T-saturation, or any mechanism chased down at the time. Check whether
  the SAME q2/q3 instability shows up again on the retuned population,
  or whether the more realistic size/flux/ellipticity distribution
  changes which quintiles (if any) are unstable. This is a genuinely
  open question, not a formality -- the retuned population's `Mr/Mf`
  distribution is meaningfully different in shape (log-normal-ish, less
  ceiling-crowded) from what produced the original instability, so it
  could easily behave differently.
- `centroid.py check`'s own table (layer shift vs. catalog shift, plus
  the ellipticity-response ratio) is worth reporting too, the same way
  the prior `bulgedisc` entry did -- it's the cleaner, lower-noise signal
  and was "as good as or better than gauss2_deep's" match last time,
  worth seeing whether that held up through the retune.
- No window (`--window-size`/`--window-flux`) has ever been established
  for `bulgedisc_deep`. Establishing one is legitimate follow-up work if
  the unwindowed number turns out too noisy to interpret (mirroring what
  `gauss2_deep`'s window took several sessions to land on) -- but don't
  reach for it reflexively; check whether the retuned population's
  smoother `Mr/Mf` distribution alone stabilises q2/q3 first.

**Gotcha to avoid:** `bias.py`'s `CATALOGS`/train-data lookup hardcodes
`"bulgedisc_deep": "moments.fits"` as the default `--train-data` (used
for the flow's standardisation, and for `--window-terms templates`). If
you render the new training catalog under a different filename, either
pass `bias.py --train-data <new-name>.fits` explicitly every time, or
update that dict entry -- otherwise `bias.py` will silently standardise
against the OLD, stale `moments.fits` while your flow was trained on the
new one, which would quietly corrupt the bias measurement rather than
error out.

## Working rules (carried over, still true)

- `shear.py train` on `bulgedisc` needs `--deriv-weight 1e4` -- see
  above, already learned the hard way twice.
- `centroid.py train` has nothing to train any more (zero free
  parameters) -- it warm-starts + runs `check`; don't wait on a training
  loop for it.
- `bias.py` on a `_deep` population needs `--batch-budget 65536` or it
  OOMs the 16GB card; don't run two GPU jobs concurrently.
- Always use `--alpha 0.5` for `bulgedisc_deep` (see above).
- `python -u <script>` from the relevant repo root.
- Any real-data numeric target pulled from `summary_templates_new.fits`
  for a NEW comparison must be re-verified against the exact filtering
  path of whatever it's being compared to -- this bit the previous
  session three separate times (flux target twice, `|e|` target once).
  Don't assume a number from HANDOFF.md's history is still the exact
  target to hit without re-deriving it the same way the comparison
  itself computes it.
