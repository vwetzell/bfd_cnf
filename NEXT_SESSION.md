# Next session — bfd_cnf

Written 2026-09-13, after a repo cleanup that cut `dev/` from ~150 scripts to
the ~40 still-live ones (see `git log` for the two cleanup commits on
`feat/centroid-shear-conditioning`, and the removed `NEXT_SESSION_*.md` /
`NEXT_PROMPT.txt` files this supersedes). `README.md` is still the canonical
description of the pipeline; this file is just the open-thread task list.

## Where things stand

`bulk_g2v3d_full2.eqx` / `shear_g2v3d_full2.eqx` (2026-09-12) is the current
best flow: `dm/dg` RMS residual against bfd's exact derivatives is 1.4-2.7%
(`logs/shear_g2v3d_full2.log`), after `models/shear.py` got a split
coefficient bound (`_COEFF_MAX_SPIN0`) and a point-source edge taper
(`_edge_factor`) — see commit `4e48ade`. `bias.py` also gained a
nearest-neighbour support-density guard (`build_support_density`,
`in_support_density`) for selection-term draws, in the same commit.

Two threads are still open on top of that.

## Thread 1 — PSF-anisotropy (PSFE) leak

Full reasoning in `PSFE_PROVENANCE.md` — read that first. Short version: the
ellipticity of the PSF leaks into `dc2` through the `Sigma_X` channel (linear
in `e_psf`, `[[psf-leak-is-the-sigma-x-channel]]`), confirmed again on the
`gauss2_v3d`/mcfix grid at ~3σ (`logs/bias_g2v3d_psfe*_mcfix.log`,
`m1 ~ +0.026-0.028` corrected, consistent across `psfe00`/`psfe1p05`).

Live tooling (all in `dev/`): `psfe_compare.py`, `psfe_joint_slope*.py`,
`psfe_angle_diagnostic.py`, `psfe_paired_diff.py`, `psfe_size_offline.py` (+
`_reseed.py`), `psfe_ess_by_orientation.py`, `psf_anisotropy_probe.py`, driven
by `bias_psfe*.sh` / `render_psfe.sh` / `psfe_afterburn*.sh`. Cached selection
terms live in `logs/phase_a/terms_*_mcfix.npz`.

**Not yet answered**: whether tightening the size ceiling shrinks `dc1`/`dc2`
(`psfe_size_offline.py`'s question, added 2026-09-11 — check if it ran to a
conclusion before picking this back up) and whether the leak's angular
dependence (`psfe_angle_diagnostic.py`, `psfe_ess_by_orientation.py`) narrows
it further. `NEXT_SESSION_psfe_rerender.md` (now removed) had a rerender plan
for `gauss2_fwd`/`noise_sigma=1.86` catalogs to match the trained
SigmaXBlockLayer's coverage — check whether that rerender happened before
re-deriving it; if not, it's still the right next step for testing this
against a population/depth the flow was actually trained on.

## Thread 2 — leader-draw / R_s instability

New 2026-09-12. `selection_terms_score`'s score-function estimator
concentrates on single draws: on `gauss2_v3d` one draw carried 61% of `R_s11`
before the density guard, 9% after it. The guard (`build_support_density`,
`bias.py`) rejects draws whose shape sits outside the training catalog's own
nearest-neighbour manifold — but the analysis in that function's docstring
found a real limit: the worst leader sits deep inside real-template density
(k-NN distance 0.46 vs. cutoff 4.6) despite having an unphysical exact
preimage (`log_p_theta = -inf` on `gauss2`'s known closed form) — the
reachable manifold folds back on itself near enough that no density estimate
over shape invariants alone can catch it.

Live tooling (`dev/`): `leader_census_logp.py`, `leader_logp_check.py` (does
the flow's own log-prob flag the leader as anomalous — if yes, a log-prob
cutoff might catch what the density guard can't), `lipschitz_check.py` (is
the leader's local Q/R sensitivity anomalous vs. real-template neighbors),
`fold_check.py` (is the Mr/Mf~3.0-3.3 fold a genuine feature of the true
moment map, checked analytically against `truth.py`), `boundary_shape_check.py`
(is the reachable-set boundary predictable from shape alone).

**Not yet answered**: whether any of these four checks found a usable second
signal (log-prob, local Lipschitz constant, fold geometry) to catch what the
k-NN density guard misses. Run them and read their output before building
anything new — they were written but not yet reported on as of the cleanup.

## Working rules (carried over from HANDOFF.md, still true)

- `shear.py train` on `bulgedisc` needs `--deriv-weight 1e4`.
- `--alpha 0.5` for any `_deep` population — `1.0` doesn't converge.
- `pqr_streamed` seeds chunk `c` as `seed + 7919*c`: raising `S` at a fixed
  seed nests the smaller draw set, so an `S` scan at one seed is not a
  convergence check.
- Two JAX processes do not fit on the 16GB card at `2^24` — serialize GPU
  jobs, and prefer `dev/window_scan.py`'s cache (`logs/phase_a/terms_*.npz`)
  for a re-solve so it doesn't touch the GPU at all.
