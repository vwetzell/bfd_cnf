# Where to start — bfd_cnf

This file used to be a full snapshot of the repo's state, and it rotted within
two days of being written.  It is now an index, and the three documents it
points at are the ones kept current.

## What this project is

A conditional normalizing flow standing in for the finite template set of the
BFD shear estimator (Bernstein et al. 2016, MNRAS 459, 4467).  The flow *is*
`P(m | g, Sigma_X)`, the density eq. (38) approximates with a sum over
delta functions; `bias.py` then measures the multiplicative and additive shear
bias it produces against known injected shear on simulated populations.

## The documents

| | |
|---|---|
| `README.md` | **Current state.**  What each phase does, how to run it, and every number that is believed today. Read this first. |
| `HANDOFF.md` | The dated investigation log, 2026-08-12 onward. Read its top-of-file warning before trusting anything in the older half. |
| `NOISY_BINNING_BIAS.md` | Why binning a noisy `m1` profile on a latent quantity biases it even under a perfect prior. Read before believing any binned noisy profile anywhere. |
| git log | Where the 2026-08-18/19 sessions' detail lives — the commit messages are long and carry the measurements. |

## The numbers that matter, as of 2026-08-19

* Noiseless, bulgedisc, converged chain: `m1 = -0.06 +/- 0.01`.  Limited by
  the **spin-0 shear response**, which is an architecture question, not a
  training one.
* Noisy end to end (image noise, found centroid, centroid layer, `C_M`
  integral), in the analysis window with eq. (45)-(46)'s selection terms:
  consistent with zero, error-bar limited.
* Those two disagree by 0.06 on the *same flows*, and that is the top open
  question.  See README.md's "What is not measured".

## Not implemented

The Poisson/sky branch (eq. 53-55), magnification, multi-band/multi-exposure,
and a `C_M` that varies across the catalog.  Everything here is the
postage-stamp branch with one noise level.
