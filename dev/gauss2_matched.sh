#!/bin/bash
# Both gauss2 arms on MATCHED settings: --floor-eps 1e-3 --window-guard 100.
#
# Why each knob, both measured 2026-09-05/06 and not guesses:
#
#   --floor-eps 1e-3   The floor's job is NOT rescuing zero-weight targets (it
#                      rescues none: 4364 at eps 0, 1e-3 and 1e-2 alike).  It
#                      kills GARBAGE-SCORE VARIANCE -- where the flow's
#                      extrapolation returns nonsense, d(mixed)/d(lp) ~ 0, so a
#                      nonsense score contributes nothing instead of nonsense.
#                      On the deep arm that took the uncorrected bar from
#                      +/-116.64 to +/-0.00599.  The SHALLOW arm was never run
#                      with it and still carries +/-0.0366 -- 6x the deep arm's
#                      bar on the catalog with HALF the noise, which is
#                      backwards and is what run 1 tests.
#                      eps is not sensitive: a decade moves m1 by 5e-05.
#
#   --window-guard 100 `sane_targets` for the prior sample.  Removes the
#                      unphysical draws that carry `R_s`'s heavy tail
#                      ([[rs-tail-is-unphysical-draws]]): 91% of the leaders
#                      invert to no galaxy at all.  Converged -- 1000 -> 100
#                      moves corrected m1 by 8e-04, below tau, while taking the
#                      top-draw share 9% -> 0.8%.
#
# The deep arm has only ever run at guard 1000 (top draw still 8%), so run 2
# also closes that gap.  After this both arms differ ONLY in depth, which is
# the comparison the pair was rendered for.
#
# THE SIZE WINDOW IS (2.2, 3.2).  It was `2.2 1e9` here and in every other
# script until 2026-09-06, and the ceiling is not cosmetic:
#
#   * without it the window admits objects whose measured Mr/Mf is at or past
#     `POINT_SOURCE` = 3.692575 -- SMALLER THAN A POINT SOURCE, noise-scattered
#     and physically impossible.  They are 8.05% of the no-ceiling window;
#     `in_domain` rejects most of their kernel cloud, so their per-target
#     integral barely converges and they carry essentially ALL of the deep
#     arm's draw noise (seed-to-seed spread on windowed uncorrected m1: 0.01815
#     without the ceiling, 0.00179 with it);
#   * it MOVES THE ANSWER.  Uncorrected m1 goes -0.04698 -> +0.03196 (shallow)
#     and -0.05122 -> +0.02753 (deep).
#
# Re-solved off the saved PQR at 2^24 (`dev/window_scan.py`), gauss2 still
# passes and the arms now agree:
#
#   g2v3   corrected m1 = -0.00344 +/- 0.01398   (0.25 sigma from zero)
#   g2v3d  corrected m1 = -0.00228 +/- 0.01624   (0.14 sigma)
#
# against -0.00128 +/- 0.01196 and -0.01738 +/- 0.01251 at the void window --
# i.e. the deep arm's old 1.4-sigma tension WAS the missing ceiling.  The price
# is that `R_s12` runs 12-13% of `R_s11` instead of 2.4%: eq. (45)-(46) is
# harder to converge with a ceiling ([[size-ceiling-correction-fails]]), so
# read the isotropy monitors before believing a corrected number here.
#
# ONE JAX GPU JOB AT A TIME.  ~40 min each.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

COMMON="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536
        --n-targets 20000 --gauge auto --window-size 2.2 3.2
        --window-flux 2500 50000 --window-terms score
        --window-draws 16777216 --floor-eps 1e-3 --window-guard 100"

echo "=== gauss2_v3  floor 1e-3 + guard 100 ==="
python -u bias.py --pop gauss2_v3 --flow flows/centroid_g2v3.eqx $COMMON \
    --save-pqr pqr/g2v3_20k_matched.npz 2>&1 | tee logs/bias_g2v3_matched.log

echo "=== gauss2_v3d  floor 1e-3 + guard 100 ==="
python -u bias.py --pop gauss2_v3d --flow flows/centroid_g2v3d.eqx $COMMON \
    --save-pqr pqr/g2v3d_20k_matched.npz 2>&1 | tee logs/bias_g2v3d_matched.log

echo "MATCHED SCAN DONE"
