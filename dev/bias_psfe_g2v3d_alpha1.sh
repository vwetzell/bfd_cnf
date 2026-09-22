#!/bin/bash
# The gauss2_v3d PSF-anisotropy bias runs, mirroring dev/bias_psfe.sh's
# structure but on the population/depth/flow the trained SigmaXBlockLayer
# actually covers -- see PSFE_PROVENANCE.md #6-7 and
# NEXT_SESSION_psfe_rerender.md for why bulgedisc_v3_psfe* can't be used to
# test that flow.
#
# UPDATED 2026-09-14 per NEXT_SESSION.md "Thread 1 -- RESUME CONFIG": the
# flow and window below are Thread 2/3's fix, not the original ones (see git
# history for the pre-update version). Flags now mirror the validated-null
# 20k pooled score-boundary scan exactly:
#   flow flows/centroid_g2v3d_full2_jac.eqx   Thread 2's --jac-weight fix
#     (centroid_g2v3d_sigmaxblock_multiscale.eqx is the pre-fix flow).
#   --window-size 2.2 3.2 --window-flux 1500 20000   the ONE window today's
#     scan validated null (m1 = -0.007 +/- 0.011 pooled); flux_hi=50000
#     (the old value) was independently shown biased at 3.3 sigma -- do not
#     widen without re-validating.
#   --samples 8192 --alpha 1.0 --chunk 2048 --batch-budget 16384
#     --no-zero-arm --support --floor-eps 0   from bias_g2v3d_full2_jac_
#     centroid_20k.log, NOT the old --gauge auto --floor-eps 1e-3
#     --window-guard 100 (that combo was measured against the OLD flow's
#     catastrophe and is no longer the validated path, though it agreed to
#     0.0006 on the wider window when cross-checked).
#   --window-terms score --window-draws 16777216   2^24, not 2^20 -- smaller
#     draw counts are NOT validated on this window (leader share was as high
#     as 341% at 2^20 on the old window, purely from undersampling).
#
# ONE JAX GPU JOB AT A TIME, ~40 min each per bias_psfe.sh's own timing note.
# Minimal rerun: baseline + one orientation (psfe1) x all 3 amplitudes = 4
# configs, budget ~2.5-3 h serial.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

FLOW=flows/centroid_g2v3d_full2_jac.eqx

for pop in gauss2_v3d_psfe00 \
           gauss2_v3d_psfe1p02 gauss2_v3d_psfe1p05 gauss2_v3d_psfe1p10; do
    tag=${pop#gauss2_v3d_}
    python -u bias.py --pop "$pop" --flow $FLOW \
        --samples 8192 --alpha 1.0 --chunk 2048 --batch-budget 16384 \
        --n-targets 20000 --no-zero-arm --support --floor-eps 0 \
        --window-size 2.2 3.2 --window-flux 1500 20000 \
        --window-terms score --window-draws 16777216 \
        --save-pqr "pqr/g2v3d_full2_jac_alpha1_${tag}.npz" \
        2>&1 | tee "logs/bias_g2v3d_full2_jac_alpha1_${tag}.log"
done
echo "bias g2v3d psfe (full2_jac, restricted window) done"
