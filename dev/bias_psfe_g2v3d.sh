#!/bin/bash
# The gauss2_v3d PSF-anisotropy bias runs, mirroring dev/bias_psfe.sh's
# structure but on the population/depth/flow the trained SigmaXBlockLayer
# actually covers -- see PSFE_PROVENANCE.md #6-7 and
# NEXT_SESSION_psfe_rerender.md for why bulgedisc_v3_psfe* can't be used to
# test that flow.
#
# Flags mirror dev/g2v3d_500k.sh (gauss2_v3d's own bias recipe) crossed with
# dev/bias_psfe.sh (the psfe driver's window/gauge/samples):
#   --floor-eps 1e-3 --window-guard 100   gauss2-only, mandatory
#     ([[gauss2-passes-with-floor-and-guard]] -- without them the deep arm
#     returned m1 = -5.75 +/- 116.64).
#   --window-size 2.2 3.2 --window-flux 2500 50000   the same window every
#     gauss2_v3d/bulgedisc_v3 bias run uses (bulgedisc_v3_psfe*, g2v3d_500k).
#   --samples 8192 --alpha 0.5 --chunk 2048 --batch-budget 16384
#     --window-draws 1048576   the 20k-target psfe scale, from bias_psfe.sh,
#     with chunk/batch-budget cut 4x from bias_psfe.sh's bulgedisc numbers:
#     the SigmaXBlockLayer flow (5 independent coefficient nets, more params
#     in the condition path than the old single-trace CentroidMarginalize)
#     OOM'd the g-Hessian at 65536/4096 on a clean 16GB card (measured,
#     "gauss2_v3d_psfe00", plus arm) even with no other GPU process running.
#   --gauge auto --window-terms score   mandatory (R is an IS artifact under
#     gauge P; templates has no on-data analogue).
#
# ONE JAX GPU JOB AT A TIME, ~40 min each per bias_psfe.sh's own timing note
# -- ten configs here (one baseline + 3 orientations x 3 amplitudes) run
# strictly serially, so budget ~6-7 h total.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

FLOW=flows/centroid_g2v3d_sigmaxblock_multiscale.eqx

for pop in gauss2_v3d_psfe00 \
           gauss2_v3d_psfe1p02 gauss2_v3d_psfe2p02 gauss2_v3d_psfe1m02 \
           gauss2_v3d_psfe1p05 gauss2_v3d_psfe2p05 gauss2_v3d_psfe1m05 \
           gauss2_v3d_psfe1p10 gauss2_v3d_psfe2p10 gauss2_v3d_psfe1m10; do
    tag=${pop#gauss2_v3d_}
    python -u bias.py --pop "$pop" --flow $FLOW \
        --samples 8192 --alpha 0.5 --chunk 2048 --batch-budget 16384 \
        --n-targets 20000 --gauge auto \
        --floor-eps 1e-3 --window-guard 100 \
        --window-size 2.2 3.2 --window-flux 2500 50000 \
        --window-terms score --window-draws 1048576 \
        --save-pqr "pqr/g2v3d_${tag}.npz" \
        2>&1 | tee "logs/bias_g2v3d_${tag}.log"
done
echo "bias g2v3d psfe done"
