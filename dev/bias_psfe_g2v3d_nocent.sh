#!/bin/bash
# Thread 1's --no-centroid bisection (PSFE_PROVENANCE.md #8d, proposed but
# never run): does the PSFE c1/c2 leak get WORSE without the SigmaXBlockLayer
# (partial suppression by the trained centroid layer) or stay the SAME (no
# suppression at all)? Mirrors dev/bias_psfe_g2v3d.sh exactly (same window,
# same target-integration/selection-term flags, same "Thread 1 -- RESUME
# CONFIG" validated setup) except:
#   --flow flows/shear_g2v3d_full2_jac.eqx --no-centroid   no SigmaXBlockLayer
#     at all, instead of flows/centroid_g2v3d_full2_jac.eqx.
#   distinct log/pqr filenames (_nocent suffix) so this run cannot collide
#     with or overwrite the existing centroid-layer cache.
#
# Minimal rerun per dev/bias_psfe_g2v3d.sh's own comment: baseline + one
# orientation (psfe1) x all 3 amplitudes = 4 configs, ~2.5-3 h serial.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

FLOW=flows/shear_g2v3d_full2_jac.eqx

for pop in gauss2_v3d_psfe00 \
           gauss2_v3d_psfe1p02 gauss2_v3d_psfe1p05 gauss2_v3d_psfe1p10; do
    tag=${pop#gauss2_v3d_}
    python -u bias.py --pop "$pop" --flow $FLOW --no-centroid \
        --samples 8192 --alpha 0.5 --chunk 2048 --batch-budget 16384 \
        --n-targets 20000 --no-zero-arm --support --floor-eps 0 \
        --window-size 2.2 3.2 --window-flux 1500 20000 \
        --window-terms score --window-draws 16777216 \
        --save-pqr "pqr/g2v3d_full2_jac_${tag}_nocent.npz" \
        2>&1 | tee "logs/bias_g2v3d_full2_jac_${tag}_nocent.log"
done
echo "bias g2v3d psfe no-centroid bisection (minimal rerun) done"
