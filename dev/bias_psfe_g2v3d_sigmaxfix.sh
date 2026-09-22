#!/bin/bash
# Re-run of dev/bias_psfe_g2v3d.sh (psfe1) plus the psfe2 (C01) sweep,
# against flows/centroid_g2v3d_full2_jac_sigmaxfix.eqx -- the SigmaXBlockLayer
# retrained after fixing the non-covariant c*E*conj(X) term in _ellipticity
# (models/bijections.py:1951-1952) to the covariant c*proj*E form.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

FLOW=flows/centroid_g2v3d_full2_jac_sigmaxfix.eqx

for pop in gauss2_v3d_psfe00 \
           gauss2_v3d_psfe1p02 gauss2_v3d_psfe1p05 gauss2_v3d_psfe1p10 \
           gauss2_v3d_psfe2p02 gauss2_v3d_psfe2p05 gauss2_v3d_psfe2p10; do
    tag=${pop#gauss2_v3d_}
    python -u bias.py --pop "$pop" --flow $FLOW \
        --samples 8192 --alpha 0.5 --chunk 2048 --batch-budget 16384 \
        --n-targets 20000 --no-zero-arm --support --floor-eps 0 \
        --window-size 2.2 3.2 --window-flux 1500 20000 \
        --window-terms score --window-draws 16777216 \
        --save-pqr "pqr/g2v3d_full2_jac_sigmaxfix_${tag}.npz" \
        2>&1 | tee "logs/bias_g2v3d_full2_jac_sigmaxfix_${tag}.log"
done
echo "bias g2v3d psfe sigmaxfix (psfe1 + psfe2) done"
