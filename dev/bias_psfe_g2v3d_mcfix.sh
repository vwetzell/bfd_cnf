#!/bin/bash
# Reduced 4-config (psfe00 + 3 orientations at amplitude 0.05) psfe bias
# grid on the net_dipquad galaxy-ellipticity + Mc/Mr checkpoint
# (flows/centroid_g2v3d_sigmaxblock_mc_lr3e4_s0.eqx), mirroring the
# densegrid follow-up's own reduced-grid pattern
# (dev/bias_psfe_g2v3d.sh / PSFE_PROVENANCE.md #8d for the full recipe and
# the baseline slope this compares against).
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

FLOW=flows/centroid_g2v3d_sigmaxblock_mc_lr3e4_s0.eqx
SUFFIX=_mcfix

for pop in gauss2_v3d_psfe00 \
           gauss2_v3d_psfe1p05 gauss2_v3d_psfe2p05 gauss2_v3d_psfe1m05; do
    tag=${pop#gauss2_v3d_}
    python -u bias.py --pop "$pop" --flow $FLOW \
        --samples 8192 --alpha 0.5 --chunk 2048 --batch-budget 16384 \
        --n-targets 20000 --gauge auto \
        --floor-eps 1e-3 --window-guard 100 \
        --window-size 2.2 3.2 --window-flux 2500 50000 \
        --window-terms score --window-draws 1048576 \
        --save-pqr "pqr/g2v3d_${tag}${SUFFIX}.npz" \
        2>&1 | tee "logs/bias_g2v3d_${tag}${SUFFIX}.log"
done
echo "bias g2v3d psfe mcfix done"
