#!/bin/bash
# Phase D.1: the gauss2 chain at TWO depths, end to end.
#
# `gauss2_fwd` is the only population where P(m|g), Q and R are closed form
# (`truth.py`), so GUIDING_PRINCIPLES 3.2 says a bulgedisc number is not
# interpretable until the same measurement passes here.  Two tags:
#
#   g2v3   noise_sigma 0.93, median flux S/N 19.0   -- matches bulgedisc_v3
#   g2v3d  noise_sigma 1.86, median flux S/N  9.6   -- SAME galaxies, 2x deep
#
# The pair is the point.  Same population, same seed, same galaxies row for
# row; `SIG_XY` comes out at exactly 2x (222.12547 vs 111.06273), so depth is
# the only thing that differs.  That isolates what the centroid layer is worth
# against a KNOWN answer rather than only against zero.
#
# Each tag needs its OWN copy grid: the grid is only valid within a factor 1.4
# in sigma_XY and this is 2.0x, so g2v3's copies are void for g2v3d.
#
# ONE JAX GPU JOB AT A TIME -- the 16 GB card OOMs with a second JAX process on
# it, so copies (CPU, all cores) all run first, then the two trainings, then
# the two bias runs, strictly in sequence.
#
# Usage: bash dev/gauss2_chain.sh          # everything
#        bash dev/gauss2_chain.sh bias     # just the measurement
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr flows

TAGS=("g2v3 0.93" "g2v3d 1.86")
COMMON="NPRIOR=100000 NTARGET=500000 SIZE=500k POP=gauss2_fwd"

copies_all() {
    for spec in "${TAGS[@]}"; do
        set -- $spec
        echo "=== copies $1 (noise_sigma $2) ==="
        env $COMMON SIGMA=$2 TAG=$1 bash rebuild_v3.sh copies \
            2>&1 | tee logs/copies_$1.log
    done
}

train_all() {
    for spec in "${TAGS[@]}"; do
        set -- $spec
        echo "=== train $1 (noise_sigma $2) ==="
        env $COMMON SIGMA=$2 TAG=$1 bash rebuild_v3.sh train
    done
}

# Same flags as `dev/bias_amplitude.sh`, and they have to stay the same: the
# whole value of a gauss2 number is that it is comparable to the bulgedisc one.
#   --gauge auto      mandatory ([[r-is-an-is-artifact]]): gauge P's Var[score]
#                     is infinite and took m1 -1.27 -> -0.005 on bulgedisc.
#   --window-terms score, --window-draws 2^24   the pathwise path without
#                     --window-fd has no finite mean; 2^20 carries +/-4e-03.
#   The window is in Mf, not S/N, and the two tags share galaxies -- so the
#   SAME window selects the SAME galaxies at both depths, which is what makes
#   the pair a controlled comparison.
bias_all() {
    for spec in "gauss2_v3 g2v3" "gauss2_v3d g2v3d"; do
        set -- $spec
        echo "=== bias $1 ==="
        python -u bias.py --pop "$1" --flow flows/centroid_$2.eqx \
            --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536 \
            --n-targets 20000 --gauge auto \
            --window-size 2.2 3.2 --window-flux 2500 50000 \
            --window-terms score --window-draws 16777216 \
            --save-pqr pqr/$2_20k.npz 2>&1 | tee logs/bias_$2_20k.log
    done
}

case "${1:-all}" in
    copies) copies_all ;;
    train)  train_all ;;
    bias)   bias_all ;;
    all)    copies_all; train_all; bias_all ;;
    *) echo "usage: $0 {all|copies|train|bias}"; exit 2 ;;
esac
echo "GAUSS2 CHAIN DONE"
