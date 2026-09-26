#!/bin/bash
# 2026-09-25 -- THIS FILE IS THE PROVENANCE RECORD for
#   gauss2_v4n_176k   : 8x more real g2v4 targets, independent of the 22k
#   gauss2_v4n_closed8: 8 more closed-loop populations (flow = truth)
# Why: real-vs-closed corrected m1 is +0.014 +/- 0.008, limited by the 22k
# real targets; the closed chunk scatter that calibrates the comparison came
# from only 4 chunks ([[real-vs-closed-gap-is-size-edge]]).
#
# The 22k targets' render command was never recorded; this reconstructs it
# from their FITS header (SEED 1, NOISESIG 0.93, IMGNOISE, gauss2_fwd, g1 =
# +/-0.02 / 0, region 3000 20000 2.2 3.2 with the default pads), changing
# only --seed (2) and --n.  bias.py flags are exactly the 22k / 4x runs'.
#
# Serial on purpose: CPU renders first, then GPU jobs one at a time
# ([[no-jax-cpu-alongside-gpu]]).
#
# Usage: bash dev/more_targets_and_chunks.sh [render|closed|bias_real|bias_closed|all]
set -e -o pipefail
cd "$(dirname "$0")/.."

S=../bfd_cnf_imsims
N=176000
SEED=2
NCLOSED=$((8 * 65372))
BIAS_FLAGS="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --flow flows/centroid_g2v4n_Ke.eqx"

render() {
    for arm in "g1p02 0.02" "g1m02 -0.02" "g0 0.0"; do
        set -- $arm
        ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.sim \
            --n $N --seed $SEED --pop gauss2_fwd --noise-sigma 0.93 --add-noise \
            --g1 $2 --region 3000 20000 2.2 3.2 --nproc 24 \
            --out data/targets_g2v4_${1}_176k.fits ) 2>&1 | grep -v Warning
    done
}

closed() {
    python -u dev/closed_loop.py catalogs --flow flows/centroid_g2v4n_Ke.eqx \
        --n $NCLOSED --seed 12 --pop gauss2_v4n_closed8
}

bias_real() {
    python -u bias.py --pop gauss2_v4n_176k $BIAS_FLAGS --save-pqr pqr/g2v4n_Ke_176k.npz
}

bias_closed() {
    python -u bias.py --pop gauss2_v4n_closed8 $BIAS_FLAGS --save-pqr pqr/closed8_g2v4n_Ke.npz
}

case "${1:-all}" in
    render) render > logs/render_g2v4_176k.log ;;
    closed) closed > logs/closed_loop_catalogs_closed8.log ;;
    bias_real) bias_real > logs/bias_g2v4n_Ke_176k.log 2>&1 ;;
    bias_closed) bias_closed > logs/bias_closed8_g2v4n_Ke.log 2>&1 ;;
    all)
        render > logs/render_g2v4_176k.log
        closed > logs/closed_loop_catalogs_closed8.log 2>&1
        bias_real > logs/bias_g2v4n_Ke_176k.log 2>&1
        bias_closed > logs/bias_closed8_g2v4n_Ke.log 2>&1
        echo ALL DONE ;;
    *) echo "usage: $0 {render|closed|bias_real|bias_closed|all}"; exit 2 ;;
esac
