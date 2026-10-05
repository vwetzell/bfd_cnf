#!/bin/bash
# 2026-09-27 -- the S/N-8 bulgedisc_g2n chain trained to CONVERGENCE (train_loop.fit via
# --converge: constant LR, reduce-on-plateau, stop on a paired held-out test) at large
# batch, replacing dev/converge_sn8.sh's fixed-step 1x/2x study (GPU at <50% power at
# batch 1024).  Recipe otherwise as dev/sn8_diag2.sh's anisotropic layer.
# Order: s0's varobs bias (its chain already trained), then seeds 1-2 trained + gated,
# the layer-level d2(shift) check on all three, their biases, and the paired m1 table.
# A seed whose centroid flow exists is not retrained.  All GPU, strictly serial.
# Usage: bash dev/converge_chain.sh > logs/cv2/converge_chain.log 2>&1
set -e -o pipefail
cd "$(dirname "$0")/.."
D=../bfd_cnf_imsims/data; L=logs/cv2; F=flows/cv2; mkdir -p $L $F pqr
M=$D/moments_bulgedisc_g2_bdg2n.fits
C=$D/copies_bulgedisc_g2_bdg2n_sn8.fits
SC=$(python -c "print(3.674 / 0.93)")
NOCB=--xla_gpu_enable_command_buffer=
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005"

train() {   # seed
    local s=$1 r=s$1
    [ -f $F/centroid_$r.eqx ] && { echo "$r exists, not retrained"; return; }
    python -u bulk.py train --data $M --flow $F/bulk_base_$r.eqx --converge \
        --batch 16384 --lr 3e-3 --seed $s > $L/bulk_base_$r.log 2>&1
    python -u bulk.py train --data $M --flow $F/bulk_$r.eqx --chart-from $F/bulk_base_$r.eqx \
        --jac-weight 5.0 --converge --batch 16384 --lr 3e-3 --seed $s > $L/bulk_$r.log 2>&1
    python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $F/bulk_$r.eqx \
        --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --seed $s > $L/shear_$r.log 2>&1
    XLA_FLAGS=$NOCB python -u centroid.py train-sigmax --multi-scale --fit-K --anisotropic \
        --n-aniso 200 --aniso-max 0.10 --copies $C --init $F/shear_$r.eqx \
        --flow $F/centroid_$r.eqx --converge --batch 8192 --seed $s > $L/centroid_$r.log 2>&1
    echo "$r trained $(date)"
    NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n python -u dev/closed_loop.py selection \
        --flow $F/centroid_$r.eqx --h 0.005 0.02 > $L/gate_$r.log 2>&1 || echo "gate $r FAILED"
    for mode in main structure; do
        FLOW=$F/centroid_$r.eqx COPIES=$C NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n \
            python -u dev/centroid_e_response.py $mode > $L/eresp_${mode}_$r.log 2>&1 || echo "eresp $r FAILED"
    done
}

bias_run() {   # seed
    local r=s$1
    [ -f pqr/cv2_$r.npz ] && { echo "bias $r exists"; return; }
    python -u bias.py --pop bulgedisc_g2n_176k_varobs_sn8 --flow $F/centroid_$r.eqx $BF \
        --save-pqr pqr/cv2_$r.npz > $L/bias_cv2_$r.log 2>&1
    echo "bias $r done $(date)"
}

bias_run 0
train 1
train 2
python -u dev/converge_checks.py d2layer flows/centroid_bdg2n_sn8_Ke_an.eqx \
    $F/centroid_s0.eqx $F/centroid_s1.eqx $F/centroid_s2.eqx > $L/d2layer.log 2>&1 || echo "d2layer FAILED"
bias_run 1
bias_run 2
LOGS=$L python -u dev/converge_checks.py pair cv2_s0 cv2_s1 cv2_s2 > $L/pair.log 2>&1 || echo "pair FAILED"
echo "ALL DONE $(date)"
