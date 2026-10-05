#!/bin/bash
# 2026-09-27 -- diagnostics for the broken S/N-8 selection terms (dev/bdg2n.sh DEPTH=sn8):
# R_s negative and anisotropic, gate R11 flipping sign between h = 0.02 and 0.05, centroid
# layer ellipticity response ratio 1.047.  Serial; the CPU render runs before any GPU job.
#  (2) render: sn8 targets under a ROUND PSF, fixed noise 3.674 (no --obs-sd), seed 2
#  (1) h-scan: gate (round C_M) at h = 0.005 0.01 0.02 0.05; bias.py selection-only
#      (--n-targets 64, see dev/selseed_varobs.sh) at --window-fd 0.005 0.01 0.05 on the
#      sn8 varobs catalog and, as a control, the shallow one
#  (4) centroid: dev/centroid_e_response.py (by |e|, and 'structure' by flux/size) for the
#      sn8 and shallow layers against their own copies
#  (2) bias: the round sn8 catalog, full run
# Usage: bash dev/sn8_diag.sh > logs/sn8_diag.log 2>&1   -> logs/sn8diag/
set -e -o pipefail
cd "$(dirname "$0")/.."
S=../bfd_cnf_imsims; D=$S/data; L=logs/sn8diag; mkdir -p $L
F=flows/centroid_bdg2n_sn8_Ke.eqx; F0=flows/centroid_bdg2n_Ke.eqx
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02"
SC=$(python -c "print(3.674 / 0.93)")

for arm in "g1p02 0.02" "g1m02 -0.02" "g0 0.0"; do
    set -- $arm
    ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.sim \
        --n 520000 --seed 2 --pop bulgedisc_g2 --noise-sigma 3.674 --add-noise \
        --g1 $2 --nproc 24 --out data/targets_bdg2_${1}_176k_sn8r.fits ) 2>&1 | grep -v Warning
done
echo "render done $(date)"

NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n python -u dev/closed_loop.py selection --flow $F \
    --h 0.005 0.01 0.02 0.05 > $L/gate_hscan_sn8.log 2>&1
for h in 0.005 0.01 0.05; do
    python -u bias.py --pop bulgedisc_g2n_176k_varobs_sn8 --flow $F $BF --n-targets 64 \
        --window-fd $h > $L/sel_sn8_h$h.log 2>&1
    python -u bias.py --pop bulgedisc_g2n_176k_varobs --flow $F0 $BF --n-targets 64 \
        --window-fd $h > $L/sel_shallow_h$h.log 2>&1
done
echo "h-scan done $(date)"

for c in "sn8 $F copies_bulgedisc_g2_bdg2n_sn8.fits $SC" "shallow $F0 copies_bulgedisc_g2_bdg2n.fits 1"; do
    set -- $c
    for mode in main structure; do
        FLOW=$2 COPIES=$D/$3 NOISE_SCALE=$4 TRAIN_POP=bulgedisc_g2n \
            python -u dev/centroid_e_response.py $mode > $L/centroid_${mode}_$1.log 2>&1 || echo "centroid $1 $mode FAILED"
    done
done
echo "centroid done $(date)"

python -u bias.py --pop bulgedisc_g2n_176k_sn8r --flow $F $BF \
    --save-pqr pqr/bdg2n_Ke_176k_sn8r.npz > $L/bias_sn8r.log 2>&1
echo "ALL DONE $(date)"
