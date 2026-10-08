#!/bin/bash
# 2026-10-06 -- inference-side template deconvolution (--deconvolve-templates) on the noisy-copies
# DES flow: kernel and selection C_M -> C_M - <C_tmpl>.  Subset = first 90k catalog rows (52,021
# targets, the prefix of pqr/desm_tnc_s2_des.npz, same seed -> paired), then 4 window re-solves.
# Usage: bash dev/desm_dt_sub.sh > logs/desm/dt_sub_chain.log 2>&1      GPU, serial.
#   full catalog: TAG=dt NT= bash dev/desm_dt_sub.sh > logs/desm/dt_chain.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
L=logs/desm; FC=flows/desm/centroid_tnc_s2.eqx; TAG=${TAG:-dt_sub}; NT=${NT---n-targets 90000}
P=pqr/desm_tnc_s2_${TAG}.npz
BF="--pop bulgedisc_g2n_176k_des --flow $FC --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768
  --no-zero-arm --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 $NT --deconvolve-templates"
t0=$SECONDS
[ -f $P ] || python -u bias.py $BF --window-size 2.2 3.5 --window-flux 2200 90000 --save-pqr $P \
    > $L/bias_desm_tnc_s2_${TAG}.log 2>&1 || { echo "bias FAILED"; exit 1; }
echo "WALL $((SECONDS - t0)) s" >> $L/bias_desm_tnc_s2_${TAG}.log
for spec in "sel20k 2.2 3.5 2200 20000" "cv6win 2.2 3.2 3000 20000" "s32_f2200 2.2 3.2 2200 20000" "s35_f3000 2.2 3.5 3000 20000"; do
    set -- $spec
    python -u bias.py $BF --window-size $2 $3 --window-flux $4 $5 --load-pqr $P > $L/bias_desm_s2_tnc_${TAG}_$1.log 2>&1 \
        || { echo "re-solve $1 FAILED"; exit 1; }
    echo "== $1"; grep -E "P_s =|R_s =" $L/bias_desm_s2_tnc_${TAG}_$1.log
done
echo "ALL DONE $(date)"
