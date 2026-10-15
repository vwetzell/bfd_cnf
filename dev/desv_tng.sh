#!/bin/bash
# 2026-10-11: --template-noise-g on the 300k-slice flows s2-s6 (kernel C_M - Sigma_t(g), Sigma_t(g) = C_t + g cbar
# (G + G^T)).  The forward check (fwd_check.py sigt) closed the size-edge response deficit with it; G is the
# dD-dM template-noise covariance from dev/noise_G.py (noisy templates only).  Selection terms are unchanged by
# it (the (Mf, Mr) block of G + G^T is zero), so the re-solves reuse the old seeds and windows.
# Usage: bash dev/desv_tng.sh > logs/desv/chain_tng.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm; NT=300000
BF="--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz"
die() { echo "$1 FAILED $(date)"; exit 1; }
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
resolve() {   # tag pqr flow size_lo size_hi flux_lo flux_hi seed [extra]
    local tag=$1 p=$2 fl=$3; shift 3
    [ -f $L/bias_$tag.log ] && grep -q "windowed, corrected" $L/bias_$tag.log && return 0
    python -u bias.py $BF --flow $fl --window-size $1 $2 --window-flux $3 $4 --seed $5 ${@:6} --load-pqr $p \
        > $L/bias_$tag.log 2>&1 || die "re-solve $tag"
    grep -E "windowed, corrected" $L/bias_$tag.log | sed "s/^/  $tag /"
}
W="full90k:2.2 3.5 2200 90000 sel20k:2.2 3.5 2200 20000 cv6win:2.2 3.2 3000 20000"
for s in 3 4 5 6 2; do
    P=pqr/desv_tnc15c_s${s}tng_sub.npz; fl=$F/centroid_tnc15c_s$s.eqx; t0=$SECONDS
    [ -f $P ] || { python -u bias.py $BF --flow $fl --window-size 2.2 3.5 --window-flux 2200 90000 \
        --n-targets $NT --save-pqr $P > $L/bias_s${s}tng_sub.log 2>&1 \
        && echo "WALL $((SECONDS - t0)) s" >> $L/bias_s${s}tng_sub.log; } || die "bias s${s}tng"
    echo "s${s}tng integrated $(date)"
    echo "$W" | tr ' ' '\n' | paste -d' ' - - - - | while read -r spec; do
        name=${spec%%:*}; set -- ${spec#*:}
        for seed in 0 1 2 3; do resolve s${s}tngsub_${name}_sd$seed $P $fl $1 $2 $3 $4 $seed --n-targets $NT; done
    done
done
echo "ALL DONE $(date)"
