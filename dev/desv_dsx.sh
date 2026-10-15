#!/bin/bash
# 2026-10-10 -- Sigma_X deconvolution test (user, option 1): desv s3 exactly as its s3sub run in dev/desv.sh, plus
#   bias.py --deconvolve-sigmax (targets' and selection draws' Sigma_X minus the bdg2d templates' mean, 7.2%).
#   Noisy templates are recentred on their own noisy centroid, so the prior already carries the geometric
#   centroid jitter at Sigma_X,t; the copies add the targets' Sigma_X on top (scratchpad copy_deficit.py /
#   jweight.py, 2026-10-10).  Paired with pqr/desv_tnc15c_s3_sub.npz: same flow, targets, draws.
# KNOWN LIMIT: 1.27% of targets fall below the layer's trained Sigma_X range (0.41% without the flag; pre-flight
#   gate is 1%) -- the layer extrapolates there.  Option 2 (copies at --sigma-range 1.6 + retrained layer) if needed.
# Usage: bash dev/desv_dsx.sh > logs/desv/chain_dsx.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
L=logs/desv; F=flows/desm; NT=300000; r=s3dsx; cf=$F/centroid_tnc15c_s3.eqx; P=pqr/desv_tnc15c_${r}_sub.npz
BF="--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --deconvolve-sigmax"
die() { echo "$1 FAILED $(date)"; exit 1; }
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
[ -f $P ] || python -u bias.py $BF --flow $cf --window-size 2.2 3.5 --window-flux 2200 90000 \
    --n-targets $NT --save-pqr $P > $L/bias_${r}_sub.log 2>&1 || die "bias $r subset"
grep -E "deconvolve|ESS =" $L/bias_${r}_sub.log
for spec in "full90k 2.2 3.5 2200 90000" "sel20k 2.2 3.5 2200 20000" "cv6win 2.2 3.2 3000 20000"; do
    set -- $spec
    for seed in 0 1 2 3; do
        tag=${r}sub_$1_sd$seed
        [ -f $L/bias_$tag.log ] && grep -q "windowed, corrected" $L/bias_$tag.log && continue
        python -u bias.py $BF --flow $cf --window-size $2 $3 --window-flux $4 $5 --seed $seed --n-targets $NT \
            --load-pqr $P > $L/bias_$tag.log 2>&1 || die "re-solve $tag"
        grep -E "P_s =|windowed, corrected" $L/bias_$tag.log | sed "s/^/  $tag /"
    done
done
echo "ALL DONE $(date)"
