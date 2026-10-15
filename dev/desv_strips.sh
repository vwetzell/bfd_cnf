#!/bin/bash
# 2026-10-12: strip-only selection re-solves of the plain 1M s2 run (pqr/desv_tnc15c_s2.npz): the large-size
# strip (Mr/Mf 3.2-3.5, Mf 3000-20000) and the faint strip (Mr/Mf 2.2-3.2, Mf 2200-3000) as windows of their own.
# The --template-noise-g shift is added offline on the 300k slice (selection terms are unchanged by it).
# Usage: bash dev/desv_strips.sh > logs/desv/chain_strips.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
L=logs/desv; F=flows/desm
BF="--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates"
die() { echo "$1 FAILED $(date)"; exit 1; }
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
for spec in "large3000:3.2 3.5 3000 20000" "faint32:2.2 3.2 2200 3000"; do
    name=${spec%%:*}; set -- ${spec#*:}
    for seed in 0 1 2 3; do
        tag=s2_${name}_sd$seed
        [ -f $L/bias_$tag.log ] && grep -q "windowed, corrected" $L/bias_$tag.log && continue
        python -u bias.py $BF --flow $F/centroid_tnc15c_s2.eqx --window-size $1 $2 --window-flux $3 $4 --seed $seed \
            --load-pqr pqr/desv_tnc15c_s2.npz > $L/bias_$tag.log 2>&1 || die "re-solve $tag"
        grep -E "P_s =|R_s =|windowed, corrected" $L/bias_$tag.log | sed "s/^/  $tag /"
    done
done
echo "ALL DONE $(date)"
