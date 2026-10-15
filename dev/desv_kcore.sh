#!/bin/bash
# 2026-10-12: core-window (cv6win: Mr/Mf 2.2-3.2, Mf 3000-20000) selection re-solves of the existing k-split
# 300k-prefix runs (pqr/desv_tnc15c_k{0-3}{,g}_sub.npz, dev/desv_split.sh).  Four independent 100k template sets:
# their scatter is the template-realization error of the core m1, their mean the core m1 over template draws.
# One selection seed each (seed scatter ~1e-4).
# Usage: bash dev/desv_kcore.sh > logs/desv/chain_kcore.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm
BF="--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates"
die() { echo "$1 FAILED $(date)"; exit 1; }
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
for r in k0 k1 k2 k3 k0g k1g k2g k3g; do
    tag=${r}sub_cv6win_sd0
    [ -f $L/bias_$tag.log ] && grep -q "windowed, corrected" $L/bias_$tag.log && continue
    python -u bias.py $BF --flow $F/centroid_tnc15c_$r.eqx --train-data $D/moments_bulgedisc_g2_bdg2d_${r%g}.fits \
        --window-size 2.2 3.2 --window-flux 3000 20000 --seed 0 --n-targets 300000 \
        --load-pqr pqr/desv_tnc15c_${r}_sub.npz > $L/bias_$tag.log 2>&1 || die "re-solve $tag"
    grep -E "P_s =|windowed, corrected" $L/bias_$tag.log | sed "s/^/  $tag /"
done
echo "ALL DONE $(date)"
