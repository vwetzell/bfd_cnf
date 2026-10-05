#!/bin/bash
# 2026-10-03 -- code-level verification of the cv6 pipeline (peel + adapter + pool + e=0 fix).
#  (2) closed loop: targets drawn from cv6 ITSELF (consistent-J noise), production bias.py
#      with the pool -> any m1 is numerics, not the flow (cv2's closed loop: +0.0002).
#  (3) real sn8r with the pool OFF, paired against the pool-on run (same kernel seeds).
# Usage: bash dev/verify_cv6.sh > logs/cv6/verify.log 2>&1      GPU strictly serial.
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
L=logs/cv6; F=flows/cv6/centroid_s2.eqx
BF="--flow $F --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005 --window-jacobian full"
for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done

REAL_POP=bulgedisc_g2n_176k_sn8r TRAIN_POP=bulgedisc_g2n python -u dev/closed_loop.py catalogs \
    --flow $F --n 520000 --pop bdg2n_sn8r_closedJ_cv6 --jacobian > $L/closed_cat.log 2>&1 \
    || { echo "closed catalogs FAILED"; tail -5 $L/closed_cat.log; exit 1; }
grep -E "wrote|ratio" $L/closed_cat.log
run() {  # pop out-tag extra-flags
    t0=$SECONDS; python -u bias.py --pop $1 $BF $3 --save-pqr pqr/$2.npz > $L/bias_$2.log 2>&1 || echo "bias $2 FAILED"
    echo "WALL $((SECONDS - t0)) s" >> $L/bias_$2.log
    echo "== $2"; grep -E "windowed|R_s =|had no draw|kept the plain|pool|WALL" $L/bias_$2.log
}
run bdg2n_sn8r_closedJ_cv6 closedJ_cv6_sn8r "--pool-reuse 8"
run bulgedisc_g2n_176k_sn8r cv6_s2_sn8r_nopool "--pool-reuse 0"
echo "== pool off - pool on (sn8r, paired)"
JAX_PLATFORMS=cpu python -u dev/pair_sn8r_varobs.py pqr/cv6_s2_sn8r.npz $L/bias_cv6_s2_sn8r.log sn8r \
    pqr/cv6_s2_sn8r_nopool.npz $L/bias_cv6_s2_sn8r_nopool.log sn8r 2>&1 | grep -v Warning | tail -1
echo "ALL DONE $(date)"
