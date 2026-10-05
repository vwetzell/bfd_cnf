#!/bin/bash
# 2026-09-26 -- selection-term MC error for the pooled varobs m1 (dev/varobs_pool.py).
# Every batch's production bias.py ran --seed 0, so the selection-term prior draws
# are shared and their MC error doesn't average down across batches.  Reruns ONLY
# the selection terms (--n-targets 64: the selection branch reads the full zero
# catalog itself, N_ns counts the full catalog; --prefilter-sample 0 so --seed
# touches nothing else) at seeds 1-5, one seed per replica across all six batches.
# Seed 0 reproduces the production P_s/Q_s/R_s exactly.  ~6.7 min per run.
# Usage: bash dev/selseed_varobs.sh   -> logs/selseed/bias_<tag>_seed<r>.log
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/selseed
BIAS_FLAGS="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --flow flows/centroid_g2v4n_Ke.eqx"
for r in ${SEEDS:-1 2 3 4 5}; do
    for b in "gauss2_v4n_176k_varobs g2v4n_Ke_176k_varobs" \
             $(for s in 3 4 5 6 7; do echo "gauss2_v4n_176k_s${s}_varobs:g2v4n_Ke_176k_s${s}_varobs"; done); do
        b=${b/:/ }; set -- $b
        python -u bias.py --pop $1 $BIAS_FLAGS --n-targets 64 --seed $r \
            > logs/selseed/bias_${2}_seed${r}.log 2>&1
    done
    echo "seed $r done $(date)"
done
echo ALL DONE
