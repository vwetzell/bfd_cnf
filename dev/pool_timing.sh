#!/bin/bash
# 2026-10-02 pool vs no pool on 20k sn8r rows (cv5 s2), serial on the GPU.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005 --window-jacobian full"
for k in 0 8; do
  t0=$SECONDS; python -u bias.py --pop bulgedisc_g2n_176k_sn8r --flow flows/cv5/centroid_s2.eqx \
    $BF --n-targets 20000 --pool-reuse $k --save-pqr logs/pool/pqr_k$k.npz > logs/pool/bias_k$k.log 2>&1
  echo "WALL $((SECONDS - t0)) s" >> logs/pool/bias_k$k.log
  grep -E "windowed|pool|WALL|targets INTEGRATED" logs/pool/bias_k$k.log
done
echo ALL DONE
