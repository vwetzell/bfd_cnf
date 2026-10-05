#!/bin/bash
# 2026-10-01 -- true galaxies + model noise (dev/truegal_loop.py) through the cv2 s0
# flow, the flow whose closed loop (closedJ, +0.0002) and sn8r sims run already exist.
# Usage: bash dev/truegal_chain.sh > logs/truegal_chain.log 2>&1   (after the seed-2 copies build)
set -eo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false BFD_LEGACY_CHART=1   # cv2 = pre-E_MAX chart
D=../bfd_cnf_imsims/data; F=flows/cv2/centroid_s0.eqx
[ -f $D/truegal_sn8r_g0.fits ] || python -u dev/truegal_loop.py --copies $D/copies_bulgedisc_g2_bdg2n_sn8_seed2.fits > logs/truegal_cat.log 2>&1
for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done
python -u bias.py --pop bdg2n_sn8r_truegal --flow $F --samples 8192 --alpha 0.5 --chunk 4096 \
    --batch-budget 32768 --no-zero-arm --support --floor-eps 0 --window-size 2.2 3.2 \
    --window-flux 3000 20000 --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005 \
    --window-jacobian full --save-pqr pqr/truegal_cv2s0.npz > logs/truegal_bias.log 2>&1
python -u dev/truegal_pair.py pqr/cv2_s0_sn8r_J.npz bulgedisc_g2n_176k_sn8r pqr/truegal_cv2s0.npz bdg2n_sn8r_truegal > logs/truegal_pair.log 2>&1
grep -E "windowed|R_s =|P_s" logs/truegal_bias.log; cat logs/truegal_pair.log
echo "ALL DONE $(date)"
