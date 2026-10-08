#!/bin/bash
# 2026-10-06 -- DES-matched s2 run re-solved at Mf 2200-20000 (user: 20k-90k is template-sparse,
# focus on Mf <= 20000).  Reuses pqr/desm_s2_des.npz's per-target Q, R (--load-pqr); only the
# selection terms are recomputed.  Flags = dev/desmatch.sh's BF with the window changed.
# Window via env: SIZE="2.2 3.5" FLUX="2200 20000" TAG=sel20k (defaults) -- edge-localisation re-solves;
# PQR / FLOW env pick another saved run (its own flow).
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
SIZE=${SIZE:-2.2 3.5}; FLUX=${FLUX:-2200 20000}; TAG=${TAG:-sel20k}; LOG=logs/desm/bias_desm_s2_$TAG.log
PQR=${PQR:-pqr/desm_s2_des.npz}; FLOW=${FLOW:-flows/desm/centroid_s2.eqx}
t0=$SECONDS
python -u bias.py --pop bulgedisc_g2n_176k_des --flow $FLOW \
  --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm \
  --support --floor-eps 0 --window-size $SIZE --window-flux $FLUX \
  --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005 --window-jacobian full \
  --window-taper 0.05 150 --load-pqr $PQR > $LOG 2>&1
echo "WALL $((SECONDS - t0)) s" >> $LOG
grep -E "loaded|windowed|P_s =|Q_s =|R_s =|top draw|WALL|Error" $LOG
