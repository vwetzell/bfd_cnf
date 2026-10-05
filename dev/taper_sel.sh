#!/bin/bash
# 2026-10-03 -- selection terms for the TAPERED base window (--window-taper 0.05 150),
# both catalogs, same settings/seed as dev/window_shift.py; combined offline with the
# full runs' per-target Q, R by dev/taper_check.py.   GPU strictly serial.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm --support
  --floor-eps 0 --g 0.02 --window-fd 0.005 --n-targets 2000 --window-jacobian full
  --window-size 2.2 3.2 --window-flux 3000 20000 --window-taper 0.05 150"
for t in varobs_sn8 sn8r; do
    log=logs/cv6/taper_sel_$t.log
    grep -q "R_s =" $log 2>/dev/null || python -u bias.py --pop bulgedisc_g2n_176k_$t \
        --flow flows/cv6/centroid_s2.eqx $BF > $log 2>&1 || { echo "$t FAILED"; exit 1; }
    echo "$t done"; grep -E "P_s|R_s =" $log
done
echo "RUN DONE $(date)"
