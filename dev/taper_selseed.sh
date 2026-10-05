#!/bin/bash
# 2026-10-03 -- tapered-window selection terms at extra seeds 1-3 (seed 0 = dev/taper_sel.sh),
# both catalogs at the same seed so the varobs - sn8r gap keeps common draws.  --seed moves
# only the selection prior draws here (--n-targets 2000, no pre-filter sample).  GPU serial.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm --support
  --floor-eps 0 --g 0.02 --window-fd 0.005 --n-targets 2000 --window-jacobian full
  --window-size 2.2 3.2 --window-flux 3000 20000 --window-taper 0.05 150"
mkdir -p logs/cv6/taper_selseed
for s in 1 2 3; do for t in varobs_sn8 sn8r; do
    log=logs/cv6/taper_selseed/${t}_seed$s.log
    grep -q "R_s =" $log 2>/dev/null || python -u bias.py --pop bulgedisc_g2n_176k_$t \
        --flow flows/cv6/centroid_s2.eqx $BF --seed $s > $log 2>&1 || { echo "$t seed $s FAILED"; exit 1; }
    echo "$t seed $s done $(date +%T)"; grep -E "P_s|R_s =" $log
done; done
echo "RUN DONE $(date)"
