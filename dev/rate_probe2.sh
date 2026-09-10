#!/bin/bash
# Two more HOST-path points (10k, 15k) to confirm the 5k/20k slope is linear.
# Same flags as dev/rate_probe.sh -- only --n-targets moves, so every fixed
# cost stays constant and a 4-point fit should have negligible curvature.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# ONE JAX job at a time: wait out the device probe's shell, then any stragglers.
while [ -d /proc/882652 ]; do sleep 20; done
while pgrep -f 'bias.py --pop gauss2_v3d' > /dev/null; do sleep 20; done

one() {
    local n=$1 t0 t1
    t0=$(date +%s)
    python -u bias.py \
        --pop gauss2_v3d --flow flows/centroid_g2v3d.eqx \
        --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536 \
        --n-targets "$n" --gauge auto --no-zero-arm \
        --prefilter-pad 0.2 --prefilter-sample 0.15 \
        --window-size 2.2 3.2 --window-flux 2500 50000 \
        --window-terms score --window-draws 1048576 \
        --floor-eps 1e-3 --window-guard 100 \
        > "logs/rate_$n.log" 2>&1
    t1=$(date +%s)
    echo "$n $((t1 - t0))"          # no bc on this box
}

one 10000
one 15000
echo "RATE PROBE 2 DONE"
