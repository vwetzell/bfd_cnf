#!/bin/bash
# Per-target integration rate, EXCLUDING compilation.
#
# Two runs at different --n-targets, identical in every other respect.  All
# fixed costs (compile, flow load, selection terms, window draws) are the same
# constant in both, so the SLOPE (dt/dN) is the pure integration rate and the
# INTERCEPT is the fixed overhead.  No barriers, no BFD_TIMING -- those insert
# `block_until_ready` and destroy the device/host overlap being measured.
#
# Cache deliberately OFF for both, so the compile term is equal and cancels.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
OUT=/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/95a440e8-6cd7-49ee-ab19-6f427455062f/scratchpad
mkdir -p "$OUT"

one() {   # $1 = n-targets
    local n=$1 t0 t1
    t0=$(date +%s.%N)
    python -u bias.py \
        --pop gauss2_v3d --flow flows/centroid_g2v3d.eqx \
        --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536 \
        --n-targets "$n" --gauge auto --no-zero-arm \
        --prefilter-pad 0.2 --prefilter-sample 0.15 \
        --window-size 2.2 3.2 --window-flux 2500 50000 \
        --window-terms score --window-draws 1048576 \
        --floor-eps 1e-3 --window-guard 100 \
        > "logs/rate_$n.log" 2>&1
    t1=$(date +%s.%N)
    echo "$n $(echo "$t1 - $t0" | bc)" | tee -a "$OUT/rate.txt"
}

: > "$OUT/rate.txt"
one 5000
one 20000

python - "$OUT/rate.txt" <<'PY'
import sys, re
n, t = zip(*[map(float, l.split()) for l in open(sys.argv[1])])
# integrated count is what actually gets the 2-arm treatment
integ = [int(re.search(r"integrating (\d+)/", open(f"logs/rate_{int(k)}.log").read()).group(1))
         for k in n]
slope = (t[1] - t[0]) / (integ[1] - integ[0])
print(f"\n  N={int(n[0])}  integrated={integ[0]}  wall={t[0]:.1f}s")
print(f"  N={int(n[1])}  integrated={integ[1]}  wall={t[1]:.1f}s")
print(f"\n  per integrated target (2 arms): {slope*1e3:.2f} ms")
print(f"  per target per arm:             {slope*1e3/2:.2f} ms")
print(f"  fixed overhead (compile+terms): {t[0] - slope*integ[0]:.1f} s")
PY
