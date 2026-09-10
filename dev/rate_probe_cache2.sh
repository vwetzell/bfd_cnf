#!/bin/bash
# DEVICE path (BFD_DEVICE_PQR=1) with the JAX persistent compilation cache ON.
#
# The cache changes the INTERCEPT (compile), not the slope -- but only once it
# is warm.  A cold first run POPULATES it, so if the N points were measured
# straight away the first would carry the full compile and the slope would be
# biased low.  Hence: prime twice at a fixed N, THEN measure.  Two primes
# because the cache is documented to need them ([[jax-cache-not-bit-identical]]).
#
# Prime #2 vs the first measurement (both N=5000) is the warm-stability check:
# they must agree, or the cache is still filling and the fit is not trustworthy.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

while pgrep -f 'bias.py --pop gauss2_v3d' > /dev/null; do sleep 20; done

one() {   # $1 = n-targets, $2 = log tag
    local n=$1 tag=$2 t0 t1
    t0=$(date +%s)
    BFD_DEVICE_PQR=1 python -u bias.py \
        --pop gauss2_v3d --flow flows/centroid_g2v3d.eqx \
        --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536 \
        --n-targets "$n" --gauge auto --no-zero-arm \
        --prefilter-pad 0.2 --prefilter-sample 0.15 \
        --window-size 2.2 3.2 --window-flux 2500 50000 \
        --window-terms score --window-draws 1048576 \
        --floor-eps 1e-3 --window-guard 100 \
        > "logs/$tag.log" 2>&1
    t1=$(date +%s)
    echo "$tag $n $((t1 - t0))"
}



for n in 5000 10000 15000 20000; do one $n "ratecache2_$n"; done
echo "RATE PROBE CACHE PASS2 DONE"
