#!/bin/bash
# 2^28 selection terms, which OOM'd before `--draw-chunk` existed.
#
# The failure was ONE allocation: `flow.base_dist.sample(key, (n,))` builds all
# n base points on the card at once -- 2^28 x 5 x float64 = 10.02 GiB on a
# 16 GB card, and the traceback pointed at the affine bijection's matmul.  The
# transform loop below it was already chunked at 16384; only the sampling was
# not.  `--draw-chunk` chunks it, filling a preallocated host array in place
# (10.7 GB at 2^28, against 46 GB free).
#
# The chunked path uses `jr.fold_in` per chunk, because `sample` at a FIXED key
# NESTS -- k calls with one key return k IDENTICAL blocks, which would have
# quietly turned 2^28 draws into 2^24 draws repeated 16 times and reported a
# 4x-too-tight answer.  That is the [[pqr-streamed-seed-nests]] trap.  So the
# stream differs from an unchunked run and the cache name carries `_c<chunk>`.
#
# STAGE 1 validates that: 2^24 chunked into 4 must land inside the spread of
# the six unchunked 2^24 seeds.  If it lands outside, the chunking is wrong and
# stage 2 is meaningless -- that is the whole point of running it first.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

run() {   # run <log2> <chunk> <seed>
    python -u dev/window_scan.py --pop gauss2_v3e \
        --flow flows/centroid_g2v3d.eqx --pqr pqr/g2v3e_s0.npz \
        --log2-draws "$1" --draw-chunk "$2" --no-scan --seed "$3" \
        2>&1 | tee "logs/rs_c${1}_s$3.log" \
        | grep -E "nominal|chunks of|top draw|Error|Traceback|RESOURCE" || true
}

case "${1:-all}" in
  validate) for s in 0 7; do echo "=== 2^24 chunked x4, seed $s ==="
                run 24 4194304 "$s"; done ;;
  full)     for s in 0 7; do echo "=== 2^28 chunked x16, seed $s ==="
                run 28 16777216 "$s"; done ;;
  all)      bash "$0" validate; bash "$0" full; echo "RS 2P28 ALL DONE" ;;
esac
