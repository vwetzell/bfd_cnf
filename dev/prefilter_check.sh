#!/bin/bash
# Does `--prefilter-pad` change the windowed answer?  It must not.
#
# Out-of-window targets reach eq. (45)-(46) only as the count `n_ns` (`ghat`
# masks their Q and R away), and `n_ns` is taken from the FULL catalog before
# the pre-filter -- so skipping their integration is supposed to be free.
#
# Run on gauss2_v3d, the DEEP arm, deliberately: that is where the
# out-of-window population is pathological (4338/19945 zero-weight, all of them
# outside the window), so if the pre-filter can break anything it breaks here.
#
# Not bit-identical by construction: `pqr_streamed` seeds per target BATCH, so
# subsetting the targets re-partitions the batches and every target draws a
# different kernel realisation.  GUIDING_PRINCIPLES 3.1 measured that as 1.0e-08
# on windowed m1, so agreement should be far tighter than tau -- a difference
# above ~1e-05 is a bug, not noise.
#
# 2^20 window draws, not 2^24: both runs share the seed so the selection-term
# MC cancels in the comparison, and it is 20x cheaper.
set -e -o pipefail
cd "$(dirname "$0")/.."
S=${OUT:-/tmp/prefilter_check}
mkdir -p "$S"

COMMON=(--pop gauss2_v3 --flow flows/centroid_g2v3.eqx
        --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536
        --n-targets "${N:-6000}" --gauge auto
        --window-size 2.2 3.2 --window-flux 2500 50000
        --window-terms score --window-draws 1048576
        --floor-eps 1e-3 --window-guard 100)

echo "=== FULL ==="
t0=$SECONDS
python -u bias.py "${COMMON[@]}" > "$S/full.log" 2>&1
t_full=$((SECONDS - t0))

echo "=== PREFILTER pad 0.2 ==="
t0=$SECONDS
python -u bias.py "${COMMON[@]}" --prefilter-pad 0.2 > "$S/pre.log" 2>&1
t_pre=$((SECONDS - t0))

for f in full pre; do
    echo "--- $f ---"
    grep -E "pre-filter|excluded|zero-weight|windowed,|targets INTEGRATED|targets, g1" \
        "$S/$f.log" || true
done
echo "--- wall: full ${t_full}s, prefiltered ${t_pre}s ---"
