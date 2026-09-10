#!/bin/bash
# Does the DEEP arm's windowed m1 depend on the kernel draw realisation?
#
# GUIDING_PRINCIPLES 3.1 says per-target MC draw noise is negligible at
# S = 8192: "a same-data seed change moves windowed uncorrected m1 by 1.0e-08".
# That was measured on the SHALLOW arm.  This runs the same null on
# `gauss2_v3d`, where ESS is starved and 22% of targets carry no weight at all
# -- exactly the regime where the claim is least likely to hold.
#
# Same targets, same window, same everything; only `--draw-seed` moves.  Run A
# is the default seed (`noise_seed + 1000`), i.e. `prefilter_check.sh`'s
# `full.log`, so only B is run here.
#
# If B agrees with A at ~1e-08, per-target draw noise is negligible at depth
# too, and the pre-filter's disagreement is a bug in the pre-filter.
# If B moves by ~0.02 like the pre-filter did, the pre-filter is innocent and
# every deep-arm number carries an unquoted draw-noise term -- including
# gauss2_v3d's headline corrected m1 = -0.01738 +/- 0.01251.
set -e -o pipefail
cd "$(dirname "$0")/.."
S=${OUT:-/tmp/prefilter_check}
mkdir -p "$S"

COMMON=(--pop gauss2_v3d --flow flows/centroid_g2v3d.eqx
        --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536
        --n-targets "${N:-6000}" --gauge auto
        --window-size 2.2 3.2 --window-flux 2500 50000
        --window-terms score --window-draws 1048576
        --floor-eps 1e-3 --window-guard 100)

echo "=== B: --draw-seed ${SEED:-1007} ==="
t0=$SECONDS
python -u bias.py "${COMMON[@]}" --draw-seed "${SEED:-1007}" \
    > "$S/seedB.log" 2>&1
echo "  $((SECONDS - t0))s"

for f in full seedB; do
    [ -f "$S/$f.log" ] || continue
    echo "--- $f ---"
    grep -E "targets kept|zero-weight|windowed," "$S/$f.log" || true
done
