#!/bin/bash
# The gauss2 measurement with the `R_s` unphysical-draw defect guarded.
#
# `[[rs-tail-is-unphysical-draws]]`: `in_support` is a BOX and the reachable
# set is a CURVED subset of it, so the flow puts mass in the gap and its
# extrapolated Q, R there are garbage -- 91% of the draws carrying `R_s11`
# invert to no galaxy at all.  `--window-guard` is `sane_targets` applied to
# the prior sample, at the same factor 1000.
#
# It is a TRUNCATION, so 1000 and 100 are both run on the shallow arm: the
# spread between them IS the systematic, and one value on its own would be a
# number with no error bar.  The deep arm additionally carries --floor-eps,
# which is the separate fix for its 22% zero-weight targets -- those are two
# different defects and this run does not separate them, so read the deep
# number only against the deep no-guard/no-floor baseline.
#
# ONE JAX GPU JOB AT A TIME.  ~40 min each.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

COMMON="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536
        --n-targets 20000 --gauge auto --window-size 2.2 3.2
        --window-flux 2500 50000 --window-terms score"

# Smoke test FIRST: 2^16 draws and 200 targets exercises the whole guard path
# in ~1 min, so a typo cannot burn two hours of card time.  set -e stops here.
#
# --batch-budget IS NOT OPTIONAL, even though the sample count is tiny.
# `batch = (batch_budget or 131072) // chunk`, so leaving it out at chunk 512
# gives 256 targets at once against the real runs' 16, and the
# forward-over-reverse Hessian then asks for 13 GiB and the card OOMs.  The
# budget, not the sample count, is what bounds the memory.
echo "=== smoke test: guard path ==="
python -u bias.py --pop gauss2_v3 --flow flows/centroid_g2v3.eqx \
    --samples 512 --alpha 0.5 --chunk 512 --batch-budget 8192 \
    --n-targets 200 --gauge auto \
    --window-size 2.2 3.2 --window-flux 2500 50000 --window-terms score \
    --window-draws 65536 --window-guard 1000 2>&1 | tee logs/guard_smoke.log \
    | grep -E "score guard|R_s =|uncorrected" \
    || { echo "SMOKE TEST FAILED -- see logs/guard_smoke.log"; exit 1; }

echo "=== gauss2_v3  guard 1000 ==="
python -u bias.py --pop gauss2_v3 --flow flows/centroid_g2v3.eqx $COMMON \
    --window-draws 16777216 --window-guard 1000 \
    --save-pqr pqr/g2v3_20k_guard1000.npz 2>&1 | tee logs/bias_g2v3_guard1000.log

echo "=== gauss2_v3  guard 100 (sensitivity) ==="
python -u bias.py --pop gauss2_v3 --flow flows/centroid_g2v3.eqx $COMMON \
    --window-draws 16777216 --window-guard 100 \
    --save-pqr pqr/g2v3_20k_guard100.npz 2>&1 | tee logs/bias_g2v3_guard100.log

echo "=== gauss2_v3d  guard 1000 + floor 1e-3 ==="
python -u bias.py --pop gauss2_v3d --flow flows/centroid_g2v3d.eqx $COMMON \
    --window-draws 16777216 --window-guard 1000 --floor-eps 1e-3 \
    --save-pqr pqr/g2v3d_20k_guard1000_eps1e-3.npz 2>&1 \
    | tee logs/bias_g2v3d_guard1000_eps1e-3.log

echo "GUARD SCAN DONE"
