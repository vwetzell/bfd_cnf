#!/bin/bash
# gauss2_v3d (DEEP arm) at the full 500k targets -- the Phase D headline run.
#
# WHY THIS IS NOW AFFORDABLE.  Measured 2026-09-06, all verified bit-identical
# on m1/c1/c2/ghat except where noted:
#
#   psi log-det by chain rule (`make_psi_ld`)            -2.6%
#   draws stay on device (`mixture_draws(to_host=False)`) -3%
#   shear reads Q,R from `response_tensors`               -5%
#   `--no-zero-arm`                                      -30%
#   `--prefilter-pad 0.2 --prefilter-sample 0.15`        ~-50% of what is left
#
# 30 -> 27.1 ms per target per arm, two arms not three, half the targets
# integrated:  500000 x 2 x 0.0271 x 0.50 ~= **3.8 h**, against ~7.5 h before
# the pre-filter and ~11 h before any of it.
#
# WHAT CHANGED IN THE SCIENCE -- read before trusting any old number:
#
#   * **the size window is (2.2, 3.2)**.  Every script hardcoded `2.2 1e9`
#     until 2026-09-06.  Without the ceiling the window admits objects measured
#     SMALLER THAN A POINT SOURCE (Mr/Mf past 3.692575), 8% of the window, and
#     they carry essentially all of the deep arm's draw noise.  The ceiling
#     moves uncorrected m1 by ~0.08 and takes the deep seed-to-seed spread from
#     0.01815 to 0.00179.  Same cut for bulgedisc: the two priors match at p50
#     3.149 vs 3.150 and both are 0.00% past POINT_SOURCE.
#   * **2^24 window draws is NOT converged at this window.**  Corrected m1
#     moved 9.0e-03 between draw seeds at 2^24; at 2^26 it moved 2.4e-04 and
#     the R_s12 monitor fell 4.8x for 16x draws, i.e. the 1/sqrt(N) rate.  So
#     this run takes the terms at 2^24 only for an immediate number and the
#     QUOTABLE value comes from re-solving offline -- `ghat` is affine in
#     (P_s, Q_s, R_s), so `dev/window_scan.py` re-solves off the saved PQR with
#     NO target integration.  Do that at 2^26 on two seeds; go to 2^28 to quote
#     against tau.
#   * `--window-terms score` is the FLOW-based correction and is mandatory --
#     never `templates`.  `--gauge auto` likewise: gauge P's Var[score] is
#     infinite.
#   * `--floor-eps 1e-3 --window-guard 100` on every gauss2 run.  Without them
#     the deep arm returned m1 = -5.75 +/- 116.64.
#
# ONE JAX GPU JOB AT A TIME -- a single run sits near 14 GB on the 16 GB card.
# Launch under `setsid` so a session-level stop cannot kill it.
#
# Usage:  setsid bash dev/g2v3d_500k.sh   # ~3.8 h
#         bash dev/g2v3d_500k.sh resolve  # just the offline re-solve
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

PQR=pqr/g2v3d_500k.npz

run() {
    # `--batch-budget` bounds MEMORY independently of `--samples`:
    # batch = budget // chunk, so OMITTING it makes the batch LARGER and the
    # card OOMs.  Keep it.
    python -u bias.py \
        --pop gauss2_v3d --flow flows/centroid_g2v3d.eqx \
        --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536 \
        --n-targets 500000 --gauge auto --no-zero-arm \
        --prefilter-pad 0.2 --prefilter-sample 0.15 \
        --window-size 2.2 3.2 --window-flux 2500 50000 \
        --window-terms score --window-draws 16777216 \
        --floor-eps 1e-3 --window-guard 100 \
        --save-pqr "$PQR" 2>&1 | tee logs/bias_g2v3d_500k.log
}

# The number to quote.  Two seeds at 2^26: their spread IS the selection MC
# error, and `R_s12`/traceless are forced zeros so their size is that error
# without differencing.  Minutes each, no target work.
resolve() {
    for seed in 0 7; do
        echo "=== re-solve 2^26, seed $seed ==="
        python -u dev/window_scan.py --pop gauss2_v3d \
            --flow flows/centroid_g2v3d.eqx --pqr "$PQR" \
            --log2-draws 26 --no-scan --seed "$seed" \
            2>&1 | tee "logs/resolve_g2v3d_500k_s$seed.log" | tail -6
    done
}

case "${1:-all}" in
    run)     run ;;
    resolve) resolve ;;
    all)     run; resolve ;;
    *) echo "usage: $0 {all|run|resolve}"; exit 2 ;;
esac
echo "G2V3D 500K DONE"
