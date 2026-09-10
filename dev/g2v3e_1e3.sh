#!/bin/bash
# gauss2_v3e -- the same population and depth as g2v3d at 9.2x the targets, to
# put sigma(m1) at 1e-3.  Plan and justification: NEXT_SESSION_1e3.md.
#
# THIS MEASURES sigma, NOT |m1|.  The 500k run gave m1 = -0.0030 +/- 0.0031,
# one sigma from zero and sitting right on the old ~5e-3 centroid floor.  At
# sigma = 1e-3 that becomes a 3-sigma detection or a clean null.  Neither is
# available today, and no target count touches the centroid spin-2 undershoot
# that is the known floor ([[road-to-1e-3]]).
#
# ARITHMETIC, measured off the 500k run and not assumed:
#   k = sigma * sqrt(N_in) = 0.00303 * sqrt(117247) = 1.038
#   sigma = 1e-3  ->  N_in = 1.08M  ->  catalog 4.6M at the 23.5% in-window rate
#   integrated = 4.6M x 50.9% (pre-filter) x 2 arms = 4.68M target-arms
#   at 23.32 ms (BFD_DEVICE_PQR=1)  ->  ~30 h
#
# SLICED, because a single 30 h job that dies at hour 29 loses everything.
# Five 920k slices, each ~6 h with its own --save-pqr; `dev/pqr_cat.py` joins
# them and the re-solve cannot tell the difference.  The per-slice m1 is a free
# consistency check -- read the UNCORRECTED one, its bar is 5x tighter than the
# corrected one at 2^24 draws.
#
# EVERY FLAG THAT IS NOT OPTIONAL:
#   --floor-eps 1e-3 --window-guard 100   without them the deep arm returned
#                                         m1 = -5.75 +/- 116.64
#   --window-size 2.2 3.2                 the CEILING.  Without it the window
#                                         admits objects smaller than a point
#                                         source and every number is void.
#   --window-terms score                  flow-only; never `templates`
#   --gauge auto                          gauge P's Var[score] is infinite
#   --batch-budget 65536                  bounds memory; omitting it OOMs the
#                                         16 GB card (a run sits near 14 GB)
#   BFD_DEVICE_PQR=1                      23.32 vs 26.04 ms/target/arm.  It
#                                         only wins past ~52k integrated
#                                         targets; this run integrates 2.3M.
#
# ONE JAX JOB AT A TIME -- GPU *and* CPU.  A probe beside a
# GPU job crashed this box into a swap spiral and a reboot.
#
# Usage:  setsid bash dev/g2v3e_1e3.sh smoke     # 5k, ~2 min, run this FIRST
#         setsid bash dev/g2v3e_1e3.sh slice 0   # one slice, ~6 h
#         setsid bash dev/g2v3e_1e3.sh null      # draw-seed null, after slice 0
#         setsid bash dev/g2v3e_1e3.sh run       # all 5 slices, ~30 h
#         bash dev/g2v3e_1e3.sh cat              # join the slice PQRs
#         bash dev/g2v3e_1e3.sh resolve          # offline, the quotable number
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

export BFD_DEVICE_PQR=1
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax

N_SLICE=920000
SLICES=5

bias() {   # bias <n> <offset> <extra...>
    local n=$1 off=$2; shift 2
    python -u bias.py \
        --pop gauss2_v3e --flow flows/centroid_g2v3d.eqx \
        --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536 \
        --n-targets "$n" --target-offset "$off" \
        --gauge auto --no-zero-arm \
        --prefilter-pad 0.2 --prefilter-sample 0.15 \
        --window-size 2.2 3.2 --window-flux 2500 50000 \
        --window-terms score --window-draws 16777216 \
        --floor-eps 1e-3 --window-guard 100 "$@"
}

# The render used the wrong SIGMA or POP if this does not reproduce the 500k
# run's pre-filter fractions: ~50.9% integrated, ~23.5% of the catalog in the
# window.  Two minutes, and it is the only thing standing between a typo in the
# render environment and 30 wasted GPU hours.
smoke() {
    # 2^22 draws: the smoke test is about the pre-filter fractions, not m1,
    # and 5000 targets cannot measure a bias anyway.
    bias 5000 0 --window-draws 4194304 2>&1 | tee logs/bias_g2v3e_smoke.log
    grep -E "pre-filter|targets kept" logs/bias_g2v3e_smoke.log
}

slice() {   # slice <i>
    local i=$1
    echo "=== slice $i / offset $((i * N_SLICE)) ==="
    bias $N_SLICE $((i * N_SLICE)) --save-pqr "pqr/g2v3e_s$i.npz" \
        2>&1 | tee "logs/bias_g2v3e_s$i.log"
}

run() {
    for i in $(seq 0 $((SLICES - 1))); do slice "$i"; done
    cat_pqr
}

cat_pqr() {
    python dev/pqr_cat.py pqr/g2v3e_all.npz \
        $(for i in $(seq 0 $((SLICES - 1))); do echo pqr/g2v3e_s$i.npz; done)
}

# [[draw-noise-not-negligible-at-depth]]: the deep arm's per-target draw noise
# does not fully converge in S.  The (2.2, 3.2) window excludes most of the
# offenders, but CHECK, do not assume: re-integrate slice 0 at a different draw
# seed and re-solve BOTH files in one window_scan call.  The galaxy set and the
# window are identical between the two (the window is cut on the catalog's
# observed moments, not on the draws) and the selection terms are shared, so
# their m1 difference is draw noise and nothing else -- 2^24 is plenty for it.
# Above ~3e-4, raise --samples before quoting anything.
#
# Run this AFTER slice 0 and BEFORE slices 1-4: it costs 6 GPU-h and catches a
# draw-noise problem at hour 12 instead of hour 36.
null() {
    bias $N_SLICE 0 --draw-seed 4242 --save-pqr pqr/g2v3e_s0_seed4242.npz \
        2>&1 | tee logs/bias_g2v3e_s0_seed4242.log
    python -u dev/window_scan.py --pop gauss2_v3e \
        --flow flows/centroid_g2v3d.eqx \
        --pqr pqr/g2v3e_s0.npz pqr/g2v3e_s0_seed4242.npz \
        --log2-draws 24 --no-scan 2>&1 | tee logs/null_g2v3e_s0.log \
        | grep -E "re-solve|corrected m1"
}

# `ghat` is affine in (P_s, Q_s, R_s), so the selection terms re-solve off the
# saved PQR with NO target integration.  2^28, not 2^26: the seed spread is
# 2.4e-4 at 2^26, a quarter of a 1e-3 bar.  ~2.7 h each, CPU, no GPU work --
# so nothing else may be running.  The two-seed spread IS the selection MC
# error; R_s12 and the traceless part are forced zeros, so their magnitude is
# that error without differencing.
resolve() {
    for seed in 0 7; do
        echo "=== re-solve 2^28, seed $seed ==="
        python -u dev/window_scan.py --pop gauss2_v3e \
            --flow flows/centroid_g2v3d.eqx --pqr pqr/g2v3e_all.npz \
            --log2-draws 28 --no-scan --seed "$seed" \
            2>&1 | tee "logs/resolve_g2v3e_s$seed.log" | tail -6
    done
}

case "${1:-}" in
    smoke)   smoke ;;
    slice)   slice "$2" ;;
    run)     run ;;
    cat)     cat_pqr ;;
    null)    null ;;
    resolve) resolve ;;
    *) echo "usage: $0 {smoke|run|null|resolve}"; exit 2 ;;
esac
echo "G2V3E $1 DONE"
