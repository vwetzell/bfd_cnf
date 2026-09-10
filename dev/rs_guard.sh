#!/bin/bash
# The guarded ladder, after the unguarded 2^28 blew up.
#
# `bias.py` passes `guard=--window-guard` (the runs use 100) into
# `selection_terms_score`; `dev/window_scan.py` passed nothing, so every
# offline re-solve was an UNGUARDED estimator while the in-run number was
# guarded.  Two different estimands, silently.
#
# It survived 2^24 and 2^26 on luck.  At 2^28 seed 0 the code's own rail
# fired -- "one draw carries 100% of R_s11 (Mf = 1999, Mr/Mf = 1.741)" -- and
# R_s11 came back 3.4e8 against a true ~0.062, m1 = -0.61.  Note where that
# draw sits: Mf = 1999 is BELOW the flux cut of 2500 and Mr/Mf = 1.741 is
# below the size cut of 2.2, so it enters only through the smoothed window
# probability F, with an unphysical score.  That is
# [[rs-tail-is-unphysical-draws]] exactly, and I was wrong to read 2^24/2^26's
# clean 1/sqrt(N) scaling as evidence the tail was not there -- those rungs
# simply never sampled it.
#
# The guard is a TRUNCATION and R_s moves with it, so this quotes the
# sensitivity rather than one value: the same seeds at 2^26 guarded and
# unguarded, then 2^28 guarded.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

run() {   # run <log2> <chunk> <seed> <guard>
    python -u dev/window_scan.py --pop gauss2_v3e \
        --flow flows/centroid_g2v3d.eqx --pqr pqr/g2v3e_s0.npz \
        --log2-draws "$1" --draw-chunk "$2" --no-scan --seed "$3" \
        --window-guard "$4" \
        2>&1 | tee "logs/rs_g$4_${1}_s$3.log" \
        | grep -E "nominal|chunks of|top draw|WARNING|Error|Traceback|RESOURCE" || true
}

# 2^26 guarded, against the six unguarded 2^26 seeds already on disk: this is
# the truncation's cost, and it is what gets quoted next to the number.
for s in 0 7 1 2; do echo "=== 2^26 guard 100, seed $s ==="; run 26 16777216 "$s" 100; done
# 2^28 guarded, both seeds -- seed 0 is the one that exploded unguarded.
for s in 0 7; do echo "=== 2^28 guard 100, seed $s ==="; run 28 16777216 "$s" 100; done
echo "RS GUARD LADDER FINISHED"
