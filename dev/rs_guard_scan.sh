#!/bin/bash
# Map the guard truncation, because it is now the LARGEST error in the number.
#
#   unguarded, 2^26, 6 seeds:   R_s11 +0.06192   m1 -0.00869 +/- 0.00049
#   guard 100, 2^26, 4 seeds:   R_s11 +0.07267   m1 -0.01290 +/- 0.00069
#   guard 100, 2^28, 2 seeds:   R_s11 +0.07095   m1 -0.01227 +/- 0.00055
#
# Guarded 2^26 vs 2^28 agree at 0.7 sigma, so WITH the guard the estimator is
# converged in draws and the top-draw share falls as it should (0.5% -> 0.2%).
# But guard-100 against unguarded is a 4.2e-3 shift in m1 at 5.0 sigma -- four
# times the 1.0e-3 statistical bar the 26 remaining GPU-hours would buy.  So
# the truncation, not the galaxy count, is the binding error.
#
# Note the SIGN: truncating a heavy-tailed mean usually lowers it, and here it
# RAISES R_s (0.062 -> 0.073) and P_s (0.2326 -> 0.2416).  The guard is cutting
# draws whose (R + QQ^T) is large and NEGATIVE, not just large.  Whatever the
# right cut is, "drop the biggest |Q|, |R|" is not obviously it.
#
# This walks the factor at fixed draws and seeds.  If m1 plateaus over a decade
# of guard, quote the plateau; if it slides all the way, the guard is choosing
# the answer and the fix is upstream -- the unphysical draws themselves
# ([[rs-tail-is-unphysical-draws]]: in_support is a BOX, the manifold CURVED).
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
for guard in 30 300 1000 3000; do
  for s in 0 7; do
    echo "=== 2^26 guard $guard, seed $s ==="
    python -u dev/window_scan.py --pop gauss2_v3e \
        --flow flows/centroid_g2v3d.eqx --pqr pqr/g2v3e_s0.npz \
        --log2-draws 26 --draw-chunk 16777216 --no-scan --seed "$s" \
        --window-guard "$guard" \
        2>&1 | tee "logs/rs_g${guard}_26_s$s.log" \
        | grep -E "nominal|top draw|WARNING|Error|Traceback|RESOURCE" || true
  done
done
echo "GUARD SCAN FINISHED"
