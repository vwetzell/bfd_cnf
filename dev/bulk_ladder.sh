#!/bin/bash
# Bulk convergence ladder, with the auxiliary loss terms REMOVED.
#
# The question: HANDOFF.md records noiseless m1 going -0.0126 -> -0.0600
# monotonically as the bulk converges (4k -> 150k steps), i.e. the density fit
# improving while the shear bias gets 5x worse.  Every one of those points was
# measured with shear.train's deriv_weight = 1e4 silently active (argparse
# default, never overridden in retrain.sh) supervising the response against
# bfd's per-template dm/dg.
#
# This re-runs the same ladder on a pure-NLL objective.  Two outcomes:
#   * anti-correlation gone  -> the supervision was the cause
#   * anti-correlation stays -> it is the spin-0 response error HANDOFF.md
#                               already fingered, and it is structural
#
# `shear.py check` is a GENUINE held-out validation for the first time here:
# with the supervision off, the dm/dg comparison is no longer a training
# diagnostic.  Its numbers matter as much as m1.
#
# ~1.5 h on a 4080S.  Logs per stage; summary at the end.
cd /home/vwetzell/gitrepos/bfd_cnf || exit 1
D=../bfd_cnf_imsims/data
L=logs/ladder
mkdir -p "$L" flows/ladder

for N in 4000 20000 60000 150000; do
    B=flows/ladder/bulk_$N.eqx
    S=flows/ladder/shear_$N.eqx
    echo "=== bulk $N steps ==="
    python -u bulk.py train --data $D/moments.fits --flow "$B" --steps $N \
        > "$L/bulk_$N.log" 2>&1 || { echo "bulk $N FAILED"; exit 1; }
    echo "=== shear from bulk $N (NLL only) ==="
    python -u shear.py train --data $D/moments.fits --flow "$S" --init "$B" \
        > "$L/shear_$N.log" 2>&1 || { echo "shear $N FAILED"; exit 1; }
    echo "=== noiseless m1 from bulk $N ==="
    python -u bias.py --flow "$S" --pop bulgedisc --samples 0 \
        > "$L/bias_$N.log" 2>&1 || { echo "bias $N FAILED"; exit 1; }
done

echo
echo "==================== LADDER SUMMARY ===================="
printf '%-12s %-12s %s\n' "bulk steps" "val nll" "noiseless m1"
for N in 4000 20000 60000 150000; do
    V=$(grep -o 'val nll [0-9.]*' "$L/shear_$N.log" | tail -1 | awk '{print $3}')
    M=$(grep -oE 'm1 = [+-][0-9.]+ \+/- [0-9.]+' "$L/bias_$N.log" | tail -1)
    printf '%-12s %-12s %s\n' "$N" "${V:-?}" "${M:-?}"
done
echo
echo "--- shear check (held-out dm/dg, supervision OFF) ---"
for N in 4000 20000 60000 150000; do
    echo "bulk $N:"; sed -n '/dm\/dg/,$p' "$L/shear_$N.log" | tail -8
done
