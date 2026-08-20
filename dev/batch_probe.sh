#!/bin/bash
# Can a pure-NLL shear layer learn the response if you just give it a bigger batch?
#
# `shear.train`'s docstring says the failure mode is a VARIANCE floor, not a
# rate -- "the knob is `batch`, not `steps`".  Batch is a convergence knob
# (monotone, no optimum to find), so if this works the training recipe stays
# free of tuning constants.
#
# Fixed bulk (bulk_60000.eqx) and fixed steps, so batch is the only variable.
# Metric is `shear.py check`'s held-out dm/dg residual -- a genuine validation
# now that nothing supervises it -- not an expensive bias.py run.
#
# Target to beat, from flows/shear.eqx trained WITH deriv_weight = 1e4:
#     dm/dg     Mf 17.10%  Mr 28.24%  M1 2.65%  M2 2.60%  Mc 45.48%
# Baseline at batch 1024 (pure NLL):
#     dm/dg     Mf 68.12%  Mr 84.37%  M1 9.57%  M2 9.52%  Mc 94.06%
cd /home/vwetzell/gitrepos/bfd_cnf || exit 1
D=../bfd_cnf_imsims/data
B=flows/ladder/bulk_60000.eqx
L=logs/batch_probe
mkdir -p "$L" flows/probe

for BS in 1024 4096 16384 65536; do
    echo "=== batch $BS ==="
    python -u shear.py train --data $D/moments.fits \
        --flow flows/probe/shear_b$BS.eqx --init "$B" --batch $BS \
        > "$L/shear_b$BS.log" 2>&1 || { echo "batch $BS FAILED"; continue; }
done

echo
echo "================== BATCH PROBE: dm/dg residual ==================="
printf '%-10s %-9s %8s %8s %8s %8s %8s\n' batch "val nll" Mf Mr M1 M2 Mc
for BS in 1024 4096 16384 65536; do
    F="$L/shear_b$BS.log"
    [ -f "$F" ] || continue
    V=$(grep -o 'val nll [0-9.]*' "$F" | tail -1 | awk '{print $3}')
    R=$(grep -A2 'RMS residual' "$F" | grep '^dm/dg' | tr -s ' ')
    printf '%-10s %-9s %s\n' "$BS" "${V:-?}" "$(echo "$R" | cut -d' ' -f2-)"
done
echo "supervised   40.4058     17.10%   28.24%    2.65%    2.60%   45.48%"
echo
echo "=================== second order d2m/dg2 ========================="
for BS in 1024 4096 16384 65536; do
    F="$L/shear_b$BS.log"
    [ -f "$F" ] || continue
    printf '%-10s %s\n' "$BS" "$(grep '^d2m/dg2' "$F" | tr -s ' ' | cut -d' ' -f2-)"
done
echo "supervised    2.58%    6.45%   24.96%   24.93%   12.06%"
