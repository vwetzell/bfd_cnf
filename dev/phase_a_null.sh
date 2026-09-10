#!/usr/bin/env bash
# The null run for A2's window scan: the SAME scan with the selection terms
# drawn from a different seed.
#
# Why this is not optional.  A2 was read off two pqr files, `v11_plain` and
# `v11_psfe00`, and they gave the same `flux_lo` slope (+4.2e-03 and +4.8e-03
# per +/-10%, same sign).  That looks like corroboration and is not: the two
# are independent in their GALAXIES but share `Sigma_X` and `C_M` bit for bit,
# so they were re-solved against the IDENTICAL selection terms.  A slope caused
# by an error in those terms would reproduce across them exactly.
#
# Changing only the draw seed separates the two: if the slope is selection-term
# MC error it moves, and if it is a real dependence of the corrected answer on
# where the flux cut sits it stays.  Section 7: "a null run is part of the
# experiment, not an optional extra".
set -e -o pipefail
cd "$(dirname "$0")/.."

# Do not overlap with the main Phase A pass -- 2^24 twice does not fit on the
# 16 GB card.
while pgrep -f "dev/window_scan.py" > /dev/null; do sleep 60; done

python -u dev/window_scan.py --pop bulgedisc_v3 --seed 7 \
    --pqr pqr/v11_plain.npz pqr/v11_psfe00.npz \
    2>&1 | tee logs/phase_a/circular_scan_s7.log

echo "PHASE A NULL DONE"
