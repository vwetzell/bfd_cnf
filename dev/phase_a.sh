#!/usr/bin/env bash
# Phase A: re-quote every saved v11 run at 2^24 selection draws, and scan the
# window locally on the circular-PSF config.  No target work -- `ghat` is
# affine in (P_s, Q_s, R_s), so this is a re-solve off `pqr/*.npz`.
#
# `plain` and `psfe00` are bit-identical in C_M AND Sigma_X, so one pass over
# the draws serves all three circular-PSF pqr files (including the MC null).
#
# `-o pipefail` is not optional: without it `| tee` masks python's exit code
# and the loop marches on through every remaining config after a failure.
#
# One config at a time.  A concurrent JAX process on the same GPU produced
# `RESOURCE_EXHAUSTED ... 9 alive graphs` at 2^24 -- the 16 GB card does not
# hold two of these.
set -e -o pipefail
cd "$(dirname "$0")/.."

L=logs/phase_a
mkdir -p "$L"

# Do not start until nothing else is on the GPU.
while pgrep -f "python -m pytest" > /dev/null; do sleep 30; done

python -u dev/window_scan.py --pop bulgedisc_v3 \
    --pqr pqr/v11_plain.npz pqr/v11_psfe00.npz pqr/v11_psfe00_s7.npz \
    2>&1 | tee "$L/circular_scan.log"

for c in psfe1p psfe2p psfe1m; do
    python -u dev/window_scan.py --pop "bulgedisc_v3_$c" \
        --pqr "pqr/v11_$c.npz" --no-scan 2>&1 | tee "$L/$c.log"
done

echo "PHASE A DONE"
