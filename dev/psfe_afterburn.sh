#!/bin/bash
# Afterburner for dev/bias_psfe_g2v3d.sh: re-solve each config's saved PQR at
# 2^24 window draws instead of the in-run 2^20, via dev/window_scan.py's
# offline resolve.  `ghat` is affine in (P_s, Q_s, R_s), so this needs no
# target re-integration -- just one pass of flow evaluations over the prior
# draws (~10 min at 2^24 per window_scan.py's own comment), then a numpy
# re-solve.  See the user's ask: 2^20 corrected-m1 carries an unquantified
# ~9e-03 MC-convergence error on top of its printed bootstrap +/-
# (dev/g2v3d_500k.sh's own note), and 2^24 is what retires that.
#
# ONE JAX GPU JOB AT A TIME: bias_psfe_g2v3d.sh is still running its own
# target integration when this is launched, so each config's re-solve WAITS
# for bias.py to be off the GPU (no other bias.py process alive) before it
# runs, and re-checks after -- never runs alongside the main job.
#
# `--no-prefilter-weights`: bias_psfe_g2v3d.sh did not use --prefilter-pad,
# so its PQR files carry no `prefilter_w` and window_scan.py would otherwise
# refuse them.
# `--window-guard 100`: matches bias_psfe_g2v3d.sh's own --window-guard.
# `--no-scan`: nominal window only (A1 re-quote + A3 c=0), no local boundary
# perturbation scan -- that is a separate, deliberate ask if wanted later.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr logs/phase_a

FLOW=flows/centroid_g2v3d_sigmaxblock_multiscale.eqx

for tag in psfe00 psfe1p02 psfe2p02 psfe1m02 \
           psfe1p05 psfe2p05 psfe1m05 psfe1p10 psfe2p10 psfe1m10; do
    pqr="pqr/g2v3d_${tag}.npz"
    out="logs/resolve_g2v3d_${tag}.log"
    [ -f "$out" ] && { echo "skip $tag: $out already exists"; continue; }

    echo "waiting for $pqr ..."
    while [ ! -f "$pqr" ]; do sleep 30; done

    echo "waiting for the GPU to be free ..."
    while pgrep -f "bias.py --pop" > /dev/null; do sleep 30; done
    sleep 5   # let a just-exited process release its GPU context

    echo "=== resolving $tag at 2^24 draws ==="
    python -u dev/window_scan.py --pop "gauss2_v3d_${tag}" --flow $FLOW \
        --pqr "$pqr" --no-scan --no-prefilter-weights --window-guard 100 \
        2>&1 | tee "$out"
done
echo "psfe afterburn done"
