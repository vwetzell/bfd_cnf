#!/bin/bash
# Afterburner for dev/bias_psfe_g2v3d_mcfix.sh: re-solve each config's saved
# PQR at 2^24 window draws, mirroring dev/psfe_afterburn.sh. Explicit
# --cache paths (suffix _mcfix): window_scan.py's default cache key does NOT
# include the flow path, so without this a re-solve on this new checkpoint
# would silently load the BASELINE checkpoint's stale cached selection terms
# for the same pop name (see dev/psfe_joint_slope_densegrid.py's identical
# precaution).
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr logs/phase_a

FLOW=flows/centroid_g2v3d_sigmaxblock_mc_lr3e4_s0.eqx
SUFFIX=_mcfix

for tag in psfe00 psfe1p05 psfe2p05 psfe1m05; do
    pqr="pqr/g2v3d_${tag}${SUFFIX}.npz"
    out="logs/resolve_g2v3d_${tag}${SUFFIX}.log"
    [ -f "$out" ] && { echo "skip $tag: $out already exists"; continue; }

    echo "waiting for $pqr ..."
    while [ ! -f "$pqr" ]; do sleep 30; done

    echo "waiting for the GPU to be free ..."
    while pgrep -f "bias.py --pop" > /dev/null; do sleep 30; done
    sleep 5

    echo "=== resolving $tag at 2^24 draws (mcfix) ==="
    python -u dev/window_scan.py --pop "gauss2_v3d_${tag}" --flow $FLOW \
        --pqr "$pqr" --no-scan --no-prefilter-weights --window-guard 100 \
        --cache "logs/phase_a/terms_gauss2_v3d_${tag}_24_s0_1w_g100${SUFFIX}.npz" \
        2>&1 | tee "$out"
done
echo "psfe afterburn mcfix done"
