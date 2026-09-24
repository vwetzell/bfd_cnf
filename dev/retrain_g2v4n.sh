#!/bin/bash
# g2v4n full retrain + health gate + bias estimate -- unattended overnight
# chain (2026-09-22).  THIS FILE IS THE PROVENANCE RECORD for the flows it
# produces.  Waits for the prior/copies renders (dev/render_g2v4n.sh) if they
# are still running, then trains bulk -> shear -> centroid sequentially (ONE
# GPU job at a time), runs the R_s health gate, and finishes with a bias
# estimate on the EXISTING g2v4 target catalogs (fixed isotropic PSF).
#
# Recipe: bulk baseline (no jac, supplies the chart) -> bulk with
# --jac-weight 5.0 (chart frozen) -- the single lever this session's sweep
# found actually suppressed the diagnosed leader net 155x, even though it
# did not fully close the health gate alone -- -> shear (--deriv-weight 1e4)
# -> centroid (train-sigmax --multi-scale; no --anisotropic, PSF is isotropic
# here). Step counts match rebuild_v3.sh's convergence gates (150k/150k/60k)
# and PSFE_PROVENANCE's centroid setting (6000 steps, batch 8192).
set -e -o pipefail
cd "$(dirname "$0")/.."
D=../bfd_cnf_imsims/data
mkdir -p logs flows pqr

wait_for() {
    local logfile=$1 marker=$2
    echo "waiting on: $marker (polling $logfile every 30s)"
    while ! grep -q "$marker" "$logfile" 2>/dev/null; do sleep 30; done
    echo "  -> ready"
}

wait_for logs/render_g2v4n_prior.log "wrote data/moments_gauss2_fwd_g2v4n.fits"

echo "=== 2a bulk baseline (no jac) ==="
python -u bulk.py train --data $D/moments_gauss2_fwd_g2v4n.fits \
    --flow flows/bulk_g2v4n_base.eqx --steps 150000 \
    2>&1 | tee logs/bulk_g2v4n_base.log

echo "=== 2b bulk, jac_weight=5.0, chart frozen from 2a ==="
python -u bulk.py train --data $D/moments_gauss2_fwd_g2v4n.fits \
    --flow flows/bulk_g2v4n.eqx --chart-from flows/bulk_g2v4n_base.eqx \
    --jac-weight 5.0 --steps 150000 \
    2>&1 | tee logs/bulk_g2v4n.log

echo "=== 2c shear ==="
python -u shear.py train --data $D/moments_gauss2_fwd_g2v4n.fits \
    --flow flows/shear_g2v4n.eqx --init flows/bulk_g2v4n.eqx \
    --deriv-weight 1e4 --steps 60000 \
    2>&1 | tee logs/shear_g2v4n.log

echo "=== leader corner on the SHEAR flow ==="
python -u dev/leader_corner_plot.py --flow flows/shear_g2v4n.eqx \
    --pop gauss2_v4n --out dev/leader_corner_g2v4n_shear.png \
    2>&1 | tee logs/leader_corner_g2v4n_shear.log || true

wait_for logs/render_g2v4n_copies.log "wrote data/copies_gauss2_fwd_g2v4n.fits"

echo "=== provenance check (prior + copies + targets) ==="
bash dev/render_g2v4n.sh check 2>&1 | tee logs/check_g2v4n.log || true

echo "=== 2d centroid (train-sigmax, multi-scale, isotropic) ==="
python -u centroid.py train-sigmax --multi-scale \
    --copies $D/copies_gauss2_fwd_g2v4n.fits --init flows/shear_g2v4n.eqx \
    --flow flows/centroid_g2v4n.eqx --steps 6000 --batch 8192 \
    2>&1 | tee logs/centroid_g2v4n.log

echo "=== final health gate: CRN finite-difference R_s (isotropy R11=R22, R12=0) ==="
python -u dev/closed_loop.py selection --flow flows/centroid_g2v4n.eqx \
    2>&1 | tee logs/closed_loop_selection_g2v4n.log

echo "=== flow-vs-truth diagnostic (caveat: prior is now noisy, so a"
echo "residual flow/truth mismatch now legitimately includes noise"
echo "broadening, not purely a training artifact -- read alongside the"
echo "health gate, not as a standalone pass/fail) ==="
python -u dev/flow_vs_truth_check.py --flow flows/centroid_g2v4n.eqx \
    --pop gauss2_v4n --stage centroid \
    2>&1 | tee logs/flow_vs_truth_g2v4n.log || true

echo "=== leader corner plot (final centroid flow's shear-stage sibling) ==="
python -u dev/leader_corner_plot.py --flow flows/shear_g2v4n.eqx \
    --pop gauss2_v4n --out dev/leader_corner_g2v4n.png \
    2>&1 | tee logs/leader_corner_g2v4n.log || true

echo "=== bias estimate on the EXISTING g2v4 targets (fixed isotropic PSF) ==="
python -u bias.py --pop gauss2_v4n --flow flows/centroid_g2v4n.eqx \
    --samples 8192 --alpha 0.5 --chunk 2048 --batch-budget 16384 \
    --no-zero-arm --support --floor-eps 0 \
    --window-size 2.2 3.2 --window-flux 3000 20000 \
    --window-terms score --window-draws 16777216 \
    --prefilter-pad 0.2 --prefilter-sample 0 \
    --save-pqr pqr/g2v4n_20k.npz \
    2>&1 | tee logs/bias_g2v4n_20k.log

echo "=== ALL DONE g2v4n ==="
