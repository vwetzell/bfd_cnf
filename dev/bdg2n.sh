#!/bin/bash
# 2026-09-26 -- PROVENANCE RECORD for bulgedisc_g2n: the g2v4n chain (dev/render_g2v4n.sh
# + dev/retrain_g2v4n.sh + the Ke centroid) rebuilt on a bulge+disc population, then the
# 176k varied-conditions test (dev/psfe_g2v4.sh TAG=varobs) on it.
#
# Population `bulgedisc_g2` (imsims sim.py BULGEDISC_G2_*): bulgedisc's box with size
# median / log spread / ellipticity width retuned so its MEASURED moments match the
# g2v4n gauss2_fwd prior (dev/bulgedisc_calib.py, logs/bulgedisc_calib.log): Mr/Mf and
# |e| quantiles agree to <= 0.04 / 0.003; Mc/Mr's low tail sits ~0.1 lower (Sersic).
# Unlike gauss2_fwd, its density is not closed form, and the region sampler is
# gauss2-only, so the targets are the FULL population: N=520000 (~ g2v4 176k's NPOP
# 519313, so ~176k fall in the padded window region).
#
# Everything else as g2v4n: prior 500k at noise 0.093 (seed 0), copies at the
# targets' 0.93, bulk base 150k -> bulk jac 5.0 150k -> shear 1e4 60k -> centroid
# train-sigmax --multi-scale --fit-K 6000.  Targets: seed 2, noise 0.93, seed-2
# varobs conditions (psf e 0.03/-0.02 +/- 0.02, sigma 0.42 +/- 0.012, noise +/- 5%).
# Strictly serial: CPU renders never overlap GPU training ([[no-jax-cpu-alongside-gpu]]).
# DEPTH=sn8: targets at noise 3.674 (x3.95), mean flux S/N 8.0 at the Mf = 3000 window
# edge (31.6 at 0.93; checked on a 60k render).  The prior and bulk/shear flows are
# reused unchanged (the templates' depth does not depend on the survey's); only the
# copies (Sigma_X scales with the TARGETS' noise) and the centroid layer are rebuilt.
# Noise sd stays 5% of the mean.  `deep` runs copies -> targets -> centroid -> gate -> bias.
# Usage: [DEPTH=sn8] bash dev/bdg2n.sh {prior|copies|train|centroid|gate|targets|bias|all|deep}
set -e -o pipefail
cd "$(dirname "$0")/.."
S=../bfd_cnf_imsims
D=$S/data
POP=bulgedisc_g2
TAG=bdg2n
NT=520000
case ${DEPTH:-} in "") TN=0.93; NSD=0.047; SUF="" ;; sn8) TN=3.674; NSD=0.1837; SUF=_sn8 ;;
  *) echo "DEPTH?"; exit 2 ;; esac
OBS="--psf-e1 0.03 --psf-e2 -0.02 --psf-sigma 0.42 --obs-sd 0.02 0.02 0.012 $NSD"
mkdir -p logs flows pqr

prior() {
    ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.sim \
        --n 500000 --seed 0 --pop $POP --noise-sigma 0.093 --add-noise --nproc 24 \
        --out data/moments_${POP}_${TAG}.fits ) 2>&1 | grep -v Warning
}

copies() {
    ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.copies build \
        --n 500000 --seed 0 --pop $POP --noise-sigma $TN --procs 24 \
        --out data/copies_${POP}_${TAG}${SUF}.fits ) 2>&1 | grep -v Warning
}

train() {
    M=$D/moments_${POP}_${TAG}.fits
    python -u bulk.py train --data $M --flow flows/bulk_${TAG}_base.eqx --steps 150000 \
        > logs/bulk_${TAG}_base.log 2>&1
    python -u bulk.py train --data $M --flow flows/bulk_$TAG.eqx \
        --chart-from flows/bulk_${TAG}_base.eqx --jac-weight 5.0 --steps 150000 \
        > logs/bulk_$TAG.log 2>&1
    python -u shear.py train --data $M --flow flows/shear_$TAG.eqx --init flows/bulk_$TAG.eqx \
        --deriv-weight 1e4 --steps 60000 > logs/shear_$TAG.log 2>&1
}

centroid() {
    python -u centroid.py train-sigmax --multi-scale --fit-K \
        --copies $D/copies_${POP}_${TAG}${SUF}.fits --init flows/shear_$TAG.eqx \
        --flow flows/centroid_${TAG}${SUF}_Ke.eqx --steps 6000 --batch 8192 \
        > logs/centroid_${TAG}${SUF}_Ke.log 2>&1
}

gate() {   # CRN finite-difference R_s: isotropy R11 = R22, R12 = 0
    NOISE_SCALE=$(python -c "print($TN / 0.93)") TRAIN_POP=bulgedisc_g2n python -u dev/closed_loop.py selection \
        --flow flows/centroid_${TAG}${SUF}_Ke.eqx > logs/closed_loop_selection_$TAG$SUF.log 2>&1
}

targets() {
    for arm in "g1p02 0.02" "g1m02 -0.02" "g0 0.0"; do
        set -- $arm
        ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.sim \
            --n $NT --seed 2 --pop $POP --noise-sigma $TN --add-noise \
            --g1 $2 $OBS --nproc 24 --out data/targets_bdg2_${1}_176k_varobs${SUF}.fits ) 2>&1 | grep -v Warning
    done
}

bias() {
    python -u bias.py --pop bulgedisc_g2n_176k_varobs${SUF} --flow flows/centroid_${TAG}${SUF}_Ke.eqx \
        --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm \
        --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000 \
        --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 \
        --save-pqr pqr/${TAG}_Ke_176k_varobs${SUF}.npz > logs/bias_${TAG}_Ke_176k_varobs${SUF}.log 2>&1
}

case "${1:-}" in
    prior) prior > logs/render_${TAG}_prior.log ;;
    copies) copies > logs/render_${TAG}${SUF}_copies.log ;;
    train) train ;; centroid) centroid ;; gate) gate ;;
    targets) targets > logs/render_${TAG}_176k_varobs${SUF}.log ;;
    bias) bias ;;
    all) prior > logs/render_${TAG}_prior.log; echo "prior done"
         copies > logs/render_${TAG}_copies.log; echo "copies done"
         targets > logs/render_${TAG}_176k_varobs.log; echo "targets done"
         train; centroid; echo "train done"
         gate || echo "gate FAILED (continuing)"
         bias; echo "ALL DONE" ;;
    deep) copies > logs/render_${TAG}${SUF}_copies.log; echo "copies done"
          targets > logs/render_${TAG}_176k_varobs${SUF}.log; echo "targets done"
          centroid; echo "centroid done"
          gate || echo "gate FAILED (continuing)"
          bias; echo "ALL DONE" ;;
    *) echo "usage: $0 {prior|copies|train|centroid|gate|targets|bias|all|deep}"; exit 2 ;;
esac
