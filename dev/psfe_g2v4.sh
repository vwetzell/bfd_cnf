#!/bin/bash
# 2026-09-25 -- PROVENANCE RECORD for gauss2_v4n_psfe1p05.
# Does an elliptical PSF bias the current flow (centroid_g2v4n_Ke) at the
# precision of the real 176k run (m1 +/-0.0024, c +/-7e-4)?  The 22k real
# targets re-rendered with psf_e1 = 0.05, everything else identical (seed 1,
# n 22000, region, pads) -> same galaxies and pixel noise row for row, so the
# difference against pqr/g2v4n_Ke_20k.npz is paired (dev/psfe_g2v4_paired.py).
# The prior is untouched: deconvolved moments are PSF-independent; the PSF
# reaches the estimator only through the targets' cov (C_M) and cov_odd
# (Sigma_X) ([[isotropy-identity-needs-circular-psf]]).
# SET=176k: the same test on the 176k targets (seed 2, dev/more_targets_and_chunks.sh),
# paired against pqr/g2v4n_Ke_176k.npz.
# TAG=varobs: per-galaxy Gaussian observing conditions (imsims --obs-sd), PSF e
# (0.03 +/- 0.02, -0.02 +/- 0.02), PSF sigma 0.42 +/- 0.012 (1.05 +/- 0.03 x 0.4),
# noise 0.93 +/- 0.047 (5%).  Sigma_X then stays inside the centroid layer's
# trained 0.714-1.4x range for 99% of galaxies; a 10% noise sd would put 16% out.
# SET=tiny: 300 targets, a smoke test of the per-target C_M path in bias.py.
# Usage: [SET=tiny|22k|176k] [TAG=psfe1p05|psfs1p10|varobs] bash dev/psfe_g2v4.sh [render|bias|all]
set -e -o pipefail
cd "$(dirname "$0")/.."
S=../bfd_cnf_imsims
# TAG/PSF_ARGS pick the PSF change; default is the psf_e1 = 0.05 test.
# psfs1p10: PSF sigma 0.4 -> 0.44 arcsec (+10%, round).
TAG=${TAG:-psfe1p05}
case $TAG in psfe1p05) PSF_ARGS="--psf-e1 0.05" ;; psfs1p10) PSF_ARGS="--psf-sigma 0.44" ;;
  varobs) PSF_ARGS="--psf-e1 0.03 --psf-e2 -0.02 --psf-sigma 0.42 --obs-sd 0.02 0.02 0.012 0.047" ;; *) echo "TAG?"; exit 2 ;; esac
SET=${SET:-22k}
case $SET in tiny) N=300; SEED=2; POP=gauss2_v4n_tiny ;; 22k) N=22000; SEED=1; POP=gauss2_v4n ;; 176k) N=176000; SEED=2; POP=gauss2_v4n_176k ;; *) echo "SET?"; exit 2 ;; esac
BIAS_FLAGS="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --flow flows/centroid_g2v4n_Ke.eqx"

render() {
    for arm in "g1p02 0.02" "g1m02 -0.02" "g0 0.0"; do
        set -- $arm
        ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.sim \
            --n $N --seed $SEED --pop gauss2_fwd --noise-sigma 0.93 --add-noise \
            --g1 $2 $PSF_ARGS --region 3000 20000 2.2 3.2 --nproc 24 \
            --out data/targets_g2v4_${1}_${SET}_${TAG}.fits ) 2>&1 | grep -v Warning
    done
}

bias() {
    python -u bias.py --pop ${POP}_${TAG} $BIAS_FLAGS --save-pqr pqr/g2v4n_Ke_${SET}_${TAG}.npz
}

case "${1:-all}" in
    render) render > logs/render_g2v4_${SET}_${TAG}.log ;;
    bias) bias > logs/bias_g2v4n_Ke_${SET}_${TAG}.log 2>&1 ;;
    all) render > logs/render_g2v4_${SET}_${TAG}.log
         bias > logs/bias_g2v4n_Ke_${SET}_${TAG}.log 2>&1
         echo ALL DONE ;;
    *) echo "usage: $0 {render|bias|all}"; exit 2 ;;
esac
