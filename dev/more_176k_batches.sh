#!/bin/bash
# 2026-09-25 -- PROVENANCE RECORD for gauss2_v4n_176k_s{3..7}_varobs: five more 176k
# real g2v4 target batches (seeds 3-7), render and bias.py flags identical to
# gauss2_v4n_176k (dev/more_targets_and_chunks.sh) except --seed and the
# per-galaxy varied conditions (user choice: confounds the round-PSF
# comparisons, but a varied-conditions run is needed anyway).  Spreads are
# TAG=varobs's (dev/psfe_g2v4.sh; noise sd 5% of its mean); the MEANS differ
# per seed (PSF e1, e2, sigma, noise below).  Each seed's Sigma_X keeps
# <=0.8% of galaxies outside the centroid layer's trained 0.714-1.4x range
# (600-galaxy renders; seed 6 noise lowered 0.97 -> 0.95 for that).  With it,
# ~1M targets -> m1 +/- ~0.001.  Serial: CPU render, then GPU bias, per seed
# ([[no-jax-cpu-alongside-gpu]]).
# Usage: [SEEDS="3 4 5 6 7"] bash dev/more_176k_batches.sh
set -e -o pipefail
cd "$(dirname "$0")/.."
S=../bfd_cnf_imsims
obs_args() {   # per-seed means: psf_e1 psf_e2 psf_sigma noise_sigma
    case $1 in 3) set -- -0.03 0.02 0.40 0.90 ;; 4) set -- 0.00 0.04 0.44 0.88 ;;
      5) set -- 0.04 0.01 0.38 0.95 ;; 6) set -- -0.02 -0.03 0.41 0.95 ;;
      7) set -- 0.02 -0.04 0.43 0.91 ;; *) echo "no means for seed $1" >&2; exit 2 ;; esac
    echo "--psf-e1 $1 --psf-e2 $2 --psf-sigma $3 --noise-sigma $4 --obs-sd 0.02 0.02 0.012 $(python -c "print(round(0.05 * $4, 5))")"
}
BIAS_FLAGS="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --flow flows/centroid_g2v4n_Ke.eqx"
for SEED in ${SEEDS:-3 4 5 6 7}; do
    OBS_ARGS=$(obs_args $SEED)
    for arm in "g1p02 0.02" "g1m02 -0.02" "g0 0.0"; do
        set -- $arm
        ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.sim \
            --n 176000 --seed $SEED --pop gauss2_fwd --add-noise \
            --g1 $2 $OBS_ARGS --region 3000 20000 2.2 3.2 --nproc 24 \
            --out data/targets_g2v4_${1}_176k_s${SEED}_varobs.fits ) 2>&1 | grep -v Warning
    done > logs/render_g2v4_176k_s${SEED}_varobs.log
    python -u bias.py --pop gauss2_v4n_176k_s${SEED}_varobs $BIAS_FLAGS \
        --save-pqr pqr/g2v4n_Ke_176k_s${SEED}_varobs.npz > logs/bias_g2v4n_Ke_176k_s${SEED}_varobs.log 2>&1
    echo "seed $SEED done $(date)"
done
echo ALL DONE
