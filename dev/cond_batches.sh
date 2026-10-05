#!/bin/bash
# 2026-10-04 -- PROVENANCE RECORD: the condition tests of the BFD-2016 gap-closing plan.
# Every catalog re-renders the SEED-2 sn8 galaxies + noise draws (exactly dev/sn8_diag.sh's
# sn8r render otherwise), so each pairs row-for-row with the sn8r baseline.  cv6 s2 flow,
# tapered window; in-run selection terms at --seed 0 (common random numbers across runs).
#   cond: A PSF e (0.0354, 0.0354) | B PSF sigma 0.44 | C noise 4.41 (x1.2) |
#         D Moffat beta 3.5 (FWHM-matched) at A's e        -- fixed conditions, --pool-reuse 8
#   condE: E = D with the PSF ellipticity REVERSED: E - D cancels Moffat's own noise
#         structure, leaving 2 x the Moffat PSF leakage (D's bar alone was 2e-4)
#   wide: W per-galaxy PSF e sd 0.03/0.03, sigma sd 0.02, noise sd 0.367 (no pool possible)
#   flow: cv6 s3 flow on sn8r batches s3-s5 (flow-training scatter; s2 already exists)
# Sigma_X of every condition checked inside the cv6 centroid's trained cloud first.
# Serial: CPU render, then GPU bias ([[no-jax-cpu-alongside-gpu]]).
# Usage: bash dev/cond_batches.sh {cond|wide|flow} >> logs/cv6/cond_batches.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; L=logs/cv6
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005 --window-jacobian full
  --window-taper 0.05 150"
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
render() {   # tag, extra sim flags...
    local tag=$1; shift
    for arm in "g1p02 0.02" "g1m02 -0.02" "g0 0.0"; do
        set -- $arm "$@"; local a=$1 g=$2; shift 2
        local out=data/targets_bdg2_${a}_176k_sn8_$tag.fits
        [ -f $S/$out ] || ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.sim \
            --n 520000 --seed 2 --pop bulgedisc_g2 --noise-sigma 3.674 --add-noise --g1 $g "$@" \
            --nproc 24 --out ${out%.fits}_tmp.fits && mv ${out%.fits}_tmp.fits $out ) 2>&1 | grep -v Warning \
            || { echo "render $tag $a FAILED"; exit 1; }
    done
    echo "$tag rendered $(date +%T)"
}
run_bias() {   # pop, flow, pqr name, extra flags
    local t0=$SECONDS; ollama_off
    [ -f pqr/$3.npz ] || python -u bias.py --pop $1 --flow $2 $BF $4 --save-pqr pqr/$3.npz \
        > $L/bias_$3.log 2>&1 || { echo "bias $3 FAILED"; exit 1; }
    echo "WALL $((SECONDS - t0)) s" >> $L/bias_$3.log
    echo "== $3 $(date +%T)"; grep -E "windowed|P_s =|top draw" $L/bias_$3.log
}
F2=flows/cv6/centroid_s2.eqx; F3=flows/cv6/centroid_s3.eqx
MODE=$1
case $1 in
  cond) for spec in "condA --psf-e1 0.0354 --psf-e2 0.0354" "condB --psf-sigma 0.44" \
                    "condC --noise-sigma 4.41" "condD --psf-profile moffat --psf-e1 0.0354 --psf-e2 0.0354"; do
            set -- $spec; tag=$1; shift
            render $tag "$@"
            run_bias bulgedisc_g2n_176k_sn8_$tag $F2 cv6_s2_$tag "--pool-reuse 8"
        done ;;
  condE) render condE --psf-profile moffat --psf-e1 -0.0354 --psf-e2 -0.0354
        run_bias bulgedisc_g2n_176k_sn8_condE $F2 cv6_s2_condE "--pool-reuse 8" ;;
  wide) render condW --obs-sd 0.03 0.03 0.02 0.367
        run_bias bulgedisc_g2n_176k_sn8_condW $F2 cv6_s2_condW "" ;;
  flow) for s in 3 4 5; do run_bias bulgedisc_g2n_176k_sn8r_s$s $F3 cv6_s3_sn8r_s$s "--pool-reuse 8"; done ;;
  *) echo "usage: $0 {cond|condE|wide|flow}"; exit 2 ;;
esac
echo "$MODE DONE $(date)"
