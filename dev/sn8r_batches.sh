#!/bin/bash
# 2026-10-03 -- PROVENANCE RECORD for bulgedisc_g2n_176k_sn8r_s{3,4,5}: three more sn8r
# target batches (render exactly as dev/sn8_diag.sh, --seed 3/4/5), each through cv6 s2
# with the tapered window (--window-taper 0.05 150; per-target Q, R are window-independent)
# and --pool-reuse 8.  Pooled with seed 2 by dev/sn8r_pool.py.  Serial: CPU render, then
# GPU bias, per seed ([[no-jax-cpu-alongside-gpu]]).
# Usage: [SEEDS="3 4 5"] bash dev/sn8r_batches.sh > logs/cv6/sn8r_batches.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; L=logs/cv6
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005 --window-jacobian full
  --window-taper 0.05 150 --pool-reuse 8"
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
for s in ${SEEDS:-3 4 5}; do
    for arm in "g1p02 0.02" "g1m02 -0.02" "g0 0.0"; do
        set -- $arm; out=data/targets_bdg2_${1}_176k_sn8r_s$s.fits
        [ -f $S/$out ] || ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.sim \
            --n 520000 --seed $s --pop bulgedisc_g2 --noise-sigma 3.674 --add-noise \
            --g1 $2 --nproc 24 --out ${out%.fits}_tmp.fits && mv ${out%.fits}_tmp.fits $out ) 2>&1 | grep -v Warning \
            || { echo "render s$s $1 FAILED"; exit 1; }
    done
    echo "s$s rendered $(date +%T)"
    ollama_off; t0=$SECONDS
    [ -f pqr/cv6_s2_sn8r_s$s.npz ] || python -u bias.py --pop bulgedisc_g2n_176k_sn8r_s$s \
        --flow flows/cv6/centroid_s2.eqx $BF --save-pqr pqr/cv6_s2_sn8r_s$s.npz \
        > $L/bias_cv6_s2_sn8r_s$s.log 2>&1 || { echo "bias s$s FAILED"; exit 1; }
    echo "WALL $((SECONDS - t0)) s" >> $L/bias_cv6_s2_sn8r_s$s.log
    echo "== s$s $(date +%T)"; grep -E "windowed|pool:" $L/bias_cv6_s2_sn8r_s$s.log
done
echo "ALL DONE $(date)"
