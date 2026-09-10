#!/bin/bash
# A second, independent psf_e=0 20k draw (seed 2), to characterise
# galaxy-sample MC scatter directly: psfe00's own corrected m1 (+0.0275 at
# 2^24) ran ~0.02-0.04 above three prior 20k draws from a different session
# (-0.011 to +0.006), and no population/moment or selection-region difference
# explained it -- see PSFE_PROVENANCE.md.  This renders, integrates, and
# resolves a THIRD 20k draw of the exact same population/pipeline/flags as
# psfe00, at a different render seed, so its scatter can be compared directly.
#
# Render is CPU-bound for bulgedisc but GPU-bound for gauss2_fwd (measured
# this session -- 12-way parallel OOMs a 16GB card); 4-way concurrent with a
# capped memory fraction is what dev/render_psfe.sh settled on, reused here
# via the same script's non-bulgedisc path, just for the single e00 config.
#
# ONE JAX GPU JOB AT A TIME throughout -- the render, then bias.py, then the
# window_scan.py afterburn, run strictly in sequence.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

S=../bfd_cnf_imsims
SEED=2
N=20000
SIGMA=1.86

echo "=== rendering psf_e=0, seed $SEED ==="
for arm in "g0 0.0" "g1p02 0.02" "g1m02 -0.02"; do
    set -- $arm
    ( cd $S && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
        python -u -m imsims.sim --n $N --seed $SEED --pop gauss2_fwd --noise-sigma $SIGMA \
        --psf-e1 0.0 --psf-e2 0.0 --g1 $2 --add-noise \
        --out data/targets_v3psfe00_gauss2fwd_d186_s2_$1_20k.fits )
done

echo "=== bias.py, identical flags to gauss2_v3d_psfe00 ==="
python -u bias.py --pop gauss2_v3d_psfe00_s2 \
    --flow flows/centroid_g2v3d_sigmaxblock_multiscale.eqx \
    --samples 8192 --alpha 0.5 --chunk 2048 --batch-budget 16384 \
    --n-targets 20000 --gauge auto --floor-eps 1e-3 --window-guard 100 \
    --window-size 2.2 3.2 --window-flux 2500 50000 \
    --window-terms score --window-draws 1048576 \
    --save-pqr pqr/g2v3d_psfe00_s2.npz \
    2>&1 | tee logs/bias_g2v3d_psfe00_s2.log

echo "=== afterburn: 2^24 resolve ==="
python -u dev/window_scan.py --pop gauss2_v3d_psfe00_s2 \
    --flow flows/centroid_g2v3d_sigmaxblock_multiscale.eqx \
    --pqr pqr/g2v3d_psfe00_s2.npz --no-scan --no-prefilter-weights \
    --window-guard 100 \
    2>&1 | tee logs/resolve_g2v3d_psfe00_s2.log

echo "psfe00 replicate done"
