#!/bin/bash
# Phase C2: the e_psf amplitude scan, 0.05 and 0.10 against the existing 0.20.
#
# The SCALING EXPONENT is the point.  Measured off the catalogs alone
# (2026-09-04), the two channels an elliptical PSF can reach the estimator
# through separate cleanly by their power law:
#
#     Sigma_X spin-2 split   ~ e_psf^1.017     (linear)
#     C_M     spin-2 split   ~ e_psf^2.030     (quadratic)
#
# so a fitted d log|dc| / d log e_psf near 1 puts the leak in Sigma_X and near
# 2 puts it in C_M -- the same question the no-centroid bisect (C1) answers,
# but this also gives the extrapolation from the exaggerated 0.2 down to a
# DES-like 0.05, which the bisect does not.
#
# Only FOUR runs, not six: with e_psf = 0 the amplitude does nothing, and the
# e00 catalogs at tags 05/10 are byte-identical in `moments` to tag 20
# (verified).  `pqr/v11_psfe00.npz` is therefore the baseline for every point
# on the scan, and re-running it would only add MC noise to the difference.
#
# Same flags as `dev/bias_psfe.sh` -- these have to be comparable to the 0.2
# runs galaxy for galaxy, and every config shares --n and --seed.
#
# --window-draws 2^24, not the 2^20 that script used: at 2^20 a corrected m1
# carries an unquoted +/-4e-03 (GUIDING_PRINCIPLES 3.1).  The uncorrected
# differences these runs are read from do not depend on it, but there is no
# reason to write another 2^20 number to disk.
#
# ONE JAX GPU JOB AT A TIME.  ~40 min each; the 16 GB card OOMs at 2^24 with a
# second JAX process on it.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

for pop in bulgedisc_v3_psfe1p05 bulgedisc_v3_psfe2p05 \
           bulgedisc_v3_psfe1p10 bulgedisc_v3_psfe2p10; do
    tag=${pop#bulgedisc_v3_}
    python -u bias.py --pop "$pop" --flow flows/centroid_v11.eqx \
        --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536 \
        --n-targets 20000 --gauge auto \
        --window-size 2.2 3.2 --window-flux 2500 50000 \
        --window-terms score --window-draws 16777216 \
        --save-pqr pqr/v11_${tag}.npz \
        2>&1 | tee logs/bias_v11_${tag}.log
done
echo "AMPLITUDE SCAN DONE"
