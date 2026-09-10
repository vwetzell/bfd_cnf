#!/bin/bash
# Phase-0 bias runs: the elliptical-PSF orientation triplet against v10.
#
# Same flags as the run that produced `pqr/v10_score.npz` and
# `logs/bias_v10_score.log`, which IS the e_psf = 0 baseline -- it is not
# re-run, so the baseline carries no fresh MC noise of its own and the
# difference is the cleanest thing available.  Reconstructed from that log:
#   "streaming 8192 draws/target in chunks of 4096 (16 targets at a time)"
#     -> --samples 8192 --chunk 4096 --batch-budget 65536
#   "alpha = 0.5"                          -> --alpha 0.5
#   "support indicator on"                 -> --support (the default)
#   "window: size (2.2, 3.2), flux (2500, 50000)"
#   "19994 targets" of 200000              -> --n-targets 20000, minus 6 that
#                                             failed to recentre
#
# The e = 0 arm is re-run here rather than reusing `pqr/v10_score.npz`: these
# catalogs are 20k draws, and `sample_population(20000, rng(1))` is not the
# first 20000 rows of the 200k draw, so the old baseline covers DIFFERENT
# galaxies and would not pair.  `--n-targets 20000` now takes the whole file.
#
# --gauge auto is mandatory ([[r-is-an-is-artifact]]).  --window-terms score is
# the only estimator with finite variance for THIS window; the pathwise path
# has no finite mean and is void.  --save-pqr always: ghat is affine in the
# selection terms, so any later variant is a re-solve with no GPU.
#
# ONE JAX GPU JOB AT A TIME.  ~40 min each.
#
# Usage:  bash dev/bias_psfe.sh              # the three elliptical configs
#         bash dev/bias_psfe.sh nocentroid   # the bisect: shear_v10, no layer
# pipefail matters here: every run is piped into `tee`, and without it a failed
# (or killed) python is masked by tee's own exit 0, so `set -e` never fires and
# the loop marches on through the remaining configs.  Observed doing exactly
# that.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs pqr

if [ "${1:-}" = "nocentroid" ]; then
    FLOW=flows/shear_v10.eqx; SUF=_nocent; EXTRA=--no-centroid
else
    FLOW=flows/centroid_v11.eqx; SUF=; EXTRA=
fi

# `bulgedisc_v3` (the 200k catalog the v10 headline numbers came from) is run
# FIRST and on the same flow as the rest: the chart fix changes the centroid
# layer's map, so every number in this study has to be re-measured on v11 or
# they are not comparable to each other.  It does not pair against the psfe
# configs -- different draw -- it is the v10-vs-v11 reference point.
for pop in bulgedisc_v3 bulgedisc_v3_psfe00 bulgedisc_v3_psfe1p \
           bulgedisc_v3_psfe2p bulgedisc_v3_psfe1m; do
    tag=${pop#bulgedisc_v3}; tag=${tag#_}; tag=${tag:-plain}
    python -u bias.py --pop "$pop" --flow $FLOW $EXTRA \
        --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536 \
        --n-targets 20000 --gauge auto \
        --window-size 2.2 3.2 --window-flux 2500 50000 \
        --window-terms score --window-draws 1048576 \
        --save-pqr pqr/v11_${tag}${SUF}.npz \
        2>&1 | tee logs/bias_v11_${tag}${SUF}.log
done
echo "bias psfe done"
