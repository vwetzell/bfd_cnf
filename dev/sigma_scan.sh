#!/bin/bash
# Does the centroid layer's CLOSED FORM get Sigma_X right, or only at the one
# point it was calibrated at?
#
# The layer has no trainable parameters: Sigma_X enters as exact algebra
# (Sigma_u = J^-1 Sigma_X J^-T, then (I +/- R Sigma_u)), so its Sigma_X
# dependence is derived, not fitted, and heteroscedastic targets cost nothing.
# The one fitted thing is the scalar `gain` patching the Gaussian-in-k ansatz's
# ~30% ellipticity undershoot -- and that gain is CONSTANT in Sigma_X.  It
# multiplies a displacement that is itself ~ Sigma_X, so one scalar is right
# only if the ansatz's FRACTIONAL error does not move with depth.  This scans
# it.
#
# `--sigma-scale S` reweights the same copy catalog to S^2 Sigma_X, so BOTH
# sides of the comparison move: `weighted_copy_mean` re-derives the catalog
# truth at the new depth and `dm_dsigma` re-evaluates the layer there.  The
# copy grid is only valid within a factor SIGRANGE = 1.4 in sigma_XY, which is
# what bounds the scan -- outside it the CALIBRATION route runs out before the
# formula does.
#
# A ratio flat in S means the one scalar transfers and a varying-depth catalog
# needs no new machinery.  A ratio that trends means the gain has to become a
# function of Sigma_X (or the ansatz has to improve) before heteroscedastic
# targets are trustworthy.
#
# Usage: bash dev/sigma_scan.sh [flow] [copies]
set -e
cd "$(dirname "$0")/.."
FLOW=${1:-flows/centroid_v3.eqx}
COPIES=${2:-../bfd_cnf_imsims/data/copies_bulgedisc_v3.fits}

for s in 0.72 0.80 0.90 1.00 1.10 1.25 1.40; do
    echo "=== sigma-scale $s"
    python -u centroid.py check --copies "$COPIES" --flow "$FLOW" \
        --sigma-scale "$s" 2>&1 | grep -E "Sigma_X =|ESS|ellipticity|^  M"
done
