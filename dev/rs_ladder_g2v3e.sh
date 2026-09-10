#!/bin/bash
# Is the selection term R_s CONVERGED in prior draws, or still drifting?
#
# The two-seed spread the plan uses as its convergence test measures VARIANCE
# at fixed N.  It cannot see a DRIFT with N, and the existing logs show one:
#
#     2^22   R_s11 0.07209        (smoke)
#     2^24   R_s11 0.06694        (every in-run value)
#     2^26   R_s11 0.06271 / 0.06411  (seeds 0 / 7)
#
# -7.1% then -5.3% per 4x in draws, against a seed half-spread of 1.1%.  The
# drift is 5x the scatter, so it is systematic.  It matters because ghat is
# affine in R_s: the correction is -0.0250 of m1 at R_s11 = 0.06694, so 5% of
# R_s is 1.3e-3 of m1 -- LARGER than the 1.0e-3 statistical bar the 33 GPU-h
# of slices is being spent to buy.
#
# `ghat` needs no target integration for this, so the whole ladder re-solves
# off slice 0's saved PQR.  On the GPU: ~4 min at 2^24, ~16 at 2^26, ~64 at
# 2^28.  Two seeds per rung separates drift from scatter.
#
# Suspected mechanism is already on record: [[rs-tail-is-unphysical-draws]] --
# Hill index < 2, so R_s's estimator has INFINITE VARIANCE and its sample mean
# converges slowly; the seed spread then understates the error by construction.
# The signature is in the logs: the top single draw's share of R_s11 went UP
# from 1.9% at 2^24 to 3.5% at 2^26 seed 0, where a well-behaved mean would
# have it fall ~4x.  If that is the cause, more draws is the WRONG lever --
# the tail is unphysical draws (in_support is a box, the manifold is curved)
# and the fix is the support test.
set -e -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PQR=${PQR:-pqr/g2v3e_s0.npz}
# 2^28 OOMs: `selection_terms` batches the TERMS but not the draw generation,
# so `flow.base_dist.sample` asks for one 10 GiB allocation on a 16 GB card.
# No code change needed to get past it -- R_s is a sample MEAN, so k
# independent 2^26 estimates average to 2^(26 + log2 k) precision AND give a
# real standard error, where a 2-seed spread gives one degree of freedom and
# fooled me once already (the 2^24 pair happens to sit 14% apart).
for n in 24 26; do
  for seed in 1 2 3 4; do
    echo "=== 2^$n seed $seed ==="
    python -u dev/window_scan.py --pop gauss2_v3e \
        --flow flows/centroid_g2v3d.eqx --pqr "$PQR" \
        --log2-draws "$n" --no-scan --seed "$seed" \
        2>&1 | tee "logs/rs_ladder_${n}_s${seed}.log" \
        | grep -E "nominal|top draw carries"
  done
done
echo "RS LADDER DONE"
