#!/bin/bash
# g2v4n: g2v4's flow chain retrained on a NOISY, 500k prior/templates catalog
# instead of the noiseless 100k one -- THIS FILE IS THE PROVENANCE RECORD.
#
# Motivation (2026-09-22, overnight autonomous run per user instruction): a
# real BFD template catalog always carries SOME residual image noise (a deep
# coadd is not literally noiseless), so this renders that explicitly at 1/10
# the targets' noise_sigma (0.093 vs 0.93) rather than approximating it with
# an artificial jitter at the moment level. It also happens to be a candidate
# fix for the R_s leader/health-gate pathology chased all last session
# (jac_weight and self_jac_weight both diagnosed as insufficient): noise
# naturally smooths the density in sparse joint-coverage regions, and 5x more
# templates (500k) directly addresses the joint-conditional-sparsity gaps
# `dev/leader_sparsity_chart.py` found underneath both leaders.
#
# Targets are UNCHANGED -- reuses g2v4's existing 22k-per-arm target catalogs
# (fixed isotropic PSF, PSF_E=(0,0)) so the bias estimate at the end is a
# direct like-for-like comparison against g2v4's own targets.
#
# copies_gauss2_fwd_g2v4n.fits stays NOISELESS (it is an analytic shift-grid
# construction from the galaxy parameters via bfd.makeTemplates, never a
# rendered image -- imsims/copies.py's `_chunk` calls `sim._measure` on a
# noiseless `draw(...)`) and MUST be built at the TARGETS' noise_sigma=0.93,
# not the prior's 0.093: Sigma_X marginalises the TARGETS' centroid error, not
# the templates'. Only `--n` grows (100k -> 500k), to match the prior.
#
# Point-source ceiling: image noise on the prior can legitimately push a
# MEASURED point past Mr/Mf >= POINT_SOURCE (same statistical effect as on
# noisy targets). The chart has no hard domain restriction that would break on
# this (slot 1 is a bare ratio; `concentration_phi` floors its internal Newton
# solve and stays smooth outside (0, POINT_SOURCE) by construction) and
# `bulk.py train`'s NLL loss calls `model.log_prob` directly, not the
# `SupportedFlow` wrapper that hard-zeros support -- so training is safe as
# is. `dev/check_provenance.py` was updated to report the overshoot RATE
# (asserted < 5%) instead of demanding zero, when the prior's IMGNOISE header
# says it is a noisy prior.
#
# Usage:  bash dev/render_g2v4n.sh prior    # CPU, ~5x the g2v4 prior render
#         bash dev/render_g2v4n.sh copies   # CPU, ~5x the g2v4 copies build
#         bash dev/render_g2v4n.sh check    # dev/check_provenance.py
set -e -o pipefail
cd "$(dirname "$0")/.."

S=../bfd_cnf_imsims
D=$S/data
POP=gauss2_fwd
TAG=g2v4n
NPRIOR=500000
TARGET_SIGMA=0.93
PRIOR_SIGMA=0.093     # 1/10 of the targets', DERIVED (see header comment)
SEED_PRIOR=0
SIZE=22k              # g2v4's target SIZE tag, reused unchanged
NPROC=${NPROC:-8}

prior() {
    mkdir -p "$D" logs
    ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.sim \
        --n $NPRIOR --seed $SEED_PRIOR --pop $POP --noise-sigma $PRIOR_SIGMA \
        --add-noise --nproc $NPROC \
        --out data/moments_${POP}_${TAG}.fits )
    echo "prior render done"
}

copies() {
    mkdir -p "$D" logs
    ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.copies build \
        --n $NPRIOR --seed $SEED_PRIOR --pop $POP --noise-sigma $TARGET_SIGMA \
        --out data/copies_${POP}_${TAG}.fits )
    echo "copies done"
}

check() {
    # tag=g2v4 (targets, reused unchanged), prior_tag=g2v4n (new prior/copies)
    python -u dev/check_provenance.py g2v4 $TAG $SIZE $POP
}

case "${1:-}" in
    prior)  prior ;;
    copies) copies ;;
    check)  check ;;
    *) echo "usage: $0 {prior|copies|check}"; exit 2 ;;
esac
