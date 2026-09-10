#!/bin/bash
# Full clean rebuild of the bulgedisc chain -- "v3".  THIS FILE IS THE
# PROVENANCE RECORD.  The 2026-09-02 audit found that no log, script or shell
# history anywhere recorded how the `_v2` flows were trained; their arguments
# survived only as prose in HANDOFF.md.  Every artifact below is produced by a
# command in this file and nowhere else.
#
# What changed against the `_v2` chain, and why:
#
#   * noise_sigma = 0.93 EVERYWHERE, including the noiseless prior catalog.
#     Derived, not inherited: it matches this population's median flux S/N
#     (19.2) to the real DES/COSMOS template catalog's.  See the comment on
#     `imsims.sim.NOISE_SIGMA`.  The `_v2` prior was rendered at the old 1.0
#     default while its targets and copies used an explicit 0.9, leaving the
#     prior's `cov`/`cov_odd` 23% wrong -- inert, because `bias.load_cov` reads
#     the TARGET catalog, but there was nothing preventing it from biting.
#
#   * The targets are drawn on seed 1, the prior and copies on seed 0.  In the
#     `_v2` chain everything was seed 0 and prior/targets differed only because
#     `--n` differed, which is an accident, not independence.  The three target
#     arms MUST share a seed with each other -- that is what makes +g and -g the
#     same galaxies under the same noise field, which is the antithetic pairing
#     the bias estimate rests on.
#
#   * bulk trains to 150000 steps, the converged value retrain.sh documents.
#     The `_v2` bulk ran 20000, where the measured m1 is still a function of the
#     step count (retrain.sh's table: -0.0227 at 20k against -0.0600 at 150k).
#
# Everything the previous chain produced was moved to `archive_20260902/` under
# imsims/data, flows, logs, plots and dev.  Nothing here reads it.
#
# Cost: ~35 min rendering (targets run in parallel), ~30 min copies, then the
# GPU chain.  ONE JAX GPU JOB AT A TIME -- the training steps are sequential on
# purpose.
#
# Usage:  bash rebuild_v3.sh render   # catalogs (CPU, parallel)
#         bash rebuild_v3.sh copies   # copy grid (CPU, all cores)
#         bash rebuild_v3.sh train    # bulk -> shear -> centroid (GPU)
#
# ponytail: POP/TAG/N are env overrides so the SAME recipe builds the gauss2
# chain -- one script, one provenance record, no second copy to drift.
#   POP=gauss2 TAG=g2v3 NTARGET=500000 SIZE=500k bash rebuild_v3.sh render
#   SIGMA=1.86 POP=gauss2_fwd TAG=g2v3d NTARGET=500000 SIZE=500k \
#       bash rebuild_v3.sh render      # the 2x-deeper arm, median S/N 9.5
set -e -o pipefail
cd "$(dirname "$0")"

S=../bfd_cnf_imsims
D=$S/data
# 0.93 is the DERIVED depth (median flux S/N 19.2, matching the real DES/COSMOS
# catalog) and is what every `_v3`/`g2v3` artifact was built at -- do not change
# the default.  The override exists to render a DEEPER arm of the same
# population: `cov` scales exactly as SIGMA^2, so S/N goes as 1/SIGMA, and any
# override must carry its own TAG so it cannot overwrite a 0.93 catalog.
SIGMA=${SIGMA:-0.93}
POP=${POP:-bulgedisc}
TAG=${TAG:-v3}
NPRIOR=${NPRIOR:-100000}
NTARGET=${NTARGET:-200000}
SIZE=${SIZE:-200k}
# Workers per render job.  The measure loop is one Python call per galaxy, so a
# job pegs exactly ONE core however many threads JAX has lying around -- four
# jobs on a 32-thread box left 28 threads idle.  `render` runs 4 jobs at once,
# so NPROC=8 is the box.  The catalog is byte-identical at any NPROC
# (tests/test_sim_nproc.py), so this is a speed knob and nothing else.
NPROC=${NPROC:-8}
SEED_PRIOR=0
SEED_TARGET=1

render() {
    mkdir -p "$D"
    # JAX_PLATFORMS=cpu: gauss2 replaces bfd's shear-derivative columns with
    # exact autodiff ones (`sim._overwrite_exact_derivs`), so a render is a JAX
    # process.  Four of them on the 16 GB card is CUDA_ERROR_OUT_OF_MEMORY --
    # measured, three of four arms died silently -- and the GPU is for training
    # anyway.  Harmless for the populations that never call jax.
    local pids=()
    # The prior / "template" catalog: noiseless moments, so the BFD prior is the
    # population itself.  --noise-sigma still matters -- it sets the `cov` and
    # `cov_odd` columns that describe the depth a target would be measured at.
    ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.sim \
        --n $NPRIOR --seed $SEED_PRIOR --pop $POP --noise-sigma $SIGMA \
        --nproc $NPROC --out data/moments_${POP}_${TAG}.fits ) &
    pids+=($!)

    # The three target arms: image noise in the stamp and a real recenter(), on
    # ONE seed so they share galaxies and noise field.
    for arm in "g0 0.0" "g1p02 0.02" "g1m02 -0.02"; do
        set -- $arm
        ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.sim \
            --n $NTARGET --seed $SEED_TARGET --pop $POP --noise-sigma $SIGMA \
            --g1 $2 --add-noise --nproc $NPROC \
            --out data/targets_${TAG}_$1_${SIZE}.fits ) &
        pids+=($!)
    done
    # A bare `wait` reports the LAST job's status only, which is how a failed
    # arm reached "render done" with exit 0.  Wait on each pid.
    for p in "${pids[@]}"; do wait "$p"; done
    echo "render done"
}

copies() {
    # Same galaxies as the prior catalog (same --n, same --seed), which is what
    # lets the copy sum and the flow share a prior galaxy for galaxy.  The audit
    # confirmed that held for _v2 and it is asserted below for _v3.
    # JAX_PLATFORMS=cpu here for the same reason as `render`, plus a sharper
    # one: gauss2's population is drawn by a JAX Newton solve, so a GPU-backend
    # copies run and a CPU-backend prior run disagree at ~3e-14 -- different
    # galaxies, and near the acceptance tolerance a DIFFERENT SET of them.
    # Measured; `check_provenance`'s POPULATION check is what caught it.
    ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.copies \
        --n $NPRIOR --seed $SEED_PRIOR --pop $POP --noise-sigma $SIGMA \
        --out data/copies_${POP}_${TAG}.fits )
    echo "copies done"
    check
}

# Every assertion here is one of the 2026-09-02 audit's five bug shapes.  Run it
# before anything trains on the catalogs, not after.
check() { python -u dev/check_provenance.py "$TAG" "$TAG" "$SIZE" "$POP"; }

train() {
    mkdir -p flows logs
    # 150000: converged.  Below it the measured bias is a function of the
    # schedule rather than of the method -- see retrain.sh's table.
    python -u bulk.py train --data $D/moments_${POP}_${TAG}.fits \
        --flow flows/bulk_${TAG}.eqx --steps 150000 2>&1 | tee logs/bulk_${TAG}.log

    # --deriv-weight 1e4 is not a tuning knob: NLL alone provably cannot
    # identify the spin-0 response, and 1e4 is what makes the derivative term
    # comparable to the ~35 nat NLL.  60000 steps is what the _v2 chain used;
    # the response fit is flat from ~20000 on.
    python -u shear.py train --data $D/moments_${POP}_${TAG}.fits \
        --flow flows/shear_${TAG}.eqx --init flows/bulk_${TAG}.eqx \
        --deriv-weight 1e4 --steps 60000 2>&1 | tee logs/shear_${TAG}.log

    # Warm-starts bulk+shear from the shear checkpoint and grafts the chart, so
    # --init must be THIS chain's shear flow.  Sigma_X comes from the copies
    # catalog's GALAXIES table, which is why the copies must be rendered at the
    # same depth as the targets.
    # Fits the k^4 spin-2 bracket coefficient against the copy catalog's own
    # weighted shifts -- NOT by likelihood, which cannot see it (dev/c_scan.py).
    # Add `--holdout 0.5` to re-run the within-population validation split.
    python -u centroid.py train --copies $D/copies_${POP}_${TAG}.fits \
        --flow flows/centroid_${TAG}.eqx --init flows/shear_${TAG}.eqx \
        2>&1 | tee logs/centroid_${TAG}.log
    echo "train done"
}

case "${1:-}" in
    render) render ;;
    copies) copies ;;
    check)  check ;;
    train)  check && train ;;
    *) echo "usage: $0 {render|copies|check|train}"; exit 2 ;;
esac
