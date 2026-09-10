#!/bin/bash
# Phase-0 target catalogs for the PSF-anisotropy test: an ORIENTATION TRIPLET
# plus its own baseline.  A single-config c2 cannot tell a leak from the
# scatter already there; the signature is that dc ROTATES WITH THE PSF.
#
#   e_psf = ( 0,  0)  -> baseline, differenced away
#   e_psf = (+e,  0)  -> leaks into c1
#   e_psf = ( 0, +e)  -> rotates into c2
#   e_psf = (-e,  0)  -> c1 flips sign
#
# N = 20000, not 200000.  `bias.py` was always run at --n-targets 20000 (the
# v10 baseline used 19994 of a 200k file), so this is the SAME statistical
# power for a tenth of the render -- but it does mean the e = 0 arms are
# re-rendered here rather than reusing `targets_v3_*`: `sample_population(20000,
# rng(1))` is NOT the first 20000 rows of `sample_population(200000, rng(1))`,
# so pairing against the old `pqr/v10_score.npz` would silently compare
# different galaxies.  Render its own baseline, pair within this set.
#
# NO new PRIOR catalog: the latent moments are PSF-independent bit for bit
# (gate G0, dev/psf_anisotropy_probe.py), so `moments_bulgedisc_v3.fits` is the
# correct --train-data for every config.  The PSF's ellipticity reaches the
# estimator only through the TARGET catalog's `cov` and `cov_odd`, which is
# where bias.py reads C_M and Sigma_X from.
#
# All four configs share --n and --seed, so every arm of every config is the
# same galaxies under the same noise field -- that shared seed is the
# antithetic pairing the bias rests on, and it is also what makes the baseline
# difference cancel the population sample noise.
#
# Cores: `make_catalog` is a sequential per-galaxy loop, so the only lever that
# does not touch the RNG stream is processes x threads.  12 jobs x 2 threads
# keeps 24 of 32 busy; sharding one catalog across processes would change its
# draw sequence and break the pairing, so it is not done.
#
# Amplitude 0.2, exaggerated on purpose: at a DES-like 0.05 a real leak hides
# under the 0.0008 rebuild floor.  Scale down after, and the SCALING EXPONENT
# localises the channel for free -- Sigma_X's split is linear in e_psf
# (measured +17.1% at 0.2), C_M's spin-2 split quadratic (+1.68% at 0.2).
#
# Usage:  bash dev/render_psfe.sh [amplitude] [pop] [noise_sigma]
#         default pop=bulgedisc, noise_sigma=0.93 (the original grid, filenames
#         unchanged for that default so it doesn't collide with what's already
#         on disk).  A non-default pop/sigma gets a suffix tag on every
#         filename instead, e.g. gauss2_fwd/1.86 -> ..._gauss2fwd_d186_...
set -e
cd "$(dirname "$0")/.."

S=../bfd_cnf_imsims
E=${1:-0.2}
POP=${2:-bulgedisc}
SIGMA=${3:-0.93}
TAG=$(python -c "print(f'{$E:.2f}'.replace('.','')[1:])")   # 0.2 -> 20
N=20000
SEED=1
if [ "$POP" = "bulgedisc" ] && [ "$SIGMA" = "0.93" ]; then
    PTAG=""
else
    PTAG="_$(echo $POP | tr -d '_')_d$(python -c "print(f'{$SIGMA:.2f}'.replace('.',''))")"
fi

# bulgedisc's render is a CPU pixel loop, so 12-way parallel is free lunch.
# Any other population (gauss2_fwd included) goes through JAX/GPU for the
# moment map: 12 concurrent processes each PREALLOCATING their own ~75% GPU
# slab blows out a 16GB card (measured: CUDA_ERROR_OUT_OF_MEMORY at 20k), but
# going fully serial just leaves the GPU idle between each short job's
# python/JAX startup -- cap each process's slab so a handful fit at once.
GPU_PAR=4
fail=0
if [ "$POP" = "bulgedisc" ]; then
    pids=()
    for cfg in "e00 0.0 0.0" "e1p $E 0.0" "e2p 0.0 $E" "e1m -$E 0.0"; do
        set -- $cfg
        name=$1; e1=$2; e2=$3
        for arm in "g0 0.0" "g1p02 0.02" "g1m02 -0.02"; do
            set -- $arm
            ( cd $S && OMP_NUM_THREADS=2 python -u -m imsims.sim \
                --n $N --seed $SEED --pop $POP --noise-sigma $SIGMA \
                --psf-e1 $e1 --psf-e2 $e2 --g1 $2 --add-noise \
                --out data/targets_v3psf${name}${TAG}${PTAG}_$1_20k.fits ) &
            pids+=($!)
        done
    done
    # Bare `wait` returns 0 regardless of what the jobs did, so `set -e` does
    # NOT catch a failed render -- same class of silent pass-through as a
    # missing `pipefail`.  Wait on each pid and keep its status.
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
else
    jobs=()
    for cfg in "e00 0.0 0.0" "e1p $E 0.0" "e2p 0.0 $E" "e1m -$E 0.0"; do
        set -- $cfg
        name=$1; e1=$2; e2=$3
        for arm in "g0 0.0" "g1p02 0.02" "g1m02 -0.02"; do
            set -- $arm
            jobs+=("cd $S && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=$(python -c "print(0.9/$GPU_PAR)") python -u -m imsims.sim --n $N --seed $SEED --pop $POP --noise-sigma $SIGMA --psf-e1 $e1 --psf-e2 $e2 --g1 $2 --add-noise --out data/targets_v3psf${name}${TAG}${PTAG}_$1_20k.fits")
        done
    done
    printf '%s\n' "${jobs[@]}" | xargs -P $GPU_PAR -I{} bash -c '{}' || fail=1
fi
[ $fail -eq 0 ] || { echo "RENDER FAILED" >&2; exit 1; }
echo "render done (e_psf = $E, tag $TAG, pop $POP, sigma $SIGMA)"
