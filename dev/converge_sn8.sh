#!/bin/bash
# 2026-09-27 -- DEFINITIVE convergence test of every layer of the S/N-8 bulgedisc_g2n chain
# (user: settle it for all layers).  Chains retrained FROM SCRATCH with the production recipe
# (dev/bdg2n.sh + centroid --anisotropic, [[isotropic-layer-unconstrained-in-sigmax-e]]) at
# 1x and 2x steps, 3 seeds each; seed s0 at 1x is the existing chain (flows/*bdg2n*, _an).
#   1x: bulk base 150k -> bulk jac 5.0 150k -> shear 1e4 60k  -> centroid fit_K 3000 + 18000
#   2x: bulk base 300k -> bulk jac 5.0 300k -> shear 1e4 120k -> centroid fit_K 6000 + 36000
# Cosine LR decays to 0 at the last step in every trainer, so a flat loss tail proves nothing;
# convergence = 2x agrees with 1x beyond the seed scatter, on:
#   (a) val NLL (bulk, shear; fixed last-10% split, so paired), shear dm/dg + d2m/dg2 vs bfd;
#   (b) centroid: e-response by flux tercile vs copies, round gate R_s, d2m/dg2 vs the sims;
#   (c) m1 of the 176k sn8 varobs run, same targets and bias --seed, paired across chains.
# Phase 1 (all training + cheap checks, ~5 h) finishes before any bias run (~2.6 h each).
# All GPU, strictly serial.  Usage: bash dev/converge_sn8.sh > logs/cvg/converge_sn8.log 2>&1
set -e -o pipefail
cd "$(dirname "$0")/.."
D=../bfd_cnf_imsims/data; L=logs/cvg; mkdir -p $L flows/cvg
M=$D/moments_bulgedisc_g2_bdg2n.fits
C=$D/copies_bulgedisc_g2_bdg2n_sn8.fits
SC=$(python -c "print(3.674 / 0.93)")
NOCB=--xla_gpu_enable_command_buffer=   # anisotropic centroid OOMs without (centroid only)
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005"
RUNS="s0_2x s1_1x s1_2x s2_1x s2_2x"

cen() { echo flows/cvg/centroid_$1.eqx; }
# the existing seed-0 1x chain
ln -sf ../centroid_bdg2n_sn8_Ke_an.eqx flows/cvg/centroid_s0_1x.eqx
cp logs/centroid_bdg2n_sn8_Ke_an.log $L/centroid_s0_1x.log
cp logs/bulk_bdg2n_base.log $L/bulk_base_s0_1x.log; cp logs/bulk_bdg2n.log $L/bulk_s0_1x.log
cp logs/shear_bdg2n.log $L/shear_s0_1x.log

train() {   # $1 = s<seed>_<k>x
    local s=${1%%_*}; s=${s#s}; local k=${1##*_}; k=${k%x}
    local F=flows/cvg
    python -u bulk.py train --data $M --flow $F/bulk_base_$1.eqx --steps $((150000 * k)) --seed $s \
        > $L/bulk_base_$1.log 2>&1
    python -u bulk.py train --data $M --flow $F/bulk_$1.eqx --chart-from $F/bulk_base_$1.eqx \
        --jac-weight 5.0 --steps $((150000 * k)) --seed $s > $L/bulk_$1.log 2>&1
    python -u shear.py train --data $M --flow $F/shear_$1.eqx --init $F/bulk_$1.eqx \
        --deriv-weight 1e4 --steps $((60000 * k)) --seed $s > $L/shear_$1.log 2>&1
    XLA_FLAGS=$NOCB python -u centroid.py train-sigmax --multi-scale --fit-K --fit-K-steps $((3000 * k)) \
        --anisotropic --n-aniso 200 --aniso-max 0.10 --copies $C --init $F/shear_$1.eqx \
        --flow $(cen $1) --steps $((18000 * k)) --batch 8192 --seed $s > $L/centroid_$1.log 2>&1
}

checks() {
    NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n python -u dev/closed_loop.py selection --flow $(cen $1) \
        --h 0.005 0.02 > $L/gate_$1.log 2>&1 || echo "gate $1 FAILED"
    for mode in main structure; do
        FLOW=$(cen $1) COPIES=$C NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n \
            python -u dev/centroid_e_response.py $mode > $L/eresp_${mode}_$1.log 2>&1 || echo "eresp $1 $mode FAILED"
    done
}

for r in $RUNS; do train $r; echo "trained $r $(date)"; done
for r in s0_1x $RUNS; do checks $r; done
NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n python -u dev/converge_checks.py d2m \
    $(for r in s0_1x $RUNS; do cen $r; done) > $L/d2m.log 2>&1 || echo "d2m FAILED"
echo "PHASE 1 DONE $(date)"

for r in $RUNS; do
    python -u bias.py --pop bulgedisc_g2n_176k_varobs_sn8 --flow $(cen $r) $BF \
        --save-pqr pqr/cv_$r.npz > $L/bias_cv_$r.log 2>&1
    echo "bias $r done $(date)"
done
python -u dev/converge_checks.py pair cv_s0_1x cv_s1_1x cv_s2_1x cv_s0_2x cv_s1_2x cv_s2_2x > $L/pair.log 2>&1
echo "ALL DONE $(date)"
