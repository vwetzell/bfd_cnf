#!/bin/bash
# 2026-10-09 -- template-realization test (user): the desv +0.002 corrected m1 is common to flows s2-s6, which
#   all trained on ONE 500k noisy-template catalog.  Bernstein+16 (sec. 5) give every batch an independent
#   template set; here the catalog is split into 5 disjoint 100k sets (dev/split_templates.py), each with its
#   own copies, and one flow is trained per set (seed 2 for all, as s2; the optimizer still differs through
#   the data).  Scatter across sets = template realization (+ optimizer) at 100k; mean of sets vs the 500k
#   flows = finite-template systematic.  User OK'd the 100k cut of the training set for this test.
# Each flow is integrated on the first 300k catalog rows (pairs with s2-s6's prefix runs) and re-solved in ONE
#   window, Mf 2200-20000, Mr/Mf 2.2-3.5 (sel20k), at 4 selection seeds (user, 2026-10-09).
# Same training commands and gates as dev/desv.sh; --train-data points every stage AND bias.py at the split.
# Usage: bash dev/desv_split.sh > logs/desv/chain_split.log 2>&1   (KS="0 2" limits both loops to those splits)
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; D=$S/data; L=logs/desv; F=flows/desm; mkdir -p $L pqr
POP=bulgedisc_g2; SM=1.5; TPOP=bulgedisc_g2n_1m_desv; SX=$D/targets_bdg2_g0_1m_desv.fits; NT=300000; SEED=2
BF="--pop $TPOP --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates"
die() { echo "$1 FAILED $(date)"; exit 1; }

[ -f $D/moments_${POP}_bdg2d_k4.fits ] && [ -f $D/copies_${POP}_bdg2d_tn15c_k4.fits ] || {
    python -u dev/split_templates.py $D/moments_${POP}_bdg2d.fits $D/copies_${POP}_bdg2d_tn15c.fits 5 || die split; }

ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &

# --- GPU: one flow per template split ---
for k in ${KS-0 1 2 3 4}; do
    r=k$k; M=$D/moments_${POP}_bdg2d_$r.fits; C=$D/copies_${POP}_bdg2d_tn15c_$r.fits
    gate() { python -u dev/tail_gate.py --kind $1 $2 --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log | grep -q PASS; }
    [ -f $F/bulk_base_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_base_$r.eqx --converge \
        --batch 16384 --lr 3e-3 --tail-weight 1.0 --seed $SEED > $L/bulk_base_$r.log 2>&1 \
        && gate bulk $F/bulk_base_$r.eqx; } || die bulk_base_$r
    [ -f $F/bulk_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_$r.eqx --chart-from $F/bulk_base_$r.eqx \
        --jac-weight 5.0 --tail-weight 1.0 --converge --batch 16384 --lr 3e-3 --seed $SEED > $L/bulk_$r.log 2>&1 \
        && gate bulk $F/bulk_$r.eqx; } || die bulk_$r
    [ -f $F/shear_$r.eqx ] || { python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $F/bulk_$r.eqx \
        --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --seed $SEED > $L/shear_$r.log 2>&1 \
        && gate shear $F/shear_$r.eqx; } || die shear_$r
    cf=$F/centroid_tnc15c_$r.eqx; cl=$L/centroid_tnc15c_$r.log
    cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; die "centroid_tnc15c_$r $1"; }
    [ -f $cf ] || systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M \
        --init $F/shear_$r.eqx --flow $cf --converge --batch 8192 --seed $SEED \
        --g-resp-weight 1.0 --g-resp-grid --adapter > $cl 2>&1 || cfail train
    gate centroid $cf || cfail "tail gate"
    er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
    awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || cfail "e-response gate (ratio $er)"
    python -u dev/eresp_scan.py $C $M $cf 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log || cfail "Sigma_X scan gate"
    tail -1 $L/eresp_scan_c_$r.log
    echo "$r trained (e-response $er) $(date)"
done

# --- GPU: prefix integration (first 300k rows, same window/pad as s2-s6), then the sel20k re-solves ---
for k in ${KS-0 1 2 3 4}; do
    r=k$k; M=$D/moments_${POP}_bdg2d_$r.fits; P=pqr/desv_tnc15c_${r}_sub.npz
    [ -f $P ] || python -u bias.py $BF --flow $F/centroid_tnc15c_$r.eqx --train-data $M --window-size 2.2 3.5 \
        --window-flux 2200 90000 --n-targets $NT --save-pqr $P > $L/bias_${r}_sub.log 2>&1 || die "bias $r subset"
    for seed in 0 1 2 3; do
        tag=${r}sub_sel20k_sd$seed
        [ -f $L/bias_$tag.log ] && grep -q "windowed, corrected" $L/bias_$tag.log && continue
        python -u bias.py $BF --flow $F/centroid_tnc15c_$r.eqx --train-data $M --window-size 2.2 3.5 --window-flux 2200 20000 \
            --seed $seed --n-targets $NT --load-pqr $P > $L/bias_$tag.log 2>&1 || die "re-solve $tag"
        grep -E "P_s =|windowed, corrected" $L/bias_$tag.log | sed "s/^/  $tag /"
    done
done
echo "ALL DONE $(date)"
