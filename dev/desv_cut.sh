#!/bin/bash
# 2026-10-13: censored template cut (--template-cut 1200 1.5: measured Mf >= 1200 and Mr/Mf >= 1.5) in every stage.
# bulk/shear: cut templates enter only through log P(unselected) (bulk.log_p_unselected), so the flow stays the
# full-population density; centroid: their copies are dropped.  Whole chain retrained, seeds 4/5/6 (tags x4/x5/x6),
# bulk/shear as dev/desv_last.sh, centroid fixed budget 200k (option 1), plus x4d = seed 4 centroid at 400k on
# shear_x4 (doubling).  Then the CPU per-template fit.  Usage: bash dev/desv_cut.sh > logs/desv/chain_cut.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm; SM=1.5; CUT="--template-cut 1200 1.5"
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2d.fits; C=$D/copies_${POP}_bdg2d_tn15c.fits; SX=$D/targets_bdg2_g0_1m_desv.fits
SP=/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/7a140eea-a3cd-4c43-a042-7378af54bad3/scratchpad
die() { echo "$1 FAILED $(date)"; exit 1; }
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
gate() { python -u dev/tail_gate.py --kind $1 $2 --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log | grep -q PASS; }
upstream() {   # tag seed
    local r=$1 s=$2
    [ -f $F/bulk_base_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_base_$r.eqx --converge $CUT \
        --batch 16384 --lr 3e-3 --tail-weight 1.0 --ema 0.9995 --seed $s > $L/bulk_base_$r.log 2>&1 \
        && gate bulk $F/bulk_base_$r.eqx; } || die bulk_base_$r
    [ -f $F/bulk_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_$r.eqx --chart-from $F/bulk_base_$r.eqx $CUT \
        --jac-weight 5.0 --tail-weight 1.0 --converge --batch 16384 --lr 3e-3 --ema 0.9995 --seed $s > $L/bulk_$r.log 2>&1 \
        && gate bulk $F/bulk_$r.eqx; } || die bulk_$r
    [ -f $F/shear_$r.eqx ] || { python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $F/bulk_$r.eqx $CUT \
        --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --ema 0.9995 --seed $s > $L/shear_$r.log 2>&1 \
        && gate shear $F/shear_$r.eqx; } || die shear_$r
    echo "$r upstream: $(grep -h 'CONVERGED\|NOT CONVERGED' $L/bulk_base_$r.log $L/bulk_$r.log $L/shear_$r.log | tr '\n' ' ')  $(date)"
}
cent() {   # tag seed steps shear-tag
    local r=$1 seed=$2 steps=$3 up=$4
    local cf=$F/centroid_tnc15c_$r.eqx cl=$L/centroid_tnc15c_$r.log
    [ -f $cf ] && return 0
    systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M $CUT \
        --init $F/shear_$up.eqx --flow $cf --steps $steps --batch 8192 --seed $seed \
        --g-resp-weight 1.0 --g-resp-grid --adapter > $cl 2>&1 || die "$r train"
    local g=PASS; gate centroid $cf || g=FAIL
    local er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
    python -u dev/eresp_scan.py $C $M $cf $CUT 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log
    echo "$r (seed $seed, $steps steps, shear_$up): tail gate $g, e-resp $er; $(tail -1 $L/eresp_scan_c_$r.log); last full-loss $(grep -o 'full-loss (all 403 scales) [0-9.]*' $cl | tail -1 | awk '{print $NF}')  $(date)"
}
for s in 4 5 6; do upstream x$s $s; cent x$s $s 200000 x$s; done
cent x4d 4 400000 x4
echo "GPU done $(date)"
JAX_PLATFORMS=cpu python -u $SP/cent_resp.py 100000 x4 x5 x6 x4d 2>&1 | grep -v "Warning\|np.log" > $L/cent_resp_cut.log
grep -E "^x[0-9]|x1.000" $L/cent_resp_cut.log
echo "ALL DONE $(date)"
