#!/bin/zsh
# 2026-10-14 overnight (serial GPU).  Shear-stage censored regularizer (shear.py --jac-weight 5, option 1):
#  A. seed 4 (x4r): shear (init bulk_x4) -> centroid 200k -> bias 300k slice + sel20k/cv6win re-solves
#     -> paired censored-share test (dev/../scratchpad censor_share2) on x4r
#  B. seed 5 (x5r): the whole cut procedure, bulk_base -> ... -> bias + 2 re-solves
# Gate failures stop that phase only (reported), never retrained around.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm; SM=1.5; CUT=(--template-cut 1200 1.5)
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2d.fits; C=$D/copies_${POP}_bdg2d_tn15c.fits; SX=$D/targets_bdg2_g0_1m_desv.fits
SP=/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/7a140eea-a3cd-4c43-a042-7378af54bad3/scratchpad
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
gate() { python -u dev/tail_gate.py --kind $1 $2 --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log | grep -q PASS; }
fail() { echo "$1 FAILED $(date)"; return 1; }
BF=(--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz)
bulk_stages() {  # tag seed
    local r=$1 s=$2
    [ -f $F/bulk_base_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_base_$r.eqx --converge $CUT \
        --batch 16384 --lr 3e-3 --tail-weight 1.0 --ema 0.9995 --seed $s > $L/bulk_base_$r.log 2>&1 \
        && gate bulk $F/bulk_base_$r.eqx; } || { fail bulk_base_$r; return 1; }
    [ -f $F/bulk_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_$r.eqx --chart-from $F/bulk_base_$r.eqx $CUT \
        --jac-weight 5.0 --tail-weight 1.0 --converge --batch 16384 --lr 3e-3 --ema 0.9995 --seed $s > $L/bulk_$r.log 2>&1 \
        && gate bulk $F/bulk_$r.eqx; } || { fail bulk_$r; return 1; }
    echo "$r bulk: $(grep -h 'CONVERGED\|kept rows' $L/bulk_base_$r.log $L/bulk_$r.log | tr '\n' ' ') $(date)"
}
shear_stage() {  # tag seed bulk-tag
    local r=$1 s=$2 b=$3
    [ -f $F/shear_$r.eqx ] || { python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $F/bulk_$b.eqx $CUT --jac-weight 5.0 \
        --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --ema 0.9995 --seed $s > $L/shear_$r.log 2>&1 \
        && gate shear $F/shear_$r.eqx; } || { fail shear_$r; return 1; }
    echo "$r shear: $(grep -h 'CONVERGED\|^val nll' $L/shear_$r.log | tr '\n' ' ') $(grep -c 'nll nan' $L/shear_$r.log) nan-prints  $(date)"
}
cent() {   # tag seed
    local r=$1 s=$2 cf=$F/centroid_tnc15c_$1.eqx cl=$L/centroid_tnc15c_$1.log
    cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; fail "centroid_tnc15c_$r $1"; }
    [ -f $cf ] && return 0
    systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M $CUT \
        --init $F/shear_$r.eqx --flow $cf --steps 200000 --batch 8192 --seed $s \
        --g-resp-weight 1.0 --g-resp-grid --adapter > $cl 2>&1 || { cfail train; return 1; }
    gate centroid $cf || { cfail "tail gate"; return 1; }
    local er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
    awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || { cfail "e-response gate (ratio $er)"; return 1; }
    python -u dev/eresp_scan.py $C $M $cf $CUT 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log || { cfail "Sigma_X scan gate"; return 1; }
    echo "$r centroid: e-resp $er; $(tail -1 $L/eresp_scan_c_$r.log); last full-loss $(grep -o 'full-loss (all 403 scales) [0-9.]*' $cl | tail -1 | awk '{print $NF}')  $(date)"
}
biasrun() {  # tag
    local r=$1 cf=$F/centroid_tnc15c_$1.eqx P=pqr/desv_tnc15c_$1tng_sub.npz
    [ -f $P ] || python -u bias.py $BF --flow $cf --window-size 2.2 3.5 --window-flux 2200 90000 \
        --n-targets 300000 --save-pqr $P > $L/bias_$r.tng_sub.log 2>&1 || { fail "bias $r"; return 1; }
    echo "$r bias full90k: $(grep -E 'dropped|top draw|traceless|windowed, (un)?corrected|outlier' $L/bias_$r.tng_sub.log | tr -s ' ' | tr '\n' '|')"
    for spec in "sel20k:2.2 3.5 2200 20000" "cv6win:2.2 3.2 3000 20000"; do
        local name=${spec%%:*}; set -- ${=${spec#*:}}
        python -u bias.py $BF --flow $cf --window-size $1 $2 --window-flux $3 $4 --seed 0 --n-targets 300000 \
            --load-pqr $P > $L/bias_${r}sub_${name}_sd0.log 2>&1 || fail "re-solve $r $name"
        echo "  $r $name: $(grep -E 'top draw|windowed, corrected' $L/bias_${r}sub_${name}_sd0.log | tr -s ' ' | tr '\n' '|')"
    done
}
censor_share() {  # tag
    local r=$1
    local BS=($BF --flow $F/centroid_tnc15c_$r.eqx --window-size 2.2 3.5 --window-flux 2200 90000 --n-targets 40000 --window-draws 1048576)
    python -u bias.py $BS --save-pqr pqr/censor_share_base.npz > $L/censor_share_base_$r.log 2>&1 || { fail "censor share base $r"; return 1; }
    python -u $SP/bias_censor_mask.py $BS --save-pqr pqr/censor_share_mask.npz > $L/censor_share_mask_$r.log 2>&1 || { fail "censor share mask $r"; return 1; }
    echo "$r censor share:"; grep -h "dropping\|^  m1 =" $L/censor_share_base_$r.log $L/censor_share_mask_$r.log
    python $SP/censor_share2.py 2>/dev/null
}
echo "=== A: x4r $(date)"
{ shear_stage x4r 4 x4 && cent x4r 4 && biasrun x4r && censor_share x4r; } || echo "phase A stopped"
echo "=== B: x5r $(date)"
{ bulk_stages x5r 5 && shear_stage x5r 5 x5r && cent x5r 5 && biasrun x5r; } || echo "phase B stopped"
echo "ALL DONE $(date)"
