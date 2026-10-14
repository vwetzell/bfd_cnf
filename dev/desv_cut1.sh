#!/bin/bash
# 2026-10-13: ONE seed (4, tag x4) through the censored-template-cut procedure (--template-cut 1200 1.5 in every
# stage, see dev/desv_cut.sh) all the way to the bias stage: bulk_base -> bulk -> shear (as dev/desv_last.sh) ->
# centroid fixed budget 200k (option 1, last EMA params) -> 300k-slice bias with --template-noise-g + 12 re-solves
# (as dev/desv_s4n.sh).  Gates fatal.  Usage: bash dev/desv_cut1.sh > logs/desv/chain_cut1.log 2>&1
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
cent() {   # tag seed steps shear-tag; gates fatal here (a bias run follows)
    local r=$1 seed=$2 steps=$3 up=$4
    local cf=$F/centroid_tnc15c_$r.eqx cl=$L/centroid_tnc15c_$r.log
    cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; die "centroid_tnc15c_$r $1"; }
    [ -f $cf ] && return 0
    systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M $CUT \
        --init $F/shear_$up.eqx --flow $cf --steps $steps --batch 8192 --seed $seed \
        --g-resp-weight 1.0 --g-resp-grid --adapter > $cl 2>&1 || cfail train
    gate centroid $cf || cfail "tail gate"
    local er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
    awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || cfail "e-response gate (ratio $er)"
    python -u dev/eresp_scan.py $C $M $cf $CUT 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log || cfail "Sigma_X scan gate"
    echo "$r (seed $seed, $steps steps, shear_$up): e-resp $er; $(tail -1 $L/eresp_scan_c_$r.log); last full-loss $(grep -o 'full-loss (all 403 scales) [0-9.]*' $cl | tail -1 | awk '{print $NF}')  $(date)"
}
NT=300000
BF="--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz"
r=x4
upstream $r 4
cent $r 4 200000 $r
cf=$F/centroid_tnc15c_$r.eqx; P=pqr/desv_tnc15c_${r}tng_sub.npz; t0=$SECONDS
[ -f $P ] || { python -u bias.py $BF --flow $cf --window-size 2.2 3.5 --window-flux 2200 90000 \
    --n-targets $NT --save-pqr $P > $L/bias_${r}tng_sub.log 2>&1 \
    && echo "WALL $((SECONDS - t0)) s" >> $L/bias_${r}tng_sub.log; } || die "bias ${r}tng"
grep -E "P_s =|R_s =|top draw|windowed" $L/bias_${r}tng_sub.log
echo "${r}tng integrated $(date)"
for spec in "full90k:2.2 3.5 2200 90000" "sel20k:2.2 3.5 2200 20000" "cv6win:2.2 3.2 3000 20000"; do
    name=${spec%%:*}; set -- ${spec#*:}
    for seed in 0 1 2 3; do
        tag=${r}tngsub_${name}_sd$seed
        [ -f $L/bias_$tag.log ] && grep -q "windowed, corrected" $L/bias_$tag.log && continue
        python -u bias.py $BF --flow $cf --window-size $1 $2 --window-flux $3 $4 --seed $seed --n-targets $NT \
            --load-pqr $P > $L/bias_$tag.log 2>&1 || die "re-solve $tag"
        grep -E "windowed, corrected" $L/bias_$tag.log | sed "s/^/  $tag /"
    done
done
echo "ALL DONE $(date)"
