#!/bin/zsh
# 2026-10-15 (user: per-component net-size ladder, wave 1 = 1/2 width).  Each rung shrinks ONE stage and reuses
# the x6r stages upstream of it; everything else is the x6r chain / x6rA bias recipe verbatim (seed 6).
#   c2: centroid --sigmax-width 16 --adapter-width 16, on shear_x6r
#   sw64 (was s2): shear --shear-width 64, on bulk_x6r; centroid at x6r size on top
#   bw32 (was b2): bulk --bulk-width 32 with bulk_base_x6r's chart (as x6r), then shear + centroid at x6r size
# Sizes ride in each flow's <flow>.json sidecar (bulk.net_kw).  Per rung: gates, full90k bias on 300k (no zero
# arm), selection seeds 1-3, sel20k/cv6win, then a 10k-target timing run (budget 32768); x6r timed the same way
# first as the same-session baseline.  Gate failures stop that rung only (reported), never retrained around.
# Usage: zsh dev/netsize.sh > logs/desv/netsize.log 2>&1
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; D=$S/data; L=logs/desv; F=flows/desm; SM=1.5; CUT=(--template-cut 1200 1.5); T=$L/ns_time
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2d.fits; C=$D/copies_${POP}_bdg2d_tn15c.fits; SX=$D/targets_bdg2_g0_1m_desv.fits
mkdir -p $T
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
gate() { python -u dev/tail_gate.py --kind $1 $2 --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log | grep -q PASS; }
fail() { echo "$1 FAILED $(date)"; return 1; }
BF=(--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz $CUT)
rep() { grep -E 'dropp|outlier|top draw|traceless|windowed, (un)?corrected' $1 | tr -s ' ' | tr '\n' '|'; }

bulk_stage() {   # tag seed size-flags...
    local r=$1 s=$2; shift 2
    [ -f $F/bulk_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_$r.eqx --chart-from $F/bulk_base_x6r.eqx $CUT \
        --jac-weight 5.0 --tail-weight 1.0 --converge --batch 16384 --lr 3e-3 --ema 0.9995 --seed $s "$@" > $L/bulk_$r.log 2>&1 \
        && gate bulk $F/bulk_$r.eqx; } || { fail bulk_$r; return 1; }
    echo "$r bulk: $(grep -h 'CONVERGED\|kept rows\|net sizes' $L/bulk_$r.log | tr '\n' ' ') $(date)"
}
shear_stage() {  # tag seed init size-flags...
    local r=$1 s=$2 init=$3; shift 3
    [ -f $F/shear_$r.eqx ] || { python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $init $CUT --jac-weight 5.0 \
        --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --ema 0.9995 --seed $s "$@" > $L/shear_$r.log 2>&1 \
        && gate shear $F/shear_$r.eqx; } || { fail shear_$r; return 1; }
    echo "$r shear: $(grep -h 'CONVERGED\|^val nll\|net sizes' $L/shear_$r.log | tr '\n' ' ') $(date)"
}
cent() {   # tag seed init size-flags...
    local r=$1 s=$2 init=$3 cf=$F/centroid_tnc15c_$1.eqx cl=$L/centroid_tnc15c_$1.log; shift 3
    cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; mv -f $cf.json ${cf%.eqx}.failed.json 2>/dev/null; fail "centroid_tnc15c_$r $1"; }
    [ -f $cf ] && return 0
    systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M $CUT \
        --init $init --flow $cf --steps 200000 --batch 8192 --seed $s \
        --g-resp-weight 1.0 --g-resp-grid --adapter "$@" > $cl 2>&1 || { cfail train; return 1; }
    gate centroid $cf || { cfail "tail gate"; return 1; }
    local er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
    awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || { cfail "e-response gate (ratio $er)"; return 1; }
    python -u dev/eresp_scan.py $C $M $cf $CUT 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log || { cfail "Sigma_X scan gate"; return 1; }
    echo "$r centroid: e-resp $er; $(tail -1 $L/eresp_scan_c_$r.log); last full-loss $(grep -o 'full-loss (all 403 scales) [0-9.]*' $cl | tail -1 | awk '{print $NF}'); $(grep -h 'net sizes' $cl) $(date)"
}
timing() {   # tag: 10k desv targets, x6rA flags, timestamped for dev/batch_sweep_summary.py
    local r=$1 out=$T/${1}_bb_32768.log
    [ -f $out ] && return 0
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $T/${r}_mem.txt &
    local smi=$!
    python -u bias.py $BF --flow $F/centroid_tnc15c_$r.eqx --window-size 2.2 3.5 --window-flux 2200 90000 \
        --n-targets 10000 2>&1 | while IFS= read -r l; do print -r -- "$(date +%s.%N) $l"; done > $out
    kill $smi 2>/dev/null; wait $smi 2>/dev/null
    echo "$r timing: $(python dev/batch_sweep_summary.py $out | sed 's/.*: bb/bb/')  peak mem $(sort -n $T/${r}_mem.txt | tail -1) MiB"
}
biasrun() {  # tag
    local r=$1 cf=$F/centroid_tnc15c_$1.eqx P=pqr/desv_tnc15c_$1A.npz
    [ -f $T/${r}_bb_32768.log ] && return 0   # rung already finished (timing is its last step)
    [ -f $P ] || python -u bias.py $BF --flow $cf --window-size 2.2 3.5 --window-flux 2200 90000 \
        --n-targets 300000 --save-pqr $P > $L/bias_${r}A.log 2>&1 || { fail "bias $r"; return 1; }
    echo "$r bias full90k sd0: $(rep $L/bias_${r}A.log)"
    for sd in 1 2 3; do
        python -u bias.py $BF --flow $cf --window-size 2.2 3.5 --window-flux 2200 90000 --seed $sd --n-targets 300000 \
            --load-pqr $P > $L/bias_${r}A_full90k_sd$sd.log 2>&1 || fail "re-solve $r sd$sd"
        echo "  $r full90k sd$sd: $(grep -E 'top draw|windowed, corrected' $L/bias_${r}A_full90k_sd$sd.log | tr -s ' ' | tr '\n' '|')"
    done
    for spec in "sel20k:2.2 3.5 2200 20000" "cv6win:2.2 3.2 3000 20000"; do
        local name=${spec%%:*}; set -- ${=${spec#*:}}
        python -u bias.py $BF --flow $cf --window-size $1 $2 --window-flux $3 $4 --seed 0 --n-targets 300000 \
            --load-pqr $P > $L/bias_${r}A_${name}_sd0.log 2>&1 || fail "re-solve $r $name"
        echo "  $r $name: $(grep -E 'top draw|windowed, corrected' $L/bias_${r}A_${name}_sd0.log | tr -s ' ' | tr '\n' '|')"
    done
    timing $r
}

echo "=== baseline timing $(date)"; timing x6r
echo "=== c2 $(date)"
{ cent c2 6 $F/shear_x6r.eqx --sigmax-width 16 --adapter-width 16 && biasrun c2; } || echo "c2 stopped"
# Tags must not collide with old seed-N flows (s2 = seed 2 from 2026-10-05 was silently reused in the first launch).
echo "=== sw64 $(date)"
{ shear_stage sw64 6 $F/bulk_x6r.eqx --shear-width 64 && cent sw64 6 $F/shear_sw64.eqx && biasrun sw64; } || echo "sw64 stopped"
echo "=== bw32 $(date)"
{ bulk_stage bw32 6 --bulk-width 32 && shear_stage bw32 6 $F/bulk_bw32.eqx && cent bw32 6 $F/shear_bw32.eqx && biasrun bw32; } || echo "bw32 stopped"
echo "NETSIZE WAVE1 DONE $(date)"
