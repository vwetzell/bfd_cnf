#!/bin/bash
# 2026-10-12: is the centroid-stage seed scatter the Sigma_X cloud draw?  Seeds 5, 6, 7 trained on s4's cloud
# (--aniso-seed 6 = seed 4 + 2): {s4, s4c5, s4c6, s4c7} share one cloud; paired by training seed with {s4r5, s4r6, s4r}
# (own clouds).  Flags as dev/desv.sh.  Then forward check + CPU per-template fit over all seven.
# Usage: bash dev/desv_cloud.sh > logs/desv/chain_cloud.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm; SM=1.5
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2d.fits; C=$D/copies_${POP}_bdg2d_tn15c.fits; SX=$D/targets_bdg2_g0_1m_desv.fits
SP=/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/7a140eea-a3cd-4c43-a042-7378af54bad3/scratchpad
die() { echo "$1 FAILED $(date)"; exit 1; }
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
gate() { python -u dev/tail_gate.py --kind $1 $2 --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log | grep -q PASS; }
run() {   # tag seed extra...
    local r=$1 seed=$2; shift 2
    local cf=$F/centroid_tnc15c_$r.eqx cl=$L/centroid_tnc15c_$r.log
    cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; die "centroid_tnc15c_$r $1"; }
    [ -f $cf ] && return 0
    systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M \
        --init $F/shear_s4.eqx --flow $cf --converge --batch 8192 --seed $seed \
        --g-resp-weight 1.0 --g-resp-grid --adapter "$@" > $cl 2>&1 || cfail train
    gate centroid $cf || cfail "tail gate"
    local er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
    awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || cfail "e-response gate (ratio $er)"
    python -u dev/eresp_scan.py $C $M $cf 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log || cfail "Sigma_X scan gate"
    echo "$r (seed $seed $*): $(grep 'resp MDD \[best\]\|CONVERGED' $cl | tr '\n' ' ') e-resp $er; $(tail -1 $L/eresp_scan_c_$r.log)  $(date)"
}
run s4c5 5 --aniso-seed 6
run s4c6 6 --aniso-seed 6
run s4c7 7 --aniso-seed 6
python -u $SP/fwd_jitter.py jitter s4 s4c5 s4c6 s4c7 s4r5 s4r6 s4r > $L/fwd_cloud.log 2>&1 || die fwd
grep -E "^==|^core|^sel20k|^large|^faint" $L/fwd_cloud.log
echo "GPU done $(date)"
JAX_PLATFORMS=cpu python -u $SP/cent_resp.py 100000 s4c5 s4c6 s4c7 2>&1 | grep -v "Warning\|np.log" > $L/cent_resp_cloud.log
grep -E "^s4|x1.000" $L/cent_resp_cloud.log
echo "ALL DONE $(date)"
