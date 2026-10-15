#!/bin/bash
# 2026-10-12: does the centroid fit still drift at the stop?  Seeds 4 and 5 as s4/s4r5 (own clouds), EMA saved at
# every check, then 40k steps past the stop at the final LR with a Polyak average (FLOW_ck*/FLOW_avg*).
# Usage: bash dev/desv_ckpt.sh > logs/desv/chain_ckpt.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm; SM=1.5
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2d.fits; C=$D/copies_${POP}_bdg2d_tn15c.fits
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
run() {   # tag seed extra...
    local r=$1 seed=$2; shift 2
    local cf=$F/centroid_tnc15c_$r.eqx cl=$L/centroid_tnc15c_$r.log
    [ -f $cf ] && return 0
    systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M \
        --init $F/shear_s4.eqx --flow $cf --converge --batch 8192 --seed $seed \
        --g-resp-weight 1.0 --g-resp-grid --adapter --save-checks "$@" > $cl 2>&1 || { echo "$r train FAILED $(date)"; exit 1; }
    echo "$r: $(grep -h 'CONVERGED' $cl)  $(date)"
}
run s4k4 4 --tail-steps 40000
run s4k5 5 --tail-steps 40000
echo "GPU done $(date)"
