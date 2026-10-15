#!/bin/bash
# 2026-10-12: is the flow-to-flow core-response scatter iterate jitter or separate optima?  On s4 (core R_e -3.1e-3):
#   s4a  s4's trained chain continued 20k steps, cosine LR 1e-4 -> 0 with EMA (centroid.py --resume): annealing
#   s4r  centroid stage retrained from s4's shear checkpoint with another seed (7), same flags as dev/desv.sh
# then the forward check on s4, s4a, s4r, s2 with common draws (paired differences vs s4).
# Usage: bash dev/desv_s4jitter.sh > logs/desv/chain_s4jitter.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; D=$S/data; L=logs/desv; F=flows/desm
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2d.fits; C=$D/copies_${POP}_bdg2d_tn15c.fits; SM=1.5
die() { echo "$1 FAILED $(date)"; exit 1; }
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
CF="--multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M
  --init $F/shear_s4.eqx --batch 8192 --g-resp-weight 1.0 --g-resp-grid --adapter"
run() {   # tag extra...
    local tag=$1; shift
    [ -f $F/centroid_tnc15c_$tag.eqx ] && return 0
    systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax $CF --flow $F/centroid_tnc15c_$tag.eqx \
        "$@" > $L/centroid_tnc15c_$tag.log 2>&1 || die "train $tag"
    echo "$tag: $(awk '/ellipticity response/' $L/centroid_tnc15c_$tag.log | tail -1)  $(date)"
}
run s4a --resume $F/centroid_tnc15c_s4.eqx --steps 20000 --lr 1e-4 --seed 4
run s4r --converge --seed 7
python -u /tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/7a140eea-a3cd-4c43-a042-7378af54bad3/scratchpad/fwd_jitter.py jitter \
    > $L/fwd_jitter.log 2>&1 || die fwd_jitter
grep -A3 "^== R_e" $L/fwd_jitter.log; grep "^core\|^sel20k\|^large\|^faint" $L/fwd_jitter.log
echo "ALL DONE $(date)"
