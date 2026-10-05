#!/bin/bash
# 2026-09-30 -- dev/emax_chain.sh plus bulk.py --tail-weight (the |e| marginal must fall to 0 at E_MAX).
# (models/bijections.py), seeds 0-2, with dev/tail_gate.py after every stage: a seed
# whose flow piles mass at high |e| (cv2 s2's failure) stops there and is not biased.
# Then the J-model varobs bias per surviving seed and the paired m1 table.
# All GPU, strictly serial.  Usage: bash dev/emax_chain.sh > logs/cv3/chain.log 2>&1
# Usage: SEEDS=2 bash dev/tail_chain.sh > logs/cv4/chain.log 2>&1   (pairs against cv3 s0, s1)
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # on-demand: the centroid's 2 GB of jit constants
D=../bfd_cnf_imsims/data; L=logs/cv4; F=flows/cv4; mkdir -p $L $F pqr
ln -sf ../cv3/bias_cv3_s0.log ../cv3/bias_cv3_s1.log $L/ 2>/dev/null
M=$D/moments_bulgedisc_g2_bdg2n.fits
C=$D/copies_bulgedisc_g2_bdg2n_sn8.fits
SX=$D/$(python -c "import bias; print(bias.CATALOGS['bulgedisc_g2n_176k_varobs_sn8']['plus'])").fits
SC=$(python -c "print(3.674 / 0.93)")
NOCB=--xla_gpu_enable_command_buffer=
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005"

gate() {   # kind flow; SOFT_GATE=sN logs that seed's failure but carries on (is a pile AT the ceiling harmless?)
    python -u dev/tail_gate.py --kind $1 $2 --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log \
        || [[ " $SOFT_GATE " == *" $r "* ]]
}
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
# user: unload any ollama model on the GPU.  Something reloads astrosage (OOM'd both cv3
# centroids mid-stage), so a watcher unloads every 30 s for the chain's lifetime.
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &

train() {   # seed; nonzero if a stage or its gate fails
    local s=$1 r=s$1
    [ -f $F/centroid_$r.eqx ] && { echo "$r exists, not retrained"; return; }
    # each stage skipped if its output exists (resume)
    ollama_off; [ -f $F/bulk_base_$r.eqx ] || python -u bulk.py train --data $M --flow $F/bulk_base_$r.eqx --converge \
        --batch 16384 --lr 3e-3 --tail-weight 1.0 --seed $s > $L/bulk_base_$r.log 2>&1 && gate bulk $F/bulk_base_$r.eqx || return 1
    ollama_off; [ -f $F/bulk_$r.eqx ] || python -u bulk.py train --data $M --flow $F/bulk_$r.eqx --chart-from $F/bulk_base_$r.eqx \
        --jac-weight 5.0 --tail-weight 1.0 --converge --batch 16384 --lr 3e-3 --seed $s > $L/bulk_$r.log 2>&1 \
        && gate bulk $F/bulk_$r.eqx || return 1
    ollama_off; [ -f $F/shear_$r.eqx ] || python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $F/bulk_$r.eqx \
        --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --seed $s > $L/shear_$r.log 2>&1 \
        && gate shear $F/shear_$r.eqx || return 1
    ollama_off; XLA_FLAGS=$NOCB python -u centroid.py train-sigmax --multi-scale --fit-K --anisotropic \
        --n-aniso 200 --aniso-max 0.10 --copies $C --init $F/shear_$r.eqx \
        --flow $F/centroid_$r.eqx --converge --batch 8192 --seed $s > $L/centroid_$r.log 2>&1 \
        && gate centroid $F/centroid_$r.eqx || { mv -f $F/centroid_$r.eqx $F/centroid_$r.failed 2>/dev/null; return 1; }
    echo "$r trained $(date)"
    NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n python -u dev/closed_loop.py selection \
        --flow $F/centroid_$r.eqx --h 0.005 0.02 > $L/gate_$r.log 2>&1 || echo "selection gate $r FAILED"
}

ok=()
for s in ${SEEDS:-2}; do
    if train $s; then ok+=(cv4_s$s); else echo "s$s FAILED (training or tail gate) $(date)"; fi
done
for t in "${ok[@]}"; do
    [ -f pqr/$t.npz ] && { echo "bias $t exists"; continue; }
    ollama_off
    python -u bias.py --pop bulgedisc_g2n_176k_varobs_sn8 --flow $F/centroid_${t#cv4_}.eqx $BF \
        --save-pqr pqr/$t.npz > $L/bias_$t.log 2>&1 && echo "bias $t done $(date)" || echo "bias $t FAILED"
done
LOGS=$L python -u dev/converge_checks.py pair cv3_s0 cv3_s1 "${ok[@]}" > $L/pair.log 2>&1 || echo "pair FAILED"
echo "ALL DONE $(date)"
