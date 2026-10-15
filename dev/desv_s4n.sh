#!/bin/bash
# 2026-10-12: one seed (s4) through the whole chain with the new shear stopping rule (shear.py --stop-on nod2,
# default): shear -> centroid (as dev/desv.sh) -> 300k-slice bias with --template-noise-g (as dev/desv_tng.sh)
# -> re-solves in full90k/sel20k/cv6win -> paired forward check against the old s4.  Bulk stage reused (bulk_s4).
# Usage: bash dev/desv_s4n.sh > logs/desv/chain_s4n.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; D=$S/data; L=logs/desv; F=flows/desm; NT=300000; SM=1.5; r=s4n
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2d.fits; C=$D/copies_${POP}_bdg2d_tn15c.fits; SX=$D/targets_bdg2_g0_1m_desv.fits
SP=/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/7a140eea-a3cd-4c43-a042-7378af54bad3/scratchpad
BF="--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz"
die() { echo "$1 FAILED $(date)"; exit 1; }
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
gate() { python -u dev/tail_gate.py --kind $1 $2 --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log | grep -q PASS; }

[ -f $F/shear_$r.eqx ] || { python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $F/bulk_s4.eqx \
    --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --seed 4 --ckpt $F/shtest/shear_$r > $L/shear_$r.log 2>&1 \
    && gate shear $F/shear_$r.eqx; } || die shear_$r
grep -E "converge: (plateau|CONVERGED)" $L/shear_$r.log
JAX_PLATFORMS=cpu python $SP/ckpt_resp.py $F/shear_s4.eqx $F/shear_$r.eqx 2>&1 | grep -v Warn

cf=$F/centroid_tnc15c_$r.eqx; cl=$L/centroid_tnc15c_$r.log
cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; die "centroid_tnc15c_$r $1"; }
[ -f $cf ] || systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
    --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M \
    --init $F/shear_$r.eqx --flow $cf --converge --batch 8192 --seed 4 \
    --g-resp-weight 1.0 --g-resp-grid --adapter > $cl 2>&1 || cfail train
gate centroid $cf || cfail "tail gate"
er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || cfail "e-response gate (ratio $er)"
python -u dev/eresp_scan.py $C $M $cf 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log || cfail "Sigma_X scan gate"
tail -1 $L/eresp_scan_c_$r.log
echo "$r trained (e-response $er) $(date)"

P=pqr/desv_tnc15c_${r}tng_sub.npz; t0=$SECONDS
[ -f $P ] || { python -u bias.py $BF --flow $cf --window-size 2.2 3.5 --window-flux 2200 90000 \
    --n-targets $NT --save-pqr $P > $L/bias_${r}tng_sub.log 2>&1 \
    && echo "WALL $((SECONDS - t0)) s" >> $L/bias_${r}tng_sub.log; } || die "bias ${r}tng"
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
python -u $SP/fwd_jitter.py jitter s4 $r > $L/fwd_s4n.log 2>&1 || die fwd_s4n
grep -E "^==|^core|^sel20k|^large|^faint" $L/fwd_s4n.log
echo "ALL DONE $(date)"
