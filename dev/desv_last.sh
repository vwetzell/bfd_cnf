#!/bin/bash
# 2026-10-13: the whole desv chain retrained with train_loop.fit returning the LAST params (never a best-of pick)
# and a Wilcoxon signed-rank stop test; bulk and shear now keep the same 0.9995 EMA as the centroid stage.
# Seeds 4/5/6 (tags l4/l5/l6), flags otherwise as dev/desv.sh.  Then the forward check (GPU) and the per-template
# fit (CPU) on all three: is the seed scatter still ~1.1e-3?
# Usage: bash dev/desv_last.sh > logs/desv/chain_last.log 2>&1
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
for s in 4 5 6; do
    r=l$s
    [ -f $F/bulk_base_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_base_$r.eqx --converge \
        --batch 16384 --lr 3e-3 --tail-weight 1.0 --ema 0.9995 --seed $s > $L/bulk_base_$r.log 2>&1 \
        && gate bulk $F/bulk_base_$r.eqx; } || die bulk_base_$r
    [ -f $F/bulk_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_$r.eqx --chart-from $F/bulk_base_$r.eqx \
        --jac-weight 5.0 --tail-weight 1.0 --converge --batch 16384 --lr 3e-3 --ema 0.9995 --seed $s > $L/bulk_$r.log 2>&1 \
        && gate bulk $F/bulk_$r.eqx; } || die bulk_$r
    [ -f $F/shear_$r.eqx ] || { python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $F/bulk_$r.eqx \
        --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --ema 0.9995 --seed $s > $L/shear_$r.log 2>&1 \
        && gate shear $F/shear_$r.eqx; } || die shear_$r
    cf=$F/centroid_tnc15c_$r.eqx; cl=$L/centroid_tnc15c_$r.log
    cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; die "centroid_tnc15c_$r $1"; }
    [ -f $cf ] || { systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M \
        --init $F/shear_$r.eqx --flow $cf --converge --batch 8192 --seed $s \
        --g-resp-weight 1.0 --g-resp-grid --adapter > $cl 2>&1 || cfail train
      gate centroid $cf || cfail "tail gate"
      er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
      awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || cfail "e-response gate (ratio $er)"
      python -u dev/eresp_scan.py $C $M $cf 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log || cfail "Sigma_X scan gate"; }
    echo "$r: $(grep -h 'CONVERGED\|NOT CONVERGED' $L/bulk_base_$r.log $L/bulk_$r.log $L/shear_$r.log $cl | tr '\n' ' ')"
    echo "$r e-resp $(awk '/ellipticity response/ {print $NF}' $cl | tail -1); $(tail -1 $L/eresp_scan_c_$r.log)  $(date)"
done
python -u $SP/fwd_jitter.py jitter l4 l5 l6 s4 > $L/fwd_last.log 2>&1 || die fwd
grep -E "^==|^core|^sel20k|^large|^faint" $L/fwd_last.log
echo "GPU done $(date)"
JAX_PLATFORMS=cpu python -u $SP/cent_resp.py 100000 l4 l5 l6 2>&1 | grep -v "Warning\|np.log" > $L/cent_resp_last.log
grep -E "^l|x1.000" $L/cent_resp_last.log
echo "ALL DONE $(date)"
