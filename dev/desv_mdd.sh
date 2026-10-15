#!/bin/bash
# 2026-10-12: centroid stage with the conditional-mean response term (centroid.py --resp-mdd-weight), s4: same init
# (shear_s4), seed and flags as centroid_tnc15c_s4 plus the term -> s4m.  Gates as dev/desv.sh, then the paired
# forward check against s4 (GPU), then CPU: per-template response fit, noise-free population response, weighted MDD.
# Usage: bash dev/desv_mdd.sh > logs/desv/chain_s4m.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm; SM=1.5; r=s4m; LAM=${LAM:-10}
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2d.fits; C=$D/copies_${POP}_bdg2d_tn15c.fits; SX=$D/targets_bdg2_g0_1m_desv.fits
SP=/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/7a140eea-a3cd-4c43-a042-7378af54bad3/scratchpad
die() { echo "$1 FAILED $(date)"; exit 1; }
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
gate() { python -u dev/tail_gate.py --kind $1 $2 --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log | grep -q PASS; }

cf=$F/centroid_tnc15c_$r.eqx; cl=$L/centroid_tnc15c_$r.log
cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; die "centroid_tnc15c_$r $1"; }
[ -f $cf ] || systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
    --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M \
    --init $F/shear_s4.eqx --flow $cf --converge --batch 8192 --seed 4 \
    --g-resp-weight 1.0 --g-resp-grid --adapter --resp-mdd-weight $LAM > $cl 2>&1 || cfail train
grep "resp MDD\|CONVERGED" $cl
gate centroid $cf || cfail "tail gate"
er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || cfail "e-response gate (ratio $er)"
python -u dev/eresp_scan.py $C $M $cf 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log || cfail "Sigma_X scan gate"
tail -1 $L/eresp_scan_c_$r.log
echo "$r trained (e-response $er) $(date)"

python -u $SP/fwd_jitter.py jitter s4 $r > $L/fwd_$r.log 2>&1 || die fwd_$r
grep -E "^==|^core|^sel20k|^large|^faint" $L/fwd_$r.log
echo "GPU done $(date)"
JAX_PLATFORMS=cpu python -u $SP/cent_resp.py 100000 s4 $r 2>&1 | grep -v "Warning\|np.log" > $L/cent_resp_$r.log; cat $L/cent_resp_$r.log
JAX_PLATFORMS=cpu python -u $SP/pop_resp.py 1000000 $r 2>&1 | grep -v "Warning\|np.log" > $L/pop_resp_$r.log
python $SP/pop_an.py s4 $r 2>&1 | grep -v Warning > $L/pop_an_$r.log; cat $L/pop_an_$r.log
MDD_WEIGHT=1 python $SP/mdd_check.py 40000 40 s4 $r 2>&1 | tail -6
echo "ALL DONE $(date)"
