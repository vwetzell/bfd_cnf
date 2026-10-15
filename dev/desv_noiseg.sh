#!/bin/bash
# 2026-10-09 -- sheared-template-noise fix (user): a noisy template's dm_dg shears its own pixel noise, so
#   the shear stage and the centroid stage's response target were regressed onto E[D_T|T] = E[D(m)|T] - G grad
#   log p_t (Tweedie), worth +0.0006 to +0.0009 in desv corrected m1.  --noise-g adds the flow's own
#   G grad log p_t back to both targets (shear.noise_g_term; G from dev/noise_G.py, pure noise stamps).
# Paired with split k0 (dev/desv_split.sh): same 100k templates, same bulk_k0, same seed, same 300k targets,
#   same sel20k window and 4 selection seeds -- only the shear and centroid stages retrain.
# Usage: bash dev/desv_noiseg.sh [split, default k0] >> logs/desv/chain_noiseg.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; D=$S/data; L=logs/desv; F=flows/desm; mkdir -p $L pqr
POP=bulgedisc_g2; SM=1.5; TPOP=bulgedisc_g2n_1m_desv; SX=$D/targets_bdg2_g0_1m_desv.fits; NT=300000; SEED=2
BF="--pop $TPOP --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates"
die() { echo "$1 FAILED $(date)"; exit 1; }
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &

G=$D/noiseG_${POP}_bdg2d.npz
[ -f $G ] || python -u dev/noise_G.py $D/moments_${POP}_bdg2d.fits $G 200000 2>&1 | grep -v Warning || die noise_G

r=${1:-k0}; t=${r}g; M=$D/moments_${POP}_bdg2d_$r.fits; C=$D/copies_${POP}_bdg2d_tn15c_$r.fits
[ -f $L/flow_tweedie_$r.log ] || python -u dev/flow_tweedie.py $F/shear_$r.eqx $M 2>&1 | grep -A5 -e "--- window, val" | tee $L/flow_tweedie_$r.log
gate() { python -u dev/tail_gate.py --kind $1 $2 --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log | grep -q PASS; }
[ -f $F/shear_$t.eqx ] || { python -u shear.py train --data $M --flow $F/shear_$t.eqx --init $F/bulk_$r.eqx \
    --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --seed $SEED --noise-g $G > $L/shear_$t.log 2>&1 \
    && gate shear $F/shear_$t.eqx; } || die shear_$t
grep -- "--noise-G" $L/shear_$t.log
python -u dev/flow_tweedie.py $F/shear_$t.eqx $M 2>&1 | grep -A5 -e "--- window, val" | tee $L/flow_tweedie_$t.log
cf=$F/centroid_tnc15c_$t.eqx; cl=$L/centroid_tnc15c_$t.log
cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; die "centroid_tnc15c_$t $1"; }
[ -f $cf ] || systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
    --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M \
    --init $F/shear_$t.eqx --flow $cf --converge --batch 8192 --seed $SEED \
    --g-resp-weight 1.0 --g-resp-grid --adapter --noise-g $G > $cl 2>&1 || cfail train
grep -- "--noise-g" $cl
gate centroid $cf || cfail "tail gate"
er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || cfail "e-response gate (ratio $er)"
python -u dev/eresp_scan.py $C $M $cf 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$t.log || cfail "Sigma_X scan gate"
tail -1 $L/eresp_scan_c_$t.log
echo "$t trained (e-response $er) $(date)"

P=pqr/desv_tnc15c_${t}_sub.npz
[ -f $P ] || python -u bias.py $BF --flow $cf --train-data $M --window-size 2.2 3.5 \
    --window-flux 2200 90000 --n-targets $NT --save-pqr $P > $L/bias_${t}_sub.log 2>&1 || die "bias $t subset"
for seed in 0 1 2 3; do
    tag=${t}sub_sel20k_sd$seed
    [ -f $L/bias_$tag.log ] && grep -q "windowed, corrected" $L/bias_$tag.log && continue
    python -u bias.py $BF --flow $cf --train-data $M --window-size 2.2 3.5 --window-flux 2200 20000 \
        --seed $seed --n-targets $NT --load-pqr $P > $L/bias_$tag.log 2>&1 || die "re-solve $tag"
    grep -E "P_s =|windowed, corrected" $L/bias_$tag.log | sed "s/^/  $tag /"
done
echo "ALL DONE $(date)"
