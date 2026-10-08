#!/bin/bash
# 2026-10-06 -- DES-matched s2 with a centroid stage trained on NOISY-template copies.
# The first chain (dev/desmatch.sh) trained bulk/shear on noisy bdg2d templates but the
# centroid stage on copies of NOISELESS galaxy images -- unavailable on real data (user rule,
# memory noisy-templates-no-noiseless-info).  Copies here come from the bdg2d template images
# themselves (imsims.copies --template-noise 0.611 0.055: GALAXIES == bdg2d rows exactly);
# Sigma_X / C_M stay at target depth (2.289).  bulk/shear reused unchanged (noisy-only), so
# the bias run pairs per target with pqr/desm_s2_des.npz; same BF flags, same pre-filter.
# Usage: bash dev/desm_tnc.sh > logs/desm/tnc_chain.log 2>&1     CPU render, then GPU, serial.
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; D=$S/data; L=logs/desm; F=flows/desm
M=$D/moments_bulgedisc_g2_bdg2d.fits
C=$D/copies_bulgedisc_g2_bdg2d_tn.fits
SX=$D/targets_bdg2_g0_176k_des.fits
r=s2; s=2
die() { echo "$1 FAILED $(date)"; exit 1; }

if [ ! -f $C ]; then
    ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.copies build --n 500000 --seed 0 \
        --pop bulgedisc_g2 --noise-sigma 2.289 --template-noise 0.611 0.055 --procs 24 \
        --out data/copies_bulgedisc_g2_bdg2d_tn_tmp.fits \
        && mv data/copies_bulgedisc_g2_bdg2d_tn_tmp.fits data/copies_bulgedisc_g2_bdg2d_tn.fits ) 2>&1 \
        | grep -v Warning || die "copies render"
fi
echo "copies done $(date)"

ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
FC=$F/centroid_tnc_$r.eqx
[ -f $FC ] || { XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
    --multi-scale --anisotropic --n-aniso 200 --aniso-max 0.10 --copies $C --train-data $M \
    --init $F/shear_$r.eqx --flow $FC --converge --batch 8192 --seed $s \
    --g-resp-weight 1.0 --g-resp-grid --adapter > $L/centroid_tnc_$r.log 2>&1 \
    && python -u dev/tail_gate.py --kind centroid $FC --data $M --sx-from $SX 2>&1 | grep "tail gate" \
       | tee -a $L/tail_gate.log | grep -q PASS; } || { mv -f $FC $FC.failed 2>/dev/null; die centroid; }
er=$(awk '/ellipticity response/ {print $NF}' $L/centroid_tnc_$r.log | tail -1)
echo "centroid e-response ratio $er"
awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || die "centroid e-response gate (ratio $er)"

t0=$SECONDS
P=pqr/desm_tnc_${r}_des.npz
[ -f $P ] || python -u bias.py --pop bulgedisc_g2n_176k_des --flow $FC \
  --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm \
  --support --floor-eps 0 --window-size 2.2 3.5 --window-flux 2200 90000 \
  --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005 --window-jacobian full \
  --window-taper 0.05 150 --save-pqr $P > $L/bias_desm_tnc_${r}_des.log 2>&1 || die bias
echo "WALL $((SECONDS - t0)) s" >> $L/bias_desm_tnc_${r}_des.log
grep -E "windowed|P_s =|Q_s =|R_s =|top draw|WALL" $L/bias_desm_tnc_${r}_des.log
for spec in "2.2 3.5|2200 20000|tnc_sel20k" "2.2 3.2|3000 20000|tnc_cv6win" \
            "2.2 3.2|2200 20000|tnc_s32_f2200" "2.2 3.5|3000 20000|tnc_s35_f3000"; do
    IFS='|' read -r sz fx tg <<< "$spec"; echo "== $tg  size $sz  flux $fx"
    SIZE="$sz" FLUX="$fx" TAG="$tg" PQR=$P FLOW=$FC bash dev/desm_sel20k.sh || die "re-solve $tg"
done
echo "ALL DONE $(date)"
