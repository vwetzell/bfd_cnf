#!/bin/bash
# 2026-10-09 -- template-noise rung (user): is the desv +0.002-0.003 corrected m1 caused by template noise?
#   The sn8r chain with ~noiseless templates (bdg2n) was clean (5-flow m1 -0.0003 +/- 0.0005); desv differs by
#   (1) noisy templates, (2) DES depth / window, (3) per-target PSF + noise variation.  This isolates (1).
# Paired against desv s3 (flows/desm/centroid_tnc15c_s3.eqx, pqr/desv_tnc15c_s3_sub.npz) on the SAME 300k targets:
#   templates  bdg2n (noise 0.093) instead of bdg2d (0.611 +/- 0.055); same galaxies, same unit noise field
#   bulk/shear flows/cv6/shear_s3.eqx -- dev/cv6_seed.sh, the same bulk/shear commands as dev/desv.sh, seed 3
#   copies     same build as desv's tn15 (noise 2.289, --sigma-range 1.5), --template-noise 0.093 0 so the
#              GALAXIES table IS the bdg2n catalog (asserted), same cull rule (dev/cull_copies.py 0.01)
#   centroid   desv's exact command, seed 3, --train-data bdg2n; same gates (tail, e-response, Sigma_X scan)
#   bias       desv's BF incl. --deconvolve-templates (C_t from bdg2n: 0.17% of C_M), --train-data bdg2n,
#              first 300k rows, re-solved at sel20k / full90k / cv6win x 4 selection seeds like s3sub
# Only template noise differs.  Pre-existing cv6 centroid layers are NOT usable: built at sn8 Sigma_X, 86.6%
#   of desv targets fall outside their trained range (checked 2026-10-09).
# Usage: bash dev/desv_noiseless.sh > logs/desv/chain_noiseless.log 2>&1     CPU render, then GPU, strictly serial.
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; D=$S/data; L=logs/desv; F=flows/desm; mkdir -p $L pqr
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2n.fits; SM=1.5; SEED=3; r=n15c_s3
C0=$D/copies_${POP}_bdg2n_des15.fits; C=$D/copies_${POP}_bdg2n_des15c.fits
TPOP=bulgedisc_g2n_1m_desv; SX=$D/targets_bdg2_g0_1m_desv.fits; NT=300000
INIT=flows/cv6/shear_s$SEED.eqx; cf=$F/centroid_$r.eqx; cl=$L/centroid_$r.log; P=pqr/desv_${r}_sub.npz
BF="--pop $TPOP --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --train-data $M"
die() { echo "$1 FAILED $(date)"; exit 1; }
[ -f $INIT ] || die "missing $INIT"

# --- CPU: copies (desv's tn15 build with bdg2n template noise), row-alignment check, cull, coverage pre-flight ---
[ -f $C0 ] || { ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.copies build --n 500000 --seed 0 \
    --pop $POP --noise-sigma 2.289 --template-noise 0.093 0 --sigma-range $SM --procs 24 \
    --out data/copies_${POP}_bdg2n_des15_tmp.fits && mv data/copies_${POP}_bdg2n_des15_tmp.fits data/copies_${POP}_bdg2n_des15.fits ) \
    2>&1 | grep -v Warning; } || die "copies render"
JAX_PLATFORMS=cpu python - <<EOF || die "GALAXIES != bdg2n templates"
import fitsio, numpy as np
a = fitsio.read("$C0", ext="GALAXIES", columns=["moments"])["moments"]
b = fitsio.read("$M", columns=["moments"])["moments"]
d = np.abs(a.astype(float) - b.astype(float)).max(0) / np.abs(b.astype(float)).std(0)
print("copies GALAXIES vs bdg2n moments: max |diff| / sd per moment", d)
assert len(a) == len(b) and d.max() < 1e-5
EOF
[ -f $C ] || { python -u dev/cull_copies.py $C0 ${C%.fits}_tmp.fits 0.01 > $L/cull_n15.log 2>&1 \
    && mv ${C%.fits}_tmp.fits $C; } || die "copies cull"
tail -3 $L/cull_n15.log
python dev/desv_check.py $SX $C $M $SM | tee $L/preflight_n15.log; grep -q "PASS" $L/preflight_n15.log || die "pre-flight"
echo "copies done $(date)"

# --- GPU: centroid stage (desv's command), gates ---
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; die "centroid_$r $1"; }
[ -f $cf ] || systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
    --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M \
    --init $INIT --flow $cf --converge --batch 8192 --seed $SEED \
    --g-resp-weight 1.0 --g-resp-grid --adapter > $cl 2>&1 || cfail train
python -u dev/tail_gate.py --kind centroid $cf --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log \
    | grep -q PASS || cfail "tail gate"
er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || cfail "e-response gate (ratio $er)"
python -u dev/eresp_scan.py $C $M $cf 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log || cfail "Sigma_X scan gate"
tail -1 $L/eresp_scan_c_$r.log
echo "$r trained (e-response $er) $(date)"

# --- GPU: prefix integration (same window/pad as desv s3sub), then the re-solves ---
[ -f $P ] || python -u bias.py $BF --flow $cf --window-size 2.2 3.5 --window-flux 2200 90000 \
    --n-targets $NT --save-pqr $P > $L/bias_${r}_sub.log 2>&1 || die "bias $r subset"
for spec in "full90k 2.2 3.5 2200 90000" "sel20k 2.2 3.5 2200 20000" "cv6win 2.2 3.2 3000 20000"; do
    set -- $spec
    for seed in 0 1 2 3; do
        tag=${r}sub_$1_sd$seed
        [ -f $L/bias_$tag.log ] && grep -q "windowed, corrected" $L/bias_$tag.log && continue
        python -u bias.py $BF --flow $cf --window-size $2 $3 --window-flux $4 $5 --seed $seed --n-targets $NT \
            --load-pqr $P > $L/bias_$tag.log 2>&1 || die "re-solve $tag"
        grep -E "P_s =|windowed, corrected" $L/bias_$tag.log | sed "s/^/  $tag /"
    done
done
echo "ALL DONE $(date)"
