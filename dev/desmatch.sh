#!/bin/bash
# 2026-10-05 -- DES-Y6-matched depth, bulgedisc_g2 population kept (user).  sigma_f is 99.6 per
# unit pixel noise and the population's flux is in DES Mf units, so absolute sigma_f = S/N match:
#   templates noise 0.611 +/- 0.055 -> sigma_f 60.9 (DES summary_templates_new median, +/- 9%)
#   targets   noise 2.289 +/- 0.28  -> sigma_f 228  (DES Y6 grid median, +/- 12%; DES p99 tail not modelled)
#   window    Mf 2200-90000 (DES's effective lower edge), Mr/Mf 2.2-3.5
# 60k test renders vs DES (dev/des_match.py): target sigma_f within 2.2%, S/N p1-p50 7-9% low;
# template sigma_f within 2.1%, template S/N in the target window within a few % to p84.
# Templates with J <= 0 (4 rows, faint, noise-made) are dropped at load (bulk.physical).
# --prefilter-pad 0.3: the taper's 4 tau = 600 must fit below the 2200 flux edge (0.2 left 440).
# Template noise is IGNORED in the likelihood (user): flows learn prior (x) C_tmpl, bias uses full C_M.
# Chain = dev/cv6_seed.sh recipe, seed 2, then the bias run (per-target C_M, no pool).
# Usage: bash dev/desmatch.sh > logs/desm/chain.log 2>&1      CPU renders, then GPU, strictly serial.
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; D=$S/data; L=logs/desm; F=flows/desm; mkdir -p $L $F pqr
POP=bulgedisc_g2; TAG=bdg2d; TN=2.289; TSD=0.28
M=$D/moments_${POP}_$TAG.fits
C=$D/copies_${POP}_$TAG.fits
SX=$D/targets_bdg2_g0_176k_des.fits
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.5 --window-flux 2200 90000
  --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005 --window-jacobian full
  --window-taper 0.05 150"
s=2; r=s$s
die() { echo "$1 FAILED $(date)"; exit 1; }
render() {   # out args...: skip if present, write via _tmp so a killed render never looks done
    local out=$1; shift
    [ -f $D/$out ] && return 0
    ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.$* --out data/${out%.fits}_tmp.fits \
        && mv data/${out%.fits}_tmp.fits data/$out ) 2>&1 | grep -v Warning
}

# --- CPU renders ---
render moments_${POP}_$TAG.fits sim --n 500000 --seed 0 --pop $POP --noise-sigma 0.611 \
    --obs-sd 0 0 0 0.055 --add-noise --nproc 24 || die "prior render"
render copies_${POP}_$TAG.fits copies build --n 500000 --seed 0 --pop $POP --noise-sigma $TN --procs 24 \
    || die "copies render"
for arm in "g1p02 0.02" "g1m02 -0.02" "g0 0.0"; do
    set -- $arm
    render targets_bdg2_${1}_176k_des.fits sim --n 520000 --seed 2 --pop $POP --noise-sigma $TN \
        --obs-sd 0 0 0 $TSD --add-noise --g1 $2 --nproc 24 || die "target render $1"
done
echo "renders done $(date)"
python dev/des_match.py $D/targets_bdg2_g0_176k_des.fits $M > $L/des_match.log 2>&1; cat $L/des_match.log

# --- GPU: chain (dev/cv6_seed.sh), tail gate after every stage ---
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
gate() { python -u dev/tail_gate.py --kind $1 $2 --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log; }
[ -f $F/bulk_base_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_base_$r.eqx --converge \
    --batch 16384 --lr 3e-3 --tail-weight 1.0 --seed $s > $L/bulk_base_$r.log 2>&1 \
    && gate bulk $F/bulk_base_$r.eqx; } || die bulk_base
[ -f $F/bulk_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_$r.eqx --chart-from $F/bulk_base_$r.eqx \
    --jac-weight 5.0 --tail-weight 1.0 --converge --batch 16384 --lr 3e-3 --seed $s > $L/bulk_$r.log 2>&1 \
    && gate bulk $F/bulk_$r.eqx; } || die bulk
[ -f $F/shear_$r.eqx ] || { python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $F/bulk_$r.eqx \
    --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --seed $s > $L/shear_$r.log 2>&1 \
    && gate shear $F/shear_$r.eqx; } || die shear
[ -f $F/centroid_$r.eqx ] || { XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
    --multi-scale --anisotropic --n-aniso 200 --aniso-max 0.10 --copies $C --train-data $M \
    --init $F/shear_$r.eqx --flow $F/centroid_$r.eqx --converge --batch 8192 --seed $s \
    --g-resp-weight 1.0 --g-resp-grid --adapter > $L/centroid_$r.log 2>&1 \
    && gate centroid $F/centroid_$r.eqx; } || { mv -f $F/centroid_$r.eqx $F/centroid_$r.failed 2>/dev/null; die centroid; }
# centroid e-response gate (cv6 layers 0.999-1.005; the stale-whitening run gave 0.82)
er=$(awk '/ellipticity response/ {print $NF}' $L/centroid_$r.log | tail -1)
awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || die "centroid e-response gate (ratio $er)"
echo "$r trained $(date)"

# --- GPU: bias (per-target C_M -> no prior-draw pool) ---
t0=$SECONDS
[ -f pqr/desm_${r}_des.npz ] || python -u bias.py --pop bulgedisc_g2n_176k_des --flow $F/centroid_$r.eqx $BF \
    --save-pqr pqr/desm_${r}_des.npz > $L/bias_desm_${r}_des.log 2>&1 || die bias
echo "WALL $((SECONDS - t0)) s" >> $L/bias_desm_${r}_des.log
grep -E "windowed|P_s =|Q_s =|R_s =|top draw|WALL" $L/bias_desm_${r}_des.log
echo "ALL DONE $(date)"
