#!/bin/bash
# 2026-10-03 -- training-seed scatter of the cv6 chain.  Full retrain at a fresh seed
# (bulk_base -> bulk -> shear as dev/tail_chain.sh trained cv4 s2, then the cv6 centroid
# stage as dev/cv6_chain.sh), tail gate after every stage, then sn8r (pool) and varobs bias
# runs paired against cv6 s2 on the same targets: B - A = flow-to-flow m1 scatter with
# the galaxy shape noise cancelled.  (cv2-era J model: s0 vs s1 = +0.0029 +/- 0.0002.)
# Usage: SEED=3 bash dev/cv6_seed.sh > logs/cv6/seed3.log 2>&1      GPU strictly serial.
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/cv6; F=flows/cv6; mkdir -p $L $F pqr
M=$D/moments_bulgedisc_g2_bdg2n.fits
C=$D/copies_bulgedisc_g2_bdg2n_sn8.fits
SX=$D/$(python -c "import bias; print(bias.CATALOGS['bulgedisc_g2n_176k_varobs_sn8']['plus'])").fits
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005 --window-jacobian full"
s=${SEED:-3}; r=s$s
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &   # something reloads astrosage
gate() { python -u dev/tail_gate.py --kind $1 $2 --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log; }
die() { echo "$r $1 FAILED $(date)"; exit 1; }

# each stage skipped if its output exists (resume)
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
    --multi-scale --anisotropic --n-aniso 200 --aniso-max 0.10 --copies $C \
    --init $F/shear_$r.eqx --flow $F/centroid_$r.eqx --converge --batch 8192 --seed $s \
    --g-resp-weight 1.0 --g-resp-grid --adapter > $L/centroid_$r.log 2>&1 \
    && gate centroid $F/centroid_$r.eqx; } || { mv -f $F/centroid_$r.eqx $F/centroid_$r.failed 2>/dev/null; die centroid; }
echo "$r trained $(date)"

for spec in "bulgedisc_g2n_176k_sn8r cv6_${r}_sn8r 8" "bulgedisc_g2n_176k_varobs_sn8 cv6_$r 0"; do
    set -- $spec
    t0=$SECONDS; [ -f pqr/$2.npz ] || python -u bias.py --pop $1 --flow $F/centroid_$r.eqx $BF --pool-reuse $3 \
        --save-pqr pqr/$2.npz > $L/bias_$2.log 2>&1 || echo "bias $2 FAILED"
    echo "WALL $((SECONDS - t0)) s" >> $L/bias_$2.log
    echo "== $2"; grep -E "windowed|pool|WALL" $L/bias_$2.log
done
pair() { python -u dev/pair_sn8r_varobs.py "$@" 2>&1 | grep -v "Warning\|r = m"; }
echo "== $r - s2, sn8r (paired)"
pair pqr/cv6_s2_sn8r.npz $L/bias_cv6_s2_sn8r.log sn8r pqr/cv6_${r}_sn8r.npz $L/bias_cv6_${r}_sn8r.log sn8r
echo "== $r - s2, varobs (paired)"
pair pqr/cv6_s2.npz $L/bias_cv6_s2.log varobs_sn8 pqr/cv6_$r.npz $L/bias_cv6_$r.log varobs_sn8
echo "ALL DONE $(date)"
