#!/bin/bash
# 2026-10-04 -- more cv6 training seeds for the flow-scatter measurement.  s3 vs s2 showed
# the targets agree (uncorrected dm ~1e-4) and the difference (-0.0006, 4 sigma) is in the
# SELECTION terms; so each new flow needs only training + its tapered selection terms
# (6.5 min), plus ONE target batch for the first new flow to confirm the target-side claim.
# Training = dev/cv6_seed.sh's stages verbatim (tail gate after each).
# Usage: SEEDS="4 5" bash dev/flow_seeds.sh >> logs/cv6/flow_seeds.log 2>&1   GPU strictly serial.
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/cv6; F=flows/cv6
M=$D/moments_bulgedisc_g2_bdg2n.fits
C=$D/copies_bulgedisc_g2_bdg2n_sn8.fits
SX=$D/$(python -c "import bias; print(bias.CATALOGS['bulgedisc_g2n_176k_varobs_sn8']['plus'])").fits
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --g 0.02 --window-fd 0.005 --window-jacobian full --window-taper 0.05 150"
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
gate() { python -u dev/tail_gate.py --kind $1 $2 --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log; }
for s in ${SEEDS:-4 5}; do
    r=s$s; die() { echo "$r $1 FAILED $(date)"; }
    echo "$r training $(date +%T)"
    [ -f $F/bulk_base_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_base_$r.eqx --converge \
        --batch 16384 --lr 3e-3 --tail-weight 1.0 --seed $s > $L/bulk_base_$r.log 2>&1 \
        && gate bulk $F/bulk_base_$r.eqx; } || { die bulk_base; continue; }
    [ -f $F/bulk_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_$r.eqx --chart-from $F/bulk_base_$r.eqx \
        --jac-weight 5.0 --tail-weight 1.0 --converge --batch 16384 --lr 3e-3 --seed $s > $L/bulk_$r.log 2>&1 \
        && gate bulk $F/bulk_$r.eqx; } || { die bulk; continue; }
    [ -f $F/shear_$r.eqx ] || { python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $F/bulk_$r.eqx \
        --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --seed $s > $L/shear_$r.log 2>&1 \
        && gate shear $F/shear_$r.eqx; } || { die shear; continue; }
    [ -f $F/centroid_$r.eqx ] || { XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --anisotropic --n-aniso 200 --aniso-max 0.10 --copies $C \
        --init $F/shear_$r.eqx --flow $F/centroid_$r.eqx --converge --batch 8192 --seed $s \
        --g-resp-weight 1.0 --g-resp-grid --adapter > $L/centroid_$r.log 2>&1 \
        && gate centroid $F/centroid_$r.eqx; } || { mv -f $F/centroid_$r.eqx $F/centroid_$r.failed 2>/dev/null; die centroid; continue; }
    echo "$r trained $(date +%T)"
    # tapered selection terms, seed 0 (same draws as every other run), as dev/taper_sel.sh
    log=$L/taper_sel_sn8r_flow_$r.log
    grep -q "R_s =" $log 2>/dev/null || python -u bias.py --pop bulgedisc_g2n_176k_sn8r --flow $F/centroid_$r.eqx \
        $BF --n-targets 2000 > $log 2>&1 || { die selterms; continue; }
    echo "$r selection terms $(date +%T)"; grep -E "P_s =|R_s =|top draw" $log
    # target batches on the flows in BATCH_FLOWS (default s4), render seeds TBATCH (default 3).
    # 2026-10-05: s4 showed target and selection sides do NOT separate -- every flow needs its own batch
    if [[ " ${BATCH_FLOWS:-4} " == *" $s "* ]]; then
      for b in ${TBATCH:-3}; do
        t0=$SECONDS; o=cv6_${r}_sn8r_s$b
        [ -f pqr/$o.npz ] || python -u bias.py --pop bulgedisc_g2n_176k_sn8r_s$b --flow $F/centroid_$r.eqx \
            $BF --prefilter-pad 0.2 --prefilter-sample 0 --pool-reuse 8 --save-pqr pqr/$o.npz \
            > $L/bias_$o.log 2>&1 || { die target_batch_s$b; continue; }
        echo "WALL $((SECONDS - t0)) s" >> $L/bias_$o.log
        echo "== $o $(date +%T)"; grep -E "windowed" $L/bias_$o.log
      done
    fi
done
echo "flow_seeds DONE $(date)"
