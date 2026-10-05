#!/bin/bash
# 2026-10-02 -- cv6 = factorised centroid stage: g-blind SigmaXBlockLayer (Sigma_X only,
# closed-form log-det, peeled out of the g-autodiff) + CentroidShearAdapter carrying the
# centroid stage's shear response (first order in g, spin-covariant in (z, g, Sigma_X)).
# The copy-mean response loss supervises the adapter over the WHOLE Sigma_X grid
# (--g-resp-grid).  No --fit-K (net_K unused when g-blind).  Bias runs use the shared
# prior-draw pool where the catalog has one Sigma_X (sn8r, truegal; varobs falls back).
# Truegal2 (part 1 of the old chain) finished: truegal - truegal2 = +0.0076 +/- 0.0035.
# Usage: bash dev/cv6_chain.sh > logs/cv6/chain.log 2>&1      GPU strictly serial.
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/cv6; F=flows/cv6; mkdir -p $L $F pqr
C=$D/copies_bulgedisc_g2_bdg2n_sn8.fits
SX=$D/$(python -c "import bias; print(bias.CATALOGS['bulgedisc_g2n_176k_varobs_sn8']['plus'])").fits
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005 --window-jacobian full
  --pool-reuse 8"
for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done
pair() { python -u dev/truegal_pair.py "$@" 2>&1 | grep -v "Warning\|r = m"; }

r=s2
[ -f $F/centroid_$r.eqx ] || XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
    --multi-scale --anisotropic --n-aniso 200 --aniso-max 0.10 --copies $C \
    --init flows/cv4/shear_$r.eqx --flow $F/centroid_$r.eqx --converge --batch 8192 --seed 2 \
    --g-resp-weight 1.0 --g-resp-grid --adapter > $L/centroid_$r.log 2>&1 || { echo "cv6 centroid FAILED"; exit 1; }
python -u dev/tail_gate.py --kind centroid $F/centroid_$r.eqx --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log \
    || { echo "cv6 tail gate FAILED"; exit 1; }
echo "cv6 trained $(date)"
python -u dev/flow_vs_truegal.py $F/centroid_$r.eqx 2>&1 | grep -v "Warning\|r = m\|e1 = m\|return np.log\|v = {" \
    | sed -n '/NOISELESS/,$p' > $L/flow_vs_truegal_$r.log; cat $L/flow_vs_truegal_$r.log
for spec in "bulgedisc_g2n_176k_sn8r cv6_${r}_sn8r" "bdg2n_sn8r_truegal truegal_cv6$r" "bulgedisc_g2n_176k_varobs_sn8 cv6_$r"; do
    set -- $spec
    t0=$SECONDS; [ -f pqr/$2.npz ] || python -u bias.py --pop $1 --flow $F/centroid_$r.eqx $BF \
        --save-pqr pqr/$2.npz > $L/bias_$2.log 2>&1 || echo "bias $2 FAILED"
    echo "WALL $((SECONDS - t0)) s" >> $L/bias_$2.log
    echo "== $2"; grep -E "windowed|R_s =|pool|WALL" $L/bias_$2.log
done
pair pqr/cv6_${r}_sn8r.npz bulgedisc_g2n_176k_sn8r pqr/truegal_cv6$r.npz bdg2n_sn8r_truegal > $L/pair_sims_vs_truegal.log
cat $L/pair_sims_vs_truegal.log
echo "ALL DONE $(date)"
