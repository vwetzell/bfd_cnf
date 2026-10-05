#!/bin/bash
# 2026-10-01 -- centroid layer retrained with centroid.py --g-resp-weight (whole-chain shear
# response supervised against the J-lensed copy means), on cv4's tail-penalised bulk+shear.
# Paired against cv4_s2: same bulk/shear, only the centroid layer differs.
# Usage: SEEDS=2 bash dev/gresp_chain.sh > logs/cv5/chain.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/cv5; F=flows/cv5; mkdir -p $L $F pqr
ln -sf ../cv4/bias_cv4_s2.log $L/ 2>/dev/null
C=$D/copies_bulgedisc_g2_bdg2n_sn8.fits
SX=$D/$(python -c "import bias; print(bias.CATALOGS['bulgedisc_g2n_176k_varobs_sn8']['plus'])").fits
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005"
ok=()
for s in ${SEEDS:-2}; do
    r=s$s
    [ -f $F/centroid_$r.eqx ] || XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --fit-K --anisotropic --n-aniso 200 --aniso-max 0.10 --copies $C \
        --init flows/cv4/shear_$r.eqx --flow $F/centroid_$r.eqx --converge --batch 8192 --seed $s \
        --g-resp-weight 1.0 > $L/centroid_$r.log 2>&1 || { echo "$r centroid FAILED"; continue; }
    python -u dev/tail_gate.py --kind centroid $F/centroid_$r.eqx --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log \
        || { echo "$r tail gate FAILED"; continue; }
    python -u dev/centroid_g_split.py $F/centroid_$r.eqx > $L/centroid_g_split_$r.log 2>&1
    echo "$r trained $(date)"
    [ -f pqr/cv5_$r.npz ] || python -u bias.py --pop bulgedisc_g2n_176k_varobs_sn8 --flow $F/centroid_$r.eqx $BF \
        --save-pqr pqr/cv5_$r.npz > $L/bias_cv5_$r.log 2>&1 || { echo "bias $r FAILED"; continue; }
    ok+=(cv5_$r); echo "bias $r done $(date)"
done
LOGS=$L python -u dev/converge_checks.py pair cv4_s2 "${ok[@]}" > $L/pair.log 2>&1 || echo "pair FAILED"
echo "ALL DONE $(date)"
