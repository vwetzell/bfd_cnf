#!/bin/zsh
# 2026-10-14: where does the cv6win +0.002 live?  Option-A PQRs re-solved on a size x flux grid.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm
BF=(--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz
  --template-cut 1200 1.5)
for r in x4r x5r; do
  for sz in "2.2 3.2" "3.2 3.5"; do for fl in "2200 3000" "3000 20000" "20000 90000"; do
    set -- ${=sz} ${=fl}; n=${r}A_s$1-$2_f$3-$4
    python -u bias.py $BF --flow $F/centroid_tnc15c_$r.eqx --window-size $1 $2 --window-flux $3 $4 --seed 0 \
        --n-targets 300000 --load-pqr pqr/desv_tnc15c_${r}A.npz > $L/strip_$n.log 2>&1 || echo "$n FAILED"
    echo "$n: $(grep -E 'windowed, (un)?corrected|top draw' $L/strip_$n.log | tr -s ' ' | tr '\n' '|')"
  done; done
done
echo "STRIPS DONE $(date)"
