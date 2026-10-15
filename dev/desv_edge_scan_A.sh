#!/bin/zsh
# 2026-10-14: does cv6win's +0.002 track its upper size edge?  Mr/Mf (2.2, X) x Mf (3000, 20000), option-A PQRs.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm
BF=(--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz
  --template-cut 1200 1.5)
for r in x4r x5r; do for X in 3.0 3.1 3.3 3.4 3.5; do
    n=${r}A_s2.2-${X}_f3000-20000
    python -u bias.py $BF --flow $F/centroid_tnc15c_$r.eqx --window-size 2.2 $X --window-flux 3000 20000 --seed 0 \
        --n-targets 300000 --load-pqr pqr/desv_tnc15c_${r}A.npz > $L/strip_$n.log 2>&1 || echo "$n FAILED"
    echo "$n: $(grep -E 'windowed, (un)?corrected|top draw|P_s =' $L/strip_$n.log | tr -s ' ' | tr '\n' '|')"
done; done
echo "EDGE DONE $(date)"
