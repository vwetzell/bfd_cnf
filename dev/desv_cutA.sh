#!/bin/zsh
# 2026-10-14: option A -- bias.py --template-cut zeroes target draws below the flow's training cut
# (fixes the x4r Mf~12 single-draw R11 spike).  x4r and x5r: full90k 300k + sel20k/cv6win re-solves.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm
BF=(--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz
  --template-cut 1200 1.5)
for r in x4r x5r; do
    cf=$F/centroid_tnc15c_$r.eqx P=pqr/desv_tnc15c_${r}A.npz
    [ -f $P ] || python -u bias.py $BF --flow $cf --window-size 2.2 3.5 --window-flux 2200 90000 \
        --n-targets 300000 --save-pqr $P > $L/bias_${r}A.log 2>&1 || { echo "bias ${r}A FAILED"; continue; }
    echo "${r}A full90k: $(grep -E 'zeroed|dropping|top draw|traceless|windowed, (un)?corrected' $L/bias_${r}A.log | tr -s ' ' | tr '\n' '|')"
    for spec in "sel20k:2.2 3.5 2200 20000" "cv6win:2.2 3.2 3000 20000"; do
        name=${spec%%:*}; set -- ${=${spec#*:}}
        python -u bias.py $BF --flow $cf --window-size $1 $2 --window-flux $3 $4 --seed 0 --n-targets 300000 \
            --load-pqr $P > $L/bias_${r}A_${name}_sd0.log 2>&1 || echo "re-solve ${r}A $name FAILED"
        echo "  ${r}A $name: $(grep -E 'top draw|windowed, corrected' $L/bias_${r}A_${name}_sd0.log | tr -s ' ' | tr '\n' '|')"
    done
done
echo "A DONE $(date)"
