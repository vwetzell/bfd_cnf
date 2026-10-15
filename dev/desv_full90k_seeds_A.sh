#!/bin/zsh
# 2026-10-14: selection-draw MC error of the headline full90k window -- seeds 1-3 x 2^24 (seed 0 = bias_{r}A.log).
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm
BF=(--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz
  --template-cut 1200 1.5)
for r in x4r x5r; do for sd in 1 2 3; do
    lg=$L/bias_${r}A_full90k_sd$sd.log
    python -u bias.py $BF --flow $F/centroid_tnc15c_$r.eqx --window-size 2.2 3.5 --window-flux 2200 90000 --seed $sd \
        --n-targets 300000 --load-pqr pqr/desv_tnc15c_${r}A.npz > $lg 2>&1 || echo "${r} sd$sd FAILED"
    echo "${r} full90k sd$sd: $(grep -E 'windowed, corrected|top draw|P_s =' $lg | sed -E 's/c1 = .*//; s/Q_s.*//' | tr -s ' ' | tr '\n' '|')"
done; done
echo "SEEDS90K DONE $(date)"
