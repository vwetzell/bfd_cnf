#!/bin/zsh
# 2026-10-14: bright strip (Mf 20000-90000) selection terms are under-sampled (top draw 12-25% of R_s11).
# 4 independent selection-draw seeds x 2^24 per window; option-A PQRs.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm
BF=(--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz
  --template-cut 1200 1.5)
for r in x4r x5r; do for sz in "2.2 3.5" "2.2 3.2" "3.2 3.5"; do for sd in 0 1 2 3; do
    set -- ${=sz}; n=${r}A_s$1-$2_f20000-90000
    lg=$L/strip_$n.log; [ $sd = 0 ] || lg=$L/strip_${n}_sd$sd.log
    [ -s $lg ] && grep -q "windowed, corrected" $lg || python -u bias.py $BF --flow $F/centroid_tnc15c_$r.eqx \
        --window-size $1 $2 --window-flux 20000 90000 --seed $sd --n-targets 300000 \
        --load-pqr pqr/desv_tnc15c_${r}A.npz > $lg 2>&1 || echo "$n sd$sd FAILED"
    echo "$n sd$sd: $(grep -E 'windowed, corrected|top draw|P_s =' $lg | sed -E 's/c1 = .*//; s/Q_s.*//' | tr -s ' ' | tr '\n' '|')"
done; done; done
echo "BRIGHT DONE $(date)"
