#!/bin/zsh
# 2026-10-14: (1) is corrected m1 taper-invariant?  core band 2.2-3.2 x 3000-90000 at taper (0.10,150), (0.15,225)
# and full90k at (0.15,150) -- widest the saved 300k PQRs cover; (2) the compact bright corner 3.2-3.5 x 20000-90000
# on the remaining 700k targets (independent of the 300k), pre-filtered to the corner.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv; F=flows/desm
BF=(--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz
  --template-cut 1200 1.5)
rep() { echo "$1: $(grep -E 'pre-filter|windowed, (un)?corrected|top draw|P_s =' $L/$1.log | tr -s ' ' | tr '\n' '|')"; }
for spec in "2.2 3.2 3000 90000 0.10 150" "2.2 3.2 3000 90000 0.15 225" "2.2 3.5 2200 90000 0.15 150"; do
  set -- ${=spec}
  for r in x4r x5r; do
    n=taper_${r}A_s$1-$2_f$3-$4_t$5-$6
    python -u bias.py $BF --prefilter-pad 0.3 --window-taper $5 $6 --flow $F/centroid_tnc15c_$r.eqx --window-size $1 $2 \
        --window-flux $3 $4 --seed 0 --n-targets 300000 --load-pqr pqr/desv_tnc15c_${r}A.npz > $L/$n.log 2>&1 || echo "$n FAILED"
    rep $n
  done
done
for r in x4r x5r; do
  n=corner700k_${r}A
  python -u bias.py $BF --prefilter-pad 0.1 --window-taper 0.05 150 --flow $F/centroid_tnc15c_$r.eqx --window-size 3.2 3.5 \
      --window-flux 20000 90000 --seed 0 --target-offset 300000 --n-targets 700000 \
      --save-pqr pqr/desv_tnc15c_${r}A_corner700k.npz > $L/$n.log 2>&1 || echo "$n FAILED"
  rep $n
done
echo "TAPERCORNER DONE $(date)"
