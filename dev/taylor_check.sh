#!/bin/zsh
# 2026-10-16: Taylor-peel g-Hessian (bias.log_conv_is_taylor) vs full autodiff, same targets and draws.
# Usage: zsh dev/taylor_check.sh [FLOW] [N] [BB...]   -> logs/desv/taylor/
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv/taylor; FLOW=${1:-flows/desm/centroid_tnc15c_bw32.eqx}; N=${2:-4000}; shift 2 2>/dev/null
BBS=(${@:-65536}); mkdir -p $L pqr/taylor
BF=(--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz
  --template-cut 1200 1.5 --window-size 2.2 3.5 --window-flux 2200 90000 --n-targets $N --flow $FLOW)
t=${FLOW:t:r}
[ -f pqr/taylor/${t}_full.npz ] || BFD_TIMING=1 BFD_TAYLOR=0 python -u bias.py $BF --batch-budget 65536 --save-pqr pqr/taylor/${t}_full.npz > $L/${t}_full.log 2>&1
for bb in $BBS; do
  BFD_TIMING=1 python -u bias.py $BF --batch-budget $bb --save-pqr pqr/taylor/${t}_taylor_bb$bb.npz > $L/${t}_taylor_bb$bb.log 2>&1
done
echo "TAYLOR CHECK DONE $(date)"
