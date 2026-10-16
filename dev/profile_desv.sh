#!/bin/zsh
# 2026-10-16: per-galaxy cost breakdown of bias.py on desv (x6rA flags): BFD_TIMING buckets for bw32 at two
# budgets, then dev/profile_layers_desv.py per-layer g-Hessian cost for x6r and bw32.  Run after the sweep (serial GPU).
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/desv/profile; F=flows/desm; mkdir -p $L
BF=(--pop bulgedisc_g2n_1m_desv --samples 8192 --alpha 0.5 --chunk 4096 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz
  --template-cut 1200 1.5 --window-size 2.2 3.5 --window-flux 2200 90000 --n-targets 4000)
for bb in ${=BBS:-32768 65536}; do
    BFD_TIMING=1 python -u bias.py $BF --batch-budget $bb --flow $F/centroid_tnc15c_bw32.eqx 2>&1 \
        | while IFS= read -r l; do print -r -- "$(date +%s.%N) $l"; done > $L/timing_bw32_bb$bb.log
    echo "bb $bb: $(grep -E '^[0-9.]+ +draw ' $L/timing_bw32_bb$bb.log | tail -1 | cut -d' ' -f2-)"
done
python -u dev/profile_layers_desv.py $F/centroid_tnc15c_x6r.eqx $F/centroid_tnc15c_bw32.eqx 2>&1 | grep -v Warning > $L/layers.log
cat $L/layers.log
echo "PROFILE DONE $(date)"
