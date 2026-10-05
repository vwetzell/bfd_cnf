#!/bin/bash
# 2026-09-29: resume dev/converge_chain.sh after centroid_s1 OOM'd (ollama held 1.5 GB of GPU).
# JAX allocates on demand (no 75% pool) so the 2 GB of jit constants fit beside ollama; bulk/shear s1 are already trained.
set -e -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; L=logs/cv2; F=flows/cv2; r=s1
C=$D/copies_bulgedisc_g2_bdg2n_sn8.fits
SC=$(python -c "print(3.674 / 0.93)")
XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax --multi-scale --fit-K --anisotropic \
    --n-aniso 200 --aniso-max 0.10 --copies $C --init $F/shear_$r.eqx \
    --flow $F/centroid_$r.eqx --converge --batch 8192 --seed 1 > $L/centroid_$r.log 2>&1
echo "$r trained $(date)"
NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n python -u dev/closed_loop.py selection \
    --flow $F/centroid_$r.eqx --h 0.005 0.02 > $L/gate_$r.log 2>&1 || echo "gate $r FAILED"
for mode in main structure; do
    FLOW=$F/centroid_$r.eqx COPIES=$C NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n \
        python -u dev/centroid_e_response.py $mode > $L/eresp_${mode}_$r.log 2>&1 || echo "eresp $r FAILED"
done
bash dev/converge_chain.sh
