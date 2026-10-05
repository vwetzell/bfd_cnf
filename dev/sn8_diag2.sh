#!/bin/bash
# 2026-09-27 -- follow-ups to dev/sn8_diag.sh (S/N-8 bulgedisc_g2n, logs/sn8diag/):
#  (1) converged round-PSF m1: bias.py selection-only (--n-targets 64) on the sn8r catalog at
#      --window-fd 0.0025 0.005 0.01, to recombine offline with pqr/bdg2n_Ke_176k_sn8r.npz
#  (3) bright-end e-response overshoot (+18% in the window): the sn8 layer was undertrained
#      (user: 6000 steps is not converged; EMA loss still falling, 0.72 vs shallow 0.49).
#      Same isotropic recipe at 18000 steps (3x), then dev/centroid_e_response.py main/structure
#  (2) varobs breakage: the sn8 layer retrained with --anisotropic (200 random Sigma_X points,
#      |e| <= 0.10 across the 1.4 scale margin; varobs PSF |e| is 0.036 +/- 0.02), 18000 steps
#      as (3) so only the anisotropy differs from it; gate, e-response, then the full varobs
#      bias at --window-fd 0.005 (h = 0.02 is truncation-biased at this depth, diag #1)
# All GPU, strictly serial.  Usage: bash dev/sn8_diag2.sh > logs/sn8_diag2.log 2>&1
# (anisotropic stage rerun as ONLY_AN=1 after the OOM)
# (relaunched 2026-09-27 as WAIT=<pid of the h=0.0025 job> HS="0.005 0.01" after the 6000 -> 18000 fix)
set -e -o pipefail
cd "$(dirname "$0")/.."
D=../bfd_cnf_imsims/data; L=logs/sn8diag; mkdir -p $L
C=$D/copies_bulgedisc_g2_bdg2n_sn8.fits
F=flows/centroid_bdg2n_sn8_Ke.eqx
FL=flows/centroid_bdg2n_sn8_Ke_long.eqx; FA=flows/centroid_bdg2n_sn8_Ke_an.eqx
BF="--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --window-size 2.2 3.2 --window-flux 3000 20000
  --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02"
HS=${HS:-0.0025 0.005 0.01}
if [ -n "${WAIT:-}" ]; then while kill -0 $WAIT 2>/dev/null; do sleep 30; done; fi
SC=$(python -c "print(3.674 / 0.93)")

if [ -z "${ONLY_AN:-}" ]; then
for h in $HS; do
    python -u bias.py --pop bulgedisc_g2n_176k_sn8r --flow $F $BF --n-targets 64 \
        --window-fd $h > $L/sel_sn8r_h$h.log 2>&1
done
echo "(1) done $(date)"

python -u centroid.py train-sigmax --multi-scale --fit-K --copies $C --init flows/shear_bdg2n.eqx \
    --flow $FL --steps 18000 --batch 8192 > logs/centroid_bdg2n_sn8_Ke_long.log 2>&1
for mode in main structure; do
    FLOW=$FL COPIES=$C NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n \
        python -u dev/centroid_e_response.py $mode > $L/centroid_${mode}_sn8_long.log 2>&1 || echo "long $mode FAILED"
done
echo "(3) done $(date)"
fi

# 203 scales capture 2 GB of constants; CUDA graph instantiation OOMed at step ~500 -> no command buffers
XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax --multi-scale --fit-K --anisotropic --n-aniso 200 --aniso-max 0.10 \
    --copies $C --init flows/shear_bdg2n.eqx \
    --flow $FA --steps 18000 --batch 8192 > logs/centroid_bdg2n_sn8_Ke_an.log 2>&1
NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n python -u dev/closed_loop.py selection --flow $FA \
    --h 0.005 0.02 > $L/gate_hscan_sn8_an.log 2>&1 || echo "gate FAILED"
for mode in main structure; do
    FLOW=$FA COPIES=$C NOISE_SCALE=$SC TRAIN_POP=bulgedisc_g2n \
        python -u dev/centroid_e_response.py $mode > $L/centroid_${mode}_sn8_an.log 2>&1 || echo "an $mode FAILED"
done
python -u bias.py --pop bulgedisc_g2n_176k_varobs_sn8 --flow $FA $BF --window-fd 0.005 \
    --save-pqr pqr/bdg2n_Ke_176k_varobs_sn8_an.npz > $L/bias_sn8_an.log 2>&1
echo "ALL DONE $(date)"
