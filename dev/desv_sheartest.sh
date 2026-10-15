#!/bin/bash
# 2026-10-12: does the s4 shear stage converge once the LR is lowered?  The --converge runs stop at 5k-10k steps,
# still at the full LR, because the held-out total rises afterwards.  Here: fixed 60k-step cosine schedule (6e-3 -> 0),
# held-out terms (NLL mean/median/max, dm/dg, d2m/dg2, held-out spin-2 response) logged and a checkpoint saved every 5k.
# Usage: bash dev/desv_sheartest.sh > logs/desv/chain_sheartest.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=../bfd_cnf_imsims/data; M=$D/moments_bulgedisc_g2_bdg2d.fits; F=flows/desm
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
python -u shear.py train --data $M --flow $F/shtest/shear_s4c.eqx --init $F/bulk_s4.eqx --deriv-weight 1e4 \
    --steps 60000 --batch 16384 --lr 6e-3 --seed 4 --check-every 5000 --ckpt $F/shtest/shear_s4c \
    > logs/desv/shear_s4c.log 2>&1 || { echo "FAILED $(date)"; exit 1; }
grep "val terms" logs/desv/shear_s4c.log
echo "ALL DONE $(date)"
