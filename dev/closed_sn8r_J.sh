#!/bin/bash
# Closed loop at sn8r (cv2 s0 flow): does the consistent-J model recover m1 = 0 on
# targets drawn from its own density, and the plain model on plain targets? (2026-09-29)
set -e
F=flows/cv2/centroid_s0.eqx
export REAL_POP=bulgedisc_g2n_176k_sn8r TRAIN_POP=bulgedisc_g2n
python -u dev/closed_loop.py catalogs --flow $F --n 520000 --pop bdg2n_sn8r_closedJ --jacobian > logs/closed_sn8r_J_cat.log 2>&1
python -u dev/closed_loop.py catalogs --flow $F --n 520000 --pop bdg2n_sn8r_closed > logs/closed_sn8r_cat.log 2>&1
BF="--flow $F --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm --support --floor-eps 0
    --window-size 2.2 3.2 --window-flux 3000 20000 --prefilter-pad 0.2 --prefilter-sample 0 --g 0.02 --window-fd 0.005"
python -u bias.py --pop bdg2n_sn8r_closedJ $BF --window-jacobian full --save-pqr pqr/closedJ_sn8r_J.npz > logs/closed_sn8r_J_full.log 2>&1
python -u bias.py --pop bdg2n_sn8r_closed $BF --window-jacobian none --save-pqr pqr/closed_sn8r_none.npz > logs/closed_sn8r_none.log 2>&1
