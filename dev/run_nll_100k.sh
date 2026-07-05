#!/usr/bin/env bash
# 100k independent PQR for the freshly retrained nll flow (prior_flow_xy_nll.eqx, 830k).
# Backups handled separately. Flags corrected for current integrate_grid.py:
#   augmentation is OFF by default; --augment turns it ON (the old --no-augment is gone).
set -euo pipefail
cd /home/vwetzell/gitrepos/bfd_cnf

S=data/raw2standard_stats_rebuild.npz
COMMON="--independent --n-targets 100000 --flux-min 1500 --flux-max 90000 --stats-file $S --q flows/q_flow_xy.eqx --prior flows/prior_flow_xy_nll.eqx"

echo "##### nll_noaug start $(date) #####"
python -m bfd_cnf.integrate_grid $COMMON           --out data/pqr_indep_nll_noaug.npz
echo "##### nll_aug start $(date) #####"
python -m bfd_cnf.integrate_grid $COMMON --augment --out data/pqr_indep_nll_aug.npz
echo "##### ALL DONE $(date) #####"
