#!/usr/bin/env bash
# Sweep fixed Sigma_X log_scale (e=0) on the SAME 10k targets, nll flow, to see how m
# depends on the noise scale alone. Key is fixed in the CLI so the selection is identical
# across points. Bracket the target median (11.41) over the train range (10.5-13.0);
# 11.94 = canonical sigma_XY=391.4.
set -euo pipefail
cd /home/vwetzell/gitrepos/bfd_cnf
# Stats come from the flow's sidecar (flows/prior_flow_xy_nll.eqx.stats.npz),
# rebuilt from the current templates_train.fits — no --stats-file override.
COMMON="--independent --n-targets 10000 --flux-min 1500 --flux-max 90000 --q flows/q_flow_xy.eqx --prior flows/prior_flow_xy_nll.eqx --fixed-e1 0 --fixed-e2 0"

for LS in 10.5 11.0 11.41 11.94 12.5 13.0; do
  echo "##### fixedls=$LS start $(date) #####"
  python -m bfd_cnf.integrate_grid $COMMON --fixed-log-scale "$LS" --out "data/pqr_indep_nll_fixedls_${LS}_jun27.npz"
done
echo "##### ALL DONE $(date) #####"
