#!/usr/bin/env bash
# 100k independent PQR: canonical vs nll flow, augmentation on vs off.
# Sequential (parallel GPU runs OOM). Correct rebuilt standardizer.
set -e
cd /home/vwetzell/gitrepos/bfd_cnf
S=data/raw2standard_stats_rebuild.npz
COMMON="--independent --n-targets 100000 --flux-min 1500 --flux-max 90000 --stats-file $S"

run () {  # $1=tag $2=prior $3=extra-flags $4=out
  echo "##### BEGIN $1 #####"
  python -m bfd_cnf.integrate_grid $COMMON --prior "$2" --q flows/q_flow_xy.eqx $3 --out "$4" 2>&1 | grep -vE "cuda_timer|Delay kernel|Nsight"
  echo "##### END $1 (out=$4) #####"
}

run canonical_aug   flows/prior_flow_xy.eqx     ""            data/pqr_indep_canonical_aug.npz
run canonical_noaug flows/prior_flow_xy.eqx     "--no-augment" data/pqr_indep_canonical_noaug.npz
run nll_aug         flows/prior_flow_xy_nll.eqx ""            data/pqr_indep_nll_aug.npz
run nll_noaug       flows/prior_flow_xy_nll.eqx "--no-augment" data/pqr_indep_nll_noaug.npz
echo "##### ALL DONE #####"
