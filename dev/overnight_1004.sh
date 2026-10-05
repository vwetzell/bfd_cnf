#!/bin/bash
# 2026-10-04 overnight chain (serial): Moffat-flip catalog E, flow seeds s4-s6 (training +
# tapered selection terms; s4 also one target batch), then the offline analysis.
cd "$(dirname "$0")/.."
L=logs/cv6; R=$L/overnight_report.log
echo "START $(date)"
bash dev/cond_batches.sh condE >> $L/cond_batches.log 2>&1; echo "condE exit $? $(date +%T)"
SEEDS="4 5 6" bash dev/flow_seeds.sh >> $L/flow_seeds.log 2>&1; echo "flow_seeds exit $? $(date +%T)"
{ echo "== condE - condD (Moffat PSF e reversed: diff = 2 x leakage at e = (0.0354, 0.0354))"
  JAX_PLATFORMS=cpu python -u dev/pair_sn8r_varobs.py --taper pqr/cv6_s2_condD.npz $L/bias_cv6_s2_condD.log sn8_condD \
      pqr/cv6_s2_condE.npz $L/bias_cv6_s2_condE.log sn8_condE 2>&1 | grep "^taper"
  echo "== condE - base"
  JAX_PLATFORMS=cpu python -u dev/pair_sn8r_varobs.py --taper pqr/cv6_s2_sn8r.npz $L/taper_sel_sn8r.log sn8r \
      pqr/cv6_s2_condE.npz $L/bias_cv6_s2_condE.log sn8_condE 2>&1 | grep "^taper"
  echo "== flow s4 vs s2 on batch sn8r_s3 (target-side check)"
  JAX_PLATFORMS=cpu python -u dev/flow_pool.py s4 2>&1 | grep -v Warn
  echo "== flow-averaged selection terms"
  JAX_PLATFORMS=cpu python -u dev/flow_sel_avg.py 2>&1 | grep -v Warn
} > $R 2>&1
echo "ALL DONE $(date)"
