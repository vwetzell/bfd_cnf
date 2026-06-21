#!/usr/bin/env bash
# Check 2: does the nda flow's m-bias depend on the centroid C_X scale?
# Re-integrate the canonical (nda-weighted 510k) flow at two FIXED log_scales:
#   11.94 = matched to the t04 grid generation noise (SIG_XY=391)
#   11.41 = the targets' actual median C_X
# Same target set/flux cut as the per-target run that gave m = -0.215.
set -u
cd /home/vwetzell/gitrepos/bfd_cnf
for LS in 11.94 11.41; do
  tag=$(echo "$LS" | tr -d '.')
  log="logs/integrate_100k_nda510k_fixedls${tag}.log"
  echo "[$(date '+%T')] === fixed-log-scale=$LS  ->  $log ==="
  python -u -m bfd_cnf.integrate_grid \
    --n-targets 100000 --flux-min 3000 --flux-max 90000 \
    --fixed-log-scale "$LS" --fixed-e1 0 --fixed-e2 0 \
    --out "data/pqr_grid_100k_nda510k_fixedls${tag}.npz" > "$log" 2>&1
  echo "[$(date '+%T')] --- fixed-log-scale=$LS result ---"
  grep -E "multiplicative bias m|m point=" "$log" || echo "  (no result line — check $log)"
done
echo "[$(date '+%T')] === CHECK2 DONE ==="
