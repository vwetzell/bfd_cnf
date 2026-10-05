#!/bin/bash
# 2026-10-03 -- window shift of the cv6 s2 varobs-vs-sn8r gap: selection terms for 3 extra
# windows on each catalog (base = the full runs' own, symlinked), then dev/window_gap.py.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAC=full FLOW=flows/cv6/centroid_s2.eqx WIN="base flux6k size28 size35 size30 size31 size33 size34"
TAG=varobs_sn8 PQR=pqr/cv6_s2.npz LOGS=logs/cv6/wshift_varobs python -u dev/window_shift.py run || exit 1
TAG=sn8r PQR=pqr/cv6_s2_sn8r_nopool.npz LOGS=logs/cv6/wshift_sn8r python -u dev/window_shift.py run || exit 1
echo "RUN DONE $(date)"
