#!/bin/zsh
# 2026-10-15: --batch-budget sweep on the x6r flow, first 10000 desv targets, x6rA flags.  Output lines are
# timestamped so the integration loop can be timed alone; peak GPU memory sampled every 0.5 s.  Stops
# escalating at the first OOM / failure.  Summarise with dev/batch_sweep_summary.py.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Env: FLOW (default x6r), TAG (log dir suffix), BBS (budgets), BFD_TIMING passes through to bias.py.
D=../bfd_cnf_imsims/data; L=logs/desv/bsweep${TAG:+_$TAG}; F=flows/desm; mkdir -p $L
FLOW=${FLOW:-$F/centroid_tnc15c_x6r.eqx}
BF=(--samples 8192 --alpha 0.5 --chunk 4096 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz
  --template-cut 1200 1.5)
for bb in ${=BBS:-8192 16384 32768 65536 131072 262144}; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $L/mem_$bb.txt &
    smi=$!
    python -u bias.py --pop bulgedisc_g2n_1m_desv $BF --batch-budget $bb --flow $FLOW \
        --window-size 2.2 3.5 --window-flux 2200 90000 --n-targets 10000 2>&1 \
        | while IFS= read -r l; do print -r -- "$(date +%s.%N) $l"; done > $L/bb_$bb.log
    rc=${pipestatus[1]}
    kill $smi 2>/dev/null; wait $smi 2>/dev/null
    peak=$(sort -n $L/mem_$bb.txt | tail -1)
    oom=$(grep -ciE 'RESOURCE_EXHAUSTED|out of memory|OOM' $L/bb_$bb.log)
    echo "bb $bb exit $rc peak_mem ${peak} MiB oom_lines $oom  $(grep -E 'windowed, corrected' $L/bb_$bb.log | tr -s ' ' | cut -c1-90)"
    if [ $rc -ne 0 ] || [ $oom -gt 0 ]; then echo "stopping escalation at $bb"; break; fi
done
echo "BATCH SWEEP DONE $(date)"
