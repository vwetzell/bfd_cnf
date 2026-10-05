#!/bin/bash
# 2026-10-05 -- after the memory-reaped overnight chain: s5 target batch (is its +13% R_s
# matched in its targets?), finish s6 (centroid restarts from scratch; bulk/shear reused) + its
# batch, then s4 on render seeds 4-5.  Every batch = own targets + own selection terms.
cd "$(dirname "$0")/.."; L=logs/cv6
echo "START $(date)"
SEEDS="5 6" BATCH_FLOWS="5 6" TBATCH=3 bash dev/flow_seeds.sh >> $L/flow_seeds.log 2>&1; echo "s5/s6 exit $? $(date +%T)"
SEEDS=4 BATCH_FLOWS=4 TBATCH="4 5" bash dev/flow_seeds.sh >> $L/flow_seeds.log 2>&1; echo "s4 batches exit $? $(date +%T)"
echo "ALL DONE $(date)"
