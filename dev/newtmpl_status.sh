#!/usr/bin/env bash
# One-line status for the from-scratch new-template convergence run.
# Reads cumulative state from the metrics JSON (total_steps is correct across
# OOM/restarts), the in-progress fraction from the newest python log, and the
# OOM-resume count from the newest watchdog log.
setopt NO_NOMATCH 2>/dev/null || true
cd /home/vwetzell/gitrepos/bfd_cnf
META=logs/converge_scratch_newtmpl.json
plog=$(ls -t logs/converge_scratch_newtmpl_*.log 2>/dev/null | head -1); plog=${plog:-/dev/null}
wlog=$(ls -t logs/watchdog_newtmpl_*.log 2>/dev/null | head -1); wlog=${wlog:-/dev/null}

summary=$(python3 - "$META" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    h = [x for x in d.get("history", []) if x.get("block", 0) > 0]
    if not h:
        print("no completed block yet"); raise SystemExit
    b = h[-1]; L = b["loss"]; cv = b["converged"]
    S = ("EquivariantAutoregressiveLayer", "ExplicitPolyLast", "SigmaXCouplingLayer")
    nconv = sum(int(cv[s]) for s in S)
    print(f"steps={b['total_steps']} (blk{b['block']}) lossMed={L['median']:.4f} "
          f"lr={b.get('lr', float('nan')):.2e} stages_converged={nconv}/3 "
          f"[eq{int(cv[S[0]])} ply{int(cv[S[1]])} sx{int(cv[S[2]])}]")
except SystemExit:
    pass
except Exception:
    print("(metrics json not ready)")
PY
)
prog=$(grep -aoE "[0-9]+/10000" "$plog" 2>/dev/null | tail -1)
res=$(grep -ac "resuming from latest ckpt" "$wlog" 2>/dev/null)
printf '[%s] %s | in-progress=%s | OOM-resumes=%s' \
  "$(date '+%F %T')" "${summary:-no data}" "${prog:-0/10000}" "${res:-0}"
