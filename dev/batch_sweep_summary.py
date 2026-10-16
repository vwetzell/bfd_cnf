"""Integration-loop throughput per --batch-budget from dev/batch_sweep.sh's timestamped logs.
Usage: python dev/batch_sweep_summary.py [GLOB]   (default logs/desv/bsweep/bb_*.log)"""
import glob, re, sys
for p in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "logs/desv/bsweep/bb_*.log"), key=lambda s: int(re.search(r"bb_(\d+)", s)[1])):
    bb = int(re.search(r"bb_(\d+)", p)[1])
    rows = [l.split(" ", 1) for l in open(p) if " " in l]
    prog = [(float(t), l) for t, l in rows if re.search(r"^\s+\d+/\d+ targets$", l.rstrip("\n"))]
    t0 = float(rows[0][0]); tend = float(rows[-1][0])
    if len(prog) < 2:
        print(f"bb {bb:7d}: no integration progress (failed early?)  wall {tend - t0:.0f} s"); continue
    n = int(re.search(r"/(\d+)", prog[0][1])[1])
    # each arm prints 8/n first and ~n last; time from first arm's first line to second arm's last line
    first = [i for i, (_, l) in enumerate(prog) if re.search(r"^\s+\d+/", l) and int(re.search(r"(\d+)/", l)[1]) <= 64]
    a1, a2 = first[0], first[1] if len(first) > 1 else None
    span1 = prog[a2 - 1][0] - prog[a1][0] if a2 else float("nan")
    span2 = prog[-1][0] - prog[a2][0] if a2 else float("nan")
    done1 = int(re.search(r"(\d+)/", prog[a2 - 1][1])[1]) - int(re.search(r"(\d+)/", prog[a1][1])[1])
    done2 = int(re.search(r"(\d+)/", prog[-1][1])[1]) - int(re.search(r"(\d+)/", prog[a2][1])[1])
    rate = (done1 + done2) / (span1 + span2)
    print(f"{p}: bb {bb:7d} ({bb // 4096:2d} targets/batch): {n} targets/arm, arm loops {span1:.0f}+{span2:.0f} s, "
          f"{rate:.1f} target-arms/s ({1000 / rate:.1f} ms each), total wall {tend - t0:.0f} s")
