#!/bin/bash
# Does raising `--samples` cure the deep arm's draw noise?
#
# [[draw-noise-not-negligible-at-depth]]: on gauss2_v3d, two draw seeds at
# S = 8192 move windowed uncorrected m1 by 1.15e-02 on IDENTICAL targets.  If
# the per-target IS estimator has finite variance that falls as 1/sqrt(S), more
# draws fix it.  If it does not fall, the variance is heavy-tailed and no draw
# count helps -- the regime `[[alpha1-ess-does-not-scale]]` and
# `[[r-is-an-is-artifact]]` describe -- and the deep arm needs a different
# estimator rather than more compute.
#
# MEASURED ON PER-TARGET Q, NOT ON m1.  Two seeds give only ONE sample of the
# m1 spread, whose own uncertainty is ~70% -- far too weak to tell 1/sqrt(S)
# from no scaling.  The same two runs give ~1900 in-window targets, each an
# independent measurement of the per-target draw error, so rms|dQ| is pinned to
# a few percent and IS what drives the ensemble sums.  m1 is reported too, as a
# cross-check, not as the evidence.
#
# `--no-zero-arm` throughout: the zero arm costs a third and feeds nothing here.
# Note [[pqr-streamed-seed-nests]]: raising S at fixed seed reuses the smaller
# draw set, so the two S values are correlated -- fine, the quantity compared is
# the spread BETWEEN seeds at fixed S.
set -e -o pipefail
cd "$(dirname "$0")/.."
S=${OUT:-/tmp/prefilter_check}
mkdir -p "$S"

COMMON=(--pop gauss2_v3d --flow flows/centroid_g2v3d.eqx
        --alpha 0.5 --chunk 4096 --batch-budget 65536
        --n-targets "${N:-6000}" --gauge auto --no-zero-arm
        --window-size 2.2 3.2 --window-flux 2500 50000
        --window-terms score --window-draws 1048576
        --floor-eps 1e-3 --window-guard 100)

for s in 8192 16384; do
    for seed in 1000 1007; do
        echo "=== S=$s draw-seed $seed ==="
        t0=$SECONDS
        python -u bias.py "${COMMON[@]}" --samples "$s" --draw-seed "$seed" \
            --save-pqr "$S/pqr_${s}_${seed}.npz" > "$S/s${s}_${seed}.log" 2>&1
        echo "  $((SECONDS - t0))s"
    done
done

python - "$S" <<'EOF'
import sys, numpy as np
S = sys.argv[1]
def win(o, size=(2.2,3.2), flux=(2500.,50000.)):
    r = o[:,1]/o[:,0]
    return (r>size[0])&(r<size[1])&(o[:,0]>flux[0])&(o[:,0]<flux[1])
print(f"\n{'S':>7} {'rms|dQ1|':>10} {'p50|dQ1|':>10} {'rms|dR11|':>11} {'n':>6}")
prev = None
for s in (8192, 16384):
    a = np.load(f"{S}/pqr_{s}_1000.npz"); b = np.load(f"{S}/pqr_{s}_1007.npz")
    k = win(a["obs_plus"])
    dq = (a["plus_q"] - b["plus_q"])[k][:, 0]
    dr = (a["plus_r"] - b["plus_r"])[k][:, 0, 0]
    rms = float(np.sqrt(np.mean(dq**2)))
    print(f"{s:>7} {rms:>10.4g} {np.median(np.abs(dq)):>10.4g} "
          f"{float(np.sqrt(np.mean(dr**2))):>11.4g} {k.sum():>6}")
    if prev is not None:
        print(f"\n  rms|dQ1| ratio 8192->16384: {rms/prev:.3f}"
              f"   (1/sqrt(2) = 0.707 if the variance is finite;"
              f" ~1.0 if no draw count helps)")
    prev = rms
EOF
echo
grep -H "windowed, uncorrected" "$S"/s8192_*.log "$S"/s16384_*.log || true
