"""Per-target Q/R agreement of two saved PQRs on the same targets and draws (dev/taylor_check.sh).
Usage: python dev/taylor_compare.py A.npz B.npz"""
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
assert (a["target_rows"] == b["target_rows"]).all()
for k in ("plus_q", "plus_r", "minus_q", "minus_r"):
    x, y = a[k], b[k]
    ok = np.isfinite(x).all(axis=tuple(range(1, x.ndim))) & np.isfinite(y).all(axis=tuple(range(1, y.ndim)))
    d = np.abs(x[ok] - y[ok]); s = np.abs(x[ok]).reshape(ok.sum(), -1).max(1)
    rel = d.reshape(ok.sum(), -1).max(1) / s
    print(f"{k}: {ok.sum()}/{len(ok)} finite in both  per-target max rel diff median {np.median(rel):.1e}  "
          f"p99 {np.percentile(rel, 99):.1e}  max {rel.max():.1e}  sum-level rel {np.abs(x[ok].sum(0) - y[ok].sum(0)).max() / np.abs(x[ok].sum(0)).max():.1e}")
