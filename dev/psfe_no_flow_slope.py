"""Paired-bootstrap dc1/d(psf_e1), dc2/d(psf_e2) from dev/psfe_no_flow_check.py's
cached per-target Q,R (dev/_psfe_no_flow_cache/*.npz).

All 10 configs share the SAME 20000-row target catalogs in the SAME row
order (verified: psfe_no_flow_check.py's `good` mask kept every row for
every config), so unlike dev/psfe_paired_diff.py no nearest-neighbor
matching is needed -- rows are already aligned by construction. One shared
bootstrap resample index, applied to every config, gives the properly
correlated paired error directly.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import bias as B

CACHE = os.environ.get("CACHE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_psfe_no_flow_cache"))
BOOT = int(os.environ.get("BOOT", 500))
SEED = int(os.environ.get("SEED", 0))


def load(tag):
    z = np.load(f"{CACHE}/{tag}.npz")
    return z


if __name__ == "__main__":
    files = sorted(glob.glob(f"{CACHE}/*.npz"))
    tags = [os.path.splitext(os.path.basename(f))[0] for f in files]
    data = {t: load(t) for t in tags}
    # each config's own window/ESS cut drops a slightly different handful of
    # rows near the boundary; row POSITION is otherwise a valid common
    # galaxy id (all 10 configs share the same seed-1 galaxies/noise in the
    # same row order), so restrict every config to the intersection of kept
    # indices before pairing.
    common = None
    for t in tags:
        s = set(data[t]["idx"].tolist())
        common = s if common is None else (common & s)
    common = np.array(sorted(common))
    print(f"{len(common)} rows common to all {len(tags)} configs "
          f"(out of {min(len(data[t]['idx']) for t in tags)}-"
          f"{max(len(data[t]['idx']) for t in tags)} per config)")

    def sub(t):
        pos = np.searchsorted(data[t]["idx"], common)
        return {k: data[t][k][pos] for k in ("qp", "rp", "qm", "rm")}

    data = {t: {**sub(t), "psf_e1": data[t]["psf_e1"], "psf_e2": data[t]["psf_e2"]}
            for t in tags}
    n = len(common)

    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, n, (BOOT, n))

    c1 = {}
    c2 = {}
    e1 = {}
    e2 = {}
    for t in tags:
        d = data[t]
        m1, c1p, c2p = B.bias(d["qp"], d["rp"], d["qm"], d["rm"])
        c1[t], c2[t] = c1p, c2p
        e1[t], e2[t] = float(d["psf_e1"]), float(d["psf_e2"])
        print(f"{t:>10s}  psf_e=({e1[t]:+.3f},{e2[t]:+.3f})  "
              f"m1={m1:+.4f}  c1={c1p:+.5f}  c2={c2p:+.5f}")

    base = "psfe00"
    print(f"\n{'config':>10s}{'psf_e1':>9s}{'psf_e2':>9s}"
          f"{'dc1':>12s}{'+/-':>10s}{'dc2':>12s}{'+/-':>10s}")
    rows = []
    for t in tags:
        if t == base:
            continue
        d, d0 = data[t], data[base]
        dc1s, dc2s = [], []
        for b in boot_idx:
            m1b, c1b, c2b = B.bias(d["qp"][b], d["rp"][b], d["qm"][b], d["rm"][b])
            m10, c10, c20 = B.bias(d0["qp"][b], d0["rp"][b], d0["qm"][b], d0["rm"][b])
            dc1s.append(c1b - c10)
            dc2s.append(c2b - c20)
        dc1, dc1e = c1[t] - c1[base], np.std(dc1s, ddof=1)
        dc2, dc2e = c2[t] - c2[base], np.std(dc2s, ddof=1)
        rows.append((t, e1[t], e2[t], dc1, dc1e, dc2, dc2e))
        print(f"{t:>10s}{e1[t]:9.3f}{e2[t]:9.3f}"
              f"{dc1:12.5f}{dc1e:10.5f}{dc2:12.5f}{dc2e:10.5f}", flush=True)

    # weighted least-squares slope through the origin (baseline already
    # subtracted, so dc(psf_e=0) = 0 by construction), separately for the
    # e1-axis and e2-axis configs
    e1r = np.array([r[1] for r in rows])
    e2r = np.array([r[2] for r in rows])
    dc1r = np.array([r[3] for r in rows])
    dc1e = np.array([r[4] for r in rows])
    dc2r = np.array([r[5] for r in rows])
    dc2e = np.array([r[6] for r in rows])

    m1sel = e1r != 0
    m2sel = e2r != 0
    w1 = 1.0 / dc1e[m1sel] ** 2
    s1 = (w1 * dc1r[m1sel] * e1r[m1sel]).sum() / (w1 * e1r[m1sel] ** 2).sum()
    s1e = 1.0 / np.sqrt((w1 * e1r[m1sel] ** 2).sum())
    w2 = 1.0 / dc2e[m2sel] ** 2
    s2 = (w2 * dc2r[m2sel] * e2r[m2sel]).sum() / (w2 * e2r[m2sel] ** 2).sum()
    s2e = 1.0 / np.sqrt((w2 * e2r[m2sel] ** 2).sum())
    print(f"\ndc1/d(psf_e1) = {s1:+.4f} +/- {s1e:.4f} ({s1 / s1e:+.1f} sigma)")
    print(f"dc2/d(psf_e2) = {s2:+.4f} +/- {s2e:.4f} ({s2 / s2e:+.1f} sigma)")
