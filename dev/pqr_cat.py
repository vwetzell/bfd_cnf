#!/usr/bin/env python
"""Concatenate the PQR files of a sliced bias.py run into one.

A 30 h run is split with `--target-offset` into slices that each write their
own `--save-pqr`; every array in the file is per-target, so the join is a
concatenate on axis 0 and the re-solve (`dev/window_scan.py`) cannot tell the
difference.  Order the slices by offset on the command line -- nothing here
sorts them, and `dev/bias.compare` pairs two files row by row.

Usage:  python dev/pqr_cat.py out.npz slice0.npz slice1.npz ...
"""
import sys

import numpy as np


def cat(out, paths):
    ds = [np.load(p) for p in paths]
    keys = set(ds[0].files)
    for p, d in zip(paths, ds):
        if set(d.files) != keys:
            raise SystemExit(f"{p} has different keys: {sorted(set(d.files) ^ keys)}")
    merged = {k: np.concatenate([d[k] for d in ds]) for k in keys}
    np.savez_compressed(out, **merged)
    n = len(merged["plus_q"])
    print(f"wrote {out}: {n} targets from {len(paths)} slices "
          f"({', '.join(str(len(d['plus_q'])) for d in ds)})")
    if "prefilter_w" not in keys:
        print("WARNING: no prefilter_w -- window_scan will refuse this file")


def demo():
    import tempfile, os
    with tempfile.TemporaryDirectory() as t:
        rng = np.random.default_rng(0)
        parts, want = [], {}
        for i, n in enumerate((3, 5)):
            d = {"plus_q": rng.normal(size=(n, 2)),
                 "plus_r": rng.normal(size=(n, 2, 2)),
                 "prefilter_w": np.full(n, 1.0 + i)}
            p = os.path.join(t, f"s{i}.npz"); np.savez(p, **d); parts.append(p)
            for k, v in d.items():
                want.setdefault(k, []).append(v)
        out = os.path.join(t, "all.npz")
        cat(out, parts)
        got = np.load(out)
        for k, vs in want.items():
            assert np.array_equal(got[k], np.concatenate(vs)), k
        assert len(got["plus_q"]) == 8
    print("demo ok")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "demo":
        demo()
    elif len(sys.argv) < 4:
        raise SystemExit(__doc__)
    else:
        cat(sys.argv[1], sys.argv[2:])
