"""Backfill standardiser sidecars for already-trained flows.

Newer training runs write ``<flow>.eqx.stats.npz`` automatically (see
``models.bijections.save_stats``).  Flows trained before that wrote nothing, so
their standardiser must be reconstructed from the training template table — NOT
copied from a possibly-stale ``data/raw2standard_stats*.npz``.

The rebuild reads only the ``moments``/``cov`` columns in chunks and applies the
exact ``_finalize_dataset`` standardiser formula (the full pipeline's quadratic
fit + resampling are irrelevant to mean/std and OOM on a low-RAM box).  The
derivative tail-cuts are skipped: they trim ~0.02%/dim and move mean/std by
<1e-3 (the table is stable to <1e-3 on a 10M-row sample).

Provenance report: prints the rebuilt ``c1 = mean[1]/std[1]`` (the size constant
baked into the flow architecture, derived per-run from the standardiser) next to
``--reference`` (the npz the flow was last known to match).  A
mismatch with the reference means the template table changed since the flow was
trained, so the rebuilt stats would warp this flow — heed the warning.

Usage
-----
    python -m bfd_cnf.backfill_stats                       # canonical prior+q
    python -m bfd_cnf.backfill_stats --flow flows/x.eqx ...
"""

from __future__ import annotations

import argparse
import os

import bfd
import fitsio
import numpy as np

from .config import (
    DATA_DIR, PRIOR_FLOW_PATH, Q_FLOW_PATH, TRAIN_FITS_PATH,
)
from .data import quality_cut_mask
from .models.bijections import RawMomentStandardize, save_stats

_CHUNK = 4_000_000


def rebuild_standardize_full(fits_path: str = TRAIN_FITS_PATH) -> RawMomentStandardize:
    """Recompute the full 4-vector standardiser (mean, std) in one chunked pass."""
    sums = np.zeros(4)
    sqs = np.zeros(4)
    n = 0
    fits = fitsio.FITS(fits_path)
    try:
        h = fits[1]
        n_total = h.get_nrows()
        for s in range(0, n_total, _CHUNK):
            e = min(s + _CHUNK, n_total)
            blk = h.read(rows=np.arange(s, e), columns=["moments", "cov"])
            mom = blk["moments"][:, :4].astype(np.float64)
            cov = bfd.MomentCovariance.bulkUnpack(blk["cov"])[:, :4, :4].astype(np.float64)
            mom = mom[quality_cut_mask(mom, cov)]
            if not len(mom):
                continue
            t = np.stack([
                np.log10(mom[:, 0]),
                mom[:, 1] / mom[:, 0],
                mom[:, 2] / mom[:, 1],
                mom[:, 3] / mom[:, 1],
            ], axis=1)
            sums += t.sum(0)
            sqs += (t * t).sum(0)
            n += len(mom)
            print(f"  rows {e}/{n_total}  kept so far {n}")
    finally:
        fits.close()

    mean = sums / n
    std = np.sqrt(np.maximum(sqs / n - mean ** 2, 0.0))
    mean[2:] = 0.0
    std[2:] = std[2:].mean()
    print(f"rebuilt from {n} quality-cut templates")
    return RawMomentStandardize(mean=mean, std=std)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow", nargs="+", default=[PRIOR_FLOW_PATH, Q_FLOW_PATH],
                    help="Flow .eqx files to write sidecars for (all share the "
                         "one training standardiser).")
    ap.add_argument("--reference", default=os.path.join(
                        DATA_DIR, "raw2standard_stats_newtmpl.npz"),
                    help="npz the flow was last known to match (provenance check).")
    ap.add_argument("--rtol", type=float, default=2e-3,
                    help="Tolerance for the reference-match check.")
    ap.add_argument("--no-write", action="store_true",
                    help="Only report; do not write sidecars.")
    args = ap.parse_args()

    print(f"Rebuilding standardiser from {TRAIN_FITS_PATH} (chunked)...")
    r2s = rebuild_standardize_full()
    mean, std = np.asarray(r2s.mean), np.asarray(r2s.std)
    c1 = float(mean[1] / std[1])
    print(f"\n  rebuilt mean={np.round(mean, 5)} std={np.round(std, 5)}")
    print(f"  rebuilt c1 = mean[1]/std[1] = {c1:.6f}  (derived per-run; saved in the sidecar)")

    matches_ref = None
    if os.path.exists(args.reference):
        d = np.load(args.reference)
        matches_ref = (np.allclose(mean, d["mean"], atol=args.rtol)
                       and np.allclose(std, d["std"], atol=args.rtol))
        print(f"  reference {os.path.basename(args.reference)}: "
              f"mean={np.round(np.asarray(d['mean']), 5)} "
              f"std={np.round(np.asarray(d['std']), 5)}  "
              f"-> {'MATCH' if matches_ref else 'DIFFERS'}")
    else:
        print(f"  reference {args.reference} not found; skipping provenance check.")

    if matches_ref is False:
        print("\nWARNING: rebuilt standardiser DIFFERS from the reference the flow "
              "was last known to match — the template table changed since the flow "
              "was trained.  Using these stats will warp this flow's outputs.")

    if args.no_write:
        return 0
    for f in args.flow:
        print(f"  wrote {save_stats(f, r2s)}")
    return 0 if matches_ref is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
