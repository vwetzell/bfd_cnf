"""
Join the two row-aligned new-template files into one lean training table.

Inputs (row-aligned — same coadd id and moments at every row; verified per block):
  data/templates.hdf5         : derivs(7,10) + nda  → PRIMARY source for everything
                                except the covariance
  data/templates_summary.fits : packed covariance  → cov only (the one column the
                                HDF5 lacks)

The HDF5 is the authoritative BFD Template output; the summary is a derived export
whose only unique column is the covariance (its derivs/moments are bit-identical to
the HDF5's).  Neither file alone is trainable: the HDF5 lacks cov, and the summary
lacks the centroid (MX,MY) and nda the ELBO's L(X|C_X) centroid weighting and the
nda weighting need.  This writes exactly the columns the trainer consumes
(see bfd_cnf.data.load_training_table, which consumes them) plus the coadd id:

  id        i8            coadd id                                   (hdf5)
  moments   f4 (4,)       [Mf, Mr, M1, M2]            = derivs[0:4,0] (hdf5)
  cov       f4 (4,4)      covariance of [Mf, Mr, M1, M2]             (summary)
  dm_dg     f4 (4,2)      dM/dg : [...,0]=dg1, [...,1]=dg2
                          = derivs[0:4, (DG1=1, DG2=2)]              (hdf5)
  d2m_dg2   f4 (4,2,2)    d2M/dg2 (symmetric): [0,0]=g1g1, [0,1]=[1,0]=g1g2, [1,1]=g2g2
                          = derivs[0:4, (DG1DG1=4, DG1DG2=5, DG2DG2=6)] (hdf5)
  centroid  f4 (2,)       [MX, MY]                    = derivs[5:7,0] (hdf5)
  nda       f4            BFD sky_density x da weight                 (hdf5)

All rows are written (no quality cut) by default — training applies its own cuts —
unless --cut is given.  Stored as float32 to match the original training precision.
"""

import argparse
import os

import fitsio
import h5py
import numpy as np
import yaml

import bfd

_OUT_DTYPE = [
    ("id", "i8"),
    ("moments", "f4", (4,)),
    ("cov", "f4", (4, 4)),
    ("dm_dg", "f4", (4, 2)),
    ("d2m_dg2", "f4", (4, 2, 2)),
    ("centroid", "f4", (2,)),
    ("nda", "f4"),
]
# Pqr derivative-column indices in the HDF5 derivs(7,10) (per bfd.pqr.Pqr).
_D0, _DG1, _DG2, _DG1DG1, _DG1DG2, _DG2DG2 = 0, 1, 2, 4, 5, 6
# The summary is read only for its covariance (+ id, for the per-block alignment check).
_SUMMARY_COLS = ["id", "covariance"]


def _read_tier_meta(hdf5_path):
    """Return the BFD noise-tier metadata omap from the HDF5 as a flat dict."""
    if not os.path.exists(hdf5_path):
        return {}
    with h5py.File(hdf5_path, "r") as f:
        ds = f.get("templates.__table_column_meta__")
        if ds is None:
            return {}
        blob = "\n".join(bytes(x).decode("latin1") for x in ds[:])
    out = {}
    try:
        for item in yaml.safe_load(blob).get("meta", []):
            if isinstance(item, (tuple, list)) and len(item) == 2:
                out[str(item[0])] = item[1]
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="data/templates_summary.fits")
    ap.add_argument("--hdf5", default="data/templates.hdf5")
    ap.add_argument("--out", default="data/templates_train.fits")
    ap.add_argument("--block", type=int, default=1_000_000,
                    help="Rows per block (memory ~1 GB/block; default 1M).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N rows (0 = all; for a quick test).")
    ap.add_argument("--cut", action="store_true",
                    help="Apply bfd_cnf.data.quality_cut_mask before writing.")
    args = ap.parse_args()

    quality_cut_mask = None
    if args.cut:
        from bfd_cnf.data import quality_cut_mask  # noqa: local import keeps deps light

    tier = _read_tier_meta(args.hdf5)
    header = [{"name": "JOINSRC1", "value": os.path.basename(args.hdf5),
              "comment": "PRIMARY: moments, derivs, centroid, nda, id"},
             {"name": "JOINSRC2", "value": os.path.basename(args.summary),
              "comment": "covariance only"}]
    _hdr_map = {"sigmaXY": "SIGMAXY", "sigmaFlux": "SIGFLUX", "fluxMin": "FLUXMIN",
                "weightN": "WEIGHTN", "weightSigma": "WGTSIGMA", "sigmaMax": "SIGMAMAX",
                "sigmaStep": "SIGSTEP", "tierNumber": "TIERNUM", "weightType": "WGTTYPE"}
    for k, kk in _hdr_map.items():
        if k in tier:
            header.append({"name": kk, "value": tier[k], "comment": f"BFD tier {k}"})

    if os.path.exists(args.out):
        os.remove(args.out)

    fsum = fitsio.FITS(args.summary)
    hf = h5py.File(args.hdf5, "r")
    fout = fitsio.FITS(args.out, "rw")
    n_total = n_written = 0
    try:
        hsum = fsum[1]
        ds = hf["templates"]
        N = hsum.get_nrows()
        if ds.shape[0] != N:
            raise ValueError(f"row-count mismatch: summary {N} vs hdf5 {ds.shape[0]}")
        if args.limit > 0:
            N = min(N, args.limit)
        print(f"Joining {N:,} rows  ({args.summary} + {args.hdf5}) -> {args.out}")

        for s in range(0, N, args.block):
            e = min(s + args.block, N)
            rows = np.arange(s, e)
            hrec = ds[s:e]                                   # PRIMARY (hdf5)
            srec = hsum.read(columns=_SUMMARY_COLS, rows=rows)  # cov only
            if not np.array_equal(hrec["id"], srec["id"]):
                raise ValueError(f"files not row-aligned in block [{s}:{e})")

            d = hrec["derivs"][:, :4, :].astype(np.float64)  # (n,4 even moments,10 Pqr)
            mom = d[:, :, _D0]                               # [Mf,Mr,M1,M2]
            centroid = hrec["derivs"][:, 5:7, _D0].astype(np.float64)   # [MX,MY]
            cov = bfd.MomentCovariance.bulkUnpack(srec["covariance"])[:, :4, :4]  # (summary)

            if args.cut:
                keep = quality_cut_mask(mom, cov)
            else:
                keep = np.ones(len(mom), dtype=bool)

            n = int(keep.sum())
            out = np.zeros(n, dtype=_OUT_DTYPE)
            out["id"] = hrec["id"][keep]
            out["moments"] = mom[keep]
            out["cov"] = cov[keep]
            out["dm_dg"][:, :, 0] = d[:, :, _DG1][keep]
            out["dm_dg"][:, :, 1] = d[:, :, _DG2][keep]
            out["d2m_dg2"][:, :, 0, 0] = d[:, :, _DG1DG1][keep]
            cross = d[:, :, _DG1DG2][keep]
            out["d2m_dg2"][:, :, 0, 1] = cross
            out["d2m_dg2"][:, :, 1, 0] = cross
            out["d2m_dg2"][:, :, 1, 1] = d[:, :, _DG2DG2][keep]
            out["centroid"] = centroid[keep]
            out["nda"] = hrec["nda"][keep]

            if n_written == 0:
                fout.write(out, header=header)
            else:
                fout[-1].append(out)
            n_total += len(mom)
            n_written += n
            print(f"  [{e:>10,}/{N:,}]  kept {n_written:,}", flush=True)
    finally:
        fsum.close()
        hf.close()
        fout.close()

    keep_frac = 100.0 * n_written / max(n_total, 1)
    print(f"Done: wrote {n_written:,} of {n_total:,} rows ({keep_frac:.1f}%) to {args.out}")


if __name__ == "__main__":
    main()
