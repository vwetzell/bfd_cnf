"""Split a template catalog and its row-aligned copies catalog into K disjoint row ranges.

  python dev/split_templates.py MOMENTS.fits COPIES.fits K

Writes MOMENTS_k{i}.fits (rows [i n/K, (i+1) n/K) of every table HDU) and COPIES_k{i}.fits
(those GALAXIES/POPULATION rows plus their copies, `gal` re-based to 0) for i < K.  The copies
catalog is grouped by `gal` and its GALAXIES table is row for row the moments catalog
(asserted), so each split is an independent template set with its own copies -- the
paper's per-batch template sets (Bernstein+16 sec. 5).  2026-10-09: template-realization test.
"""
import sys

import fitsio
import numpy as np

STRUCT = ("XTENSION", "BITPIX", "NAXIS", "PCOUNT", "GCOUNT", "TFIELDS", "TTYPE", "TFORM",
          "TDIM", "TUNIT", "EXTNAME", "EXTEND", "SIMPLE")


def header(h):
    return [{"name": k, "value": h[k]} for k in h.keys() if not k.startswith(STRUCT)]


def write(out, hdus):
    """hdus: [(extname, table, header)]; the (empty) primary HDU is fitsio's own."""
    with fitsio.FITS(out + ".tmp", "rw", clobber=True) as f:
        for name, data, hdr in hdus:
            f.write(data, extname=name or None, header=hdr)
    import os
    os.replace(out + ".tmp", out)


mpath, cpath, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
fm, fc = fitsio.FITS(mpath), fitsio.FITS(cpath)
n = fm[1].get_nrows()
gal = fc["COPIES"].read(columns=["gal"])["gal"]
assert (np.diff(gal) >= 0).all() and fc["GALAXIES"].get_nrows() == n
assert np.array_equal(fc["GALAXIES"].read(columns=["moments"])["moments"],
                      fm[1].read(columns=["moments"])["moments"]), "copies GALAXIES not row-aligned"
edges = [i * n // K for i in range(K + 1)]
for i in range(K):
    lo, hi = edges[i], edges[i + 1]
    mo = mpath.replace(".fits", f"_k{i}.fits")
    write(mo, [(h.get_extname(), h[lo:hi], header(h.read_header())) for h in fm[1:]])
    a, b = np.searchsorted(gal, [lo, hi])
    cop = fc["COPIES"][int(a):int(b)]
    cop["gal"] -= lo
    co = cpath.replace(".fits", f"_k{i}.fits")
    write(co, [("COPIES", cop, header(fc["COPIES"].read_header()))]
          + [(name, fc[name][lo:hi], header(fc[name].read_header())) for name in ("GALAXIES", "POPULATION")])
    print(f"split {i}: templates [{lo}, {hi})  copies [{a}, {b}) = {b - a}  -> {mo}, {co}", flush=True)
