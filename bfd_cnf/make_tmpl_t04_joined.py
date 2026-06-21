"""
Rebuild tmpl_t04_joined.fits by joining tmpl_t04.fits with the covariance
column from summary_templates_new.fits.

Matching key: tmpl_t04["id"] == summary_templates_new["coadd_ID"]
Added column:  covariance (15 x float64, packed lower-triangular 5x5)

Usage
-----
    python make_tmpl_t04_joined.py

Output is written to the same directory as the input files (BFD_CNF_DIR).
Rows in tmpl_t04 with no matching coadd_ID in summary_templates_new are
included but their covariance entries will be zero-filled.

"""

import os
import numpy as np
import fitsio

BFD_CNF_DIR = os.environ.get("BFD_DATA_DIR", "/home/vwetzell/Documents/BFD_cNF")

T04_PATH      = f"{BFD_CNF_DIR}/tmpl_t04.fits"
SUMMARY_PATH  = f"{BFD_CNF_DIR}/summary_templates_new.fits"
OUTPUT_PATH   = f"{BFD_CNF_DIR}/tmpl_t04_joined.fits"

print(f"Reading {T04_PATH} ...")
with fitsio.FITS(T04_PATH) as _f:
    tmpl_t04 = _f[1].read()
    hdr = _f[1].read_header()
print(f"  {len(tmpl_t04):,} rows")

print(f"Reading covariance column from {SUMMARY_PATH} ...")
with fitsio.FITS(SUMMARY_PATH) as _f:
    templates = _f[1].read(columns=["coadd_ID", "covariance"])
print(f"  {len(templates):,} rows")

# Preserve header keywords from tmpl_t04
keep_keys = ["TIER_NUM", "WT_TYPE", "WT_SIGMA", "SIG_MAX",
             "SIG_STEP", "SIG_FLUX", "SIG_XY", "FLUX_MIN"]

# Join: append covariance from summary_templates_new matched on id == coadd_ID
print("Joining on id == coadd_ID ...")
lookup = {int(v): i for i, v in enumerate(templates["coadd_ID"])}
indices = np.array([lookup.get(int(v), -1) for v in tmpl_t04["id"]])
matched = indices != -1
print(f"  {matched.sum():,} / {len(tmpl_t04):,} rows matched")

new_dtype = np.dtype(
    tmpl_t04.dtype.descr + [("covariance", templates.dtype["covariance"])]
)
result = np.zeros(len(tmpl_t04), dtype=new_dtype)
for field in tmpl_t04.dtype.names:
    result[field] = tmpl_t04[field]
result["covariance"][matched] = templates["covariance"][indices[matched]]
del tmpl_t04, templates

header = [{"name": k, "value": hdr[k]} for k in keep_keys if k in hdr]

tmp_path = OUTPUT_PATH + ".tmp"
print(f"Writing {tmp_path} ...")
fitsio.write(tmp_path, result, header=header, clobber=True)
os.replace(tmp_path, OUTPUT_PATH)
# Make the file read-only so cfitsio cannot write its "in use" marker when
# subsequently opened for reading.  If cfitsio gets write access it marks the
# file header on open and clears the mark on close; a process that is killed
# before close (e.g. OOM) leaves a stale mark, causing cfitsio's recovery
# logic to truncate the file on the next open → "tried to move past end of
# file".  With read-only permissions cfitsio falls back silently to a true
# read-only open and never writes to the file.
os.chmod(OUTPUT_PATH, 0o444)

print("Done.")
print(f"Verify: {OUTPUT_PATH}")
