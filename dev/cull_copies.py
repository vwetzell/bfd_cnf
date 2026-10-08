"""Cull unphysical template copies from an `imsims.copies` catalog.  CPU, numpy only.  (2026-10-07)

A copy is UNPHYSICAL if Mf < 0, Mr < 0 or |e| = hypot(M1, M2) / Mr > 1 -- the
weight-kernel positivity bounds a noiseless galaxy can never cross; noise-dominated
templates do.  The point-source bound (Mr/Mf floor) is NOT one of them, and neither is
Mc < 0 (user, 2026-10-07: that alone would cull ~8% of templates).
Per template: if >= FRAC of its copies are unphysical, ALL its copies go (the template is
noise-dominated; its copy means can leave the chart's domain -- row 172291 of the tn15 set
had copy-mean Mr = -17 and set the spin-2 loss scale 10-100x too large); otherwise only
the unphysical copies go.  GALAXIES / POPULATION are copied unchanged, so a culled template
simply has no copies (weighted_copy_mean's keep mask drops it from the centroid stage).
Usage: python dev/cull_copies.py IN.fits OUT.fits [FRAC=0.01]
"""
import sys
import numpy as np, fitsio

CHUNK = 10_000_000
SKIP = ("XTENSION", "BITPIX", "NAXIS", "PCOUNT", "GCOUNT", "TFIELDS", "TTYPE", "TFORM", "TDIM", "EXTNAME")


def unphysical(m):
    m = m.astype(np.float64)
    return (m[:, 0] < 0) | (m[:, 1] < 0) | (np.hypot(m[:, 2], m[:, 3]) > np.abs(m[:, 1]))


def header(h):
    out = fitsio.FITSHDR()
    for r in h.records():
        if r["name"] and not r["name"].startswith(SKIP):
            out.add_record(r)
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    with fitsio.FITS(src) as f:
        n_gal, n_cp = f["GALAXIES"].get_nrows(), f["COPIES"].get_nrows()
        tot, bad = np.zeros(n_gal), np.zeros(n_gal)
        why = np.zeros(4, np.int64)
        for a in range(0, n_cp, CHUNK):
            c = f["COPIES"].read(columns=["gal", "moments"], rows=np.arange(a, min(a + CHUNK, n_cp)))
            m = c["moments"].astype(np.float64)
            u = unphysical(m)
            why += [(m[:, 0] < 0).sum(), (m[:, 1] < 0).sum(), (m[:, 4] < 0).sum(),
                    (np.hypot(m[:, 2], m[:, 3]) > np.abs(m[:, 1])).sum()]   # Mc<0 reported, not culled
            tot += np.bincount(c["gal"], minlength=n_gal)
            bad += np.bincount(c["gal"], u, minlength=n_gal)
        f_bad = np.divide(bad, tot, out=np.zeros(n_gal), where=tot > 0)
        drop_tpl = f_bad >= frac
        print(f"unphysical copies: {int(bad.sum())} of {n_cp} ({bad.sum() / n_cp:.3%}); "
              f"Mf<0 {why[0]}, Mr<0 {why[1]}, |e|>1 {why[3]} (overlapping); Mc<0 {why[2]} not culled")
        print(f"templates with any unphysical copy {(bad > 0).sum()}; with >= {frac:.1%}: {drop_tpl.sum()} "
              f"of {n_gal} -> all their copies dropped ({tot[drop_tpl].sum():.0f} copies)")
        print(f"bad-fraction quantiles among templates with any: "
              f"p50 {np.median(f_bad[bad > 0]):.4f} p90 {np.percentile(f_bad[bad > 0], 90):.4f} max {f_bad.max():.4f}")
        kept = 0
        with fitsio.FITS(dst, "rw", clobber=True) as g:
            for a in range(0, n_cp, CHUNK):
                c = f["COPIES"].read(rows=np.arange(a, min(a + CHUNK, n_cp)))
                c = c[~(unphysical(c["moments"]) | drop_tpl[c["gal"]])]
                kept += len(c)
                if a == 0:
                    h = header(f["COPIES"].read_header())
                    g.write(c, extname="COPIES", header=h)
                else:
                    g["COPIES"].append(c)
            g["COPIES"].write_keys([{"name": "CULLFRAC", "value": frac, "comment": "template dropped if >= this frac unphysical"},
                                    {"name": "CULLTPL", "value": int(drop_tpl.sum()), "comment": "templates with all copies dropped"},
                                    {"name": "CULLCPY", "value": int(n_cp - kept), "comment": "copies dropped in total"}])
            for ext in ("GALAXIES", "POPULATION"):
                g.write(f[ext].read(), extname=ext, header=header(f[ext].read_header()))
    print(f"wrote {dst}: {kept} copies kept, {n_cp - kept} dropped")
