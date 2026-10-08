"""Sim depth vs DES Y6: S/N, sigma_f and Mr/Mf quantiles.  (2026-10-05)  CPU, numpy only.

DES targets  = ~/Documents/BFD_cNF/merged_masked_bfd_grid_p.npy (every row lies in
DES's window Mf 1500-90000, Mr/Mf 2.2-3.5; `select` is a flux-bin index, not pass/fail),
DES templates = summary_templates_new.fits (deep field, uncut).  Sim catalogs named on the
command line are classed as targets or templates by their file name ("moments_" =
templates) and cut to the window below (targets) or left uncut (templates).
The sim's population flux is calibrated in DES Mf units, so absolute sigma_f matching
IS S/N matching: target noise 2.289 -> sigma_f 228, template noise 0.611 -> 60.9.
Usage: python dev/des_match.py [sim.fits ...]
"""
import os, sys
import numpy as np, fitsio, bfd

DES = os.path.expanduser("~/Documents/BFD_cNF/")
FLUX, SIZE = (2200.0, 90000.0), (2.2, 3.5)   # 2200 = DES's effective (detection) lower edge
P = (1, 5, 16, 50, 84, 95, 99)


def stats(m, cov_packed, window):
    m = np.asarray(m, np.float64)
    vf = bfd.MomentCovariance.bulkUnpack(np.asarray(cov_packed, np.float64))[:, 0, 0]
    r = m[:, 1] / m[:, 0]
    k = (vf > 0) & np.isfinite(r)
    if window:
        k &= (m[:, 0] > FLUX[0]) & (m[:, 0] < FLUX[1]) & (r > SIZE[0]) & (r < SIZE[1])
    sf = np.sqrt(vf[k])
    return dict(n=int(k.sum()), frac=k.mean(), sn=np.percentile(m[k, 0] / sf, P),
                sf=np.percentile(sf, P), r=np.percentile(r[k], (16, 50, 84)))


def show(lab, s, ref=None):
    print(f"{lab:44s} n={s['n']:8d} ({s['frac']:.3f})\n"
          f"    S/N  p{P}: {np.round(s['sn'], 1)}\n"
          f"    sigF p{P}: {np.round(s['sf'], 1)}\n"
          f"    Mr/Mf p16/50/84: {np.round(s['r'], 2)}")
    if ref is not None:
        print(f"    vs DES: S/N ratio {np.round(s['sn'] / ref['sn'], 3)}  sigF ratio {np.round(s['sf'] / ref['sf'], 3)}")


g = np.load(DES + "merged_masked_bfd_grid_p.npy", mmap_mode="r")
tgt = stats(g["moments"], g["covariance"], True)
t = fitsio.read(DES + "summary_templates_new.fits", columns=["moments", "covariance"])
tmp = stats(t["moments"], t["covariance"], False)
show(f"DES targets (window {FLUX}, {SIZE})", tgt)
show("DES templates (uncut)", tmp)
for f in sys.argv[1:]:
    d = fitsio.read(f, columns=["moments", "cov"])
    is_tmpl = os.path.basename(f).startswith("moments_")
    show(os.path.basename(f) + (" [template]" if is_tmpl else " [target]"),
         stats(d["moments"], d["cov"], not is_tmpl), tmp if is_tmpl else tgt)
