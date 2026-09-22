"""Extends dev/psf_anisotropy_probe.py's "gate G0" check to NONZERO shift.

Gate G0 (b93e27b) confirmed M^G(u=0) -- the moment at the RECENTERED point --
is PSF-independent bit-for-bit, exactly matching the paper's own claim (BFD
2015 section 2.1: "M^G is independent of the observational conditions...
corrected for the PSF, so we do not need to recalculate M^G as the PSF
varies"). That licensed reusing one fixed-PSF (psf_e=0) copies/template
catalog for every psf_e config throughout this codebase.

But eq. (35)-(36)'s centroid marginalisation sums M(u) over a GRID of
shifted positions, u != 0 -- and the paper's own PSF-independence claim was
never stated for u != 0. This script tests it directly: the SAME galaxy,
shifted to the SAME grid of u offsets, once through a circular PSF and once
through an elliptical PSF (equal area) -- do the shifted moments M(u) agree?

    python -u dev/psf_anisotropy_shift_probe.py
"""
import sys

import numpy as np

sys.path.insert(0, "../bfd_cnf_imsims")

import bfd
from imsims import sim

E_PSF = [0.0, 0.05, 0.10, 0.20]
SHIFTS = [(0.0, 0.0), (0.05, 0.0), (0.0, 0.05), (0.05, 0.05), (0.1, 0.0), (0.0, 0.1)]


def psf_cov(e, sigma=sim.PSF_SIGMA):
    m = np.array([[1.0 + e, 0.0], [0.0, 1.0 - e]]) / np.sqrt(1.0 - e * e)
    return sigma**2 * m / sim.PIXEL_SCALE**2


def draw(g, cpsf, n=sim.STAMP_N):
    ctr = (n / 2, n / 2)
    parts = [(g["flux"] * (1.0 - g["bulge_frac"]),
              sim._cov_gal(g["sigma"], g["e1"], g["e2"])),
             (g["flux"] * g["bulge_frac"],
              sim._cov_gal(g["sigma"] * g["bulge_ratio"],
                           g["bulge_e1"], g["bulge_e2"]))]
    im = np.zeros((n, n))
    for flux, cov_gal in parts:
        im += bfd.drawGauss(np.asarray(cov_gal) / sim.PIXEL_SCALE**2 + cpsf,
                            shape=(n, n), ctr=ctr, flux=flux)
    return im


def moments_at_shifts(g, e, wt, shifts, n=sim.STAMP_N):
    """M(u) at each requested shift u, through a PSF of ellipticity e --
    NO recentring (we choose u ourselves), matching the copy-grid's own
    `mc.getMoment(xy[0], xy[1])` usage in `bfd.MomentCalculator.makeTemplates`."""
    cpsf = psf_cov(e)
    ctr = (n / 2, n / 2)
    im = draw(g, cpsf)
    psf = bfd.drawGauss(cpsf, shape=(n, n), ctr=ctr)
    kd = bfd.simpleImage(im, ctr, psf, pixel_scale=sim.PIXEL_SCALE,
                         pixel_noise=sim.NOISE_SIGMA, pad_factor=sim.PAD_FACTOR)
    mc = bfd.MomentCalculator(kd, wt)
    # Recentre once to find the TRUE detection point (matches makeTemplates'
    # own xy_offset convention: shifts are relative to the recentred origin).
    mc.xyshift, mc.badcenter, _ = mc.recenter()
    x0, y0 = mc.xyshift
    return np.array([np.asarray(mc.getMoment(x0 + dx, y0 + dy).even, dtype=float)
                     for dx, dy in shifts])


if __name__ == "__main__":
    wt = bfd.KBlackmanHarris(weightSigma=sim.WEIGHT_SIGMA)
    rows = sim.sample_population(6, np.random.default_rng(0))

    base = [moments_at_shifts(g, 0.0, wt, SHIFTS) for g in rows]
    print(f"{len(rows)} galaxies, shifts {SHIFTS}\n")
    for e in E_PSF[1:]:
        out = [moments_at_shifts(g, e, wt, SHIFTS) for g in rows]
        # relative moment difference at EACH shift, vs the circular-PSF baseline
        for si, (dx, dy) in enumerate(SHIFTS):
            d = np.array([np.abs(o[si] - b[si]) / np.abs(b[0]).max()
                         for o, b in zip(out, base)])
            print(f"  e_psf={e:.2f}  u=({dx:+.2f},{dy:+.2f})  "
                  f"max|dM|/|M(0)|max = {d.max():.3e}  (median {np.median(d):.3e})")
        print()

    print("If |dM(u)|/|M(0)| stays at roundoff for u=0 but GROWS with |u| at\n"
          "nonzero e_psf, the shift-grid curvature d^2M/du^2 depends on PSF\n"
          "ellipticity -- meaning reusing a psf_e=0 copies pool for every psf_e\n"
          "config discards exactly the term that would need to be there to\n"
          "cancel the Sigma_X-driven leak, the way the paper's OWN matched-PSF\n"
          "validation test (templates AND targets under the same elliptical\n"
          "PSF) implicitly did.")
