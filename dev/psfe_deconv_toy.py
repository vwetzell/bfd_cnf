"""Thread 1 closure plan, Step 3 (lightweight version, user's call 2026-09-15):
does BFD's own deconvolution/weight-function truncation leak PSF ellipticity
into a galaxy's moments even for a PERFECTLY ROUND, noiseless, centred galaxy?

No noise, no centroid solve (`recenter=False`, known centre), no population,
no MC, no flow -- just `imsims.sim.psf_cov`/`_measure`, the exact same
render+deconvolve path every other catalog in this repo goes through. A
round galaxy (e1=e2=0) has no shape of its own to confuse with a PSF-driven
signal: if the deconvolution + KBlackmanHarris weight were exact for an
elliptical PSF, M1/M2 (the spin-2 moments) would come out EXACTLY zero at
every psf_e. Any nonzero M1/M2 here is unambiguously a property of BFD's own
weight-truncated deconvolution algebra under an anisotropic PSF -- Step 3's
"is this baked into the base bfd package" question, answered directly rather
than via a from-scratch closed-form derivation.

Usage: python dev/psfe_deconv_toy.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "bfd_cnf_imsims"))

import numpy as np

import bfd
from imsims import sim

SIGMA = 2.0     # galaxy size in pixels, well-resolved against PSF_SIGMA/pixel
FLUX = 1.0e5
AMPS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]


def measure_round(psf_e1, psf_e2):
    sim.PSF_E = (psf_e1, psf_e2)
    n = sim.STAMP_N
    ctr = (n / 2, n / 2)
    cov_gal = SIGMA**2 * np.eye(2)                 # perfectly round
    cov_psf = sim.psf_cov()
    gal = bfd.drawGauss(cov_gal / sim.PIXEL_SCALE**2 + cov_psf,
                        shape=(n, n), ctr=ctr, flux=FLUX)
    wt = bfd.KBlackmanHarris(weightSigma=sim.WEIGHT_SIGMA)
    mc = sim._measure(gal, wt, noise_sigma=sim.NOISE_SIGMA, n=n, recenter=False)
    m = np.asarray(mc.getMoment(0.0, 0.0).even, dtype=float)
    return m  # [Mf, Mr, M1, M2, Mc]


if __name__ == "__main__":
    print(f"{'psf_e1':>8s}{'psf_e2':>8s}{'Mf':>12s}{'Mr':>10s}"
          f"{'M1':>12s}{'M2':>12s}{'Mc':>10s}")
    rows = []
    for e in AMPS:
        m = measure_round(e, 0.0)
        rows.append((e, 0.0, m))
        print(f"{e:8.3f}{0.0:8.3f}{m[0]:12.4f}{m[1]:10.4f}"
              f"{m[2]:12.6f}{m[3]:12.6f}{m[4]:10.4f}")
    for e in AMPS[1:]:
        m = measure_round(0.0, e)
        rows.append((0.0, e, m))
        print(f"{0.0:8.3f}{e:8.3f}{m[0]:12.4f}{m[1]:10.4f}"
              f"{m[2]:12.6f}{m[3]:12.6f}{m[4]:10.4f}")

    e1s = np.array([r[0] for r in rows if r[1] == 0.0 and r[0] != 0.0])
    m1s = np.array([r[2][2] for r in rows if r[1] == 0.0 and r[0] != 0.0])
    e2s = np.array([r[1] for r in rows if r[0] == 0.0 and r[1] != 0.0])
    m2s = np.array([r[2][3] for r in rows if r[0] == 0.0 and r[1] != 0.0])
    sl = lambda y, x: (y * x).sum() / (x * x).sum()
    print(f"\ndM1/d(psf_e1) = {sl(m1s, e1s):+.6f}   "
          f"dM2/d(psf_e2) = {sl(m2s, e2s):+.6f}")
