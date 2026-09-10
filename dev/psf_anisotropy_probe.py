"""Does an ELLIPTICAL PSF make the latent moment distribution anisotropic?

It decides whether the chart has to change.  `RawMomentStandardize._effective`
forces the spin-2 mean to zero and one shared spin-2 scale, which is only
legal if the population the flow models is isotropic.

BFD's moments are PSF-CORRECTED (`momentcalc.simpleImage`):

    kval /= kpsf                                # signal: PSF divided out
    kvar /= kpsf.real**2 + kpsf.imag**2         # noise:  amplified by 1/|T|^2

so with `I_obs(k) = T(k) I_true(k) + n(k)`, the signal part of a moment is

    M_i = INT d2k W(k) [I_obs/T] f_i(k) = INT d2k W(k) I_true(k) f_i(k)

with NO `T` in it, while the noise part keeps `1/|T|^2`.  If that holds, an
elliptical PSF leaves `p(m)` alone and moves only `C_M` and `Sigma_X`.

Measured here rather than argued: the same galaxy is convolved with a circular
and with an elliptical PSF (equal area), measured through its OWN PSF model,
and the two moment vectors compared.

    python -u dev/psf_anisotropy_probe.py
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "../bfd_cnf_imsims")

import bfd
from imsims import sim


def psf_cov(e, sigma=sim.PSF_SIGMA):
    """Elliptical PSF covariance at fixed AREA, in pixel^2.

    Fixed determinant so `e` changes the shape and nothing else -- otherwise a
    moment shift could just be the PSF getting bigger.
    """
    m = np.array([[1.0 + e, 0.0], [0.0, 1.0 - e]]) / np.sqrt(1.0 - e * e)
    return sigma ** 2 * m / sim.PIXEL_SCALE ** 2


def draw(g, cpsf, n=sim.STAMP_N):
    """`sim.draw_bulge_disc` with the PSF covariance made an argument."""
    ctr = (n / 2, n / 2)
    parts = [(g["flux"] * (1.0 - g["bulge_frac"]),
              sim._cov_gal(g["sigma"], g["e1"], g["e2"])),
             (g["flux"] * g["bulge_frac"],
              sim._cov_gal(g["sigma"] * g["bulge_ratio"],
                           g["bulge_e1"], g["bulge_e2"]))]
    im = np.zeros((n, n))
    for flux, cov_gal in parts:
        im += bfd.drawGauss(np.asarray(cov_gal) / sim.PIXEL_SCALE ** 2 + cpsf,
                            shape=(n, n), ctr=ctr, flux=flux)
    return im


def measure(g, e, wt, n=sim.STAMP_N):
    """(even moments, even covariance, odd covariance) through a PSF of
    ellipticity `e` -- the galaxy is convolved with it and deconvolved by it,
    which is what a correct PSF model in real data means."""
    cpsf = psf_cov(e)
    ctr = (n / 2, n / 2)
    im = draw(g, cpsf)
    psf = bfd.drawGauss(cpsf, shape=(n, n), ctr=ctr)
    kd = bfd.simpleImage(im, ctr, psf, pixel_scale=sim.PIXEL_SCALE,
                         pixel_noise=sim.NOISE_SIGMA, pad_factor=sim.PAD_FACTOR)
    mc = bfd.MomentCalculator(kd, wt)
    mc.xyshift, mc.badcenter, _ = mc.recenter()
    c = mc.getCovariance()
    return (np.asarray(mc.getMoment(0.0, 0.0).even, dtype=float),
            np.asarray(c.even, dtype=float), np.asarray(c.odd, dtype=float))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=6, help="galaxies to test")
    p.add_argument("--e", type=float, nargs="*", default=[0.0, 0.05, 0.10])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    wt = bfd.KBlackmanHarris(weightSigma=sim.WEIGHT_SIGMA)
    rows = sim.sample_population(a.n, np.random.default_rng(a.seed))

    print(f"PSF sigma {sim.PSF_SIGMA}\" at fixed area, weight sigma "
          f"{sim.WEIGHT_SIGMA}\"\n{a.n} bulge+disc galaxies\n")

    base = [measure(g, 0.0, wt) for g in rows]
    for e in a.e:
        out = [measure(g, e, wt) for g in rows]
        dm = np.array([np.abs(o[0] - b[0]) / np.abs(b[0]).max()
                       for o, b in zip(out, base)])
        ce = np.mean([o[1] for o in out], axis=0)
        co = np.mean([o[2] for o in out], axis=0)
        s2 = 2 * (ce[2, 2] - ce[3, 3]) / (ce[2, 2] + ce[3, 3])
        sx = 2 * (co[0, 0] - co[1, 1]) / (co[0, 0] + co[1, 1])
        print(f"e_psf = {e:.2f}")
        print(f"  moments:  max |dM|/|M|max over galaxies = {dm.max():.3e}"
              f"   (median {np.median(dm):.3e})")
        print(f"  C_M:      spin-2 split (C11-C22)/mean = {s2:+.2%}"
              f"     Cov(Mf,M1)/sqrt(CffC11) = "
              f"{ce[0, 2] / np.sqrt(ce[0, 0] * ce[2, 2]):+.3f}")
        print(f"  Sigma_X:  split (C00-C11)/mean = {sx:+.2%}\n")

    print("If |dM|/|M| stays at roundoff the LATENT moments are PSF-free and\n"
          "the chart's forced spin-2 mean is correct as it stands; only C_M\n"
          "and Sigma_X carry the PSF's ellipticity.")


if __name__ == "__main__":
    main()
