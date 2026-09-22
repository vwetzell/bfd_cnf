"""Does this repo's PSFE leak collapse under the paper's own operating point?

The reference paper (arXiv:1508.05655, BFD) never sees the additive-shear
leak this repo found at its own default operating point.  Its own SS4.2 test
differs from this repo's default in three knobs at once: a narrower
ellipticity prior (`sigma_e=0.2` in eq. 58, vs. this repo's wide/broad
default), a S/N cut of 8-20, and a wider weight-function/PSF ratio (paper
~2.75, this repo's default 0.65).  This script reproduces all three at once
on the gauss2_fwd population and checks whether the debiased M1 intercept
against psf_e1 collapses relative to the raw (uncut, un-debiased) number --
an order-of-magnitude / qualitative check, not a literal reproduction of the
paper's own `c` (which is a shear-bias units, not M1 units).

No trained flow, no bias.py, no saved catalog -- pure numpy/bfd against
imsims.sim's own rendering, same cost class as psfe_deconv_toy_noisy.py.

Usage: N=300 python dev/psfe_paper_operating_point.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "bfd_cnf_imsims"))

import numpy as np

import bfd
from bfd.moment import Moment
from imsims import sim

N = int(os.environ.get("N", 300))
SEED = int(os.environ.get("SEED", 0))
SIGMA_E = float(os.environ.get("SIGMA_E", 0.2))    # paper's eq. 58, Table 1
PSF_E1 = float(os.environ.get("PSF_E1", 0.05))
WEIGHT_SIGMA = float(os.environ.get("WEIGHT_SIGMA", 1.5))
SN_LO, SN_HI = 8.0, 20.0


def render_config(psf_e1, pop, noise_draws, wt):
    """Render + measure `pop` at PSF anisotropy `psf_e1`, reusing the SAME
    per-galaxy noise draws across configs (antithetic pairing that cancels
    per-galaxy shape-noise scatter in the config-to-config difference, the
    same trick psfe_deconv_toy_noisy.py relies on)."""
    sim.PSF_E = (psf_e1, 0.0)
    n = len(pop)
    m1 = np.full(n, np.nan)
    sn = np.full(n, np.nan)
    bad = 0
    for j in range(n):
        img_clean = sim.draw_bulge_disc(pop[j], shear=(0.0, 0.0))
        mc = sim._measure(img_clean + noise_draws[j], wt,
                          noise_sigma=sim.NOISE_SIGMA, n=sim.STAMP_N,
                          recenter=True)
        if mc.badcenter:
            bad += 1
            continue
        m = np.asarray(mc.getMoment(0.0, 0.0).even, dtype=float)
        m1[j] = m[Moment.M1]
        cov = mc.getCovariance()
        sn[j] = m[Moment.M0] / np.sqrt(cov.even[Moment.M0, Moment.M0])
    return m1, sn, bad


if __name__ == "__main__":
    # bfd.Moment has no MF alias -- the flux moment is M0 (checked here rather
    # than hardcoding indices without confirmation, per the resolved values).
    assert Moment.M0 == 0 and Moment.M1 == 2, "Moment index layout changed"

    print(f"N={N} SEED={SEED} SIGMA_E={SIGMA_E} PSF_E1={PSF_E1} "
          f"WEIGHT_SIGMA={WEIGHT_SIGMA} (paper operating point: narrow "
          f"ellipticity prior + wide weight/PSF ratio + S/N cut "
          f"{SN_LO}-{SN_HI})\n")

    sim.WEIGHT_SIGMA = WEIGHT_SIGMA
    wt = bfd.KBlackmanHarris(weightSigma=sim.WEIGHT_SIGMA)

    rng = np.random.default_rng(SEED)
    pop = sim.sample_population_gauss2_fwd(
        N, rng, sim.GAUSS2_FWD_SIZE_LOGMEDIAN, sim.GAUSS2_FWD_ELLIP_SIGMA)
    # Paper's own eq. 58 ellipticity prior, sigma_e=0.2 (Table 1) -- overwrite
    # the wide default the population sampler hardcodes.  Separate rng draw
    # (not reused from the flux/sigma draw above) so the two stay independent.
    pop["e1"], pop["e2"] = sim._ellipticity(rng, N, sigma_e=SIGMA_E)
    # gauss2 shares one ellipticity across bulge+disc by construction
    # (draw_bulge_disc uses e1/e2 for both components' shapes) -- bulge_e1/
    # bulge_e2 must be overwritten to match, or the two components end up
    # with different ellipticities, which is a different (uncontrolled)
    # population than gauss2_fwd.
    pop["bulge_e1"], pop["bulge_e2"] = pop["e1"], pop["e2"]

    noise_draws = rng.normal(0.0, sim.NOISE_SIGMA, (N, sim.STAMP_N, sim.STAMP_N))

    m1_base, sn_base, bad_base = render_config(0.0, pop, noise_draws, wt)
    m1_psf, _, bad_psf = render_config(PSF_E1, pop, noise_draws, wt)

    resid1 = m1_psf - m1_base
    ok_all = np.isfinite(resid1)
    raw_mean = np.nanmean(resid1[ok_all])
    raw_sem = np.nanstd(resid1[ok_all], ddof=1) / np.sqrt(ok_all.sum())

    sn_cut = (sn_base > SN_LO) & (sn_base < SN_HI)
    sel = ok_all & sn_cut
    e1_gal = pop["e1"][sel]
    y = resid1[sel]
    coeffs, cov_fit = np.polyfit(e1_gal, y, 1, cov=True)
    b, a = coeffs[0], coeffs[1]
    a_err = np.sqrt(cov_fit[1, 1])

    print(f"bad centroids: baseline={bad_base}  psf_e1={PSF_E1}={bad_psf}")
    print(f"converged (either config): {ok_all.sum()} / {N}")
    print(f"S/N-cut survivors ({SN_LO} < S/N < {SN_HI}): {sel.sum()}\n")

    print(f"BEFORE (uncut, un-debiased): mean(resid1) = "
          f"{raw_mean:+.5f} +/- {raw_sem:.5f}  (N={ok_all.sum()})")
    print(f"AFTER  (S/N-cut, e1-debiased intercept a): a = "
          f"{a:+.5f} +/- {a_err:.5f}  (N={sel.sum()}, slope b={b:+.5f})\n")

    print("Paper comparison (order-of-magnitude only -- these are M1 moment "
          "units, not the paper's shear-bias c; paper reports "
          "c = (-1.3 +/- 0.9)e-5 at the psf_e amplitudes it tested): "
          f"did the intercept collapse relative to the raw number? "
          f"|a|/|raw_mean| = "
          f"{(abs(a) / abs(raw_mean)) if raw_mean != 0 else float('nan'):.3f}")
