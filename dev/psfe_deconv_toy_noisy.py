"""Follow-up to dev/psfe_deconv_toy.py: same perfectly-round galaxy, but now
WITH real pixel noise and a genuine centroid solve (`recenter=True`), across
many independent noise realizations -- does the RECOVERED M1/M2, averaged
over noise, come out nonzero under an anisotropic PSF?

The noiseless/known-centre toy (dev/psfe_deconv_toy.py) found EXACTLY zero
M1/M2 leak at every psf_e up to 0.20 -- BFD's static deconvolution is exact
for a round, centred galaxy. But that toy cannot engage Sigma_X/centroid
marginalisation at all (no noise, no recenter), and Step 1
(dev/psfe_no_flow_check.py) found the leak DOES reproduce through the
Sigma_X-reweighted flow-free machinery. This script closes the gap: same
round galaxy, but with `_row_noise`-style pixel noise and `mc.recenter()`
actually solving X=0, so a genuine (noisy) centroid error and a genuine
per-target Sigma_X exist. If mean(M1)/mean(M2) over many noise draws comes
out nonzero, that pins the leak to the centroid-solve step itself -- no
galaxy asymmetry, no population, no flow, no importance sampling anywhere
in the loop.

Usage: N=4000 python dev/psfe_deconv_toy_noisy.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "bfd_cnf_imsims"))

import numpy as np

import bfd
from imsims import sim

SIGMA = 2.0
FLUX = 1.0e5
NOISE_SIGMA = float(os.environ.get("NOISE_SIGMA", sim.NOISE_SIGMA))
N = int(os.environ.get("N", 4000))
SEED = int(os.environ.get("SEED", 0))
AMPS = [0.0, 0.05, 0.10, 0.20]


def clean_image(psf_e1, psf_e2):
    sim.PSF_E = (psf_e1, psf_e2)
    n = sim.STAMP_N
    ctr = (n / 2, n / 2)
    cov_gal = SIGMA**2 * np.eye(2)
    cov_psf = sim.psf_cov()
    return bfd.drawGauss(cov_gal / sim.PIXEL_SCALE**2 + cov_psf,
                         shape=(n, n), ctr=ctr, flux=FLUX)


def run(psf_e1, psf_e2, noise_draws, wt):
    """`noise_draws` (n_draws, STAMP_N, STAMP_N) is SHARED across every
    config call -- same pixel noise field, only the clean (PSF-convolved)
    image differs -- so the per-draw shape-noise scatter (tens of units on
    M1/M2 for a single realization) cancels in the cross-config comparison
    almost entirely, the same antithetic-pairing trick every other psfe grid
    in this repo relies on."""
    gal0 = clean_image(psf_e1, psf_e2)
    m1s, m2s, bad = [], [], 0
    for noise in noise_draws:
        img = gal0 + noise
        mc = sim._measure(img, wt, noise_sigma=NOISE_SIGMA, n=sim.STAMP_N,
                          recenter=True)
        if mc.badcenter:
            bad += 1
            m1s.append(np.nan)
            m2s.append(np.nan)
            continue
        m = np.asarray(mc.getMoment(0.0, 0.0).even, dtype=float)
        m1s.append(m[2])
        m2s.append(m[3])
    return np.asarray(m1s), np.asarray(m2s), bad


if __name__ == "__main__":
    print(f"noise_sigma={NOISE_SIGMA}  N={N} PAIRED draws/config "
          f"(same noise field reused across every psf_e)\n")
    rng = np.random.default_rng(SEED)
    noise_draws = rng.normal(0.0, NOISE_SIGMA, (N, sim.STAMP_N, sim.STAMP_N))
    wt = bfd.KBlackmanHarris(weightSigma=sim.WEIGHT_SIGMA)

    base_m1, base_m2, base_bad = run(0.0, 0.0, noise_draws, wt)
    print(f"{'psf_e1':>8s}{'psf_e2':>8s}{'dM1':>12s}{'+/-':>10s}"
          f"{'dM2':>12s}{'+/-':>10s}{'bad':>6s}")
    rows = []
    for e in AMPS[1:]:
        m1, m2, bad = run(e, 0.0, noise_draws, wt)
        d1, d2 = m1 - base_m1, m2 - base_m2
        ok = np.isfinite(d1) & np.isfinite(d2)
        dm1, dm1e = np.nanmean(d1), np.nanstd(d1[ok], ddof=1) / np.sqrt(ok.sum())
        dm2, dm2e = np.nanmean(d2), np.nanstd(d2[ok], ddof=1) / np.sqrt(ok.sum())
        rows.append(("e1", e, dm1, dm1e, dm2, dm2e))
        print(f"{e:8.3f}{0.0:8.3f}{dm1:12.5f}{dm1e:10.5f}"
              f"{dm2:12.5f}{dm2e:10.5f}{bad + base_bad:6d}", flush=True)
    for e in AMPS[1:]:
        m1, m2, bad = run(0.0, e, noise_draws, wt)
        d1, d2 = m1 - base_m1, m2 - base_m2
        ok = np.isfinite(d1) & np.isfinite(d2)
        dm1, dm1e = np.nanmean(d1), np.nanstd(d1[ok], ddof=1) / np.sqrt(ok.sum())
        dm2, dm2e = np.nanmean(d2), np.nanstd(d2[ok], ddof=1) / np.sqrt(ok.sum())
        rows.append(("e2", e, dm1, dm1e, dm2, dm2e))
        print(f"{0.0:8.3f}{e:8.3f}{dm1:12.5f}{dm1e:10.5f}"
              f"{dm2:12.5f}{dm2e:10.5f}{bad + base_bad:6d}", flush=True)

    e1 = np.array([e for ax, e, *_ in rows if ax == "e1"])
    dm1 = np.array([r[2] for r in rows if r[0] == "e1"])
    dm1e = np.array([r[3] for r in rows if r[0] == "e1"])
    e2 = np.array([e for ax, e, *_ in rows if ax == "e2"])
    dm2 = np.array([r[4] for r in rows if r[0] == "e2"])
    dm2e = np.array([r[5] for r in rows if r[0] == "e2"])
    w1 = 1.0 / dm1e**2
    s1 = (w1 * dm1 * e1).sum() / (w1 * e1**2).sum()
    s1e = 1.0 / np.sqrt((w1 * e1**2).sum())
    w2 = 1.0 / dm2e**2
    s2 = (w2 * dm2 * e2).sum() / (w2 * e2**2).sum()
    s2e = 1.0 / np.sqrt((w2 * e2**2).sum())
    print(f"\ndM1/d(psf_e1) = {s1:+.5f} +/- {s1e:.5f} ({s1 / s1e:+.1f} sigma)")
    print(f"dM2/d(psf_e2) = {s2:+.5f} +/- {s2e:.5f} ({s2 / s2e:+.1f} sigma)")
