"""
bias.py
=======
Multiplicative and additive shear bias of a trained flow, measured on the imsims
targets -- noiseless, or with a noise realization integrated over.

The BFD ensemble estimator (Bernstein & Armstrong 2014 sec. 3) maximises

    log L(g) = sum_i log P(M_i | g)

over g, expanded to second order about g = 0,

    ghat = -[ sum_i d2 logP_i/dg2 ]^-1 [ sum_i d logP_i/dg ].

Those two derivatives ARE BFD's PQR combinations -- d logP/dg = Q/P and
d2 logP/dg2 = R/P - Q Q^T/P^2 (paper eq. 12-13, 45-46) -- so differentiating
`log P` with respect to g gives them without ever forming P, Q and R
separately, and `ghat` above is the paper's eq. (19).

What `log P(M_i|g)` is depends on the target's noise:

* **noiseless targets** (`--samples 0`): the noise kernel is a delta and the
  integral collapses to a point evaluation of the prior at the target's own
  moments -- no sampling, no weights, no ESS.
There are two kinds of noisy target, and the catalog's `IMGNOISE` header says
which:

* **noise added in moment space** (`add_noise`, the older catalogs).  Exact only
  at a KNOWN centre, where the moments are a linear functional of the image.
* **noise added in the image**, then measured with a real `recenter()` (the
  `*_noisy_*` catalogs).  Here the centroid is FOUND, so the moments carry a
  centroid error and the prior must marginalise over it -- which is the centroid
  layer, switched on automatically for such a catalog.  `add_noise` is not
  applied to these: the noise is already in them, in the only place it can be
  and still let the detection respond to it.

* **noisy targets** (`--samples S`): the target carries M = M^G + M^n with
  M^n ~ N(0, C_M) (paper eq. 5), and P is the prior convolved with that noise,

      P(M_i | g) = INT dm P(m | g) L(M_i - m),   L = N(., C_M)

  which is the continuum limit of the paper's template sum, eq. (38).  It is
  estimated by importance sampling S draws from a proposal (`log_conv_is`):
  the pure noise kernel (`log_conv`, `mixture_draws` at alpha = 1, the default)
  or a defensive mixture of the kernel and the prior (`mixture_draws`,
  `--alpha < 1`).

  WHICH PROPOSAL WINS DEPENDS ON DEPTH, and the earlier blanket verdict against
  prior-aware proposals held only where the kernel was already adequate:

      noise_sigma      alpha = 1            alpha = 0.5
      1.0              median ESS 129       median ESS 77        (mixture worse)
      2.73             median ESS 25        median ESS 103       (mixture better)
                       frac(ESS<10) 0.30    frac(ESS<10) 0.013

  At noise_sigma 2.73 the kernel is genuinely starved -- C_M is 7.45x wider in
  variance, so draws centred on M_i rarely land on the prior's support -- and
  that is exactly the regime `mixture_draws` was written for.  The tail is what
  moves, not just the median.

  Two caveats before reaching for it.  It costs ~15x per draw (it samples from
  the flow, running the shear Newton solve and the centroid fixed point), so
  brute-force alpha = 1 at 16x the draws is ~3x cheaper for the same ESS.  And
  alpha < 1 draws part of the proposal FROM THE FLOW, so two runs with different
  flows no longer share random numbers and `--compare`'s pairing weakens.

  A base-space Laplace/Student-t proposal was tried and was worse than either
  (frac(ESS < 10) 0.52 against the kernel's 0.09, unrescued by widening, tail
  weight or mode-quality); it has been removed.

Bias comes from the +g/-g pair, which is the SAME galaxies sheared both ways
(same seed, and the same noise realization), so the paired difference carries
no shape noise:

    m1 = (ghat1[+] - ghat1[-]) / (2 g) - 1        c = (ghat[+] + ghat[-]) / 2

Only m1 is measurable here: the catalogs are sheared along g1 only, so there is
no lever arm on m2.  The unsheared catalog gives c independently.

No selection is applied -- every target is used, at any flux -- so P(s|g) = 1
and the non-detection terms of paper eq. (45)-(46) are absent.  Cut on a noisy
flux and they stop being absent.

Usage:
    python bias.py --flow flows/shear.eqx
    python bias.py --flow flows/shear.eqx --samples 1024 --n-targets 100000
    # image noise + recentring; the centroid layer switches on from the header
    python bias.py --flow flows/centroid.eqx --pop bulgedisc_noisy --samples 2048
    # the same targets without it -- what the marginalisation is worth
    python bias.py --flow flows/shear.eqx --no-centroid --pop bulgedisc_noisy --samples 2048
"""

from __future__ import annotations

import argparse
import os
import time

import bfd
import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from flowjax.bijections import Chain, Invert
from flowjax.distributions import Transformed

import bulk
import shear
# `safe_point` lives beside `in_domain` in models/bijections.py: the centroid
# TRAINING loss needs the identical guard, and the last bug here was one copy
# of it not learning about a new ceiling.
from models.bijections import in_domain, safe_point, SigmaXBlockLayer
from models.centroid import CentroidMarginalize

# Both are located BY TYPE at bijections[1] when a centroid/Sigma_X layer is
# present: `CentroidMarginalize` (the Gaussian-ansatz layer) or `SigmaXBlockLayer`
# (the trained-coefficient layer that replaces it). Both implement the same
# AbstractBijection interface (.transform/.inverse/.transform_and_log_det), so
# every caller below that only calls through that interface needs no other change.
_CENTROID_LAYER_TYPES = (CentroidMarginalize, SigmaXBlockLayer)


# Full float32 matmuls, not the TF32 the GPU defaults to.  TF32 keeps 10
# mantissa bits, so a log-density that accumulates through a ~12-layer flow
# comes out with ~1 nat of error -- and XLA picks TF32 kernels by autotuning,
# which is a per-PROCESS decision, so two runs of the SAME code disagreed by
# 1.2 nats on draws carrying real weight.  That is fatal here twice over: the
# paired +g/-g comparison assumes two processes compute identical weights from
# identical draws, and 1 nat is a factor e in a weight.  Measured on this
# machine: cross-process max |dlog_wt| 1.2 -> 1.4e-3, for 7% throughput.
# Only the inference path needs this; training noise at TF32 is harmless.
jax.config.update("jax_default_matmul_precision", "highest")

# `BFD_TIMING=1` splits `pqr_streamed`'s inner loop into device and host time,
# separated by an explicit `jax.block_until_ready` so the blocking `np.asarray`
# cannot charge the GPU's time to the merge.  Off by default: the barrier
# serialises dispatch, so a timed run is slightly SLOWER than a real one and
# the split is what it is measuring, not the total.
#
# Written because `nvidia-smi --query-gpu=utilization.gpu` cannot answer the
# question it looks like it answers: it reports the fraction of time at least
# one kernel was RESIDENT, not that the SMs were busy, so a launch-bound loop
# of tiny kernels reads ~100% exactly like a saturated one.
_TIME = ({"draw": 0.0, "lambda": 0.0, "dispatch": 0.0, "device": 0.0,
          "merge": 0.0, "finish": 0.0, "wall": 0.0}
         if os.environ.get("BFD_TIMING") else None)

# (start_batch, stop_batch, outdir) from BFD_PROFILE="start:stop:dir".
_PROF = None
if os.environ.get("BFD_PROFILE"):
    _a, _b, _d = os.environ["BFD_PROFILE"].split(":")
    _PROF = (int(_a), int(_b), _d)

# Catalogs, as (label, filename stem), per population.  The +/- pair share a
# seed and so are the same galaxies; the g=0 run is the same galaxies again.
# The target selection window lives in bulk.py -- it cannot live here because
# shear.py needs it too and cannot import bias.py without a cycle -- and is
# re-exported here under its established name.
SIZE_WINDOW, FLUX_WINDOW = bulk.SIZE_WINDOW, bulk.FLUX_WINDOW

CATALOGS = {
    "bulgedisc": {"plus": "targets_g1p02_1M", "minus": "targets_g1m02_1M",
                  "zero": "targets_g0_1M"},
    "sersic": {"plus": "sersic_tg1p_20k", "minus": "sersic_tg1m_20k",
               "zero": "sersic_tg1z_20k"},
    # Pixel noise in the IMAGE and a real recenter(), so the centroid is found
    # rather than assumed.  These need the centroid layer -- and they must NOT
    # be handed to `add_noise`, which is only valid at a fixed centre.  The
    # IMGNOISE header says which kind a catalog is; `main` reads it rather than
    # trusting the name.
    "bulgedisc_noisy": {"plus": "targets_noisy_g1p02_200k",
                        "minus": "targets_noisy_g1m02_200k",
                        "zero": "targets_noisy_g0_200k"},
    # Same population at noise_sigma = 2.73, which puts the 5th percentile of
    # flux S/N at 10 and the median at 19.  That is where the centroid
    # marginalisation is worth +2e-3 to +1.4e-2 rather than the +3.5e-4 it is at
    # the shallower depth -- i.e. where a with/without comparison can resolve
    # it.  Needs flows trained on copies_bulgedisc_deep.fits: the copy grid is
    # only valid within a factor 1.4 in sigma_XY, and this is 2.73x.
    "bulgedisc_deep": {"plus": "targets_deep_g1p02_200k",
                       "minus": "targets_deep_g1m02_200k",
                       "zero": "targets_deep_g0_200k"},
    # Same recipe as bulgedisc_deep, rendered against the 2026-08-27 real-data
    # retune (see HANDOFF.md). "_deep"'s targets/flows predate the retune and
    # are kept as-is for comparison.
    "bulgedisc_deep_v2": {"plus": "targets_deep_g1p02_200k_v2",
                          "minus": "targets_deep_g1m02_200k_v2",
                          "zero": "targets_deep_g0_200k_v2"},
    # The 2026-09-02 clean rebuild -- `rebuild_v3.sh`, which is the only place
    # these were produced and the only record of how.  Same population as _v2,
    # re-rendered at the DERIVED depth noise_sigma = 0.93 (median flux S/N 19.2,
    # matching the real DES/COSMOS template catalog) with the prior on seed 0
    # and all three target arms on seed 1, so prior and targets are independent
    # draws by construction rather than by accident of differing --n.  Every
    # _v2 and earlier catalog is under data/archive_20260902/.
    "bulgedisc_v3": {"plus": "targets_v3_g1p02_200k",
                     "minus": "targets_v3_g1m02_200k",
                     "zero": "targets_v3_g0_200k"},
    # The same galaxies and noise field as bulgedisc_v3, rendered through a
    # FIXED ELLIPTICAL PSF at |e| = 0.2 (dev/render_psfe.sh) -- an orientation
    # triplet, because the signature of a PSF leak is that dc ROTATES WITH THE
    # PSF and a single config's c2 cannot be told from the scatter already
    # there.  Difference each against bulgedisc_v3, which cancels the known
    # c1 = -2.02e-03, c2 = -4.01e-03 and the population sample noise.
    #
    # These reuse the v3 FLOWS and the v3 PRIOR unchanged, and that is not a
    # shortcut: BFD's moments are PSF-corrected, so the latent moments are
    # PSF-independent bit for bit over 2000 galaxies at e_psf = 0.2 (gate G0).
    # The PSF's ellipticity reaches the estimator only through this catalog's
    # own `cov` (C_M spin-2 split +1.68%, second order in e) and `cov_odd`
    # (Sigma_X split +17.1%, first order).
    # 20k, not 200k: `bias.py` was always run at --n-targets 20000, so this is
    # the same statistical power -- but `sample_population(20000, rng(1))` is
    # NOT the first 20000 rows of the 200k draw, so `_psfe00` is this set's OWN
    # e_psf = 0 baseline and `pqr/v10_score.npz` must NOT be paired against it.
    "bulgedisc_v3_psfe00": {"plus": "targets_v3psfe0020_g1p02_20k",
                            "minus": "targets_v3psfe0020_g1m02_20k",
                            "zero": "targets_v3psfe0020_g0_20k"},
    "bulgedisc_v3_psfe1p": {"plus": "targets_v3psfe1p20_g1p02_20k",
                            "minus": "targets_v3psfe1p20_g1m02_20k",
                            "zero": "targets_v3psfe1p20_g0_20k"},
    "bulgedisc_v3_psfe2p": {"plus": "targets_v3psfe2p20_g1p02_20k",
                            "minus": "targets_v3psfe2p20_g1m02_20k",
                            "zero": "targets_v3psfe2p20_g0_20k"},
    "bulgedisc_v3_psfe1m": {"plus": "targets_v3psfe1m20_g1p02_20k",
                            "minus": "targets_v3psfe1m20_g1m02_20k",
                            "zero": "targets_v3psfe1m20_g0_20k"},
    # The amplitude scan.  The SCALING EXPONENT localises the channel for
    # free and it is the only way to extrapolate the e_psf = 0.2 bound down to
    # a DES-like 0.05: measured off the catalogs alone (2026-09-04),
    # `Sigma_X`'s spin-2 split runs as e^1.017 and `C_M`'s as e^2.030, so an
    # effect that scales linearly lives in `Sigma_X` and one that scales
    # quadratically lives in `C_M`.
    #
    # NO e00 entry at 05/10: `dev/render_psfe.sh` re-renders the baseline at
    # every amplitude, but with e_psf = 0 the amplitude does nothing and the
    # three files are byte-identical in `moments` (verified).  So
    # `bulgedisc_v3_psfe00` and `pqr/v11_psfe00.npz` are the baseline for the
    # whole scan, which saves two 40-minute runs.
    # 0.02 was added after the 0.05/0.10/0.20 scan came back with an exponent
    # of 0.3: quadratic (`C_M`) is excluded at 3.4 sigma but linear (`Sigma_X`)
    # only at 1.9, and an e_psf-INDEPENDENT offset fits best -- which would be
    # an artifact, not a leak, since a spin-2 response to a spin-2 perturbation
    # must vanish at least linearly.  At 0.02 the two hypotheses predict
    # dc2 = -5.8e-04 (constant) against -1.0e-04 (linear), ~2.7 sigma apart.
    "bulgedisc_v3_psfe2p02": {"plus": "targets_v3psfe2p02_g1p02_20k",
                              "minus": "targets_v3psfe2p02_g1m02_20k",
                              "zero": "targets_v3psfe2p02_g0_20k"},
    "bulgedisc_v3_psfe1p05": {"plus": "targets_v3psfe1p05_g1p02_20k",
                              "minus": "targets_v3psfe1p05_g1m02_20k",
                              "zero": "targets_v3psfe1p05_g0_20k"},
    "bulgedisc_v3_psfe2p05": {"plus": "targets_v3psfe2p05_g1p02_20k",
                              "minus": "targets_v3psfe2p05_g1m02_20k",
                              "zero": "targets_v3psfe2p05_g0_20k"},
    "bulgedisc_v3_psfe1p10": {"plus": "targets_v3psfe1p10_g1p02_20k",
                              "minus": "targets_v3psfe1p10_g1m02_20k",
                              "zero": "targets_v3psfe1p10_g0_20k"},
    "bulgedisc_v3_psfe2p10": {"plus": "targets_v3psfe2p10_g1p02_20k",
                              "minus": "targets_v3psfe2p10_g1m02_20k",
                              "zero": "targets_v3psfe2p10_g0_20k"},
    # The analytic population: two co-elliptical Gaussians whose moments were
    # DRAWN from a chosen density rather than pushed forward from galaxy
    # parameters, so P(m|g), Q and R are known in closed form -- `truth.py`.
    # This is the only population where a measured m or c can be compared
    # against what it should have been, rather than only against zero.
    "gauss2": {"plus": "gauss2_g1p02_1M", "minus": "gauss2_g1m02_1M",
               "zero": "gauss2_g0_1M"},
    # The analytic population rebuilt by the v3 recipe -- `POP=gauss2 TAG=g2v3
    # NPRIOR=100000 NTARGET=500000 SIZE=500k bash rebuild_v3.sh` -- so it sits
    # under the SAME chart, architecture and noise_sigma as `bulgedisc_v3`.
    # Every gauss2 artifact older than this is in `archive_20260902/` and
    # predates the chart fix and the free-net spin-2 stretch; none of it is
    # comparable.  GUIDING_PRINCIPLES 3.2: a bulgedisc number is not
    # interpretable until the same measurement passes here.  500k targets puts
    # galaxy sample variance at ~7.6e-04 (gauss2_deep measured +/-1.2e-03 at
    # 200k), i.e. below tau = 1e-3, which 20k cannot reach.
    # `gauss2_fwd`, FORWARD-sampled: the population is drawn in galaxy
    # parameters and pushed through the moment map, so there is no rejection
    # step and the realised population IS the chosen density (`truth.log_p_theta`
    # is its closed form).  The rejection-sampled `gauss2`/`gauss2_real` above
    # keep only 20.3% of P0's draws and land 0.41 sd wide on the size axis
    # against the 0.84 they were fitted to; this one matches bulgedisc_v3's
    # moments on all five chart axes (sd ratios 1.00/0.89/0.87/0.95/0.95).
    "gauss2_v3": {"plus": "targets_g2v3_g1p02_500k",
                  "minus": "targets_g2v3_g1m02_500k",
                  "zero": "targets_g2v3_g0_500k"},
    # The SAME galaxies at 2x the depth: noise_sigma 1.86, median flux S/N 9.6
    # against g2v3's 19.0, `SIG_XY` exactly 2x (222.12547 vs 111.06273).  The
    # depth is the only difference, so the pair isolates what the centroid
    # layer is worth -- at 19.0 the marginalisation was worth +3.5e-4 on the
    # old population and +2e-3 to +1.4e-2 at its "deep" setting, and this is
    # the first time that comparison can be made where P(m|g), Q and R are
    # closed form.  Its own prior AND its own copy grid: `Sigma_X` scales with
    # noise_sigma, and the copy grid is only valid within a factor 1.4 in it.
    "gauss2_v3d": {"plus": "targets_g2v3d_g1p02_500k",
                   "minus": "targets_g2v3d_g1m02_500k",
                   "zero": "targets_g2v3d_g0_500k"},
    # The same population and depth at 9.2x the targets, for sigma(m1) = 1e-3.
    # Its prior is g2v3d's file unchanged (TRAIN_DATA below): RawMomentStandardize
    # is frozen at training time, so the new targets MUST be standardised against
    # the file the flows were trained on.  Do not retrain, do not re-render it.
    "gauss2_v3e": {"plus": "targets_g2v3e_g1p02_4600k",
                   "minus": "targets_g2v3e_g1m02_4600k",
                   "zero": "targets_g2v3e_g0_4600k"},
    # A 2k version of the same, for smoke-testing the pipeline end to end
    # without waiting on a 1M-galaxy render.  Far too small to measure a bias
    # with -- it is there so `check_flow_vs_truth` can be exercised in minutes.
    "gauss2_2k": {"plus": "gauss2_g1p02_2k", "minus": "gauss2_g1m02_2k",
                  "zero": "gauss2_g0_2k"},
    # gauss2 with image noise at the same noise_sigma=2.73 depth as
    # bulgedisc_deep, co-elliptical so it carries no bulge/disc misalignment --
    # the control for whether the Mr/Mf 3.0-3.2 response-scatter floor is
    # specific to that hidden variable.  Needs centroid flows trained on
    # copies_gauss2_deep.fits.
    "gauss2_deep": {"plus": "targets_gauss2_deep_g1p02_200k",
                    "minus": "targets_gauss2_deep_g1m02_200k",
                    "zero": "targets_gauss2_deep_g0_200k"},
}

# The catalog each population's flows were standardised on (RawMomentStandardize
# is fixed at training time, so `main` has to rebuild the same split) --
# NOT auto-derived from `--pop`, because a naive fallback silently pointed
# every non-"sersic" population at bulgedisc's moments.fits (caught when
# gauss2_deep's selection-term correction came out byte-identical to
# bulgedisc_deep's).  bulgedisc's noisy/deep variants are the same underlying
# population as "bulgedisc", just measured with image noise, so they share its
# noiseless moments.fits.
TRAIN_DATA = {
    "bulgedisc": "moments.fits", "bulgedisc_noisy": "moments.fits",
    "bulgedisc_deep": "moments.fits",
    "bulgedisc_deep_v2": "moments_bulgedisc_v2.fits",
    "bulgedisc_v3": "moments_bulgedisc_v3.fits",
    # The v3 prior serves the elliptical-PSF configs unchanged -- the latent
    # moments are PSF-independent bit for bit (gate G0), so re-rendering it at
    # e_psf != 0 would reproduce this file exactly.
    "bulgedisc_v3_psfe00": "moments_bulgedisc_v3.fits",
    "bulgedisc_v3_psfe1p": "moments_bulgedisc_v3.fits",
    "bulgedisc_v3_psfe2p": "moments_bulgedisc_v3.fits",
    "bulgedisc_v3_psfe1m": "moments_bulgedisc_v3.fits",
    "bulgedisc_v3_psfe2p02": "moments_bulgedisc_v3.fits",
    "bulgedisc_v3_psfe1p05": "moments_bulgedisc_v3.fits",
    "bulgedisc_v3_psfe2p05": "moments_bulgedisc_v3.fits",
    "bulgedisc_v3_psfe1p10": "moments_bulgedisc_v3.fits",
    "bulgedisc_v3_psfe2p10": "moments_bulgedisc_v3.fits",
    "sersic": "moments_sersic.fits",
    "gauss2": "gauss2_g0_1M.fits", "gauss2_2k": "gauss2_g0_2k.fits",
    "gauss2_v3": "moments_gauss2_fwd_g2v3.fits",
    "gauss2_v3d": "moments_gauss2_fwd_g2v3d.fits",
    "gauss2_v3e": "moments_gauss2_fwd_g2v3d.fits",   # deliberate: see CATALOGS
    "gauss2_deep": "gauss2_g0_1M.fits",
}


def condition(g, sigma_x):
    """The flow's condition vector, given the shear and (optionally) Sigma_X.

    Without the centroid layer it is just g.  With it the chain carries
    [g1, g2, C00, C01, C11] -- shear reads the first two, centroid the last
    three.  Everything downstream differentiates with respect to `g` ALONE and
    treats `sigma_x` as a fixed per-target constant, which is what makes Q and R
    (eq. 12-13) still mean what the estimator needs: the target's noise
    properties are data, not something lensing moves.
    """
    return g if sigma_x is None else jnp.concatenate([g, sigma_x])


def load_cov(path):
    """C_M, the even-moment noise covariance of a target (paper eq. 9).

    It is a pure noise covariance -- set by the PSF, the weight function and the
    depth, not by the galaxy -- so in these sims every target shares one C_M and
    one Cholesky factor serves for all of them.  Heteroscedastic catalogs would
    only mean carrying a factor per target.
    """
    c = bfd.MomentCovariance.bulkUnpack(fitsio.read(path)["cov"])
    assert np.allclose(c, c[0]), "C_M varies between targets; see load_cov"
    return np.asarray(c[0], dtype=np.float64)


def add_noise(m, cov, seed):
    """One noise realization on the targets: M = M^G + M^n, M^n ~ N(0, C_M).

    Drawing the noise in moment space rather than redrawing noisy stamps is not
    an approximation.  The moments are a linear functional of the image at a
    FIXED centre (paper eq. 5, 7-8) and the pixel noise is Gaussian, so M^n is
    exactly multivariate normal with the covariance eq. (9) already stores.
    What it does assume is the known centre: letting the detection re-find
    X = 0 in the presence of noise is the centroid marginalisation, deferred.
    """
    L = np.linalg.cholesky(cov)
    return m + np.random.default_rng(seed).standard_normal(m.shape) @ L.T


def kernel_draws(cov, n, samples, seed):
    """`samples` offsets per target, drawn from the noise kernel N(0, C_M).

    Antithetic in pairs: eps and -eps both appear, which cancels the leading
    term of the estimator's error where the prior varies linearly across the
    kernel -- which is most of it, the kernel being narrow next to the
    population.  Independent draws PER TARGET, deliberately: sharing one set
    across targets would correlate their Monte Carlo errors so they no longer
    average down in the ensemble sums of eq. (45)-(46).

    Drawn with `jax.random` ON DEVICE.  This used numpy: ~164k float64 normals
    per chunk generated on the host, then pushed H2D, which the profiler caught
    stalling the GPU for 85 ms per 6 target batches (2.9% of wall) -- the card
    idled while Python made random numbers.

    THIS CHANGED THE RNG STREAM.  Every number moves, exactly as a new seed
    would; the estimator is unbiased under either stream, so the comparison
    against earlier runs is STATISTICAL, not bit-for-bit.  PQR files written
    before this change cannot be paired against ones written after -- re-run
    both arms of any comparison you care about.

    float32 rather than numpy's float64 costs nothing: with x64 off (it is,
    until the bootstrap at the end of `main`) the caller's `jnp.asarray`
    downcast these to float32 the moment they reached the device anyway.
    """
    L = np.linalg.cholesky(cov)          # 5x5 constant; the RNG was the cost
    half = jr.normal(jr.key(seed), (n, samples // 2, 5),
                     dtype=jnp.float32) @ jnp.asarray(L.T, jnp.float32)
    return jnp.concatenate([half, -half], axis=1)


def log_conv_is(flow, m_i, draws, log_wt, g, ok=None):
    """log (1/S) sum_s exp(log_wt_s) P(draws_s | g) -- the general importance-
    sampled estimator of the convolution P(M_i|g) = INT dm P(m|g) L(M_i - m),
    the continuum limit of the paper's sum over templates, eq. (38).

    `g` is the flow's full condition vector -- just the shear without the
    centroid layer, or [g1, g2, C00, C01, C11] with it (see `condition`).  It is
    forwarded verbatim, so callers differentiating with respect to shear vary
    only its leading two entries.

    `draws` are absolute points in moment space (not offsets from M_i) and
    `log_wt_s` is the log ratio of L(M_i - draws_s) to whatever proposal
    density produced draws_s, fixed once and reused unchanged as g varies --
    so Q and R (eq. 12-13) still come from differentiating ONE estimator with
    common random numbers, exactly as in `log_conv`.  `log_conv`'s pure-kernel
    draws are the alpha = 1 special case, log_wt = 0 everywhere: the kernel is
    its own proposal, so L cancels out of the ratio entirely.
    """
    # The prior's chart cannot evaluate a draw with Mf <= 0, Mr <= 0, or
    # Mr/Mf at or above the point-source ceiling -- and needs no evaluating:
    # no galaxy has a negative flux or size or is larger than the PSF is small,
    # the true prior is zero there, and zero weight is the right answer.  The
    # ceiling is the half that matters here: the kernel is wide enough that a
    # well-resolved target still puts draws past it, and before the chart was
    # bounded the flow answered those with a positive density.
    if ok is None:
        # `draws` are RAW moments and `flow` still carries the chart, so the
        # domain test belongs right here.
        ok = in_domain(draws)
        dummy = safe_point(m_i)
    else:
        # `draws` have already been through the peel, so they are STANDARDISED
        # coordinates and `in_domain` -- which reads Mf, Mr, Mc -- is meaningless
        # on them.  The caller tested the raw draw before peeling and hands the
        # answer down.  The stand-in only has to be finite: it is masked out of
        # the logsumexp below and exists solely so the flow is never evaluated
        # at a NaN, whose gradient would poison the whole target.
        dummy = jnp.zeros_like(draws[0])
    lp = flow.log_prob(jnp.where(ok[:, None], draws, dummy), condition=g)
    return (jax.nn.logsumexp(jnp.where(ok, lp + log_wt, -jnp.inf))
            - jnp.log(draws.shape[0]))


def log_conv(flow, m_i, eps, g):
    """log INT dm P(m|g) L(M_i - m), by Monte Carlo over the noise kernel.

    L is symmetric, so drawing m = M_i + eps with eps ~ N(0, C_M) makes

        Phat = (1/S) sum_s P(M_i + eps_s | g)

    an unbiased estimator of the convolution, with the flow standing in for
    sum_G p_G delta(m - M^G).  The draws do not depend on g, so Q and R
    (eq. 12-13) come from differentiating THIS estimator: one estimator
    differentiated, with common random numbers, not three noisy ones divided.
    This is `log_conv_is` with the kernel as its own proposal (log_wt = 0);
    see `mixture_draws` for the general, defensive-mixture proposal.
    """
    return log_conv_is(flow, m_i, m_i + eps, jnp.zeros(eps.shape[0]), g)


def make_psi(flow, peeled):
    """`Psi_g`: the RAW-moment diffeomorphism the shear layer pushes the prior by.

    Shear enters the chain in exactly one place, and everything ahead of it --
    the chart, and the centroid layer when there is one -- is g-independent.  So
    `P(.|g)` is the pushforward of `P(.|0)` through

        Psi_g = Phi^-1 . ShearResponse.shear(., g) . Phi,   Phi = chart (. centroid)

    which is the identity at g = 0.  `peeled` says which coordinate the caller
    holds: True for the standardised one `centroid_transform` returns (Phi has
    already run, so only its INVERSE is applied here), False for raw moments.
    Either way the result is a raw moment, which is where the noise kernel lives.
    """
    bij = flow.bijection.bijection.bijections
    chart = bij[0]
    cen = bij[1] if isinstance(bij[1], _CENTROID_LAYER_TYPES) else None
    sh = bij[2] if cen is not None else bij[1]

    def psi(x, g, sigma_x):
        cond = (g if sigma_x is None else
                jnp.concatenate([jnp.zeros_like(g), sigma_x]))
        if peeled:
            z = x
        else:
            # BOTH halves of Phi on the way in, or Psi_0 is not the identity:
            # applying `cen.inverse` on the way out without `cen.transform` on
            # the way in leaves `chart^-1 . cen^-1 . chart`, which missed by a
            # median 0.76% (p99 4.5%) of the moment -- a silent O(1%) error in
            # every Q and R it touched.
            z = chart.transform(x)
            if cen is not None:
                z = cen.transform(z, cond)
        y = sh.shear(z, g)
        return chart.inverse(y if cen is None else cen.inverse(y, cond))

    return psi


def _val_grad_hess(f):
    """`(f(0), grad f(0), hess f(0))` for a scalar `f` of a 2-vector g.

    One forward-over-reverse pass per g direction yields the value, the
    gradient AND a column of the Hessian.

    NEGATIVE, measured 2026-09-06: `jax.linearize` -- which evaluates the
    primal once and hands back a linear map to apply per tangent -- is a WASH
    (11.5 s against 11.3 s per 400 targets).  XLA already common-subexpression-
    eliminates the shared primal between the two `jvp`s, so the duplication is
    only apparent.  Kept as two `jvp`s because `linearize` holds the linearised
    residuals live across both tangent applications and so costs peak memory
    for nothing, which at 13.7/16.4 GB is a real risk.
    """
    zero = jnp.zeros(2)
    vg = jax.value_and_grad(f)
    (val, df), lin = jax.linearize(vg, zero)       # see `one_b`: shared primal
    _, h0 = lin(jnp.array([1.0, 0.0]))
    _, h1 = lin(jnp.array([0.0, 1.0]))
    return val, df, jnp.stack([h0, h1], axis=-1)


def _pad_cond(g, cond_shape):
    """`g` widened with zeros to `cond_shape`, for flowjax's shape check."""
    if cond_shape is None or cond_shape[-1] == g.shape[-1]:
        return g
    return jnp.concatenate(
        [g, jnp.zeros(cond_shape[-1] - g.shape[-1], g.dtype)])


def make_psi_ld(flow, peeled):
    """`(Psi_g(x), log|det dPsi_g/dx|)` -- the same map as `make_psi`, with its
    log-det by the CHAIN RULE instead of a Jacobian of the whole composition.

    This is a speed fix and nothing else; the number is the same one.  `Psi` is
    a composition of four or five bijections, so its log-det is the sum of
    theirs, and every one of them already knows its own analytically or from a
    much shorter derivative:

      * the chart is triangular -- its log-det is a handful of logs, free;
      * the centroid layer's is `slogdet(jacfwd(unmarginalize))`, 5 JVPs of one
        network pass;
      * the shear layer's is `-slogdet(jacfwd(unshear))` at the image point --
        5 JVPs of `unshear`, which is a closed-form polynomial in g.

    What it replaces is `slogdet(jax.jacfwd(psi))`: 5 JVPs of the WHOLE chain,
    and the chain contains `ShearResponse.shear`, which is itself three nested
    `jacfwd` calls (models/shear.py:403).  Differentiating a triply-nested
    jacfwd five times over, inside a forward-over-reverse Hessian in g, is
    where `log_conv_is_blend` spent its wall clock -- measured at 30 ms per
    target per arm, i.e. 25 h for a 1M-target run.

    The two agree to float32 roundoff at g = 0 and differ at O(g^3), because
    `shear` inverts `unshear`'s g-series only to second order (its own
    docstring) while `inverse_and_log_det` reads the log-det off `unshear` at
    the image point.  Q and R are the first two g-derivatives at g = 0, so
    that difference cannot reach them -- and `tests/test_psi_logdet.py` checks
    both claims rather than taking this paragraph's word for it.
    """
    bij = flow.bijection.bijection.bijections
    chart = bij[0]
    cen = bij[1] if isinstance(bij[1], _CENTROID_LAYER_TYPES) else None
    sh = bij[2] if cen is not None else bij[1]

    def psi_ld(x, g, sigma_x):
        cond = (g if sigma_x is None else
                jnp.concatenate([jnp.zeros_like(g), sigma_x]))
        ld = jnp.zeros((), x.dtype)
        if peeled:
            z = x
        else:
            z, d = chart.transform_and_log_det(x)
            ld = ld + d
            if cen is not None:
                z, d = cen.transform_and_log_det(z, cond)
                ld = ld + d
        # `sh.shear` is the bijection's INVERSE direction, so its log-det comes
        # from `inverse_and_log_det` -- see make_psi's `sh.shear(z, g)`.
        # `make_psi` calls the bare `shear` method and so skips flowjax's
        # condition-shape check; the wrapped `inverse_and_log_det` enforces it,
        # so pad g out to the declared width.  `unshear` reads `condition[:2]`
        # and nothing else (models/shear.py:393), so the padding is inert and
        # this is the same map `make_psi` builds.
        y, d = sh.inverse_and_log_det(z, _pad_cond(g, sh.cond_shape))
        ld = ld + d
        if isinstance(cen, CentroidMarginalize):
            # NOT `cen.inverse_and_log_det`.  That returns the
            # inverse-function-theorem value `-log|det d unmarginalize|`, which
            # is the log-det of `marginalize` only if the two are exact
            # inverses -- and they are not: both are CLOSED FORM, `_transport`
            # with +1 and -1 (models/centroid.py:498-518), an approximate pair.
            # Using it leaves 5.6e-04 in the log-det at g = 0, where `Psi` is
            # the identity and the two centroid terms must cancel exactly.
            # `jacfwd(marginalize)` is the map actually applied here, which is
            # what `slogdet(jacfwd(psi))` differentiated, and it is still only
            # 5 JVPs of a closed form.
            ld = ld + jnp.linalg.slogdet(
                jax.jacfwd(cen.marginalize)(y, cond))[1]
            y = cen.marginalize(y, cond)
        elif cen is not None:
            # `SigmaXBlockLayer` is invertible BY CONSTRUCTION (closed-form
            # both ways, no approximate transport pair), so its own
            # `inverse_and_log_det` needs no such correction -- it IS the map
            # `make_psi` applies (`cen.inverse(y, cond)`).
            y, d = cen.inverse_and_log_det(y, cond)
            ld = ld + d
        y, d = chart.inverse_and_log_det(y)
        return y, ld + d

    return psi_ld


def log_conv_is_kernel(psi, r0, x, draws_raw, c, g, sigma_x, cinv, ok):
    """Gauge-K estimator of log P(M|g): the shear moves the KERNEL, not the prior.

    `log_conv_is` holds the draws fixed and evaluates `p(. | g)`, so every
    g-derivative lands on the flow's own score `d_g log p`.  Under pi that score
    has a Hill tail index of 1.20 on bulgedisc_v2's faint end -- **its variance
    does not exist**, so `R = E[d2_g log p] + Var[d_g log p]` is not merely noisy
    but unestimable, and what the mixture proposal actually reports is whatever
    its weights truncate the tail to (see `dev/hmc_ref.py`).

    Substituting `m = Psi_g(u)` in eq. (38)'s integral moves the g out of the
    density and into the kernel, exactly:

        P(M|g) = INT p(m|g) L(M - m) dm = INT p(u|0) L(M - Psi_g(u)) du

    so with draws from the same g-independent proposal q,

        Phat(g) = (1/S) sum_s exp(c_s) L(M - Psi_g(u_s)),   c_s = log p(u_s|0)/q(u_s)

    `c = log p(u_s|0) + log_wt_s` is g-INDEPENDENT and precomputed by
    `_gauge_k_c` -- exactly `log_conv_is`'s own summand at g = 0 -- so the flow
    is evaluated once and never inside the g-autodiff.  The per-draw derivative
    is then a Gaussian score, bounded by construction, which is the property
    BFD's template sum has and this estimator did not.

    What is added to `c` is the kernel's CHANGE, never the kernel itself:

        ll(g) - ll(0) = r0.Cinv.delta - delta.Cinv.delta / 2,   delta = Psi_g(u) - u

    with `r0 = M - u`.  Forming `ll(g)` and `ll(0)` separately and subtracting
    would be catastrophic in float32: the mixture's flow-component draws land far
    from M for a bright target, where each term reaches ~1e12 and their
    difference is O(1).  That cancellation zeroed HALF the catalog ("980 targets
    had no draw with any weight") before this was written as a difference.  It
    also makes the estimator exactly `log_conv_is` at g = 0, since delta = 0
    there, and drops the kernel normalisation entirely.

    `x` is whatever coordinate the caller holds (see `make_psi`'s `peeled`);
    `draws_raw` and `r0` are in RAW moments, because L is Gaussian in those.
    """
    delta = jax.vmap(psi, in_axes=(0, None, None))(x, g, sigma_x) - draws_raw
    dll = (jnp.einsum("si,ij,sj->s", r0, cinv, delta)
           - 0.5 * jnp.einsum("si,ij,sj->s", delta, cinv, delta))
    return (jax.nn.logsumexp(jnp.where(ok, c + dll, -jnp.inf))
            - jnp.log(x.shape[0]))


def log_conv_is_blend(flow, psi_ld, m_raw, x, r0, log_wt, g, sigma_x, cinv, lam,
                      ok):
    """The ONE estimator: shear the draws by `lam` of the way, the prior the rest.

    Gauge P (`log_conv_is`) and gauge K (`log_conv_is_kernel`) are the endpoints
    of a family, and each one FAILS at the other's end -- measured on
    bulgedisc_v2, `B = Var_pi[score]` is 74.9 vs 4.3 at the faint end and 0.115
    vs 7930 at the bright one (`dev/lambda_scan.py`).  Neither is the answer and
    a per-target switch between them is not either.

    Translate the draws rigidly instead.  For ANY smooth `t(g)` with `t(0) = 0`,
    substituting `m = u + t(g)` in eq. (38) is exact and the Jacobian is 1:

        P(M|g) = INT p(u + t(g) | g) L(M - u - t(g)) du

    Move each draw by its OWN partial displacement, `t_s(g) = Psi_{lam g}(u_s) -
    u_s`, and pay the Jacobian `log|det dPsi_{lam g}/du|` that a per-draw map
    costs.  A RIGID translation by the target's displacement is exact and much
    cheaper but does not work: it only cancels the prior's g-dependence while
    the cloud is narrow, so it fixed the bright end and left q1/q2 at
    -1.06/-1.29, exactly where the wide kernel puts the draws far from M.  Then

      * lam = 0 is `log_conv_is` exactly -- the draws do not move and the
        derivative is the prior's score;
      * lam = 1 moves them by the full displacement, so to the extent the cloud
        is narrow the prior term cancels and the derivative is the kernel's;
      * in between the per-draw score is `(1 - lam) X_P + lam X_K` exactly, by
        the continuity equation -- a linear blend of the two, both of which have
        pi-mean Q.  So lam is a CONTROL VARIATE: every lam is unbiased, and
        `_blend_lambda` picks the minimum-variance one per target.

    Measured `B` at that lam, by flux quintile: 2.69, 4.21, 3.60, 1.54, 0.043 --
    flat, against endpoints spanning four orders of magnitude.

    The kernel term is again added as a DIFFERENCE (see `log_conv_is_kernel`) so
    no `log L` is ever formed alone.
    """
    # `psi_ld` returns the log-det by the chain rule over the layers rather than
    # from `slogdet(jacfwd(psi))`, which differentiated `ShearResponse.shear`'s
    # three nested jacfwds five times over -- see `make_psi_ld`.  Same number,
    # and it is what makes a 1M-target run affordable.
    y, ld = jax.vmap(psi_ld, in_axes=(0, None, None))(x, lam * g, sigma_x)
    delta = y - x
    dll = (jnp.einsum("si,ij,sj->s", r0, cinv, delta)
           - 0.5 * jnp.einsum("si,ij,sj->s", delta, cinv, delta))
    lp = flow.log_prob(jnp.where(ok[:, None], y, safe_point(m_raw)),
                       condition=condition(g, sigma_x))
    return (jax.nn.logsumexp(jnp.where(ok, lp + ld + log_wt + dll, -jnp.inf))
            - jnp.log(x.shape[0]))


def _blend_lambda(flow, psi_ld, m_raw, x, r0, log_wt, sigma_x, cinv, ok,
                  stride=1):
    """The minimum-variance `lam` for `log_conv_is_blend`, from the draws.

    The per-draw score of that estimator is `X0 + lam Z`, where `X0` is its
    score at lam = 0 (gauge P's) and `Z` is what the translation adds.  Every
    lam has the same pi-mean, so `Z` is a mean-zero control variate and

        lam* = -Cov_pi(X0, Z) / Var_pi(Z)

    on the g1 component (g2 matches by isotropy).

    `Z` is taken as the DIFFERENCE of the estimator's own score at lam = 1 and
    lam = 0, not from a closed form.  Writing it out by hand as `X_K - X_P` is
    wrong: the translation also drags the prior, contributing a
    `grad_m log p . dPsi/dg` term and a divergence, and a lam fitted to the
    idealised blend minimises the wrong quadratic -- measured, that put m1 at
    -1.06, WORSE than either endpoint, with 233 of 2000 targets tripping the
    |Q|/|R| guard.

    Clipped to [0, 1]: outside it neither endpoint is being interpolated.

    ponytail: lam is fitted on the SAME draws it is used on, correlating it with
    the estimate at O(1/S) -- the order `_merge_finish`'s jackknife already
    removes.  Cross-fit across chunks if that is ever shown to matter.
    """
    zero = jnp.zeros(2)
    # `stride` fits lam on every `stride`-th draw.  This is the cheapest real
    # lever on wall clock: `score` is two full gradient passes over the chunk
    # and `lp0` a third forward, all to produce ONE SCALAR per target, and they
    # run once per target batch on top of the two Hessian chunks -- measured at
    # ~29% of a production run.  It is also the safest, because EVERY lam is
    # unbiased (the control-variate argument above): a noisier lam costs a
    # little variance in Q and R and moves no expectation.  The draws are iid,
    # so a stride is a fair subsample.  Default 1 = unchanged.
    sub = slice(None, None, stride)
    x, r0, log_wt, ok = x[sub], r0[sub], log_wt[sub], ok[sub]
    xs = jnp.where(ok[:, None], x, safe_point(m_raw))

    def score(lam):
        def one(z, r):
            def f(g):
                y, ld = psi_ld(z, lam * g, sigma_x)
                return (flow.log_prob(y, condition=condition(g, sigma_x)) + ld
                        + jnp.dot(r, cinv @ (y - z)))
            return jax.grad(f)(zero)[0]
        return jax.vmap(one)(xs, r0)

    x0, x1 = score(0.0), score(1.0)
    z = x1 - x0

    lp0 = jax.vmap(lambda v: flow.log_prob(
        v, condition=condition(zero, sigma_x)))(xs)
    fin = jnp.isfinite(x0) & jnp.isfinite(z)
    pi = jax.nn.softmax(jnp.where(ok & fin, lp0 + log_wt, -jnp.inf))
    # Drop zero-weight draws BEFORE squaring: gauge P's score reaches 1e15 on
    # draws whose pi is exactly 0, and the float32 square overflows past 1.8e19,
    # after which 0 * inf NaNs the target.  They carry no weight, so no
    # expectation changes.  Kept draws peak near 1e4.
    keep = pi > 0
    pi = jnp.where(keep, pi, 0.0)
    a = jnp.where(keep, x0, 0.0)
    b = jnp.where(keep, z, 0.0)
    da, db = a - pi @ a, b - pi @ b
    vzz = pi @ jnp.where(keep, db * db, 0.0)
    vaz = pi @ jnp.where(keep, da * db, 0.0)
    lam = jnp.where(vzz > 0, -vaz / jnp.where(vzz > 0, vzz, 1.0), 0.0)
    return jnp.clip(jnp.nan_to_num(lam), 0.0, 1.0)


def _gauge_k_c(flow, x, log_wt, m_raw, sigma_x, ok, peeled):
    """The g-independent part of gauge K: `log_conv_is`'s summand, frozen at g=0.

    Gauge K adds only the kernel's CHANGE to this (see `log_conv_is_kernel`), so
    `log L(M - u)` stays folded inside `log_wt` where `mixture_draws` put it and
    is never formed on its own -- which is what keeps a far draw's ~1e12 out of
    float32.
    """
    dummy = jnp.zeros_like(x[0]) if peeled else safe_point(m_raw)
    lp0 = flow.log_prob(jnp.where(ok[:, None], x, dummy),
                        condition=condition(jnp.zeros(2), sigma_x))
    return lp0 + log_wt


def mixture_draws(flow, m, cov, samples, alpha, seed, batch=None, sigma_x=None,
                  to_host=True):
    """Draws and log-importance-weights from the defensive mixture proposal

        q(m) = alpha * N(m; M_i, C_M) + (1 - alpha) * P(m | g=0)

    `flow` here is the PROPOSAL -- it need not be the flow whose density is
    being integrated.  A proposal only has to COVER the posterior, not match
    it, so callers may pass a flow shared across runs (`--proposal-flow`) to
    keep the flow-component draws common random numbers even when the flow
    actually being evaluated differs between runs.

    Pure-kernel sampling (`kernel_draws`) starves whenever C_M is wide next to
    the prior: the ESS can sit below 10 for a real fraction of targets no
    matter how many draws are spent, because every draw comes from the SAME
    place and none of them land where the prior actually has mass.  Mixing in
    draws from the prior itself fixes that: since q >= alpha * L(M_i - m)
    everywhere, the importance weight w = P(m|g)/q(m) is bounded by
    sup_m P(m|g) / alpha, so the estimator has finite variance regardless of
    how badly L and the prior overlap.

    Both components are drawn and their densities evaluated at g = 0, NOT at
    the g the caller will eventually want -- the proposal only has to COVER
    the posterior, not track it.  Sampling from P(.|g) would put the draws
    inside the g-autodiff and add a pathwise gradient term through the
    sampler; freezing the proposal at g = 0 keeps every g-dependence in the
    numerator of `log_conv_is`, preserving the "one estimator differentiated,
    common random numbers" property the module docstring describes.

    Returns (draws, log_wt), float64 numpy arrays of shape (n, samples, 5) and
    (n, samples), matching `kernel_draws`'s convention except that `draws` are
    absolute moment-space points, not offsets (log_conv_is needs the flow-
    component draws, which are not offsets from anything).

    `to_host=False` returns DEVICE arrays instead.  `pqr_streamed`'s draws go
    straight back into a jitted function, so the numpy round trip there is pure
    loss -- and worse than its bytes, because `np.asarray` on a device array
    DRAINS THE PIPELINE, serialising the GPU against the host once per chunk.
    Same reasoning as `centroid_transform`'s own `to_host`.  The values are
    bit-identical either way; this only changes where they live.
    """
    n = len(m)
    # Sampling runs the flow BASE -> DATA, which for the shear layer is a Newton
    # solve plus a jacfwd log-det per draw (models/shear.py) -- several times the
    # cost and memory of the data -> base log_prob direction, even at g = 0 where
    # the layer is the identity.  So the chunk is sized on `batch * samples`, the
    # same budget `pqr` uses for its forward-over-reverse Hessian.
    batch = max(1, 131_072 // samples) if batch is None else batch
    n_k = round(alpha * samples)
    n_k -= n_k % 2                       # kernel half stays antithetic (pairs)
    n_f = samples - n_k
    # The alpha >= 1 branch below skips the flow entirely, so the evenness trim
    # turning an ODD `samples` into n_f = 1 would silently hand back one draw
    # fewer than asked -- a shape change, not a weight one.  Every caller passes
    # a power of two; this is the net under that.
    if alpha >= 1.0 and n_f:
        raise ValueError(f"alpha >= 1 needs an even `samples`; got {samples}")

    # float64 here is deliberate and is NOT a wasted host copy: it is a no-op
    # when `m` already is float64, and callers that enable x64 (the tests do)
    # rely on it to keep the proposal in double.  With x64 off the `jnp.asarray`
    # below downcasts to float32 on the way to the device, as the flow needs.
    m64 = np.asarray(m, dtype=np.float64)
    kernel = kernel_draws(cov, n, n_k, seed)     # (n, n_k, 5) offsets, empty if n_k=0

    L = jnp.asarray(np.linalg.cholesky(cov))
    log_diag = jnp.sum(jnp.log(jnp.diag(L)))

    # Guard log(alpha) and log(1 - alpha) at the 0/1 endpoints so neither
    # produces a NaN; -inf * (finite log-density) is fine, it just drops that
    # component out of the logaddexp mixture density below.
    log_alpha = -jnp.inf if alpha <= 0.0 else float(np.log(alpha))
    log_1ma = -jnp.inf if alpha >= 1.0 else float(np.log1p(-alpha))

    key = jr.key(seed) if n_f else None
    draws_out, wt_out = [], []
    for i in range(0, n, batch):
        m_i = jnp.asarray(m64[i:i + batch])
        kern_i = jnp.asarray(kernel[i:i + batch])
        sx_i = (None if sigma_x is None else
                jnp.asarray(sigma_x[i:i + batch], dtype=jnp.float32))

        if alpha >= 1.0:
            # q IS the kernel, so log_wt = log_k - log_q is identically zero and
            # log_k need never be formed -- it is a triangular solve over every
            # draw whose result is then subtracted from itself.  There is no
            # flow work in this branch at all, so `_mixture_chunk` -- built to
            # hoist exactly that work -- is never called here.
            x = m_i[:, None, :] + kern_i
            if to_host:
                draws_out.append(np.asarray(x, dtype=np.float32))
                wt_out.append(np.zeros(x.shape[:-1], dtype=np.float32))
            else:
                draws_out.append(x.astype(jnp.float32))
                wt_out.append(jnp.zeros(x.shape[:-1], jnp.float32))
            continue

        sub = None
        if n_f:
            key, sub = jr.split(key)
        x, log_wt = _mixture_chunk(flow, m_i, kern_i, sx_i, sub, L, log_diag,
                                   log_alpha, log_1ma, n_f)
        # float32 on the host: these are (n, samples, 5) and at 32k draws/target
        # float64 would be 26 GB per catalog.  Nothing is lost -- `_over_targets`
        # casts to f32 on the way to the device anyway, because the flow is f32.
        if to_host:
            draws_out.append(np.asarray(x, dtype=np.float32))
            wt_out.append(np.asarray(log_wt, dtype=np.float32))
        else:
            draws_out.append(x.astype(jnp.float32))
            wt_out.append(log_wt.astype(jnp.float32))

    cat = np.concatenate if to_host else jnp.concatenate
    return cat(draws_out), cat(wt_out)


def split_centroid(flow):
    """Peel the data-adjacent centroid layer off the front of the chain.

    Returns `(rest, layer)`, or `(flow, None)` if there is no centroid layer.

    This is worth doing because of WHERE the layer sits.  In data -> base order
    the chain is [centroid, shear, raw2standard, bulk], so the centroid layer is
    applied FIRST, to the raw target moments, conditioned only on Sigma_X.  Both
    its input and its condition are independent of g, so

        log p(x | g, Sigma) = log p_rest(centroid(x, Sigma) | g)
                              + log |det d centroid / dx|

    and that second term is exactly g-independent: it contributes NOTHING to Q
    or R (eq. 12-13).  Evaluating it once, outside the autodiff, keeps the
    forward-over-reverse Hessian from ever traversing the layer's jacfwd log-det
    -- which is 5 extra JVPs of the coefficient network per draw, and what makes
    the Hessian run out of memory at the batch sizes the shear-only flow used.

    The peel is exact, not an approximation; see `centroid_transform`.

    RESTRICTED to a standalone (3,)-conditioned layer, not a chained (5,)
    one, even though `models/centroid.py`'s `CentroidMarginalize` no longer
    reads g at all (module docstring, "No shear-conditioning") and so is
    mathematically safe to peel either way.  Verified empirically NOT safe in
    practice: peeling a chained layer and reconstructing `rest =
    Invert(Chain(bij[2:]).merge_chains())` reproduces `flow.log_prob` exactly
    (matches to float32 roundoff), but `bias.py`'s actual Q/R -- gradient and
    HESSIAN of log_prob w.r.t. g, computed through `rest` -- come out wrong
    even with `_transport` forced to a literal identity (m1 ~ -0.66 on
    `gauss2_deep` where `--no-centroid` gives ~-0.03).  So the bug is in
    autodiff through the RECONSTRUCTED sub-chain specifically, not in this
    layer's transport or in log_prob-level peeling -- root cause not yet
    found (HANDOFF.md, 2026-08-27).  Refusing here costs what it always did:
    5 extra JVPs of this (now free, coefficient-net-less) layer per draw
    inside the forward-over-reverse Hessian, no longer using a large network.
    """
    bij = flow.bijection.bijection.bijections
    # data -> base order is [raw2standard, centroid, shear, *bulk]; the chart is
    # always first now, so the centroid layer is at index 1 when it is present.
    if len(bij) < 2 or not isinstance(bij[1], _CENTROID_LAYER_TYPES):
        return flow, None
    if bij[1].cond_shape[-1] >= 5:
        return flow, None
    rest = Invert(Chain(list(bij[2:])).merge_chains())
    # Both halves are g-independent here -- the chart is a fixed
    # reparametrisation and the centroid layer reads only Sigma_X out of
    # whatever condition width it was built for -- so the pair can be
    # hoisted together and their log-dets summed.
    return Transformed(flow.base_dist, rest), (bij[0], bij[1])


@eqx.filter_jit
def _centroid_apply(peel, x, cond):
    """vmapped centroid transform + log-det, jitted ONCE at module level.

    Building the `eqx.filter_jit` wrapper inside `centroid_transform` instead
    made a fresh wrapper per call, so JAX recompiled on every chunk -- 1181 ms
    against 18 ms for the traced computation, and 91% of the whole integration
    step.  Equinox splits `layer` into its arrays and its structure, so retracing
    happens only if the structure or the shapes change, not when weights do.
    """
    chart, layer = peel

    def one(xi, ci):
        z, ld0 = chart.transform_and_log_det(xi)
        y, ld1 = layer.transform_and_log_det(z, ci)
        return y, ld0 + ld1

    return jax.vmap(one)(x, cond)


@eqx.filter_jit
def _mixture_chunk(flow, m_i, kern, sx_i, key, L, log_diag, log_alpha, log_1ma, n_f):
    """One batch of `mixture_draws`' flow work, jitted ONCE at module level.

    `flow.sample` and `flow.log_prob` each traverse a ~12-layer flow whose
    every layer holds an MLP -- hundreds of individual kernel launches,
    dispatched one at a time from Python when called eagerly.  Measured at 8
    targets x 512 draws: eager `flow.sample` 1123.8 ms, eager `flow.log_prob`
    717.0 ms, both jitted together 2.0 ms -- 922x, and the difference between
    8% and full GPU utilisation on a production run.  As with `_centroid_apply`,
    building the wrapper inside `mixture_draws` would retrace on every batch;
    built here it retraces only when a batch's shapes actually change.

    Callers must not invoke this at alpha >= 1.0: with the kernel as its own
    proposal there is no flow work to hoist (log_wt is identically zero), so
    `mixture_draws` keeps that branch outside this function entirely.
    """
    # The proposal is evaluated at g = 0, but at each target's OWN Sigma_X: the
    # centroid layer is part of the prior the proposal has to cover, and unlike
    # g it is not something the estimator differentiates.  Built HERE rather
    # than by the caller -- as an eager `jax.vmap` outside any jit it was one
    # more per-chunk dispatch, and the profiler charged those stalls to
    # `mixture_draws` (52% of all remaining GPU idle).  `sx_i is None` is a
    # static leaf, so this branch costs nothing at trace time.
    cond0 = (jnp.zeros(2) if sx_i is None else
             jax.vmap(condition, in_axes=(None, 0))(jnp.zeros(2), sx_i))
    x = m_i[:, None, :] + kern
    if n_f:
        if cond0.ndim == 1:
            flow_x = flow.sample(key, (m_i.shape[0], n_f), condition=cond0)
        else:
            # flowjax prepends sample_shape to the CONDITION's batch shape,
            # so a per-target condition already supplies the target axis --
            # asking for (B, n_f) on top of it would nest a second one.
            flow_x = jnp.moveaxis(flow.sample(key, (n_f,), condition=cond0), 0, 1)
        x = jnp.concatenate([x, flow_x], axis=1)

    def log_normal(offset):
        # log N(offset; 0, cov), via a triangular solve rather than inverting
        # cov -- the numerically stable way to get the quadratic form.
        shp = offset.shape[:-1]
        y = jax.scipy.linalg.solve_triangular(L, offset.reshape(-1, 5).T, lower=True)
        quad = jnp.sum(y * y, axis=0).reshape(shp)
        return -0.5 * quad - log_diag - 2.5 * jnp.log(2 * jnp.pi)

    log_k = log_normal(x - m_i[:, None, :])
    ok = in_domain(x)
    dummy = safe_point(m_i)
    safe = jnp.where(ok[..., None], x, dummy[:, None, :])
    # log_prob vectorises as (5),(5)->(), so a per-target condition has to
    # be broadcast along the DRAW axis as well as the target axis.
    cond_f = (cond0 if cond0.ndim == 1 else
              jnp.broadcast_to(cond0[:, None, :],
                               safe.shape[:-1] + cond0.shape[-1:]))
    log_f = jnp.where(ok, flow.log_prob(safe, condition=cond_f), -jnp.inf)
    log_q = jnp.logaddexp(log_alpha + log_k, log_1ma + log_f)

    # Where a draw is far enough out that BOTH densities underflow float32,
    # log_wt is -inf - (-inf) = NaN -- on a draw that is still nominally
    # in-domain (Mf, Mr > 0), so `log_conv_is`'s domain mask does not catch
    # it, and ONE such draw NaNs that target's whole logsumexp and with it
    # its Q and R.  A draw out where L has underflowed carries no weight, so
    # -inf is not a patch over the NaN, it IS the answer.
    log_wt = log_k - log_q
    return x, jnp.where(jnp.isfinite(log_wt), log_wt, -jnp.inf)


def centroid_transform(peel, x, sigma_x, batch=200_000, to_host=True):
    """Apply the centroid layer to `x`, returning (transformed, log-det).

    `x` is (..., 5) and `sigma_x` broadcasts against its leading axes.  The
    log-det is returned separately so the caller can fold it into an importance
    weight, which is where it belongs: it multiplies the density of that draw
    and never depends on g.  It cannot be skipped: it varies by ~18 nats across
    a chunk of kernel draws (they reach the poorly resolved region where the
    layer's T is large), and dropping it moves R by 8%.

    Everything is kept on device and the Sigma_X broadcast is done there too.
    Doing it in host numpy instead -- `np.broadcast_to(...).reshape()` plus a
    few `astype` copies of a (131072, 5) array per chunk -- cost 1181 ms against
    18 ms for the actual transform, i.e. 98% of this function and 91% of the
    whole integration step.  `to_host=False` also leaves the result on device
    for callers that feed it straight back into a jitted function.
    """
    shp = x.shape[:-1]
    xf = jnp.asarray(x, dtype=jnp.float32).reshape(-1, 5)
    sj = jnp.asarray(sigma_x, dtype=jnp.float32)
    layer = peel[1]
    # The layer keeps the chain's condition width even after being peeled off,
    # and reads Sigma_X as the LAST three entries -- so pad the shear slots.
    if layer.cond_shape[0] > 3:
        sj = jnp.concatenate(
            [jnp.zeros(sj.shape[:-1] + (layer.cond_shape[0] - 3,)), sj], axis=-1)
    sf = jnp.broadcast_to(sj.reshape((-1,) + (1,) * (len(shp) - 1) + sj.shape[-1:]),
                          shp + sj.shape[-1:]).reshape(-1, sj.shape[-1])

    z_out, ld_out = [], []
    for i in range(0, xf.shape[0], batch):
        z, ld = _centroid_apply(peel, xf[i:i + batch], sf[i:i + batch])
        z_out.append(z)
        ld_out.append(ld)
    z = jnp.concatenate(z_out).reshape(shp + (5,))
    ld = jnp.concatenate(ld_out).reshape(shp)
    if to_host:
        return np.asarray(z, dtype=np.float32), np.asarray(ld, dtype=np.float32)
    return z, ld


def _fixed_batch(fn, batch):
    """Wrap a vmapped+jitted `fn` so XLA only ever sees ONE leading dim.

    The trailing target batch is otherwise a second shape, and a second shape
    is a second full compile of the heaviest kernels in the program: measured
    on the streamed path, `one_b` and the `_blend_lambda` fit each compile
    4 times (2 arms x 2 shapes) for 173 s of a 281 s total.

    The padding is applied at the JIT BOUNDARY only -- `mixture_draws` is still
    called with the true `len(m_b)`, so the random stream is untouched (padding
    `m_b` itself would reseed the trailing batch and move its targets by far
    more than a rounding).  Pad rows are copies of row 0, and `vmap` is
    row-independent, so they cannot reach the real rows; they are sliced off
    before anything accumulates.
    """
    def go(*args):
        # `args[0]` is an array at all three call sites, so read the leading dim
        # straight off it: the generator-plus-`hasattr` scan this replaced ran on
        # EVERY chunk and showed up in the profile as 28.9 ms of GPU idle.
        k = args[0].shape[0]
        if k == batch:
            return fn(*args)
        pad = lambda a: (a if not hasattr(a, "shape") else
                         jnp.concatenate([a, jnp.repeat(a[:1], batch - k, 0)]))
        return jax.tree.map(lambda o: o[:k], fn(*(pad(a) for a in args)))
    return go


def _over_targets(one, arrays, batch):
    """Run a per-target function over the catalog in batches, in float64 out.

    `arrays` is a tuple of per-target arrays (or None, for an argument `one`
    doesn't use); vmap treats a None leaf as an empty pytree, so it is simply
    passed through unbatched -- the same trick the old (m, eps) version used.

    The flow is float32 (it was trained that way and the checkpoint stores f32),
    but a 1M-galaxy sum in float32 would lose more than the bias being measured,
    so the per-galaxy values are promoted before they are ever accumulated.

    The trailing chunk is PADDED up to `batch` and sliced back off.  A short
    last chunk is a second shape, and a second shape is a second XLA
    compilation of `one` -- the heaviest kernel in the program, measured at
    half of `pqr_streamed`'s entire compile cost (4 heavy compiles become 2).
    `vmap` is row-independent, so the pad rows cannot reach the real ones; they
    are copies of the chunk's first row purely because that is guaranteed to be
    in-domain, and they are dropped before anything accumulates.
    """
    batched = eqx.filter_jit(jax.vmap(one))
    f32 = lambda a: None if a is None else jnp.asarray(a, dtype=jnp.float32)
    n = len(arrays[0])
    # Pad to min(batch, n), NEVER to `batch`: callers pass a batch far larger
    # than the catalog (`ess` uses n=1 with batch=20000), and padding one row up
    # to 20000 asked for a 16 GiB allocation and OOMed.  There is still exactly
    # one shape per call, which is the whole point.
    nb = min(batch, n)
    out = []
    for i in range(0, n, nb):
        idx = np.arange(i, min(i + nb, n))
        k = len(idx)
        if k < nb:
            idx = np.concatenate([idx, np.full(nb - k, idx[0])])
        chunk = tuple(f32(None if a is None else a[idx]) for a in arrays)
        out.append(jax.tree.map(lambda a: np.asarray(a[:k], dtype=np.float64),
                                batched(*chunk)))
    return jax.tree.map(lambda *a: np.concatenate(a), *out)


def pqr(flow, m, draws=None, log_wt=None, batch=20000, sigma_x=None):
    """Per-target (d logP/dg, d2 logP/dg2) at g = 0, as float64.

    With `draws`/`log_wt` the target is noisy and P is the convolution above;
    without them the target is noiseless and P is a point evaluation of the
    prior -- a single flow call, no sampling, no weights.

    `sigma_x` switches on the centroid layer.  It enters the condition but NOT
    the differentiation: grad and hessian are taken with respect to the 2-vector
    g while Sigma_X is held fixed, because the target's noise properties are
    data, not something lensing moves.  Q and R therefore keep their eq. (12-13)
    meaning with the centroid marginalisation folded into P.
    """
    zero = jnp.zeros(2)

    def one(m_i, draws_i, log_wt_i, sigma_x_i):
        cond = lambda g: condition(g, sigma_x_i)
        f = (lambda g: flow.log_prob(m_i, condition=cond(g))) if draws is None else (
            lambda g: log_conv_is(flow, m_i, draws_i, log_wt_i, cond(g)))
        return jax.grad(f)(zero), jax.hessian(f)(zero)

    return _over_targets(one, (m, draws, log_wt, sigma_x), batch)


def pqr_full(flow, m, draws=None, log_wt=None, batch=20000, sigma_x=None, ok=None):
    """Per-target (log P, d logP/dg, d2 logP/dg2) at g = 0, as float64.

    Same convolution as `pqr` (noiseless without `draws`/`log_wt`, else the
    C_M integral), but also returns the value.  `pqr` doesn't, because nothing
    upstream of it needs it: `ghat`/`bias` only ever sum Q and R.  It exists
    for `write_logpqr.py` -- bfd's packed PQR (`bfd.pqr.packPqr`) has a P slot
    alongside Q and R, and here it is log P, NOT bfd's own raw P (see that
    module's docstring: `bfd.pqr.logPqr` converts a RAW packed p,q,r to a log
    one by dividing by p; ours is already log, so never round-trip it through
    that function -- there is no finite p to divide by, `log P` is O(-40) so
    `p = exp(log P)` is float32 noise, which is exactly why this project
    differentiates `log P` directly instead of forming P, Q, R separately).

    One extra forward pass over `pqr`'s jvp trick (value_and_grad already
    computes the value for free), so this is not the streamed one -- no
    streaming, or the merge's known chunk-Hessian fragility applies here too;
    see `pqr_streamed`'s docstring and this session's own grounding of it.

    `ok`, if given, is the domain mask computed on `draws` BEFORE any centroid
    peel (`in_domain(draws_raw)`), same as `pqr_streamed` threads through --
    `pqr` itself does not accept this and instead lets `log_conv_is` recompute
    `in_domain` on whatever it is handed, which is wrong once the caller has
    already peeled `draws` to standardised coordinates (see that function's
    docstring).  Measured harmless in practice (HANDOFF.md's `_mixture_chunk`
    finding: the affected draws carry ~0 weight) but there is no reason to
    reintroduce it here.
    """
    zero = jnp.zeros(2)

    def one(m_i, draws_i, log_wt_i, sigma_x_i, ok_i):
        cond = lambda g: condition(g, sigma_x_i)
        f = (lambda g: flow.log_prob(m_i, condition=cond(g))) if draws is None else (
            lambda g: log_conv_is(flow, m_i, draws_i, log_wt_i, cond(g), ok_i))
        vg = jax.value_and_grad(f)
        # ONE linearization, two tangents.  Two `jax.jvp` calls re-ran the
        # primal (a full forward+reverse traversal of the flow over every
        # draw) and threw the second one away; `linearize` shares it.
        (val, df), lin = jax.linearize(vg, zero)
        _, h0 = lin(jnp.array([1.0, 0.0]))
        _, h1 = lin(jnp.array([0.0, 1.0]))
        return val, df, jnp.stack([h0, h1], axis=-1)

    return _over_targets(one, (m, draws, log_wt, sigma_x, ok), batch)


def _merge_init(n):
    """Running state for the chunk merge.

    Chunks are INDEPENDENT draw sets, so keeping them separate (rather than
    only their running total) is what makes the delete-one jackknife in
    `_merge_finish` possible.  Per chunk we keep `log A_c` and the two
    normalised ratios `B_c/A_c`, `C_c/A_c`; the absolute scale of A never
    leaves log space, so nothing can overflow.  Storage is
    `n_chunks x 7 x batch` floats -- 28k at 64 chunks and batch 64.
    """
    return {"la": [], "b": [], "c": []}


def _merge_chunk(st, la, df, c_over_a, good):
    """Record one chunk's (log A_c, B_c/A_c, C_c/A_c)."""
    st["la"].append(np.where(good, la, -np.inf))
    st["b"].append(np.where(good[:, None], df, 0.0))
    st["c"].append(np.where(good[:, None, None], c_over_a, 0.0))
    return st


def _shares(la):
    """A_c / sum_c A_c, computed stably; all -inf gives all zeros."""
    mx = la.max(0)
    mxs = np.where(np.isfinite(mx), mx, 0.0)
    w = np.where(np.isfinite(la), np.exp(la - mxs), 0.0)
    tot = w.sum(0)
    return np.where(tot > 0.0, w / np.where(tot > 0.0, tot, 1.0), 0.0), tot > 0.0


def _merge_finish(st, jackknife=True, min_left=0.05):
    """(Q, R, n_fallback) from the per-chunk record, optionally bias-corrected.

    The plain estimator is `Q = B/A`, `R = C/A - (B/A)(B/A)^T`, and it carries
    TWO O(1/S) biases that partly cancel:

      * `(B/A)(B/A)^T` is a squared Monte-Carlo estimate, so
        `E[(B/A)(B/A)^T] = q q^T + Cov_MC(B/A)` and R comes out LARGE;
      * `C/A` and `B/A` are self-normalised ratios, whose own O(1/S) ratio bias
        pulls the other way.

    Measured in closed form (`tests/test_pqr_crossfit.py`, a Gaussian where
    `P(M|g) = N(M; g, 1+s^2)` so `j` is known): at 8 chunks of 48 draws the
    square contributes +0.00080 and the `C/A` ratio -0.00055, i.e. **opposite
    signs and the same order**, for a net +0.00025.  So the obvious cross-fit /
    U-statistic on `(B/A)^2` alone is a trap: it was tried first and it does
    cleanly kill its own term (+0.00080 -> +0.00001), but that leaves the ratio
    bias uncancelled and the total is then no better -- sometimes worse,
    sometimes better, depending on the configuration.  Not a reliable gain.

    So debias the estimator as a whole instead: a delete-one jackknife over the
    chunks.  With `theta_hat` the full-sample value and `theta_(e)` the value
    recomputed dropping chunk `e`,

        theta_jack = k theta_hat - (k - 1) mean_e theta_(e)

    which removes the entire leading O(1/S) bias of any smooth function of the
    chunk sums, whatever its source.  The leave-one-out sums cost nothing: with
    `a_c = A_c / A_tot`,

        B/A dropping e  =  (bhat - a_e b_e) / (1 - a_e)

    and likewise for `C/A`, so it is arithmetic on what is already stored.

    Both Q and R are corrected.  Q's own O(1/S) bias is odd in g by isotropy, so
    it acts multiplicatively on m1 exactly as R's does, and there is no reason
    to fix one and not the other.

    Falls back to the plain estimator, per target, when fewer than two chunks
    carry weight or one chunk holds more than `1 - min_left` of it: the
    jackknife then divides by a vanishing `1 - a_e`.  The count is returned.
    """
    la = np.stack(st["la"])                       # (k, n)
    b = np.stack(st["b"])                         # (k, n, 2)
    c = np.stack(st["c"])                         # (k, n, 2, 2)
    a, alive = _shares(la)                        # (k, n)

    bhat = np.einsum("kn,kna->na", a, b)
    chat = np.einsum("kn,knab->nab", a, c)
    r_plain = chat - np.einsum("na,nb->nab", bhat, bhat)
    k = len(la)
    if not jackknife or k < 2:
        return bhat, r_plain, int(alive.sum()) if k < 2 and jackknife else 0

    # delete-one, vectorised over chunks
    left = 1.0 - a                                                  # (k, n)
    ok = alive & (left > min_left).all(0) & (np.isfinite(la).sum(0) >= 2)
    safe = np.where(left > min_left, left, 1.0)
    b_e = (bhat[None] - a[..., None] * b) / safe[..., None]         # (k, n, 2)
    c_e = (chat[None] - a[..., None, None] * c) / safe[..., None, None]
    r_e = c_e - np.einsum("kna,knb->knab", b_e, b_e)

    q_j = k * bhat - (k - 1) * b_e.mean(0)
    r_j = k * r_plain - (k - 1) * r_e.mean(0)
    q = np.where(ok[:, None], q_j, bhat)
    r = np.where(ok[:, None, None], r_j, r_plain)
    return q, r, int((alive & ~ok).sum())


def _shares_j(la):
    """`_shares` on device.  Same expression, jnp instead of np."""
    mx = la.max(0)
    mxs = jnp.where(jnp.isfinite(mx), mx, 0.0)
    w = jnp.where(jnp.isfinite(la), jnp.exp(la - mxs), 0.0)
    tot = w.sum(0)
    return jnp.where(tot > 0.0, w / jnp.where(tot > 0.0, tot, 1.0), 0.0), tot > 0.0


def _merge_finish_j(la, b, c, jackknife=True, min_left=0.05):
    """`_merge_finish` on device, taking STACKED chunk arrays.

    Identical arithmetic to the host version -- see that docstring for why the
    delete-one jackknife is there at all (it removes the whole leading O(1/S)
    bias of the ratio estimator, worth 0.0026 in m1, so it is not optional).
    The only difference is that `lax.scan` hands the chunks over already
    stacked, so there is no `np.stack` and no Python list.

    Returns (q, r, n_fallback) with `n_fallback` a device scalar.
    """
    a, alive = _shares_j(la)
    bhat = jnp.einsum("kn,kna->na", a, b)
    chat = jnp.einsum("kn,knab->nab", a, c)
    r_plain = chat - jnp.einsum("na,nb->nab", bhat, bhat)
    k = la.shape[0]
    if not jackknife or k < 2:
        return bhat, r_plain, jnp.array(0)

    left = 1.0 - a
    ok = alive & (left > min_left).all(0) & (jnp.isfinite(la).sum(0) >= 2)
    safe = jnp.where(left > min_left, left, 1.0)
    b_e = (bhat[None] - a[..., None] * b) / safe[..., None]
    c_e = (chat[None] - a[..., None, None] * c) / safe[..., None, None]
    r_e = c_e - jnp.einsum("kna,knb->knab", b_e, b_e)

    q_j = k * bhat - (k - 1) * b_e.mean(0)
    r_j = k * r_plain - (k - 1) * r_e.mean(0)
    q = jnp.where(ok[:, None], q_j, bhat)
    r = jnp.where(ok[:, None, None], r_j, r_plain)
    return q, r, (alive & ~ok).sum()


def _merge_opg(st):
    """Per-target cross-fit outer product `q q^T`, as (n, 2, 2).

    The Fisher-scoring alternative to `R` needs `E[q q^T]`, and the obvious
    `qhat qhat^T` is biased by exactly the Monte-Carlo noise it squares:
    `E[qhat qhat^T] = q q^T + Var[eps]`, which OVERSTATES the information and
    shrinks `ghat`.  Measured on `gauss2_deep` as a -3.5% m1 offset that decays
    like O(1/S) (-0.01187 at S = 8192, -0.00979 at 32768, against a Newton
    -0.00907).

    Splitting the chunks into two halves and taking the CROSS product removes
    it outright rather than correcting it: the halves are independent draw
    sets, so `E[qhat^A qhat^B^T] = q q^T` with no `Var[eps]` term and no fitted
    constant.  Symmetrised, since only the symmetric part is a metric.

    Each half is renormalised over its own chunks (`_shares` on that subset),
    so each is the same self-normalised ratio estimator the full sample uses,
    just at half the draws.  Falls back to the plain `bhat bhat^T` when there
    is only one chunk, where there is nothing to cross-fit.
    """
    la = np.stack(st["la"])
    b = np.stack(st["b"])
    k = len(la)
    a_all, _ = _shares(la)
    bhat = np.einsum("kn,kna->na", a_all, b)
    if k < 2:
        return np.einsum("na,nb->nab", bhat, bhat)
    h = k // 2
    qs = []
    for sl in (slice(0, h), slice(h, k)):
        a_h, _ = _shares(la[sl])
        qs.append(np.einsum("kn,kna->na", a_h, b[sl]))
    qa, qb = qs
    cross = np.einsum("na,nb->nab", qa, qb)
    out = 0.5 * (cross + cross.transpose(0, 2, 1))
    # A target with no weight in one half has an undefined cross term; the
    # plain square is the only thing left and it is what `ghat` would have
    # used anyway.
    bad = ~np.isfinite(out).reshape(len(out), -1).all(1)
    if bad.any():
        out[bad] = np.einsum("na,nb->nab", bhat[bad], bhat[bad])
    return out


def pqr_streamed(flow, m, cov, samples, alpha, seed, sigma_x=None,
                 batch=64, chunk=2048, report=None,
                 proposal=None, proposal_sigma_x=None, jackknife=True,
                 opg=False, gauge="prior", lam_stride=1):
    """Per-target (d logP/dg, d2 logP/dg2) WITHOUT materialising the draws.

    `Phat = (1/S) sum_s w_s p(x_s|g)` is a plain sum over samples, and so are its
    two g-derivatives, so a target needs only three running accumulators

        A = sum_s w_s p_s      B = sum_s w_s grad p_s     C = sum_s w_s hess p_s

    from which `grad log P = B/A` and `hess log P = C/A - (B/A)(B/A)^T`.  That
    last square is a squared MONTE-CARLO estimate, so R comes out biased LARGE
    by `Cov_MC(B/A)` at O(1/S) -- ghat too small, m1 too negative, flat in g,
    and invisible to the weight ESS -- while the self-normalised ratios pull the
    other way.  `jackknife=True` (the default) removes the whole leading O(1/S)
    bias with a delete-one jackknife over the chunks; see `_merge_finish`, and
    note it CHANGES EVERY NUMBER relative to runs made before it existed.  Pass
    `jackknife=False` to reproduce those.  Memory
    is then O(targets) instead of O(targets x samples): the (n, S, 5) draw array
    is 26 GB per catalog at n = 20000, S = 32768, and it never exists here.  That
    is what makes a large S affordable, which is the only thing that fixes the
    ESS starvation at depth.

    Each chunk is evaluated with the SAME `log_conv_is` the non-streamed path
    uses, then combined: a chunk returns `f = log(A_c / n_c)`, `grad f = B_c/A_c`
    and `hess f = C_c/A_c - (B_c/A_c)(B_c/A_c)^T`, which is enough to recover
    `A_c`, `B_c/A_c` and `C_c/A_c` and merge them as an A-weighted mean.  The
    merge runs in log space against a running max, so no chunk can overflow.

    Chunks are seeded per (target batch, chunk index), so the draws are common
    random numbers across catalogs exactly as before -- at alpha = 1.  With
    alpha < 1 part of the proposal comes from the flow itself, so two runs with
    DIFFERENT flows no longer share draws and the pairing weakens; that is a
    property of the defensive mixture, not of the streaming.  `proposal` (with
    `proposal_sigma_x`) restores it: `mixture_draws` draws its flow component
    and evaluates `log_q` from `proposal`/`proposal_sigma_x` instead of
    `flow`/`sigma_x`, while everything downstream of the draws --
    `split_centroid(flow)`, the centroid transform of draws and targets, and
    `log_conv_is` -- keeps evaluating `flow` at `sigma_x`.  `proposal is None`
    (the default) is exactly today's behavior.

    Accuracy, measured against a single logsumexp over the SAME draws:

        chunks   dQ/Q     dR/R
             1   3.3e-7   8.0e-5     (no merge -- exact)
             2   5.3e-3   9.8e-3
             4   3.7e-3   4.2e-3

    The merge is algebraically exact; the residual is float32 in the per-chunk
    `d2f`, which `c_over_a` has to un-cancel.  It is conditioned on the
    per-chunk ESS, and the table above is a worst case: 256 draws where the ESS
    is 24, i.e. ~4 effective samples per chunk.  Use the largest chunk that
    fits.  The perturbation is random per target, so it enters the eq. (45)-(46)
    ensemble sums as ~dR/sqrt(N) -- 4e-5 at N = 20000 -- and a component common
    to all targets cancels between +g and -g anyway.

    `gauge` picks WHERE the shear acts inside the estimator.  "prior" is the
    original: draws fixed, `p(. | g)` evaluated, so the derivatives land on the
    flow's score.  "kernel" substitutes `m = Psi_g(u)` and puts g in the noise
    kernel instead -- the same integral, identical at g = 0, but with a bounded
    per-draw derivative.  It is not a refinement: under the posterior the score's
    Hill tail index is 1.20 on bulgedisc_v2's faint end, so "prior"'s
    `Var[d_g log p]` does not exist and no draw count converges it (see
    `log_conv_is_kernel` and `dev/hmc_ref.py`).  Default stays "prior" so old
    runs reproduce.
    """
    if gauge not in ("prior", "kernel", "auto"):
        raise ValueError(
            f"gauge must be 'prior', 'kernel' or 'auto'; got {gauge!r}")
    zero = jnp.zeros(2)
    # `flow` still carries the centroid layer, because `mixture_draws` needs the
    # FULL prior to build its proposal.  Everything downstream of the transform
    # must use the peeled flow instead -- evaluating `flow` on already
    # transformed draws would apply the layer a second time.
    flow_g, layer = split_centroid(flow)

    def one(m_i, draws_i, log_wt_i, sigma_x_i, ok_i):
        f = lambda g: log_conv_is(flow_g, m_i, draws_i, log_wt_i,
                                  condition(g, sigma_x_i), ok_i)
        # One forward-over-reverse pass per g direction yields the value, the
        # gradient AND a column of the Hessian.  Calling f, jax.grad(f) and
        # jax.hessian(f) separately -- as this did -- evaluates the flow three
        # times over, and `hessian` is `jacfwd(grad)` so it already computed the
        # first two internally and discarded them.
        return _val_grad_hess(f)

    # Gauge K differentiates the same estimator with g moved into the kernel;
    # see `log_conv_is_kernel` for why.  `c` arrives precomputed, so the flow is
    # not in the autodiff at all here -- only `psi`.
    # Gauge K needs only the POINT `Psi_g(u)`; gauge auto needs its log-det too,
    # and gets it by the chain rule rather than by differentiating the whole
    # composition (`make_psi_ld`).
    # NOT worth memoizing across arms: measured null (74.8 s -> 77.5 s, noise).
    # `eqx.filter_jit` takes the WHOLE `batched` function object as a static
    # argument hashed by identity, so pinning `psi` alone changes nothing --
    # the per-arm recompile is `batched` itself being rebuilt, not `psi`.
    psi = (make_psi_ld(flow, peeled=layer is not None) if gauge == "auto" else
           make_psi(flow, peeled=layer is not None) if gauge == "kernel" else
           None)
    cinv = jnp.asarray(np.linalg.inv(cov), jnp.float32)

    def _auto_glue(m_raw_i, x_i, d_raw_i):
        """Per-draw masking and `r0`, INSIDE the kernel.

        This ran eagerly op by op between jit calls (`in_domain`, two `where`s,
        a `safe_point` vmap, a subtract) on (batch, chunk, 5) arrays.  The
        profiler charged those 47 ms of GPU IDLE per 6 batches -- the card sat
        waiting while Python dispatched them one at a time.  Folded in here
        they fuse with the flow work and cost no launches at all.

        Same trap as gauge K: `where` masks the VALUE but 0 * inf NaNs the
        gradient, so an off-chart draw must be stood in with a safe point
        before it reaches the flow or `psi`.
        """
        ok_i = in_domain(d_raw_i)
        x_i = jnp.where(ok_i[:, None], x_i, safe_point(m_raw_i)[None, :])
        d_raw_i = jnp.where(ok_i[:, None], d_raw_i, m_raw_i[None, :])
        return x_i, m_raw_i[None, :] - d_raw_i, ok_i

    def one_b(m_raw_i, x_i, d_raw_i, lw_i, lam_i, sigma_x_i):
        x_i, r0_i, ok_i = _auto_glue(m_raw_i, x_i, d_raw_i)
        f = lambda g: log_conv_is_blend(flow_g, psi, m_raw_i, x_i, r0_i, lw_i,
                                        g, sigma_x_i, cinv, lam_i, ok_i)
        vg = jax.value_and_grad(f)
        # ONE linearization, two tangents.  Two `jax.jvp` calls re-ran the
        # primal (a full forward+reverse traversal of the flow over every
        # draw) and threw the second one away; `linearize` shares it.
        (val, df), lin = jax.linearize(vg, zero)
        _, h0 = lin(jnp.array([1.0, 0.0]))
        _, h1 = lin(jnp.array([0.0, 1.0]))
        return val, df, jnp.stack([h0, h1], axis=-1)

    # One leading dim for every jitted call, so the short LAST target batch is
    # not a second compile of these -- see `_fixed_batch`.  `nb` rather than
    # `batch` so a run with fewer targets than a batch does not pad up to a
    # size it never needs.
    nb = min(batch, len(m))
    def _lam_one(mr, x, d_raw, lw, sx):
        x, r0, ok = _auto_glue(mr, x, d_raw)
        return _blend_lambda(flow_g, psi, mr, x, r0, lw, sx, cinv, ok,
                             lam_stride)

    batched_lam = _fixed_batch(eqx.filter_jit(jax.vmap(_lam_one)), nb)

    def one_k(r0_i, x_i, dr_i, c_i, sigma_x_i, ok_i):
        f = lambda g: log_conv_is_kernel(psi, r0_i, x_i, dr_i, c_i, g,
                                         sigma_x_i, cinv, ok_i)
        vg = jax.value_and_grad(f)
        # ONE linearization, two tangents.  Two `jax.jvp` calls re-ran the
        # primal (a full forward+reverse traversal of the flow over every
        # draw) and threw the second one away; `linearize` shares it.
        (val, df), lin = jax.linearize(vg, zero)
        _, h0 = lin(jnp.array([1.0, 0.0]))
        _, h1 = lin(jnp.array([0.0, 1.0]))
        return val, df, jnp.stack([h0, h1], axis=-1)

    batched = _fixed_batch(eqx.filter_jit(jax.vmap(
        {"prior": one, "kernel": one_k, "auto": one_b}[gauge])), nb)
    batched_c = _fixed_batch(eqx.filter_jit(jax.vmap(
        lambda x, lw, mr, sx, ok: _gauge_k_c(
            flow_g, x, lw, mr, sx, ok, layer is not None))), nb)
    n_chunks = max(1, samples // chunk)

    # ------------------------------------------------------------------ #
    # Device path: one jitted call per TARGET BATCH.  The host sends the
    # moments and Sigma_X and gets back Q and R -- the draws, the masking, the
    # per-chunk Q/R and the whole delete-one merge stay on the GPU, and the
    # chunk loop is a `lax.scan` rather than a Python loop with a D2H round
    # trip per chunk.  Profiled at 95.4% duty before this; the round trip and
    # the Python dispatch around it were what was left.
    #
    # `--gauge auto` only, and not with `--opg`: `prior` and `kernel` are both
    # rejected gauges ([[r-is-an-is-artifact]]), so putting them in here would
    # triple the traced surface for paths nobody runs.  They keep the loop
    # below, unchanged.
    # ------------------------------------------------------------------ #
    n_k_c = round(alpha * chunk)
    n_k_c -= n_k_c % 2                    # kernel half stays antithetic
    n_f_c = chunk - n_k_c
    # OPT-IN (`BFD_DEVICE_PQR=1`), not the default.  MEASURED 2026-09-07 at
    # N=2000: it does what it says -- GPU duty 95.4% -> 98.4%, idle 265.6 ms ->
    # 40.7 ms across the profiled window, the host reduced to sending moments
    # and Sigma_X and taking back Q and R -- and it is still SLOWER end to end.
    # Span for the same work is flat (2584.7 ms vs 2515.1 ms) because GPU busy
    # time ROSE (2378 -> 2544 ms, kernels 58434 -> 62412: the scan and the
    # `lax.cond` cost real compute), and wall is 52% worse (727 s vs 479 s)
    # because the whole flow now traces inside a scan and `cond` traces both
    # branches.  Host stalls were only 4.6% of the window to begin with; the
    # bottleneck is the flow arithmetic, not the round trip.  Outputs are
    # statistically consistent (windowed m1 differs 5.6e-03 on a different RNG
    # stream, against ~1.2e-02 draw noise).  Worth revisiting if the compile
    # cost is brought down -- the persistent cache is the obvious lever.
    use_device = (gauge == "auto" and not opg and 0.0 < alpha < 1.0
                  and n_f_c > 0 and os.environ.get("BFD_DEVICE_PQR"))

    if use_device:
        _L = jnp.asarray(np.linalg.cholesky(cov), jnp.float32)
        _Lt = _L.T
        _log_diag = jnp.sum(jnp.log(jnp.diag(_L)))
        _log_alpha = float(np.log(alpha))
        _log_1ma = float(np.log1p(-alpha))
        _prop = flow if proposal is None else proposal
        _lam_v = jax.vmap(_lam_one)
        _qr_v = jax.vmap(one_b)

        def _draw_chunk(m_raw_j, sx_j, psx_j, key_c):
            """One chunk's draws + centroid transform, on device."""
            kk, kf = jr.split(key_c)
            half = jr.normal(kk, (m_raw_j.shape[0], n_k_c // 2, 5),
                             dtype=jnp.float32) @ _Lt
            kern = jnp.concatenate([half, -half], axis=1)
            d, lw = _mixture_chunk(_prop, m_raw_j, kern,
                                   sx_j if proposal is None else psx_j,
                                   kf, _L, _log_diag, _log_alpha, _log_1ma,
                                   n_f_c)
            lw = lw.astype(jnp.float32)
            d_raw = d.astype(jnp.float32)
            if layer is not None:
                d, ld = centroid_transform(layer, d, sx_j, to_host=False)
                lw = lw + ld
            else:
                d = d_raw
            return d, d_raw, lw

        def _record(f, df, d2f):
            """One chunk's (log A_c, B_c/A_c, C_c/A_c), masked -- `_merge_chunk`
            on device."""
            la = f + jnp.log(chunk)
            c_over_a = d2f + jnp.einsum("ia,ib->iab", df, df)
            good = (jnp.isfinite(la) & jnp.isfinite(df).all(1)
                    & jnp.isfinite(c_over_a).reshape(la.shape[0], -1).all(1))
            return (jnp.where(good, la, -jnp.inf),
                    jnp.where(good[:, None], df, 0.0),
                    jnp.where(good[:, None, None], c_over_a, 0.0))

        @eqx.filter_jit
        def _batch_device(m_raw_j, sx_j, psx_j, key_b):
            # EVERY chunk goes through the scan, including chunk 0, so only one
            # chunk's intermediates are ever live.  Unrolling chunk 0 alongside
            # the scan (to fit `lam` outside it) put two chunks in one HLO
            # module and XLA asked for 16.6 GiB on a 16 GiB card.
            #
            # `lam` is fit on chunk 0 and then CARRIED: it must be identical
            # across chunks, or the merge combines estimators of different
            # variance-optimal blends.  `lax.cond` runs the fit only on c == 0
            # -- a `jnp.where` would refit it every chunk, and the fit is ~10%
            # of runtime.
            def body(lam, c):
                d, dr, lw = _draw_chunk(m_raw_j, sx_j, psx_j,
                                        jr.fold_in(key_b, c))
                lam = jax.lax.cond(
                    c == 0,
                    lambda: _lam_v(m_raw_j, d, dr, lw, sx_j),
                    lambda: lam)
                return lam, _record(*_qr_v(m_raw_j, d, dr, lw, lam, sx_j))

            lam0 = jnp.zeros(m_raw_j.shape[0], jnp.float32)
            _, (la, b, c_) = jax.lax.scan(body, lam0, jnp.arange(n_chunks))
            q, r, n_fb = _merge_finish_j(la, b, c_, jackknife)
            return q, r, n_fb, (~_shares_j(la)[1]).sum()

    q_out, r_out, opg_out = [], [], []
    n_empty = 0
    n_fallback = 0

    for i in range(0, len(m), batch):
        # BFD_PROFILE=start:stop -- trace batches [start, stop) with
        # jax.profiler and NO added barriers, so the GPU timeline shows
        # the real duty cycle.  Off unless the env var is set.
        if _PROF is not None:
            _bi = i // batch
            if _bi == _PROF[0]:
                jax.profiler.start_trace(_PROF[2])
            if _bi == _PROF[1]:
                jax.block_until_ready(q_out[-1] if q_out else None)
                jax.profiler.stop_trace()
                print(f'  profile written to {_PROF[2]}', flush=True)
                raise SystemExit(0)
        if _TIME is not None:
            _t_batch = time.perf_counter()
        m_b = m[i:i + batch]
        sx_b = None if sigma_x is None else sigma_x[i:i + batch]
        psx_b = None if proposal_sigma_x is None else proposal_sigma_x[i:i + batch]

        if use_device:
            # Everything for this batch happens in ONE jitted call: the host
            # hands over the moments and Sigma_X and takes back Q and R.
            q_b, r_b, n_fb, n_emp = _batch_device(
                jnp.asarray(m_b, dtype=jnp.float32),
                None if sx_b is None else jnp.asarray(sx_b, jnp.float32),
                None if psx_b is None else jnp.asarray(psx_b, jnp.float32),
                jr.fold_in(jr.key(seed), i // batch))
            q_out.append(np.asarray(q_b, dtype=np.float64))
            r_out.append(np.asarray(r_b, dtype=np.float64))
            n_fallback += int(n_fb)
            n_empty += int(n_emp)
            if report and (i // batch) % report == 0:
                print(f"    {i + len(m_b)}/{len(m)} targets", flush=True)
            continue

        # The merge runs in float32.  It used to be float64 on the host, on the
        # grounds that rebuilding `c_over_a` as `hess + (B/A)(B/A)^T` undoes the
        # cancellation in `hess = C/A - (B/A)(B/A)^T` and costs 1.7% on R at low
        # ESS.  MEASURED 2026-09-07 on 1998 real targets, stratified by the very
        # quantity that drives it: float32 costs 2.4e-07 (median relative on R)
        # even in the top 1% of |B/A| (up to 16.6), 4.1e-07 on the ensemble sum,
        # and loses no target to non-finiteness.  The 1.7% does not reproduce --
        # `d2f` arrives from the device ALREADY float32, so that cancellation has
        # happened in single precision before the host ever sees it and the
        # float64 merge was guarding a step that was already lossy upstream.
        # The ENSEMBLE sum in `ghat` is still float64 and must stay so: that one
        # runs over ~1M galaxies, where float32 really would lose the signal.
        st = _merge_init(len(m_b))
        # The targets' own transform does not depend on the chunk, so do it once
        # per target batch rather than once per chunk.
        m_z = (jnp.asarray(m_b, dtype=jnp.float32) if layer is None else
               centroid_transform(layer, m_b, sx_b, to_host=False)[0])

        for c in range(n_chunks):
            # The seed carries BOTH the chunk and the target batch.  Without the
            # batch term every target batch would replay the same offsets, so
            # targets in different batches would share a noise realization --
            # the correlation `kernel_draws` is documented to avoid, because
            # correlated per-target errors stop averaging down in the eq.
            # (45)-(46) ensemble sums.  Note this makes `batch` part of the
            # random stream: two runs to be compared must use the same one.
            if _TIME is not None:
                _t = time.perf_counter()
            # `to_host=False`: these go straight back into `batched`, so the
            # numpy round trip is pure loss and its `np.asarray` drained the
            # pipeline once per chunk.  Bit-identical values.
            d, lw = mixture_draws(
                flow if proposal is None else proposal, m_b, cov, chunk, alpha,
                seed + 7919 * c + 104729 * (i // batch),
                batch=len(m_b), sigma_x=sx_b if proposal is None else psx_b,
                to_host=False)
            lw = jnp.asarray(lw, dtype=jnp.float32)
            # Test the domain on the RAW draw, while it still means something:
            # once the peel has run, the draw is in standardised coordinates and
            # the chart's ceilings are behind it.  With no peel, `log_conv_is`
            # does this itself on the same quantity.
            # ... and hand it down ONLY when the peel has run.  With no peel
            # `draws` are still raw, and `log_conv_is`'s `ok is not None` branch
            # stands bad rows in at ZERO -- a valid standardised coordinate but
            # not a valid raw moment (Mf = 0 -> log10(0)).  The `where` masks it
            # out of the value and 0 * inf NaNs the GRADIENT, killing every
            # target with even one off-chart draw (16% of bulgedisc).  `ok=None`
            # makes log_conv_is recompute the same mask with `safe_point` as the
            # stand-in, which is the whole reason that function exists.
            # Gauge K needs the mask explicitly (it never calls `log_conv_is`,
            # so nothing downstream would recompute it) and needs the RAW draw
            # kept, because the noise kernel is Gaussian in raw moments.
            d_raw = jnp.asarray(d, dtype=jnp.float32)
            # Gauge auto computes this inside its kernel (`_auto_glue`), so
            # doing it here too would be a 20 ms eager stall for nothing.
            ok_raw = (in_domain(d_raw)
                      if gauge != "auto" and (layer is not None
                                              or gauge != "prior") else None)
            if layer is not None:
                # Stay on device: the transformed draws go straight back into a
                # jitted function, so a host round-trip here is pure loss.
                d, ld = centroid_transform(layer, d, sx_b, to_host=False)
                lw = lw + ld
            else:
                d = d_raw
            sx_j = None if sx_b is None else jnp.asarray(sx_b, dtype=jnp.float32)
            if gauge == "auto":
                m_raw_j = jnp.asarray(m_b, dtype=jnp.float32)
                # The masking and `r0` now live INSIDE the kernel (`_auto_glue`)
                # rather than as eager ops here -- see that docstring.
                # lam is a property of the target, not of the chunk, so fit it
                # once on the first chunk and reuse it: it must be the SAME
                # across chunks or the merge would be combining estimators of
                # different variance-optimal blends.
                if _TIME is not None:
                    jax.block_until_ready((d, d_raw, lw))
                    _TIME["draw"] += time.perf_counter() - _t
                if c == 0:
                    if _TIME is not None:
                        _t2 = time.perf_counter()
                    lam_b = batched_lam(m_raw_j, d, d_raw, lw, sx_j)
                    if _TIME is not None:
                        jax.block_until_ready(lam_b)
                        _TIME["lambda"] += time.perf_counter() - _t2
                if _TIME is not None:
                    _t3 = time.perf_counter()
                f, df, d2f = batched(m_raw_j, d, d_raw, lw, lam_b, sx_j)
                if _TIME is not None:
                    _TIME["dispatch"] += time.perf_counter() - _t3
            elif gauge == "kernel":
                m_raw_j = jnp.asarray(m_b, dtype=jnp.float32)
                c_k = batched_c(d, lw, m_raw_j, sx_j, ok_raw)
                # Stand the off-chart draws in with a safe point BEFORE `psi`
                # ever sees them.  Masking only the logsumexp would leave `psi`
                # producing an inf whose 0 * inf gradient NaNs the whole target
                # -- 37.5% of draws, the same trap `log_conv_is`'s `dummy`
                # exists for.  These rows are already -inf in the value.
                dummy = (jnp.zeros_like(d[:, :1, :]) if layer is not None else
                         jax.vmap(safe_point)(m_raw_j)[:, None, :])
                d = jnp.where(ok_raw[..., None], d, dummy)
                d_raw = jnp.where(ok_raw[..., None], d_raw, m_raw_j[:, None, :])
                r0 = m_raw_j[:, None, :] - d_raw
                f, df, d2f = batched(r0, d, d_raw, c_k, sx_j, ok_raw)
            else:
                f, df, d2f = batched(m_z, d, lw, sx_j, ok_raw)
            if _TIME is not None:
                # `np.asarray` below BLOCKS on the device, so without an
                # explicit barrier here the GPU's time would be charged to the
                # host merge and the two could not be told apart.  This is the
                # whole point of the breakdown: `utilization.gpu` only reports
                # that A kernel was resident, not that the SMs were busy, so it
                # cannot answer "host-bound or not".
                _t = time.perf_counter()
                jax.block_until_ready((f, df, d2f))
                _TIME["device"] += time.perf_counter() - _t
                _t = time.perf_counter()
            f = np.asarray(f, dtype=np.float32)
            df = np.asarray(df, dtype=np.float32)
            d2f = np.asarray(d2f, dtype=np.float32)
            # f = log(A_c / n_c); recover C_c/A_c from hess = C/A - (B/A)(B/A)^T
            la = f + np.log(chunk)
            c_over_a = d2f + np.einsum("ia,ib->iab", df, df)

            # A chunk contributes only if it carries weight and its derivatives
            # are finite.  A chunk whose draws all underflow gives la = -inf,
            # and with the accumulator also still -inf the max trick computes
            # -inf - -inf = NaN.  That is not hypothetical: at 20000 targets it
            # hit 1 (the faintest in the catalog, flux 755) and a single NaN row
            # takes out R_tot and with it m1, c1 and c2 for the WHOLE ensemble.
            good = (np.isfinite(la) & np.isfinite(df).all(1)
                    & np.isfinite(c_over_a).reshape(len(la), -1).all(1))
            st = _merge_chunk(st, la, df, c_over_a, good)
            if _TIME is not None:
                _TIME["merge"] += time.perf_counter() - _t

        if _PROF is None and os.environ.get("BFD_DUMP_MERGE") and i // batch < 200:
            # The merge is float64 on the host on purpose; dump its INPUTS so
            # the cost of doing it in float32 instead can be measured offline
            # against the same numbers, rather than argued about.
            np.savez(f"{os.environ['BFD_DUMP_MERGE']}/merge_{i // batch:03d}.npz",
                     la=np.stack(st["la"]), b=np.stack(st["b"]),
                     c=np.stack(st["c"]))
        if _TIME is not None:
            _t = time.perf_counter()
        q_b, r_b, n_fb = _merge_finish(st, jackknife)
        q_out.append(q_b)
        r_out.append(r_b)
        if opg:
            opg_out.append(_merge_opg(st))
        n_fallback += n_fb
        n_empty += int((~_shares(np.stack(st["la"]))[1]).sum())
        if _TIME is not None:
            # `finish` is per-BATCH host work -- the delete-one jackknife and
            # `_shares`, both numpy -- and sits outside the chunk loop, so it
            # was invisible to the per-chunk buckets.
            _TIME["finish"] += time.perf_counter() - _t
            _TIME["wall"] += time.perf_counter() - _t_batch
        if report and (i // batch) % report == 0:
            print(f"    {i + len(m_b)}/{len(m)} targets", flush=True)
            if _TIME is not None:
                w = _TIME["wall"] or 1.0
                acc = sum(v for k, v in _TIME.items() if k != "wall")
                print("      " + "  ".join(
                    f"{k} {v:.1f}s {v / w:.0%}" for k, v in _TIME.items()
                    if k != "wall")
                    + f"  | unaccounted {w - acc:.1f}s {(w - acc) / w:.0%}"
                    + f"  wall {w:.1f}s", flush=True)

    if n_empty:
        # Q = R = 0 for these, so they drop out of the eq. (45)-(46) sums rather
        # than poisoning them -- but a rising count means the proposal is
        # failing, not that the targets are uninformative.
        print(f"    {n_empty} targets had no draw with any weight; "
              f"they contribute nothing", flush=True)
    if n_fallback:
        # One chunk holds nearly all of this target's weight, so the delete-one
        # jackknife would divide by a vanishing 1 - a_e and these targets keep
        # the plain, biased estimator.  A large count means the per-chunk ESS is
        # so concentrated that the debiasing cannot bite -- raise `samples`, or
        # lower `chunk` so there are more independent draw sets.
        print(f"    {n_fallback} targets kept the plain Q, R "
              f"(one chunk carries the weight)", flush=True)
    # Back to float64 on the way out: the merge is float32 (see above) but
    # `ghat` sums these over ~1M galaxies, and that sum needs the headroom.
    f64 = lambda parts: np.concatenate(parts).astype(np.float64)
    if opg:
        return f64(q_out), f64(r_out), f64(opg_out)
    return f64(q_out), f64(r_out)


def ess(flow, m, draws, log_wt, batch=20000, sigma_x=None):
    """Effective number of draws behind Phat, (sum w)^2 / sum w^2,
    w_s = exp(log_wt_s) * P(draws_s|g=0).

    This is the paper's own diagnostic for the one bias its estimator carries:
    sec. 2.5 notes that dividing Q and R by a noisy Phat biases the result
    inversely with "the number of template galaxies contributing significantly
    to the P_i sums", and here that number is the ESS.  Read it as: the induced
    m is of order 1/ESS, so it wants to be well past 1e3 to be irrelevant.
    With log_wt = 0 (the pure-kernel case) this is exactly the old w_s = P(M+eps_s).
    """
    def one(m_i, draws_i, log_wt_i, sigma_x_i):
        ok = in_domain(draws_i)
        dummy = safe_point(m_i)
        cond = condition(jnp.zeros(2), sigma_x_i)
        lp = flow.log_prob(jnp.where(ok[:, None], draws_i, dummy), condition=cond)
        lw = jnp.where(ok, lp + log_wt_i, -jnp.inf)
        return jnp.exp(2 * jax.nn.logsumexp(lw) - jax.nn.logsumexp(2 * lw))

    return _over_targets(one, (m, draws, log_wt, sigma_x), batch)


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


def window_mask(m, size, flux):
    """Boolean mask of the rows of `m` (n, 5) inside the target window:
    `size[0] < Mr/Mf < size[1]` and `flux[0] < Mf < flux[1]`.

    This is the HARD version of the cut -- membership by the target's own
    (noisy) moments, exactly what a real analysis applies.  `window_prob`
    is its soft, pre-measurement counterpart: the probability a galaxy's
    NOISY moments would land here, needed by eq. (45)-(46) because the
    estimator never gets to see the non-selected galaxies' own M.
    """
    m = np.asarray(m)
    r = m[:, 1] / m[:, 0]
    return (r > size[0]) & (r < size[1]) & (m[:, 0] > flux[0]) & (m[:, 0] < flux[1])


def window_prob(m, cov, size, flux, nodes=64):
    """F(m) = Pr[m + n lands in the window], n ~ N(0, cov[:2, :2]).

    This is eq. (30)'s `INT_{M in S} dM L(M - M^G)`, with the paper's `|J(M)|`
    weight and `L(X^G)` detection factor dropped -- see `selection_terms` for
    why that is right here and how it was checked.

    Only the (Mf, Mr) 2x2 block of `cov` enters: the window is a cut on those
    two moments alone.  The size cut `size[0] < Mr'/Mf' < size[1]` is
    equivalent to the two LINEAR constraints `size[0] Mf' < Mr' < size[1] Mf'`
    when `Mf' > 0`, and conditioning on `Mf' = Mf + sf t` reduces it to a
    single Gaussian CDF difference in the residual of Mr' after regressing out
    Mf' -- a 1-D Gauss-Legendre quadrature over `t` handles the flux cut and
    the correlation together.

    That equivalence is EXACT, not approximate, whenever `flux[0] >= 0`: `t` is
    integrated only over the flux window, so `Mf' >= flux[0]` identically
    (where the clip below bites, it bites only on the side already inside the
    window).  Checked at 2e6 noise draws on real catalog rows: zero
    disagreements with the literal ratio cut.  It FAILS for an unbounded flux
    window, where a faint `Mf` can be pushed negative and the ratio flips sign
    against the linear form -- hence the guard.

    Clipped at t in [-8, 8]: `norm.pdf` is negligible past there and the
    window's flux boundary has stopped moving `F` to float32 precision, so
    the zero gradient beyond the clip is the right answer, not lost signal.
    Returns 0 (not NaN) wherever the flux window is empty at that `Mf`
    (`t1 <= t0`), and `+/-inf` window bounds work directly: `clip` absorbs an
    infinite flux bound and `norm.cdf(+/-inf)` absorbs an infinite size one.
    """
    if np.isfinite(size).any() and not flux[0] >= 0.0:
        raise ValueError(
            f"a finite Mr/Mf window ({size}) needs a flux floor >= 0, got "
            f"flux={flux}: the ratio cut is linearised as size*Mf' vs Mr', "
            f"which inverts once Mf' can go negative.  Pass --window-flux with "
            f"a non-negative lower bound.")
    if nodes == 64:
        x, w = _GL_NODES, _GL_WEIGHTS
    else:
        x, w = np.polynomial.legendre.leggauss(nodes)
    x = jnp.asarray(x)
    w = jnp.asarray(w)

    Mf = m[..., 0]
    Mr = m[..., 1]
    sf = jnp.sqrt(cov[0, 0])
    sr = jnp.sqrt(cov[1, 1])
    rho = cov[0, 1] / (sf * sr)
    sc = sr * jnp.sqrt(1.0 - rho ** 2)

    t0 = jnp.clip((flux[0] - Mf) / sf, -8.0, 8.0)
    t1 = jnp.clip((flux[1] - Mf) / sf, -8.0, 8.0)

    u = 0.5 * (t1 + t0)[..., None] + 0.5 * (t1 - t0)[..., None] * x
    Mfp = Mf[..., None] + sf * u
    mur = Mr[..., None] + rho * sr * u

    def lin(coef):
        # `coef * Mfp` is a genuine 0 * inf trap in the GRADIENT when `coef`
        # is +/-inf (unlike the flux side's `flux[k] - Mf`, this is a
        # PRODUCT, so autodiff's product rule multiplies an infinite
        # derivative by pdf(+/-inf) = 0 downstream).  An infinite size bound
        # truly does not move with Mfp, so its gradient IS zero, not
        # indeterminate; skip the arithmetic entirely and say so, the same
        # guard `mixture_draws` applies to `log(alpha)` at the 0/1 endpoints.
        if np.isneginf(coef):
            return jnp.full_like(mur, -jnp.inf)
        if np.isposinf(coef):
            return jnp.full_like(mur, jnp.inf)
        return (coef * Mfp - mur) / sc

    a, b = lin(size[0]), lin(size[1])
    integrand = jax.scipy.stats.norm.pdf(u) * (
        jax.scipy.stats.norm.cdf(b) - jax.scipy.stats.norm.cdf(a))
    F = 0.5 * (t1 - t0) * jnp.sum(w * integrand, axis=-1)
    return jnp.where(t1 <= t0, 0.0, F)


def selection_terms(draw, z, cov, size, flux, batch=16384, fd=None,
                    kind="draw"):
    """(P_s, Q_s, R_s, Q_s_err) -- eq. (40)'s selection probability and its
    first two shear derivatives at g = 0, as float64 (), (2,), (2, 2), (2,).

    `P(s|g) = E_{m ~ P(.|g)}[F(m)]` (paper eq. 40): draw `m` from the prior at
    shear `g` and ask for the probability its noisy measurement lands in the
    window.  `draw(g, z_chunk)` supplies the prior draws -- typically the
    flow's base -> data map -- and `z` a fixed base sample large enough to
    resolve `P_s` and its derivatives to the precision the ensemble sums need;
    `Q_s_err` says whether it was.

    Two factors of the paper's eq. (30)/(38)/(40) are absent, deliberately.
    `|J(M)|`, the Jacobian of the positional moments, is a function of `M`
    alone (eq. 23, and eq. 25's note that it is independent of `X`), so it
    cancels exactly out of every `Q_i`, `R_i` -- those are g-derivatives at
    FIXED `M_i`.  `L(X^G)`, the detection factor, and its `Delta^2 u` sum are
    what the centroid layer already carries (eq. 36).  What is left is: every
    stamp holds one already-detected galaxy, the flow is the density of that
    detected population, and the only selection is this window -- so `P(s|g)`
    is just "does a random detected galaxy's noisy M land in S".

    That is an assumption, and it was checked rather than asserted: pushing the
    TRUE template population (`moments.fits`, with bfd's exact `dm_dg`) through
    `window_prob` gives `P_s = 0.3033` against the noisy catalog's own measured
    selection fraction of `0.3016` -- 0.6%.  If the dropped factors mattered
    they would show up there.  Two things the same check does NOT cover: this
    is the postage-stamp branch (eq. 45-46, `N_ns` counted), not the Poisson
    sky branch (eq. 53-55, `n Omega P_s`), and it assumes one `C_M` and one
    `Sigma_X` for the whole catalog, as these catalogs have.

    By isotropy `Q_s` is exactly zero for a spin-0 window -- `P_s` can only
    depend on `|g|^2` -- so the correction is carried entirely by `R_s`, and a
    `Q_s` significantly above `Q_s_err` means something is wrong (an
    anisotropic window, or an anisotropy in the prior).  Measured both ways it
    is consistent with zero at 1 sigma, and forcing it to zero moves `m1` by
    less than 1e-5.

    One forward-over-reverse `jax.jvp` pass per shear direction, exactly the
    idiom `pqr_full`/`pqr_streamed` use, rather than `jax.hessian` -- calling
    value, grad and hessian separately would evaluate `draw` three times over.
    Chunks are accumulated as a weighted mean (weighted by chunk size, so an
    unequal last chunk is handled correctly) and promoted to float64 on the
    host before accumulating, as `_over_targets` does.
    """
    zero = jnp.zeros(2)
    cov = jnp.asarray(cov)

    def one_chunk(z_chunk):
        f = lambda g: jnp.mean(window_prob(draw(g, z_chunk), cov, size, flux))
        vg = jax.value_and_grad(f)
        (val, dq), lin = jax.linearize(vg, zero)   # see `one_b`: shared primal
        _, h0 = lin(jnp.array([1.0, 0.0]))
        _, h1 = lin(jnp.array([0.0, 1.0]))
        return val, dq, jnp.stack([h0, h1], axis=-1)

    def one_chunk_d2(z_chunk):
        """PER-TEMPLATE d2F/dg1^2, for the concentration check below."""
        e0 = jnp.array([1.0, 0.0])
        fv = lambda g: window_prob(draw(g, z_chunk), cov, size, flux)
        d1 = lambda g: jax.jvp(fv, (g,), (e0,))[1]
        return jax.jvp(d1, (zero,), (e0,))[1]

    def one_chunk_fd(z_chunk):
        """Central differences at step `fd`, with common random numbers.

        For a SIZE window the exact `d2F/dg2` has no usable mean -- `F` is a
        step in `g` of width `sigma / |dm/dg|` and along an `Mr/Mf` boundary
        `|dm/dg|` grows with flux without bound, so `d2F ~ Mf^2` while the
        boundary shell's occupancy falls only as `1/Mf`.  Measured Hill index
        0.74-0.76 (against 2.5-2.9 for a flux floor), and the autodiff estimate
        gets WORSE with more draws: -0.169 at 2^20 to -0.797 at 2^22, one draw
        carrying half of it.

        `F` itself is a probability in [0, 1], so a central difference is a
        mean of BOUNDED terms: finite variance, 1/sqrt(n), CLT applies and the
        printed standard error means something.  Measured on `(2.2, 3.6)` at
        `fd = 0.02`: -0.2204 / -0.2268 / -0.2293 over 2^18 -> 2^22 draws, flat,
        top-draw share 0.005.  The price is an O(fd^2) bias; `fd = 0.01` gives
        -0.2144 +/- 0.0136, within 1 sigma, so that bias is <= ~0.015 here.
        Concentration climbs back as the step shrinks (share 0.005 / 0.011 /
        0.051 / 0.327 at fd = 0.02 / 0.01 / 0.005 / 0.002), which is the
        divergent h -> 0 limit reappearing -- do not "improve" this by
        shrinking `fd`.

        `fd = 0.02` is also the operating point: `ghat` solves at |g| ~ 0.02,
        and eq. (45)-(46)'s quadratic is standing in for `P_s` across that
        range, not at an infinitesimal.

        Cross term by the standard 4-point stencil, so 9 evaluations of `draw`
        against the autodiff path's ~12 primal-equivalents.
        """
        h = fd
        P = lambda a, b: window_prob(
            draw(jnp.array([a, b]), z_chunk), cov, size, flux)
        p00 = P(0.0, 0.0)
        pp0, pm0, p0p, p0m = P(h, 0.0), P(-h, 0.0), P(0.0, h), P(0.0, -h)
        ppp, ppm, pmp, pmm = P(h, h), P(h, -h), P(-h, h), P(-h, -h)
        r00 = (pp0 - 2.0 * p00 + pm0) / h ** 2
        r11 = (p0p - 2.0 * p00 + p0m) / h ** 2
        r01 = (ppp - ppm - pmp + pmm) / (4.0 * h ** 2)
        q = jnp.stack([(pp0 - pm0) / (2.0 * h), (p0p - p0m) / (2.0 * h)], -1)
        r = jnp.stack([jnp.stack([r00, r01], -1),
                       jnp.stack([r01, r11], -1)], -1)
        # per-draw r00 goes back too: the concentration probe is free here,
        # where the autodiff path needs a whole extra pass for it.
        return jnp.mean(p00), jnp.mean(q, 0), jnp.mean(r, 0), r00

    chunked = eqx.filter_jit(one_chunk_fd if fd else one_chunk)
    chunked_d2 = eqx.filter_jit(one_chunk_d2)
    # The whole STENCIL has to be finite, not just g = 0.  The autodiff path
    # only ever evaluates `draw` at zero, so testing there was enough; the `fd`
    # path evaluates at the eight offset points too, and the flow's tail turns
    # some draws non-finite at g != 0 -- one such row NaNs the chunk mean and
    # with it R_s.  Requiring finiteness at every stencil point keeps the
    # sample a FIXED subset across all nine evaluations, so the differences
    # stay paired (dropping a row at +h but keeping it at 0 would bias them).
    stencil = ([zero] if not fd else
               [jnp.array([a, b]) for a, b in
                ((0.0, 0.0), (fd, 0.0), (-fd, 0.0), (0.0, fd), (0.0, -fd),
                 (fd, fd), (fd, -fd), (-fd, fd), (-fd, -fd))])
    keep_at_zero = eqx.filter_jit(
        lambda zc: jnp.all(jnp.stack([jnp.isfinite(draw(g, zc)).all(-1)
                                      for g in stencil]), axis=0))
    n = len(z)
    ps_acc, qs_acc, rs_acc, w_acc = 0.0, np.zeros(2), np.zeros((2, 2)), 0
    qs_chunks = []
    n_dropped = 0
    d2_top = (0.0, None)     # (max |d2F/dg1^2|, that DRAW's m).  "Draw", not
                             # "template": with `--window-terms flow` these are
                             # flow samples, and the warning below said
                             # "template" unconditionally for both branches.
    for i in range(0, n, batch):
        z_chunk = jnp.asarray(z[i:i + batch])
        # Drop NON-FINITE prior draws BEFORE differentiating.  A single such
        # row NaNs the whole `jnp.mean` and with it P_s, Q_s and R_s -- which
        # is how `--window-terms flow` came back NaN for every window on
        # bulgedisc_v2: its log10(Mf) tail reaches 2e8 and about 1 draw in
        # 262144 overflows float32 (gauss2 has none, which is why this went
        # unseen).  Masking in-graph with `jnp.where` fixes the VALUE but not
        # the g-Hessian, measured directly; removing the row from the sample
        # is unconditional.  The test is at g = 0 and outside the autodiff, so
        # it is a property of the sample, not a g-dependent seam -- the same
        # reason `pqr_streamed` computes `in_domain` on the raw draw once.
        #
        # FINITENESS ONLY, deliberately: `window_prob` needs Mf and Mr, not
        # chart-domain membership, and `draw` is not required to return
        # chart-representable moments (`tests/test_selection.py` hands it an
        # analytic Gaussian population, half of which is off-chart).  Adding
        # `in_domain` here silently redefined P_s and failed that test.
        ok = np.asarray(keep_at_zero(z_chunk))
        if not ok.all():
            n_dropped += int((~ok).sum())
            z_chunk = z_chunk[jnp.asarray(ok)]
        nb = z_chunk.shape[0]
        if nb == 0:
            continue
        if fd:
            val, dq, d2q, d2_raw = chunked(z_chunk)
        else:
            val, dq, d2q = chunked(z_chunk)
            d2_raw = chunked_d2(z_chunk)
        val = float(np.asarray(val, dtype=np.float64))
        dq = np.asarray(dq, dtype=np.float64)
        d2q = np.asarray(d2q, dtype=np.float64)
        ps_acc += val * nb
        qs_acc += dq * nb
        rs_acc += d2q * nb
        w_acc += nb
        qs_chunks.append(dq)
        d2 = np.asarray(d2_raw, dtype=np.float64)
        top = np.abs(d2).max()
        if top > d2_top[0]:
            j = int(np.argmax(np.abs(d2)))
            d2_top = (top, np.asarray(draw(zero, z_chunk[j:j + 1]))[0])

    if n_dropped:
        print(f"  selection_terms: dropped {n_dropped}/{n} non-finite prior "
              f"draws ({n_dropped / n:.1e})")
    ps, qs, rs = ps_acc / w_acc, qs_acc / w_acc, rs_acc / w_acc
    qs_stack = np.stack(qs_chunks)
    # Standard error of the mean from the chunk-to-chunk scatter -- the same
    # quantity `pqr_streamed`'s per-chunk merge is diagnosing, just for this
    # one number.  Single chunk: no scatter to measure, so 0 rather than NaN.
    qs_err = (qs_stack.std(axis=0, ddof=1) / np.sqrt(len(qs_stack))
             if len(qs_stack) > 1 else np.zeros(2))

    # CONCENTRATION.  `R_s` is a sample mean over the template catalog, and a
    # template sitting ON a window boundary can own the whole thing: `F` is a
    # step in `g` of width `sigma / |dm/dg|`, so a bright template, whose
    # `dm/dg` is large against a `C_M` that does not grow with flux, has
    # `d2F/dg2 ~ 1e5` where an ordinary one has ~30.  Measured on
    # `bulgedisc_deep_v2`: with a `(2.2, 3.6)` size window ONE template of
    # 131072, at `Mf = 1.5e6` and `Mr/Mf = 2.2000`, carried 103% of `R_s` and
    # by itself put the corrected `m1` at +0.99.  A flux CEILING does the same
    # (top template 130% at `Mf = 50130`).  The value is not wrong -- float64
    # finite differences converge onto it -- it is a sample mean with no
    # effective sample size, and it enters `ghat`'s denominator directly.
    # `Q_s_err` does not catch this: isotropy pins `Q_s` at zero and the damage
    # is all in `R_s`.
    share = d2_top[0] / (w_acc * abs(rs[0, 0])) if rs[0, 0] else np.inf
    if share > 0.05:
        m_top = d2_top[1]
        print(f"  WARNING: one {kind} carries {share:.0%} of R_s11 "
              f"(Mf = {m_top[0]:.4g}, Mr/Mf = {m_top[1] / m_top[0]:.4g}) -- "
              f"R_s is boundary-dominated and the correction is unreliable; "
              f"move the window off the bright {kind}s.")
    return np.float64(ps), qs, rs, qs_err


def _score_guard(chunked, m, pilot, factor, batch):
    """Threshold on |Q|, |R| for the selection draws -- `sane_targets` for the
    prior sample instead of the targets, and the same `factor`.

    WHY (2026-09-05, `[[rs-tail-is-unphysical-draws]]`).  `in_support` is a BOX
    -- Mf>0, Mr>0, and the two point-source ratio ceilings -- but the reachable
    set of a real galaxy family is a CURVED subset of it, much smaller.  The
    flow puts mass in the gap, and its extrapolated Q, R there are garbage: on
    gauss2, 91% of the draws carrying `R_s11` invert to NO galaxy (against 1.6%
    of real ones) and the flow's |R11| runs 87x the exact value where the exact
    value exists at all.  That is the whole heavy tail -- Hill 1.01, one draw
    owning 42% of `R_s11`, and getting WORSE with more draws.

    The threshold comes from a PILOT of the first `pilot` draws rather than a
    full first pass: Q, R for every draw is the expensive part and at 2^24 a
    second pass would double the run.  A median is what is being estimated, so
    a 2^16 pilot is ample, and it is computed on the UNWINDOWED |Q|, |R| -- the
    same threshold then serves every window in a `windows=` scan, which is what
    keeps that scan paired.
    """
    # In `batch`-sized pieces, NOT one vmap over the whole pilot: `one` is a
    # forward-over-reverse Hessian, so a 65536-wide call asks for 11.7 GiB and
    # the 16 GB card OOMs.  The main loop below batches for the same reason.
    aq, ar = [], []
    for i in range(0, pilot, batch):
        q, r = chunked(jnp.asarray(m[i:min(i + batch, pilot)]))
        q = np.abs(np.asarray(q, dtype=np.float64)).max(axis=-1)
        aq.append(q)
        ar.append(np.abs(np.asarray(r, dtype=np.float64))
                  .reshape(len(q), -1).max(axis=-1))
    aq, ar = np.concatenate(aq), np.concatenate(ar)
    aq, ar = aq[np.isfinite(aq)], ar[np.isfinite(ar)]
    tq, tr = factor * np.median(aq), factor * np.median(ar)
    print(f"  score guard: |Q| > {tq:.4g} or |R| > {tr:.4g} "
          f"(median {np.median(aq):.4g} / {np.median(ar):.4g} over a "
          f"{len(aq)}-draw pilot, factor {factor:g})")
    return tq, tr


def selection_terms_score(flow, m, cov, size, flux, sigma_x=None, batch=4096,
                          windows=None, guard=0.0, guard_pilot=1 << 16):
    """(P_s, Q_s, R_s, Q_s_err) by the SCORE-FUNCTION estimator -- eq. (40)'s
    selection probability and its first two shear derivatives at g = 0.

    `windows`, if given, is a list of `(size, flux)` pairs evaluated on the SAME
    draws in ONE pass, returning a list of results and ignoring `size`/`flux`.
    That is not an optimisation detail, it is what makes section 3.4's window
    scan affordable: the expensive part -- the flow's `Q`, `R` per draw -- does
    not depend on the window at all, only the 64-node `window_prob` quadrature
    does.  Thirteen windows at 2^24 draws then cost one run rather than
    thirteen.  It also makes the scan a PAIRED comparison, since every window
    sees the identical draw set and so the identical MC noise -- which is the
    whole point when the differences being read are at the 7e-04 MC floor.

    Same estimand as `selection_terms`, differentiated on the other side of the
    integral.  `F` is a cut on the MEASURED moment and shear acts on the prior,
    so `F` carries no `g` at all:

        P_s(g) = INT P(m|g) F(m) dm

    Differentiating the DENSITY rather than the SAMPLE gives, at g = 0,

        Q_s   = E[ F(m) Q(m) ]
        R_s   = E[ F(m) (R(m) + Q(m) Q(m)^T) ]        m ~ P(.|0)

    with `Q = d log P/dg` and `R = d2 log P/dg2` -- the flow's own log-density
    derivatives, exactly what `pqr` computes for a noiseless target.

    WHY, and it is not a matter of taste.  `selection_terms` is a PATHWISE
    (reparameterisation) estimator: the sample moves with `g` and `F` is
    evaluated at the moved sample, so the chain rule brings down

        d2F/dg2 = F''(m) (dm/dg)^2 + F'(m) d2m/dg2

    and `F` is a noise-smoothed step, `F' ~ 1/sigma`, `F'' ~ 1/sigma^2`.  With
    `dm/dg` proportional to flux and `sigma` flux-independent, the per-draw term
    grows as `Mf^2/sigma^2`: measured Hill tail index 0.74-0.76, i.e. NO FINITE
    MEAN, and the estimate degrades with more draws (-0.169 at 2^20 to -0.797
    at 2^22, one draw carrying half).  `--window-fd` sidesteps that by
    differencing `F` instead, which is bounded -- but it changes the estimand
    and buys an O(h^2) bias.

    Here there is no `sigma` anywhere.  `F` is a probability in [0, 1] and `Q`,
    `R` are properties of the prior alone, so the divergence is removed by
    construction rather than truncated.  `top_share` below is the same
    concentration probe `selection_terms` prints; it should be small.

    Draws `m` must come from the prior at g = 0 -- `draw(zeros(2), z)` -- and
    are held FIXED as `g` varies, which is what makes this a likelihood-ratio
    estimator rather than a second pathwise one.
    """
    zero = jnp.zeros(2)
    e0, e1 = jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
    sx = None if sigma_x is None else jnp.asarray(sigma_x)

    def one(m_i):
        f = lambda g: flow.log_prob(m_i, condition=condition(g, sx))
        vg = jax.value_and_grad(f)
        (_, q), lin = jax.linearize(vg, zero)      # see `one_b`: shared primal
        _, h0 = lin(e0)
        _, h1 = lin(e1)
        return q, jnp.stack([h0, h1], axis=-1)

    chunked = eqx.filter_jit(jax.vmap(one))
    wins = [(size, flux)] if windows is None else list(windows)
    fprobs = [eqx.filter_jit(lambda mm, s=s, f=f: window_prob(mm, cov, s, f))
              for s, f in wins]

    tq = tr = np.inf
    n_guarded = 0
    if guard > 0.0:
        tq, tr = _score_guard(chunked, m, min(guard_pilot, len(m)), guard,
                              batch)

    n = len(m)
    nw = len(wins)
    ps_acc = np.zeros(nw)
    qs_acc = np.zeros((nw, 2))
    rs_acc = np.zeros((nw, 2, 2))
    w_acc = np.zeros(nw)
    qs_chunks = [[] for _ in range(nw)]
    top = [(0.0, None)] * nw
    n_dropped = 0
    for i in range(0, n, batch):
        mm = jnp.asarray(m[i:i + batch])
        q, r = chunked(mm)
        # Drop non-finite rows outright, as `selection_terms` does: one NaN
        # takes the whole chunk mean with it, and the flow's tail does produce
        # them.  The test is at g = 0 and outside the autodiff, so the retained
        # sample is not a g-dependent seam.  `Q`, `R` are window-independent,
        # so this mask is shared -- every window keeps the SAME draws, which is
        # what makes the scan paired.
        okqr = jnp.isfinite(q).all(-1) & jnp.isfinite(r).all(-1).all(-1)
        if guard > 0.0:
            # Window-INDEPENDENT, like the finiteness mask above, so every
            # window in a scan keeps the identical draw set.
            sane = ((jnp.abs(q).max(-1) <= tq)
                    & (jnp.abs(r).reshape(len(q), -1).max(-1) <= tr))
            n_guarded += int(okqr.sum() - (okqr & sane).sum())
            okqr = okqr & sane
        for w, fprob in enumerate(fprobs):
            F = fprob(mm)
            ok = okqr & jnp.isfinite(F)
            if w == 0:
                n_dropped += int(len(mm) - ok.sum())
            Fk, qk, rk = F[ok], q[ok], r[ok]
            if not len(Fk):
                continue
            # R_s integrand: F (R + Q Q^T).  Q_s: F Q.  P_s: F.
            rr = np.asarray(
                Fk[:, None, None] * (rk + qk[:, :, None] * qk[:, None, :]),
                dtype=np.float64)
            qq = np.asarray(Fk[:, None] * qk, dtype=np.float64)
            ff = np.asarray(Fk, dtype=np.float64)
            ps_acc[w] += ff.sum(); qs_acc[w] += qq.sum(0); rs_acc[w] += rr.sum(0)
            w_acc[w] += len(ff)
            qs_chunks[w].append(qq.mean(0))
            j = int(np.argmax(np.abs(rr[:, 0, 0])))
            if abs(rr[j, 0, 0]) > top[w][0]:
                top[w] = (abs(rr[j, 0, 0]),
                          np.asarray(mm[ok][j], dtype=np.float64))

    if n_dropped:
        print(f"  selection_terms_score: dropped {n_dropped}/{n} non-finite "
              f"prior draws ({n_dropped / n:.1e})")
    if n_guarded:
        print(f"  score guard dropped {n_guarded}/{n} draws "
              f"({n_guarded / n:.2e}) -- REPORT R_s's sensitivity to `guard`, "
              f"it is a truncation and the estimand moves with it")
    out = []
    for w, (s, f) in enumerate(wins):
        ps = ps_acc[w] / w_acc[w]
        qs, rs = qs_acc[w] / w_acc[w], rs_acc[w] / w_acc[w]
        st = np.stack(qs_chunks[w])
        qs_err = (st.std(axis=0, ddof=1) / np.sqrt(len(st)) if len(st) > 1
                  else np.zeros(2))
        share = top[w][0] / (w_acc[w] * abs(rs[0, 0])) if rs[0, 0] else np.inf
        tag = "" if windows is None else f"  [size {s}, flux {f}]"
        if share > 0.05:
            mt = top[w][1]
            print(f"  WARNING: one draw carries {share:.0%} of R_s11 "
                  f"(Mf = {mt[0]:.4g}, Mr/Mf = {mt[1] / mt[0]:.4g}) -- the "
                  f"score-function estimator is NOT supposed to concentrate; "
                  f"investigate before trusting R_s.{tag}")
        else:
            print(f"  top draw carries {share:.1%} of R_s11{tag}")
        out.append((np.float64(ps), qs, rs, qs_err))
    return out if windows is not None else out[0]


def sane_targets(qr, factor=1000.0):
    """`(finite, sane)` masks over the targets of a `{label: (Q, R)}` dict.

    A single non-finite target, or one whose |Q| or |R| is many orders of
    magnitude above the rest, takes out the whole eq. (45)-(46) sum -- and both
    sums are over EVERY catalog, so a target bad in one arm is dropped from all
    of them and the +/- pairing stays aligned galaxy for galaxy.

    The magnitude cut catches flow density-curvature spikes: a handful of fully
    `in_domain` targets (median |R| ~1e2) where the flow's local Hessian blows
    up to |R| ~1e6-1e9, a normalising-flow generalisation artifact in a
    sparsely-trained pocket, not a support-boundary effect.  Measured: dropping
    the worst 5 of 200000 in one quintile took `ghat[0]` on a g = 0 null test
    from +8.7e-3 to -3.6e-4.  With `pqr_streamed`'s merge guard, non-finite
    entries should not arise; this is the net under it.

    `Q` IS GUARDED TOO, and was not until 2026-08-19.  eq. (45)'s numerator is a
    sum of `Q/P`, so a spiking Q is exactly as fatal as a spiking R -- and it
    happened: an unwindowed noisy run returned `m1 = 2.8e18` off a single target
    the |R|-only guard let through.  A healthy population has max/median ~13 for
    |Q| and ~70 for |R|, so `factor = 1000` fires on pathology and nothing else.

    The median is taken over the finite rows of ALL arms pooled, so the
    threshold is one number for the whole measurement rather than one per arm --
    an arm cannot set a looser bar for itself by being worse.
    """
    n = len(qr["plus"][0])
    finite = np.ones(n, dtype=bool)
    for q, r in qr.values():
        finite &= np.isfinite(q).all(1) & np.isfinite(r).reshape(n, -1).all(1)
    sane = finite.copy()
    for j in (0, 1):                                    # Q, then R
        v = [np.linalg.norm(np.asarray(x[j]).reshape(n, -1), axis=1)
             for x in qr.values()]
        med = np.median(np.concatenate([x[finite] for x in v]))
        for x in v:
            sane &= x < factor * med
    return finite, sane


def ghat(q, r, sel=None, ns=None, opg=None):
    """The BFD ensemble shear estimate over the selected targets.

    `ns`, if given, is the tuple `(n_ns, P_s, Q_s, R_s)` from `selection_terms`
    -- `n_ns` the count of targets that fell OUTSIDE the window -- and applies
    eq. (45)-(46)'s non-selection term, which stands in for the sum the
    excluded targets would have contributed had they been seen.

    `opg`, if given, is `pqr_streamed(..., opg=True)`'s per-target cross-fit
    `q q^T`, and switches the observed part of the metric from the Newton step
    `-SUM r` to Fisher scoring's `SUM q q^T`.  Same likelihood equation, same
    root, different metric for the step -- and it consumes only `Q`, which
    survives comparison against BFD's template sum (corr 0.896 on the failing
    bin) where `R` does not (corr 0.319, and the wrong sign).  It is also
    positive definite by construction, so the solve cannot invert.

    The SELECTION part is left alone: each template's own `dF/dg`, `d2F/dg2`
    is exact, so there is no per-template estimation error for an outer
    product to debias.  Only the `-SUM r` term is replaced.

    That does NOT make `P_s`, `Q_s`, `R_s` error-free -- they are sample MEANS
    over a finite template catalog, and their error belongs in every quoted
    bar.  `dev/size_bands.py` propagates it by block-bootstrapping the
    catalog: negligible for a flux-only window (0.00016 against a 0.00305
    galaxy bootstrap, because `R_s11` is then determined to 0.5%), but the
    DOMINANT term for a narrow size band, where `R_s11` carries 15-26% and it
    doubles the bar.

    MEASURED VERDICT (2026-08-28): a large mitigation, NOT a fix, and it
    carries a defect of its own -- do not adopt it as a default.

      * it rescues the catastrophe: `bulgedisc_deep_v2` unwindowed m1 goes
        -1.285 (Newton) -> -0.187 (cross-fit OPG), stable over S = 8192 ->
        32768;
      * on `gauss2_deep`, where Newton is right, cross-fit OPG agrees with it
        to 2.4e-4 at S = 32768 (-0.00931 against -0.00907), so the estimator
        is sound where the model is;
      * but it does NOT reach zero on `bulgedisc_deep_v2` (-0.19 plateau), and
      * `SUM q q^T` is far more outlier-dominated than `-SUM r` -- on gauss2
        the top 1000 of 19976 targets carry 38.7% of `SUM q1^2` against 17.8%
        of `SUM -R11`, and max/median is 181 against 13.  So its bias GROWS
        WITH CATALOG SIZE as more extreme `|Q|` enter (-0.0096 at n = 3000,
        -0.033 at n = 20000 on gauss2).  Suppressing that needs a tightened
        `|Q|` cut, i.e. exactly the tuned constant this route was chosen to
        avoid.

    Kept because it is opt-in, inert by default, and the only way to reproduce
    those numbers.
    """
    if sel is not None:
        q, r = q[sel], r[sel]
        opg = None if opg is None else opg[sel]
    obs = -r.sum(0) if opg is None else opg.sum(0)
    if ns is None:
        return np.linalg.solve(obs, q.sum(0))
    n_ns, ps, qs, rs = ns
    Q = q.sum(0) - n_ns * qs / (1 - ps)
    R = obs + n_ns * (np.outer(qs, qs) / (1 - ps) ** 2 + rs / (1 - ps))
    return np.linalg.solve(R, Q)


def _split_per_arm(x):
    """`x` -> (plus, minus).  A `(plus, minus)` tuple is per-arm; anything
    else (None, an ndarray, or a valid `ns` 4-tuple `(n_ns, P_s, Q_s, R_s)`)
    is shared by both arms.  Unambiguous because a genuine per-arm `ns` value
    is itself a 4-tuple, never length 2."""
    return x if isinstance(x, tuple) and len(x) == 2 else (x, x)


def bias(qp, rp, qm, rm, g=0.02, sel=None, ns=None, opg=None):
    """(m1, c1, c2) from the +g/-g pair.

    `sel` and `ns` may each be a single value used for both arms, or a
    `(plus, minus)` tuple -- the selection is on each arm's OWN observed
    moments, so the two genuinely differ.  See `_split_per_arm` for the
    disambiguation rule.  `opg` is per-arm the same way, and switches the
    metric to Fisher scoring -- see `ghat`.
    """
    sel_p, sel_m = _split_per_arm(sel)
    ns_p, ns_m = _split_per_arm(ns)
    opg_p, opg_m = _split_per_arm(opg)
    gp = ghat(qp, rp, sel_p, ns_p, opg_p)
    gm = ghat(qm, rm, sel_m, ns_m, opg_m)
    return (gp[0] - gm[0]) / (2 * g) - 1, *(0.5 * (gp + gm))


def save_pqr(path, qr, truth, obs, w=None):
    """Write per-target Q and R so two runs can be differenced later.

    The point is the PAIRING.  Two runs over the same galaxies, the same noise
    realization and the same kernel draws differ only in the flow, so their
    difference in m1 is far better determined than either run's own bootstrap
    error -- the population scatter that dominates both cancels.  Recovering
    that needs the per-target values, which are otherwise summed away.

    `truth` is the g = 0 arm's unsheared [Mf, Mr, M1, M2, Mc] (`compare` still
    reads just the flux column, kept under its old key for that).  NOT the
    clean moments on an image-noise catalog: it is a THIRD noisy realization
    -- `targets_deep_g0_200k_v2`'s Mr/Mf reaches 4.845 against the population's
    true ceiling of 3.683.  Its value is that it is independent of both arms'
    noise and of `g`, so binning on it induces no selection (`dev/edge_share.py`).  `obs` is
    `{"plus": ..., "minus": ...}`, each arm's own OBSERVED (n, 5) moments --
    written so a selection window can be re-applied offline without rerunning
    the flow, keyed `obs_plus`/`obs_minus`.

    `w` is `--prefilter-pad`'s per-target population weight, written as
    `prefilter_w`.  Only `N_ns` consumes it, but an offline scan MUST: on a
    pre-filtered file the out-of-window rows are a subsample, so counting them
    with `(~sp).sum()` under-counts `N_ns` by 64% and moves m1 by 0.015 while
    leaving the bootstrap bar unchanged -- invisible in the error bars.
    Weights per target, not one scalar count, because a scan re-solves at
    DIFFERENT windows.
    """
    truth = np.asarray(truth)
    cols = {f"{k}_{n}": v for k, (q, r) in qr.items()
            for n, v in (("q", q), ("r", r))}
    if w is not None:
        cols["prefilter_w"] = np.asarray(w)
    np.savez_compressed(path, flux=truth[:, 0], moments=truth,
                        obs_plus=np.asarray(obs["plus"]),
                        obs_minus=np.asarray(obs["minus"]), **cols)
    print(f"wrote {path}")


def compare(path_a, path_b, g=0.02, n=200, seed=0, labels=("A", "B")):
    """Paired difference in (m1, c1, c2) between two saved runs.

    Resamples galaxies ONCE per bootstrap draw and evaluates both runs on that
    same resample, so the difference keeps the pairing.  Reports each run's own
    bias with its unpaired error, then the difference with its paired one --
    which is the number that says whether the two flows actually differ.
    """
    a, b = np.load(path_a), np.load(path_b)
    if len(a["plus_q"]) != len(b["plus_q"]):
        raise SystemExit("the two runs cover different numbers of targets")
    # Drop non-finite targets from BOTH runs jointly, so the pairing -- the
    # whole point of this comparison -- stays galaxy for galaxy.
    ok = np.ones(len(a["plus_q"]), dtype=bool)
    for d in (a, b):
        for k in ("plus", "minus"):
            ok &= np.isfinite(d[f"{k}_q"]).all(1)
            ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    if not ok.all():
        print(f"  dropping {int((~ok).sum())} targets non-finite in either run")
    a = {k: v[ok] for k, v in a.items()}
    b = {k: v[ok] for k, v in b.items()}
    get = lambda d: (d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"])
    rng = np.random.default_rng(seed)
    flux = a["flux"]
    edges = np.percentile(flux, [0, 20, 40, 60, 80, 100])

    def row(name, sel):
        ba, bb = bias(*get(a), g, sel), bias(*get(b), g, sel)
        idx0 = np.flatnonzero(sel) if sel is not None else np.arange(len(flux))
        d = [np.subtract(bias(*get(b), g, i), bias(*get(a), g, i))
             for i in (rng.choice(idx0, len(idx0)) for _ in range(n))]
        sd = np.std(np.array(d), axis=0)
        print(f"  {name:>14s} {ba[0]:>+10.4f} {bb[0]:>+10.4f} "
              f"{bb[0] - ba[0]:>+10.4f} +/- {sd[0]:.4f}")
        return sd

    print(f"\n{'':>14s} {labels[0]:>10s} {labels[1]:>10s} "
          f"{'difference (paired)':>22s}")
    row("all", None)
    for i in range(5):
        row(f"Mf q{i + 1}", (flux >= edges[i]) & (flux <= edges[i + 1]))


def bootstrap(qp, rp, qm, rm, g=0.02, n=200, seed=0, sel=None, ns=None,
              weights=None):
    """Paired bootstrap over galaxies: the same resampled index into BOTH
    catalogs, so the shape noise that the +/- pairing cancels stays cancelled.

    `sel`/`ns` follow `bias`'s per-arm convention (`_split_per_arm`).  Without
    `ns` there is no N_ns to move, so the pool is exactly the selected subset,
    resampled to its own size -- today's behaviour, kept bit-for-bit.  WITH
    `ns`, N_ns is a COUNT over the resample, not a fixed number, so the index
    is drawn over ALL targets and each arm's own mask (and so its N_ns) is
    recomputed on every draw -- that is what puts the correction's own
    sampling noise into the returned error, rather than treating it as exact.
    """
    rng = np.random.default_rng(seed)
    sel_p, sel_m = _split_per_arm(sel)
    ns_p, ns_m = _split_per_arm(ns)

    if ns_p is None and ns_m is None:
        pool_sel = sel_p if sel_p is not None else sel_m
        idx0 = np.arange(len(qp)) if pool_sel is None else np.flatnonzero(pool_sel)
        out = [bias(qp[i], rp[i], qm[i], rm[i], g)
               for i in (rng.choice(idx0, len(idx0)) for _ in range(n))]
        return np.std(np.array(out), axis=0)

    w = None if weights is None else np.asarray(weights, dtype=np.float64)

    def arm_resample(sel_a, ns_a, idx):
        if sel_a is None:
            return sel_a, ns_a
        sel_i = np.asarray(sel_a)[idx]
        if ns_a is None:
            return sel_i, None
        # `N_ns` is a COUNT OF THE POPULATION, not of the integrated set.  With
        # `--prefilter-pad` the integrated set holds every target inside the
        # padded window but only a `--prefilter-sample` share of those outside
        # it, so a bare `(~sel_i).sum()` counts the padding ring and the sample
        # and misses the rest -- ~1100 against a true ~4050.  `weights` carries
        # 1 inside the pad and 1/sample outside, so the weighted sum is an
        # unbiased N_ns per replicate AND inflates the bar by the subsampling's
        # own variance, which is the honest thing to do.
        n_ns = ((~sel_i).sum() if w is None else w[idx][~sel_i].sum())
        return sel_i, (float(n_ns),) + tuple(ns_a[1:])

    n_all = len(qp)
    out = []
    for _ in range(n):
        idx = rng.choice(n_all, n_all)
        sel_pi, ns_pi = arm_resample(sel_p, ns_p, idx)
        sel_mi, ns_mi = arm_resample(sel_m, ns_m, idx)
        out.append(bias(qp[idx], rp[idx], qm[idx], rm[idx], g,
                        sel=(sel_pi, sel_mi), ns=(ns_pi, ns_mi)))
    return np.std(np.array(out), axis=0)


def _flux_sas(s):
    """Parse "mu,sig,a,b" for `--flux-sas`; see `models.bijections.sas`."""
    v = tuple(float(x) for x in s.split(","))
    if len(v) != 4:
        raise argparse.ArgumentTypeError("--flux-sas needs mu,sig,a,b")
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--pop", choices=sorted(CATALOGS), default="bulgedisc")
    p.add_argument("--centroid", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="use a flow with the centroid-marginalisation layer. "
                        "Defaults to whatever the catalog's IMGNOISE header "
                        "says; --no-centroid forces it off, which measures what "
                        "the marginalisation is worth.")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--train-data", default=None,
                   help="catalog the flow was trained on (sets the flow's "
                        "standardisation; defaults to the matching moments file)")
    p.add_argument("--g", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flux-sas", type=_flux_sas, default=None,
                   help="fitted sinh-arcsinh warp of the flux axis as \"mu,sig,a,b\"; omit for the plain log10 this chart has always used. Gaussianises log10 Mf (skew 1.78 -> 0 on bulgedisc_v2). MUST match across bulk/shear/centroid/bias or the charts disagree.")
    p.add_argument("--floor-eps", type=float, default=0.0,
                   help="defensive floor on the PRIOR density: evaluate "
                        "(1-eps)*P_flow + eps*P_broad, with P_broad a wide "
                        "Gaussian on the chart's standardised coordinates. 0 "
                        "(default) is the raw flow. This is what addresses the "
                        "measured 30.7%% of kernel draws that come back "
                        "poisoned (log p < -1e4) while sitting comfortably "
                        "INSIDE the physical support, for targets far from the "
                        "point-source boundary -- NLL training constrains "
                        "nothing in regions with no training data, and the flow "
                        "answers -1e7 there. The floor also kills those rows' "
                        "GRADIENT, which is what actually reaches Q and R. "
                        "Moves the estimand by O(eps), deliberately: scan it.")
    p.add_argument("--floor-std", type=float, default=4.0,
                   help="width of the defensive floor's Gaussian, in units of "
                        "the chart's standardised coordinates. 4 covers the "
                        "population generously. Only matters with --floor-eps.")
    p.add_argument("--support", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="zero the prior outside the ANALYTIC physical support "
                        "(models.bijections.in_support): Mf, Mr > 0, "
                        "Mr/Mf < POINT_SOURCE, Mc/Mr < POINT_SOURCE_MC. The "
                        "integration variable is a NOISELESS template moment, "
                        "so the prior is exactly zero there. Default on. "
                        "Measured to barely move the numbers (99.9%% of "
                        "out-of-support draws were already discarded as "
                        "poisoned) -- what it buys is a boundary at the "
                        "physical surface rather than at the flow's learned "
                        "cliff, which is what dP/dg's boundary term needs.")
    p.add_argument("--gauge", choices=["prior", "kernel", "auto"], default="prior",
                   help="where the shear acts inside the C_M integral. "
                        "'prior' (default, and what every run before 2026-08-31 "
                        "used) holds the draws fixed and differentiates the "
                        "flow's density, so R picks up Var[d_g log p] -- whose "
                        "Hill tail index is 1.20 on bulgedisc_v2's faint end, "
                        "i.e. INFINITE, so R there is set by whatever the "
                        "proposal truncates the tail to. 'kernel' substitutes "
                        "m = Psi_g(u) and puts g in the noise kernel: same "
                        "integral, same draws, identical at g = 0, bounded "
                        "derivative -- but it FAILS at the bright end, where "
                        "'prior' was already fine. 'auto' is the one that works "
                        "everywhere: it translates the draws by lam of the "
                        "target's own shear displacement, with lam the "
                        "minimum-variance control-variate blend of the two "
                        "endpoints, fitted per target. Measured Var[score] by "
                        "flux quintile: 2.7/4.2/3.6/1.5/0.04, against endpoints "
                        "spanning 0.1 to 7900. See log_conv_is_blend, "
                        "dev/lambda_scan.py, dev/hmc_ref.py.")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--samples", type=int, default=0,
                   help="Monte Carlo draws per target from its noise kernel. "
                        "0 leaves the targets noiseless and P a point "
                        "evaluation; >0 adds a noise realization and integrates "
                        "the prior under C_M.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="the kernel's share of the defensive mixture proposal "
                        "(mixture_draws); 1.0, the default, is pure kernel "
                        "sampling. WHETHER IT HELPS DEPENDS ON DEPTH: at "
                        "noise_sigma 1 it measured worse (median ESS 129 -> 77), "
                        "but at 2.73 it is the opposite and the tail is what "
                        "moves -- frac(ESS<10) 30%% -> 1.3%% at alpha 0.5. Note "
                        "alpha < 1 draws part of the proposal FROM THE FLOW, so "
                        "two runs with different flows lose their common random "
                        "numbers and --compare weakens.")
    p.add_argument("--chunk", type=int, default=None,
                   help="draws evaluated at once; streams the integral in "
                        "chunks of this size and keeps only per-target running "
                        "sums, so `--samples` is not limited by memory. Defaults "
                        "to min(samples, 8192).")
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="multiply C_M by this factor (in sigma, so the variance "
                        "goes as its square) -- a moment-space stand-in for a "
                        "deeper catalog. Only meaningful where the noise is "
                        "ADDED in moment space; an IMGNOISE catalog carries its "
                        "own realization and this would rescale only the kernel.")
    p.add_argument("--noise-seed", type=int, default=1,
                   help="the targets' noise realization; shared by the +g, -g "
                        "and unsheared catalogs so the pairing still cancels "
                        "shape noise")
    p.add_argument("--draw-seed", type=int, default=None,
                   help="the importance-sampling draws' seed; defaults to "
                        "--noise-seed + 1000, which is how it used to be tied. "
                        "Vary this ALONE to measure the Monte-Carlo error: "
                        "changing --noise-seed moves the noise realization too, "
                        "so a difference across it is not pure MC.")
    p.add_argument("--batch-budget", type=int, default=None,
                   help="targets x draws held on the device at once; lower it "
                        "if the Hessian runs out of memory")
    p.add_argument("--prefilter-pad", type=float, default=None,
                   help="integrate only targets inside the window widened by "
                        "this fraction (union over arms), skipping the rest. "
                        "Out-of-window targets enter eq. (45)-(46) only as a "
                        "COUNT, which is taken from the full catalog, so the "
                        "windowed answer is unchanged -- 0.2 keeps ~42%% of a "
                        "gauss2_v3 catalog and covers a +/-10%% boundary scan")
    p.add_argument("--prefilter-sample", type=float, default=0.15,
                   help="with --prefilter-pad, also integrate this fraction of "
                        "the targets outside the padded window, so the "
                        "out-of-window population stays diagnosable (that is "
                        "where gauss2_v3d's zero-weight failure lives). 0 to "
                        "cut hard")
    p.add_argument("--no-zero-arm", action="store_true",
                   help="skip the unsheared arm's integration -- a THIRD of "
                        "the target work.  It feeds only `ghat` at g = 0; m1 "
                        "and c come from the +/- pair. Costs the independent "
                        "c = 0 test (GUIDING_PRINCIPLES 4) and perturbs "
                        "`sane_targets`, whose medians pool over all arms")
    p.add_argument("--lambda-stride", type=int, default=1,
                   help="fit --gauge auto's blend lambda on every Nth draw. "
                        "Two gradient passes and a forward, over the whole "
                        "chunk, to produce one scalar per target -- ~29%% of "
                        "the run.  Every lambda is unbiased (see "
                        "`_blend_lambda`), so this trades a little variance in "
                        "Q and R for wall clock and moves no expectation")
    p.add_argument("--save-pqr", default=None,
                   help="write per-target Q and R to this .npz, for --compare")
    p.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"), default=None,
                   help="paired difference between two --save-pqr runs, and "
                        "nothing else; the only way to resolve a shift smaller "
                        "than either run's own error bar")
    p.add_argument("--n-targets", type=int, default=None,
                   help="use only the first N targets (the integration is "
                        "`samples` flow evaluations per target)")
    p.add_argument("--target-offset", type=int, default=0,
                   help="skip the first OFF targets before --n-targets. Slices "
                        "a long run into resumable pieces; the arms stay "
                        "paired because the same slice is taken from all three")
    p.add_argument("--window-size", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="target selection window in Mr/Mf (paper eq. 40/45-46 "
                        "applied to make the cut unbiased). Needs --samples > "
                        "0: at --samples 0 the observed M IS the latent, F "
                        "collapses to a hard indicator and the correction is a "
                        "different, harder calculation, out of scope here. "
                        "Unset while --window-flux is given defaults to "
                        "(-inf, inf).")
    p.add_argument("--window-flux", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="target selection window in Mf; see --window-size.")
    p.add_argument("--window-fd", type=float, default=None,
                   metavar="H",
                   help="estimate P_s's shear derivatives by central "
                        "differences at step H instead of by autodiff. "
                        "REQUIRED for any --window-size cut: the exact d2F/dg2 "
                        "has Hill tail index ~0.75 along an Mr/Mf boundary "
                        "(d2F grows as Mf^2 while the boundary shell thins "
                        "only as 1/Mf), so the autodiff estimate gets WORSE "
                        "with more draws. F is a probability, so a difference "
                        "of it is a bounded-term mean and converges. Use 0.02, "
                        "the |g| the estimator actually solves at; do NOT "
                        "shrink it, that walks back toward the divergence.")
    p.add_argument("--window-draws", type=int, default=1 << 20,
                   help="prior draws for --window-terms score. 262144 is NOT "
                        "converged: R_s11 moves -0.182 -> -0.168 from 2^18 to "
                        "2^20 and the exact R_s11 = R_s22 identity goes from "
                        "17%% violated to 2%%. Only this estimator has finite "
                        "variance, so only it rewards more draws.")
    # DEFAULT IS `score`, AND THE SELECTION CORRECTION IS FLOW-BASED ONLY.
    # `templates` lenses the training catalog by its own exact dm/dg, which is
    # what eq. (40) literally is -- but it is not a method that exists on real
    # data (there is no catalog of true dm/dg there), and its error is set by
    # the CATALOG size, which is fixed, while the flow's is set by the DRAW
    # COUNT, which is free.  At 2^23 draws the templates are the NOISIER side
    # on every windowed case (block sem 2-6x the flow's) and far more
    # concentrated (top draw 1-2% of R_s11 against the flow's 0.2%) --
    # [[rs-flow-vs-templates-is-draw-count]].  Keep `templates` reachable for
    # dev/rs_agree.py, which cross-checks the two; do not correct a bias with
    # it.
    p.add_argument("--window-guard", type=float, default=0.0,
                   help="drop selection draws whose |Q| or |R| exceeds this "
                        "factor times the pilot median -- `sane_targets` for "
                        "the prior sample. 1000 is the same factor the target "
                        "guard uses. OFF by default because it is a "
                        "TRUNCATION: R_s moves with it, so scan it and quote "
                        "the sensitivity rather than one value.")
    p.add_argument("--window-terms", choices=["templates", "flow", "score"],
                   default="score",
                   help="where eq. (40)'s P_s, Q_s, R_s come from. 'score' "
                        "is the score-function estimator E[F (R + QQ^T)], "
                        "which differentiates the DENSITY rather than the "
                        "sample and so has no 1/sigma^2 in its integrand -- "
                        "prefer it, and it needs no --window-fd. 'templates' "
                        "lenses --train-data by its own exact dm/dg, which is "
                        "what eq. (40) literally is (a sum over G) and is exact "
                        "to the catalog's sampling; 'flow' integrates the "
                        "fitted prior instead, which is what you would have to "
                        "do on real data but currently runs P_s 4%% high.")
    p.add_argument("--proposal-flow", default=None,
                   help="DANGEROUS unless it EQUALS --flow: at alpha < 1 the "
                        "weights carry P_eval/q_proposal, and wherever the "
                        "evaluated flow has mass the proposal does not cover "
                        "that ratio explodes. Measured: sharing one proposal "
                        "across two flows put 73%% of the ensemble sum|R11| on "
                        "a SINGLE target and returned m1 = -0.84 where the "
                        "self-proposed run gave +0.023, with the ESS "
                        "diagnostic and every training metric looking normal. "
                        "Use it only for the alpha=1 kernel, where the "
                        "proposal does not involve a flow at all. "
                        "Otherwise: draw the defensive mixture's flow component from THIS"
                        "flow instead of the one being evaluated, so two runs "
                        "with different --flow still share draws and --compare "
                        "keeps its pairing. Only matters when --alpha < 1.")
    a = p.parse_args()
    draw_seed = a.noise_seed + 1000 if a.draw_seed is None else a.draw_seed

    if a.compare:
        compare(*a.compare, a.g, a.boot, a.seed,
                labels=[s.split("/")[-1][:10] for s in a.compare])
        return

    if (a.window_size is not None or a.window_flux is not None) and not a.samples:
        raise SystemExit(
            "--window-size/--window-flux need noisy targets (--samples > 0): "
            "at --samples 0 the observed M IS the latent, F collapses to a "
            "hard indicator and eq. (40)/(45)-(46)'s correction becomes a "
            "different, harder calculation that is out of scope here.")

    cat = CATALOGS[a.pop]
    path = lambda v: f"{a.data_dir}/{v}.fits"
    # IMGNOISE says the noise is already in the image, put there before a real
    # recenter().  Such a catalog must not be handed to `add_noise` (which is
    # only exact at a fixed centre) and needs the centroid layer, because its
    # moments carry a centroid error the prior has to marginalise over.
    # ext=1: imsims writes its header on the table HDU, not the primary one, so
    # a bare read_header would silently report every catalog as noiseless.
    img_noise = bool(
        fitsio.read_header(path(cat["zero"]), ext=1).get("IMGNOISE", False))
    use_centroid = a.centroid if a.centroid is not None else img_noise

    train_data = a.train_data or f"{a.data_dir}/{TRAIN_DATA[a.pop]}"
    # Rebuild exactly as `shear.py train` did -- the flow's RawMomentStandardize
    # is fixed by the training split, so the same 90% has to go in here.  The
    # centroid flow was standardised on the copy catalog's GALAXIES table
    # instead, which is the full population; --train-data selects it.
    m_train_full = shear.load(train_data)[0]
    slice90 = lambda arr: arr[:int(0.9 * len(arr))]
    m_train = m_train_full if use_centroid else slice90(m_train_full)
    flow = bulk.build_flow(jr.key(a.seed), m_train, shear=True,
                           centroid=use_centroid,
                           flux_sas=a.flux_sas)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    # The physical support indicator and the defensive floor are EVALUATION-time
    # properties of the prior, not of the trained weights: the checkpoint is
    # unchanged, and --floor-eps 0 --no-support reproduces the raw flow exactly.
    if a.floor_eps > 0.0 or a.support:
        flow = bulk.SupportedFlow(flow, eps=max(a.floor_eps, 0.0),
                                  broad_std=a.floor_std, support=a.support)
    print(f"{a.flow} on the {a.pop} targets"
          + (" (image noise, recentred; centroid layer on)" if use_centroid
             else ""))
    print(f"  prior: support indicator {'on' if a.support else 'OFF'}, "
          f"defensive floor eps = {a.floor_eps:g} (broad std {a.floor_std:g})")

    proposal = None
    if a.proposal_flow:
        # Keyed on IMGNOISE, not `use_centroid`/`--no-centroid`: --no-centroid
        # selects what is being MEASURED, and must not degrade the proposal,
        # which only has to cover the posterior the CATALOG actually has.  Its
        # training slice is keyed the same way, for the same reason the eval
        # flow's is keyed on `use_centroid` above -- get this wrong and the
        # proposal's RawMomentStandardize is silently off.
        m_train_prop = m_train_full if img_noise else slice90(m_train_full)
        proposal = bulk.build_flow(jr.key(a.seed), m_train_prop, shear=True,
                                   centroid=img_noise,
                                   flux_sas=a.flux_sas)
        proposal = eqx.tree_deserialise_leaves(a.proposal_flow, proposal)
        print(f"  proposal flow: {a.proposal_flow} "
              f"(draws shared across eval flows)")

    if img_noise and a.noise_scale != 1.0:
        raise SystemExit(
            "--noise-scale on an IMGNOISE catalog would rescale the KERNEL "
            "without touching the noise already in the moments; render a "
            "deeper catalog instead (see CATALOGS['bulgedisc_deep']).")
    if img_noise and not a.samples:
        raise SystemExit(
            "these targets carry image noise, so P(M|g) is the prior CONVOLVED "
            "with C_M -- a point evaluation of the prior at a noisy M lands in "
            "its tails and returns nonsense. Pass --samples > 0.")

    off = a.target_offset
    n = slice(off, None) if a.n_targets is None else slice(off, off + a.n_targets)
    rows = {k: fitsio.read(path(v))[n] for k, v in cat.items()}
    m = {k: np.asarray(v["moments"], dtype=np.float64) for k, v in rows.items()}
    truth = m["zero"]           # unsheared -- for binning only

    # A target whose recentring did not converge is not at a detection point, so
    # it is outside the formalism entirely.  Drop it from ALL THREE catalogs so
    # the +/- pairing still lines up galaxy for galaxy.
    keep = np.ones(len(truth), dtype=bool)
    if img_noise:
        for v in rows.values():
            keep &= ~v["badcenter"]
        if not keep.all():
            print(f"  dropping {int((~keep).sum())} targets "
                  f"({(~keep).mean():.1e}) whose recentring did not converge")
        rows = {k: v[keep] for k, v in rows.items()}
        m = {k: v[keep] for k, v in m.items()}
        truth = m["zero"]

    # `n_ns`, the count of targets the window EXCLUDES, over the full catalog.
    # eq. (45)-(46) needs the count and nothing else -- `ghat` masks the
    # excluded targets' Q and R away entirely -- so it must be taken here,
    # BEFORE any pre-filter, and never from whatever subset gets integrated.
    # Getting this from the integrated subset instead would count only the
    # padding ring and silently shrink the selection correction.
    n_out_full = None
    if a.window_size is not None or a.window_flux is not None:
        _size = tuple(a.window_size) if a.window_size is not None else (-np.inf, np.inf)
        _flux = tuple(a.window_flux) if a.window_flux is not None else (-np.inf, np.inf)
        n_out_full = {k: int((~window_mask(m[k], _size, _flux)).sum())
                      for k in ("plus", "minus")}

    prefilter = None
    prefilter_w = None
    if a.prefilter_pad is not None:
        if n_out_full is None:
            raise SystemExit("--prefilter-pad needs a window to filter on; "
                             "pass --window-size and/or --window-flux.")
        if not 0.0 <= a.prefilter_sample <= 1.0:
            raise SystemExit("--prefilter-sample must be in [0, 1]")
        # Integrate the window widened by `pad`, unioned over the arms, plus a
        # random `--prefilter-sample` share of everything else.  The padding
        # keeps `dev/window_scan.py`'s boundary perturbations inside the
        # integrated set; the sample keeps the OUT-OF-WINDOW population visible,
        # which is not cosmetic -- gauss2_v3d's 22% zero-weight targets, the
        # open IS-proposal failure, are 4338/4338 out of window (0.00% in), so
        # a hard cut would hide the one diagnostic that shows it.
        pad = a.prefilter_pad
        p_size = (_size[0] * (1 - pad), _size[1] * (1 + pad))
        p_flux = (_flux[0] * (1 - pad), _flux[1] * (1 + pad))
        prefilter = np.zeros(len(truth), dtype=bool)
        for k in ("plus", "minus"):
            prefilter |= window_mask(m[k], p_size, p_flux)
        in_pad = prefilter.copy()
        n_pad = int(in_pad.sum())
        if a.prefilter_sample > 0:
            out = np.flatnonzero(~prefilter)
            take = np.random.default_rng(a.seed).choice(
                out, size=int(round(a.prefilter_sample * len(out))),
                replace=False)
            prefilter[take] = True
        print(f"  pre-filter: integrating {int(prefilter.sum())}/{len(prefilter)} "
              f"targets ({prefilter.mean():.1%}) -- {n_pad} in the window "
              f"padded by {pad:.0%}, plus {int(prefilter.sum()) - n_pad} "
              f"({a.prefilter_sample:.0%}) sampled from outside it")
        # What each integrated target stands for in the POPULATION: itself if it
        # is inside the pad (all of those are kept), or 1/sample of the
        # out-of-pad population if it was subsampled.  Only `N_ns` consumes
        # this -- the nominal window sits INSIDE the pad, so every in-window
        # target has weight 1 and the `sum q`, `sum r` the estimator actually
        # forms are untouched.
        prefilter_w = np.where(in_pad[prefilter], 1.0,
                               0.0 if a.prefilter_sample == 0 else
                               1.0 / a.prefilter_sample)
        rows = {k: v[prefilter] for k, v in rows.items()}
        m = {k: v[prefilter] for k, v in m.items()}
        truth = m["zero"]

    # Sigma_X per target: the centroid layer's condition.  Constant in these
    # sims, read per row anyway so a varying-depth catalog needs no change here.
    # Read unconditionally: the proposal flow's centroid layer is keyed on
    # IMGNOISE (see the proposal-flow build above), so it needs Sigma_X even
    # when the evaluation flow -- keyed on `use_centroid` -- does not.
    sigma_x_all = np.asarray(rows["zero"]["cov_odd"], dtype=np.float64)
    sigma_x = sigma_x_all if use_centroid else None
    proposal_sigma_x = sigma_x_all if img_noise else None
    # The three mixture_draws call sites below all draw the flow component
    # from EITHER the proposal, if one was given, or the flow itself --
    # collapse that choice once instead of three times.  `pqr_streamed` does
    # the equivalent internally, so it is passed `proposal`/`proposal_sigma_x`
    # directly rather than through these.
    draw_flow = flow if proposal is None else proposal
    draw_sigma_x = sigma_x if proposal is None else proposal_sigma_x

    draws = log_wt = None
    batch = 20000
    chunk = None
    if a.samples:
        # `--noise-scale` reaches C_M before anything else uses it, so the
        # target's own noise realization, the kernel the proposal draws from and
        # the convolution integral all move together -- which is what makes a
        # moment-space catalog stand in for a deeper render.  It is exact only
        # for the noise; a deeper IMAGE would also re-find the centroid, which
        # is why the bulgedisc deep study renders instead of scaling.
        cov = load_cov(path(cat["zero"])) * a.noise_scale ** 2
        if not img_noise:
            # The SAME noise realization for a galaxy in all three catalogs, so
            # what the +/- difference cancels stays cancelled.
            m = {k: add_noise(v, cov, a.noise_seed) for k, v in m.items()}
        # The Hessian is forward-over-reverse, so a batch costs several times
        # `batch * samples` flow activations -- keep that product bounded.  The
        # centroid layer does NOT inflate this, because it is peeled off below
        # and never enters the autodiff.
        chunk = min(a.samples, a.chunk or 8192)
        batch = max(1, (a.batch_budget or 131_072) // chunk)
        if a.samples > chunk:
            # Streamed: the draws are generated, used and discarded per chunk,
            # so nothing of size (targets x samples) is ever held.  This is the
            # only path that reaches large `samples` -- at 20000 x 32768 the draw
            # array alone would be 26 GB per catalog.
            print(f"  streaming {a.samples} draws/target in chunks of {chunk} "
                  f"({batch} targets at a time)")
        else:
            # Each catalog needs its own draws, centred on its own noisy M_i --
            # but the SAME seed for all three, so the kernel offsets and the
            # flow's defensive draws are the same common random numbers in each,
            # the same role `eps` played when it was shared directly.
            draws, log_wt = {}, {}
            for k, v in m.items():
                draws[k], log_wt[k] = mixture_draws(
                    draw_flow, v, cov, a.samples, a.alpha, draw_seed,
                    sigma_x=draw_sigma_x)

    # Peel the centroid layer off and apply it once.  Its log-det is a property
    # of the draw and of Sigma_X, never of g, so it belongs in the importance
    # weight rather than inside the g-autodiff -- see `split_centroid`.  The
    # proposal weights themselves stay as they are: they are ratios in the
    # ORIGINAL moment space, which is where the kernel lives.
    # `pqr_streamed` and the ESS diagnostic below build their own draws, so they
    # need the moments in the ORIGINAL space, before the peel rewrites `m`.
    m_raw = dict(m)
    flow_g, layer = split_centroid(flow)
    if layer is not None:
        if draws is not None:
            for k in draws:
                draws[k], ld = centroid_transform(layer, draws[k], sigma_x)
                log_wt[k] = log_wt[k] + ld
        # `m` itself only ever enters as the kernel's centre (already used, in
        # the original space) or, with no draws, as a point evaluation -- where
        # the log-det is an additive g-independent constant and so drops out of
        # both Q and R.  Either way it is discarded here rather than tracked.
        m = {k: centroid_transform(layer, v, sigma_x)[0] for k, v in m.items()}
        print("  centroid layer applied outside the g-autodiff "
              "(its log-det is g-independent, so Q and R are unchanged)")

    if a.samples:
        # ESS on one chunk's worth of draws, over a slice of targets -- a
        # diagnostic, so it does not need the full sample count.  Report it
        # scaled to the full S, since ESS grows linearly with draws.  Stratified
        # by flux quintile (not just the first n_e rows) so a starved tail
        # concentrated in one quintile doesn't average out against the rest.
        n_e = min(2000, len(m["zero"]))
        flux_e = truth[:, 0]
        edges_e = np.percentile(flux_e, [0, 20, 40, 60, 80, 100])
        per_q = max(1, n_e // 5)
        idx = np.concatenate([
            np.flatnonzero((flux_e >= edges_e[i]) & (flux_e <= edges_e[i + 1]))[:per_q]
            for i in range(5)])
        d_e, w_e = mixture_draws(draw_flow, m_raw["zero"][idx], cov, chunk,
                                 a.alpha, draw_seed, sigma_x=None
                                 if draw_sigma_x is None else draw_sigma_x[idx])
        # NOT through the peel.  `ess` tests `in_domain` and stands bad rows in
        # at `safe_point`, both of which read RAW moments; on peeled draws they
        # are nonsense and the reported median came out NaN.  The peel is exact,
        # so the full flow on raw draws is the same number, computed where the
        # domain test means something.
        e = ess(flow, m_raw["zero"][idx], d_e, w_e, 4 * batch,
                None if sigma_x is None else sigma_x[idx])
        scale = a.samples / chunk
        print(f"integrating under C_M with {a.samples} draws/target "
              f"(alpha = {a.alpha}): ESS = {np.median(e) * scale:.0f} (median), "
              f"{np.percentile(e, 5) * scale:.0f} (5th pct), "
              f"frac<10 = {float((e * scale < 10).mean()):.3f}")
        print(f"  {'quintile':>10s}{'median ESS':>12s}{'5th pct':>10s}{'frac<10':>10s}")
        lo = 0
        for i in range(5):
            n_i = min(per_q, ((flux_e >= edges_e[i]) & (flux_e <= edges_e[i + 1])).sum())
            e_i = e[lo:lo + n_i]
            lo += n_i
            if len(e_i) == 0:
                continue
            print(f"  {f'q{i + 1}':>10s}{np.median(e_i) * scale:>12.0f}"
                  f"{np.percentile(e_i, 5) * scale:>10.0f}"
                  f"{float((e_i * scale < 10).mean()):>10.3f}")

    # The unsheared arm costs a THIRD of the target integration and feeds
    # exactly one output, `ghat` at g = 0 -- `bias` and `bootstrap` read only
    # the +/- pair, and the window selection is `sel=(sp, sm)`.  Everything
    # else the zero catalog supplies (`truth` for binning, `sigma_x`, `cov`,
    # the ESS diagnostic) is READ from it, not integrated, and is unaffected.
    #
    # It is not quite free to drop: `sane_targets` pools its medians over ALL
    # arms, so without this one both the guard threshold and the dropped-target
    # set change slightly.  And `c` from the pair, `(ghat[+] + ghat[-])/2`,
    # differs from `ghat` at g = 0 by the O(g^2) part of the response -- 4e-04
    # at g = 0.02, which is not negligible against tau.  Measured on the 20k
    # matched run they agree to ~1e-04: pair c = (-2.20e-03, -7.64e-05) against
    # unsheared ghat = (-2.30e-03, -1.65e-04).
    arms = [k for k in m_raw if not (a.no_zero_arm and k == "zero")]
    if a.samples and a.samples > chunk:
        qr = {}
        for k in arms:
            v = m_raw[k]
            print(f"  {k}:", flush=True)
            qr[k] = pqr_streamed(flow, v, cov, a.samples, a.alpha,
                                 draw_seed, sigma_x,
                                 batch=batch, chunk=chunk,
                                 report=max(1, len(v) // (batch * 10)),
                                 proposal=proposal, proposal_sigma_x=proposal_sigma_x,
                                 gauge=a.gauge, lam_stride=a.lambda_stride)
    else:
        qr = {k: pqr(flow_g, m[k], None if draws is None else draws[k],
                     None if log_wt is None else log_wt[k], batch, sigma_x)
              for k in arms}
    finite, sane = sane_targets(qr)
    if not finite.all():
        print(f"  dropping {int((~finite).sum())} targets with a non-finite "
              f"Q or R ({(~finite).mean():.1e})")
    if (finite & ~sane).any():
        n = int((finite & ~sane).sum())
        print(f"  dropping {n} targets with |Q| or |R| > 1000x the population "
              f"median ({n / len(finite):.1e}); a flow density-curvature "
              f"spike, not a domain effect")
    if not finite.all() or not sane.all():
        # By flux quintile: are the dropped/outlier targets concentrated in a
        # few quintiles, or spread evenly?  Binned on the same pre-drop truth
        # flux as the main quintile table below.
        edges_diag = np.percentile(truth[:, 0], [0, 20, 40, 60, 80, 100])
        print(f"  {'quintile':>10s}{'dropped':>10s}{'outlier':>10s}{'total':>10s}")
        for i in range(5):
            qsel = (truth[:, 0] >= edges_diag[i]) & (truth[:, 0] <= edges_diag[i + 1])
            print(f"  {f'q{i + 1}':>10s}{int((qsel & ~finite).sum()):>10d}"
                  f"{int((qsel & finite & ~sane).sum()):>10d}{int(qsel.sum()):>10d}")
    if not sane.all():
        qr = {k: (q[sane], r[sane]) for k, (q, r) in qr.items()}
        truth = truth[sane]
        # The pre-filter weights are per-target and index the same rows as
        # `qp`/`sp`, so they follow the same cut.
        if prefilter_w is not None:
            prefilter_w = prefilter_w[sane]

    qp, rp = qr["plus"]
    qm, rm = qr["minus"]
    # `m_raw` is the pre-peel, raw-moment copy and is NOT itself filtered by
    # `sane` (see its own comment above) -- filter it here, wherever it is
    # used as "each arm's own observed moments".
    obs = {"plus": m_raw["plus"][sane], "minus": m_raw["minus"][sane]}

    if a.save_pqr:
        save_pqr(a.save_pqr, qr, truth, obs, w=prefilter_w)

    if a.window_size is not None or a.window_flux is not None:
        size = tuple(a.window_size) if a.window_size is not None else (-np.inf, np.inf)
        flux = tuple(a.window_flux) if a.window_flux is not None else (-np.inf, np.inf)

        sp = window_mask(obs["plus"], size, flux)
        sm = window_mask(obs["minus"], size, flux)
        print(f"\nwindow: size {size} (Mr/Mf), flux {flux} (Mf)")
        print(f"  plus:  {int(sp.sum())}/{len(sp)} targets kept")
        print(f"  minus: {int(sm.sum())}/{len(sm)} targets kept")
        print(f"  excluded (drives the correction): plus {n_out_full['plus']}, "
              f"minus {n_out_full['minus']}, of {len(truth)} in the catalog")

        # Zero-weight targets SPLIT BY WINDOW MEMBERSHIP.  The count alone is
        # what `pqr_streamed` prints and it is ambiguous: on gauss2_v3d 4338 of
        # 19945 carry no weight, which reads as a 22% catastrophe, but every
        # one of them is OUTSIDE the window (in-window rate 0.00%) at a median
        # flux of 923 against a cut at 2500.  So it does not touch the windowed
        # number -- it is the open IS-proposal failure (`bias.py`'s "a rising
        # count means the PROPOSAL is failing"), and this is where to watch it.
        for lbl, q_a, r_a, s_a in (("plus", qp, rp, sp), ("minus", qm, rm, sm)):
            deadmask = ((np.abs(q_a).sum(1) == 0)
                        & (np.abs(r_a).reshape(len(q_a), -1).sum(1) == 0))
            if deadmask.any():
                print(f"  {lbl}: zero-weight {int(deadmask.sum())} "
                      f"({deadmask.mean():.1%}) -- in-window "
                      f"{deadmask[s_a].mean():.2%}, out-of-window "
                      f"{deadmask[~s_a].mean():.2%}")

        # P_s, Q_s, R_s are a property of the MODEL (the flow's prior at
        # g = 0), not of which arm's catalog is being corrected, so one
        # `selection_terms` call serves both arms -- only N_ns (a property of
        # each arm's own data) differs between them.
        if a.window_terms == "templates":
            # eq. (40) is a sum over TEMPLATES G, not an integral over a fitted
            # prior, and the catalog carries bfd's exact dm/dg -- so take the
            # paper at its word and lens the templates directly.  That is the
            # whole justification: it is what the equation says, and it is exact
            # to the template set's own sampling rather than to a fit of it.
            # `z` is row indices here; `selection_terms` only slices it and
            # hands it to `draw`, so no special case is needed.
            #
            # The 'flow' branch is the honest alternative -- it is what you must
            # do if the flow is to REPLACE the template sum, which is this
            # project's premise -- and the gap between the two is a direct
            # measure of the flow's density error near the window edge, worth
            # watching for its own sake.
            tm, tdm, td2m = (jnp.asarray(v, jnp.float32)
                             for v in shear.load(train_data))
            draw = lambda g, idx: shear.lens(
                tm[idx], tdm[idx], td2m[idx], jnp.broadcast_to(g, (len(idx), 2)))
            z = np.arange(len(tm))
        else:
            # FLOAT64, and not optionally.  `R_s` here is a forward-over-forward
            # second derivative through the whole bijection stack -- a far
            # longer chain than the templates' `shear.lens`, which is one
            # quadratic polynomial and is fine in float32 (checked: identical
            # `R_s11` to 4 decimals both ways).  In float32 this branch loses
            # the derivative entirely to cancellation: measured on
            # `bulgedisc_v2`, flux >= 1600, `R_s11` came out -0.223 / -0.130 /
            # -2.673 at 2^18 / 2^20 / 2^22 draws with ONE draw carrying 94-103%,
            # against -0.234 / -0.233 / -0.243 in float64 -- converged, 6%
            # concentration, and agreeing with central differences of `P_s`
            # (-0.216 at h = 1e-2, -0.221 at h = 2e-3).  Every `--window-terms
            # flow` number produced before 2026-08-31 is void.
            #
            # Toggled HERE rather than at import: everything upstream (the
            # flow's own training dtype, `pqr_streamed`, the draws) is float32
            # by design and is already computed by this point; only numpy
            # bootstrapping follows.  The flow is deserialised float32, so its
            # leaves are promoted explicitly -- x64 alone would leave them
            # float32 and the chain would silently demote right back.
            jax.config.update("jax_enable_x64", True)
            flow = jax.tree_util.tree_map(
                lambda x: (x.astype(jnp.float64) if eqx.is_inexact_array(x)
                           else x), flow)
            sx1 = (None if sigma_x is None
                   else jnp.asarray(sigma_x[0], dtype=jnp.float64))
            draw = lambda g, zz: jax.vmap(
                lambda z1: flow.bijection.transform(z1, condition(g, sx1)))(zz)
            z = flow.base_dist.sample(jr.key(a.seed + 31),
                                      (262144,)).astype(jnp.float64)
        if a.window_terms == "score":
            # MORE DRAWS than the 262144 the pathwise branch uses.  Measured on
            # v10 (`dev/window_terms_compare.py`): R_s11 runs -0.213, -0.182,
            # -0.168, -0.162 at 2^16, 2^18, 2^20, 2^21, and the EXACT isotropy
            # identity R_s11 = R_s22 -- the window cuts only on spin-0
            # quantities, so P_s can depend on |g|^2 alone -- is violated by
            # 10.2%, 17.3%, 1.7%, 4.3%.  262144 is not converged and is the
            # noisiest point of the scan.  This estimator has finite variance
            # (Hill 2.9 against 1.2-1.4 for the other two), so unlike them it
            # actually pays to draw more.
            # The branch above already promoted `flow` to float64 and drew `z`
            # from its base, so the prior sample is one `draw` call away.  It
            # is taken at g = 0 and held FIXED -- the g dependence lives in the
            # density, which is the whole point (see `selection_terms_score`).
            zero2 = jnp.zeros(2)
            zs = flow.base_dist.sample(
                jr.key(a.seed + 31), (a.window_draws,)).astype(jnp.float64)
            tr = eqx.filter_jit(jax.vmap(
                lambda z1: flow.bijection.transform(z1, condition(zero2, sx1))))
            m_draw = jnp.concatenate(
                [tr(zs[i:i + 16384]) for i in range(0, len(zs), 16384)])
            ps, qs, rs, qs_err = selection_terms_score(
                flow, m_draw, cov, size, flux, sigma_x=sx1,
                guard=a.window_guard)
        else:
            ps, qs, rs, qs_err = selection_terms(
                draw, z, cov, size, flux, fd=a.window_fd,
                kind=("template" if a.window_terms == "templates" else "draw"))
        print(f"  selection terms from the {a.window_terms}"
              + (f", central differences at h = {a.window_fd}" if a.window_fd
                 else ""))
        print(f"  P_s = {ps:.4f}   Q_s = ({qs[0]:+.3e}, {qs[1]:+.3e})   "
              f"Q_s_err = ({qs_err[0]:.1e}, {qs_err[1]:.1e})")
        # The FULL R_s, not just what `ghat` consumes: `R_s11 = R_s22` and
        # `R_s12 = 0` are forced by the window being spin-0, so the off-diagonal
        # and the trace-free part are free exactness checks -- and under an
        # ELLIPTICAL PSF they are where a spin-4 leak would appear.  Printed
        # because `--save-pqr` runs before this block and cannot carry it.
        print(f"  R_s = [[{rs[0, 0]:+.5f}, {rs[0, 1]:+.5f}], "
              f"[{rs[1, 0]:+.5f}, {rs[1, 1]:+.5f}]]   "
              f"traceless {0.5 * (rs[0, 0] - rs[1, 1]):+.3e}")

        # `n_ns` comes from `n_out_full`, counted over the WHOLE catalog before
        # any pre-filter -- `(~sp).sum()` here would count only what survived
        # into the integrated set.  With no pre-filter the two agree except for
        # the handful `sane_targets` drops, which `(~sp).sum()` excludes and
        # this does not; that is a difference of a few targets in ~13000 and it
        # is the more defensible of the two, since a target dropped for a
        # spiking Q was still excluded by the window, not by the estimator.
        ns_p = (n_out_full["plus"], ps, qs, rs)
        ns_m = (n_out_full["minus"], ps, qs, rs)

        m1w, c1w, c2w = bias(qp, rp, qm, rm, a.g, sel=(sp, sm))
        dm1w, dc1w, dc2w = bootstrap(qp, rp, qm, rm, a.g, a.boot, a.seed, sel=(sp, sm))
        print(f"  windowed, uncorrected: m1 = {m1w:+.5f} +/- {dm1w:.5f}   "
              f"c1 = {c1w:+.2e} +/- {dc1w:.1e}   c2 = {c2w:+.2e} +/- {dc2w:.1e}")

        m1c, c1c, c2c = bias(qp, rp, qm, rm, a.g, sel=(sp, sm), ns=(ns_p, ns_m))
        # `weights` is what makes N_ns a POPULATION count on each replicate
        # under `--prefilter-pad`; None (no pre-filter) keeps the old path.
        dm1c, dc1c, dc2c = bootstrap(qp, rp, qm, rm, a.g, a.boot, a.seed,
                                     sel=(sp, sm), ns=(ns_p, ns_m),
                                     weights=prefilter_w)
        print(f"  windowed, corrected:   m1 = {m1c:+.5f} +/- {dm1c:.5f}   "
              f"c1 = {c1c:+.2e} +/- {dc1c:.1e}   c2 = {c2c:+.2e} +/- {dc2c:.1e}")

    m1, c1, c2 = bias(qp, rp, qm, rm, a.g)
    dm1, dc1, dc2 = bootstrap(qp, rp, qm, rm, a.g, a.boot, a.seed)
    # With a pre-filter this block is NOT the unwindowed m1: the integrated set
    # is the padded window plus a random share of the rest, so it over-weights
    # the window.  Labelled rather than silently reinterpreted -- the sampled
    # out-of-window rows are still an unbiased view of THAT population (see the
    # per-quintile table and the zero-weight split above), which is what the
    # sample is for; the mixture's mean is not a population mean.
    if prefilter is None:
        print(f"\n{len(qp)} targets, g1 = +/-{a.g}")
    else:
        print(f"\n{len(qp)} targets INTEGRATED (pre-filtered; not the "
              f"unwindowed population), g1 = +/-{a.g}")
    print(f"  m1 = {m1:+.5f} +/- {dm1:.5f}")
    print(f"  c1 = {c1:+.2e} +/- {dc1:.1e}   c2 = {c2:+.2e} +/- {dc2:.1e}")
    if "zero" in qr:
        g0 = ghat(*qr["zero"])
        print(f"  unsheared catalog: ghat = ({g0[0]:+.2e}, {g0[1]:+.2e})")
    else:
        print("  unsheared catalog: NOT INTEGRATED (--no-zero-arm)")

    # Flux quintiles: the bias is expected to vary far more across the
    # population than its mean does, so the mean alone can hide a lot.  Binned
    # on the NOISELESS unsheared flux, which is independent of both the noise
    # realization and the shear -- binning on the measured flux would be a
    # selection on a noisy quantity, and would owe eq. (40) and (46).
    flux = truth[:, 0]
    edges = np.percentile(flux, [0, 20, 40, 60, 80, 100])
    print(f"\n{'Mf quintile':>14s}{'m1':>12s}{'c1':>12s}{'c2':>12s}")
    for i in range(5):
        sel = (flux >= edges[i]) & (flux <= edges[i + 1])
        b = bias(qp, rp, qm, rm, a.g, sel)
        d = bootstrap(qp, rp, qm, rm, a.g, max(a.boot // 4, 50), a.seed, sel)
        print(f"{f'q{i + 1}':>14s}{b[0]:>+12.4f}{b[1]:>+12.2e}{b[2]:>+12.2e}"
              f"   (+/- {d[0]:.4f})")


if __name__ == "__main__":
    main()
