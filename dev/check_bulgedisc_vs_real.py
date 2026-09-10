"""Corner plot: real DES/COSMOS template moments vs. the `bulgedisc` sim, in
the flow's own chart coordinates -- prior templates AND noisy targets overlaid.

Three data sets, no flow involved:

  * REAL      -- ~/Documents/BFD_cNF/summary_templates_new.fits, a COSMOS/DES
                 template catalog ([Mf, Mr, M1, M2, Mc] + a packed covariance).
                 Measured data, so these moments carry measurement noise.
  * TEMPLATES -- the v3 prior catalog, `moments_bulgedisc_v3.fits`.  NOISELESS
                 moments: the BFD prior is the population itself, and the depth
                 enters only through the stored `cov`.  This is what
                 `bulk.py`/`shear.py` train on and what the copy grid is built
                 from.
  * TARGETS   -- the v3 g = 0 arm, `targets_v3_g0_200k.fits`.  Pixel noise in
                 the stamp and a real `recenter()`, so these moments carry both
                 measurement noise and a centroid error.  This is what
                 `bias.py` integrates.

The three-way overlay is the point: TEMPLATES vs REAL asks whether the
population is right, and TARGETS vs REAL asks whether it is right once measured
at the same depth.  TEMPLATES is the only noiseless set, so it is expected to be
NARROWER than the other two -- that gap is the noise, not a mismatch.

A common S/N > 10 cut is applied to all three, since the real catalog is
detection-limited and an uncut sim would be compared against a truncated real
sample.  Companion to `bulk.py corner`, which overlays one catalog against a
trained flow's samples.

Run: `python dev/check_bulgedisc_vs_real.py`
"""

import sys

import fitsio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import corner
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from bulk import (COORD_LABELS, LABEL_FONTSIZE, POINT_SOURCE,  # noqa: E402
                  POINT_SOURCE_MC, TICK_LABELSIZE, to_coords)

REAL_PATH = "/home/vwetzell/Documents/BFD_cNF/summary_templates_new.fits"
D = "../bfd_cnf_imsims/data"
TEMPLATE_PATH = f"{D}/moments_bulgedisc_v3.fits"
TARGET_PATH = f"{D}/targets_v3_g0_200k.fits"
G2V3_TEMPLATE_PATH = f"{D}/moments_gauss2_fwd_g2v3.fits"
G2V3_TARGET_PATH = f"{D}/targets_g2v3_g0_500k.fits"
# The same population and galaxies at 2x the depth (noise_sigma 1.86, median
# flux S/N 9.6 against 19.0) -- SIG_XY comes out at exactly 2x, 222.12547 vs
# 111.06273, which is the check that the depth is the only thing that changed.
G2V3D_TARGET_PATH = f"{D}/targets_g2v3d_g0_500k.fits"
SN_MIN = 10.0
N_SAMPLE = 60000   # real catalog is 1.37M rows; subsample for plotting speed
N_GAUSS2 = 20000   # accepted galaxies; ~160s each, so smaller and cached
CACHE = "dev/gauss2_moments_cache.npz"
OUT = "plots/bulgedisc_vs_real_corner.png"

# [dataset, colour, whether to scatter individual points]
STYLES = [("real", "tab:blue", True),
          ("templates", "tab:orange", False),
          ("targets", "tab:green", False),
          ("gauss2", "tab:red", False),
          ("gauss2_real", "tab:purple", False),
          ("gauss2_fwd", "tab:brown", False),
          ("gauss2_fwd_cal", "tab:green", False),
          # The RENDERED g2v3 chain, read off disk rather than re-sampled --
          # the box parameters live in `sim.GAUSS2_FWD_*` now, so a live
          # re-sample here would drift from what was actually rendered.
          ("g2v3_templates", "tab:red", False),
          ("g2v3_targets", "tab:purple", False),
          ("g2v3d_targets", "tab:olive", False)]
DEFAULT_SETS = ["real", "templates", "gauss2", "gauss2_real"]


def flux_sn(m, cov_packed):
    """Mf / sqrt(Var Mf).  Slot 0 of a BFD packed even-moment covariance is
    C[0, 0], the same convention in the real catalog and in `imsims`."""
    return m[:, 0] / np.sqrt(np.clip(cov_packed[:, 0], 1e-30, None))


def load_real(path, n_sample, seed=0):
    n_total = fitsio.FITS(path)[1].get_info()["nrows"]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_total, size=min(n_sample * 3, n_total),
                             replace=False))
    d = fitsio.read(path, rows=idx)
    m = np.asarray(d["moments"], dtype=np.float64)
    sn = flux_sn(m, np.asarray(d["covariance"], dtype=np.float64))
    # Mr <= 0 is a measurement failure, not a chart edge.
    keep = (m[:, 0] > 0) & (m[:, 1] > 0) & np.isfinite(sn) & (sn > SN_MIN)
    return m[keep][:n_sample], keep.mean()


def load_sim(path, n_sample, drop_badcenter):
    d = fitsio.read(path)
    m = np.asarray(d["moments"], dtype=np.float64)
    sn = flux_sn(m, np.asarray(d["cov"], dtype=np.float64))
    keep = (m[:, 0] > 0) & (m[:, 1] > 0) & np.isfinite(sn) & (sn > SN_MIN)
    if drop_badcenter:
        # A target whose recentring did not converge is not at a detection
        # point, so it is outside the formalism -- `bias.py` drops it too.
        keep &= ~d["badcenter"]
    return m[keep][:n_sample], keep.mean()


def load_gauss2(which, n, seed=3):
    """Analytic moments of `n` ACCEPTED gauss2 galaxies.

    No rendering: this population is drawn in moment space, so
    `analytic.moments_batch` gives exactly what bfd would measure noiselessly,
    which is the right comparison against the noiseless `templates` set.

    The point of plotting it is that P0 is fitted to the bulgedisc box but
    ~70% of its draws have no galaxy behind them, and the reachable set is
    curved -- so what you get is P0 TRUNCATED, and only the accepted set is
    worth looking at.  Cached because each draw costs ~160s.
    """
    import os

    key = f"{which}_{n}_{seed}"
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if key in z:
            return _sn_cut(np.asarray(z[key]))
    sys.path.insert(0, "../bfd_cnf_imsims")
    from imsims import analytic, sim

    mu, cov = ((sim.GAUSS2_MU, sim.GAUSS2_COV) if which == "gauss2"
               else (sim.GAUSS2_REAL_MU, sim.GAUSS2_REAL_COV))
    pop = sim.sample_population_gauss2(n, np.random.default_rng(seed), mu, cov)
    rho = pop["bulge_ratio"]
    theta = np.stack([np.log(pop["flux"]), np.log(pop["sigma"]),
                      np.log(rho) - np.log1p(-rho), pop["e1"], pop["e2"]],
                     axis=-1)
    m = np.asarray(analytic.moments_batch(theta))
    old = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    np.savez(CACHE, **old, **{key: m})
    return _sn_cut(m)


# The two scalars that carry the forward-sampled two-Gaussian onto bulgedisc's
# moments.  They exist because the populations share a parameter BOX but not a
# light profile -- Sersic bulge + exponential disc, misaligned, against two
# co-elliptical Gaussians -- so the same (flux, sigma, rho, e) lands elsewhere.
# Fitted by bisection on the Mr/Mf and |e| medians and checked on a held-out
# seed: Mr/Mf 3.156 vs 3.149, Mc/Mr 5.962 vs 5.897, |e| 0.0911 vs 0.0909,
# log10 Mf 3.242 vs 3.248.  Mf needs no scale -- it is nearly profile-blind.
FWD_SIGMA_SCALE, FWD_E_SCALE = 1.3092, 1.0488


def load_gauss2_fwd(n, seed=0, sigma_scale=1.0, e_scale=1.0):
    """Two-Gaussian moments FORWARD-SAMPLED from the bulgedisc parameter box.

    No P0, no Newton solve, no rejection: draw the same flux/size/rho/e that
    `sim.sample_population` draws, and push them through the co-elliptical
    two-Gaussian's analytic moments.  Every draw is a galaxy by construction,
    so nothing truncates the distribution.

    This is the control that says how much of the gauss2/bulgedisc gap is the
    LIGHT PROFILE (two Gaussians against a Sersic bulge + exponential disc,
    and co-elliptical against misaligned) and how much is the rejection
    sampling.  The two populations share a parameter box, so any gap left here
    is profile, not sampling.
    """
    sys.path.insert(0, "../bfd_cnf_imsims")
    from imsims import analytic, sim

    rng = np.random.default_rng(seed)
    flux, size = sim._flux_sigma_copula(rng, n)
    rho = rng.uniform(0.2, 0.6, n)
    e1, e2 = sim._ellipticity_wide(rng, n)
    theta = np.stack([np.log(flux), np.log(size * sigma_scale),
                      np.log(rho) - np.log1p(-rho),
                      e1 * e_scale, e2 * e_scale], axis=-1)
    return _sn_cut(np.asarray(analytic.moments_batch(theta)))


def _sn_cut(m):
    """Apply the same S/N > SN_MIN cut the other sets get.

    gauss2's moments are analytic, so they carry no `cov` column -- but the
    depth is the same `noise_sigma`, and `C_M` is constant across targets
    (`check_provenance` asserts it), so the v3 prior's single covariance is the
    right one.  Without this the gauss2 curves would be the only uncut set and
    would look artificially broad at the faint end.
    """
    cov0 = np.asarray(fitsio.read(TEMPLATE_PATH, rows=[0])["cov"], float)[0, 0]
    keep = (m[:, 0] > 0) & (m[:, 1] > 0) & (m[:, 0] / np.sqrt(cov0) > SN_MIN)
    return m[keep], keep.mean()


def to_coords_clipped(m):
    """The chart, straight from `bulk.to_coords`.

    This used to apply its own logits to slots 1 and 2, and went on doing so
    after `RawMomentStandardize._forward_transform` dropped them -- so the axes
    were not the chart's and the numbers on them were not comparable to any
    catalog.  Call `bulk.to_coords` rather than re-deriving it, so the next
    chart change cannot silently strand this plot again.

    The clipping it did is gone with the logits: bare ratios have nowhere to
    blow up to, so an off-chart row just lands past the ceiling and the range
    is set by percentiles below.  The off-chart FRACTION is still reported --
    50.9% of real rows exceed the Mc/Mr ceiling, which is a real finding about
    the real catalog and not an artifact of this plot.
    """
    return to_coords(m)


def clip_fraction(m):
    """Fraction of rows sitting outside the chart on either bounded slot."""
    off = ((m[:, 1] / (POINT_SOURCE * m[:, 0]) >= 1.0)
           | (m[:, 4] / (POINT_SOURCE_MC * m[:, 1]) >= 1.0))
    return off.mean()


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sets", nargs="+", default=DEFAULT_SETS,
                   choices=[s[0] for s in STYLES],
                   help="which curves to overlay (default: the gauss2 "
                        "population comparison)")
    p.add_argument("--out", default=OUT)
    a = p.parse_args()

    loaders = {
        "real": lambda: load_real(REAL_PATH, N_SAMPLE),
        "templates": lambda: load_sim(TEMPLATE_PATH, N_SAMPLE,
                                      drop_badcenter=False),
        "targets": lambda: load_sim(TARGET_PATH, N_SAMPLE,
                                    drop_badcenter=True),
        "gauss2": lambda: load_gauss2("gauss2", N_GAUSS2),
        "gauss2_real": lambda: load_gauss2("gauss2_real", N_GAUSS2),
        "gauss2_fwd": lambda: load_gauss2_fwd(N_SAMPLE),
        "gauss2_fwd_cal": lambda: load_gauss2_fwd(
            N_SAMPLE, sigma_scale=FWD_SIGMA_SCALE, e_scale=FWD_E_SCALE),
        "g2v3_templates": lambda: load_sim(G2V3_TEMPLATE_PATH, N_SAMPLE,
                                           drop_badcenter=False),
        "g2v3_targets": lambda: load_sim(G2V3_TARGET_PATH, N_SAMPLE,
                                         drop_badcenter=True),
        "g2v3d_targets": lambda: load_sim(G2V3D_TARGET_PATH, N_SAMPLE,
                                          drop_badcenter=True),
    }
    styles = [s for s in STYLES if s[0] in a.sets]
    data, frac = {}, {}
    for name, _, _ in styles:
        data[name], frac[name] = loaders[name]()

    for k, m in data.items():
        print(f"{k:10s} n={len(m):6d}  kept {frac[k]:.1%} of rows at S/N > {SN_MIN:g}"
              f"  off-chart (clipped) {clip_fraction(m):.1%}"
              f"  Mf p50 {np.median(m[:, 0]):8.1f}"
              f"  Mr/Mf p50 {np.median(m[:, 1] / m[:, 0]):.3f}"
              f"  |e| p50 {np.median(np.hypot(m[:, 2], m[:, 3]) / m[:, 1]):.3f}")

    coords = {k: to_coords_clipped(m) for k, m in data.items()}

    # Percentiles over every plotted set, so no curve is silently cropped.
    # The old |t| < 8 guard is gone with the logits: it existed to keep the
    # clipped rows' +/-9.2 spike from setting the range, and bare ratios have
    # no such spike -- an off-chart row lands just past the ceiling.
    lo, hi = [], []
    for t in coords.values():
        p = np.percentile(t, [0.5, 99.5], axis=0)
        lo.append(p[0])
        hi.append(p[1])
    lo, hi = np.min(lo, axis=0), np.max(hi, axis=0)
    pad = 0.25 * (hi - lo)
    pad[3:] = 0.0                      # spin-2 axes are already symmetric
    plot_range = [tuple(r) for r in np.stack([lo - pad, hi + pad], axis=-1)]

    # density=True is not cosmetic: the sets have very different n (the
    # rejection-sampled ones are ~3x smaller), and on raw counts a smaller set
    # draws a shorter curve that reads as a NARROWER distribution.  Comparing
    # widths by eye is the entire purpose of this plot.
    style = dict(labels=COORD_LABELS, bins=60, range=plot_range, smooth=1.5,
                 plot_density=False, plot_contours=True, fill_contours=False,
                 label_kwargs={"fontsize": LABEL_FONTSIZE})

    # The off-chart fraction goes in the LEGEND: those rows sit past the
    # ceiling, outside the range the bulk sets, so the pile-up is invisible on
    # the axis and widening the range to show it would squash every real
    # feature.  Stating the number is the honest way to carry it.
    blurb = {
        "real": "real DES/COSMOS templates, measured",
        "templates": "bulgedisc v3 prior, NOISELESS, seed 0",
        "targets": "bulgedisc v3 targets g = 0, image noise + recentred",
        "gauss2": "gauss2 (P0 fitted to the OLD flux/size-independent box)",
        "gauss2_real": "gauss2_real (P0 fitted to the current bulgedisc box)",
        "gauss2_fwd": ("two-Gaussian FORWARD-sampled from the bulgedisc box "
                       "(no P0, no rejection)"),
        "gauss2_fwd_cal": (f"two-Gaussian FORWARD-sampled, CALIBRATED: "
                           f"sigma x{FWD_SIGMA_SCALE:.3f}, "
                           f"e x{FWD_E_SCALE:.3f}"),
        "g2v3_templates": "gauss2_fwd g2v3 prior, RENDERED, NOISELESS, seed 0",
        "g2v3_targets": ("gauss2_fwd g2v3 targets g = 0, RENDERED, "
                         "image noise + recentred (S/N p50 19.0)"),
        "g2v3d_targets": ("gauss2_fwd g2v3d targets g = 0, 2x DEEPER "
                          "(noise_sigma 1.86, S/N p50 9.6)"),
    }
    legend = {n: (f"{blurb[n]}  (n={len(data[n])}; "
                  f"{clip_fraction(data[n]):.0%} off-chart)")
              for n, _, _ in styles}

    fig = plt.figure(figsize=(16, 16))
    for name, colour, points in styles:
        corner.corner(coords[name], fig=fig, color=colour,
                      plot_datapoints=points,
                      # colour must be repeated here: corner only injects it
                      # into hist_kwargs when the caller has not supplied the
                      # dict, so passing `density` alone silently drew every
                      # 1-D panel in matplotlib's default blue.
                      hist_kwargs={"density": True, "color": colour},
                      **style)

    for ax in fig.axes:
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)

    # A figure-level legend and title: a corner grid's own axes are all either
    # occupied or hidden, so anchoring either one to a panel lands it on top of
    # the data.
    handles = [plt.Line2D([], [], color=c, lw=3, label=legend[n])
               for n, c, _ in styles]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.98), fontsize=17, frameon=True)
    fig.suptitle(
        "Population moments in the flow's chart coordinates "
        "(`bulk.to_coords`: bare ratios, NOT logits)\n"
        f"common cut $M_f/\\sqrt{{\\mathrm{{Var}}\\,M_f}}$ > {SN_MIN:g}; "
        f"noise_sigma = 0.93; point-source ceilings $r_*$ = "
        f"{POINT_SOURCE:.4f}, $r_c^*$ = {POINT_SOURCE_MC:.4f}.\n"
        "gauss2/gauss2_real are ACCEPTED galaxies (20.3% of P0 draws) -- the "
        "population you get, not P0.  gauss2_fwd has no rejection at all.\n"
        "1-D panels are DENSITY-normalised, so curves of different n are "
        "comparable.",
        fontsize=20, y=1.005, va="bottom")

    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
