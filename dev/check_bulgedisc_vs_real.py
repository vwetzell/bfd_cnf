"""Corner plot: real DES/COSMOS template moments vs. the (real-matched)
`bulgedisc` population, in the flow's own chart coordinates.

Companion to `bulk.py corner` (which overlays imsims moments against a
trained flow's samples) -- this overlays two DATA sets instead, to check
`imsims.sim`'s `SIZE_FLUX_SLOPE` coupling against the thing it was tuned to
match (see HANDOFF.md 2026-08-27 (cont.), "bulgedisc_deep q2/q3..." and the
session that added `SIZE_FLUX_SLOPE`).

Real moments come from ~/Documents/BFD_cNF/summary_templates_new.fits (COSMOS/
DES-like template catalog, [Mf, Mr, M1, M2, Mc] + a packed covariance, no
`IMGNOISE`/`cov_odd` -- real data, not another sim).  Cut to S/N > 10
(Mf / sqrt(Var(Mf)), `covariance`'s first packed element) since that is the
regime `SIZE_FLUX_SLOPE` was calibrated against.

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
                  POINT_SOURCE_MC, TICK_LABELSIZE)

REAL_PATH = "/home/vwetzell/Documents/BFD_cNF/summary_templates_new.fits"
BULGEDISC_PATH = "dev/moments_bulgedisc_realmatch.fits"
SN_MIN = 10.0
N_SAMPLE = 60000   # real catalog is 1.37M rows; subsample for plotting speed
OUT = "plots/bulgedisc_vs_real_corner.png"


def load_real(path, sn_min, n_sample, seed=0):
    n_total = fitsio.FITS(path)[1].get_info()["nrows"]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_total, size=min(n_sample * 2, n_total), replace=False))
    d = fitsio.read(path, rows=idx)
    m = np.asarray(d["moments"], dtype=np.float64)
    sn = m[:, 0] / np.sqrt(np.clip(d["covariance"][:, 0], 1e-30, None))
    m = m[(m[:, 0] > 0) & (sn > sn_min)]
    return m[:n_sample]


def load_bulgedisc(path):
    return np.asarray(fitsio.read(path)["moments"], dtype=np.float64)


def to_coords_clipped(m, eps=1e-4):
    """Like `bulk.to_coords`, but CLIPS Mr/Mf and Mc/Mr to just inside
    (0, ceiling) instead of `in_chart` dropping the whole row.

    `in_chart`'s drop was NOT harmless: 50.9% of real rows fail the Mc/Mr
    bound alone (real galaxies routinely exceed this chart's Mc/Mr ceiling --
    a real, separate finding, not chased here), and that failure correlates
    with size -- so dropping those rows pulled the SURVIVING real sample's
    Mr/Mf median from 3.50 (its true value, and what bulgedisc was actually
    calibrated against) down to 3.16, making an apples-to-oranges comparison
    look like a population mismatch on the Mr/Mf axis specifically. Clipping
    keeps every row -- a bad Mc/Mr no longer deletes an otherwise-fine row
    from the Mr/Mf or log Mf panels -- and instead piles the offending rows
    up at the chart boundary, which is the informative thing to show.
    """
    u = np.clip(m[:, 1] / (POINT_SOURCE * m[:, 0]), eps, 1 - eps)
    v = np.clip(m[:, 4] / (POINT_SOURCE_MC * m[:, 1]), eps, 1 - eps)
    return np.stack([np.log10(m[:, 0]), np.log(u) - np.log1p(-u),
                     np.log(v) - np.log1p(-v), m[:, 2] / m[:, 1], m[:, 3] / m[:, 1]],
                    axis=-1)


def main():
    real = load_real(REAL_PATH, SN_MIN, N_SAMPLE)
    real = real[real[:, 1] > 0]     # Mr <= 0 is a measurement failure, not a chart edge
    bd = load_bulgedisc(BULGEDISC_PATH)
    print(f"real: {len(real)} (S/N > {SN_MIN})")
    print(f"bulgedisc: {len(bd)}")

    t_real, t_bd = to_coords_clipped(real), to_coords_clipped(bd)

    # Real's Mc/Mr pileup at the clipped boundary (~50% of rows, see
    # `to_coords_clipped`) is genuine and worth SHOWING, but its own
    # percentile would set the axis range TO that spike and squash
    # everything else -- so the range comes from the unclipped bulk only
    # (bulgedisc has none; a loose |t| < 8 cut drops real's spike rows,
    # which sit within 1e-3 of the +/-9.2 clip bound).
    clean = (np.abs(t_real[:, 1]) < 8) & (np.abs(t_real[:, 2]) < 8)
    plot_range = [np.percentile(t_real[clean, i], [0.5, 99.5]) for i in range(5)]
    for i in (0, 1, 2):
        plot_range[i] += 0.25 * np.ptp(plot_range[i]) * np.array([-1.0, 1.0])
    plot_range = [tuple(r) for r in plot_range]
    style = dict(labels=COORD_LABELS, bins=60, range=plot_range, smooth=1.5,
                 plot_density=False, plot_contours=True, fill_contours=False,
                 label_kwargs={"fontsize": LABEL_FONTSIZE})

    fig = plt.figure(figsize=(16, 16))
    corner.corner(t_real, fig=fig, color="tab:blue", plot_datapoints=True,
                  smooth1d=1.0, hist_kwargs={"label": "real (S/N > 10)"}, **style)
    corner.corner(t_bd, fig=fig, color="tab:orange", plot_datapoints=False,
                  smooth1d=1.0, hist_kwargs={"label": "bulgedisc"}, **style)

    n_c = len(COORD_LABELS)
    axs = fig.axes
    axs[n_c + 1].legend(bbox_to_anchor=(0.0, 1.0), loc="lower left", fontsize=16)
    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
