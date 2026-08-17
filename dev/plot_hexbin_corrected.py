"""Noisy m1 in the (log Mf, Mr/Mf) plane: real, control, and real MINUS control.

The NOISELESS hexbin (`dev/plot_bias_flux_size.py`) needs no control -- binning
on Mr/Mf there is a function of `m`, which is exactly what the estimator sees.
This one does: once the target is noisy the estimator sees only `M`, so sorting
targets by their TRUE moments is a hidden-variable selection and biases per-cell
m1 even for a perfect density.  See NOISY_BINNING_BIAS.md.

The control (`dev/check_selfconsistency_noisy.py --save-pqr`) draws its targets
FROM THE FLOW, so its density is exact by construction and whatever it shows is
the method's own artefact.

Cell matching: the control's targets are flow draws, not catalog rows, so the
two runs share the hexagonal LATTICE (fixed `gridsize` + `extent`) but not cell
membership.  Cells are therefore matched by their centre coordinates, and a cell
is only shown where BOTH runs have at least `--mincnt` targets.

    python dev/plot_hexbin_corrected.py --real hex_real.npz --control hex_control.npz
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors                   # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402

sys.path.insert(0, ".")

import bias                                           # noqa: E402
import shear as shear_top                             # noqa: E402

G, NBOOT = 0.02, 60
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]
DATA = "../bfd_cnf_imsims/data"


def cells(ax, x, y, d, gridsize, extent, rng, nboot):
    """(centres, m1, sigma, counts) on the hexagonal lattice `gridsize`+`extent`.

    Two passes over the same lattice: one with no `C` for the counts, one with
    `C` and a reducer for m1.  mincnt=1 in both so the two agree cell for cell.
    """
    m1 = lambda i: bias.bias(d["plus_q"][i], d["plus_r"][i],
                             d["minus_q"][i], d["minus_r"][i], G)[0]
    # All three passes go through the C + reduce_C_function path.  hexbin
    # returns a DIFFERENT set of cells when C is None, so mixing the two forms
    # gives arrays that cannot be indexed against each other.
    hx = lambda fn: ax.hexbin(x, y, C=np.arange(len(x)), gridsize=gridsize,
                              extent=extent, mincnt=1, reduce_C_function=fn)
    cnt = hx(len)
    val = hx(lambda i: m1(np.asarray(i, int)))
    # 0.0, not NaN, for the sparse cells: hexbin DROPS a cell whose reducer
    # returns a non-finite value, which would desynchronise this pass from the
    # others.  They are masked by `--mincnt` downstream anyway.
    sig = hx(lambda i: np.std([m1(rng.choice(np.asarray(i, int), len(i)))
                               for _ in range(nboot)])
             if len(i) >= 20 else 0.0)
    out = (val.get_offsets(), val.get_array(), sig.get_array(), cnt.get_array())
    assert len({len(o) for o in out}) == 1, [len(o) for o in out]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--targets", default=f"{DATA}/targets_g0_1M.fits")
    p.add_argument("--gridsize", type=int, default=9)
    p.add_argument("--mincnt", type=int, default=800)
    p.add_argument("--halfrange", type=float, default=0.04)
    p.add_argument("--out", default="dev/hexbin_corrected.png")
    a = p.parse_args()

    zr = np.load(a.real)
    dr = {k: zr[k] for k in ("plus_q", "plus_r", "minus_q", "minus_r")}
    n = len(dr["plus_q"])
    m = np.asarray(shear_top.load(a.targets)[0][:n], np.float64)
    xr, yr = np.log10(m[:, 0]), m[:, 1] / m[:, 0]

    zc = np.load(a.control)
    keep = zc["keep"]
    dc = {k: zc[k][keep] for k in ("plus_q", "plus_r", "minus_q", "minus_r")}
    xc, yc = zc["x"][keep], zc["y"][keep]
    print(f"real {n} targets, control {int(keep.sum())}")

    # One lattice for both, from the REAL run's spread (the control's flow draws
    # can stray slightly further, and a shared extent is what makes the cells
    # comparable at all).
    extent = (*np.percentile(xr, [0.5, 99.5]), *np.percentile(yr, [0.5, 99.5]))
    rng = np.random.default_rng(0)

    scratch = plt.figure()
    ax0 = scratch.add_subplot(111)
    off_r, v_r, s_r, c_r = cells(ax0, xr, yr, dr, a.gridsize, extent, rng, NBOOT)
    off_c, v_c, s_c, c_c = cells(ax0, xc, yc, dc, a.gridsize, extent, rng, NBOOT)
    plt.close(scratch)

    key = lambda o: (round(float(o[0]), 6), round(float(o[1]), 6))
    look = {key(o): i for i, o in enumerate(off_c)}
    diff = np.full(len(off_r), np.nan)
    dsig = np.full(len(off_r), np.nan)
    ctrl = np.full(len(off_r), np.nan)
    for i, o in enumerate(off_r):
        j = look.get(key(o))
        if j is None or c_r[i] < a.mincnt or c_c[j] < a.mincnt:
            continue
        ctrl[i] = v_c[j]
        diff[i] = v_r[i] - v_c[j]
        dsig[i] = np.hypot(s_r[i], s_c[j])
    real = np.where(c_r >= a.mincnt, v_r, np.nan)
    real = np.where(np.isfinite(ctrl), real, np.nan)   # show the same cells
    print(f"  {int(np.isfinite(diff).sum())} cells with >= {a.mincnt} in both")

    cmap = mcolors.LinearSegmentedColormap.from_list("d", DIVERGING)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panels = [
        ("real: m1 (binned on TRUE Mr/Mf)", real, cmap,
         mcolors.CenteredNorm(0, a.halfrange)),
        ("control: flow is its own prior\n(this is pure method)", ctrl, cmap,
         mcolors.CenteredNorm(0, a.halfrange)),
        # Half the scale of the two panels above: the residual is small, and on
        # a +/-0.04 scale it reads as uniformly blank.  The two Mr/Mf ~ 3.65
        # cells saturate -- that is the edge, where the control overshoots and
        # the subtraction is between two large numbers (see the printed table).
        (f"CORRECTED = real - control\n(the flow's own contribution, "
         f"scale +/-{a.halfrange/2:.2f})", diff, cmap,
         mcolors.CenteredNorm(0, a.halfrange / 2)),
        ("corrected / sigma", np.where(dsig > 0, diff / dsig, np.nan), cmap,
         mcolors.CenteredNorm(0, 3.0)),
    ]
    for ax, (title, arr, cm, norm) in zip(axes.ravel(), panels):
        hx = ax.hexbin(xr, yr, C=np.arange(len(xr)), gridsize=a.gridsize,
                       extent=extent, mincnt=1,
                       reduce_C_function=lambda i: 0.0, cmap=cm, norm=norm)
        hx.set_array(arr)
        fig.colorbar(hx, ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\log_{10} M_f$")
        ax.set_ylabel(r"$M_r / M_f$")
    fig.suptitle("Noisy m1 in the flux-size plane, with the binning artefact "
                 "subtracted", fontsize=12)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")

    fin = np.isfinite(diff)
    order = np.argsort(-np.abs(np.where(fin & (dsig > 0), diff / dsig, 0)))
    print(f"\n  cells by |corrected|/sigma:")
    print(f"  {'log10 Mf':>9s} {'Mr/Mf':>7s} {'n_real':>7s} {'n_ctrl':>7s} "
          f"{'real':>9s} {'control':>9s} {'corrected':>19s}")
    for i in order[:int(fin.sum())]:
        if not fin[i]:
            continue
        j = look[key(off_r[i])]
        print(f"  {off_r[i][0]:9.3f} {off_r[i][1]:7.3f} {int(c_r[i]):7d} "
              f"{int(c_c[j]):7d} {v_r[i]:+9.4f} {ctrl[i]:+9.4f} "
              f"{diff[i]:+8.4f}+/-{dsig[i]:.4f} "
              f"({abs(diff[i])/dsig[i]:4.1f}s)")
    print(f"\n  real      : min {np.nanmin(real):+.4f}  max {np.nanmax(real):+.4f}")
    print(f"  control   : min {np.nanmin(ctrl):+.4f}  max {np.nanmax(ctrl):+.4f}")
    print(f"  corrected : min {np.nanmin(diff):+.4f}  max {np.nanmax(diff):+.4f}"
          f"   |>2 sigma| in {int((np.abs(diff[fin]) > 2*dsig[fin]).sum())}"
          f"/{int(fin.sum())} cells")


if __name__ == "__main__":
    main()
