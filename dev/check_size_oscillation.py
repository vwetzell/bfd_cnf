"""Is the wiggle in m1(Mr/Mf) real, or a hexbin artifact?

`dev/plot_bias_flux_size.py` shows what looks like an oscillation in m1 along
the size axis, on top of the known edge failure near the point-source ceiling.
Two independent tests, both on the cached noiseless PQR (exact pairing, no
selection term -- see that script's docstring):

1. Fine 1D profile, marginalised over flux: if the 2D hexbin's stripes are a
   real function of Mr/Mf alone, a much finer 1D binning (which has far less
   per-bin bootstrap error, because it doesn't also split on flux) should show
   the same wiggle, not average it away.
2. Split by flux (low half vs high half) and overlay: a real size effect keeps
   the same phase in both; a flux-size INTERACTION (e.g. a population mix that
   varies with both variables) shifts the phase between the two curves.

    python dev/check_size_oscillation.py [--cache dev/bias_flux_size_cache.npz]
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402

G, NBIN, NBOOT = 0.02, 60, 200

COORDS = {
    # raw Mr/Mf, and the logit `RawMomentStandardize` actually hands the flow
    # for slot 1: u = Mr/(POINT_SOURCE*Mf), z0 = logit(u).
    "raw": ("Mr/Mf  [size]", lambda y: y),
    "logit": ("logit(Mr/(POINT_SOURCE*Mf))",
              lambda y: np.log(y / (bulk.POINT_SOURCE - y))),
}


def profile(y, qp, rp, qm, rm, nbin, rng, width=False):
    if width:
        lo, hi = np.percentile(y, [0.5, 99.5])
        edges = np.linspace(lo, hi, nbin + 1)
    else:
        edges = np.quantile(y, np.linspace(0, 1, nbin + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    out = []
    for k in range(nbin):
        s = (y >= edges[k]) & (y < edges[k + 1])
        if s.sum() < 50:
            continue
        idx0 = np.flatnonzero(s)
        v = bias.bias(qp[s], rp[s], qm[s], rm[s], G)[0]
        boot = [bias.bias(qp[i], rp[i], qm[i], rm[i], G)[0]
                for i in (rng.choice(idx0, len(idx0)) for _ in range(NBOOT))]
        out.append((float(y[s].mean()), v, float(np.std(boot)), int(s.sum())))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="dev/bias_flux_size_cache.npz")
    p.add_argument("--nbin", type=int, default=NBIN)
    p.add_argument("--coord", choices=list(COORDS), default="raw")
    p.add_argument("--width", action="store_true",
                   help="equal-WIDTH bins in the chosen coord, not equal-count")
    p.add_argument("--tag", default="", help="suffix for the output png name")
    a = p.parse_args()

    d = np.load(a.cache)
    x, y_raw, qp, rp, qm, rm = (d[k] for k in ("x", "y", "qp", "rp", "qm", "rm"))
    label, xform = COORDS[a.coord]
    y = xform(y_raw)
    rng = np.random.default_rng(0)

    binning = "width" if a.width else "quantile"
    print(f"=== 1. full-population profile, {a.nbin} {binning} bins, "
          f"coord={a.coord} ===")
    full = profile(y, qp, rp, qm, rm, a.nbin, rng, width=a.width)
    for c, v, e, n in full:
        bar = "*" * int(round(abs(v) / 0.002)) if np.isfinite(v) else ""
        print(f"  {label}={c:8.3f}  n={n:6d}  m1={v:+.4f} +/- {e:.4f}  {bar}")
    sign_flips = sum(1 for a_, b_ in zip(full, full[1:])
                     if np.sign(a_[1]) != np.sign(b_[1]) and abs(a_[1]) > a_[2]
                     and abs(b_[1]) > b_[2])
    print(f"  significant sign flips bin-to-bin: {sign_flips}/{len(full) - 1}")

    med = np.median(x)
    print(f"\n=== 2. split by flux at median(log10 Mf)={med:.3f}, "
          f"{a.nbin // 2} bins each half ===")
    halves = {}
    for lab, sel in (("low flux", x < med), ("high flux", x >= med)):
        rows = profile(y[sel], qp[sel], rp[sel], qm[sel], rm[sel],
                       a.nbin // 2, rng, width=a.width)
        halves[lab] = rows
        line = "  ".join(f"{c:.2f}:{v:+.3f}" for c, v, e, n in rows)
        print(f"  {lab:9s}: {line}")

    # Drop the handful of pathological low-size bins (near-singular R blows
    # the bootstrap sd up to O(1)); keep everything else including the edge.
    keep = lambda rows: [r for r in rows if r[2] < 0.01]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    yc, v, e, _ = zip(*keep(full))
    ax.errorbar(yc, v, yerr=e, fmt="-", color="#2a78d6", lw=1.5,
               label="full population")
    for lab, color in (("low flux", "#eb6834"), ("high flux", "#008300")):
        rows = keep(halves[lab])
        yc2, v2, e2, _ = zip(*rows)
        ax.errorbar(yc2, v2, yerr=e2, fmt="--", color=color, lw=1, label=lab)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel("m1")
    ax.set_title(f"m1 vs {a.coord} size coord ({binning} bins), "
                f"marginalised over flux")
    ax.legend()
    out = f"dev/size_oscillation_{a.coord}_{binning}{a.tag}.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
