"""Paired m1 restricted to targets that a real analysis would actually keep.

The full-catalog number is dragged down by objects no survey would use: the
faintest and the barely-resolved, where the linearisation behind both the shear
and the centroid layers has already failed.  This re-cuts the SAVED per-target
Q and R -- no re-running -- on flux S/N and on resolution.

Two notes on why this is legitimate here and not in general:

* The cut is evaluated on the g = 0 catalog, so it is the SAME set of galaxies
  in both the +g and the -g arms.  A cut on the sheared catalogs would correlate
  with the shear response and is exactly what turns P(s|g) into a term you have
  to model (eq. 45-46); a cut on the unsheared moments does not, which is what
  makes the selection terms negligible rather than merely ignored.
* The moments are still NOISY (these targets carry image noise), so a cut near
  the population's edge scatters objects across the boundary.  That is why the
  interesting rows here are "confidently" resolved, well clear of the point
  source, rather than a cut placed at the limit.

`Mr/Mf` is bounded ABOVE by `bulk.POINT_SOURCE` (= 3.976167, the PSF itself);
smaller means better resolved.
"""

import sys

import fitsio
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402

DATA = "../bfd_cnf_imsims/data"
SP = ("/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/"
      "c729e54d-31f4-414c-8c5d-6d6a496207d0/scratchpad")
G, NBOOT = 0.02, 400


def load_cuts():
    """Rebuild the per-target cut variables, aligned with the saved npz rows."""
    cat = bias.CATALOGS["bulgedisc_deep"]
    rows = {k: fitsio.read(f"{DATA}/{v}.fits")[:20000] for k, v in cat.items()}
    keep = np.ones(len(rows["zero"]), dtype=bool)
    for v in rows.values():
        keep &= ~v["badcenter"]
    z = rows["zero"][keep]
    m = np.asarray(z["moments"], dtype=np.float64)
    cov = bias.load_cov(f"{DATA}/{cat['zero']}.fits")
    # Flux S/N as bias.py's own docstring defines it (5th pct 10, median 19).
    # Both cuts are FIXED boundaries in moment space -- a flux bound Mf > F and
    # a resolution bound Mr/Mf < R -- not per-target significance cuts.  That is
    # deliberate: P(s|g) for a hard boundary in the moment space the flow already
    # models is something the eq. (45)-(46) machinery can integrate later, while
    # a threshold that moves target by target is not.
    # C_M is constant across these targets (see bias.load_cov), so a flux bound
    # IS a flux-S/N bound; the S/N is carried only to label the cut.
    return m[:, 0] / np.sqrt(cov[0, 0]), m[:, 1] / m[:, 0], np.sqrt(cov[0, 0])


def paired(a, b, sel, rng):
    """Both arms' m1 with their own bootstrap sd, and the paired difference.

    The per-arm sd is what says whether a residual bias is consistent with
    zero; the paired sd is what says whether the two arms differ.  They are
    different questions and the second is much the sharper of the two, so
    quoting only it would overstate what is known about the residual.
    """
    get = lambda d: (d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"])
    ma, mb = bias.bias(*get(a), G, sel), bias.bias(*get(b), G, sel)
    idx0 = np.flatnonzero(sel)
    res = np.array([[bias.bias(*get(a), G, i)[0], bias.bias(*get(b), G, i)[0]]
                    for i in (rng.choice(idx0, len(idx0)) for _ in range(NBOOT))])
    sd_a, sd_b = res.std(0)
    return ma[0], sd_a, mb[0], sd_b, mb[0] - ma[0], float((res[:, 1] - res[:, 0]).std())


def main():
    # Default: the baseline pair.  Pass two --save-pqr npz files to re-cut a
    # different run through exactly the same selections.
    p = sys.argv[1:3] or [f"{SP}/paired_without.npz", f"{SP}/paired_with.npz"]
    print(f"without: {p[0]}\nwith:    {p[1]}")
    a, b = np.load(p[0]), np.load(p[1])
    snr, mr_mf, sig_f = load_cuts()
    assert len(snr) == len(a["plus_q"]), (len(snr), len(a["plus_q"]))
    flux = snr * sig_f

    print(f"{len(snr)} targets,  sigma(Mf) = {sig_f:.1f}, so a flux bound "
          f"Mf > F is the S/N bound F/{sig_f:.1f}\n")
    print("  Mf            percentiles 5/25/50/75/95: "
          + " ".join(f"{v:8.0f}" for v in np.percentile(flux, [5, 25, 50, 75, 95])))
    print("  flux S/N      percentiles 5/25/50/75/95: "
          + " ".join(f"{v:8.1f}" for v in np.percentile(snr, [5, 25, 50, 75, 95])))
    print("  Mr/Mf         percentiles 5/25/50/75/95: "
          + " ".join(f"{v:8.3f}" for v in np.percentile(mr_mf, [5, 25, 50, 75, 95])))
    print(f"  point source (unresolved) at Mr/Mf = {bulk.POINT_SOURCE:.3f}; "
          f"{100 * (mr_mf > bulk.POINT_SOURCE).mean():.1f}% sit above it "
          "(noise scatter past the ceiling)")

    rng = np.random.default_rng(0)
    # Flux bounds, quoted as Mf with the equivalent S/N in the label.
    fcuts = [(round(k * sig_f, -2), k) for k in (7, 10, 15, 20, 30)]
    rows = [("no cut", np.ones(len(snr), bool))]
    rows += [(f"Mf>{f:.0f}  (S/N {k})", flux > f) for f, k in fcuts]
    rows += [(f"Mr/Mf<{c}", mr_mf < c) for c in (3.8, 3.5, 3.0, 2.5)]
    f10 = round(10 * sig_f, -2)
    f20 = round(20 * sig_f, -2)
    # The resolution threshold scanned finely at fixed S/N > 10.  The cut is on
    # the NOISY Mr/Mf, whose noise sd is 0.146 against a 0.391 intrinsic spread,
    # so a nominal 3.5 lets the >3.5 tail scatter back in; this is the scan that
    # says how far below 3.5 the threshold has to sit before that stops mattering
    # -- and how much of the answer is just the ramp being sampled differently.
    rows += [(f"Mf>{f10:.0f} & Mr/Mf<{c}", (flux > f10) & (mr_mf < c))
             for c in (3.4, 3.3, 3.2, 3.1)]
    rows += [(f"Mf>{f10:.0f} & Mr/Mf<3.5", (flux > f10) & (mr_mf < 3.5)),
             (f"Mf>{f10:.0f} & Mr/Mf<3.0", (flux > f10) & (mr_mf < 3.0)),
             (f"Mf>{f10:.0f} & Mr/Mf<2.5", (flux > f10) & (mr_mf < 2.5)),
             (f"Mf>{f20:.0f} & Mr/Mf<3.0", (flux > f20) & (mr_mf < 3.0))]

    print(f"\n{'selection':>24s} {'N':>6s} {'without':>18s} {'with':>18s} "
          f"{'difference (paired)':>21s}")
    for name, sel in rows:
        if sel.sum() < 50:
            print(f"{name:>24s} {sel.sum():>6d}   too few")
            continue
        w, sw, c, sc, d, sd = paired(a, b, sel, rng)
        print(f"{name:>24s} {sel.sum():>6d} {w:>+9.4f} +/- {sw:.4f} "
              f"{c:>+9.4f} +/- {sc:.4f} {d:>+9.4f} +/- {sd:.4f}")


if __name__ == "__main__":
    main()
