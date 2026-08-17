"""Is the noisy hump explained by bulgedisc's MORPHOLOGY, or only by size?

`[[broad-hump-is-real-and-bulgedisc-specific]]`: the hump is real (immovable
under 4x ESS, proposal and chart), is not domain truncation, and gauss2 does
not have it -- gauss2 reaches m1 ~ 0.001 overall.  The structural difference is
that gauss2 is CO-ELLIPTICAL with a fixed flux share, so its five even moments
determine the galaxy and Var[Q|m] = 0; bulgedisc gives the bulge its own
ellipticity and a free `bulge_frac`, so they do not, and the shear response
scatters at fixed m.  That is the term a deterministic transport drops at
second order ([[bfd-template-sum-has-no-conditional-mean]]).

The catalogs DO carry the structural parameters -- ext "POPULATION", columns
flux/sigma/e1/e2/bulge_frac/bulge_ratio/bulge_e1/bulge_e2 -- so this needs no
new simulation output.  (HANDOFF.md previously said bulgedisc did not save
them; that was wrong.)

Two analyses, because size and morphology are correlated and the marginal
profile of either is confounded by the other:

1. **m1 vs morphology AT FIXED SIZE** -- inside a narrow Mr/Mf slice, so any
   variation is morphology's and not size's.
2. **m1 vs size AT FIXED MORPHOLOGY** -- the converse: if the hump is really a
   morphology effect wearing size's clothes, it should flatten here.

    python dev/check_hump_morphology.py --pqr <run>.npz            # noisy
    python dev/check_hump_morphology.py --pqr dev/bias_flux_size_cache_mcb.npz --cache  # noiseless 1M
"""
import argparse
import sys

import fitsio
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import shear as shear_top                            # noqa: E402

G, NBOOT = 0.02, 200
HUMP_SLICE = (3.15, 3.40)      # where the noisy hump peaks


def derived(p):
    """Morphology axes, including the two gauss2 structurally cannot have."""
    bf = np.asarray(p["bulge_frac"], float)
    return {
        # Both components contribute most at bf = 0.5; this is "how much of a
        # two-component galaxy is it at all".
        "mixedness bf(1-bf)": bf * (1.0 - bf),
        "bulge_frac": bf,
        "bulge_ratio": np.asarray(p["bulge_ratio"], float),
        # THE gauss2-impossible axis: bulge and disc ellipticity disagree, so
        # the galaxy is not co-elliptical and m does not pin the morphology.
        "misalign |de|": np.hypot(np.asarray(p["bulge_e1"], float) - np.asarray(p["e1"], float),
                                  np.asarray(p["bulge_e2"], float) - np.asarray(p["e2"], float)),
        "|e| disc": np.hypot(np.asarray(p["e1"], float), np.asarray(p["e2"], float)),
        "sigma": np.asarray(p["sigma"], float),
    }


def profile(d, v, sel, nbin, rng, label):
    """m1 in `nbin` quantile bins of `v`, restricted to `sel`."""
    vv = v[sel]
    ed = np.quantile(vv, np.linspace(0, 1, nbin + 1))
    ed[0], ed[-1] = -np.inf, np.inf
    idx_all = np.flatnonzero(sel)
    out = []
    for k in range(nbin):
        s = idx_all[(vv >= ed[k]) & (vv < ed[k + 1])]
        if len(s) < 200:
            continue
        val = bias.bias(d["plus_q"][s], d["plus_r"][s],
                        d["minus_q"][s], d["minus_r"][s], G)[0]
        b = [bias.bias(d["plus_q"][i], d["plus_r"][i], d["minus_q"][i],
                       d["minus_r"][i], G)[0]
             for i in (rng.choice(s, len(s)) for _ in range(NBOOT))]
        out.append((float(np.mean(v[s])), val, float(np.std(b)), len(s)))
    if not out:
        return None
    spread = out[-1][1] - out[0][1]
    err = np.hypot(out[-1][2], out[0][2])
    print(f"  {label:22s} " + "  ".join(f"{c:6.3f}:{m:+.4f}" for c, m, e, n in out)
          + f"   | top-bottom {spread:+.4f} +/- {err:.4f} ({abs(spread)/err:4.1f}s)")
    return spread, err


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pqr", required=True)
    p.add_argument("--targets", default="../bfd_cnf_imsims/data/targets_g0_1M.fits")
    p.add_argument("--cache", action="store_true",
                   help="the npz is a plot_bias_flux_size cache (keys qp/rp/qm/rm)")
    p.add_argument("--nbin", type=int, default=5)
    a = p.parse_args()

    z = np.load(a.pqr)
    d = ({"plus_q": z["qp"], "plus_r": z["rp"],
          "minus_q": z["qm"], "minus_r": z["rm"]} if a.cache else
         {k: z[k] for k in ("plus_q", "plus_r", "minus_q", "minus_r")})
    n = len(d["plus_q"])

    m = np.asarray(shear_top.load(a.targets)[0][:n], np.float64)
    y = m[:, 1] / m[:, 0]
    pop = fitsio.read(a.targets, ext="POPULATION")[:n]
    axes = derived(pop)
    rng = np.random.default_rng(0)
    print(f"{a.pqr}: {n} targets\n")

    hump = (y >= HUMP_SLICE[0]) & (y < HUMP_SLICE[1])
    print(f"=== 1. m1 vs MORPHOLOGY at fixed size Mr/Mf in {HUMP_SLICE}, "
          f"n={int(hump.sum())} ===")
    for lab, v in axes.items():
        profile(d, v, hump, a.nbin, rng, lab)

    print(f"\n=== 2. m1 vs SIZE at fixed morphology "
          f"(within quartiles of each axis) ===")
    for lab, v in axes.items():
        q = np.quantile(v, [0.0, 0.25, 0.75, 1.0])
        for name, sel in (("low  quartile", v < q[1]), ("high quartile", v >= q[2])):
            profile(d, y, sel & (y > 2.0), a.nbin, rng, f"{lab[:14]} {name}")


if __name__ == "__main__":
    main()
