"""Does the measured bias ROTATE WITH THE PSF?

Phase 0's whole point.  A single config's `c2` cannot be told from the scatter
already in the baseline (`c1 = -4.65e-03`, `c2 = -3.89e-03` unwindowed, and an
identical rebuild moves `m1` by sd 0.0008).  What no other term in the system
can imitate is a shift that turns with the PSF:

    e_psf = (+e, 0)   ->   dc lands in c1
    e_psf = ( 0, +e)  ->   the SAME magnitude rotates into c2
    e_psf = (-e, 0)   ->   c1 flips sign

Each config's targets are the same galaxies under the same noise field as the
baseline's (one seed, `dev/render_psfe.sh`), so the difference is PAIRED galaxy
for galaxy and the population scatter that dominates either run's own bar
cancels -- which is the only way to resolve a shift this small.

Reported here is the WINDOWED, UNCORRECTED `(m1, c1, c2)`: the window is
re-applied offline from each run's stored `obs_plus`/`obs_minus`, and the
selection terms are deliberately left out because `--save-pqr` writes before
they are computed, so the baseline npz does not carry them.  That costs
nothing for this question -- the correction is affine and nearly common across
configs -- and the PRIOR-side half of the leak is read straight off the logs'
`Q_s` and `R_s` lines instead, which is where PSF anisotropy shows up first.

    python -u dev/psfe_compare.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import bias as B


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def finite(d):
    ok = np.ones(len(d["plus_q"]), dtype=bool)
    for k in ("plus", "minus"):
        ok &= np.isfinite(d[f"{k}_q"]).all(1)
        ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    return ok


def amplitude(cfg):
    """`e_psf` magnitude from a config name: `psfe1p` -> 0.2, `psfe2p05` ->
    0.05.  The bare names predate the amplitude scan and are all at 0.2."""
    tag = cfg[len("psfe1p"):]
    return 0.2 if not tag else float(f"0.{tag}")


def catalog_rows(d, cfg, n, data_dir="../bfd_cnf_imsims/data"):
    """Which of the first `n` catalog rows each npz row came from.

    The configs do NOT have equal row counts: `bias.py` drops any target whose
    recentring failed in ANY arm, plus whatever the `sane` |Q|/|R| guard takes,
    and an elliptical PSF changes how the image noise maps into the moments, so
    each config loses a different handful (1 to 4 of 20000 here).  Pairing them
    needs galaxy IDENTITY -- comparing by row position would silently offset
    every row after the first drop.

    Recovered by matching `obs_plus`, which `save_pqr` stores verbatim, against
    the plus arm's own `moments` column.  Reconstructing the mask from
    `badcenter` alone is NOT enough and was tried: it over-counts by one on
    `psfe1m`, because `sane` dropped a row on top of the recentring failures.
    The full 5-vector is needed as the key -- Mf alone collides.

    `cfg` is the config name (`psfe1p`, `psfe2p05`, ...) and the catalog path
    is taken from `bias.CATALOGS`, not rebuilt by string surgery: the old
    version hard-coded the `20` amplitude tag, which silently pointed every
    amplitude-scan config at the 0.2 catalog and then failed the uniqueness
    assert below.  One source of truth for which file a config means.
    """
    import fitsio
    cat = np.asarray(fitsio.read(
        f"{data_dir}/{B.CATALOGS['bulgedisc_v3_' + cfg]['plus']}.fits")
        ["moments"], dtype=np.float64)[:n]
    lut = {}
    for i, k in enumerate(map(tuple, cat)):
        lut.setdefault(k, i)
    idx = np.array([lut.get(k, -1) for k in map(tuple, d["obs_plus"])])
    assert (idx >= 0).all() and len(set(idx.tolist())) == len(idx), cfg
    return idx


def triple(d, sel, size, flux):
    sp = B.window_mask(d["obs_plus"][sel], size, flux)
    sm = B.window_mask(d["obs_minus"][sel], size, flux)
    return np.array(B.bias(d["plus_q"][sel], d["plus_r"][sel],
                           d["minus_q"][sel], d["minus_r"][sel],
                           sel=(sp, sm)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="pqr/v11_psfe00.npz")
    p.add_argument("--configs", nargs="*",
                   default=["psfe1p", "psfe2p", "psfe1m"])
    p.add_argument("--suffix", default="", help='"_nocent" for the bisect')
    p.add_argument("--window-size", type=float, nargs=2, default=[2.2, 3.2])
    p.add_argument("--window-flux", type=float, nargs=2, default=[2500.0, 50000.0])
    p.add_argument("--n-catalog", type=int, default=20000,
                   help="rows bias.py was pointed at (--n-targets)")
    p.add_argument("--boot", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drop-top", type=int, default=0,
                   help="drop the k paired galaxies with the largest |R11| in "
                        "ANY run -- the concentration check")
    p.add_argument("--drop-by", default="R11", choices=("R11", "Q2"))
    a = p.parse_args()
    size, flux = tuple(a.window_size), tuple(a.window_flux)

    names = ["psfe00"] + list(a.configs)
    npz = {c: load(f"pqr/v11_{c}{a.suffix}.npz") for c in names}

    # Align on GALAXY, not row: each config lost a different few targets.
    n = a.n_catalog
    cat_of = {c: catalog_rows(npz[c], c, n) for c in names}
    # Galaxies present, and finite, in EVERY run.
    common = np.ones(n, dtype=bool)
    for c in names:
        ok = np.zeros(n, dtype=bool)
        ok[cat_of[c][finite(npz[c])]] = True
        common &= ok
    want = np.flatnonzero(common)
    row = {}
    for c in names:
        pos = np.full(n, -1)
        pos[cat_of[c]] = np.arange(len(cat_of[c]))
        row[c] = pos[want]
        assert (row[c] >= 0).all()

    # Concentration check, on the PAIRED set so the comparison stays paired.
    # `bias.py`'s |Q|/|R| guard fires at 1000x the median and `psfe1m` carries
    # a target at 572x that it therefore keeps -- one galaxy holding 7.9% of
    # SUM|R11| inside the window, against 0.13% for every other config.  A
    # difference that moves when that galaxy leaves is not a measurement.
    if a.drop_top:
        key = np.zeros(len(want))
        for c in names:
            v = (np.abs(npz[c]["plus_r"][row[c], 0, 0])
                 if a.drop_by == "R11" else np.abs(npz[c]["plus_q"][row[c], 1]))
            key = np.maximum(key, np.nan_to_num(v, nan=np.inf))
        keep = np.argsort(key)[:-a.drop_top]
        keep.sort()
        print(f"dropping the {a.drop_top} paired galaxies with the largest "
              f"|{a.drop_by}| in any run\n  (largest dropped "
              f"{key[np.argsort(key)[-a.drop_top:]].max():.3g}, largest kept "
              f"{key[keep].max():.3g}, median {np.median(key):.3g})")
        want = want[keep]
        for c in names:
            row[c] = row[c][keep]

    nrow = len(want)
    print(f"{nrow} galaxies paired across {len(names)} runs "
          f"(catalog {n}); window size {size}, flux {flux}\n")

    rng = np.random.default_rng(a.seed)
    boots = [rng.choice(nrow, nrow) for _ in range(a.boot)]
    sub = lambda c, i: triple(npz[c], row[c][i], size, flux)
    allrows = np.arange(nrow)

    b0 = sub("psfe00", allrows)
    print(f"{'run':>10} {'m1':>11} {'c1':>11} {'c2':>11}")
    for c in names:
        v = sub(c, allrows)
        lab = "e = 0" if c == "psfe00" else c
        print(f"{lab:>10} {v[0]:>+11.5f} {v[1]:>+11.3e} {v[2]:>+11.3e}")

    print(f"\npaired difference against e = 0 ({a.boot} bootstraps)")
    print(f"{'run':>10} {'dm1':>22} {'dc1':>22} {'dc2':>22}")
    for c in a.configs:
        dv = sub(c, allrows) - b0
        sd = np.std([sub(c, i) - sub("psfe00", i) for i in boots], axis=0)
        print(f"{c:>10}" + "".join(
            f"  {dv[j]:>+10.3e} +/- {sd[j]:.1e}" for j in range(3)))

    # The e1p/e1m ANTISYMMETRIC part isolates whatever is LINEAR in e_psf and
    # cancels everything even in it (including the common-mode scatter both
    # e1 configs share with the baseline).  A leak linear in e_psf must give
    # A_c1 == dc2 at e2p, because e_psf = (0, +e) is the 45-deg rotation of
    # e_psf = (+e, 0) and c1 rotates into c2.  If those two disagree, the
    # signal does not rotate and is not a PSF leak.
    if {"psfe1p", "psfe1m", "psfe2p"} <= set(a.configs):
        d1p, d1m = sub("psfe1p", allrows) - b0, sub("psfe1m", allrows) - b0
        d2p = sub("psfe2p", allrows) - b0
        A = 0.5 * (d1p - d1m)
        sdA = np.std([0.5 * ((sub("psfe1p", i) - sub("psfe00", i))
                             - (sub("psfe1m", i) - sub("psfe00", i)))
                      for i in boots], axis=0)
        sd2 = np.std([sub("psfe2p", i) - sub("psfe00", i) for i in boots],
                     axis=0)
        print(f"\nrotation test, the part LINEAR in e_psf:")
        print(f"  A_c1 = (dc1[e1p] - dc1[e1m])/2 = {A[1]:+.3e} +/- {sdA[1]:.1e}")
        print(f"  dc2[e2p]                       = {d2p[2]:+.3e} +/- {sd2[2]:.1e}")
        # The two share the psfe00 baseline and the galaxies, so their bars are
        # correlated and quoting them side by side understates the agreement.
        # Bootstrap the DIFFERENCE.
        def _rot_gap(i):
            b = sub("psfe00", i)
            return ((sub("psfe2p", i) - b)[2]
                    - 0.5 * ((sub("psfe1p", i) - b)[1]
                             - (sub("psfe1m", i) - b)[1]))
        gap = d2p[2] - A[1]
        sdg = float(np.std([_rot_gap(i) for i in boots]))
        print(f"  ratio {d2p[2] / A[1]:+.2f}x   (a leak that rotates gives 1.00)")
        print(f"  gap  = {gap:+.3e} +/- {sdg:.1e} (paired) = "
              f"{gap / sdg:+.1f} sigma from rotating")

    # --- C2: the amplitude scan ------------------------------------------
    # The SCALING EXPONENT localises the channel without needing more targets.
    # Measured off the catalogs alone (2026-09-04): `Sigma_X`'s spin-2 split
    # runs as e^1.017 and `C_M`'s as e^2.030, so an exponent near 1 puts the
    # leak in `Sigma_X` and near 2 puts it in `C_M`.
    #
    # Fit in the SIGNED linear variable, not log|dc|.  A log fit throws away
    # the sign, cannot use a point consistent with zero, and is biased by the
    # bar; `dc = A e^p` is fitted here by scanning p and least-squares-solving
    # the single linear amplitude A at each p, weighted by the paired bars.
    # The 0.05 point is expected to be under the noise (a linear channel gives
    # -1.9e-04 there against a bar that does not shrink), which is exactly why
    # all three amplitudes are fitted jointly rather than 0.05 read alone.
    for orient, chan, lab in [("psfe1p", 1, "dc1"), ("psfe2p", 2, "dc2")]:
        pts = sorted((c for c in a.configs if c.startswith(orient)),
                     key=amplitude)
        if len(pts) < 3:
            continue
        e = np.array([amplitude(c) for c in pts])
        y = np.array([(sub(c, allrows) - b0)[chan] for c in pts])
        sd = np.array([np.std([(sub(c, i) - sub("psfe00", i))[chan]
                               for i in boots]) for c in pts])
        print(f"\namplitude scan, {orient[-2:]} orientation, {lab}:")
        print(f"{'e_psf':>10}{lab:>13}{'+/- paired':>13}{'sigma':>8}")
        for k, c in enumerate(pts):
            print(f"{e[k]:>10.2f}{y[k]:>+13.3e}{sd[k]:>13.1e}"
                  f"{y[k] / sd[k]:>+8.1f}")
        w = 1.0 / sd ** 2
        ps = np.linspace(0.0, 4.0, 4001)
        chi2 = []
        for p in ps:
            b = e ** p
            amp = (w * b * y).sum() / (w * b * b).sum()
            chi2.append((w * (y - amp * b) ** 2).sum())
        chi2 = np.array(chi2)
        best = ps[int(np.argmin(chi2))]
        lo = ps[chi2 <= chi2.min() + 1.0]
        amp = ((w * e ** best * y).sum() / (w * e ** (2 * best)).sum())
        print(f"  exponent p = {best:.2f}  (1-sigma {lo.min():.2f} to "
              f"{lo.max():.2f})   -> 1 = Sigma_X, 2 = C_M")
        print(f"  {lab}(0.05) extrapolated = {amp * 0.05 ** best:+.2e}   "
              f"chi2/dof = {chi2.min() / max(len(e) - 2, 1):.2f}")

    print("\nprior-side terms, from the logs")
    for name, log in [("e = 0", f"logs/bias_v11_psfe00{a.suffix}.log")] + \
            [(c, f"logs/bias_v11_{c}{a.suffix}.log") for c in a.configs]:
        if not os.path.exists(log):
            continue
        txt = open(log).read()
        for key in ("P_s =", "R_s ="):
            m = [ln.strip() for ln in txt.splitlines() if key in ln]
            if m:
                print(f"  {name:>10}  {m[-1]}")

    print("\nA leak ROTATES: |dc1| at psfe1p ~ |dc2| at psfe2p, and psfe1m\n"
          "flips dc1's sign.  A dc that is the same in every config, or that\n"
          "sits inside its own bar, is not a PSF leak.")


if __name__ == "__main__":
    main()
