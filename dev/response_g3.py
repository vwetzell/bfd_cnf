"""Is the prior's quadratic-in-g moment model wrong, or just truncated?

Flow and template sum both lens a template by `m + g.dm_dg + g.d2m_dg2.g/2`,
so any error in the stored `dm_dg` is SHARED by both and survives everything
`dev/prior_n.py` exonerated the flow of.  The rendered catalogs measure it with
no model at all: over many galaxies the pixel noise averages out of

    D(g) = (m_+ - m_-) / 2g - dm_dg = g^2/6 d3m/dg3 + O(g^4)

so `D` is the TRUNCATION if it scales as `g^2`, and a WRONG STORED DERIVATIVE
if it does not move.  `imsims/sim.py` claims bulgedisc's first derivatives are
good to 0.009% (the known defect is in `d2m_dg2`, from KBlackmanHarris's
`W''(kmax) != 0`); this is the check of that claim at the compact end, where
`D/|dm_dg|` reaches -27%.

`G` selects which pair to read.  The two renders share `--seed 0 --n 200000
--pop bulgedisc --add-noise --noise-sigma 0.9`, so they are the same galaxies
as `targets_deep_*_200k_v2` and `dm_dg` / the binning come from the g = 0 arm.

    G=0.02 python dev/response_g3.py
    G=0.005 PLUS=$S/tg1p005_200k.fits MINUS=$S/tg1m005_200k.fits python dev/response_g3.py
"""
import os

import fitsio
import numpy as np

D = "../bfd_cnf_imsims/data"
G = float(os.environ.get("G", 0.02))
ZERO = os.environ.get("ZERO", f"{D}/targets_deep_g0_200k_v2.fits")
PLUS = os.environ.get("PLUS", f"{D}/targets_deep_g1p02_200k_v2.fits")
MINUS = os.environ.get("MINUS", f"{D}/targets_deep_g1m02_200k_v2.fits")
# Bin on a CLEAN g = 0 render when the arms are noisy: the g = 0 arm shares
# their noise field (same --seed), and recentring keeps a little of it in
# `m_+ - m_-`, so binning on that arm's own Mr/Mf correlates the bin with the
# residual being measured.  A noiseless render of the same galaxies does not.
BIN = os.environ.get("BIN", "")
SIZE = [float(x) for x in os.environ.get(
    "SIZE", "0 3.005 3.252 3.407 3.530 9").split()]
NAMES = ["Mf", "Mr", "M1", "M2", "Mc"]

if __name__ == "__main__":
    z = fitsio.read(ZERO)
    m0 = np.asarray(z["moments"], float)
    mp = np.asarray(fitsio.read(PLUS)["moments"], float)
    mm = np.asarray(fitsio.read(MINUS)["moments"], float)
    dm = np.asarray(z["dm_dg"], float)[:, 0]        # d m / d g1
    assert len(m0) == len(mp) == len(mm), "arms must be the same galaxies"
    if BIN:
        b = fitsio.read(BIN)
        assert np.allclose(np.asarray(b["dm_dg"], float)[:, 0], dm), \
            "BIN must be the same galaxies as ZERO"
        m0 = np.asarray(b["moments"], float)

    odd = (mp - mm) / (2 * G) - dm
    size = m0[:, 1] / m0[:, 0]
    print(f"g = {G:g}, {len(m0)} galaxies\n"
          f"  {os.path.basename(PLUS)} / {os.path.basename(MINUS)}\n"
          f"  binned on {os.path.basename(BIN or ZERO)}\n\n"
          f"D/<|dm_dg|> per component, binned on the g=0 arm's Mr/Mf "
          f"(sem in brackets)")
    print(f"{'Mr/Mf':>16s}{'n':>8s}" + "".join(f"{x:>18s}" for x in NAMES))
    for lo, hi in zip(SIZE[:-1], SIZE[1:]):
        k = (size >= lo) & (size < hi)
        scale = np.abs(dm[k]).mean(0)
        frac, err = odd[k].mean(0) / scale, odd[k].std(0) / np.sqrt(k.sum()) / scale
        print(f"{lo:7.3f}-{hi:<8.3f}{k.sum():8d}"
              + "".join(f"{f:+9.4f} ({e:.4f})" for f, e in zip(frac, err)))
