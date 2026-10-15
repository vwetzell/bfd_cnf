"""G = Cov(noise in dm_dg, noise in the moments) per unit template C_ff -- the noise model's
sheared-template-noise coupling, from pure-noise stamps.

  python dev/noise_G.py TEMPLATES.fits OUT.npz [N]

A noisy template's dm_dg (bfd getTemplate) shears its own pixel noise, and that noise is
correlated with the moment noise.  A layer regressed onto D_T therefore learns
E[D_T|T] = E[D(m)|T] - G grad log p_t(T) (Tweedie): the model believes template noise
shears, which biases corrected m1 by about +0.0006 to +0.0009 on desv (2026-10-09).
`--noise-G` in shear.py/centroid.py adds G grad log p_t back onto the targets.

G only depends on the measurement (weight, PSF, stamp, pixel noise), the same way C_M does.
It is measured here on N stamps of unit-variance pure noise, about the stamp centre, with
sim._measure + sim.fill_row, the same calls a template row goes through.  No galaxy and no
noiseless template, so it is allowed for noisy-template flows.  It scales with the noise
variance, so it is stored per unit C_ff: G_i = G * cov_ff,i for template i.
2026-10-09: on the bdg2d templates this matches the twin-extracted G (bdg2d vs bdg2n).
"""
import os
import sys
from multiprocessing import Pool

import fitsio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bfd_cnf_imsims"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from imsims import sim  # noqa: E402

import bfd  # noqa: E402


def _block(args):
    lo, hi = args
    wt = bfd.KBlackmanHarris(weightSigma=sim.WEIGHT_SIGMA)
    row = np.zeros(1, dtype=sim.ROW_DTYPE)[0]
    gm, cff = np.zeros((2, 5, 5)), 0.0
    for i in range(lo, hi):
        img = np.random.default_rng([20261009, i]).normal(0.0, 1.0, (sim.STAMP_N, sim.STAMP_N))
        sim.fill_row(row, sim._measure(img, wt, 1.0, recenter=False))
        gm += np.einsum("gi,j->gij", row["dm_dg"], row["moments"])
        cff += row["cov"][0]
    return gm, cff


def main():
    tpath, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
    h = fitsio.read_header(tpath, 1)
    want = {"PIXSCALE": sim.PIXEL_SCALE, "WTSIGMA": sim.WEIGHT_SIGMA, "PSFSIGMA": sim.PSF_SIGMA,
            "PSFE1": sim.PSF_E[0], "PSFE2": sim.PSF_E[1], "PSFPROF": sim.PSF_PROFILE}
    bad = {k: (h[k], v) for k, v in want.items() if k in h and h[k] != v}
    assert not bad, f"templates were measured with other settings than sim's defaults: {bad}"
    step = 2000
    with Pool(int(os.environ.get("NPROC", 24))) as pool:
        parts = pool.map(_block, [(a, min(a + step, n)) for a in range(0, n, step)])
    G = sum(p[0] for p in parts) / n
    cff = sum(p[1] for p in parts) / n      # C_ff at unit pixel noise (bfd's own covariance)
    np.savez(out, G=G / cff, n=n, tmpl=os.path.basename(tpath))
    print(f"wrote {out}: G per unit C_ff from {n} noise stamps; g1 rows f r 1 2 c x cols f r 1 2 c:")
    print(np.array2string(G[0] / cff, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
