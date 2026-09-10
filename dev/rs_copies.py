"""Is the flow-vs-template `R_s` gap the CENTROID layer?

`selection_terms`' docstring says the eq. (36) detection factor "is what the
centroid layer already carries", and the two prior routes disagree about it:

  * FLOW  -- `flow.bijection.transform(z, condition(g, sx))` runs the whole
    stack, `CentroidMarginalize` included, so its draws are the RECENTRED
    population.
  * TEMPLATES -- `shear.lens(tm, tdm, td2m, g)` on `moments_bulgedisc_v2.fits`,
    i.e. the noiseless PERFECTLY CENTRED catalog moments lensed by the
    derivative table.  No shift enumeration anywhere.

`window_prob` convolves with `C_M` in both, so the noise is common; only the
recentring is not.  This settles it without touching a checkpoint: run the
template route on the 22.7M shifted COPIES, which carry recentring by
enumeration, and compare against the same route on the galaxies they were
built from.

The copies are NOT equally weighted (eq. 36's `w = d2u |J| N(X; 0, Sigma_X)`),
and `selection_terms` takes a plain mean, so they are importance-RESAMPLED to
an equally-weighted set first.  That reuses the validated code path instead of
writing a weighted variant of it.

**The control that matters is `copies` vs `moments_100k`** -- the same 100000
galaxies, differing only in whether the copy shifts are enumerated.  The 1M
catalog is included for scale, since it is what production uses and its own
galaxy-sampling error is 3x smaller.
"""
import os
import sys

import fitsio
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "../bfd_cnf_imsims")

import bias                                             # noqa: E402
import shear                                            # noqa: E402
from imsims.copies import log_weights                   # noqa: E402

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
COPIES = os.environ.get("COPIES", f"{D}/copies_bulgedisc_v2.fits")
FD = float(os.environ.get("FD", 0.02))
NDRAW = int(os.environ.get("NDRAW", 1 << 21))
WMIN = float(os.environ.get("WMIN", 1e-4))
SEED = int(os.environ.get("SEED", 0))
# One route per PROCESS: `selection_terms` builds a fresh `eqx.filter_jit`
# closure per call and the accumulated CUBINs OOM the GPU at about the 20th
# (HANDOFF 2026-09-02 night).  Three routes x five windows is over that.
ROUTE = os.environ.get("ROUTE", "copies")
# ... and one WINDOW per process too: the OOM is cumulative WITHIN a process
# and 1M rows hits it at the fifth window.  The copies resample is cached so
# the per-window reruns cost seconds instead of a 3 GB read each.
WINDOW = os.environ.get("WINDOW", "")
CACHE = os.environ.get("CACHE", "/tmp/rs_copies_resample.npz")

WINDOWS = {                                    # name: (size, flux)
    "flux only Mf>=1600": ((-np.inf, np.inf), (1600.0, 1e9)),
    "nominal": ((2.2, 3.2), (2500.0, 50000.0)),
    "band 3.005-3.252": ((3.005, 3.252), (1600.0, 1e9)),
    "band 3.252-3.407": ((3.252, 3.407), (1600.0, 1e9)),
    "band 3.407-3.530": ((3.407, 3.530), (1600.0, 1e9)),
}


def catalog_route(path):
    """(draw, z) for the plain template route on a moments catalog."""
    m, q, r = (jnp.asarray(v, jnp.float32) for v in shear.load(path))
    return (lambda g, idx: shear.lens(m[idx], q[idx], r[idx],
                                      jnp.broadcast_to(g, (len(idx), 2))),
            np.arange(len(m)))


def copies_route(sx, rng):
    """(draw, z) for the RECENTRED route: eq. (36)-weighted copies, resampled.

    Read once, in full: 2M scattered row reads out of 22.7M cost far more than
    the ~4 GB of a straight pass, and the weights need `moments` anyway
    (`log_jacobian`).
    """
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        print(f"  {CACHE}: {len(z['m'])} resampled copies, "
              f"{len(np.unique(z['gal']))} galaxies", flush=True)
        return (lambda g, i: shear.lens(
                    jnp.asarray(z["m"])[i], jnp.asarray(z["q"])[i],
                    jnp.asarray(z["r"])[i],
                    jnp.broadcast_to(g, (len(i), 2))),
                np.arange(len(z["m"])))
    c = fitsio.read(COPIES, ext="COPIES",
                    columns=["gal", "moments", "xy", "da", "dm_dg",
                             "d2m_dg2"])
    lw = log_weights(c, sx)
    keep = lw > lw.max() + np.log(WMIN)
    w = np.exp(lw[keep] - lw[keep].max())
    w /= w.sum()
    idx = rng.choice(np.flatnonzero(keep), NDRAW, replace=True, p=w)
    print(f"  {keep.sum()} of {len(lw)} copies over WMIN; resampled {NDRAW}, "
          f"{len(np.unique(idx))} distinct, "
          f"{len(np.unique(c['gal'][idx])) if 'gal' in c.dtype.names else '?'}"
          f" galaxies", flush=True)
    np.savez(CACHE, m=np.asarray(c["moments"][idx], np.float32),
             q=np.asarray(c["dm_dg"][idx], np.float32),
             r=np.asarray(c["d2m_dg2"][idx], np.float32),
             gal=np.asarray(c["gal"][idx]))
    m = jnp.asarray(c["moments"][idx], jnp.float32)
    q = jnp.asarray(c["dm_dg"][idx], jnp.float32)
    r = jnp.asarray(c["d2m_dg2"][idx], jnp.float32)
    del c
    return (lambda g, i: shear.lens(m[i], q[i], r[i],
                                    jnp.broadcast_to(g, (len(i), 2))),
            np.arange(NDRAW))


if __name__ == "__main__":
    cov = bias.load_cov(f"{D}/{bias.CATALOGS[POP]['zero']}.fits")
    sx = fitsio.read(f"{D}/{bias.CATALOGS[POP]['zero']}.fits",
                     rows=[0])["cov_odd"][0]
    rng = np.random.default_rng(SEED)

    print(f"route {ROUTE}, seed {SEED}")
    if ROUTE == "copies":
        draw, z = copies_route(sx, rng)                 # RECENTRED
    else:
        draw, z = catalog_route(                        # CENTRED
            f"{D}/moments_bulgedisc_v2{'_1M' if ROUTE == 'moments_1M' else ''}"
            f".fits")
        print(f"  {len(z)} galaxies")

    print(f"\n{'window':>22s}{'P_s':>10s}{'R_s11':>12s}{'Q_s1':>12s}")
    for name, (size, flux) in WINDOWS.items():
        if WINDOW and name != WINDOW:
            continue
        ps, qs, rs, _ = bias.selection_terms(draw, z, cov, size, flux, fd=FD)
        print(f"{name:>22s}{ps:10.4f}{rs[0, 0]:+12.4f}{qs[0]:+12.5f}",
              flush=True)
