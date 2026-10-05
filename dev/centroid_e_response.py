"""The centroid layer's ellipticity distortion against the copies', by |e|.

`centroid.check_sigmax` reduces this to one number (the training log's
"ellipticity response ... ratio 0.665"): the fractional change of each
galaxy's ellipticity along its own axis under centroid marginalisation,
layer vs. the eq.-36-weighted copy mean at g = 0.  Here it is per |e| bin,
plus the spin-0 shifts, to see whether the shortfall grows with |e| the way
the real-vs-closed-loop score excess does.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "dev")

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import bias as B
import bulk
import centroid as C
import shear
from closed_loop import D, REAL, S2, TRAIN

import os
COPIES = os.environ.get("COPIES", f"{D}/copies_gauss2_fwd_g2v4n.fits")   # with TRAIN_POP/NOISE_SCALE (closed_loop)
FLOW = os.environ.get("FLOW", "flows/centroid_g2v4n_Ke.eqx")
EBINS = [0, 0.05, 0.08, 0.11, 0.15, 0.2, np.inf]
ROWS = 30_000_000


def main():
    m_tr = np.asarray(shear.load(f"{D}/{B.TRAIN_DATA[TRAIN]}")[0])[:20000]
    flow = eqx.tree_deserialise_leaves(FLOW, bulk.build_flow(jr.key(0), m_tr, shear=True, centroid=True))
    layer, chart = C._sigmax_layer(flow), C._chart(flow)
    sx = S2 * fitsio.read(f"{D}/{REAL['plus']}.fits", rows=[0])["cov_odd"][0]
    with fitsio.FITS(COPIES) as f:
        copies = f["COPIES"][0:ROWS]
        galaxies = f["GALAXIES"].read()
    copies = copies[copies["gal"] < copies["gal"][-1]]          # drop the partial last galaxy
    target, keep = C.weighted_copy_mean(copies, galaxies, sx)
    have = np.zeros(len(galaxies), bool); have[np.unique(copies["gal"])] = True
    sel = have[keep]
    m0, target = galaxies["moments"][keep][sel], target[sel]
    cond = jnp.concatenate([jnp.zeros(2), jnp.asarray(sx, jnp.float32)])
    f = jax.jit(jax.vmap(lambda mi: C.sigmax_dm_dsigma(layer, mi, cond, chart)))
    pred = np.concatenate([np.asarray(f(jnp.asarray(m0[i:i + 8192], jnp.float32)), np.float64)
                           for i in range(0, len(m0), 8192)])
    truth = target - m0

    ec = lambda m: (m[:, 2] + 1j * m[:, 3]) / m[:, 1]
    e0 = ec(m0)
    de_cat = ((ec(target) - e0) * np.conj(e0)).real        # per-galaxy numerators
    de_lay = ((ec(m0 + pred) - e0) * np.conj(e0)).real
    den = (e0 * np.conj(e0)).real
    ae = np.abs(e0)
    rng = np.random.default_rng(0)
    print(f"{FLOW}: {len(m0)} galaxies (copies rows 0:{ROWS})")
    print(f"overall ellipticity response ratio layer/catalog {de_lay.sum() / de_cat.sum():.4f} "
          f"(training log: 0.665)\n")
    print(f"{'|e| bin':>12s} {'n':>6s}  {'catalog de/e':>12s} {'layer de/e':>12s} {'ratio':>14s}"
          f"   {'dMr/Mr cat':>10s} {'layer':>10s}   {'dMf/Mf cat':>10s} {'layer':>10s}")
    for lo, hi in zip(EBINS[:-1], EBINS[1:]):
        k = np.flatnonzero((ae >= lo) & (ae < hi))
        rc, rl = de_cat[k].sum() / den[k].sum(), de_lay[k].sum() / den[k].sum()
        bs = np.std([de_lay[b].sum() / de_cat[b].sum() for b in (rng.choice(k, len(k)) for _ in range(300))])
        r = lambda j, x: (x[k, j] / m0[k, j]).mean()
        print(f"{lo:5.2f}-{hi:<6.2f} {len(k):6d}  {rc:+12.4e} {rl:+12.4e} {rl / rc:7.4f}+/-{bs:.4f}"
              f"   {r(1, truth):+10.3e} {r(1, pred):+10.3e}   {r(0, truth):+10.3e} {r(0, pred):+10.3e}")



def structure():
    """Isotropic Sigma_X zeroes both D e and c.proj.e in `_ellipticity`, so the
    layer's ellipticity change is 1/kappa(flux) alone.  Show it: the copies'
    fractional e change at FIXED flux varies with size and |e|; the layer's
    cannot."""
    m_tr = np.asarray(shear.load(f"{D}/{B.TRAIN_DATA[TRAIN]}")[0])[:20000]
    flow = eqx.tree_deserialise_leaves(FLOW, bulk.build_flow(jr.key(0), m_tr, shear=True, centroid=True))
    layer, chart = C._sigmax_layer(flow), C._chart(flow)
    sx = S2 * fitsio.read(f"{D}/{REAL['plus']}.fits", rows=[0])["cov_odd"][0]
    with fitsio.FITS(COPIES) as f:
        copies = f["COPIES"][0:ROWS]
        galaxies = f["GALAXIES"].read()
    copies = copies[copies["gal"] < copies["gal"][-1]]
    target, keep = C.weighted_copy_mean(copies, galaxies, sx)
    have = np.zeros(len(galaxies), bool); have[np.unique(copies["gal"])] = True
    m0, target = galaxies["moments"][keep][have[keep]], target[have[keep]]
    cond = jnp.concatenate([jnp.zeros(2), jnp.asarray(sx, jnp.float32)])
    f = jax.jit(jax.vmap(lambda mi: C.sigmax_dm_dsigma(layer, mi, cond, chart)))
    pred = np.concatenate([np.asarray(f(jnp.asarray(m0[i:i + 8192], jnp.float32)), np.float64)
                           for i in range(0, len(m0), 8192)])
    ec = lambda m: (m[:, 2] + 1j * m[:, 3]) / m[:, 1]
    e0 = ec(m0); den = np.abs(e0) ** 2
    dc = ((ec(target) - e0) * np.conj(e0)).real
    dl = ((ec(m0 + pred) - e0) * np.conj(e0)).real
    flux, size, ae = m0[:, 0], m0[:, 1] / m0[:, 0], np.abs(e0)
    fq = np.quantile(flux, [0, 1 / 3, 2 / 3, 1]); fq[-1] += 1
    for name, v in (("size Mr/Mf", size), ("|e|", ae)):
        print(f"\nde/e (copies | layer) at fixed flux, by {name} tercile within each flux tercile")
        for a_, b_ in zip(fq[:-1], fq[1:]):
            k0 = (flux >= a_) & (flux < b_)
            vq = np.quantile(v[k0], [0, 1 / 3, 2 / 3, 1]); vq[-1] += 1e-9
            cells = []
            for c_, d_ in zip(vq[:-1], vq[1:]):
                k = k0 & (v >= c_) & (v < d_)
                cells.append(f"{dc[k].sum() / den[k].sum():+.2e} | {dl[k].sum() / den[k].sum():+.2e}")
            print(f"  flux {a_:7.0f}-{b_:<7.0f}  " + "     ".join(cells))


if __name__ == "__main__":
    structure() if sys.argv[1:2] == ["structure"] else main()
