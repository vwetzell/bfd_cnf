"""Is the resolution-dependent m1 the layer's shear response being wrong?

m1 is, to first order, the fractional error in the population's modelled shear
response.  The layer's leading spin-2 coefficient A is d(M1 + i M2)/dg1
normalised by Mr -- exactly what the `dm_dg` column measures -- so the two are
directly comparable and an error in A of size x shows up as an m1 of size ~x.

Evaluated on the copy catalog's GALAXIES table: exact moments AND exact
per-galaxy derivatives, no image noise anywhere, which is the flow's own
training population.  Nothing here is regressed or reconstructed.
"""
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bulk                                          # noqa: E402
from models.shear import ShearResponse, dm_dg        # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
DATA = "../bfd_cnf_imsims/data"
FLOW = "flows/centroid_deep.eqx"
TRAIN = f"{DATA}/copies_bulgedisc_deep.fits"


def find_layer(tree):
    got = [b for b in jax.tree_util.tree_leaves(
        tree, is_leaf=lambda x: isinstance(x, ShearResponse))
        if isinstance(b, ShearResponse)]
    assert len(got) == 1, f"expected one ShearResponse, found {len(got)}"
    return got[0]


def main():
    g = fitsio.read(TRAIN, ext="GALAXIES")
    M = np.asarray(g["moments"], np.float64)
    T = np.asarray(g["dm_dg"], np.float64)[:, 0, :]            # exact d m / d g1
    m_train = np.asarray(fitsio.read(TRAIN, ext="GALAXIES")["moments"], np.float64)
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    layer = find_layer(flow)
    print(f"loaded {FLOW}; {len(M)} exact galaxies\n")

    P = np.asarray(jax.jit(jax.vmap(lambda m: dm_dg(layer, m)[0]))(jnp.asarray(M)))
    r = M[:, 1] / M[:, 0]
    # A is the response of the spin-2 pair, normalised the way the layer defines
    # it; compare the whole 2-vector so a rotation of the response shows up too.
    A_t = T[:, 2] / M[:, 1]
    A_l = P[:, 0, 2] / M[:, 1]
    q = np.quantile(r, np.linspace(0, 1, 11)); q[0], q[-1] = -np.inf, np.inf
    print(f"{'<Mr/Mf>':>9s} {'N':>6s} {'A_true':>9s} {'A_layer':>9s} "
          f"{'frac err in A':>14s} {'dMf/dg1 frac':>13s} {'dMr/dg1 frac':>13s}")
    for k in range(10):
        s = (r >= q[k]) & (r < q[k + 1])
        at, al = A_t[s].mean(), A_l[s].mean()
        print(f"{r[s].mean():9.3f} {s.sum():6d} {at:9.4f} {al:9.4f} "
              f"{al / at - 1:>+14.4f} "
              f"{P[s, 0, 0].mean() / T[s, 0].mean() - 1:>+13.4f} "
              f"{P[s, 0, 1].mean() / T[s, 1].mean() - 1:>+13.4f}")
    print(f"\nwhole population: frac err in A = {A_l.mean() / A_t.mean() - 1:+.4f}")
    lo, hi = r < 3.5, r >= 3.5
    print(f"  Mr/Mf < 3.5 ({lo.sum():6d}): {A_l[lo].mean() / A_t[lo].mean() - 1:+.4f}")
    print(f"  Mr/Mf >= 3.5 ({hi.sum():6d}): {A_l[hi].mean() / A_t[hi].mean() - 1:+.4f}")


if __name__ == "__main__":
    main()
