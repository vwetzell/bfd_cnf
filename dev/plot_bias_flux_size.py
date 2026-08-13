"""Hexbin of multiplicative bias in the flux-size (log Mf, Mr/Mf) plane.

Noiseless path: exact catalog `dm_dg`/`d2m_dg2` lens each galaxy to +/-0.02 in
g1, so PQR is a point evaluation and pairing is exact -- the same galaxy, same
`m`, enters both arms.

Bin assignment is on the UNSHEARED moments (`m[:,0]`, `m[:,1]/m[:,0]`), not on
the lensed ones.  A galaxy's hex cell is therefore identical in the +g and -g
arms, so `dP(s|g)/dg = 0` for the cut a hexagon boundary imposes and no
selection term is induced (same reasoning as `dev/check_bias_by_selection.py`,
which cuts on the g=0 catalog for the same reason).

    python dev/plot_bias_flux_size.py [--flow flows/shear.eqx]
"""
import argparse
import sys

import equinox as eqx
import jax.random as jr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402

DATA = "../bfd_cnf_imsims/data"
G = 0.02
GRIDSIZE = 40
MINCNT = 200
NBOOT = 60
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]         # blue - neutral - red


def lensed(m, dm, d2m, g):
    quad = np.array([0.5 * g[0] ** 2, g[0] * g[1], 0.5 * g[1] ** 2])
    return (m + np.einsum("i,bij->bj", np.asarray(g), dm)
            + np.einsum("i,bij->bj", quad, d2m))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--targets", default=f"{DATA}/targets_g0_1M.fits")
    p.add_argument("--train-data", default=f"{DATA}/moments.fits")
    p.add_argument("--out", default="dev/bias_flux_size.png")
    p.add_argument("--cache", default="dev/bias_flux_size_cache.npz",
                   help="save/reuse the per-target PQR here")
    a = p.parse_args()

    if a.cache and __import__("os").path.exists(a.cache):
        d = np.load(a.cache)
        x, y, qp, rp, qm, rm = (d[k] for k in ("x", "y", "qp", "rp", "qm", "rm"))
        print(f"loaded {a.cache}: {len(x)} targets")
    else:
        m, dm, d2m = (np.asarray(v, np.float64) for v in shear_top.load(a.targets))
        m_train = shear_top.load(a.train_data)[0]
        m_train = m_train[:int(0.9 * len(m_train))]
        flow = bulk.build_flow(jr.key(0), m_train, shear=True)
        flow = eqx.tree_deserialise_leaves(a.flow, flow)
        print(f"{a.flow} on {len(m)} exact noiseless targets from {a.targets}")

        qp, rp = (np.asarray(v, np.float64)
                 for v in bias.pqr(flow, lensed(m, dm, d2m, (+G, 0.0))))
        qm, rm = (np.asarray(v, np.float64)
                 for v in bias.pqr(flow, lensed(m, dm, d2m, (-G, 0.0))))
        fin = (np.isfinite(qp).all(1) & np.isfinite(rp).reshape(len(m), -1).all(1)
               & np.isfinite(qm).all(1) & np.isfinite(rm).reshape(len(m), -1).all(1))
        print(f"  dropping {int((~fin).sum())} non-finite targets")
        qp, rp, qm, rm = qp[fin], rp[fin], qm[fin], rm[fin]

        # Bin variables from the UNSHEARED catalog -- see module docstring.
        x = np.log10(m[fin, 0])
        y = m[fin, 1] / m[fin, 0]
        if a.cache:
            np.savez_compressed(a.cache, x=x, y=y, qp=qp, rp=rp, qm=qm, rm=rm)
            print(f"wrote {a.cache}")

    def m1_of(idx):
        idx = np.asarray(idx, dtype=int)
        return bias.bias(qp[idx], rp[idx], qm[idx], rm[idx], G)[0]

    def z_of(idx):
        idx = np.asarray(idx, dtype=int)
        v = m1_of(idx)
        err = bias.bootstrap(qp[idx], rp[idx], qm[idx], rm[idx], G, n=NBOOT)[0]
        return v / err if err > 0 else np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    cmap = mcolors.LinearSegmentedColormap.from_list("diverging", DIVERGING)
    extent = (*np.percentile(x, [0.2, 99.8]), *np.percentile(y, [0.2, 99.8]))

    hx1 = ax1.hexbin(x, y, C=np.arange(len(x)), reduce_C_function=m1_of,
                     gridsize=GRIDSIZE, mincnt=MINCNT, cmap=cmap, extent=extent,
                     norm=mcolors.CenteredNorm(vcenter=0, halfrange=0.02))
    fig.colorbar(hx1, ax=ax1, label="m1")
    ax1.set_title("multiplicative bias m1")

    hx2 = ax2.hexbin(x, y, C=np.arange(len(x)), reduce_C_function=z_of,
                     gridsize=GRIDSIZE, mincnt=MINCNT, cmap=cmap, extent=extent,
                     norm=mcolors.CenteredNorm(vcenter=0, halfrange=3))
    fig.colorbar(hx2, ax=ax2, label="m1 / bootstrap sd")
    ax2.set_title("m1, normalized by per-cell error")

    for ax in (ax1, ax2):
        ax.set_xlabel("log10(Mf)  [flux]")
        ax.set_ylabel("Mr/Mf  [size]")

    fig.suptitle(f"{a.flow}  ({len(x)} targets, mincnt={MINCNT})")
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
