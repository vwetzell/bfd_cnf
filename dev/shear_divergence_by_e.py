"""The shear layer's divergence term against the data's, by galaxy |e|.

A density transported by the velocity v(z) = dz/dg has score
    d log P / dg = -(grad log P . v + div v)        (continuity equation)
The flow's sheared path has v = -Q(z) (`ShearResponse.shear`), so its log-det
contributes +tr(dQ/dz) per shear component.  The data's v is the conditional
mean E[dz/dg | z] of the galaxies' exact response (J_chart dm/dg); its
divergence is estimated here by a local LINEAR regression of the exact
dz/dg on z over the k nearest prior galaxies (the whole 500k catalog as the
neighbour pool), tr of the fitted 5x5 slope.  Two k values bracket the
regression's own smoothing bias.  Evaluated at held-out galaxies.
"""
import sys
sys.path.insert(0, ".")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.spatial import cKDTree

import bias as B
import bulk
import shear

PRIOR = f"../bfd_cnf_imsims/data/{B.TRAIN_DATA['gauss2_v4n']}"
FLOW = "flows/shear_g2v4n.eqx"
EBINS = [0, 0.05, 0.08, 0.11, 0.15, 0.2, np.inf]
N_EVAL, KS = 20000, (128, 512)


def main():
    m, dm, _ = shear.load(PRIOR)
    n_tr = int(0.9 * len(m))
    flow = eqx.tree_deserialise_leaves(
        FLOW, bulk.build_flow(jr.key(0), m[:n_tr][:20000], shear=True, centroid=False))
    chart, layer = flow.bijection.bijection.bijections[:2]

    @jax.jit
    def zv(mi, dmi):
        z = chart.transform(mi)
        v = jnp.stack([jax.jvp(chart.transform, (mi,), (dmi[a],))[1] for a in range(2)], -1)
        return z, v

    @jax.jit
    def flow_div(z):
        dQ = jax.jacfwd(lambda zz: layer.response_tensors(zz)[0])(z)   # (5,2,5)
        return -jnp.einsum("iai->a", dQ)          # div of v = -Q, per shear component

    zs, vs = [], []
    for i in range(0, len(m), 16384):
        z, v = jax.vmap(zv)(jnp.asarray(m[i:i + 16384], jnp.float32), jnp.asarray(dm[i:i + 16384], jnp.float32))
        zs.append(np.asarray(z, np.float64)); vs.append(np.asarray(v, np.float64))
    Z, V = np.concatenate(zs), np.concatenate(vs)
    ok = np.isfinite(Z).all(1) & np.isfinite(V).all((1, 2))
    Z, V, mm = Z[ok], V[ok], m[ok]
    held = np.flatnonzero(np.arange(len(m))[ok] >= n_tr)
    ev = np.random.default_rng(0).choice(held, N_EVAL, replace=False)
    fd = np.concatenate([np.asarray(jax.vmap(flow_div)(jnp.asarray(Z[ev[i:i + 4096]], jnp.float32)), np.float64)
                         for i in range(0, N_EVAL, 4096)])
    tree = cKDTree(Z)
    e = np.hypot(mm[ev, 2], mm[ev, 3]) / mm[ev, 1]

    data_div = {}
    for k in KS:
        _, nb = tree.query(Z[ev], k=k + 1, workers=-1)
        nb = nb[:, 1:]                                         # exclude the point itself
        dz = Z[nb] - Z[ev][:, None, :]                         # (n,k,5)
        X = np.concatenate([np.ones(dz.shape[:2] + (1,)), dz], -1)   # (n,k,6)
        XtX = np.einsum("nki,nkj->nij", X, X)
        div = np.zeros((N_EVAL, 2))
        for a in range(2):
            XtY = np.einsum("nki,nkj->nij", X, V[nb][:, :, :, a])   # (n,6,5)
            coef = np.linalg.solve(XtX, XtY)                    # rows 1..5 = d v_j / d z_i
            div[:, a] = np.einsum("nii->n", coef[:, 1:, :])
        data_div[k] = div

    print(f"{FLOW}: divergence of the shear velocity field, {N_EVAL} held-out galaxies\n")
    print("slope of data on flow per |e| bin (1 = flow's divergence right; data is the noisy side),"
          " both shear components pooled")
    print(f"{'|e| bin':>12s} {'n':>6s}  {'rms flow':>9s}   " + "   ".join(f"{'k=' + str(k):>16s}" for k in KS))
    rng = np.random.default_rng(1)
    for lo, hi in zip(EBINS[:-1], EBINS[1:]):
        s_ = np.flatnonzero((e >= lo) & (e < hi))
        f = fd[s_]
        cells = []
        for k in KS:
            d = data_div[k][s_]
            sl = lambda i: (d[i] * f[i]).sum() / (f[i] ** 2).sum()
            bs = np.std([sl(rng.integers(0, len(s_), len(s_))) for _ in range(300)])
            cells.append(f"{sl(slice(None)):.3f}+/-{bs:.3f}")
        print(f"{lo:5.2f}-{hi:<6.2f} {len(s_):6d}  {np.sqrt((f ** 2).mean()):9.4f}   " + "   ".join(f"{c:>16s}" for c in cells))
    for k in KS:
        c = np.corrcoef(fd[:, 0], data_div[k][:, 0])[0, 1]
        print(f"per-galaxy corr(flow, data k={k}) g1: {c:.3f}   "
              f"g2: {np.corrcoef(fd[:, 1], data_div[k][:, 1])[0, 1]:.3f}")


if __name__ == "__main__":
    main()
