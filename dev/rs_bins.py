"""Split the flow-vs-template `R_s11` gap into density (bin weights) and
response (per-bin curvature), binned by the draw's own unsheared `Mf`.

`R_s11 = sum_b w_b f_b` with `f_b` the mean of the per-draw central difference
`(F(+h) + F(-h) - 2 F(0)) / h^2` over bin `b` and `w_b` the bin's share of the
population.  Both routes use the SAME bins, so the two sums are directly
comparable term by term: a gap in `w_b` is a density error, a gap in `f_b` is a
response error.
"""
import os
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
import fitsio

import bias, bulk, shear
from bias import condition, window_prob, load_cov

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
FLOW = os.environ.get("FLOW", "flows/centroid_bulgedisc_v2.eqx")
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
HI = float(os.environ.get("HI", 9e9))
NDRAW = int(os.environ.get("NDRAW", 1048576))
SEED = int(os.environ.get("SEED", 31))
FD = float(os.environ.get("FD", 0.02))
EDGES = np.array([float(x) for x in os.environ["EDGES"].split()]) \
    if "EDGES" in os.environ else \
    np.array([0, 800, 1200, 1600, 2000, 2600, 3400, 5000, 1e4, 1e9])

if __name__ == "__main__":
    train = f"{D}/{bias.TRAIN_DATA[POP]}"
    flow = bulk.build_flow(jr.key(0), shear.load(train)[0], shear=True,
                           centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    sx = jnp.asarray(fitsio.read(f"{D}/{bias.CATALOGS[POP]['zero']}.fits",
                                 rows=[0])["cov_odd"][0], dtype=jnp.float64)
    cov = load_cov(f"{D}/{bias.CATALOGS[POP]['zero']}.fits")
    size, flux = (-np.inf, HI), (FLUX_LO, 1e9)

    def curvature(draw, z, batch=16384):
        """(Mf at g=0, per-draw d2F/dg1^2) for a set of base points."""
        e = jnp.array([FD, 0.0])
        f = lambda g, zz: window_prob(draw(g, zz), cov, size, flux)

        @jax.jit
        def one(zz):
            m0 = draw(jnp.zeros(2), zz)
            return m0[:, 0], (f(e, zz) + f(-e, zz) - 2 * f(jnp.zeros(2), zz)) / FD**2
        mf, d2 = [], []
        for i in range(0, len(z), batch):
            a, b = one(z[i:i + batch])
            mf.append(np.asarray(a)); d2.append(np.asarray(b))
        return np.concatenate(mf), np.concatenate(d2)

    zf = flow.base_dist.sample(jr.key(SEED), (NDRAW,)).astype(jnp.float64)
    mf_f, d2_f = curvature(
        lambda g, zz: jax.vmap(lambda z1: flow.bijection.transform(
            z1, condition(g, sx)))(zz), zf)

    tref = os.environ.get("TREF", train)
    tm, tdm, td2m = (jnp.asarray(v, jnp.float64) for v in shear.load(tref))
    mf_t, d2_t = curvature(
        lambda g, i: shear.lens(tm[i], tdm[i], td2m[i],
                                jnp.broadcast_to(g, (len(i), 2))),
        jnp.arange(len(tm)))

    print(f"{FLOW}\n  {NDRAW} flow draws (seed {SEED}) vs {len(tm)} templates, "
          f"fd = {FD}, flux >= {FLUX_LO:g}, Mr/Mf <= {HI:g}")
    print(f"\n{'Mf bin':>16s}{'w flow':>9s}{'w tmpl':>9s}{'w ratio':>9s}"
          f"{'f flow':>10s}{'f tmpl':>10s}{'f ratio':>9s}"
          f"{'contrib flow':>14s}{'tmpl':>10s}")
    tot_f = tot_t = 0.0
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sf, st = (mf_f >= lo) & (mf_f < hi), (mf_t >= lo) & (mf_t < hi)
        wf, wt = sf.mean(), st.mean()
        ff = d2_f[sf].mean() if sf.any() else 0.0
        ft = d2_t[st].mean() if st.any() else 0.0
        tot_f += wf * ff; tot_t += wt * ft
        print(f"{lo:7.0f}-{hi:8.0f}{wf:9.4f}{wt:9.4f}{wf / wt if wt else np.nan:9.3f}"
              f"{ff:10.4f}{ft:10.4f}{ff / ft if ft else np.nan:9.3f}"
              f"{wf * ff:14.4f}{wt * ft:10.4f}")
    print(f"{'total':>16s}{'':27s}{'':29s}{tot_f:14.4f}{tot_t:10.4f}")
    # Counterfactuals: flow response on template weights, and vice versa.
    wf = np.array([((mf_f >= lo) & (mf_f < hi)).mean()
                   for lo, hi in zip(EDGES[:-1], EDGES[1:])])
    wt = np.array([((mf_t >= lo) & (mf_t < hi)).mean()
                   for lo, hi in zip(EDGES[:-1], EDGES[1:])])
    ff = np.array([d2_f[(mf_f >= lo) & (mf_f < hi)].mean() if ((mf_f >= lo) & (mf_f < hi)).any() else 0.0
                   for lo, hi in zip(EDGES[:-1], EDGES[1:])])
    ft = np.array([d2_t[(mf_t >= lo) & (mf_t < hi)].mean() if ((mf_t >= lo) & (mf_t < hi)).any() else 0.0
                   for lo, hi in zip(EDGES[:-1], EDGES[1:])])
    print(f"\ntemplate weights x flow response: {float(wt @ ff):.4f}")
    print(f"flow weights x template response: {float(wf @ ft):.4f}")
