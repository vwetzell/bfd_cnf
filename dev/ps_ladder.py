"""Is `P_s(g)` quadratic over the shear the arms actually span?

eq. (45)-(46) keeps the selection correction to SECOND order in g: one number
`R_s = d2 P_s / dg2` stands in for the whole g-dependence.  That is only valid
while `P_s(g) - P_s(0)` really is `g^2 R_s / 2` across |g| = 0.02.

`P_s` is even in g (isotropy), so `(P_s(g) - P_s(0)) / g^2` is FLAT iff the
truncation holds.  A ceiling that sweeps a dense part of the population makes
it anything but.  No derivatives here -- just `window_prob` at a ladder of g,
which is cheap.
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
HIS = [float(x) for x in os.environ.get("HIS", "9e9 3.45 3.20 3.00").split()]
GS = [float(x) for x in os.environ.get("GS", "0.0025 0.005 0.01 0.02 0.04").split()]
NDRAW = int(os.environ.get("NDRAW", 262144))
BATCH = 16384

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
    cov = jnp.asarray(load_cov(f"{D}/{bias.CATALOGS[POP]['zero']}.fits"),
                      dtype=jnp.float64)
    z = flow.base_dist.sample(jr.key(31), (NDRAW,)).astype(jnp.float64)

    # COMMON RANDOM NUMBERS, and a FIXED draw set.  The signal here is
    # `P_s(g) - P_s(0)` down at 1e-6, while dropping ONE non-finite draw of
    # 262144 moves `P_s` by 3.8e-6 -- so a mask that varies with `g` swamps the
    # measurement outright.  Take the draws finite at every g on the ladder and
    # use exactly those throughout.
    @eqx.filter_jit
    def fvals(g, zz, size):
        m = jax.vmap(lambda z1: flow.bijection.transform(
            z1, condition(jnp.asarray(g, jnp.float64), sx)))(zz)
        return window_prob(m, cov, size, (FLUX_LO, 1e9))

    def col(g, size):
        return np.concatenate([np.asarray(fvals(jnp.array([g, 0.0]),
                                                z[i:i + BATCH], size))
                               for i in range(0, NDRAW, BATCH)])

    def ladder(size):
        f = {g: col(g, size) for g in [0.0] + GS + [-g for g in GS]}
        ok = np.ones(NDRAW, bool)
        for v in f.values():
            ok &= np.isfinite(v)
        if not ok.all():
            print(f"  {int((~ok).sum())} draws dropped (non-finite at some g)")
        return {g: v[ok].mean() for g, v in f.items()}

    for hi in HIS:
        size = (-np.inf, hi)
        p = ladder(size)
        print(f"\nMr/Mf <= {hi}, flux >= {FLUX_LO:g}: P_s(0) = {p[0.0]:.8f}")
        # Even and odd parts separate the R_s term from the Q_s one; `R_eff` is
        # exactly the central difference `selection_terms(fd=h)` forms, so this
        # table and the fd scan measure the same thing two ways.
        print(f"{'g':>9s}{'even = R/2 g^2':>16s}{'R_eff = 2even/g^2':>19s}"
              f"{'odd = Q g':>13s}{'Q_eff = odd/g':>15s}")
        for g in GS:
            ev = 0.5 * (p[g] + p[-g]) - p[0.0]
            od = 0.5 * (p[g] - p[-g])
            print(f"{g:9.4f}{ev:+16.3e}{2 * ev / g ** 2:19.4f}"
                  f"{od:+13.3e}{od / g:15.2e}", flush=True)
