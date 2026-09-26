"""Shear layer's spin-0 response at the window's size edges vs exact.

Real-vs-closed windowed m1 differs only in the Mr/Mf edge bins
([[real-vs-closed-gap-is-size-edge]]).  Selection at a size cut responds to
shear through d(Mr/Mf)/dg, which is spin-0 and proportional to e -- the part of
the shear layer known to be off for round galaxies.  Here, per real in-window
target (binned on g = 0 observed Mr/Mf like the m1 split): the flow's
noiseless transport  m(g) = chart^-1(shear^-1(u, g)),  u = shear(chart(m0), 0),
finite-differenced at +/-DELTA, against the exact analytic dm/dg of the same
noiseless galaxy.  The selection-relevant average is <e1 d(Mr/Mf)/dg1>: an
isotropic population only moves a size cut through that correlation.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "dev")
sys.path.insert(0, "../bfd_cnf_imsims")
import fitsio
import jax
import jax.numpy as jnp
import numpy as np

import bias as B
from closed_loop import load_flow
from imsims import analytic

D = "../bfd_cnf_imsims/data"
DELTA = 0.02
EDGES = [2.2, 2.3, 2.45, 2.95, 3.1, 3.2]

path = f"{D}/{B.CATALOGS['gauss2_v4n']['zero']}.fits"
t = fitsio.read(path)
pop = fitsio.read(path, ext="POPULATION")
win = B.window_mask(t["moments"], (2.2, 3.2), (3000.0, 20000.0)) & ~t["badcenter"]
t, pop = t[win], pop[win].astype(pop.dtype.newbyteorder("="))
rho = pop["bulge_ratio"]
theta = jnp.stack([jnp.log(pop["flux"]), jnp.log(pop["sigma"]), jnp.log(rho) - jnp.log1p(-rho),
                   pop["e1"], pop["e2"]], -1)
m0 = np.asarray(analytic.moments_batch(theta), np.float64)
dm = np.asarray(t["dm_dg"], np.float64)[:, 0]              # exact d/dg1 at g = 0

jax.config.update("jax_enable_x64", False)   # analytic turns it on; the flow is f32
flow = load_flow("flows/centroid_g2v4n_Ke.eqx")
chain = flow.bijection.bijection.bijections
chart = chain[0]
sh = next(b for b in chain if type(b).__name__ == "ShearResponse")
sx = jnp.asarray(t["cov_odd"][0], jnp.float32)


@jax.jit
def flow_m(m, g):
    u = sh.transform(chart.transform(m), B.condition(jnp.zeros(2), sx))
    return chart.inverse(sh.inverse(u, B.condition(g, sx)))


mj = jnp.asarray(m0, jnp.float32)
fp = np.asarray(jax.vmap(lambda m: flow_m(m, jnp.array([DELTA, 0.0])))(mj), np.float64)
fm = np.asarray(jax.vmap(lambda m: flow_m(m, jnp.array([-DELTA, 0.0])))(mj), np.float64)
df = (fp - fm) / (2 * DELTA)

size = lambda m: m[:, 1] / m[:, 0]
ds_ex = dm[:, 1] / m0[:, 0] - m0[:, 1] * dm[:, 0] / m0[:, 0] ** 2      # d(Mr/Mf)/dg1
ds_fl = size(fp) / 2 / DELTA - size(fm) / 2 / DELTA
e1 = m0[:, 2] / m0[:, 1]
r_obs = size(t["moments"])
slope = lambda y, x: (x * y).sum() / (x * x).sum()

print(f"{len(t)} in-window real targets; flow = shear layer only, exact = analytic dm/dg (g1)")
print(" Mr/Mf bin     n   slope d(Mr/Mf)  <e1 dS>_flow/<e1 dS>_exact   slope dMf  slope dM1")
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    s = (r_obs >= lo) & (r_obs < hi)
    x, y = ds_ex[s], ds_fl[s]
    k = np.random.default_rng(0).integers(0, s.sum(), (200, s.sum()))
    rat = lambda i: (e1[s][i] * y[i]).mean() / (e1[s][i] * x[i]).mean()
    boot = np.std([rat(i) for i in k])
    print(f" {lo:.2f}-{hi:.2f}  {s.sum():5d}   {slope(y, x):.4f}          {rat(slice(None)):.4f} +/- {boot:.4f}"
          f"             {slope(df[s, 0], dm[s, 0]):.4f}     {slope(df[s, 2], dm[s, 2]):.4f}")
