"""Is `R_s11` converged in the number of prior draws at a TIGHT size ceiling?

`dev/window_leverage.py` says m1 moves -0.00065 per 1% of `R_s11` at
`Mr/Mf <= 3.0`, so the +0.017 there needs `R_s11` ~26% low.  Half the routes'
disagreement (flow 0.309, templates 0.267) is that size.  This walks the draw
count and the seed to separate "not enough draws" from "the two priors really
differ at the boundary".

Seeds vary with N -- `flow.base_dist.sample` at a fixed key NESTS, so raising N
alone reuses the smaller set and proves nothing (see [[pqr-streamed-seed-nests]]
for the same trap in the target draws).
"""
import os, sys
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
import fitsio

import bias, bulk, shear
from bias import condition, selection_terms, load_cov

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
FLOW = os.environ.get("FLOW", "flows/centroid_bulgedisc_v2.eqx")
FD = float(os.environ.get("FD", 0.02))
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
HIS = [float(x) for x in os.environ.get("HIS", "3.20 3.00").split()]
NS = [int(x) for x in os.environ.get("NS", "65536 262144 1048576").split()]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "31 77 123").split()]

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
    draw = lambda g, zz: jax.vmap(
        lambda z1: flow.bijection.transform(z1, condition(g, sx)))(zz)
    cov = load_cov(f"{D}/{bias.CATALOGS[POP]['zero']}.fits")
    flux = (FLUX_LO, 1e9)

    # The templates are the TRUE population, so their R_s is the reference --
    # but it is a sample mean with a known concentration problem, so report the
    # half-split spread beside it rather than trusting the value alone.
    tm, tdm, td2m = (jnp.asarray(v, jnp.float32) for v in shear.load(train))
    tdraw = lambda g, idx: shear.lens(tm[idx], tdm[idx], td2m[idx],
                                      jnp.broadcast_to(g, (len(idx), 2)))

    for hi in HIS:
        size = (-np.inf, hi)
        print(f"\nMr/Mf <= {hi}, flux >= {FLUX_LO:g}, fd = {FD}")
        print(f"{'N draws':>10s}" + "".join(f"{f'seed {s}':>12s}" for s in SEEDS)
              + f"{'mean':>10s}{'sd':>9s}")
        for n in NS:
            vals = []
            for s in SEEDS:
                z = flow.base_dist.sample(jr.key(s), (n,)).astype(jnp.float64)
                vals.append(float(
                    selection_terms(draw, z, cov, size, flux, fd=FD)[2][0, 0]))
            print(f"{n:10d}" + "".join(f"{v:12.4f}" for v in vals)
                  + f"{np.mean(vals):10.4f}{np.std(vals, ddof=1):9.4f}",
                  flush=True)
        zt = np.arange(len(tm))
        rt = selection_terms(tdraw, zt, cov, size, flux, fd=FD)[2][0, 0]
        h = len(zt) // 2
        ra = selection_terms(tdraw, zt[:h], cov, size, flux, fd=FD)[2][0, 0]
        rb = selection_terms(tdraw, zt[h:], cov, size, flux, fd=FD)[2][0, 0]
        print(f"{'templates':>10s}{rt:12.4f}   (halves {ra:.4f} / {rb:.4f})")
