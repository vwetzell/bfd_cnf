"""`R_s11` at a set of size ceilings, for one flow, against the templates.

The templates carry bfd's exact `dm/dg` and ARE the true population, so their
`R_s` is the reference the flow has to match -- and `R_s` is what the residual
climb at a tight ceiling is sensitive to (dm1 ~ -0.21 per unit `R_s11`).

Deliberately NOT an m1: the per-target Q, R in a saved `--save-pqr` run belong
to the flow that produced them, and eq. (45)-(46)'s terms are a property of
that same model.  Mixing a new flow's selection terms into an old run's scores
is not the estimator for either.
"""
import os
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
import fitsio

import bias, bulk, shear
from bias import condition, selection_terms, load_cov

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
FLOW = os.environ.get("FLOW", "flows/centroid_bulgedisc_v2.eqx")
TRAIN = os.environ.get("TRAIN", None)
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
HIS = [float(x) for x in os.environ.get("HIS", "9e9 3.45 3.3 3.2 3.0").split()]
NDRAW = int(os.environ.get("NDRAW", 1048576))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "31 77 123").split()]
FD = float(os.environ.get("FD", 0.02))

if __name__ == "__main__":
    train = TRAIN or f"{D}/{bias.TRAIN_DATA[POP]}"
    flow = bulk.build_flow(jr.key(0), shear.load(train)[0], shear=True,
                           centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    sx = jnp.asarray(fitsio.read(f"{D}/{bias.CATALOGS[POP]['zero']}.fits",
                                 rows=[0])["cov_odd"][0], dtype=jnp.float64)
    cov = load_cov(f"{D}/{bias.CATALOGS[POP]['zero']}.fits")
    draw = lambda g, zz: jax.vmap(
        lambda z1: flow.bijection.transform(z1, condition(g, sx)))(zz)

    # Templates: the reference.  Its error comes from BLOCKS, not a half-split:
    # per-template d2F/dg2 is heavy-tailed, and on 2026-09-02 the half-split
    # returned 0.0004 where the true sem was 0.0039 -- a factor of 10, which is
    # what made a 3% gap look like 30 sigma.  See `dev/rs_blocks.py`.
    tm, tdm, td2m = (jnp.asarray(v, jnp.float32) for v in shear.load(
        f"{D}/{bias.TRAIN_DATA[POP]}"))
    tdraw = lambda g, idx: shear.lens(tm[idx], tdm[idx], td2m[idx],
                                      jnp.broadcast_to(g, (len(idx), 2)))
    zt = np.arange(len(tm))
    blocks = np.array_split(zt, 10)

    print(f"{FLOW}\n  chart from {train}\n  {NDRAW} draws x {len(SEEDS)} seeds, "
          f"fd = {FD}, flux >= {FLUX_LO:g}")
    print(f"\n{'Mr/Mf hi':>10s}{'flow R_s11':>14s}{'sd':>9s}"
          f"{'tmpl R_s11':>13s}{'+/-':>9s}{'flow/tmpl':>11s}")
    flux = (FLUX_LO, 1e9)
    for hi in HIS:
        size = (-np.inf, hi)
        vals = []
        for s in SEEDS:
            z = flow.base_dist.sample(jr.key(s), (NDRAW,)).astype(jnp.float64)
            vals.append(float(selection_terms(draw, z, cov, size, flux,
                                              fd=FD)[2][0, 0]))
        rt = float(selection_terms(tdraw, zt, cov, size, flux, fd=FD)[2][0, 0])
        rb_ = [float(selection_terms(tdraw, b, cov, size, flux, fd=FD)[2][0, 0])
               for b in blocks]
        et = np.std(rb_, ddof=1) / np.sqrt(len(blocks))
        m, sd = np.mean(vals), np.std(vals, ddof=1)
        print(f"{hi:10.2f}{m:14.4f}{sd:9.4f}{rt:13.4f}{et:9.4f}"
              f"{m / rt if rt else np.nan:11.3f}", flush=True)
