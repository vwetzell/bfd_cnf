"""Does the flow's dm/dg residual, coordinate by coordinate, explain `R_s`?

The templates ARE the true population and carry bfd's exact `dm_dg`/`d2m_dg2`,
so their `R_s` is the reference.  Here one column at a time of the templates'
`dm_dg` is REPLACED by the flow's own response at the same moments (the layer's
response is a deterministic function of `m`, so this is well defined), leaving
everything else exact.  If swapping `Mr` alone carries the templates' `R_s11`
from -0.2492 to the flow's -0.2199, the chain "spin-0 response error -> weak
`R_s`" is established; if no single swap does, the 12% is coming from
elsewhere.

`R2=1` swaps `d2m_dg2` as well as `dm_dg`.
"""
import os
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
import fitsio

import bias, bulk, shear
from bias import condition, selection_terms, load_cov
from models.shear import dm_dg

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
FLOW = os.environ.get("FLOW", "flows/centroid_bulgedisc_v2.eqx")
TRAIN = os.environ.get("TRAIN", None)
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
HIS = [float(x) for x in os.environ.get("HIS", "9e9").split()]
FD = float(os.environ.get("FD", 0.02))
R2 = int(os.environ.get("R2", 0))
NAMES = ["Mf", "Mr", "M1", "M2", "Mc"]

if __name__ == "__main__":
    train = TRAIN or f"{D}/{bias.TRAIN_DATA[POP]}"
    flow = bulk.build_flow(jr.key(0), shear.load(train)[0], shear=True,
                           centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    cov = load_cov(f"{D}/{bias.CATALOGS[POP]['zero']}.fits")

    tm, tdm, td2m = (jnp.asarray(v, jnp.float64) for v in shear.load(
        f"{D}/{bias.TRAIN_DATA[POP]}"))
    layer, chart = shear._shear_layer(flow), shear._chart(flow)
    fq, fr = jax.lax.map(lambda m: dm_dg(layer, m, chart), tm, batch_size=4096)

    s = shear._scale(tm)[:, None, :]
    rms = lambda a: np.sqrt(np.mean(np.asarray(a / s) ** 2, axis=(0, 1)))
    print(f"{FLOW}\n  chart from {train}\n  {len(tm)} templates, fd = {FD}, "
          f"flux >= {FLUX_LO:g}, swap d2m/dg2 = {bool(R2)}")
    print("\nflow dm/dg residual vs bfd (RMS/RMS): " + "".join(
        f"{n}{v:8.2%}  " for n, v in zip(NAMES, rms(fq - tdm) / rms(tdm))))

    zt = np.arange(len(tm))
    h = len(zt) // 2

    def rs(q, r, idx, size):
        d = lambda g, i: shear.lens(tm[i], q[i], r[i],
                                    jnp.broadcast_to(g, (len(i), 2)))
        return float(selection_terms(d, idx, cov, size, (FLUX_LO, 1e9),
                                     fd=FD)[2][0, 0])

    cases = [("exact", None)] + [(f"swap {n}", i) for i, n in enumerate(NAMES)]
    cases += [("swap Mr,Mc", [1, 4]), ("swap all", list(range(5)))]
    for hi in HIS:
        size = (-np.inf, hi)
        print(f"\nMr/Mf <= {hi:g}\n{'case':>12s}{'R_s11':>10s}{'half-split':>12s}")
        for name, cols in cases:
            q, r = tdm, td2m
            if cols is not None:
                c = jnp.asarray(np.atleast_1d(cols))
                q = tdm.at[:, :, c].set(fq[:, :, c])
                if R2:
                    r = td2m.at[:, :, c].set(fr[:, :, c])
            v = rs(q, r, zt, size)
            # Paired comparison on the SAME templates, so the half-split is
            # only there to sanity-check the exact row; at a size ceiling it
            # is both meaningless (one template can carry 39%) and expensive
            # (a recompile per batch size, which OOMs).
            e = (abs(rs(q, r, zt[:h], size) - rs(q, r, zt[h:], size)) / 2
                 if int(os.environ.get("HALF", 1)) else np.nan)
            print(f"{name:>12s}{v:10.4f}{e:12.4f}", flush=True)
