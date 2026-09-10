"""Why does the flow degrade as Mr/Mf approaches POINT_SOURCE?

Measures, on the v3 NOISELESS prior (whose dm_dg columns are the catalog's own
exact derivatives), three things per Mr/Mf band:

  * the chart's amplification 1/(1-u), u = Mr/(POINT_SOURCE Mf), which is
    exactly `_chart_spin0_jac`'s d1 up to 1/scale;
  * the PHYSICAL size response d ln(Mr/Mf)/dg1 the catalog actually has, and
    whether it vanishes as u -> 1 the way a point source demands;
  * the flow's own response against it, split into the physical factor and the
    amplified z-space quantity the density is built on.

    PYTHONPATH=. python dev/edge_geometry.py
"""
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


import bulk                                          # noqa: E402
import shear as shear_mod                            # noqa: E402
from models.bijections import POINT_SOURCE           # noqa: E402
from models.shear import ShearResponse, dm_dg        # noqa: E402

D = "../bfd_cnf_imsims/data"
FLOW = os.environ.get("FLOW", "flows/centroid_v3.eqx")
CAT = os.environ.get("CAT", f"{D}/moments_bulgedisc_v3.fits")
N = int(os.environ.get("N", 20000))


def main():
    m, dmdg, _ = shear_mod.load(CAT)
    m = np.asarray(m, np.float64)
    dmdg = np.asarray(dmdg, np.float64)          # (n, 2, 5): d m_a / d g_b
    flow = bulk.build_flow(jr.key(0), m, shear=True, centroid=True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, flow)
    # deserialise in float32 (as written), then promote -- eqx checks dtypes
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    bij = flow.bijection.bijection.bijections
    chart = bij[0]
    layer = next(b for b in bij if isinstance(b, ShearResponse))

    m, dmdg = m[:N], dmdg[:N]
    size = m[:, 1] / m[:, 0]
    u = size / POINT_SOURCE
    amp = 1.0 / (1.0 - u)

    # the chart's own standardised slot-1 coordinate
    z = np.asarray(jax.vmap(chart.transform)(jnp.asarray(m)))
    z1 = z[:, 1]

    # PHYSICAL size response, catalog: d ln(Mr/Mf)/dg1
    phys_cat = dmdg[:, 0, 1] / m[:, 1] - dmdg[:, 0, 0] / m[:, 0]

    # the flow's, through the same definition
    first = np.asarray(jax.vmap(dm_dg, in_axes=(None, 0, None))(
        layer, jnp.asarray(m), chart)[0])            # (n, 2, 5)
    phys_flow = first[:, 0, 1] / m[:, 1] - first[:, 0, 0] / m[:, 0]

    edges = [0.0, 2.6, 2.8, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.6926]
    print(f"POINT_SOURCE = {POINT_SOURCE}, catalog max Mr/Mf = {size.max():.4f} "
          f"(u = {u.max():.5f})")
    print(f"chart slot-1 std used by the flow = "
          f"{float(chart.std[1]):.4f}, mean = {float(chart.mean[1]):.4f}\n")
    print(f"{'Mr/Mf band':>13s} {'n':>6s} {'z1 (std)':>9s} {'1/(1-u)':>9s} "
          f"{'phys cat':>10s} {'phys flow':>10s} {'ratio':>7s} "
          f"{'z-space err':>12s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (size >= lo) & (size < hi)
        if b.sum() < 5:
            print(f"  {lo:5.3f}-{hi:5.3f} {int(b.sum()):6d}   (too few)")
            continue
        pc, pf = phys_cat[b].mean(), phys_flow[b].mean()
        # what the density/score sees: the response IN z, = amp * phys / scale
        zerr = np.mean(amp[b] * (phys_flow[b] - phys_cat[b])) / float(chart.std[1])
        print(f"  {lo:5.3f}-{hi:5.3f} {int(b.sum()):6d} {z1[b].mean():9.2f} "
              f"{amp[b].mean():9.1f} {pc:10.4f} {pf:10.4f} {pf / pc:7.3f} "
              f"{zerr:12.3f}")

    print("\nis the physical response vanishing as a point source demands?")
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (size >= lo) & (size < hi)
        if b.sum() < 5:
            continue
        print(f"  {lo:5.3f}-{hi:5.3f}  d ln(Mr/Mf)/dg1 = {phys_cat[b].mean():+.4f}"
              f"   x 1/(1-u) = {np.mean(amp[b] * phys_cat[b]):+9.2f}")


if __name__ == "__main__":
    main()
