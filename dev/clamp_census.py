"""Does the centroid layer's `_EXP_MAX` tanh clamp ever actually bite?

`marginalize`/`unmarginalize` both end with

    z + _EXP_MAX * tanh((z' - z) / _EXP_MAX)

applied COMPONENTWISE, which includes the spin-2 pair `(z3, z4)`.  A
componentwise tanh does not commute with rotating that pair, so it is the only
non-equivariant operation left in the layer -- and therefore the prime suspect
for the residual `Q_s_sym ~ 3e-6` that `dev/isotropy_check.py` measures.

Whether that matters is an empirical question: if `|dz|` never approaches
`_EXP_MAX` the tanh is the identity to machine precision and the
non-equivariance is latent, not active.  Measure before fixing.

Reported as PERCENTILES, never the max: the flow's tail has a handful of
ill-conditioned draws that carry no window weight.

Sigma_X is scanned as well as taken at nominal, because the clamp has to be
clear at the WIDEST Sigma_X in play, not just the one the targets happen to
have.

    python -u dev/clamp_census.py --flow flows/centroid_v10.eqx
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import bulk
import shear
from models.centroid import _EXP_MAX, raw_from_standard, standard_from_raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_v10.eqx")
    p.add_argument("--data",
                   default="../bfd_cnf_imsims/data/moments_bulgedisc_v3.fits")
    p.add_argument("--targets",
                   default="../bfd_cnf_imsims/data/targets_v3_g0_200k.fits")
    p.add_argument("--scale", type=float, nargs="*", default=[1.0, 1.4, 2.0],
                   help="Sigma_X multipliers; 1.4 is copies.SIGMA_RANGE")
    p.add_argument("--split", type=float, nargs="*", default=[0.0, 0.2, 0.4],
                   help="spin-2 split d, lam+- = s sigma0^2 (1 +- d/2)")
    p.add_argument("--n", type=int, default=20000)
    a = p.parse_args()

    import fitsio
    m_train = shear.load(a.data)[0]
    flow = bulk.build_flow(jax.random.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    layer = _find_layer(flow)

    m = jnp.asarray(np.asarray(fitsio.read(a.data)["moments"],
                               dtype=np.float64)[:a.n])
    sx0 = np.asarray(fitsio.read(a.targets)["cov_odd"], dtype=np.float64)[0]
    mean, std = _unwrap(layer)
    z = jax.vmap(standard_from_raw, (0, None, None))(m, mean, std)

    print(f"{a.flow}: _EXP_MAX = {_EXP_MAX}, {len(z)} prior galaxies")
    print(f"nominal Sigma_X = {sx0}\n")
    print(f"{'scale':>6} {'split':>6} | "
          f"{'p50':>9} {'p99':>9} {'p99.9':>9} {'max':>9}  (|dz| spin-2)"
          f"   frac |dz|_inf > 0.1*EXP_MAX")

    for s in a.scale:
        for d in a.split:
            sx = jnp.asarray([sx0[0] * s * (1 + d / 2), sx0[1] * s,
                              sx0[2] * s * (1 - d / 2)])
            cond = jnp.concatenate([jnp.zeros(2), sx])
            dz = jax.vmap(_raw_delta, (None, 0, None))(layer, z, cond)
            r2 = np.asarray(jnp.hypot(dz[:, 3], dz[:, 4]))
            inf = np.asarray(jnp.max(jnp.abs(dz), axis=1))
            q = np.percentile(r2[np.isfinite(r2)], [50, 99, 99.9])
            print(f"{s:6.2f} {d:6.2f} | {q[0]:9.2e} {q[1]:9.2e} {q[2]:9.2e} "
                  f"{np.nanmax(r2):9.2e}   "
                  f"{np.mean(inf > 0.1 * _EXP_MAX):.3e}")

    print(f"\nThe tanh is the identity to ~(dz/_EXP_MAX)^2/3.  If p99.9 sits\n"
          f"well under {_EXP_MAX:g} the componentwise clamp never engages and its\n"
          f"non-equivariance is latent; leave it.")


def _raw_delta(layer, z, cond):
    """`x - y` BEFORE the tanh -- the quantity the clamp acts on."""
    mean, std = _unwrap(layer)
    m = raw_from_standard(z, mean, std, layer.flux_sas)
    from models.centroid import _transport, split_condition
    _, sigma_x = split_condition(cond)
    x = standard_from_raw(_transport(m, sigma_x, 1.0,
                                     *layer.bracket_coeffs(m, mean, std)),
                          mean, std, layer.flux_sas)
    return x - z


def _unwrap(layer):
    # `chart()`, not the stored constants: the layer's map is defined against
    # the symmetrised spin-2 pair (see `CentroidMarginalize.chart`).
    return layer.chart()


def _find_layer(flow):
    from models.centroid import CentroidMarginalize
    out = [x for x in jax.tree_util.tree_leaves(
        flow, is_leaf=lambda x: isinstance(x, CentroidMarginalize))
        if isinstance(x, CentroidMarginalize)]
    assert len(out) == 1, f"{len(out)} centroid layers found"
    return out[0]


if __name__ == "__main__":
    main()
