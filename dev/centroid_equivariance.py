"""How rotation-equivariant is the centroid layer, and which term breaks it?

A sky rotation by `t` rotates the moments' spin-2 pair `(M1, M2)` by `2t` and
`Sigma_X` by `t`, leaving `(Mf, Mr, Mc)` alone.  If the layer is equivariant,
`marginalize(rot z, rot Sigma_X)` equals `rot marginalize(z, Sigma_X)` exactly,
and `Q_s = 0` follows -- which `dev/isotropy_check.py` found only to `~3e-6`.

`t = 45 deg` is `(M1, M2) -> (-M2, M1)` and `Sigma_X -> [C11, -C01, C00]`, both
exact in floating point, so the residual measured here is the layer's, not the
rotation's.

There were exactly two candidates and this script separated them:

  * the `_EXP_MAX` tanh, applied COMPONENTWISE to `(z3, z4)` -- MEASURED INERT.
    Removing it changes the residual by nothing, because the spin-2 `|dz|` p99
    is 0.098 against `_EXP_MAX = 0.5` (`dev/clamp_census.py`).
  * the layer's FROZEN chart constants, read RAW where the chart itself reads
    `RawMomentStandardize._effective()` -- THE WHOLE LEAK.  Fixed by
    `CentroidMarginalize.chart()`; see its docstring.

Kept as the regression check on that fix, so the variants below reconstruct the
pre-fix code path explicitly rather than by editing the constants (`chart()`
would just re-symmetrise them).

Percentiles, never the max -- the tail has a few ill-conditioned draws.

    python -u dev/centroid_equivariance.py --flow flows/centroid_v11.eqx
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import numpy as np
from paramax import unwrap

import bulk
import shear
import models.centroid as C
from models.centroid import CentroidMarginalize


def rot45_z(z):
    """(z3, z4) -> (-z4, z3): a 45 deg sky rotation in standardised chart
    coordinates.  An exact rotation only because `chart()` forces the two slots
    to share a scale and carry zero mean -- which is the property under test."""
    return z.at[3].set(-z[4]).at[4].set(z[3])


def rot45_sigma_x(sx):
    """[C00, C01, C11] -> [C11, -C01, C00], a 45 deg rotation of Sigma_X."""
    return jnp.stack([sx[2], -sx[1], sx[0]])


def _variant(layer, raw_chart=False, clamp=True):
    """The layer with the pre-fix code path selectively restored."""
    chart = ((lambda s: (unwrap(s.mean), unwrap(s.std))) if raw_chart
             else type(layer).chart)

    class _V(type(layer)):
        def marginalize(self, y, condition):
            mean, std = chart(self)
            _, sx = C.split_condition(condition)
            m = C.raw_from_standard(y, mean, std, self.flux_sas)
            x = C.standard_from_raw(
                C._transport(m, sx, 1.0, *self.bracket_coeffs(m, mean, std)),
                mean, std, self.flux_sas)
            if not clamp:
                return x
            return y + C._EXP_MAX * jnp.tanh((x - y) / C._EXP_MAX)

    out = object.__new__(_V)
    for f in layer.__dataclass_fields__:
        object.__setattr__(out, f, getattr(layer, f))
    return out


def residual(layer, z, sx):
    """|marginalize(R z, R Sx) - R marginalize(z, Sx)|, spin-2 part."""
    def one(zi):
        c = jnp.concatenate([jnp.zeros(2), sx])
        cr = jnp.concatenate([jnp.zeros(2), rot45_sigma_x(sx)])
        a = rot45_z(layer.marginalize(zi, c))
        b = layer.marginalize(rot45_z(zi), cr)
        return jnp.hypot(b[3] - a[3], b[4] - a[4])
    return jax.vmap(one)(z)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_v10.eqx")
    p.add_argument("--data",
                   default="../bfd_cnf_imsims/data/moments_bulgedisc_v3.fits")
    p.add_argument("--targets",
                   default="../bfd_cnf_imsims/data/targets_v3_g0_200k.fits")
    p.add_argument("--n", type=int, default=20000)
    a = p.parse_args()

    m_train = shear.load(a.data)[0]
    flow = bulk.build_flow(jax.random.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    layer = next(b for b in flow.bijection.bijection.bijections
                 if isinstance(b, CentroidMarginalize))

    raw_mean, raw_std = unwrap(layer.mean), unwrap(layer.std)
    mean, std = layer.chart()
    m = jnp.asarray(np.asarray(fitsio.read(a.data)["moments"],
                               dtype=np.float64)[:a.n])
    z = jax.vmap(C.standard_from_raw, (0, None, None))(m, mean, std)
    sx = jnp.asarray(np.asarray(fitsio.read(a.targets)["cov_odd"],
                                dtype=np.float64)[0])

    print(f"{a.flow}, {len(z)} prior galaxies, 45 deg rotation")
    print(f"  stored chart spin-2: mean {np.asarray(raw_mean)[3:]}, "
          f"std ratio {float(raw_std[3] / raw_std[4]):.6f}")
    print(f"  chart() spin-2:      mean {np.asarray(mean)[3:]}, "
          f"std ratio {float(std[3] / std[4]):.6f}\n")
    print(f"{'variant':>30} | {'p50':>9} {'p90':>9} {'p99':>9} {'p99.9':>9}"
          "   (|d z_spin2|)")

    for name, kw in [("as built (chart(), clamped)", {}),
                     ("chart(), no _EXP_MAX clamp", dict(clamp=False)),
                     ("PRE-FIX raw chart, clamped", dict(raw_chart=True)),
                     ("PRE-FIX raw chart, no clamp",
                      dict(raw_chart=True, clamp=False))]:
        r = np.asarray(residual(_variant(layer, **kw), z, sx))
        r = r[np.isfinite(r)]
        q = np.percentile(r, [50, 90, 99, 99.9])
        print(f"{name:>30} | {q[0]:9.2e} {q[1]:9.2e} {q[2]:9.2e} {q[3]:9.2e}")

    print("\nThe two chart() rows should sit at the float32 floor (~1e-6) and\n"
          "be identical to each other -- that is the fix holding AND the clamp\n"
          "being inert.  The two PRE-FIX rows show what was being measured\n"
          "before, and should also be identical to each other.")


if __name__ == "__main__":
    main()
