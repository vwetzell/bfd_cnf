"""Is the flow's boundary error a DENSITY error or a RESPONSE error?

`R_s` is a flux across the window boundary: it is the density of galaxies ON
the shell times their shear response there.  The flow's `R_s11` at an
`Mr/Mf = 3.0` ceiling is 0.338 +/- 0.007 against the templates' 0.267 +/- 0.023
(~3 sigma), and the two ingredients call for opposite fixes:

  * DENSITY wrong  -> the fitted prior misplaces mass near the boundary; that
    is a training problem (coverage, weighting, capacity where it matters).
  * RESPONSE wrong -> `d(Mr/Mf)/dg` from the shear layer disagrees with bfd's
    exact `dm/dg`; that is the shear layer, and nothing about the bulk density
    or the training set would touch it.

Compares both against the template catalog, which carries bfd's exact `dm/dg`
and IS the true population, as a function of `Mr/Mf` -- so the answer comes
with the location of the damage, not just its size.

Both restricted to the flux window, since that is the context every `R_s` here
was computed in.
"""
import os
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
import fitsio

import bias, bulk, shear
from bias import condition

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
FLOW = os.environ.get("FLOW", "flows/centroid_bulgedisc_v2.eqx")
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
NDRAW = int(os.environ.get("NDRAW", 262144))
SEED = int(os.environ.get("SEED", 31))
# Chart constants come from the TRAINING catalog; a mismatch shows up as a
# gross density disagreement below, so the table is its own load check.
TRAIN = os.environ.get("TRAIN", None)
# Shear-stage checkpoints have no centroid layer; the response spread is a
# shear-layer property, so a steps comparison is cleanest evaluated there.
CENTROID = os.environ.get("CENTROID", "1") != "0"
H = float(os.environ.get("H", 0.01))
EDGES = np.array([float(x) for x in os.environ.get(
    "EDGES", "2.4 2.7 2.9 3.0 3.1 3.2 3.3 3.45 3.6 3.8 4.0").split()])
BATCH = 16384


def ratio_and_response(m, dm):
    """`(Mr/Mf, d(Mr/Mf)/dg1)` from moments and their shear derivative."""
    v = m[:, 1] / m[:, 0]
    return v, (dm[:, 1] * m[:, 0] - m[:, 1] * dm[:, 0]) / m[:, 0] ** 2


if __name__ == "__main__":
    train = TRAIN or f"{D}/{bias.TRAIN_DATA[POP]}"
    tm, tdm, _ = (np.asarray(v, np.float64) for v in shear.load(train))

    flow = bulk.build_flow(jr.key(0), shear.load(train)[0], shear=True,
                           centroid=CENTROID)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    sx = (jnp.asarray(fitsio.read(f"{D}/{bias.CATALOGS[POP]['zero']}.fits",
                                  rows=[0])["cov_odd"][0], dtype=jnp.float64)
          if CENTROID else None)
    z = flow.base_dist.sample(jr.key(SEED), (NDRAW,)).astype(jnp.float64)

    @eqx.filter_jit
    def draw(g, zz):
        return jax.vmap(lambda z1: flow.bijection.transform(
            z1, condition(jnp.asarray(g, jnp.float64), sx)))(zz)

    def moments(g):
        return np.concatenate([np.asarray(draw(jnp.array([g, 0.0]),
                                               z[i:i + BATCH]))
                               for i in range(0, NDRAW, BATCH)])

    # Central difference for the flow's dm/dg1 -- the same bounded construction
    # the selection terms use, rather than autodiff whose per-draw tail is what
    # made R_s unusable in the first place.
    m0, mp, mm = moments(0.0), moments(H), moments(-H)
    fdm = (mp - mm) / (2 * H)
    ok = np.isfinite(m0).all(1) & np.isfinite(fdm).all(1)
    fv, fr = ratio_and_response(m0[ok], fdm[ok])
    ffl = m0[ok, 0]
    # tdm is (N, 2, 5): [galaxy, shear component, moment].  Slot 0 is dm/dg1,
    # matching the flow's central difference along g1.
    tv, tr = ratio_and_response(tm, tdm[:, 0, :])

    fsel = ffl >= FLUX_LO
    tsel = tm[:, 0] >= FLUX_LO
    fv, fr = fv[fsel], fr[fsel]
    tv, tr = tv[tsel], tr[tsel]
    print(f"flow: {fsel.sum()} draws, templates: {tsel.sum()} rows "
          f"(Mf >= {FLUX_LO:g}); flow dm/dg by central difference at h = {H}")

    print(f"\n{'Mr/Mf bin':>14s}{'flow frac':>11s}{'tmpl frac':>11s}"
          f"{'ratio':>8s}{'flow <dv/dg>':>14s}{'tmpl <dv/dg>':>14s}{'ratio':>8s}")
    for i in range(len(EDGES) - 1):
        lo, hi = EDGES[i], EDGES[i + 1]
        fm = (fv >= lo) & (fv < hi)
        tmask = (tv >= lo) & (tv < hi)
        ff, tf = fm.mean(), tmask.mean()
        fresp = fr[fm].mean() if fm.any() else np.nan
        tresp = tr[tmask].mean() if tmask.any() else np.nan
        print(f"  [{lo:.2f}, {hi:.2f})" + f"{ff:11.4f}{tf:11.4f}"
              f"{ff / tf if tf else np.nan:8.3f}"
              f"{fresp:14.4f}{tresp:14.4f}"
              f"{fresp / tresp if tresp else np.nan:8.3f}")

    # The shell that matters for a 3.0 ceiling: F's derivatives live within a
    # noise sigma of the boundary, so this is the population R_s actually sees.
    for c in (3.0, 3.2, 3.3, 3.45):
        w = 0.1
        fm, tmask = (np.abs(fv - c) < w), (np.abs(tv - c) < w)
        print(f"\nshell |Mr/Mf - {c}| < {w}: "
              f"density flow/tmpl = {fm.mean() / tmask.mean():.3f}, "
              f"<dv/dg> flow {fr[fm].mean():+.4f} vs tmpl {tr[tmask].mean():+.4f} "
              f"({fr[fm].mean() / tr[tmask].mean():.3f}), "
              f"sd flow {fr[fm].std():.4f} vs tmpl {tr[tmask].std():.4f}")
