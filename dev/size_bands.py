"""Per-BAND m1 with the eq.-(40) selection correction the bands need.

Every per-size-bin `m1` in this repo so far was computed by SORTING the flux-
windowed targets into `Mr/Mf` bins and taking each bin's ratio.  That is a
selection on the observed moments, and eq. (40) says a selection on `M` puts a
term into `SUM Q`:

    E[SUM_{i in S} Q_i] = d_g P(s|g) != 0

which is exactly the correction `bias.py --window-size` applies to a size CUT.
A bin is a size cut with two bounds, so it needs the same correction -- and
without it a bin's `m1` is not an estimate of anything, which is why
`dev/band_tmpl.py` finds BFD's own exact template sum reproducing the flow's
per-bin numbers.  This applies the correction band by band.

Bands are disjoint, so unlike `dev/size_offline.py`'s nested ceilings there is
no paired-drift construction here; each band's bootstrap stands alone.

    PQR=dev/pqr_200k.npz python dev/size_bands.py 0 3.005 3.252 3.407 3.53 9e9
"""
import os
import sys

import equinox as eqx
import fitsio
import jax
import jax.random as jr
import numpy as np
import jax.numpy as jnp

import bias
import bulk
import shear
from bias import condition, load_cov, window_mask, window_prob

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
PQR = os.environ.get("PQR", "dev/pqr_200k.npz")
BOOT = int(os.environ.get("BOOT", 200))
G = float(os.environ.get("G", 0.02))
FD = os.environ.get("FD", "0.02")
FD = None if FD.lower() == "none" else float(FD)
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
# A flux CEILING is allowed but puts the window on the bright templates, where
# `selection_terms` warns that R_s is boundary-dominated; the block bootstrap
# below is what says whether that matters.
FLUX_HI = float(os.environ.get("FLUX_HI", 1e9))
# Where eq. (40)'s terms come from -- see `prior_draw`.  `flow` is correct and
# is the default; `templates` is the diagnostic, and then `TMPL` picks the
# catalog (the 1M one is the reference with a usable error bar, see
# rs-half-split-understates-error).
TERMS = os.environ.get("TERMS", "flow")
TMPL = os.environ.get("TMPL", "")
TRAIN = os.environ.get("TRAIN", "")
FLOW = os.environ.get("FLOW", "flows/centroid_bulgedisc_v2.eqx")
# 2^23, not 2^20: `R_s` is a boundary flux, so a narrow window uses a few
# percent of the sample and at 2^20 its MC error (17% on the nominal
# window) is as large as the flow-vs-template gap.  8x the draws costs
# ~15 min and cuts the selection term on m1 from 0.0057 to 0.0024, below
# the galaxy bootstrap.  The template route has no such knob -- its error
# is the catalog.
NDRAW = int(os.environ.get("NDRAW", 1 << 23))
SEED = int(os.environ.get("SEED", 31))
# `P_s`, `Q_s`, `R_s` are SAMPLE MEANS over the template catalog -- `ghat`'s
# docstring calls them "exact analytic derivatives ... not Monte-Carlo
# estimates", which is true of each template's own `d2F/dg2` but not of their
# mean.  Their error has never been in a quoted bar, and R_s is heavy-tailed
# enough that a single template can carry 20% of it.  Splitting the catalog
# into equal blocks and bootstrapping the blocks gives the sampling
# distribution of (P_s, Q_s, R_s) with their correlations intact; re-solving
# `bias.bias` at each draw, with the GALAXY sample held fixed, turns that into
# the selection component of the error on m1.  The two components come from
# different catalogs, so the total is their quadrature sum.
NBLOCK = int(os.environ.get("NBLOCK", 20))
TBOOT = int(os.environ.get("TBOOT", 500))


def stencil_moments(draw, z, batch=16384):
    """The nine lensed moment sets of `selection_terms`' fd stencil, `(9,n,5)`.

    Window-INDEPENDENT -- only `window_prob` downstream sees the window -- so
    this is computed once and reused for every window.  That matters on the
    flow route, where each new window would otherwise recompile the whole
    bijection stack, and where ~20 accumulated CUBINs OOM this GPU.

    Non-finite rows are dropped, and dropped from ALL NINE evaluations so the
    differences stay paired -- `selection_terms`' own rule.  It bites here:
    bulgedisc_v2's `log10(Mf)` tail turns a handful of draws non-finite away
    from `g = 0`.
    """
    h = FD
    gs = [(0.0, 0.0), (h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h),
          (h, h), (h, -h), (-h, h), (-h, -h)]

    @jax.jit
    def one(idx):
        ms = jnp.stack([draw(jnp.array(g), idx) for g in gs])
        return ms, jnp.isfinite(ms).all(-1).all(0)

    # Kept on the HOST.  Boolean-indexing a (9, 1e6, 5) device array costs a
    # 5-minute XLA compile of one gather fusion; numpy does it in a blink, and
    # 380 MB moves back per window in ~0.04 s.
    out = [one(jnp.asarray(z[i:i + batch])) for i in range(0, len(z), batch)]
    ms = np.concatenate([np.asarray(o[0], np.float64) for o in out], axis=1)
    good = np.concatenate([np.asarray(o[1]) for o in out])
    if not good.all():
        print(f"  dropped {int((~good).sum())}/{len(good)} non-finite prior "
              f"draws ({(~good).mean():.1e})")
    return ms[:, good]


def per_template(ms, cov, size, flux, batch=16384):
    """`(F, dF/dg, d2F/dg2)` per prior draw, from `stencil_moments`' output --
    `selection_terms` with the chunk mean not yet taken, so any subset's mean
    is one `np.mean` away.  That is what makes the block bootstrap free."""
    h = FD

    @jax.jit
    def one(m9):
        p00, pp0, pm0, p0p, p0m, ppp, ppm, pmp, pmm = [
            window_prob(m9[k], cov, size, flux) for k in range(9)]
        r01 = (ppp - ppm - pmp + pmm) / (4.0 * h ** 2)
        q = jnp.stack([(pp0 - pm0) / (2.0 * h), (p0p - p0m) / (2.0 * h)], -1)
        r = jnp.stack([jnp.stack([(pp0 - 2.0 * p00 + pm0) / h ** 2, r01], -1),
                       jnp.stack([r01, (p0p - 2.0 * p00 + p0m) / h ** 2], -1)],
                      -1)
        return p00, q, r

    out = [one(jnp.asarray(ms[:, i:i + batch]))
           for i in range(0, ms.shape[1], batch)]
    return [np.concatenate([np.asarray(o[k], np.float64) for o in out])
            for k in range(3)]


def prior_draw():
    """`(draw, z)` for the selection terms -- the flow, or the templates.

    THE FLOW IS THE RIGHT ONE and is the default here (`bias.py`'s own
    `--window-terms` still defaults to `templates`).  Eq. (40)'s `P(s|g)` is an
    expectation under the SAME prior the target `Q`, `R` came from; taking it
    from the template catalog instead mixes two priors, and the two disagree --
    `bias.py`'s help records the flow running `P_s` ~4% high.  The template
    route stays available as the density diagnostic it is.
    """
    train = TRAIN or f"{D}/{bias.TRAIN_DATA[POP]}"
    if TERMS == "templates":
        tm, tdm, td2m = (jnp.asarray(v, jnp.float32)
                         for v in shear.load(TMPL or train))
        return (lambda g, idx: shear.lens(tm[idx], tdm[idx], td2m[idx],
                                          jnp.broadcast_to(g, (len(idx), 2))),
                np.arange(len(tm)))
    # FLOAT64, not optionally -- `R_s` here is a second derivative through the
    # whole bijection stack and float32 loses it to cancellation.
    flow = bulk.build_flow(jr.key(0), shear.load(train)[0], shear=True,
                           centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    sx = jnp.asarray(fitsio.read(f"{D}/{bias.CATALOGS[POP]['zero']}.fits",
                                 rows=[0])["cov_odd"][0], dtype=jnp.float64)
    z = flow.base_dist.sample(jr.key(SEED), (NDRAW,)).astype(jnp.float64)
    return (lambda g, zz: jax.vmap(
        lambda z1: flow.bijection.transform(z1, condition(g, sx)))(zz), z)


def selection_error(blocks, sp, sm, qp, rp, qm, rm, seed=0):
    """sd of m1 induced by the template-catalog error in (P_s, Q_s, R_s)."""
    rng = np.random.default_rng(seed)
    n_p, n_m = int((~sp).sum()), int((~sm).sum())
    out = np.empty(TBOOT)
    for b in range(TBOOT):
        k = rng.integers(0, len(blocks), len(blocks))
        ps, qs, rs = (np.mean([blocks[i][j] for i in k], axis=0)
                      for j in range(3))
        out[b] = bias.bias(qp, rp, qm, rm, G, sel=(sp, sm),
                           ns=((n_p, ps, qs, rs), (n_m, ps, qs, rs)))[0]
    return out.std(ddof=1)


if __name__ == "__main__":
    edges = [float(x) for x in sys.argv[1:]] or [0, 3.005, 3.252, 3.407,
                                                 3.53, 9e9]
    d = np.load(PQR)
    ok = np.ones(len(d["plus_q"]), dtype=bool)
    for k in ("plus", "minus"):
        ok &= np.isfinite(d[f"{k}_q"]).all(1)
        ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    qp, rp = d["plus_q"][ok], d["plus_r"][ok]
    qm, rm = d["minus_q"][ok], d["minus_r"][ok]
    obs_p, obs_m = d["obs_plus"][ok], d["obs_minus"][ok]
    cov = load_cov(f"{D}/{bias.CATALOGS[POP]['zero']}.fits")
    draw, z = prior_draw()
    flux = (FLUX_LO, FLUX_HI)
    ms = stencil_moments(draw, z)

    print(f"{len(qp)} targets, flux in ({FLUX_LO:g}, {FLUX_HI:g}), fd = {FD}, "
          f"selection terms from the {TERMS} ({len(z)} draws)\n")
    print(f"blocks: {NBLOCK} x {len(z) // NBLOCK}, {TBOOT} draws\n")
    print(f"{'Mr/Mf band':>16s}{'kept':>8s}{'R_s11':>9s}{'sd(R_s11)':>11s}"
          f"{'uncorrected':>21s}{'corrected':>34s}")
    for lo, hi in zip(edges, edges[1:]):
        size = (lo, hi)
        sp, sm = window_mask(obs_p, size, flux), window_mask(obs_m, size, flux)
        pt = per_template(ms, cov, size, flux)
        # `selection_terms` is not called: it is exactly these means, and on
        # the flow route it would double the cost.  Checked equal to 6e-5 of
        # R_s11 when it was still being run alongside.
        ps, qs, rs = (v.mean(0) for v in pt)
        blocks = [tuple(v[k].mean(0) for v in pt)
                  for k in np.array_split(np.arange(len(pt[0])), NBLOCK)]
        ns = ((int((~sp).sum()), ps, qs, rs), (int((~sm).sum()), ps, qs, rs))
        u = bias.bias(qp, rp, qm, rm, G, sel=(sp, sm))
        du = bias.bootstrap(qp, rp, qm, rm, G, BOOT, 0, sel=(sp, sm))
        c = bias.bias(qp, rp, qm, rm, G, sel=(sp, sm), ns=ns)
        dc = bias.bootstrap(qp, rp, qm, rm, G, BOOT, 0, sel=(sp, sm), ns=ns)
        ds = selection_error(blocks, sp, sm, qp, rp, qm, rm)
        rs_sd = (np.std([b[2][0, 0] for b in blocks], ddof=1)
                 / np.sqrt(len(blocks)))
        print(f"{lo:7.3f}-{hi:<8.3f}{sp.mean():8.1%}{rs[0, 0]:9.3f}{rs_sd:11.4f}"
              f"{u[0]:+12.5f} +/- {du[0]:.5f}"
              f"{c[0]:+12.5f} +/- {dc[0]:.5f} +/- {ds:.5f}"
              f"  ({np.hypot(dc[0], ds):.5f})", flush=True)
    print("\ncorrected bar is  +/- galaxy bootstrap  +/- selection MC  "
          "(total in brackets)")
