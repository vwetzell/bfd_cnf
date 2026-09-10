"""Offline size-window scan, from one saved --save-pqr run.

Same trick as `fluxlo_offline.py`: per-target Q and R do not depend on the
window, so a size sweep costs one `selection_terms` call per point instead of a
full integration.  Here the flux floor is held fixed and the Mr/Mf CEILING is
walked down, because that is where the residual m1 concentrates.

`--window-fd`'s central differences are used for the selection terms (`fd`),
the route that made size bounds work.
"""
import os, sys
import numpy as np, jax.numpy as jnp

import equinox as eqx, fitsio, jax, jax.random as jr

import bias, bulk, shear
from bias import condition, window_mask, selection_terms, load_cov

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
PQR = os.environ.get("PQR", "dev/pqr_200k.npz")
BOOT = int(os.environ.get("BOOT", 200))
G = float(os.environ.get("G", 0.02))
FD = os.environ.get("FD", "0.02")
FD = None if FD.lower() == "none" else float(FD)
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
TERMS = os.environ.get("TERMS", "flow")          # flow | templates
FLOW = os.environ.get("FLOW", "flows/centroid_bulgedisc_v2.eqx")
NDRAW = int(os.environ.get("NDRAW", 262144))


def prior_draw():
    """`(draw, z)` for `selection_terms` -- the same two routes bias.py's
    `--window-terms` offers, rebuilt here so the sweep needs no rerun."""
    train = f"{D}/{bias.TRAIN_DATA[POP]}"
    if TERMS == "templates":
        tm, tdm, td2m = (jnp.asarray(v, jnp.float32) for v in shear.load(train))
        return (lambda g, idx: shear.lens(tm[idx], tdm[idx], td2m[idx],
                                          jnp.broadcast_to(g, (len(idx), 2))),
                np.arange(len(tm)))
    # FLOAT64, not optionally -- see bias.py's own note: R_s here is a
    # forward-over-forward second derivative through the whole bijection stack
    # and float32 loses it entirely to cancellation.
    flow = bulk.build_flow(jr.key(0), shear.load(train)[0], shear=True,
                           centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    sx = jnp.asarray(fitsio.read(f"{D}/{bias.CATALOGS[POP]['zero']}.fits",
                                 rows=[0])["cov_odd"][0], dtype=jnp.float64)
    return (lambda g, zz: jax.vmap(
                lambda z1: flow.bijection.transform(z1, condition(g, sx)))(zz),
            flow.base_dist.sample(jr.key(31), (NDRAW,)).astype(jnp.float64))

if __name__ == "__main__":
    his = [float(x) for x in sys.argv[1:]] or [9e9, 3.9, 3.6, 3.45, 3.3, 3.2, 3.0]
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
    flux = (FLUX_LO, 1e9)

    print(f"{len(qp)} targets, flux window ({FLUX_LO:g}, 1e9), fd = {FD}, "
          f"selection terms from the {TERMS} ({len(z)} draws)")
    print(f"\n{'Mr/Mf hi':>10s}{'kept':>8s}{'P_s':>8s}{'R_s11':>10s}"
          f"{'uncorrected':>22s}{'corrected':>22s}")
    for hi in his:
        size = (-np.inf, hi)
        sp, sm = window_mask(obs_p, size, flux), window_mask(obs_m, size, flux)
        ps, qs, rs, _ = selection_terms(draw, z, cov, size, flux, fd=FD)
        ns = ((int((~sp).sum()), ps, qs, rs), (int((~sm).sum()), ps, qs, rs))
        u = bias.bias(qp, rp, qm, rm, G, sel=(sp, sm))
        du = bias.bootstrap(qp, rp, qm, rm, G, BOOT, 0, sel=(sp, sm))
        c = bias.bias(qp, rp, qm, rm, G, sel=(sp, sm), ns=ns)
        dc = bias.bootstrap(qp, rp, qm, rm, G, BOOT, 0, sel=(sp, sm), ns=ns)
        print(f"{hi:10.2f}{sp.mean():8.1%}{ps:8.4f}{rs[0, 0]:10.3f}"
              f"{u[0]:+13.5f} +/- {du[0]:.5f}"
              f"{c[0]:+13.5f} +/- {dc[0]:.5f}", flush=True)

    # R_s is a sample mean over the templates and the WARNING says one of them
    # can own it.  Splitting the catalog in half gives two independent draws of
    # that mean: if the halves agree the concentration is harmless here.
    print(f"\n{'Mr/Mf hi':>10s}{'R_s11 (half A)':>16s}{'(half B)':>12s}"
          f"{'spread':>10s}")
    half = len(z) // 2
    for hi in his:
        size = (-np.inf, hi)
        ra = selection_terms(draw, z[:half], cov, size, flux, fd=FD)[2][0, 0]
        rb = selection_terms(draw, z[half:], cov, size, flux, fd=FD)[2][0, 0]
        print(f"{hi:10.2f}{ra:16.3f}{rb:12.3f}"
              f"{abs(ra - rb) / max(abs(ra + rb) / 2, 1e-12):10.1%}", flush=True)

    # The windows are nested subsets of ONE catalog, so their unpaired errors
    # above are heavily correlated; resample galaxies once and evaluate every
    # window on that same resample to get the drift's own error.
    print(f"\npaired drift in corrected m1 vs no size cut")
    rng = np.random.default_rng(0)
    n = len(qp)
    cache = {}
    for hi in his:
        size = (-np.inf, hi)
        sp, sm = window_mask(obs_p, size, flux), window_mask(obs_m, size, flux)
        ps, qs, rs, _ = selection_terms(draw, z, cov, size, flux, fd=FD)
        cache[hi] = (sp, sm, ps, qs, rs)
    draws = {hi: [] for hi in his}
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        for hi in his:
            sp, sm, ps, qs, rs = cache[hi]
            sp, sm = sp[idx], sm[idx]
            draws[hi].append(bias.bias(
                qp[idx], rp[idx], qm[idx], rm[idx], G, sel=(sp, sm),
                ns=((int((~sp).sum()), ps, qs, rs),
                    (int((~sm).sum()), ps, qs, rs)))[0])
    ref = np.array(draws[his[0]])
    for hi in his:
        dd = np.array(draws[hi]) - ref
        print(f"{hi:10.2f}   dm1 = {dd.mean():+.5f} +/- {dd.std(ddof=1):.5f}")
