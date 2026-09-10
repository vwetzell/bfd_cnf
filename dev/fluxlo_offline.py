"""Sweep the selection window offline, from ONE saved run.

Per-target Q and R do not depend on the window: the window only chooses which
targets enter the eq. (45)-(46) sums and adds the selection terms, and those
come from the TEMPLATES (`--window-terms templates`), never from the flow.  So a
flux_lo scan does not need one 20k `bias.py` run per point -- it needs one run
with `--save-pqr` and this.

`save_pqr` already writes each arm's observed moments for exactly this purpose.
The only work per window is `selection_terms` over the template catalog, which
is seconds, against ~an hour for a 20k integration.
"""
import os, sys
import numpy as np, jax.numpy as jnp

import bias, shear
from bias import window_mask, selection_terms, load_cov

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
PQR = os.environ.get("PQR", "dev/pqr_auto_20k.npz")
BOOT = int(os.environ.get("BOOT", 200))
G = float(os.environ.get("G", 0.02))


def arms(d):
    """(qp, rp, qm, rm) with the non-finite targets dropped from BOTH arms."""
    ok = np.ones(len(d["plus_q"]), dtype=bool)
    for k in ("plus", "minus"):
        ok &= np.isfinite(d[f"{k}_q"]).all(1)
        ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    return ok, (d["plus_q"][ok], d["plus_r"][ok],
                d["minus_q"][ok], d["minus_r"][ok])


if __name__ == "__main__":
    los = [float(x) for x in sys.argv[1:]] or [1200, 1600, 2000, 2500, 3000, 4000]
    d = np.load(PQR)
    ok, (qp, rp, qm, rm) = arms(d)
    obs_p, obs_m = d["obs_plus"][ok], d["obs_minus"][ok]
    cov = load_cov(f"{D}/{bias.CATALOGS[POP]['zero']}.fits")
    tm, tdm, td2m = (jnp.asarray(v, jnp.float32)
                     for v in shear.load(f"{D}/{bias.TRAIN_DATA[POP]}"))
    draw = lambda g, idx: shear.lens(tm[idx], tdm[idx], td2m[idx],
                                     jnp.broadcast_to(g, (len(idx), 2)))
    z = np.arange(len(tm))

    m1u, du = bias.bias(qp, rp, qm, rm, G), bias.bootstrap(qp, rp, qm, rm, G, BOOT, 0)
    print(f"{len(qp)} targets; unwindowed m1 = {m1u[0]:+.5f} +/- {du[0]:.5f}\n")
    print(f"{'flux_lo':>9s}{'kept':>8s}{'P_s':>8s}{'uncorrected':>22s}"
          f"{'corrected':>22s}")
    size = (-np.inf, np.inf)
    for lo in los:
        flux = (lo, 1e9)
        sp, sm = window_mask(obs_p, size, flux), window_mask(obs_m, size, flux)
        ps, qs, rs, _ = selection_terms(draw, z, cov, size, flux)
        ns_p, ns_m = (int((~sp).sum()), ps, qs, rs), (int((~sm).sum()), ps, qs, rs)
        u = bias.bias(qp, rp, qm, rm, G, sel=(sp, sm))
        du_ = bias.bootstrap(qp, rp, qm, rm, G, BOOT, 0, sel=(sp, sm))
        c = bias.bias(qp, rp, qm, rm, G, sel=(sp, sm), ns=(ns_p, ns_m))
        dc = bias.bootstrap(qp, rp, qm, rm, G, BOOT, 0, sel=(sp, sm), ns=(ns_p, ns_m))
        print(f"{lo:9.0f}{sp.mean():8.1%}{ps:8.4f}"
              f"{u[0]:+13.5f} +/- {du_[0]:.5f}"
              f"{c[0]:+13.5f} +/- {dc[0]:.5f}", flush=True)

    # The windows are NESTED subsets of one catalog, so the per-point bootstrap
    # errors above are heavily correlated and their quadrature sum is the wrong
    # yardstick for the drift.  Resample galaxies ONCE per draw and evaluate
    # every flux_lo on that same resample -- the same pairing `bias.compare`
    # uses between two runs, applied between two windows.
    print(f"\npaired drift vs flux_lo = {los[0]:.0f} "
          f"(same galaxy resample for every window)")
    rng = np.random.default_rng(0)
    n = len(qp)
    masks, sels = {}, {}
    for lo in los:
        flux = (lo, 1e9)
        sp, sm = window_mask(obs_p, size, flux), window_mask(obs_m, size, flux)
        ps, qs, rs, _ = selection_terms(draw, z, cov, size, flux)
        masks[lo], sels[lo] = (sp, sm), (ps, qs, rs)
    draws = {lo: [] for lo in los}
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        for lo in los:
            sp, sm = masks[lo][0][idx], masks[lo][1][idx]
            ps, qs, rs = sels[lo]
            draws[lo].append(bias.bias(
                qp[idx], rp[idx], qm[idx], rm[idx], G, sel=(sp, sm),
                ns=((int((~sp).sum()), ps, qs, rs),
                    (int((~sm).sum()), ps, qs, rs)))[0])
    ref = np.array(draws[los[0]])
    for lo in los:
        d_ = np.array(draws[lo]) - ref
        print(f"{lo:9.0f}   dm1 = {d_.mean():+.5f} +/- {d_.std(ddof=1):.5f}")
