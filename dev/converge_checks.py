"""Convergence study for the S/N-8 bulgedisc_g2n chain (dev/converge_sn8.sh).

`d2layer FLOW...`: each centroid layer's own d2(shift)/dg1^2 over flux > 3000
(no window) against the sims' recentring and the copies' lensed target.

`pair TAG...`: windowed corrected/uncorrected m1, c1, c2 of each
pqr/{TAG}.npz (selection terms parsed from logs/cvg/bias_{TAG}.log, NPOP
calibrated to reproduce the log's corrected m1), and every run's m1 minus
the FIRST tag's, paired: same targets, same bootstrap resamples.  Then the
seed mean and seed sd (per step multiple if tags end in _1x / _2x).  LOGS
sets the bias-log directory (default logs/cvg).

usage: python dev/converge_checks.py d2layer flows/a.eqx ...
       python dev/converge_checks.py pair cv_s0_1x cv_s1_1x ...
"""
import os
import re
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "dev")
import numpy as np

SIZE, FLUX, G = (2.2, 3.2), (3000.0, 20000.0), 0.02
D = "../bfd_cnf_imsims/data"


def d2layer(paths, h=0.02, n_cop=120000, n_sims=120000):
    """The centroid layer's own d2(shift)/dg1^2, ratio of sums over flux > 3000
    (NO window: selecting on noisy moments inflates the sims' recentring d2 from
    +0.002 to +0.044 on Mf, and flow draws carry no per-draw noise, so a window
    mean of flow d2m is not an observable -- [[centroid-loss-does-not-pin-second-order]]).
    References: the sims' recentring (paired-arm d2M minus the fixed-centroid
    d2m_dg2 column) and the copies' (d2 of the sheared-weight mean of the
    copies' LENSED moments, minus the galaxy's own d2m_dg2)."""
    import jax
    import jax.numpy as jnp
    import fitsio
    import bias as B
    import centroid as C
    import closed_loop as CL
    from imsims.copies import log_weights
    cols = [0, 1, 4]
    fmt = lambda v: "  ".join(f"{x:+.4f}" for x in np.asarray(v)[cols])
    print(f"{'d2(recentring)/dg1^2, sum/sum, flux>3000':44s} Mf       Mr       Mc")
    c = B.CATALOGS["bulgedisc_g2n_176k_sn8r"]
    Z = fitsio.read(f"{D}/{c['zero']}.fits", columns=["moments", "d2m_dg2", "cov_odd"])
    Pm, Mm = (fitsio.read(f"{D}/{c[a]}.fits", columns=["moments"])["moments"] for a in ("plus", "minus"))
    m0 = Z["moments"].astype(float)
    w = np.where(m0[:, 0] > FLUX[0])[0][:n_sims]
    sims = (Pm + Mm - 2 * m0) / G**2 - Z["d2m_dg2"][:, 0]
    print(f"{'sims (paired arms - fixed-centroid column)':44s} {fmt(sims[w].sum(0) / m0[w].sum(0))}")
    copies, gal = C.load_copies(f"{D}/copies_bulgedisc_g2_bdg2n_sn8.fits")
    cc = copies[copies["gal"] < n_cop]; sxc = gal["cov_odd"][0]

    def M(g):
        g = np.asarray(g, float); quad = np.array([.5 * g[0]**2, g[0] * g[1], .5 * g[1]**2])
        wt = np.exp(log_weights(cc, sxc, g=g if g.any() else None))
        mc = (cc["moments"].astype(np.float64) + np.einsum("i,nim->nm", g, cc["dm_dg"])
              + np.einsum("i,nim->nm", quad, cc["d2m_dg2"]))
        den = np.bincount(cc["gal"], wt, n_cop)
        return np.stack([np.bincount(cc["gal"], wt * mc[:, j], n_cop) for j in range(5)], 1) / den[:, None]
    M0 = M((0, 0)); sel = M0[:, 0] > FLUX[0]
    t = (M((h, 0)) + M((-h, 0)) - 2 * M0) / h**2 - np.asarray(gal["d2m_dg2"][:n_cop, 0], np.float64)
    print(f"{'copies (lensed moments, sheared weights)':44s} {fmt(t[sel].sum(0) / M0[sel].sum(0))}")
    sx = jnp.asarray(Z["cov_odd"][0], jnp.float32)
    for p in paths:
        fl = CL.load_flow(p); chart, layer = C._chart(fl), C._sigmax_layer(fl)

        def d2(m):
            z = chart.transform(m)
            f = lambda s: chart.inverse(layer.inverse_and_log_det(z, jnp.concatenate([jnp.array([s, 0.]), sx]))[0])
            d = lambda s: jax.jvp(f, (s,), (1.0,))[1]
            return jax.jvp(d, (0.,), (1.,))[1]
        f = jax.jit(jax.vmap(d2))
        L = np.concatenate([np.asarray(f(jnp.asarray(m0[w[i:i + 10000]], jnp.float32)))
                            for i in range(0, len(w), 10000)])
        L = np.where(np.isfinite(L), L, 0.)
        print(f"{os.path.basename(p):44s} {fmt(L.sum(0) / m0[w].sum(0))}", flush=True)


def pair(tags, nboot=200, logs=os.environ.get("LOGS", "logs/cvg")):
    import fitsio
    from scipy.optimize import brentq
    import bias as B
    src = open("dev/varobs_pool.py").read().split("Bs = [load")[0]
    ns = {}
    exec(src, ns)
    sel_terms, est = ns["sel_terms"], ns["est"]
    pop = "bulgedisc_g2n_176k_varobs_sn8"
    npop0 = fitsio.read_header(f"{D}/{B.CATALOGS[pop]['plus']}.fits", 1)["NPOP"]
    runs = []
    for tag in tags:
        d = np.load(f"pqr/{tag}.npz")
        k = {x: d[x] for x in ("plus_q", "plus_r", "minus_q", "minus_r")}
        for a in ("plus", "minus"):
            ok = np.isfinite(k[a + "_q"]).all(1) & np.isfinite(k[a + "_r"]).all((1, 2))
            k["s" + a[0]] = B.window_mask(d["obs_" + a], SIZE, FLUX) & ok
        log = f"{logs}/bias_{tag}.log"
        st = sel_terms(log)
        m1_log = float(re.search(r"windowed, corrected:\s+m1 = ([-+.\de]+)", open(log).read()).group(1))
        npop = brentq(lambda n: est([(k, n, st, None)], [np.arange(len(k["plus_q"]))])[0] - m1_log,
                      0.5 * npop0, 1.5 * npop0)
        runs.append((tag, (k, npop, st, None), [tuple(r) for r in d["obs_plus"]]))
    # flows drop slightly different targets (region/safe_point): pair on the common rows
    common = set.intersection(*(set(r[2]) for r in runs))
    for tag, (k, *_), keys in runs:
        rows = np.array([i for i, x in enumerate(keys) if x in common])
        rows = rows[np.lexsort(np.array([keys[i] for i in rows]).T)]
        for x in list(k):
            k[x] = k[x][rows]
        print(f"{tag}: {len(keys)} targets, {len(rows)} common")
    n = len(common)
    full = [np.arange(n)]
    rng = np.random.default_rng(0)
    idx = [[rng.choice(n, n)] for _ in range(nboot)]
    P = np.array([est([r[1]], full) for r in runs])                        # (run, 6)
    Bt = np.array([[est([r[1]], i) for i in idx] for r in runs])           # (run, boot, 6)
    print(f"{'run':14s}{'m1 corr':>20s}{'c1':>11s}{'c2':>11s}{'m1 uncorr':>11s}{'m1 - ' + tags[0]:>24s}")
    for j, (tag, _, _) in enumerate(runs):
        dm = Bt[j, :, 0] - Bt[0, :, 0]
        print(f"{tag:14s}{P[j, 0]:+11.5f}+/-{Bt[j, :, 0].std():.5f}{P[j, 1]:+11.1e}{P[j, 2]:+11.1e}"
              f"{P[j, 3]:+11.5f}{P[j, 0] - P[0, 0]:+15.5f}+/-{dm.std():.5f}")
    grp = {}
    for j, (tag, _, _) in enumerate(runs):
        grp.setdefault(tag.rsplit("_", 1)[-1] if "_" in tag else "all", []).append(j)
    print("\nper step multiple: seed mean, seed sd (paired: same targets)")
    for g, js in grp.items():
        v = P[js, 0]
        print(f"  {g}: n={len(js)}  mean m1 {v.mean():+.5f}  seed sd {v.std(ddof=1) if len(js) > 1 else float('nan'):.5f}")
    if {"1x", "2x"} <= grp.keys():
        a, b = P[grp["1x"], 0], P[grp["2x"], 0]
        err = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"  2x - 1x: {b.mean() - a.mean():+.5f} +/- {err:.5f} (seed scatter; galaxies cancel in the pairing)")


if __name__ == "__main__":
    {"d2layer": d2layer, "pair": pair}[sys.argv[1]](sys.argv[2:])
