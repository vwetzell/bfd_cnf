"""Does the sub-3.5 ramp track how far each target's C_M integral REACHES?

`dev/sweep_ramp_cut.py` showed that zeroing the flow's own noiseless ramp
(`--score-weight 3`) leaves ~78% of the noisy one, and the noisy ramp is the
same size with the centroid layer off as on.  What is left in between is the
convolution: `P(M|g) = INT dm P(m|g) L(M_i - m)` evaluates the flow over a
C_M-wide neighbourhood of every target, and for anything near `Mr/Mf ~ 3.4`
that neighbourhood includes the 3.5-3.9 region where the density is known to be
wrong.  A point evaluation never touches it; an IS integral touches it for
every target.

This measures the reach directly: per target, the share of the importance
weight coming from draws with `Mr/Mf > RCUT`,

    reach_i = sum_{s: Mr/Mf > RCUT} w_s  /  sum_s w_s,
    w_s = exp(log_wt_s) P(draws_s | g = 0),

which is the same weight `bias.ess` reduces, with a different reduction.  The
draws are classified in the ORIGINAL moment space, where the point-source
boundary lives, while the density is evaluated in the centroid-marginalised
space -- so both forms of each draw have to be carried.

**The discriminating table is the last one.**  `reach` correlates strongly with
`Mr/Mf` by construction, so a ramp in `reach` alone proves nothing.  What the
convolution hypothesis predicts, and resolution-per-se does not, is that m1
varies with `reach` AT FIXED noise-free `Mr/Mf`.

    python dev/check_convolution_reach.py [pqr.npz] [--samples 4096]
"""
import argparse
import os
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear                                         # noqa: E402
from dev.check_bias_flatness import noise_free       # noqa: E402

DATA = "../bfd_cnf_imsims/data"
SP = ("/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/"
      "c729e54d-31f4-414c-8c5d-6d6a496207d0/scratchpad")
RCUT, ALPHA, SEED, NBOOT = 3.5, 0.5, 1001, 300   # RCUT overridable
NB = 3


def reach(flow, m_raw, cov, sigma_x, samples, tbatch, rcut=RCUT):
    """Per-target share of the IS weight coming from draws above RCUT."""
    flow_g, layer = bias.split_centroid(flow)
    assert layer is not None, "expected a centroid flow"

    @eqx.filter_jit
    def one(m_t, d_t, lw, sx, above, ok):
        cond = bias.condition(jnp.zeros(2), sx)
        dummy = bias.safe_point(m_t)
        lp = flow_g.log_prob(jnp.where(ok[:, None], d_t, dummy), condition=cond)
        l = jnp.where(ok, lp + lw, -jnp.inf)
        return jnp.exp(jax.nn.logsumexp(jnp.where(above, l, -jnp.inf))
                       - jax.nn.logsumexp(l))

    out = []
    for i in range(0, len(m_raw), tbatch):
        s = slice(i, i + tbatch)
        d, lw = bias.mixture_draws(flow, m_raw[s], cov, samples, ALPHA, SEED,
                                   sigma_x=sigma_x[s])
        ok = np.asarray(bias.in_domain(d))
        # Classified BEFORE the centroid transform: RCUT is a boundary in the
        # original moment space (bulk.POINT_SOURCE lives there), and the
        # transform shifts Mr/Mf by an amount that grows exactly where we are
        # trying to measure.
        above = ok & (np.divide(d[:, :, 1], d[:, :, 0],
                                out=np.zeros_like(d[:, :, 0]), where=ok) > rcut)
        d_t, ld = bias.centroid_transform(layer, d, sigma_x[s])
        m_t, _ = bias.centroid_transform(layer, m_raw[s], sigma_x[s])
        out.append(np.asarray(jax.vmap(one)(
            jnp.asarray(m_t), jnp.asarray(d_t), jnp.asarray(lw + ld),
            jnp.asarray(sigma_x[s]), jnp.asarray(above), jnp.asarray(ok))))
        print(f"  {min(i + tbatch, len(m_raw))}/{len(m_raw)}", flush=True)
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pqr", nargs="?", default=f"{SP}/paired_with.npz")
    p.add_argument("--flow", default="flows/centroid_deep.eqx")
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--tbatch", type=int, default=500)
    p.add_argument("--rcut", type=float, default=RCUT,
                   help="the boundary reach is measured above; POINT_SOURCE "
                        "(3.976167) asks how much weight the bounded chart "
                        "removes outright, 3.5 how much sits in the band whose "
                        "density is merely wrong")
    p.add_argument("--cache", default=None,
                   help="npz to save/reuse the per-target reach in")
    a = p.parse_args()

    cat = bias.CATALOGS["bulgedisc_deep"]
    rows = {k: fitsio.read(f"{DATA}/{v}.fits")[:20000] for k, v in cat.items()}
    keep = np.ones(len(rows["zero"]), bool)
    for v in rows.values():
        keep &= ~v["badcenter"]
    z = rows["zero"][keep]
    M0 = np.asarray(z["moments"], np.float64)
    sigma_x = np.asarray(z["cov_odd"], np.float64)
    cov = bias.load_cov(f"{DATA}/{cat['zero']}.fits")

    # Same build as `bias.main` for a centroid flow: standardised on the FULL
    # training population, not the 90% slice.
    m_train = shear.load(f"{DATA}/moments.fits")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    print(f"{a.flow}, {len(M0)} targets, {a.samples} draws, alpha={ALPHA}")

    sel = M0[:, 0] > 2800
    rhat = noise_free(z, M0[:, 1] / M0[:, 0], sel)
    if a.cache and os.path.exists(a.cache):
        w = np.load(a.cache)["reach"]
        print(f"  reach from {a.cache}")
    else:
        w = reach(flow, M0, cov, sigma_x, a.samples, a.tbatch, a.rcut)
        if a.cache:
            np.savez_compressed(a.cache, reach=w)
    # `reach` is built from the target's OWN noisy moments, so binning m1 on it
    # raw is the mistake that produced the spurious +4%/-49% pattern: the same
    # noise draw sits in the binning variable and in Q.  `what` is the part of
    # reach predictable from the exact per-galaxy derivatives, hence noise-free
    # by construction, and it is the only version whose grid can be believed.
    what = noise_free(z, w, sel)
    print(f"  noise-free reach: R^2 = "
          f"{1 - ((w - what)[sel] ** 2).mean() / w[sel].var():.3f}")

    # The base SELECTION stays at 3.5 whatever --rcut is: it defines the
    # targets a real analysis keeps, not the boundary reach is measured above.
    base = sel & (rhat < RCUT)
    d = np.load(a.pqr)
    rng = np.random.default_rng(0)
    est = lambda s: (bias.ghat(d["plus_q"], d["plus_r"], s)[0]
                     - bias.ghat(d["minus_q"], d["minus_r"], s)[0]) / 0.04 - 1

    def boot(s):
        i0 = np.flatnonzero(s)
        return float(np.std([est(rng.choice(i0, len(i0))) for _ in range(NBOOT)]))

    print(f"\nreach above Mr/Mf={a.rcut}, over the base selection (N={base.sum()})")
    print("  percentiles 5/25/50/75/95: "
          + " ".join(f"{v:.4f}" for v in np.percentile(w[base], [5, 25, 50, 75, 95])))
    print(f"  Spearman rank corr with noise-free Mr/Mf: "
          f"{np.corrcoef(np.argsort(np.argsort(w[base])), np.argsort(np.argsort(rhat[base])))[0, 1]:+.3f}")

    def edges(x, n):
        e = np.quantile(x[base], np.linspace(0, 1, n + 1))
        e[0], e[-1] = -np.inf, np.inf
        return e

    if not np.isfinite(what[base]).any() or w[base].ptp() == 0:
        # A degenerate reach is itself the answer, not a crash: it says the
        # boundary asked about carries no weight, so anything that acts only
        # there cannot move the bias.  Measured for --rcut = POINT_SOURCE.
        print(f"\n  reach above {a.rcut} is identically {w[base].mean():.2g} "
              "-- nothing to bin, and nothing an intervention there could remove")
        return

    we, re_ = edges(what, NB), edges(rhat, NB)
    for nm, x, e in (("noise-free reach", what, we),
                     ("raw (noisy) reach", w, edges(w, NB)),
                     ("noise-free Mr/Mf", rhat, re_)):
        print(f"\nmarginal in {nm}")
        for k in range(NB):
            s = base & (x >= e[k]) & (x < e[k + 1])
            print(f"  {e[k]:8.4f}-{e[k + 1]:<8.4f} {s.sum():6d}  "
                  f"m1 = {est(s):+.4f} +/- {boot(s):.4f}")

    # The discriminator: m1 against reach WITHIN a resolution tercile.  If the
    # ramp is resolution per se, the rows are flat; if it is the convolution
    # reaching the bad region, m1 climbs left to right inside every row.
    print(f"\nm1 on a {NB}x{NB} grid (rows: noise-free Mr/Mf tercile, "
          "cols: noise-free reach tercile)")
    print(f"{'Mr/Mf':>14s}" + "".join(
        f"{f'reach {we[j]:.3f}-{we[j + 1]:.3f}':>22s}" for j in range(NB)))
    slopes = []
    for i in range(NB):
        line = f"{f'{re_[i]:.2f}-{re_[i + 1]:.2f}':>14s}"
        row = []
        for j in range(NB):
            s = (base & (rhat >= re_[i]) & (rhat < re_[i + 1])
                 & (what >= we[j]) & (what < we[j + 1]))
            if s.sum() < 80:
                line += f"{'--':>22s}"
                row.append(None)
                continue
            v, e = est(s), boot(s)
            row.append((v, e))
            line += f"{f'{v:+.4f}+/-{e:.4f}':>22s}"
        print(line)
        if row[0] and row[-1]:
            slopes.append((row[-1][0] - row[0][0],
                           np.hypot(row[-1][1], row[0][1])))
    if slopes:
        v = np.array([s[0] for s in slopes])
        e = np.array([s[1] for s in slopes])
        wgt = 1 / e ** 2
        print(f"\nwithin-row (high reach - low reach) at fixed resolution: "
              + ", ".join(f"{a_:+.4f}+/-{b:.4f}" for a_, b in slopes))
        print(f"  inverse-variance mean {(v * wgt).sum() / wgt.sum():+.4f} "
              f"+/- {np.sqrt(1 / wgt.sum()):.4f}")


if __name__ == "__main__":
    main()
