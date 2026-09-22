"""BFD's own template sum vs the flow, per target, across the size edge.

The residual bias at `Mf >= 1600` sits in `Mr/Mf` 3.26-3.59 and three
explanations are already excluded (the shear layer's response, `R_s`, and
finite-S/the jackknife -- see HANDOFF 2026-09-02).  What was never done in that
band is the direct comparison: what does BFD's OWN estimator give for the same
targets?

The reference here is the paper's eq. (35)-(36) sum over shifted template
COPIES, which is the honest one for a recentred noisy target:

    P(M|g) ~ SUM_c w_c N(M - m_c(g); C_M),   w_c = d2u |J| N(X_c; 0, Sigma_X)

so the centroid marginalisation is carried by ENUMERATION, exactly where the
flow carries it with a deterministic `CentroidMarginalize` transport.  Q and R
are then the first two g-derivatives of `log P`, in closed form (no autodiff,
no importance sampling, no gauge): with whitened residual `u = A(M - m_c)`,
`A = chol(C_M)^-1`, and the copy's own exact `dm_dg`/`d2m_dg2`,

    d_i s_c = u . Aq_i          d_ij s_c = -Aq_i . Aq_j + u . Ar_ij

and `Q`, `R` are the softmax-weighted first/second cumulants of `s`.

Everything is offline: one pass over the copies per target block, fp32 for the
pair arithmetic (the exponent carries ~1e-4 of absolute error, far below the
weights' own spread) with float64 reductions.

    PQR=dev/pqr_auto_20k_S32k.npz NPB=200 python dev/band_tmpl.py
"""
import os
import sys

import fitsio
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "../bfd_cnf_imsims")

import bias                                             # noqa: E402
from imsims.copies import log_weights                   # noqa: E402

D = "../bfd_cnf_imsims/data"
PQR = os.environ.get("PQR", "dev/pqr_auto_20k_S32k.npz")
POP = os.environ.get("POP", "bulgedisc_deep_v2")
COPIES = os.environ.get("COPIES", f"{D}/copies_bulgedisc_v2.fits")
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
PCT = [float(x) for x in os.environ.get("PCT", "0 50 75 90 98").split()]
NPB = int(os.environ.get("NPB", 200))       # targets sampled per size bin
G = float(os.environ.get("G", 0.02))
GUARD = float(os.environ.get("GUARD", 1000))
WMIN = float(os.environ.get("WMIN", 1e-4))  # copy-weight floor, see below
TB = int(os.environ.get("TB", 32))          # targets per block
KB = int(os.environ.get("KB", 262144))      # copies per block
SEED = int(os.environ.get("SEED", 0))
ESSMIN = float(os.environ.get("ESSMIN", 1000))  # template-sum coverage floor
RESTRICT = os.environ.get("RESTRICT", "") not in ("", "0")
CACHE = os.environ.get(
    "CACHE", "/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/"
             "9fa62fdc-58c1-4456-953f-8dd72c66fd88/scratchpad/band_tmpl.npz")


def whitened_templates(cov):
    """Copy moments and derivatives whitened by `C_M`, with their eq.-36 weights.

    Cached: reading and whitening 22.7M copies costs minutes, the comparison
    itself seconds.  Copies below `WMIN` of the peak weight are dropped -- at
    1e-4 that is 47% of them carrying 1.2e-4 of the total weight, which is two
    orders below the template-sum's own sampling error.
    """
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    sx = fitsio.read(f"{D}/{bias.CATALOGS[POP]['zero']}.fits",
                     rows=[0])["cov_odd"][0]
    c = fitsio.read(COPIES, ext="COPIES")
    lw = log_weights(c, sx)
    keep = lw > lw.max() + np.log(WMIN)
    print(f"  {keep.sum()} of {len(lw)} copies kept "
          f"({np.exp(lw[keep] - lw.max()).sum() / np.exp(lw - lw.max()).sum():.6f}"
          f" of the weight)")
    a = np.linalg.inv(np.linalg.cholesky(cov))           # (5,5), C^-1 = A^T A
    m = np.asarray(c["moments"][keep], np.float64) @ a.T
    q = np.asarray(c["dm_dg"][keep], np.float64) @ a.T   # (K,2,5)
    r = np.asarray(c["d2m_dg2"][keep], np.float64) @ a.T  # (K,3,5)
    f32 = lambda x: np.ascontiguousarray(x, np.float32)
    out = dict(
        logw=f32(lw[keep] - lw.max()),
        m2=f32((m * m).sum(-1)), m=f32(m), q=f32(q), r=f32(r),
        # unwhitened Mf and Mr/Mf, so the prior can be reweighted onto the same
        # subpopulation the targets were binned into (see RESTRICT below)
        gal=np.asarray(c["gal"][keep], np.int32),   # dev/prior_n.py subsamples
        mf=f32(c["moments"][keep][:, 0]),
        size=f32(c["moments"][keep][:, 1] / c["moments"][keep][:, 0]),
        # per-copy, target-independent pieces of the two derivative formulae
        mq=f32(np.einsum("ka,kia->ki", m, q)),
        mr=f32(np.einsum("ka,kia->ki", m, r)),
        qq=f32(np.stack([(q[:, 0] * q[:, 0]).sum(-1),
                         (q[:, 0] * q[:, 1]).sum(-1),
                         (q[:, 1] * q[:, 1]).sum(-1)], -1)),
    )
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **out)
    return out


IJ = ((0, 0), (0, 1), (1, 1))


def _logs(mw, t):
    """`log w_c - 0.5 |M - m_c|^2_C` for every (target, copy) pair."""
    d2 = ((mw * mw).sum(-1, keepdims=True) - 2.0 * (mw @ t["m"].T)
          + t["m2"][None])
    return t["logw"][None] - 0.5 * d2


@jax.jit
def _smax_block(mw, t):
    return _logs(mw, t).max(-1)


@jax.jit
def _block(mw, t, smax):
    """One (targets x copies) block -> unnormalised (S0, S1, S2, SUM w^2).

    `smax` shifts the exponent; a second pass is cheaper than an online
    logsumexp and leaves nothing to get subtly wrong.
    """
    w = jnp.exp(_logs(mw, t) - smax[:, None])
    a = jnp.stack([mw @ t["q"][:, i].T - t["mq"][:, i].T for i in range(2)])
    b = jnp.stack([mw @ t["r"][:, i].T - t["mr"][:, i].T for i in range(3)])
    s2 = jnp.stack([b[k] - t["qq"][:, k][None] + a[i] * a[j]
                    for k, (i, j) in enumerate(IJ)])
    f64 = lambda x: jnp.sum(x, axis=-1, dtype=jnp.float64)
    return f64(w), f64(w * a), f64(w * s2), f64(w * w)


def template_pqr(obs, t, a):
    """(Q, R, ESS) from the copy sum, for observed moments `obs` (n,5)."""
    n = len(obs)
    mw = jnp.asarray(np.asarray(obs, np.float64) @ a.T, jnp.float32)
    q, r, ess = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros(n)
    nk = len(t["m"])
    for lo in range(0, n, TB):
        x = mw[lo:lo + TB]
        sm = jnp.full((len(x),), -jnp.inf)
        for k in range(0, nk, KB):
            sm = jnp.maximum(sm, _smax_block(
                x, {kk: v[k:k + KB] for kk, v in t.items()}))
        s0 = s2w = jnp.zeros(len(x), jnp.float64)
        s1 = jnp.zeros((2, len(x)), jnp.float64)
        s2 = jnp.zeros((3, len(x)), jnp.float64)
        for k in range(0, nk, KB):
            d0, d1, d2_, dw = _block(
                x, {kk: v[k:k + KB] for kk, v in t.items()}, sm)
            s0, s1, s2, s2w = s0 + d0, s1 + d1, s2 + d2_, s2w + dw
        qq = np.asarray(s1 / s0).T
        rr = np.asarray(s2 / s0)
        q[lo:lo + TB] = qq
        ij = ((0, 0), (0, 1), (1, 1))
        for k, (i, j) in enumerate(ij):
            r[lo:lo + TB, i, j] = r[lo:lo + TB, j, i] = rr[k] - qq[:, i] * qq[:, j]
        ess[lo:lo + TB] = np.asarray(s0 * s0 / s2w)
        if os.environ.get("VERBOSE"):
            print(f"    {lo + len(x)}/{n}", flush=True)
    return q, r, ess


def selfcheck(t, a, obs, n=8, k=20000):
    """The closed-form Q, R against autodiff through the same sum. SELFCHECK=1."""
    sub = {kk: jnp.asarray(v[:k], jnp.float64) for kk, v in t.items()}
    mw = jnp.asarray(np.asarray(obs[:n], np.float64) @ a.T)

    def logp(g, x):                       # one target, exact lensed templates
        quad = jnp.array([0.5 * g[0] ** 2, g[0] * g[1], 0.5 * g[1] ** 2])
        m = sub["m"] + jnp.einsum("i,kia->ka", g, sub["q"]) \
            + jnp.einsum("i,kia->ka", quad, sub["r"])
        d = x[None] - m
        return jax.scipy.special.logsumexp(
            sub["logw"] - 0.5 * (d * d).sum(-1))

    g0 = jnp.zeros(2)
    q = jax.vmap(lambda x: jax.grad(logp)(g0, x))(mw)
    r = jax.vmap(lambda x: jax.hessian(logp)(g0, x))(mw)
    saved_kb, saved_tb = KB, TB
    globals().update(KB=k, TB=n)
    qc, rc, _ = template_pqr(obs[:n], {kk: v[:k] for kk, v in t.items()}, a)
    globals().update(KB=saved_kb, TB=saved_tb)
    print(f"  selfcheck: max|dQ| {np.abs(np.asarray(q) - qc).max():.2e}, "
          f"max|dR| {np.abs(np.asarray(r) - rc).max():.2e} "
          f"(|Q|~{np.abs(q).max():.1f}, |R|~{np.abs(r).max():.1f})")


def m1(qp, rp, qm, rm):
    return ((qp[:, 0].sum() - qm[:, 0].sum()) / (2 * G)
            / (0.5 * (-rp[:, 0, 0].sum() - rm[:, 0, 0].sum())) - 1)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    d = np.load(PQR)
    ok = np.all([np.isfinite(d[f"{k}_{n}"]).reshape(len(d["flux"]), -1).all(1)
                 for k in ("plus", "minus") for n in "qr"], axis=0)
    ok &= d["flux"] >= FLUX_LO
    if GUARD:
        for n, sl in (("q", (slice(None), 0)), ("r", (slice(None), 0, 0))):
            v = [np.abs(d[f"{k}_{n}"][sl]) for k in ("plus", "minus")]
            med = np.median(np.concatenate([x[ok] for x in v]))
            for x in v:
                ok &= x < GUARD * med
    idx = np.flatnonzero(ok)
    mom = d["moments"][idx]
    c = mom[:, 1] / mom[:, 0]
    edges = np.percentile(c, PCT + [100.0])
    rng = np.random.default_rng(SEED)

    cov = bias.load_cov(f"{D}/{bias.CATALOGS[POP]['plus']}.fits")
    a = np.linalg.inv(np.linalg.cholesky(cov))
    t = {k: jnp.asarray(v) for k, v in whitened_templates(cov).items()}
    print(f"{len(idx)} targets pass (flux >= {FLUX_LO:g}, guard {GUARD:g}); "
          f"{len(t['m'])} copies\n")
    if os.environ.get("SELFCHECK"):
        selfcheck(t, a, d["obs_plus"][idx])

    # A target whose nearest copies are 60 whitened sigma away gets an ESS of a
    # few tens out of 12M and a |Q| of 100 -- 100k galaxies do not cover the
    # bright/compact corner, exactly the failure the flow exists to fix.  Those
    # rows say nothing about the flow, so both estimators are read on the
    # ESS-covered subset and the coverage fraction is reported alongside.
    print(f"{'size pct':>10s}{'n':>6s}{'kept':>6s}{'Mr/Mf':>7s}{'ESS':>8s}"
          f"{'|Q1| flow':>11s}{'|Q1| tmpl':>11s}{'corrQ':>7s}{'slopeQ':>8s}"
          f"{'-R11 flow':>11s}{'-R11 tmpl':>11s}{'corrR':>7s}{'slopeR':>8s}"
          f"{'m1 flow':>10s}{'m1 tmpl':>10s}")
    for i in range(len(PCT)):
        s = np.flatnonzero((c >= edges[i]) & (c <= edges[i + 1]))
        s = rng.permutation(s)[:NPB]
        j = idx[s]
        # RESTRICT: the prior the estimator uses must be the prior of the
        # SELECTED population.  Binning targets by size selects a
        # subpopulation, so restricting the template sum to the same band is
        # the direct test of whether the per-bin bias is that mismatch.
        tb = t
        if RESTRICT:
            lo = edges[i] if i else -np.inf
            hi_ = edges[i + 1] if i + 1 < len(edges) - 1 else np.inf
            mfr = jnp.stack([t["mf"], t["mf"] * t["size"]], -1)
            lf = jnp.concatenate([
                jnp.log(bias.window_prob(mfr[k:k + 250000].astype(jnp.float64),
                                         cov, (lo, hi_), (FLUX_LO, 1e9)) + 1e-300)
                for k in range(0, len(mfr), 250000)])
            tb = dict(t, logw=(t["logw"] + lf).astype(jnp.float32))
            print(f"  prior reweighted by F(m) for Mr/Mf [{lo:.3f}, {hi_:.3f}]:"
                  f" effective copies {float(jnp.exp(lf).sum()):.0f}", flush=True)
        qt, rt, es = {}, {}, {}
        for arm in ("plus", "minus"):
            qt[arm], rt[arm], es[arm] = template_pqr(d[f"obs_{arm}"][j], tb, a)
        qf = {k: d[f"{k}_q"][j] for k in ("plus", "minus")}
        rf = {k: d[f"{k}_r"][j] for k in ("plus", "minus")}
        good = (es["plus"] > ESSMIN) & (es["minus"] > ESSMIN)
        cut = lambda x: {k: v[good] for k, v in x.items()}
        qt, rt, qf, rf = cut(qt), cut(rt), cut(qf), cut(rf)
        cat = lambda x: np.concatenate([x["plus"], x["minus"]])
        q1f, q1t = cat(qf)[:, 0], cat(qt)[:, 0]
        r1f, r1t = -cat(rf)[:, 0, 0], -cat(rt)[:, 0, 0]
        sl = lambda y, x: (y * x).sum() / (x * x).sum()
        hi = PCT[i + 1] if i + 1 < len(PCT) else 100.0
        print(f"{PCT[i]:4.0f}-{hi:<5.0f}{len(s):6d}{int(good.sum()):6d}"
              f"{np.median(c[s]):7.2f}{np.median(cat(es)[np.tile(good, 2)]):8.0f}"
              f"{np.median(np.abs(q1f)):11.2f}{np.median(np.abs(q1t)):11.2f}"
              f"{np.corrcoef(q1f, q1t)[0, 1]:7.2f}{sl(q1f, q1t):8.2f}"
              f"{np.median(r1f):11.2f}{np.median(r1t):11.2f}"
              f"{np.corrcoef(r1f, r1t)[0, 1]:7.2f}{sl(r1f, r1t):8.2f}"
              f"{m1(qf['plus'], rf['plus'], qf['minus'], rf['minus']):+10.4f}"
              f"{m1(qt['plus'], rt['plus'], qt['minus'], rt['minus']):+10.4f}",
              flush=True)
