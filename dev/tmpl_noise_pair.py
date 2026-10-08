"""Paired template-sum test of template noise at DES depth.  (2026-10-06)  GPU, no flow.

BFD's own estimator, P(M|g) = SUM_t N(M - m_t(g); C_M), evaluated on the same DES-matched
targets twice: once with the noisy templates (bdg2d, noise 0.611, the flow run's training data)
and once with the clean ones (bdg2n, noise 0.093) -- the SAME seed-0 galaxies, each with its
own exact dm_dg / d2m_dg2.  Delta m1 (noisy - clean) is the bias template noise alone puts into
BFD, with no flow, no deconvolution and no heteroscedasticity question.  The plain template sum
has no centroid marginalisation; that bias is common to both sets and cancels to first order.

Target C_M is exactly s_i^2 * C0 (pixel noise only scales it), so each target's kernel is the
C0-whitened residual divided by s_i^2 (the log-det is constant per target and drops out):
    s_t = -|d|^2 / 2 s^2,   d_i s_t = d.q_i / s^2,   d_ij s_t = (d.r_ij - q_i.q_j) / s^2
Q, R are the softmax-weighted first cumulant and (second cumulant + mean Hessian) of s.
Bins: plus-arm observed Mf 2200-3000 and 3000-4500 (the flow run gives -0.0166 / -0.0068 there,
the Gaussian toy -0.0071 / -0.0021); window weights are the run's tapered Mf 2200-20000 window.
Result (2026-10-06, 261k targets, 94 s/173k): noisy - clean = -0.0042 +/- 0.0002 (2200-3000),
-0.0011 +/- 0.0002 (3000-4500), -0.0012 +/- 0.0022 (4500-8000); above Mf 8000 the sum is
ESS-starved (median ESS 14) and meaningless, which also swamps the whole-window row.
Usage: python dev/tmpl_noise_pair.py [N_TARGETS]     (SELFCHECK=1 checks Q, R vs autodiff)
"""
import os, sys
import numpy as np, fitsio, bfd, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B, bulk

D, G = "../bfd_cnf_imsims/data/", 0.02
B.WINDOW_TAPER = (0.05, 150.0)
S, F = (2.2, 3.5), (2200.0, 20000.0)
BINS = ((1600, 3000), (3000, 4500), (4500, 8000), (8000, 20600))   # 1600/20600: the taper reaches 4 tau = 600 past the edges
TB, KB = 256, 32768
IJ = ((0, 0), (0, 1), (1, 1))


def templates(tag, keep, a):
    t = fitsio.read(D + f"moments_bulgedisc_g2_{tag}.fits", columns=["moments", "dm_dg", "d2m_dg2"])
    f32 = lambda x: jnp.asarray(np.ascontiguousarray(x, np.float32))
    return dict(m=f32(np.asarray(t["moments"][keep], float) @ a.T),
                q=f32(np.asarray(t["dm_dg"][keep], float) @ a.T),       # (K,2,5)
                r=f32(np.asarray(t["d2m_dg2"][keep], float) @ a.T))     # (K,3,5)


@jax.jit
def _block(x, u, m, q, r, sm_old):
    """(TB targets) x (KB templates) -> running max, and sums of w, w a, w s2, w^2 at that shift."""
    d = x[:, None, :] - m[None]                                    # (TB,KB,5)
    ls = -0.5 * u[:, None] * (d * d).sum(-1)
    smax = jnp.maximum(sm_old, ls.max(-1))
    w = jnp.exp(ls - smax[:, None])
    a = u[None, :, None] * jnp.einsum("tkc,kic->itk", d, q)      # (2,TB,KB)
    dr = jnp.einsum("tkc,kjc->jtk", d, r)                          # (3,TB,KB)
    qq = jnp.stack([(q[:, i] * q[:, j]).sum(-1) for i, j in IJ])  # (3,KB)
    s2 = jnp.stack([u[:, None] * (dr[k] - qq[k][None]) + a[i] * a[j] for k, (i, j) in enumerate(IJ)])
    f64 = lambda v: jnp.sum(v, -1, dtype=jnp.float64)
    return smax, f64(w), f64(w * a), f64(w * s2), f64(w * w)


def template_pqr(obs, s2, t, a):
    n, nk = len(obs), t["m"].shape[0]
    xs = jnp.asarray(np.asarray(obs, float) @ a.T, jnp.float32)
    us = jnp.asarray(1.0 / s2, jnp.float32)
    Q, R, ess = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros(n)
    for lo in range(0, n, TB):
        x, u = xs[lo:lo + TB], us[lo:lo + TB]
        sm, acc = jnp.full(len(x), -jnp.inf), None
        for k in range(0, nk, KB):                                 # online max-shifted sums
            new, *o = _block(x, u, t["m"][k:k + KB], t["q"][k:k + KB], t["r"][k:k + KB], sm)
            if acc is not None:
                e = jnp.exp((sm - new).astype(jnp.float64))
                o = [p * e ** (2 if i == 3 else 1) + v for i, (p, v) in enumerate(zip(acc, o))]
            sm, acc = new, o
        s0, s1, s2_, sw = (np.asarray(v) for v in acc)
        q = (s1 / s0).T
        Q[lo:lo + TB] = q
        for k, (i, j) in enumerate(IJ):
            R[lo:lo + TB, i, j] = R[lo:lo + TB, j, i] = s2_[k] / s0 - q[:, i] * q[:, j]
        ess[lo:lo + TB] = s0 * s0 / sw
    return Q, R, ess


def selfcheck(obs, s2, t, a, n=4, k=4096):
    sub = {kk: v[:k] for kk, v in t.items()}
    def logp(g, x, u):
        quad = jnp.array([0.5 * g[0] ** 2, g[0] * g[1], 0.5 * g[1] ** 2])
        m = sub["m"] + jnp.einsum("i,kic->kc", g, sub["q"]) + jnp.einsum("i,kic->kc", quad, sub["r"])
        return jax.scipy.special.logsumexp(-0.5 * u * ((x[None] - m) ** 2).sum(-1))
    xs = jnp.asarray(np.asarray(obs[:n], float) @ a.T, jnp.float32)
    us = jnp.asarray(1.0 / s2[:n], jnp.float32)
    g0 = jnp.zeros(2)
    q = np.asarray(jax.vmap(lambda x, u: jax.grad(logp)(g0, x, u))(xs, us))
    r = np.asarray(jax.vmap(lambda x, u: jax.hessian(logp)(g0, x, u))(xs, us))
    qc, rc, _ = template_pqr(obs[:n], s2[:n], sub, a)
    print(f"selfcheck: max|dQ|/|Q| {np.abs(q - qc).max() / np.abs(q).max():.1e}  "
          f"max|dR|/|R| {np.abs(r - rc).max() / np.abs(r).max():.1e}")


def m1(c, w):            # c: (..., N, 4) = [qp1, qm1, -rp11, -rm11] per row; w: window weights
    s = (c * w).sum(-2)
    return (s[..., 0] - s[..., 1]) / (2 * G) / (0.5 * (s[..., 2] + s[..., 3])) - 1


if __name__ == "__main__":
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cat = {a: fitsio.read(D + f"targets_bdg2_{a}_176k_des.fits", columns=["moments", "cov", "badcenter"])
           for a in ("g1p02", "g1m02", "g0")}
    ok = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
    obs = {a: np.asarray(cat[a]["moments"], float) for a in ("g1p02", "g1m02")}
    cov = {a: bfd.MomentCovariance.bulkUnpack(np.asarray(cat[a]["cov"], float)) for a in obs}
    fp = obs["g1p02"][:, 0]
    rows = np.flatnonzero(ok & (fp >= BINS[0][0]) & (fp < BINS[-1][1]))
    if n_max:
        rows = np.sort(np.random.default_rng(0).choice(rows, min(n_max, len(rows)), replace=False))
    C0 = cov["g1p02"][rows].mean(0)
    C0 /= C0[0, 0] / np.mean([cov[a][rows, 0, 0] for a in obs])   # scale: mean target C_M
    A = np.linalg.inv(np.linalg.cholesky(C0))
    s2 = {a: cov[a][rows, 0, 0] / C0[0, 0] for a in obs}
    w = np.stack([np.asarray(B.window_mask(obs[a][rows], S, F), float) for a in obs], -1)
    keep_t = bulk.physical(np.asarray(fitsio.read(D + "moments_bulgedisc_g2_bdg2d.fits", columns=["moments"])["moments"], float))
    print(f"{len(rows)} targets (plus-arm Mf {BINS[0][0]}-{BINS[-1][1]}), {keep_t.sum()} templates per set, "
          f"window weight sum plus/minus {w.sum(0).round(1)}", flush=True)
    res = {}
    for tag in ("bdg2d", "bdg2n"):
        t = templates(tag, keep_t, A)
        if os.environ.get("SELFCHECK"):
            selfcheck(obs["g1p02"][rows], s2["g1p02"], t, A)
        out = {a: template_pqr(obs[a][rows], s2[a], t, A) for a in obs}
        (qp, rp, ep), (qm, rm, em) = out["g1p02"], out["g1m02"]
        res[tag] = (np.stack([qp[:, 0], qm[:, 0], -rp[:, 0, 0], -rm[:, 0, 0]], -1), ep, em)
        print(f"  {tag} done", flush=True)
    # the flow run's per-target Q, R on the same rows (npz rows are a subset of the catalog's)
    z = np.load("pqr/desm_s2_des.npz")
    g0f = np.asarray(cat["g0"]["moments"][:, 0], float)
    srt = np.argsort(g0f)
    zr = srt[np.searchsorted(g0f[srt], z["moments"][:, 0])]
    pos = np.full(len(g0f), -1); pos[zr] = np.arange(len(zr))
    fr = pos[rows]
    have = fr >= 0
    flow = np.zeros((len(rows), 4))
    flow[have] = np.stack([z["plus_q"][fr[have], 0], z["minus_q"][fr[have], 0],
                           -z["plus_r"][fr[have], 0, 0], -z["minus_r"][fr[have], 0, 0]], -1)
    print(f"flow PQR found for {have.mean():.4f} of the rows")
    cs = np.random.default_rng(1).poisson(1.0, (400, len(rows), 1)).astype(float)
    cn, cc = res["bdg2d"][0], res["bdg2n"][0]
    fpr = fp[rows]
    print(f"{'plus Mf':>12s} {'n':>6s} {'noisy tmpl':>18s} {'clean tmpl':>18s} {'noisy - clean':>18s} "
          f"{'flow (same rows)':>18s} {'ESS med n/c':>12s}")
    for lo, hi in ((2200, 3000), (3000, 4500), (4500, 8000), (8000, 20000), (1600, 20600)):
        b = ((fpr >= lo) & (fpr < hi) & have)[:, None]
        wb = w[:, :1] * b, w[:, 1:] * b
        wb = np.concatenate([wb[0], wb[1], wb[0], wb[1]], -1)      # arm weight per column
        f = lambda c: (m1(c, wb), m1(cs * c, wb).std())
        (vn, en), (vc, ec), (vf, ef) = f(cn), f(cc), f(flow)
        dd = m1(cs * cn, wb) - m1(cs * cc, wb)
        ess = lambda e: np.median(e[b[:, 0] & (w[:, 0] > 0.5)])
        print(f"{lo:5d}-{hi:5d} {int(b.sum()):6d} {vn:+.4f} +/- {en:.4f} {vc:+.4f} +/- {ec:.4f} "
              f"{vn - vc:+.4f} +/- {dd.std():.4f} {vf:+.4f} +/- {ef:.4f} "
              f"{ess(res['bdg2d'][1]):6.0f}/{ess(res['bdg2n'][1]):.0f}")
