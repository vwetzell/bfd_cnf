"""Shared prior-draw pool (`bias._pool_build`/`_pool_stats`) == today's estimator
on the SAME draws.  (2026-10-02)

One chunk for 8 sn8r targets: kernel draws + pool draws picked by `idx`.
  A: `batched`-style `one` over all draws together (today's path)
  B: `one` over the kernel draws + `_pool_stats` over the picked pool draws,
     combined by A-weighting (what `pqr_streamed(pool_reuse>0)` does)
must agree to float32 roundoff.
  python dev/pool_check.py [flow.eqx]
"""
import sys
import numpy as np, fitsio, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
sys.path.insert(0, "."); sys.path.insert(0, "../bfd_cnf_imsims")
import bias as B, bulk, shear
jax.config.update("jax_default_matmul_precision", "highest")

D = "../bfd_cnf_imsims/data/"
cat = B.CATALOGS["bulgedisc_g2n_176k_sn8r"]
r = fitsio.read(f"{D}{cat['plus']}.fits", columns=["moments", "cov_odd"], rows=np.arange(3000))
cov = B.load_cov(f"{D}{cat['zero']}.fits")
m = r["moments"].astype(np.float64)
m8 = m[B.window_mask(m, (2.2, 3.2), (3000, 20000))][:8]
sx = np.repeat(r["cov_odd"][:1].astype(np.float64), 8, 0)
mt = shear.load(D + "moments_bulgedisc_g2_bdg2n.fits")[0]
flow = bulk.load_flow(sys.argv[1] if len(sys.argv) > 1 else "flows/cv5/centroid_s2.eqx",
                      mt, shear=True, centroid=True)
flow_g, layer = B.split_centroid(flow)
print("peeled layer:", layer is not None)

alpha, nk, nf = 0.5, 2048, 2048
L = jnp.asarray(np.linalg.cholesky(cov), jnp.float32)
ld0 = jnp.sum(jnp.log(jnp.diag(L)))
la_, l1_ = float(np.log(alpha)), float(np.log1p(-alpha))
sx0 = jnp.asarray(sx[0], jnp.float32)
pool = B._pool_build(flow, flow_g, layer, sx0, jr.key(3), B._POOL_SUB)
idx = jr.randint(jr.key(4), (8, nf), 0, B._POOL_SUB)
mj, sxj = jnp.asarray(m8, jnp.float32), jnp.asarray(sx, jnp.float32)
kern = B.kernel_draws(cov, 8, nk, 11)


def one(m_i, d, lw, s, ok):
    return B._val_grad_hess(lambda g: B.log_conv_is(flow_g, m_i, d, lw, B.condition(g, s), ok))


onev = eqx.filter_jit(jax.vmap(one))


def prep(offsets):
    d, lw = B._mixture_chunk(flow, mj, offsets, sxj, None, L, ld0, la_, l1_, 0)
    d = d.astype(jnp.float32); lw = lw.astype(jnp.float32) - B._log_jac(d)
    ok = B.in_domain(d) if layer is not None else None
    if layer is not None:
        d, ldc = B.centroid_transform(layer, d, sx, to_host=False)
        lw = lw + ldc
    return d, lw, ok


def stats(f, df, d2f, n):
    return np.asarray(f) + np.log(n), np.asarray(df), np.asarray(d2f) + np.einsum("ia,ib->iab", df, df)


# A: all draws together
allo = jnp.concatenate([kern, pool["u"][idx] - mj[:, None, :]], axis=1)
dA, lwA, okA = prep(allo)
laA, dfA, caA = stats(*onev(mj, dA, lwA, sxj, okA), nk + nf)
# B: kernel via `one` + pool via `_pool_stats`
dK, lwK, okK = prep(kern)
fK = onev(mj, dK, lwK, sxj, okK)
lak, dfk, cak = stats(*fK, nk)
lap, dfp, cap, _, _ = [np.asarray(v) for v in B._pool_stats(mj, idx, pool, L, ld0, la_, l1_)]
laB = np.logaddexp(lak, lap)
wk, wp = np.exp(lak - laB), np.exp(lap - laB)
dfB = wk[:, None] * dfk + wp[:, None] * dfp
caB = wk[:, None, None] * cak + wp[:, None, None] * cap
rel = lambda a, b: np.max(np.abs(a - b) / (np.abs(b) + 1e-3 * np.abs(b).max()))
print(f"pool weight share {np.round(wp, 3)}")
print(f"max |la diff| {np.max(np.abs(laA - laB)):.2e}   rel df {rel(dfB, dfA):.2e}   rel C/A {rel(caB, caA):.2e}")
assert np.max(np.abs(laA - laB)) < 1e-3 and rel(dfB, dfA) < 1e-3 and rel(caB, caA) < 1e-3
print("POOL == TODAY on identical draws: OK")
# The fused chunk (`_pool_chunk`, what pqr_streamed calls) == the manual combine.
laF, dfF, caF, _, _ = [np.asarray(v) for v in B._pool_chunk(
    *fK, jnp.float32(np.log(nk)), mj, jr.key(4), nf, pool, L, ld0, la_, l1_,
    jnp.zeros(B._POOL_SUB), jnp.float32(0))]
print(f"fused: max |la diff| {np.max(np.abs(laF - laB)):.2e}   rel df {rel(dfF, dfB):.2e}   rel C/A {rel(caF, caB):.2e}")
assert np.max(np.abs(laF - laB)) < 1e-4 and rel(dfF, dfB) < 1e-4 and rel(caF, caB) < 1e-4
print("FUSED == MANUAL: OK")
# The whole fused chunk (`_pool_full_chunk`: kernel half + pool half in one
# dispatch, what pqr_streamed calls now) == prep + `one` + `_pool_chunk`.
laX, dfX, caX, _, _ = [np.asarray(v) for v in B._pool_full_chunk(
    flow, flow_g, layer, False, mj, mj, kern, sxj, jr.key(4), nf, pool, L, ld0, la_, l1_,
    jnp.zeros(B._POOL_SUB), jnp.float32(0), jnp.float32(np.log(nk)))]
print(f"full: max |la diff| {np.max(np.abs(laX - laF)):.2e}   rel df {rel(dfX, dfF):.2e}   rel C/A {rel(caX, caF):.2e}")
assert np.max(np.abs(laX - laF)) < 1e-4 and rel(dfX, dfF) < 1e-4 and rel(caX, caF) < 1e-4
print("FULL FUSED == STEPWISE: OK")
