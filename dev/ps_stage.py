"""Which training stage sets each flow's P_s error?  (2026-10-05)  CPU, run alone.

Flow-to-flow scatter in corrected m tracks P_s vs the data's in-window fraction (corr -0.91,
5 flows).  Here every stage of every cv6 flow (bulk_base -> bulk -> shear -> centroid) is
sampled at g = 0 and scored three ways against the training set (noiseless moments):
  F    = E[window_prob] (sn8r C_M, J ratio, taper) -- for the centroid flow this is P_s itself
  flux = fraction with 3000 < Mf < 20000             (noiseless, hard)
  size = fraction with 2.2 < Mr/Mf < 3.2 among the flux-window rows (noiseless, hard)
  step = paired change in F from the previous stage (same base draws; centroid vs shear)
Bulk/shear model the noiseless moments, so the reference is the same statistic on the
training data.  The centroid flow's F is compared with the data's in-window fraction
0.17818 +/- 0.00027 (dev/flow_sel_avg.py).
Usage: JAX_PLATFORMS=cpu python dev/ps_stage.py [N_DRAWS]
"""
import os, sys
import fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np, equinox as eqx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B, bulk, shear

D, S, F, FL = "../bfd_cnf_imsims/data/", (2.2, 3.2), (3000, 20000), "flows/cv6/"
B.WINDOW_TAPER = (0.05, 150.0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 20
P_DATA = 0.17818
cat = D + B.CATALOGS["bulgedisc_g2n_176k_sn8r"]["zero"] + ".fits"
cov = jnp.asarray(B.load_cov(cat), jnp.float32)
assert cov.ndim == 2
sx = jnp.asarray(fitsio.read(cat, rows=[0])["cov_odd"][0], jnp.float32)
m_tr = np.asarray(shear.load(D + "moments_bulgedisc_g2_bdg2n.fits")[0], np.float64)
Fw = jax.jit(lambda x: B.window_prob(x, cov, S, F))


def stats(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x).all(1)]
    f = np.concatenate([np.asarray(Fw(jnp.asarray(x[i:i + 65536], jnp.float32)))
                        for i in range(0, len(x), 65536)])
    r, inf = x[:, 1] / x[:, 0], (x[:, 0] > F[0]) & (x[:, 0] < F[1])
    return np.array([f.mean(), inf.mean(), np.mean((r[inf] > S[0]) & (r[inf] < S[1]))]), f


ref, _ = stats(m_tr)
print(f"training data ({len(m_tr)} rows): F {ref[0]:.5f}  flux {ref[1]:.5f}  size {ref[2]:.5f};  "
      f"data in-window fraction (noisy targets) {P_DATA}")
print(f"{N} draws/flow; columns = value - training-data value (x1e3), F MC sd in brackets")
for s in (2, 3, 4, 5, 6):
    prev = None
    for st in ("bulk_base", "bulk", "shear", "centroid"):
        p = f"{FL}{st}_s{s}.eqx"
        if not os.path.exists(p):
            continue
        fl = bulk.load_flow(p, m_tr[:20000], key=jr.key(0), shear=st in ("shear", "centroid"),
                            centroid=st == "centroid")
        c = B.condition(jnp.zeros(2), sx) if st == "centroid" else (jnp.zeros(2) if st == "shear" else None)
        f = eqx.filter_jit(lambda z: jax.vmap(lambda zi: fl.bijection.transform(zi, c))(z))
        z = fl.base_dist.sample(jr.key(7), (N,))
        x = np.concatenate([np.asarray(f(z[i:i + 65536])) for i in range(0, N, 65536)])
        assert np.isfinite(x).all(), p
        v, fw = stats(x)
        e = fw.std() / np.sqrt(N)
        step = "" if prev is None else f"  step {(fw - prev).mean() * 1e3:+6.2f} ({(fw - prev).std() / np.sqrt(N) * 1e3:.2f})"
        prev = fw
        d = (v - ref) * 1e3
        extra = f"   P_s {v[0]:.5f} ({(v[0] - P_DATA) / 0.00027:+.1f} sigma vs data)" if st == "centroid" else ""
        print(f"  s{s} {st:>9s}: F {d[0]:+6.2f} ({e * 1e3:.2f})  flux {d[1]:+6.2f}  size {d[2]:+6.2f}{step}{extra}",
              flush=True)
