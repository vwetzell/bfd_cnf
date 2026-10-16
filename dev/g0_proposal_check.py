"""mixture_draws with the full proposal vs bias.g0_proposal: identical draws/weights? ms per target-arm.
Usage: python dev/g0_proposal_check.py FLOW"""
import sys, time; sys.path.insert(0, ".")
import bfd, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
D = "../bfd_cnf_imsims/data"; T, S = 16, 4096
cat = fitsio.read(f"{D}/{B.CATALOGS['bulgedisc_g2n_1m_desv']['plus']}.fits", rows=np.arange(1000, 1000 + T),
                  columns=["moments", "cov", "cov_odd"])
m = np.asarray(cat["moments"][:, :5], np.float64)
cov = np.asarray(bfd.MomentCovariance.bulkUnpack(cat["cov"]), np.float64)[:, :5, :5]
sx = np.asarray(cat["cov_odd"], np.float32)
m_train = shear.load(f"{D}/{B.TRAIN_DATA['bulgedisc_g2n_1m_desv']}")[0]
flow = bulk.SupportedFlow(bulk.load_flow(sys.argv[1], m_train, key=jr.key(0), shear=True, centroid=True), eps=0.0)
out = {}
for name, f in (("g0", B.g0_proposal(flow)), ("full", flow), ("g0", B.g0_proposal(flow)), ("full", flow)):
    md = lambda s: B.mixture_draws(f, m, cov, S, 0.5, s, batch=T, sigma_x=sx, to_host=False)
    jax.block_until_ready(md(7)); t = time.perf_counter()
    for s in range(8, 28): jax.block_until_ready(md(s))
    print(f"{name}: {(time.perf_counter() - t) / 20 / T * 8192 / S * 1e3:.3f} ms/target-arm", flush=True)
    out[name] = [np.asarray(a) for a in md(7)]
(d0, w0), (d1, w1) = out["full"], out["g0"]
print("draws identical:", np.array_equal(d0, d1, equal_nan=True), " max |dd|", np.nanmax(np.abs(d0 - d1)))
f0, f1 = np.isfinite(w0), np.isfinite(w1)
print("weights finite-mask identical:", np.array_equal(f0, f1), " identical:", np.array_equal(w0[f0], w1[f0]),
      " max |dw|", np.abs(w0[f0 & f1] - w1[f0 & f1]).max())
