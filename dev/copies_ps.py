# P_s straight from the copies (no flow): sum_u w(u) F(m(u)) / N_gal, with the
# copies' own noiseless-|J(u)| weights, plain F vs the paper's noisy-J F_J.
# Population: sn8r (round PSF, fixed noise) -- the conditions the copies were made for.
import sys, numpy as np, fitsio, jax, jax.numpy as jnp
sys.path.insert(0, "../bfd_cnf_imsims")
import bias as B
from imsims.copies import log_weights
D = "../bfd_cnf_imsims/data/"
W = {"base": ((2.2, 3.2), (3000, 20000)), "flux4k": ((2.2, 3.2), (4000, 20000)),
     "flux6k": ((2.2, 3.2), (6000, 20000)), "size30": ((2.2, 3.0), (3000, 20000)),
     "size28": ((2.2, 2.8), (3000, 20000)), "size24lo": ((2.4, 3.2), (3000, 20000)),
     "size35": ((2.2, 3.5), (3000, 20000))}
tz = D + "targets_bdg2_g0_176k_sn8r.fits"
C = B.load_cov(tz); assert C.ndim == 2
sx = fitsio.read(tz, columns=["cov_odd"], rows=[0])["cov_odd"][0]
print("C_M diag", np.diag(C).round(0), " Tr(BC)*4 =", C[1,1]-C[2,2]-C[3,3], " Sigma_X", sx)
# sims fractions, averaged over the three arms
sims = {}
for k in W:
    fr = []
    for a in ("g0", "g1p02", "g1m02"):
        m = fitsio.read(D + f"targets_bdg2_{a}_176k_sn8r.fits", columns=["moments"])["moments"]
        fr.append(B.window_mask(np.asarray(m, float), *W[k]).mean())
    sims[k] = np.mean(fr)
fp = D + "copies_bulgedisc_g2_bdg2n_sn8.fits"
f = fitsio.FITS(fp); n = f["COPIES"].get_nrows(); ngal = f["GALAXIES"].get_nrows()
Cj = jnp.asarray(C)
fns = {(k, v): jax.jit(lambda m, s=W[k][0], fl=W[k][1], v=v: B.window_prob(m, Cj, s, fl, jac=v))
       for k in W for v in ("none", "full")}
acc = {kv: 0.0 for kv in fns}; wsum = 0.0
step = 4_000_000
for i in range(0, n, step):
    c = f["COPIES"].read(columns=["moments", "xy", "da"], rows=np.arange(i, min(i + step, n)))
    w = np.exp(log_weights(c, sx))
    wsum += w.sum()
    m = jnp.asarray(np.asarray(c["moments"], np.float64))
    for kv, g in fns.items():
        acc[kv] += float(np.dot(w, np.asarray(g(m), np.float64)))
    print(f"  {min(i+step,n)}/{n}", flush=True)
print(f"\nsum w / N_gal = {wsum/ngal:.5f}  (mean detection prob of the copies' galaxies)")
se = lambda p: np.sqrt(p * (1 - p) / 520000)
print(f"{'window':>9s} {'sims':>8s} {'copies F':>9s} {'copies F_J':>10s}   (F - sims, F_J - sims) in sims sigma")
for k in W:
    p0, pj = acc[(k, "none")] / ngal, acc[(k, "full")] / ngal
    print(f"{k:>9s} {sims[k]:>8.4f} {p0:>9.4f} {pj:>10.4f}   {(p0-sims[k])/se(sims[k]):+6.1f} {(pj-sims[k])/se(sims[k]):+6.1f}")
