"""Step 1-2 of the leader diagnosis: at the top-R_s leader draws, compare the
flow's (Q, R) against truth.pqr_sigma's exact (Q, R) and measure local template
support (10-NN distance in the flow's chart coords vs the bulk median).

    python dev/leader_truth_check.py --flow flows/centroid_g2v4n.eqx
"""
import argparse, sys
import equinox as eqx, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
sys.path.insert(0, ".")
import bias as B, bulk, shear

p = argparse.ArgumentParser()
p.add_argument("--pop", default="gauss2_v4n")
p.add_argument("--flow", default="flows/centroid_g2v4n.eqx")
p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
p.add_argument("--log2-draws", type=int, default=20)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--out", default="dev/leaders_g2v4n.npy")
a = p.parse_args()

m_train = np.asarray(shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0], dtype=np.float64)
flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
flow = eqx.tree_deserialise_leaves(a.flow, flow)
jax.config.update("jax_enable_x64", True)
flow = jax.tree_util.tree_map(
    lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
import truth
cat = B.CATALOGS[a.pop]
cov = B.load_cov(f"{a.data_dir}/{cat['zero']}.fits")
sx = jnp.asarray(np.asarray(fitsio.read(f"{a.data_dir}/{cat['zero']}.fits")["cov_odd"],
                            dtype=np.float64)[0])
zero, e0, e1 = jnp.zeros(2), jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])

def one(m_i):
    f = lambda g: flow.log_prob(m_i, condition=B.condition(g, sx))
    (_, q), lin = jax.linearize(jax.value_and_grad(f), zero)
    _, h0 = lin(e0); _, h1 = lin(e1)
    return q, jnp.stack([h0, h1], axis=-1)

chunked = eqx.filter_jit(jax.vmap(one))
fprob = eqx.filter_jit(lambda mm: B.window_prob(mm, cov, (2.2, 3.2), (3000.0, 20000.0)))
n = 1 << a.log2_draws
zs = flow.base_dist.sample(jr.key(31), (n,)).astype(jnp.float64)
tr = eqx.filter_jit(jax.vmap(lambda z: flow.bijection.transform(z, B.condition(zero, sx))))
m0 = np.concatenate([np.asarray(tr(zs[i:i + 16384])) for i in range(0, n, 16384)])
pv, pm = [], []
for i in range(0, n, a.batch):
    mm = jnp.asarray(m0[i:i + a.batch]); q, r = chunked(mm); F = fprob(mm)
    ok = jnp.isfinite(q).all(-1) & jnp.isfinite(r).all(-1).all(-1) & jnp.isfinite(F)
    if not int(ok.sum()): continue
    Fk, qk, rk = np.asarray(F[ok]), np.asarray(q[ok]), np.asarray(r[ok])
    v = Fk * rk[:, 0, 0] + Fk * qk[:, 0] ** 2
    j = np.argsort(np.abs(v))[::-1][:64]
    pv.append(v[j]); pm.append(np.asarray(mm[ok])[j])
v, mt = np.concatenate(pv), np.concatenate(pm)
o = np.argsort(np.abs(v))[::-1][:10]
L, vL = mt[o], v[o]
np.save(a.out, L)
qf, rf = [np.asarray(x) for x in chunked(jnp.asarray(L))]
qt, rt = [np.asarray(x) for x in jax.vmap(lambda m: truth.pqr_sigma(m, sx))(jnp.asarray(L))]
from scipy.spatial import cKDTree
tc = bulk.to_coords(m_train[np.random.default_rng(0).choice(len(m_train), 200_000, replace=False)])
tc = tc[np.isfinite(tc).all(-1)]
mu, sd = tc.mean(0), tc.std(0)
tree = cKDTree((tc - mu) / sd)
d_l = tree.query((bulk.to_coords(L) - mu) / sd, k=10)[0][:, -1]
d_b = tree.query((tc[:5000] - mu) / sd, k=11)[0][:, -1]
print(f"10-NN dist bulk median {np.median(d_b):.3f}  p95 {np.percentile(d_b,95):.3f}")
print("rank   Mf   Mr/Mf | |contrib|  R11_flow  R11_truth ratio | Q1_flow Q1_truth | Q1^2/R11 flow | 10NN/bulk")
for k in range(10):
    print(f"{k:3d} {L[k,0]:7.0f} {L[k,1]/L[k,0]:6.3f} | {abs(vL[k]):8.3g} "
          f"{rf[k,0,0]:9.3g} {rt[k,0,0]:9.3g} {rf[k,0,0]/rt[k,0,0]:7.2f} | "
          f"{qf[k,0]:8.3g} {qt[k,0]:8.3g} | {qf[k,0]**2/abs(rf[k,0,0]):6.2f} | "
          f"{d_l[k]/np.median(d_b):5.1f}")
