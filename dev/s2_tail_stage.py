"""Which training stage put s2's mass at |e| -> 1?  Sample every stage of s1 and s2
(bulk_base, bulk, shear at g=0) on common base draws; |e| tail vs training data. (2026-09-30)"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear
m = np.asarray(shear.load("../bfd_cnf_imsims/data/moments_bulgedisc_g2_bdg2n.fits")[0])[:20000]
N = 1 << 21
for s in ("s1", "s2"):
    for st in ("bulk_base", "bulk", "shear"):
        sh = st == "shear"
        fl = eqx.tree_deserialise_leaves(f"flows/cv2/{st}_{s}.eqx", bulk.build_flow(jr.key(0), m, shear=sh))
        z = fl.base_dist.sample(jr.key(1), (N,))
        c = jnp.zeros(2) if sh else None
        f = eqx.filter_jit(lambda z: jax.vmap(lambda zi: fl.bijection.transform(zi, c))(z))
        x = np.concatenate([np.asarray(f(z[i:i + (1 << 18)]), np.float64) for i in range(0, N, 1 << 18)])
        e = np.hypot(x[:, 2], x[:, 3]) / x[:, 1]
        print(f"{s} {st:9s} frac|e|>0.8 {(e > .8).mean():.1e}  >0.95 {(e > .95).mean():.1e}  max {np.nanmax(e):.5f}  nonfinite {(~np.isfinite(x).all(1)).sum()}", flush=True)
