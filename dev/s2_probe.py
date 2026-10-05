"""Why is seed 2's flow broken (R_s halved, per-target R11 ~ -1.7e4)?  (2026-09-30)
Common base draws through s1 and s2: find draws whose sample moves wildly with g
(or leaves the chart), then locate the layer.  GPU; run alone."""
import os, sys
sys.path.insert(0, ".")
os.environ.setdefault("TRAIN_POP", "bulgedisc_g2n"); os.environ.setdefault("REAL_POP", "bulgedisc_g2n_176k_varobs_sn8")
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, fitsio
sys.argv = sys.argv[:1]
import dev.closed_loop as CL
import bias as B

sx = jnp.asarray(fitsio.read(f"{CL.D}/{CL.REAL['plus']}.fits", rows=[0])["cov_odd"][0], jnp.float32)
F = {s: CL.load_flow(f"flows/cv2/centroid_{s}.eqx") for s in ("s1", "s2")}
print(type(F["s1"].bijection).__name__, [type(b).__name__ for b in getattr(F["s1"].bijection, "bijection", F["s1"].bijection).bijections])
z = F["s1"].base_dist.sample(jr.key(0), (1 << 20,))
h = 0.005
for s, fl in F.items():
    d = CL.sampler(fl, sx)
    mp, mm, m0 = (np.asarray(d(z, jnp.array(g, jnp.float32)), np.float64) for g in ([h, 0], [-h, 0], [0, 0]))
    d2 = (mp + mm - 2 * m0) / h ** 2
    d1 = (mp - mm) / (2 * h)
    bad = ~np.isfinite(m0).all(1)
    print(f"\n{s}: nonfinite {bad.sum()}  |d2m/dg2| pct50/99.99/max per moment",
          np.nanpercentile(np.abs(d2), [50, 99.99, 100], axis=0).round(0).tolist())
    print(f"   |dm/dg| max", np.nanmax(np.abs(d1), 0).round(0).tolist())
    np.savez(f"/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/7a140eea-a3cd-4c43-a042-7378af54bad3/scratchpad/probe_{s}.npz", m0=m0, d1=d1, d2=d2)

# ---- trace the worst s2 draws layer by layer (generative order: last list entry first)
P = {s: np.load(f"/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/7a140eea-a3cd-4c43-a042-7378af54bad3/scratchpad/probe_{s}.npz") for s in F}
lim = np.nanmax(np.abs(P["s1"]["d2"]), 0)
out = (np.abs(P["s2"]["d2"]) > lim).any(1)
print(f"\ns2 draws beyond s1's max |d2m/dg2|: {out.sum()} of {len(out)}")
idx = np.argsort(-np.abs(P["s2"]["d2"]).max(1))[:6]
print("worst s2 m0 [Mf Mr M1 M2 Mc]:\n", P["s2"]["m0"][idx].round(1), "\n  same z in s1:\n", P["s1"]["m0"][idx].round(1))
for s, fl in F.items():
    layers = list(fl.bijection.bijection.bijections)[::-1]
    zz = z[idx]
    ys = {}
    for g in ([h, 0], [-h, 0], [0, 0]):
        c = B.condition(jnp.array(g, jnp.float32), sx)
        y, tr = zz, []
        for L in layers:
            y = jax.vmap(lambda yi: L.inverse(yi, c))(y); tr.append(np.asarray(y, np.float64))
        ys[tuple(g)] = tr
    print(f"\n{s}: per layer (generative order) max over draws of |d2y/dg2| (all 5 coords) and |y|")
    for k, L in enumerate(layers):
        a, b, c0 = ys[(h, 0)][k], ys[(-h, 0)][k], ys[(0, 0)][k]
        print(f"  {k:2d} {type(L).__name__:32s} d2 {np.abs((a + b - 2 * c0) / h ** 2).max(0).round(1)}  |y| {np.abs(c0).max(0).round(2)}")
