"""Centroid-layer e-response vs Sigma_X scale, by flux tercile -- the margin gate.  GPU.  (2026-10-07)

Ratio = layer / catalog copy-mean e-response, as `check_sigmax`, at Sigma_X x [1/1.5 .. 1.5] on 40k
galaxies.  PASS if the pooled, faint and middle terciles stay inside [1-TOL, 1+TOL] at every scale
(bright: catalog response ~0, ratio is noise, not gated).  Good culled-copy layers sat within 0.97-1.06;
the failures that motivated this (one unphysical template, pooled amp term) at 0.39-1.21 / 0.71.
Usage: python dev/eresp_scan.py COPIES TRAIN_DATA FLOW... [--tol 0.08]
"""
import sys
sys.path.insert(0, ".")
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
import centroid as C, bulk, shear

argv = sys.argv[1:]
tol = 0.08
if "--tol" in argv:
    k = argv.index("--tol"); tol = float(argv[k + 1]); del argv[k:k + 2]
N = 40000
cut = None
if "--template-cut" in argv:   # check only the templates the layer was fit on
    k = argv.index("--template-cut"); cut = (float(argv[k + 1]), float(argv[k + 2])); del argv[k:k + 3]
cp, td, flows = argv[0], argv[1], argv[2:]
copies, galaxies = C.load_copies(cp)
sx0 = np.asarray(galaxies["cov_odd"][0], np.float64)
S = [1 / 1.5, 1 / 1.4, 1 / 1.2, 1.0, 1.2, 1.4, 1.5]
res = C.weighted_copy_mean_many(copies, galaxies, [sx0 * s ** 2 for s in S])
del copies
m_train = shear.load(td)[0]
tmpl = bulk.build_flow(jr.key(2), m_train, shear=True, centroid=True, flux_sas=None, adapter=True)
m0all = np.asarray(galaxies["moments"], np.float64)
e = lambda m: (m[:, 2] + 1j * m[:, 3]) / m[:, 1]
ok = True
for path in flows:
    flow = eqx.tree_deserialise_leaves(path, tmpl)
    layer, chart = C._sigmax_layer(flow), C._chart(flow)
    print(f"\n{path}")
    worst = 0.0
    for s, (tgt, keep) in zip(S, res):
        m0, t = m0all[keep], tgt
        if cut:
            sel = np.asarray(bulk.selected(m0, cut))
            m0, t = m0[sel], t[sel]
        m0, t = m0[:N], t[:N]
        cond = jnp.tile(jnp.concatenate([jnp.zeros(2), jnp.asarray(sx0 * s ** 2, jnp.float32)]), (N, 1))
        pred = np.asarray(jax.vmap(C.sigmax_dm_dsigma, in_axes=(None, 0, 0, None))(
            layer, jnp.asarray(m0, jnp.float32), cond, chart), np.float64)
        e0 = e(m0)
        q = np.quantile(m0[:, 0], [0, 1 / 3, 2 / 3, 1])
        r = lambda mm, w: (e(mm[w]) * np.conj(e0[w])).real.sum() / (np.abs(e0[w]) ** 2).sum() - 1
        ratios = []
        for lo, hi in [(q[0], q[3] + 1)] + list(zip(q[:-1], q[1:])):
            w = (m0[:, 0] >= lo) & (m0[:, 0] < hi)
            ratios.append(r(m0 + pred, w) / r(t, w))
        worst = max(worst, *(abs(x - 1) for x in ratios[:3]))
        print(f"  scale {s:.3f}  catalog {r(t, np.ones(len(m0), bool)):+.4e}  ratio all {ratios[0]:.4f}"
              f"  flux terciles {' '.join(f'{x:.4f}' for x in ratios[1:])}", flush=True)
    good = worst <= tol
    ok &= good
    print(f"eresp scan {'PASS' if good else 'FAIL'}: max |ratio-1| (pooled/faint/middle) {worst:.4f}, tol {tol}")
sys.exit(0 if ok else 1)
