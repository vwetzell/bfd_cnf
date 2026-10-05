"""Flow vs templates, stage by stage: which layer carries the faint-end e1 under-response? (2026-10-01)

Per (Mf band, size 2.0-3.4), g1 = +/-H, common base draws:
  stage shear:    bulk+shear flow           vs templates (training parents, unit weight, analytic lensing)
  stage centroid: full flow (sn8r Sigma_X)  vs copies, eq-36 weights with |J| LENSED
                  ([[copies-g-weights-freeze-J]]: log_weights(g=) alone is +8% biased)
Reports band mass fraction at g=0 and d<e1>/dg1, ratio flow/reference, jackknife errors.
  python dev/stage_response.py flows/cv4 s2
"""
import sys
import numpy as np, fitsio, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
sys.path.insert(0, "."); sys.path.insert(0, "../bfd_cnf_imsims")
import bias as B, bulk, shear
from imsims.copies import log_weights, log_jacobian

D = "../bfd_cnf_imsims/data/"
H, NCH, CH, SIZE = 0.02, 32, 500_000, (2.0, 3.4)
BANDS = [(2000, 3000), (3000, 3500), (3500, 4000), (4000, 5000), (5000, 8000), (8000, 20000)]
FD, S = sys.argv[1], sys.argv[2]


def stats(m, w):
    """per band: sum w, sum w e1.  Plus total weight in the last row (for mass fractions)."""
    s, e1 = m[:, 1] / m[:, 0], m[:, 2] / m[:, 1]
    out = np.zeros((len(BANDS) + 1, 2))
    for b, (lo, hi) in enumerate(BANDS):
        k = (m[:, 0] >= lo) & (m[:, 0] < hi) & (s > SIZE[0]) & (s < SIZE[1])
        out[b] = w[k].sum(), (w[k] * e1[k]).sum()
    out[-1, 0] = w.sum()
    return out


def summarize(p, m, z):
    """(chunks, bands+1, 2) sums at +H, -H, 0 -> (frac, resp) per band, jackknife errors."""
    def val(P, M, Z):
        return np.stack([Z[:-1, 0] / Z[-1, 0],
                         (P[:-1, 1] / P[:-1, 0] - M[:-1, 1] / M[:-1, 0]) / (2 * H)], 1)
    P, M, Z = p.sum(0), m.sum(0), z.sum(0)
    v = val(P, M, Z)
    jk = np.array([val(P - p[i], M - m[i], Z - z[i]) for i in range(len(p))])
    n = len(p)
    return v, np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))


mt = shear.load(D + "moments_bulgedisc_g2_bdg2n.fits")[0]
sx = fitsio.read(D + "targets_bdg2_g0_176k_sn8r.fits", columns=["cov_odd"], rows=[0])["cov_odd"][0]
res = {}
for stage, cent in (("shear", False), ("centroid", True)):
    fl = eqx.tree_deserialise_leaves(f"{FD}/{stage}_{S}.eqx", bulk.build_flow(jr.key(0), mt, shear=True, centroid=cent))
    tr = eqx.filter_jit(jax.vmap(lambda z, c: fl.bijection.transform(z, c), in_axes=(0, None)))
    cond = (lambda g: B.condition(jnp.array([g, 0.]), jnp.asarray(sx, jnp.float32))) if cent else \
           (lambda g: jnp.array([g, 0.], jnp.float32))
    acc = [[], [], []]
    for i in range(NCH):
        z = fl.base_dist.sample(jr.key(500 + i), (CH,))
        out = [np.asarray(tr(z, cond(g)), np.float64) for g in (H, -H, 0.0)]
        ok = np.all([np.isfinite(o).all(1) for o in out], 0)
        for a, o in zip(acc, out):
            a.append(stats(o[ok], np.ones(ok.sum())))
    res[stage] = summarize(*map(np.array, acc))
    print(f"flow {stage} done", flush=True)

# templates: split into 32 chunks for the jackknife
pm, pdm, pd2m = (fitsio.read(D + "moments_bulgedisc_g2_bdg2n.fits", columns=[k])[k].astype(np.float64)
                 for k in ("moments", "dm_dg", "d2m_dg2"))
acc = [[], [], []]
for idx in np.array_split(np.arange(len(pm)), NCH):
    for a, sg in zip(acc, (1, -1, 0)):
        a.append(stats(pm[idx] + sg * H * pdm[idx, 0] + 0.5 * H * H * sg * sg * pd2m[idx, 0], np.ones(len(idx))))
ref = {"shear": summarize(*map(np.array, acc))}

f = fitsio.FITS(D + "copies_bulgedisc_g2_bdg2n_sn8.fits")["COPIES"]
n = f.get_nrows()
acc = [[], [], []]
for idx in np.array_split(np.arange(n), NCH):        # contiguous rows = whole galaxies per chunk
    c = f.read(columns=["moments", "xy", "da", "dm_dg", "d2m_dg2", "dxy_dg", "d2xy_dg2"], rows=idx)
    m0 = np.asarray(c["moments"], np.float64)
    for a, sg in zip(acc, (1, -1, 0)):
        ml = m0 + sg * H * c["dm_dg"][:, 0] + 0.5 * H * H * sg * sg * c["d2m_dg2"][:, 0]
        lw = log_weights(c, sx, g=np.array([sg * H, 0.0])) + log_jacobian(ml) - log_jacobian(m0)
        w = np.where(np.isfinite(lw), np.exp(lw), 0.0)
        a.append(stats(ml, w))
ref["centroid"] = summarize(*map(np.array, acc))
print("references done\n", flush=True)

print(f"{FD} {S}: size {SIZE}, H = {H}; ratio = flow / reference")
for stage in ("shear", "centroid"):
    (vf, ef), (vr, er) = res[stage], ref[stage]
    print(f"\nstage {stage}  (reference: {'templates' if stage == 'shear' else 'copies, lensed J'})")
    print(f"{'Mf band':>11s} {'frac flow':>10s} {'ref':>8s} {'ratio':>16s}   {'de1/dg flow':>12s} {'ref':>8s} {'ratio':>16s}")
    for b, (lo, hi) in enumerate(BANDS):
        r = vf[b] / vr[b]
        re = np.abs(r) * np.hypot(ef[b] / vf[b], er[b] / vr[b])
        print(f"{lo:>5d}-{hi:<5d} {vf[b, 0]:>10.5f} {vr[b, 0]:>8.5f} {r[0]:>8.4f}+/-{re[0]:.4f}   "
              f"{vf[b, 1]:>12.4f} {vr[b, 1]:>8.4f} {r[1]:>8.4f}+/-{re[1]:.4f}")
