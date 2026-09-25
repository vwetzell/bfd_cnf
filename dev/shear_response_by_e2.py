"""Follow-up to dev/shear_response_by_e.py, same held-out prior, by galaxy |e|.

1. Propagate the response to the OBSERVED ellipticity e1 = M1/Mr: mean
   d e1/dg1 for the exact path, the flow's path, and two hybrids that swap in
   the flow's response for ONE block only (spin-0 slots z0-z2, or spin-2 slots
   z3-z4) with the exact response for the other -- which block owns the
   high-|e| deficit.
2. Second order.  The flow's sheared path in chart coordinates is (the
   closed-form inverse in `ShearResponse.shear`)
       X(g) = z - Q g + 1/2 g.[(C + C^T) - R].g,   C_iab = sum_j dQ_ia/dz_j Q_jb
   against the exact z(lens(m, g)) with the galaxy's own dm/dg, d2m/dg2.
   Slope of the flow's d2X/dg1^2 on the exact one, per block, per bin.
"""
import sys
sys.path.insert(0, ".")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import bias as B
import bulk
import shear

PRIOR = f"../bfd_cnf_imsims/data/{B.TRAIN_DATA['gauss2_v4n']}"
FLOW = sys.argv[1] if len(sys.argv) > 1 else "flows/shear_g2v4n.eqx"
EBINS = [0, 0.05, 0.08, 0.11, 0.15, 0.2, np.inf]


def main():
    m, dm, d2m = shear.load(PRIOR)
    n_tr = int(0.9 * len(m))
    flow = eqx.tree_deserialise_leaves(
        FLOW, bulk.build_flow(jr.key(0), m[:n_tr][:20000], shear=True, centroid=False))
    chart, layer = flow.bijection.bijection.bijections[:2]
    mh, dmh, d2mh = m[n_tr:], dm[n_tr:], d2m[n_tr:]

    e1 = lambda mm: mm[2] / mm[1]
    g1 = lambda f: jax.jacfwd(f)(0.0)                    # d/dg1 at 0
    g11 = lambda f: jax.jacfwd(jax.jacfwd(f))(0.0)      # d2/dg1^2 at 0

    @jax.jit
    def one(mi, dmi, d2mi):
        lens = lambda t: mi + t * dmi[0] + 0.5 * t * t * d2mi[0]      # g = (t, 0)
        z = chart.transform(mi)
        Q, R = layer.response_tensors(z)
        dQ = jax.jacfwd(lambda zz: layer.response_tensors(zz)[0])(z)  # (5,2,5)
        C = jnp.einsum("iaj,jb->iab", dQ, Q)
        M2 = (C + jnp.swapaxes(C, 1, 2)) - R                          # (5,2,2)
        flow_z = lambda t: z - t * Q[:, 0] + 0.5 * t * t * M2[:, 0, 0]
        exact_z = lambda t: chart.transform(lens(t))
        dz_ex, dz_fl = g1(exact_z), g1(flow_z)
        hyb0 = jnp.concatenate([dz_fl[:3], dz_ex[3:]])                # flow spin-0 only
        hyb2 = jnp.concatenate([dz_ex[:3], dz_fl[3:]])                # flow spin-2 only
        de = lambda dz: jax.jvp(lambda zz: e1(chart.inverse(zz)), (z,), (dz,))[1]
        return (jnp.stack([de(dz_ex), de(dz_fl), de(hyb0), de(hyb2)]),
                g11(exact_z), g11(flow_z))

    out = [jax.vmap(one)(*(jnp.asarray(x[i:i + 4096], jnp.float32) for x in (mh, dmh, d2mh)))
           for i in range(0, len(mh), 4096)]
    de, h_ex, h_fl = (np.concatenate([np.asarray(o[k], np.float64) for o in out]) for k in range(3))
    e = np.hypot(mh[:, 2], mh[:, 3]) / mh[:, 1]
    slope = lambda a, b: (a * b).sum() / (b * b).sum()
    rng = np.random.default_rng(0)

    print(f"{FLOW}: {len(mh)} held-out prior galaxies\n")
    print("1. mean d(e1)/dg1, e1 = M1/Mr   [ratio to exact]")
    print(f"{'|e| bin':>12s} {'exact':>9s} {'flow':>16s} {'flow spin0 only':>18s} {'flow spin2 only':>18s}")
    for lo, hi in zip(EBINS[:-1], EBINS[1:]):
        k = (e >= lo) & (e < hi)
        v = de[k].mean(0); se = de[k].std(0) / np.sqrt(k.sum())
        print(f"{lo:5.2f}-{hi:<6.2f} {v[0]:+.4f}  " + "  ".join(
            f"{v[i]:+.4f} [{v[i] / v[0]:.4f}]" for i in (1, 2, 3)))

    print("\n2. second order d2z/dg1^2: slope of flow on exact (1 = unbiased), bootstrap err")
    print(f"{'|e| bin':>12s} {'spin-0 block':>18s} {'spin-2 block':>18s}")
    for lo, hi in zip(EBINS[:-1], EBINS[1:]):
        idx = np.flatnonzero((e >= lo) & (e < hi))
        cells = []
        for blk in (slice(0, 3), slice(3, 5)):
            s = slope(h_fl[idx, blk], h_ex[idx, blk])
            bs = np.std([slope(h_fl[b, blk], h_ex[b, blk]) for b in (rng.choice(idx, len(idx)) for _ in range(200))])
            cells.append(f"{s:.4f}+/-{bs:.4f}")
        print(f"{lo:5.2f}-{hi:<6.2f} " + "  ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
