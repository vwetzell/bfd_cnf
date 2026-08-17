"""Is the reported ESS telling the truth, and is the peel's log-det what breaks it?

`bias.main` measures ESS on ONE chunk (2048 draws) and multiplies by S/chunk,
on the grounds that "ESS grows linearly with draws".  That holds only if the
weights have finite variance: with heavy tails a small sample sees only the
body of the weight distribution, so the per-draw ESS FALLS as S grows and the
scaled number is an overestimate -- the reported 1582 could be a fiction.

The suspect is the centroid layer's log-det, which `pqr_streamed` folds into
the weight (`lw = lw + ld`).  It is the correct change of variables -- draws
live in raw moment space, the peeled flow's density lives in z -- but it varies
~18 nats across a chunk, so it can concentrate the weights on its own.

Two numbers per S, over the same targets and the same draws:

    with ld     what the estimator actually uses
    without ld  the weight the kernel alone would give

If ESS/S is flat in S, the linear scaling is honest.  If it falls, the reported
ESS is an overestimate and every "converged" claim resting on it is soft.

    python dev/check_ess_logdet.py [-n 200] [--alpha 0.5]
"""
import argparse
import sys

import equinox as eqx
import fitsio
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                            # noqa: E402
import bulk                                            # noqa: E402
import shear as shear_top                              # noqa: E402

DATA = "../bfd_cnf_imsims/data"
LADDER = [2048, 8192, 32768, 131072]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_deep.eqx")
    p.add_argument("--targets", default=f"{DATA}/targets_deep_g0_200k.fits")
    p.add_argument("--train-data", default=f"{DATA}/moments.fits")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1001)   # bias.py's noise_seed + 1000
    a = p.parse_args()

    rows = fitsio.read(a.targets)[:4 * a.n]
    rows = rows[~rows["badcenter"]][:a.n]
    m_raw = np.asarray(rows["moments"], dtype=np.float64)
    sigma_x = np.asarray(rows["cov_odd"], dtype=np.float64)
    cov = bias.load_cov(a.targets)

    m_train = shear_top.load(a.train_data)[0]          # no 0.9 slice: centroid on
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    flow_g, layer = bias.split_centroid(flow)

    # The targets go through the layer exactly as `pqr_streamed` sends them.
    m_z = bias.centroid_transform(layer, m_raw, sigma_x, to_host=False)[0]

    print(f"{len(m_raw)} targets, alpha = {a.alpha}, {a.flow}")
    print(f"{'S':>8}  {'ESS with ld':>22}  {'ESS/S':>8}   "
          f"{'ESS without ld':>22}  {'ESS/S':>8}")
    for s in LADDER:
        # targets x draws is what blows up the GPU, so hold that product fixed
        tb = max(1, 262144 // s)
        d, lw = bias.mixture_draws(flow, m_raw, cov, s, a.alpha, a.seed,
                                   batch=tb, sigma_x=sigma_x)
        dz, ld = bias.centroid_transform(layer, d, sigma_x, to_host=False)
        lw = jnp.asarray(lw, dtype=jnp.float32)
        out = []
        for w in (lw + ld, lw):
            e = bias.ess(flow_g, m_z, dz, w, tb, jnp.asarray(sigma_x))
            out.append((np.median(e), np.percentile(e, 5)))
        print(f"{s:>8}  {out[0][0]:>10.1f} (5th {out[0][1]:>6.1f})  "
              f"{out[0][0] / s:>8.4f}   "
              f"{out[1][0]:>10.1f} (5th {out[1][1]:>6.1f})  {out[1][0] / s:>8.4f}")

    print("\nESS/S flat  -> linear scaling honest, reported ESS is real\n"
          "ESS/S falls -> weights heavy-tailed, reported ESS is an overestimate")


if __name__ == "__main__":
    main()
