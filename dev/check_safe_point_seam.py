"""`_mixture_chunk` masks the RAW draw; the flow it evaluates has the layer on it.

`bias._mixture_chunk` builds the proposal weight as

    ok    = in_domain(x)                     # x is the RAW draw
    log_f = where(ok, flow.log_prob(x), -inf)
    log_q = logaddexp(log_alpha + log_k, log_1ma + log_f)

but `flow` there is the FULL flow, centroid layer included, so its density at x
is only defined when `centroid(x, Sigma_X)` is on the chart.  Testing the raw
draw instead splits the population of draws four ways, and two of the cells are
wrong:

  A  in_domain(x), NOT in_domain(z)  -- log_f is NaN/-inf on a draw the
     downstream mask kills anyway.  Harmless: zero weight IS the right answer.
  B  NOT in_domain(x), in_domain(z)  -- log_f is forced to -inf, so log_q drops
     the flow component entirely and log_wt = log_k - log_alpha - log_k =
     -log(alpha).  But `pqr_streamed` masks on z, so the draw COUNTS, carrying a
     weight inflated by whatever the flow component was really worth.

B is a target bias, not a variance one: draw count cannot touch it, it exists
only at alpha < 1, and only on the centroid path.  Those are exactly the three
properties of the +0.0070 centroid-control residual.

This measures how big B is: its share of draws, its share of the weight in
Phat, and the factor by which those draws are over-weighted.

    python dev/check_safe_point_seam.py [-n 200] [--samples 8192]
"""
import argparse
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                            # noqa: E402
import bulk                                            # noqa: E402
import shear as shear_top                              # noqa: E402
from models.bijections import in_domain                # noqa: E402

DATA = "../bfd_cnf_imsims/data"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_deep.eqx")
    p.add_argument("--targets", default=f"{DATA}/targets_deep_g0_200k.fits")
    p.add_argument("--train-data", default=f"{DATA}/moments.fits")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--samples", type=int, default=8192)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1001)
    a = p.parse_args()

    rows = fitsio.read(a.targets)[:4 * a.n]
    rows = rows[~rows["badcenter"]][:a.n]
    m_raw = np.asarray(rows["moments"], dtype=np.float64)
    sigma_x = np.asarray(rows["cov_odd"], dtype=np.float64)
    cov = bias.load_cov(a.targets)

    m_train = shear_top.load(a.train_data)[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    flow_g, layer = bias.split_centroid(flow)

    tb = max(1, 262144 // a.samples)
    x, log_wt = bias.mixture_draws(flow, m_raw, cov, a.samples, a.alpha, a.seed,
                                   batch=tb, sigma_x=sigma_x)
    z, ld = bias.centroid_transform(layer, x, sigma_x, to_host=False)

    ok_x = np.asarray(in_domain(jnp.asarray(x)))
    ok_z = np.asarray(in_domain(z))
    cell_a = ok_x & ~ok_z
    cell_b = ~ok_x & ok_z
    print(f"{len(m_raw)} targets x {a.samples} draws, alpha = {a.alpha}")
    print(f"  in_domain(raw) but not in_domain(z)   [harmless]  "
          f"{cell_a.mean():.2e}")
    print(f"  in_domain(z) but not in_domain(raw)   [COUNTED]   "
          f"{cell_b.mean():.2e}")
    if not cell_b.any():
        print("\n  cell B is empty -- the seam cannot bite.")
        return

    # What Phat actually sums: w_s = exp(log_wt + ld) * P(z_s | g=0).
    m_z = bias.centroid_transform(layer, m_raw, sigma_x, to_host=False)[0]
    one = eqx.filter_jit(jax.vmap(lambda zz, sx: flow_g.log_prob(
        zz, condition=bias.condition(jnp.zeros(2), sx))))
    sx_j = jnp.asarray(sigma_x, jnp.float32)
    lp = np.concatenate([np.asarray(one(z[i:i + tb], sx_j[i:i + tb]), np.float64)
                         for i in range(0, len(m_raw), tb)])
    lw = np.asarray(log_wt, np.float64) + np.asarray(ld, np.float64)
    lw_tot = np.where(ok_z, lp + lw, -np.inf)
    mx = lw_tot.max(axis=1, keepdims=True)
    w = np.where(np.isfinite(lw_tot), np.exp(lw_tot - mx), 0.0)
    share = (w * cell_b).sum(1) / np.maximum(w.sum(1), 1e-300)
    print(f"  cell B's share of Phat: median {np.median(share):.2e}, "
          f"90th pct {np.percentile(share, 90):.2e}, max {share.max():.2e}")

    # How wrong is their weight?  log_q used = log_alpha + log_k (flow component
    # dropped); log_q correct also carries the flow, whose full-flow density at
    # x is the peeled density at z plus the layer's log-det.
    L = np.linalg.cholesky(cov)
    off = (x - m_raw[:, None, :]).reshape(-1, 5).T
    y = np.linalg.solve(L, off)
    log_k = (-0.5 * (y * y).sum(0) - np.log(np.diag(L)).sum()
             - 2.5 * np.log(2 * np.pi)).reshape(x.shape[:2])
    log_f = lp + np.asarray(ld, np.float64)
    log_q_used = np.log(a.alpha) + log_k
    log_q_true = np.logaddexp(log_q_used, np.log1p(-a.alpha) + log_f)
    over = (log_q_true - log_q_used)[cell_b]        # log of the inflation factor
    print(f"  over-weighting of cell B draws: median {np.exp(np.median(over)):.3f}x, "
          f"90th pct {np.exp(np.percentile(over, 90)):.3f}x, "
          f"max {np.exp(over.max()):.3f}x")

    # Net effect on log Phat, per target: what correcting cell B would do.
    corr = np.where(cell_b, np.exp(-(log_q_true - log_q_used)), 1.0)
    d = np.log(np.maximum((w * corr).sum(1), 1e-300)) - np.log(np.maximum(w.sum(1), 1e-300))
    print(f"  d log Phat if cell B were fixed: median {np.median(d):+.2e}, "
          f"mean {d.mean():+.2e}, max |{np.abs(d).max():.2e}|")


if __name__ == "__main__":
    main()
