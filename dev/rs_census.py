"""Which draws carry `R_s11`, and is the concentration one draw or a heavy tail?

`bias.py` warns when the top draw owns >5% of `R_s11`.  Both gauss2 runs on
2026-09-05 tripped it -- 13% at `g2v3`, 17% at `g2v3d` -- and both named a draw
at `Mf ~ 1.732e+04, Mr/Mf ~ 3.40`, i.e. bright and close to the point-source
ceiling (3.6926).  That warning reports ONE number and cannot tell apart the
two things it could mean:

  * one bad draw -- a sample-mean accident, curable by a guard, and `R_s`
    is otherwise a converged quantity;
  * a heavy-tailed integrand -- `R_s` then has no effective sample size, more
    draws do not help, and the corrected `m1` is not a measurement.

The estimator is `R_s = E[F (R + Q Q^T)]` (`bias.selection_terms_score`), so
this walks the same draws and censuses the per-draw contribution
`F (R11 + Q1^2)`:

  * top-k shares and `R_s11` re-evaluated with the top k dropped;
  * the R11 vs Q1^2 split on the leaders -- `Q1^2 >= 0` always, so a Q-driven
    spike is a steep SCORE and an R-driven one is density CURVATURE, and they
    have different fixes;
  * a Hill tail index over the top order statistics.  `selection_terms_score`'s
    docstring claims Hill 2.9 for this estimator against 1.2-1.4 for the
    pathwise ones -- i.e. a finite mean and variance.  Below ~2 the variance is
    gone; below ~1 so is the mean.
  * where the leaders sit in the chart.

Run: python dev/rs_census.py --pop gauss2_v3 --flow flows/centroid_g2v3.eqx
"""
import argparse
import os
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402

KEEP = 4096          # order statistics retained; enough for Hill and drop-k


def hill(sorted_desc, k):
    """Hill estimator of the tail index from the top k order statistics."""
    x = sorted_desc[:k + 1]
    return 1.0 / np.mean(np.log(x[:k] / x[k])) if x[k] > 0 else np.inf


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pop", default="gauss2_v3", choices=sorted(B.CATALOGS))
    p.add_argument("--flow", required=True)
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--log2-draws", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(2500.0, 50000.0))
    p.add_argument("--save", default=None,
                   help="npz for the retained order statistics: the leaders' "
                        "moments and their F R11 / F Q1^2 split, so the exact "
                        "density can be evaluated at the SAME points")
    a = p.parse_args()

    cat = B.CATALOGS[a.pop]
    targets = f"{a.data_dir}/{cat['zero']}.fits"
    cov = B.load_cov(targets)
    sigma_x = jnp.asarray(
        np.asarray(fitsio.read(targets)["cov_odd"], dtype=np.float64)[0])
    n = 1 << a.log2_draws
    print(f"flow {a.flow}   pop {a.pop}   {n} draws (2^{a.log2_draws})")
    print(f"window size {a.window_size}  flux {a.window_flux}\n", flush=True)

    # bias.py's construction exactly -- centroid flows are standardised on the
    # FULL population, so no 0.9 slice, and jr.key(seed + 31) is its draw key.
    m_train = shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    flow = bulk.SupportedFlow(flow, eps=0.0, support=True)

    zs = flow.base_dist.sample(jr.key(a.seed + 31), (n,)).astype(jnp.float64)
    tr = eqx.filter_jit(jax.vmap(lambda z1: flow.bijection.transform(
        z1, B.condition(jnp.zeros(2), sigma_x))))
    m0 = np.concatenate([np.asarray(tr(zs[i:i + 16384]))
                         for i in range(0, n, 16384)])
    del zs

    zero, e0, e1 = jnp.zeros(2), jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])

    def one(m_i):
        f = lambda g: flow.log_prob(m_i, condition=B.condition(g, sigma_x))
        vg = jax.value_and_grad(f)
        (_, q), (_, h0) = jax.jvp(vg, (zero,), (e0,))
        _, (_, h1) = jax.jvp(vg, (zero,), (e1,))
        return q, jnp.stack([h0, h1], axis=-1)

    chunked = eqx.filter_jit(jax.vmap(one))
    fprob = eqx.filter_jit(
        lambda mm: B.window_prob(mm, cov, a.window_size, a.window_flux))

    total = 0.0          # sum of F (R11 + Q1^2) over kept draws
    kept = 0
    pool_v, pool_m, pool_rq = [], [], []   # top order statistics so far
    for i in range(0, n, a.batch):
        mm = jnp.asarray(m0[i:i + a.batch])
        q, r = chunked(mm)
        F = fprob(mm)
        ok = (jnp.isfinite(q).all(-1) & jnp.isfinite(r).all(-1).all(-1)
              & jnp.isfinite(F))
        Fk, qk, rk = np.asarray(F[ok]), np.asarray(q[ok]), np.asarray(r[ok])
        if not len(Fk):
            continue
        r11 = Fk * rk[:, 0, 0]
        q11 = Fk * qk[:, 0] ** 2
        v = r11 + q11
        total += v.sum()
        kept += len(v)
        # Retain only the largest |v| seen so far; that is all the census needs.
        j = np.argsort(np.abs(v))[::-1][:KEEP]
        pool_v.append(v[j]); pool_rq.append(np.stack([r11[j], q11[j]], 1))
        pool_m.append(np.asarray(mm[ok])[j])
        if len(pool_v) > 64:
            pool_v, pool_m, pool_rq = _trim(pool_v, pool_m, pool_rq)
    pool_v, pool_m, pool_rq = _trim(pool_v, pool_m, pool_rq)
    v, mtop, rq = pool_v[0], pool_m[0], pool_rq[0]

    rs11 = total / kept
    print(f"kept {kept}/{n} draws   R_s11 = {rs11:+.6f}\n")

    print("  drop-k          R_s11      change     top-k share")
    run = 0.0
    for k in (0, 1, 3, 10, 30, 100, 300, 1000):
        run = v[:k].sum()
        val = (total - run) / (kept - k)
        print(f"  {k:5d}      {val:+.6f}   {val - rs11:+.6f}      "
              f"{abs(run) / abs(total):7.2%}")

    print(f"\n  Hill tail index over |F (R11 + Q1^2)|:")
    av = np.abs(v)
    for k in (50, 200, 1000, 3000):
        if k < len(av):
            print(f"    k = {k:5d}   alpha = {hill(av, k):.2f}")
    print("    (alpha < 2: infinite variance;  alpha < 1: infinite mean)")

    print(f"\n  the 10 leading draws "
          f"(POINT_SOURCE = {bulk.POINT_SOURCE:.4f}, "
          f"POINT_SOURCE_MC = {bulk.POINT_SOURCE_MC:.4f}):")
    print(f"    {'F(R11+Q1^2)':>13} {'F R11':>12} {'F Q1^2':>12} "
          f"{'Mf':>10} {'Mr/Mf':>8} {'Mc/Mr':>8}")
    for i in range(min(10, len(v))):
        m = mtop[i]
        print(f"    {v[i]:>13.4g} {rq[i, 0]:>12.4g} {rq[i, 1]:>12.4g} "
              f"{m[0]:>10.4g} {m[1] / m[0]:>8.4f} {m[4] / m[1]:>8.4f}")

    frac_q = np.abs(rq[:100, 1]).sum() / np.abs(rq[:100]).sum()
    print(f"\n  of the top 100 draws' |contribution|, "
          f"{frac_q:.0%} is Q1^2 (score) and {1 - frac_q:.0%} is R11 (curvature)")

    if a.save:
        np.savez(a.save, moments=mtop, v=v, r11=rq[:, 0], q1sq=rq[:, 1],
                 rs11=rs11, kept=kept, n=n)
        print(f"\n  wrote {a.save}")


def _trim(pv, pm, prq):
    v, m, rq = np.concatenate(pv), np.concatenate(pm), np.concatenate(prq)
    j = np.argsort(np.abs(v))[::-1][:KEEP]
    return [v[j]], [m[j]], [rq[j]]


if __name__ == "__main__":
    main()
