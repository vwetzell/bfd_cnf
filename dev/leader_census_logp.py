"""For the draws that actually drive R_s's instability (top |contribution|
order statistics, same census as dev/rs_census.py), where does each one sit
in the flow's OWN log_prob distribution relative to real templates in its
own Mr/Mf band?

Follow-up to dev/leader_logp_check.py, which found the single worst known
leader at the 2.7th local percentile (16.5 nats below the local median) --
real signal, unlike the moment-space k-NN density test, which saw it as
unremarkably dense. This checks whether that holds across the population of
leaders or was a one-off.

Same setup as last session's rs_hull_leader_check.py: gauss2_v3d,
flows/shear_g2v3d_full2.eqx, centroid=False, sigma_x=None, window
(2.2,3.2)/(2500,50000).

Run: python dev/leader_census_logp.py --log2-draws 22
"""
import argparse
import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402

KEEP = 4096
TOPN = 25


def _trim(pool_v, pool_m, pool_rq):
    v = np.concatenate(pool_v)
    m = np.concatenate(pool_m)
    rq = np.concatenate(pool_rq)
    j = np.argsort(np.abs(v))[::-1][:KEEP]
    return [v[j]], [m[j]], [rq[j]]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pop", default="gauss2_v3d", choices=sorted(B.CATALOGS))
    p.add_argument("--flow", default="flows/shear_g2v3d_full2.eqx")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--log2-draws", type=int, default=22)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(2500.0, 50000.0))
    a = p.parse_args()

    cat = B.CATALOGS[a.pop]
    targets = f"{a.data_dir}/{cat['zero']}.fits"
    cov = B.load_cov(targets)
    n = 1 << a.log2_draws
    print(f"flow {a.flow}   pop {a.pop}   {n} draws (2^{a.log2_draws})")

    m_train = shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=False)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)

    # Real-template log_prob, for the local-percentile lookup below.
    m_train64 = np.asarray(m_train, dtype=np.float64)
    lp_fn = eqx.filter_jit(jax.vmap(
        lambda m: flow.log_prob(m, condition=B.condition(jnp.zeros(2), None))))
    lp_train = np.concatenate([
        np.asarray(lp_fn(jnp.asarray(m_train64[i:i + 16384])))
        for i in range(0, len(m_train64), 16384)])
    finite = np.isfinite(lp_train)
    lp_train, m_train64 = lp_train[finite], m_train64[finite]
    ratio_train = m_train64[:, 1] / m_train64[:, 0]

    zs = flow.base_dist.sample(jr.key(a.seed + 31), (n,)).astype(jnp.float64)
    tr = eqx.filter_jit(jax.vmap(lambda z1: flow.bijection.transform(
        z1, B.condition(jnp.zeros(2), None))))
    m0 = np.concatenate([np.asarray(tr(zs[i:i + 16384]))
                         for i in range(0, n, 16384)])
    del zs

    zero, e0, e1 = jnp.zeros(2), jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])

    def one(m_i):
        f = lambda g: flow.log_prob(m_i, condition=B.condition(g, None))
        vg = jax.value_and_grad(f)
        (lp, q), lin = jax.linearize(vg, zero)
        _, h0 = lin(e0)
        _, h1 = lin(e1)
        return lp, q, jnp.stack([h0, h1], axis=-1)

    chunked = eqx.filter_jit(jax.vmap(one))
    fprob = eqx.filter_jit(
        lambda mm: B.window_prob(mm, cov, a.window_size, a.window_flux))

    total, kept = 0.0, 0
    pool_v, pool_m, pool_rq = [], [], []
    for i in range(0, n, a.batch):
        mm = jnp.asarray(m0[i:i + a.batch])
        lp, q, r = chunked(mm)
        F = fprob(mm)
        ok = (jnp.isfinite(q).all(-1) & jnp.isfinite(r).all(-1).all(-1)
              & jnp.isfinite(F) & jnp.isfinite(lp))
        Fk = np.asarray(F[ok])
        if not len(Fk):
            continue
        qk, rk, lpk = np.asarray(q[ok]), np.asarray(r[ok]), np.asarray(lp[ok])
        r11 = Fk * rk[:, 0, 0]
        q11 = Fk * qk[:, 0] ** 2
        v = r11 + q11
        total += v.sum()
        kept += len(v)
        j = np.argsort(np.abs(v))[::-1][:KEEP]
        pool_v.append(v[j])
        pool_rq.append(lpk[j])   # reuse the rq slot to carry log_prob through
        pool_m.append(np.asarray(mm[ok])[j])
        if len(pool_v) > 64:
            pool_v, pool_m, pool_rq = _trim(pool_v, pool_m, pool_rq)
    pool_v, pool_m, pool_rq = _trim(pool_v, pool_m, pool_rq)
    v, mtop, lptop = pool_v[0], pool_m[0], pool_rq[0]

    rs11 = total / kept
    print(f"kept {kept}/{n} draws   R_s11 = {rs11:+.6f}\n")

    order = np.argsort(np.abs(v))[::-1][:TOPN]
    print(f"{'rank':>4} {'|contrib|':>12} {'Mr/Mf':>8} {'Mc/Mr':>8} "
          f"{'log_prob':>10} {'local med':>10} {'local pct':>10} {'n_near':>7}")
    for rank, j in enumerate(order):
        m = mtop[j]
        r_ratio, c_ratio = m[1] / m[0], m[4] / m[1]
        near = np.abs(ratio_train - r_ratio) < 0.05 * r_ratio
        n_near = int(near.sum())
        if n_near >= 20:
            med = float(np.median(lp_train[near]))
            pct = 100.0 * np.mean(lp_train[near] < lptop[j])
        else:
            med, pct = np.nan, np.nan
        print(f"{rank:>4} {abs(v[j]):>12.4g} {r_ratio:>8.4f} {c_ratio:>8.4f} "
              f"{lptop[j]:>10.3f} {med:>10.3f} {pct:>9.2f}% {n_near:>7}")


if __name__ == "__main__":
    main()
