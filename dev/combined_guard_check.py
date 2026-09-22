"""Which single guard, or combination, best separates the top-25 R_s leader
census from real templates: shape-invariant density (bias.build_support_density),
bend-conditioner density (bend_density_guard_check.py), Lipschitz ratio
(lipschitz_check.py/lipschitz_guard_check.py), or local log_prob percentile
(leader_logp_check.py)?

Runs ONE top-25-by-|contribution| census (same recipe/seed as the other guard
scripts) and scores every leader on all four metrics plus two combined scores
(log_prob-percentile x Lipschitz-ratio, and bend-density-ratio x Lipschitz-ratio),
each calibrated against the real-to-real / real-catalog baseline in-band.

Run: python dev/combined_guard_check.py --log2-draws 20
"""
import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402
from dev.bend_density_guard_check import (build_flow_and_data,  # noqa: E402
                                           build_bend_density,
                                           make_conditioner_fn)

TOPN = 25
N_REAL_SAMPLE = 400


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pop", default="gauss2_v3d")
    p.add_argument("--flow", default="flows/shear_g2v3d_full2.eqx")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--log2-draws", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(2500.0, 50000.0))
    a = p.parse_args()

    flow, m_train = build_flow_and_data(a)
    bijections = flow.bijection.bijection.bijections
    cond_fn = make_conditioner_fn(bijections)
    cat = B.CATALOGS[a.pop]
    cov = B.load_cov(f"{a.data_dir}/{cat['zero']}.fits")

    shape_density = B.build_support_density(m_train)
    bend_density = build_bend_density(cond_fn, m_train)
    bend_tree, bend_scale, bend_cutoff, bend_k = bend_density

    zero, e0, e1 = jnp.zeros(2), jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])

    def one(m_i):
        f = lambda g: flow.log_prob(m_i, condition=B.condition(g, None))
        vg = jax.value_and_grad(f)
        (lp, q), lin = jax.linearize(vg, zero)
        _, h0 = lin(e0)
        _, h1 = lin(e1)
        return lp, q, jnp.stack([h0, h1], axis=-1)

    one_jit = eqx.filter_jit(one)
    chunked = eqx.filter_jit(jax.vmap(one))
    fprob = eqx.filter_jit(
        lambda mm: B.window_prob(mm, cov, a.window_size, a.window_flux))

    # --- census ---
    n = 1 << a.log2_draws
    print(f"{n} draws (2^{a.log2_draws})")
    zs = flow.base_dist.sample(jr.key(a.seed + 31), (n,)).astype(jnp.float64)
    tr = eqx.filter_jit(jax.vmap(lambda z1: flow.bijection.transform(
        z1, B.condition(jnp.zeros(2), None))))
    m0 = np.concatenate([np.asarray(tr(zs[i:i + 16384]))
                         for i in range(0, n, 16384)])
    del zs

    pool_v, pool_m = [], []
    for i in range(0, n, a.batch):
        mm = jnp.asarray(m0[i:i + a.batch])
        lp, q, r = chunked(mm)
        F = fprob(mm)
        ok = (jnp.isfinite(q).all(-1) & jnp.isfinite(r).all(-1).all(-1)
              & jnp.isfinite(F) & jnp.isfinite(lp))
        Fk = np.asarray(F[ok])
        if not len(Fk):
            continue
        qk, rk = np.asarray(q[ok]), np.asarray(r[ok])
        v = Fk * rk[:, 0, 0] + Fk * qk[:, 0] ** 2
        j = np.argsort(np.abs(v))[::-1][:512]
        pool_v.append(v[j])
        pool_m.append(np.asarray(mm[ok])[j])

    v = np.concatenate(pool_v)
    mtop = np.concatenate(pool_m)
    order = np.argsort(np.abs(v))[::-1][:TOPN]
    leaders = mtop[order]
    contrib = np.abs(v[order])

    # --- per-leader metrics ---
    inv = B._shape_invariants(m_train)
    ok_inv = np.isfinite(inv).all(-1)
    inv, m_train_ok = inv[ok_inv], m_train[ok_inv]
    scale = inv.std(0)
    tree = cKDTree(inv / scale)

    def qr_vec(m_i):
        _, q, r = one_jit(jnp.asarray(m_i))
        return np.concatenate([np.asarray(q).ravel(), np.asarray(r).ravel()])

    def lipschitz_to_nearest_real(m_i):
        inv_i = B._shape_invariants(m_i[None, :])[0] / scale
        d, j = tree.query(inv_i, k=1)
        dq = np.linalg.norm(qr_vec(m_i) - qr_vec(m_train_ok[j]))
        return dq / d if d > 0 else np.inf

    lp_fn = eqx.filter_jit(jax.vmap(
        lambda m: flow.log_prob(m, condition=B.condition(jnp.zeros(2), None))))
    m_train64 = np.asarray(m_train, dtype=np.float64)
    lp_train = np.concatenate([
        np.asarray(lp_fn(jnp.asarray(m_train64[i:i + 16384])))
        for i in range(0, len(m_train64), 16384)])
    finite = np.isfinite(lp_train)
    lp_train, m_train64 = lp_train[finite], m_train64[finite]
    ratio_train = m_train64[:, 1] / m_train64[:, 0]

    lp_leaders = np.asarray(lp_fn(jnp.asarray(leaders)))
    lip_leaders = np.array([lipschitz_to_nearest_real(m) for m in leaders])
    U_leaders = np.asarray(eqx.filter_jit(jax.vmap(cond_fn))(jnp.asarray(leaders)))
    bend_dist_leaders, _ = bend_tree.query(U_leaders / bend_scale, k=bend_k)
    bend_ratio_leaders = bend_dist_leaders[:, -1] / bend_cutoff
    caught_shape = ~B.in_support_density(leaders, shape_density)

    # local log_prob percentile per leader (band = +-5% in Mr/Mf)
    pct_leaders = np.full(TOPN, np.nan)
    for i, m in enumerate(leaders):
        r_ratio = m[1] / m[0]
        near = np.abs(ratio_train - r_ratio) < 0.05 * r_ratio
        if near.sum() >= 20:
            pct_leaders[i] = 100.0 * np.mean(lp_train[near] < lp_leaders[i])

    # real-to-real baselines, restricted to the leaders' Mr/Mf band (2.7-3.5)
    ratio = m_train_ok[:, 1] / m_train_ok[:, 0]
    band = (ratio > 2.7) & (ratio < 3.5)
    idx_band = np.where(band)[0]
    rng = np.random.default_rng(0)
    sample = rng.choice(idx_band, size=min(N_REAL_SAMPLE, len(idx_band)), replace=False)
    lips_real = []
    for i in sample:
        d2, j2 = tree.query(inv[i] / scale, k=2)
        d, j = d2[1], j2[1]
        dq = np.linalg.norm(qr_vec(m_train_ok[i]) - qr_vec(m_train_ok[j]))
        lips_real.append(dq / d)
    lips_real = np.array(lips_real)
    real_p99 = np.percentile(lips_real, 99)
    real_max = lips_real.max()

    # --- single-metric catch rates ---
    catch_shape = caught_shape
    catch_bend = bend_ratio_leaders > 1.0
    catch_lip_max = lip_leaders > real_max
    catch_lip_p99 = lip_leaders > real_p99
    catch_logp5 = pct_leaders < 5.0  # bottom-5%-locally rule

    # --- combined scores ---
    # normalize each leader's lipschitz ratio by real_p99 (>1 = anomalous)
    lip_norm = lip_leaders / real_p99
    logp_anom = np.nan_to_num(100.0 - pct_leaders, nan=0.0) / 100.0  # 0..1, higher=more anomalous
    bend_norm = bend_ratio_leaders  # >1 = outside cutoff

    combo_logp_lip = logp_anom * np.log1p(lip_norm)
    combo_bend_lip = bend_norm * np.log1p(lip_norm)
    # OR-combination: flagged by any of the three single guards
    catch_or = catch_shape | catch_bend | catch_lip_p99 | catch_logp5

    print(f"\nreal-to-real Lipschitz baseline (n={len(lips_real)}): "
          f"p99={real_p99:.4g} max={real_max:.4g}")

    hdr = (f"{'rank':>4} {'contrib':>10} {'Mr/Mf':>7} {'shape':>7} {'bend':>7} "
           f"{'lip@p99':>8} {'lip@max':>8} {'logp<5%':>8} {'OR-any':>7} "
           f"{'combo_lp*lip':>13} {'combo_bd*lip':>13}")
    print(hdr)
    for i in range(TOPN):
        print(f"{i:>4} {contrib[i]:>10.3g} {leaders[i,1]/leaders[i,0]:>7.4f} "
              f"{'REJ' if catch_shape[i] else '.':>7} "
              f"{'REJ' if catch_bend[i] else '.':>7} "
              f"{'CATCH' if catch_lip_p99[i] else '.':>8} "
              f"{'CATCH' if catch_lip_max[i] else '.':>8} "
              f"{'CATCH' if catch_logp5[i] else '.':>8} "
              f"{'CATCH' if catch_or[i] else '.':>7} "
              f"{combo_logp_lip[i]:>13.3f} {combo_bend_lip[i]:>13.3f}")

    print(f"\ncatch rate out of {TOPN} leaders:")
    print(f"  shape-invariant density guard:      {catch_shape.sum()}/{TOPN}")
    print(f"  bend-conditioner density guard:     {catch_bend.sum()}/{TOPN}")
    print(f"  Lipschitz @ real p99:                {catch_lip_p99.sum()}/{TOPN}")
    print(f"  Lipschitz @ real max:                {catch_lip_max.sum()}/{TOPN}")
    print(f"  local log_prob bottom-5%:            {catch_logp5.sum()}/{TOPN}")
    print(f"  OR of (shape, bend, lip@p99, logp5):  {catch_or.sum()}/{TOPN}")
    print(f"\ncombined scores are continuous (no hard cutoff calibrated here); "
          f"compare their rank order to |contrib| by eye above -- a good "
          f"combined guard should assign its highest scores to the leaders "
          f"with the largest |contrib|, not just count catches.")


if __name__ == "__main__":
    main()
