"""Does a Lipschitz-ratio threshold (`lipschitz_check.py`'s per-draw Q/R
sensitivity to its nearest real template, in shape-invariant space) catch
the leaders that BOTH density guards (`bias.build_support_density` and
`bend_density_guard_check.py`'s bend-conditioner variant) missed?

Runs the same top-25-by-|contribution| census as `bend_density_guard_check.py`,
computes each leader's Lipschitz ratio against its nearest real template
(same recipe as `lipschitz_check.py`), and compares to the real-to-real
baseline distribution in the same Mr/Mf band -- i.e. a threshold set at the
real-to-real max would flag a leader iff its ratio exceeds every ordinary
pair of adjacent real templates.

Run: python dev/lipschitz_guard_check.py --log2-draws 20
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
from dev.bend_density_guard_check import build_flow_and_data  # noqa: E402

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
    cat = B.CATALOGS[a.pop]
    cov = B.load_cov(f"{a.data_dir}/{cat['zero']}.fits")

    # --- census: same as bend_density_guard_check.py ---
    n = 1 << a.log2_draws
    print(f"{n} draws (2^{a.log2_draws})")
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
        (_, q), lin = jax.linearize(vg, zero)
        _, h0 = lin(e0)
        _, h1 = lin(e1)
        return q, jnp.stack([h0, h1], axis=-1)

    one_jit = eqx.filter_jit(one)
    chunked = eqx.filter_jit(jax.vmap(one))
    fprob = eqx.filter_jit(
        lambda mm: B.window_prob(mm, cov, a.window_size, a.window_flux))

    pool_v, pool_m = [], []
    for i in range(0, n, a.batch):
        mm = jnp.asarray(m0[i:i + a.batch])
        q, r = chunked(mm)
        F = fprob(mm)
        ok = jnp.isfinite(q).all(-1) & jnp.isfinite(r).all(-1).all(-1) & jnp.isfinite(F)
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

    # --- Lipschitz ratio, same recipe as lipschitz_check.py ---
    def qr_vec(m_i):
        q, r = one_jit(jnp.asarray(m_i))
        return np.concatenate([np.asarray(q).ravel(), np.asarray(r).ravel()])

    inv = B._shape_invariants(m_train)
    ok = np.isfinite(inv).all(-1)
    inv, m_train_ok = inv[ok], m_train[ok]
    scale = inv.std(0)
    tree = cKDTree(inv / scale)

    def lipschitz_to_nearest_real(m_i):
        inv_i = B._shape_invariants(m_i[None, :])[0] / scale
        d, j = tree.query(inv_i, k=1)
        dq = np.linalg.norm(qr_vec(m_i) - qr_vec(m_train_ok[j]))
        return dq / d if d > 0 else np.inf

    lip_leaders = np.array([lipschitz_to_nearest_real(m) for m in leaders])

    ratio = m_train_ok[:, 1] / m_train_ok[:, 0]
    band = (ratio > 2.7) & (ratio < 3.5)
    idx_band = np.where(band)[0]
    rng = np.random.default_rng(0)
    sample = rng.choice(idx_band, size=min(N_REAL_SAMPLE, len(idx_band)),
                        replace=False)
    lips_real = []
    for i in sample:
        d2, j2 = tree.query(inv[i] / scale, k=2)
        d, j = d2[1], j2[1]
        dq = np.linalg.norm(qr_vec(m_train_ok[i]) - qr_vec(m_train_ok[j]))
        lips_real.append(dq / d)
    lips_real = np.array(lips_real)
    real_max = lips_real.max()
    real_p99 = np.percentile(lips_real, 99)

    print(f"\nreal-to-real baseline ({len(lips_real)} pairs): "
          f"p50={np.percentile(lips_real,50):.4g} p99={real_p99:.4g} max={real_max:.4g}")

    print(f"\n{'rank':>4} {'|contrib|':>12} {'Mr/Mf':>8} {'lipschitz':>12} "
          f"{'x real p99':>11} {'x real max':>11} {'flagged@max':>12}")
    for rank, (m, lip) in enumerate(zip(leaders, lip_leaders)):
        flagged = lip > real_max
        print(f"{rank:>4} {abs(v[order[rank]]):>12.4g} {m[1]/m[0]:>8.4f} "
              f"{lip:>12.4g} {lip/real_p99:>10.2f}x {lip/real_max:>10.2f}x "
              f"{'FLAGGED' if flagged else 'missed':>12}")

    n_flag_max = int((lip_leaders > real_max).sum())
    n_flag_p99 = int((lip_leaders > real_p99).sum())
    print(f"\nthreshold = real-to-real MAX: flags {n_flag_max}/{TOPN} leaders")
    print(f"threshold = real-to-real p99: flags {n_flag_p99}/{TOPN} leaders")


if __name__ == "__main__":
    main()
