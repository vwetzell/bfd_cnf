"""Build the bend-layer-conditioner density guard `bend_conditioner_density_check.py`
proposed (a `build_support_density`-style global k-NN cutoff, but in the bend
net's own 4-d input space `[z0,z1,z2, log1p(|e|^2)-W0]` instead of the 5-d
shape-invariant space `bias.build_support_density` uses) and check it against
the same top-25-by-contribution leader census `leader_census_logp.py` used,
comparing catch rate to the existing shape-invariant guard.

Run: python dev/bend_density_guard_check.py --log2-draws 20
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

TOPN = 25
_W0 = 1.0986122886681098


def build_flow_and_data(a):
    m_train = shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=False)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    return flow, np.asarray(m_train, dtype=np.float64)


def make_conditioner_fn(bijections):
    def conditioner(m_i):
        x = jnp.asarray(m_i)
        for b in bijections[:2]:
            c = jnp.zeros(2) if b.cond_shape is not None else None
            x = b.transform_and_log_det(x, c)[0]
        x = bijections[2].spin0.transform_and_log_det(x)[0]
        q = x[3] ** 2 + x[4] ** 2
        return jnp.concatenate([x[:3], (jnp.log1p(q) - _W0)[None]])
    return conditioner


def build_bend_density(cond_fn, m_train, n_max=100_000, seed=0, k=10):
    """Same global-cutoff k-NN recipe as `bias.build_support_density`, over
    the bend net's own 4-d conditioner space instead of shape invariants."""
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(m_train), n_max, replace=False)
           if len(m_train) > n_max else np.arange(len(m_train)))
    batched = eqx.filter_jit(jax.vmap(cond_fn))
    U = np.concatenate([np.asarray(batched(jnp.asarray(m_train[idx[i:i + 16384]])))
                        for i in range(0, len(idx), 16384)])
    ok = np.isfinite(U).all(-1)
    U = U[ok]
    scale = U.std(0)
    tree = cKDTree(U / scale)
    self_dist, _ = tree.query(U / scale, k=k + 1)
    cutoff = self_dist[:, k].max()
    print(f"  bend-conditioner density: {len(U)} templates, {k}-NN spacing "
          f"median {np.median(self_dist[:, k]):.3g}, cutoff {cutoff:.3g}")
    return tree, scale, cutoff, k


def in_support_bend(m_batch, cond_fn, density):
    tree, scale, cutoff, k = density
    batched = eqx.filter_jit(jax.vmap(cond_fn))
    U = np.concatenate([np.asarray(batched(jnp.asarray(m_batch[i:i + 16384])))
                        for i in range(0, len(m_batch), 16384)])
    dist, _ = tree.query(U / scale, k=k)
    return dist[:, -1] <= cutoff


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

    bend_density = build_bend_density(cond_fn, m_train)
    shape_density = B.build_support_density(m_train)

    cat = B.CATALOGS[a.pop]
    cov = B.load_cov(f"{a.data_dir}/{cat['zero']}.fits")

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

    caught_shape = B.in_support_density(leaders, shape_density)
    caught_bend = in_support_bend(leaders, cond_fn, bend_density)

    print(f"\n{'rank':>4} {'|contrib|':>12} {'Mr/Mf':>8} {'shape-guard':>12} "
          f"{'bend-guard':>11}")
    for rank, (m, cs, cb) in enumerate(zip(leaders, caught_shape, caught_bend)):
        print(f"{rank:>4} {abs(v[order[rank]]):>12.4g} {m[1]/m[0]:>8.4f} "
              f"{'IN-SUPPORT' if cs else 'REJECTED':>12} "
              f"{'IN-SUPPORT' if cb else 'REJECTED':>11}")

    n_rej_shape = int((~caught_shape).sum())
    n_rej_bend = int((~caught_bend).sum())
    print(f"\nshape-invariant guard rejects {n_rej_shape}/{TOPN} top leaders")
    print(f"bend-conditioner guard rejects {n_rej_bend}/{TOPN} top leaders")


if __name__ == "__main__":
    main()
