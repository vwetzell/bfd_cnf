"""Corner plot of the gauss2_v3d template population's 5 moments
(Mf, Mr, M1, M2, Mc), with the top-10 R_s-leader draws (same
top-|contribution| census as leader_census_logp.py) overlaid as points.

Run: python dev/leader_corner_plot.py --log2-draws 20 --out leader_corner.png
"""
import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import corner  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

TOPN = 10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pop", default="gauss2_v4")
    p.add_argument("--flow", default="flows/shear_g2v4_ema1e3.eqx")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--log2-draws", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(3000.0, 20000.0))
    p.add_argument("--out", default="dev/leader_corner.png")
    a = p.parse_args()

    m_train = shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=False)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    m_train = np.asarray(m_train, dtype=np.float64)

    cat = B.CATALOGS[a.pop]
    cov = B.load_cov(f"{a.data_dir}/{cat['zero']}.fits")

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
    print("top-10 leaders (Mf, Mr, M1, M2, Mc, |contrib|):")
    for m, vv in zip(leaders, v[order]):
        print(f"  {m}  |contrib|={abs(vv):.4g}")

    # Same chart bulk.py's own `corner` mode plots in: to_coords is
    # RawMomentStandardize's forward transform (log10 Mf, Mr/Mf, Mc/Mr,
    # M1/Mr, M2/Mr), so this overlay sits in the flow's own working space.
    d = bulk.to_coords(m_train)
    d = d[np.isfinite(d).all(-1)]
    leaders_t = bulk.to_coords(leaders)

    plot_range = [np.percentile(d[:, i], [0.05, 99.95]) for i in range(5)]
    for i in (0, 1, 2):
        plot_range[i] += 0.25 * np.ptp(plot_range[i]) * np.array([-1.0, 1.0])
    plot_range = [tuple(r) for r in plot_range]
    style = dict(labels=bulk.COORD_LABELS, bins=500, range=plot_range, smooth=5.0,
                 plot_density=False, plot_contours=True, fill_contours=False,
                 label_kwargs={"fontsize": bulk.LABEL_FONTSIZE})

    fig = plt.figure(figsize=(16, 16))
    corner.corner(d, fig=fig, color="tab:blue", plot_datapoints=True, smooth1d=1.0,
                  hist_kwargs={"label": f"{a.pop} templates"}, **style)
    corner.overplot_points(fig, leaders_t, marker="o", color="red", markersize=4,
                            zorder=600, label=f"top-{TOPN} R_s leaders")

    n_c = len(bulk.COORD_LABELS)
    axs = fig.axes

    def mark(coord, value, color="tab:orange", **kw):
        for row in range(n_c):
            for col in range(row + 1):
                ax = axs[row * n_c + col]
                if col == coord:
                    ax.axvline(value, lw=1, color=color, zorder=500, **kw)
                    kw = {}
                elif row == coord and row != col:
                    ax.axhline(value, lw=1, color=color, zorder=500)

    mark(1, bulk.POINT_SOURCE, label="Point source")
    mark(2, bulk.POINT_SOURCE_MC - 2.0 * bulk.POINT_SOURCE)
    mark(3, 0.0)
    mark(4, 0.0)
    # Selection window: size is a direct cut on coord 1 (Mr/Mf); flux is a
    # cut on Mf, which coord 0 holds as log10(Mf).
    mark(1, a.window_size[0], color="tab:green", ls="--", label="Size window")
    mark(1, a.window_size[1], color="tab:green", ls="--")
    if np.isfinite(a.window_flux[0]) and a.window_flux[0] > 0:
        mark(0, np.log10(a.window_flux[0]), color="tab:green", ls="--",
             label="Flux window")
    if np.isfinite(a.window_flux[1]):
        mark(0, np.log10(a.window_flux[1]), color="tab:green", ls="--")
    axs[n_c + 1].legend(bbox_to_anchor=(0.0, 1.0), loc="lower left", fontsize=16)
    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=bulk.TICK_LABELSIZE)

    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
