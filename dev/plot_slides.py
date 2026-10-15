"""The first three 'BFD cNF Plots' slides, remade from the current desv chain (2026-10-15).

Ported from the pre-fresh-start bfd_cnf/plot_corner.py and plot_shear_derivs{,_diff}.py (commit 19750cc):
  1. corner: the bdg2d noisy templates vs the flow's template-level prior (shear stage at g = 0; bulk and shear
     are frozen in the centroid stage, so this IS the shipped chain's prior), with a Mc Mf / Mr^2 axis;
  2. P and its g-derivatives on the M1/Mr-M2/Mr plane at fixed Mf, Mr/Mf, Mc, from the full centroid chain at
     the desv targets' mean Sigma_X;
  3. the same panels at the PSF e1 +0.05 catalog's mean Sigma_X minus the baseline's.
P is the conditional density of (M1/Mr, M2/Mr) on the plotted slice: p(m) over its integral on the plane at g = 0.
Usage: python dev/plot_slides.py [--tag x6r] [--out plots/slides]
"""
import argparse
import os
import sys

import fitsio
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
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

D = "../bfd_cnf_imsims/data/"
TEMPLATES = D + "moments_bulgedisc_g2_bdg2d.fits"
TARGETS = D + "targets_bdg2_g0_1m_desv.fits"
TARGETS_PSFE = D + "targets_bdg2_g0_1m_desv_psfe1p05.fits"
CUT, WIN_FLUX, WIN_SIZE = (1200.0, 1.5), (2200.0, 90000.0), (2.2, 3.5)
LABELS = [r"$\log_{10}M_f$", r"$M_r/M_f$", r"$M_1/M_r$", r"$M_2/M_r$", r"$M_c M_f/M_r^2$"]
RANGE = [(2.5, 6.0), (1.0, 4.2), (-0.8, 0.8), (-0.8, 0.8), (1.5, 2.6)]
# Point-source locus (log Mf free): a star's deconvolved image is flat in k, so these depend only on the
# weight -- KBlackmanHarris(0.65) through imsims.sim._measure, identical at PSF sigma 0.36/0.40/0.44 and e1 0.05.
STAR = {1: 3.69257, 2: 0.0, 3: 0.0, 4: 1.80418}
KEYS = ["p", "dg1", "dg2", "dg1dg1", "dg2dg2", "dg1dg2"]
DLAB = [r"P", r"\partial P/\partial g_1", r"\partial P/\partial g_2", r"\partial^2 P/\partial g_1^2",
        r"\partial^2 P/\partial g_2^2", r"\partial^2 P/\partial g_1\partial g_2"]


def load(path, m, centroid):
    jax.config.update("jax_enable_x64", False)   # x64 at build time breaks the deserialise
    fl = bulk.load_flow(path, m[:20000], key=jr.key(0), shear=True, centroid=centroid)
    jax.config.update("jax_enable_x64", True)
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if isinstance(x, jax.Array)
                                  and jnp.issubdtype(x.dtype, jnp.floating) else x, fl)


def coords(m):
    c = np.column_stack([np.log10(m[:, 0]), m[:, 1] / m[:, 0], m[:, 2] / m[:, 1], m[:, 3] / m[:, 1],
                         m[:, 4] * m[:, 0] / m[:, 1] ** 2])
    return c[np.all(np.isfinite(c), axis=1) & (m[:, 0] > 0)]


def mean_sigma_x(path):
    return np.asarray(fitsio.read(path, columns=["cov_odd"])["cov_odd"], np.float64).mean(0)


def overlay(fig):
    """Target window (green) and the template cut (grey dashed) on the 5x5 corner grid."""
    ax = np.array(fig.axes).reshape(5, 5)
    lf, (s0, s1) = np.log10(WIN_FLUX), WIN_SIZE
    win = {0: lf, 1: (s0, s1)}
    cut = {0: [np.log10(CUT[0])], 1: [CUT[1]]}
    for j in range(5):
        for v in win.get(j, ()):
            ax[j, j].axvline(v, c="tab:green", lw=1)
        for v in cut.get(j, ()):
            ax[j, j].axvline(v, c="grey", ls="--", lw=1)
        for i in range(j + 1, 5):
            for v in cut.get(j, ()):
                ax[i, j].axvline(v, c="grey", ls="--", lw=1)
            for v in cut.get(i, ()):
                ax[i, j].axhline(v, c="grey", ls="--", lw=1)
            if j in win and i in win:
                ax[i, j].add_patch(mpatches.Rectangle((lf[0], s0), lf[1] - lf[0], s1 - s0, fill=False,
                                                      ec="tab:green", lw=1, zorder=100))
            elif j in win:
                for v in win[j]:
                    ax[i, j].axvline(v, c="tab:green", lw=1)
    for j, v in STAR.items():
        ax[j, j].axvline(v, c="tab:red", lw=1)
        for i in range(j + 1, 5):
            ax[i, j].axvline(v, c="tab:red", lw=1, zorder=500)
        for k in range(j):
            ax[j, k].axhline(v, c="tab:red", lw=1, zorder=500)
    ax[0, 0].plot([], [], c="tab:red", label="Stars")
    ax[0, 0].plot([], [], c="tab:green", label="Target window")
    ax[0, 0].plot([], [], c="grey", ls="--", label=r"Template cut ($M_f\geq1200$, $M_r/M_f\geq1.5$)")


def plot_corner(tag, out, n, cut):
    m = np.asarray(shear.load(TEMPLATES)[0], np.float64)
    fl = load(f"flows/desm/shear_{tag}.eqx", m, False)
    f = jax.jit(jax.vmap(lambda z: fl.bijection.transform(z, jnp.zeros(2))))
    z = fl.base_dist.sample(jr.key(1), (n,)).astype(jnp.float64)
    x = np.concatenate([np.asarray(f(z[i:i + (1 << 17)])) for i in range(0, n, 1 << 17)])
    rng = np.random.default_rng(0)
    if cut:     # the flow is censored below the cut: compare where it is trained to match, same cut on both
        m, x = m[bulk.selected(m, CUT)], x[bulk.selected(x, CUT)]
    obs = coords(m[rng.choice(len(m), min(n, len(m)), replace=False)])
    flo = coords(x)
    k = min(len(obs), len(flo))
    obs, flo = obs[:k], flo[:k]
    print(f"corner: {k:,} templates vs {k:,} flow draws ({len(x) - len(coords(x)):,} non-finite/Mf<=0 draws dropped)")
    fig = plt.figure(figsize=(20, 20))
    kw = dict(labels=LABELS, bins=200, range=RANGE, plot_density=False, fill_contours=False, smooth=2.0,
              smooth1d=1.0, label_kwargs={"fontsize": 30}, fig=fig)
    corner.corner(obs, color="tab:blue", plot_datapoints=True, **kw)
    corner.corner(flo, color="tab:orange", plot_datapoints=False, **kw)
    overlay(fig)
    ax = fig.axes
    ax[0].plot([], [], c="tab:blue", label=f"Templates (bdg2d, noisy, {'cut' if cut else 'uncut'})")
    ax[0].plot([], [], c="tab:orange", label=f"Prior flow ({tag}, g = 0{', same cut' if cut else ''})")
    ax[0].legend(bbox_to_anchor=(1.3, 1.0), loc="upper left", fontsize=18)
    for a in ax:
        a.tick_params(labelsize=18)
    p = os.path.join(out, f"corner_templates_vs_flow_{tag}{'_cut' if cut else '_uncut'}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def planes(fl, sx, mf, mrmf, mc_ratio, n=121, r=0.3):
    e = np.linspace(-r, r, n)
    E1, E2 = np.meshgrid(e, e)              # rows = M2/Mr, cols = M1/Mr
    mr = mrmf * mf
    m = np.column_stack([np.full(E1.size, mf), np.full(E1.size, mr), E1.ravel() * mr, E2.ravel() * mr,
                         np.full(E1.size, mc_ratio * mr ** 2 / mf)])
    sx = jnp.asarray(sx)

    def prob(mi, g):
        return jnp.exp(fl.log_prob(mi, condition=B.condition(g, sx)))

    g0 = jnp.zeros(2)
    p = jax.jit(jax.vmap(prob, (0, None)))
    d1 = jax.jit(jax.vmap(jax.grad(prob, 1), (0, None)))
    d2 = jax.jit(jax.vmap(jax.hessian(prob, 1), (0, None)))
    mj = jnp.asarray(m)
    P, G, H = (np.asarray(f(mj, g0)) for f in (p, d1, d2))
    norm = P.sum() * (e[1] - e[0]) ** 2      # p(e) = p(m) Mr^2, so P / its integral over (M1/Mr, M2/Mr)
    out = {"p": P, "dg1": G[:, 0], "dg2": G[:, 1], "dg1dg1": H[:, 0, 0], "dg2dg2": H[:, 1, 1],
           "dg1dg2": H[:, 0, 1]}
    return {k: v.reshape(n, n) / norm for k, v in out.items()}, e


def panels(pl, e, title, path, delta=False):
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    ext = (e[0], e[-1], e[0], e[-1])
    for a, k, lab in zip(ax.ravel(), KEYS, DLAB):
        v = np.abs(pl[k]).max() + 1e-300
        seq = k == "p" and not delta
        im = a.imshow(pl[k], origin="lower", extent=ext, interpolation="none", aspect="auto",
                      cmap="Blues" if seq else "RdBu", vmin=0 if seq else -v, vmax=v)
        a.set_xlabel(r"$M_1/M_r$", fontsize=18)
        a.set_ylabel(r"$M_2/M_r$", fontsize=18)
        a.axvline(0, c="grey", lw=1)
        a.axhline(0, c="grey", lw=1)
        plt.colorbar(im, ax=a).set_label(rf"$\Delta {lab}$" if delta else rf"${lab}$", size=18)
    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_derivs(tag, out, mf, mrmf):
    m = np.asarray(shear.load(TEMPLATES)[0], np.float64)
    fl = load(f"flows/desm/centroid_tnc15c_{tag}.eqx", m, True)
    c = coords(m)
    near = (np.abs(10 ** c[:, 0] / mf - 1) < 0.1) & (np.abs(c[:, 1] - mrmf) < 0.1)
    mc_ratio = float(np.median(c[near, 4]))      # Mc fixed at the templates' median Mc Mf/Mr^2 on the slice
    sx0, sx1 = mean_sigma_x(TARGETS), mean_sigma_x(TARGETS_PSFE)
    e_psf = lambda s: ((s[0] - s[2]) / (s[0] + s[2]), 2 * s[1] / (s[0] + s[2]))
    print(f"slice Mf {mf:g}, Mr/Mf {mrmf:g}, Mc Mf/Mr^2 {mc_ratio:.3f} ({near.sum()} templates); "
          f"Sigma_X base {sx0} (e {e_psf(sx0)}), psfe {sx1} (e {e_psf(sx1)})")
    slab = (rf"$\log_{{10}}M_f={np.log10(mf):.2f}$ ($M_f={mf:g}$), $M_r/M_f={mrmf:g}$, "
            rf"$M_cM_f/M_r^2={mc_ratio:.2f}$, $g=(0,0)$")
    p0, e = planes(fl, sx0, mf, mrmf, mc_ratio)
    panels(p0, e, rf"{tag}: {slab}, $\Sigma_X$ = desv targets' mean",
           os.path.join(out, f"shear_derivs_{tag}.png"))
    p1, _ = planes(fl, sx1, mf, mrmf, mc_ratio)
    panels({k: p1[k] - p0[k] for k in KEYS}, e,
           rf"{tag}: PSF $e_1$ +0.05 minus baseline ($\Sigma_X$ of each catalog), {slab}",
           os.path.join(out, f"shear_derivs_diff_psfe1p05_{tag}.png"), delta=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="x6r")
    ap.add_argument("--out", default="plots/slides")
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--mf", type=float, default=2000.0)
    ap.add_argument("--mrmf", type=float, default=3.3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plot_derivs(a.tag, a.out, a.mf, a.mrmf)
    for cut in (True, False):
        plot_corner(a.tag, a.out, a.n, cut)
