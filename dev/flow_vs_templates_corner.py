"""Corner plot of the template moments (blue) with the flow's own g=0 draws
overlaid (red contours), in the flow's transformed coordinates.

    python dev/flow_vs_templates_corner.py --flow flows/centroid_g2v4n.eqx
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
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import corner  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--pop", default="gauss2_v4n")
p.add_argument("--flow", default="flows/centroid_g2v4n.eqx")
p.add_argument("--stage", choices=["bulk", "shear", "centroid"], default="centroid")
p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
p.add_argument("--n", type=int, default=200_000)
p.add_argument("--out", default="dev/flow_vs_templates_corner.png")
a = p.parse_args()

m_train = np.asarray(shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0], dtype=np.float64)
flow = bulk.build_flow(jr.key(0), m_train, shear=a.stage != "bulk",
                       centroid=a.stage == "centroid")
flow = eqx.tree_deserialise_leaves(a.flow, flow)
jax.config.update("jax_enable_x64", True)
flow = jax.tree_util.tree_map(
    lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
sx = None
if a.stage == "centroid":
    sx = jnp.asarray(np.asarray(fitsio.read(
        f"{a.data_dir}/{B.CATALOGS[a.pop]['zero']}.fits")["cov_odd"], dtype=np.float64)[0])
zs = flow.base_dist.sample(jr.key(31), (a.n,)).astype(jnp.float64)
tr = eqx.filter_jit(jax.vmap(lambda z: flow.bijection.transform(
    z, B.condition(jnp.zeros(2), sx))))
m_flow = np.concatenate([np.asarray(tr(zs[i:i + 16384])) for i in range(0, a.n, 16384)])

rng = np.random.default_rng(0)
d_t = bulk.to_coords(m_train[rng.choice(len(m_train), a.n, replace=False)])
d_f = bulk.to_coords(m_flow)
d_t, d_f = d_t[np.isfinite(d_t).all(-1)], d_f[np.isfinite(d_f).all(-1)]
print(f"templates {len(d_t)}, flow draws {len(d_f)} (dropped non-finite)")

rg = [np.percentile(d_t[:, i], [0.05, 99.95]) for i in range(5)]
for i in (0, 1, 2):
    rg[i] += 0.25 * np.ptp(rg[i]) * np.array([-1.0, 1.0])
style = dict(labels=bulk.COORD_LABELS, bins=200, range=[tuple(r) for r in rg],
             smooth=2.0, plot_density=False, plot_contours=True,
             fill_contours=False, smooth1d=1.0, plot_datapoints=False,
             label_kwargs={"fontsize": bulk.LABEL_FONTSIZE})
fig = plt.figure(figsize=(16, 16))
corner.corner(d_t, fig=fig, color="tab:blue",
              hist_kwargs={"label": f"{a.pop} templates"}, **style)
corner.corner(d_f, fig=fig, color="tab:red",
              hist_kwargs={"label": "flow draws (g=0)"}, **style)
fig.axes[bulk.COORD_LABELS.__len__() + 1].legend(
    handles=[plt.Line2D([], [], color="tab:blue", label="templates"),
             plt.Line2D([], [], color="tab:red", label="flow")],
    bbox_to_anchor=(0.0, 1.0), loc="lower left", fontsize=16)
for ax in fig.axes:
    ax.tick_params(axis="both", which="major", labelsize=bulk.TICK_LABELSIZE)
fig.savefig(a.out, dpi=130, bbox_inches="tight")
print(f"wrote {a.out}")
