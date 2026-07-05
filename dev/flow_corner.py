"""Corner plot: flow g=0 samples vs nda*L(X|C_X)-weighted training templates,
in transformed coords [log10 Mf, Mr/Mf, M1/Mr, M2/Mr] at log_scale=11.4, e=0.

Run:  JAX_PLATFORMS=cpu PYTHONPATH=. python dev/flow_corner.py
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

from bfd_cnf.integrate_grid import load_prior_flow, load_stats
from bfd_cnf.models.flows import _batch_log_L_X
from bfd_cnf.config import TRAIN_FITS_PATH, PLOTS_DIR

FLOW = "flows/prior_flow_xy_nll.eqx"
SX = jnp.array([11.4, 0.0, 0.0])
N = 200_000
LABELS = ["log10 Mf", "Mr/Mf", "M1/Mr", "M2/Mr"]

flow = load_prior_flow(jax.random.PRNGKey(0), FLOW, "flows/q_flow_xy.eqx")
r2s = load_stats(FLOW)
mean = np.asarray(r2s.mean); std = np.asarray(r2s.std)
to_std = jax.jit(jax.vmap(lambda m: r2s.transform_and_log_det(m)[0]))

# weighted templates -> transformed coords
F = fitsio.FITS(TRAIN_FITS_PATH)[1]
ntot = F.get_nrows()
starts = np.linspace(0, ntot - 20_000, N // 20_000).astype(int)
mo, ce, nd = [], [], []
for s in starts:
    r = F.read(rows=np.arange(s, s + 20_000), columns=["moments", "centroid", "nda"])
    mo.append(np.asarray(r["moments"]).reshape(-1, 5)[:, :4])
    ce.append(np.asarray(r["centroid"]).reshape(-1, 2))
    nd.append(np.asarray(r["nda"]).reshape(-1))
mo = np.concatenate(mo); ce = np.concatenate(ce); nd = np.concatenate(nd)
ok = (mo[:, 0] > 0) & (mo[:, 1] > 0) & np.all(np.isfinite(mo), 1) & (nd > 0)
mo, ce, nd = mo[ok], ce[ok], nd[ok]
nd = np.minimum(nd, np.percentile(nd, 99.9))
w = nd * np.exp(np.asarray(_batch_log_L_X(jnp.asarray(ce), SX)))
xt = np.asarray(to_std(jnp.asarray(mo))) * std + mean          # transformed coords

# flow samples -> transformed coords
cond = jnp.concatenate([jnp.zeros(2), SX])
xs = np.asarray(flow.sample(jax.random.key(1), (N,), condition=cond)) * std + mean

# shared axis ranges (robust, weighted for templates)
def wq(a, ww, qs):
    o = np.argsort(a); a, ww = a[o], ww[o]
    c = np.cumsum(ww) / ww.sum()
    return np.interp(qs, c, a)
rng = []
for i in range(4):
    lo = min(wq(xt[:, i], w, 0.002), np.percentile(xs[:, i], 0.2))
    hi = max(wq(xt[:, i], w, 0.998), np.percentile(xs[:, i], 99.8))
    rng.append((lo, hi))

ckw = dict(range=rng, bins=60, labels=LABELS, plot_datapoints=False,
           plot_density=False, no_fill_contours=True, smooth=1.0,
           levels=(0.393, 0.865, 0.989), hist_kwargs=dict(density=True))
fig = corner.corner(xt, weights=w, color="C0", **ckw)
corner.corner(xs, color="C3", fig=fig, **ckw)
fig.legend(handles=[plt.Line2D([], [], color="C0", label="templates (nda·L weighted)"),
                    plt.Line2D([], [], color="C3", label="flow  p(x|g=0, log_scale=11.4)")],
           loc="upper right", fontsize=12, frameon=False)
fig.suptitle("Prior flow vs weighted training templates", fontsize=13)
out = os.path.join(PLOTS_DIR, "flow_vs_templates_corner.png")
os.makedirs(PLOTS_DIR, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}")
