"""Overlay per-galaxy g2 distributions: flow at two effective-sample levels
(effN~700 and ~11000) vs the closed-form analytic, for both shear groups."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bfd_cnf.plot_per_galaxy_shear import per_object_g

lo = np.load("data/pqr_grid_100k_mf3000_90000_plateau310k_lores1024x16.npz")
hi = np.load("data/pqr_grid_100k_mf3000_90000_plateau310k_hires8192x32.npz")

RANGE, BINS = 1.0, 200
edges = np.linspace(-RANGE, RANGE, BINS + 1)

# (label, npz, pqr-key-prefix, color, linestyle)
sources = [
    ("flow  effN~700  (1024x16)", lo, "pqr_", "C0", "-"),
    ("flow  effN~11000 (8192x32)", hi, "pqr_", "C2", "--"),
    ("analytic (closed-form)", hi, "pqr_sim_", "C3", ":"),
]


def g2(pqr):
    g = per_object_g(pqr.astype(np.float64))[1]   # index 1 == g2
    return g[np.isfinite(g)]


fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for col, (grp, title) in enumerate([("p", "+ shear group"), ("m", "- shear group")]):
    a = ax[col]
    for label, d, key, color, ls in sources:
        g = g2(d[key + grp])
        iqr = float(np.subtract(*np.percentile(g, [75, 25])))
        t1 = float(np.mean(np.abs(g) > 1.0))
        a.hist(g, bins=edges, density=True, histtype="step", color=color, ls=ls,
               lw=1.6, label=f"{label}  (IQR {iqr:.3f}, |g|>1 {t1:.3f})")
    a.axvline(0.0, color="k", ls="--", lw=1)          # applied g2 = 0
    a.set_title(rf"per-galaxy $g_2$: {title}", fontsize=12)
    a.set_xlabel(r"per-galaxy $g_2$ estimate"); a.set_ylabel("density")
    a.set_xlim(-RANGE, RANGE); a.legend(fontsize=9)

fig.suptitle(r"$g_2$ per-galaxy distributions: flow at two effN vs analytic "
             "(width is set by the model, not the sampler)", fontsize=13)
fig.tight_layout()
out = "plots/g2_distributions_overlay.png"
fig.savefig(out, dpi=140); plt.close(fig)
print("saved", out)
