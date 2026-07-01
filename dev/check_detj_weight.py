"""Robust check: which weighted template marginal does the converged flow match?
Compares weighted quantiles (p10/p50/p90) of the flow's g=0 sample against the
templates under 4 weightings, plus flow-sample health and blue weight concentration.
"""
import numpy as np, jax, jax.numpy as jnp
from bfd_cnf.data import load_training_dataset
from bfd_cnf.models.flows import _batch_log_L_X
from bfd_cnf.plot_corner import load_flow, sample_flow, to_coords, PLOT_RANGE

LS = 11.94
COORDS = ["log10 Mf", "Mr/Mf", "M1/Mr", "M2/Mr"]


def wq(x, w, qs=(0.1, 0.5, 0.9)):
    w = np.clip(w, 0, None); i = np.argsort(x); x, w = x[i], w[i]
    cw = np.cumsum(w) - 0.5 * w; cw /= w.sum()
    return np.interp(qs, cw, x)


class A:
    prior = "flows/prior_flow_xy_nll.eqx"; q = None; stats = None


ds = load_training_dataset()
m = np.asarray(ds["moments_jnp"], np.float64)
X = np.asarray(ds["centroid_moments_jnp"], np.float64)[:, :2]
nda = np.asarray(ds["nda"], np.float64)
wfull = np.asarray(ds["weights"], np.float64)   # batch-sampler proposal
inbox = wfull > 0
m, X, nda, wfull = m[inbox], X[inbox], nda[inbox], wfull[inbox]

c = to_coords(m)  # drops non-finite; realign weights on the same mask
fin = np.all(np.isfinite(np.column_stack([
    np.log10(np.abs(m[:, 0])), m[:, 1] / m[:, 0],
    m[:, 2] / m[:, 1], m[:, 3] / m[:, 1]])), axis=1)
m, X, nda, wfull = m[fin], X[fin], nda[fin], wfull[fin]
detj = np.clip(0.25 * (m[:, 1] ** 2 - m[:, 2] ** 2 - m[:, 3] ** 2), 0.0, None)
logL = np.asarray(_batch_log_L_X(jnp.asarray(X), jnp.array([LS, 0.0, 0.0])))
L = np.exp(logL - logL.max())

# Is nda_HT ∝ 1/detj (so nda·L ∝ 1/Mf² → low-flux pile-up)?  corr in log space.
lm = np.log(nda); ld = np.log(np.clip(detj, 1e-30, None))
print(f"\ncorr(log nda_HT, log detj) = {np.corrcoef(lm, ld)[0,1]:+.3f}  "
      f"(−1 ⇒ nda_HT ∝ 1/detj)")

W = {"count(unwt)": np.ones(len(m)), "nda·L": nda * L,
     "nda·detj·L": nda * detj * L, "detj·L (no nda)": detj * L,
     "wfull·L (proposal)": wfull * L, "L only": L.copy()}

prior_trained, raw2standard, key = load_flow(A(), "train")
flow_c = sample_flow(prior_trained, raw2standard, key, 300_000, LS)

print("\nFraction of FLOW samples inside PLOT_RANGE per coord:")
for i, name in enumerate(COORDS):
    lo, hi = PLOT_RANGE[i]
    print(f"  {name:<9} {np.mean((flow_c[:, i] >= lo) & (flow_c[:, i] <= hi)):.3f}")

print(f"\nWeighted quantiles p10/p50/p90:")
print(f"{'coord':<9} {'FLOW':>22} | " + " | ".join(f"{k:>22}" for k in W))
for i, name in enumerate(COORDS):
    fq = np.quantile(flow_c[:, i], [0.1, 0.5, 0.9])
    row = f"{name:<9} {fq[0]:6.2f}/{fq[1]:6.2f}/{fq[2]:6.2f} | "
    row += " | ".join("{:6.2f}/{:6.2f}/{:6.2f}".format(*wq(c[:, i], w)) for w in W.values())
    print(row)

print("\nBlue weight concentration (ESS%% and top-10 weight share):")
for k, w in W.items():
    w = w / w.sum(); ess = 1.0 / np.sum(w ** 2)
    top10 = np.sort(w)[-10:].sum()
    print(f"  {k:<16} ESS={100*ess/len(w):5.2f}%  top-10 share={100*top10:5.2f}%")
