"""Test the over-smoothing hypothesis: is the flow's g=0 prior p_theta(x|0,Sigma_X)
BROADER than the (nda*L(X|C_X)-weighted) template density it should match?

Why this matters: the integrated response is  Q/P = shift_rate * d/dmu log(prior * N)(mu)
-- the score of the KERNEL-SMOOTHED prior at the target.  A broader prior gives a
flatter smoothed-prior => smaller score => Q and R both suppressed:
    Q_flow/Q_ana ~ (sigma_true^2 + sigma_k^2) / (sigma_flow^2 + sigma_k^2)
So if sigma_flow > sigma_true (flow over-smoothed), Q_flow<Q_ana -- exactly the
0.76-0.78 integrated deficit measured in flow_shear_response_map.py.

Compares, at a representative Sigma_X (log_scale=11.4, e=0):
  * marginal std / excess-kurtosis of the flow's g=0 samples vs the weighted templates
  * per (log10 Mf, Mr/Mf) cell: ellipticity (M1_std) spread sigma_flow vs sigma_true,
    overlaid with the observed Qf/Qa, Rf/Ra from the grid PQR npz.
NOTE Fourier moments: SMALL Mr/Mf = LARGE galaxy.

Run:  JAX_PLATFORMS=cpu PYTHONPATH=. python dev/flow_prior_width_check.py
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
import fitsio

from bfd_cnf.integrate_grid import load_prior_flow, load_stats
from bfd_cnf.models.flows import _batch_log_L_X
from bfd_cnf.config import TRAIN_FITS_PATH

FLOW = "flows/prior_flow_xy_nll.eqx"
NPZ = "data/pqr_grid_100k_nll.npz"
SX = jnp.array([11.4, 0.0, 0.0])       # representative median target Sigma_X
N_SAMPLE = 300_000
N_TMPL_ROWS = 300_000                   # rows (x3 copies) read, spread across file

flow = load_prior_flow(jax.random.PRNGKey(0), FLOW, "flows/q_flow_xy.eqx")
r2s = load_stats(FLOW)
mean = np.asarray(r2s.mean); std = np.asarray(r2s.std)
to_std = jax.jit(jax.vmap(lambda m: r2s.transform_and_log_det(m)[0]))


# ---- weighted moment stats -------------------------------------------------
def wstats(x, w):
    w = w / w.sum()
    mu = np.sum(w[:, None] * x, 0)
    d = x - mu
    var = np.sum(w[:, None] * d**2, 0)
    sd = np.sqrt(var)
    skew = np.sum(w[:, None] * d**3, 0) / sd**3
    kurt = np.sum(w[:, None] * d**4, 0) / var**2 - 3.0
    return mu, sd, skew, kurt


# ---- weighted templates ----------------------------------------------------
F = fitsio.FITS(TRAIN_FITS_PATH)[1]
ntot = F.get_nrows()
starts = np.linspace(0, ntot - 20_000, N_TMPL_ROWS // 20_000).astype(int)
mo, ce, nd = [], [], []
for s in starts:
    r = F.read(rows=np.arange(s, s + 20_000), columns=["moments", "centroid", "nda"])
    mo.append(np.asarray(r["moments"]).reshape(-1, 5)[:, :4])
    ce.append(np.asarray(r["centroid"]).reshape(-1, 2))
    nd.append(np.asarray(r["nda"]).reshape(-1))
mo = np.concatenate(mo); ce = np.concatenate(ce); nd = np.concatenate(nd)
ok = (mo[:, 0] > 0) & (mo[:, 1] > 0) & np.all(np.isfinite(mo), 1) & (nd > 0)
mo, ce, nd = mo[ok], ce[ok], nd[ok]
nd = np.minimum(nd, np.percentile(nd, 99.9))   # match training nda_clip_percentile=99.9
logL = np.asarray(_batch_log_L_X(jnp.asarray(ce), SX))
w = nd * np.exp(logL - logL.max())            # nda * L(X|C_X); shift for stability
wn = w / w.sum(); ess = 1.0 / np.sum(wn**2)
xt = np.asarray(to_std(jnp.asarray(mo)))       # standardized template moments
print(f"templates: {xt.shape[0]:,} copies (weighted by nda*L(X|C_X), log_scale={float(SX[0])}); "
      f"weight ESS = {ess:,.0f} ({100*ess/len(w):.1f}% of copies)")
_, sd_uw, _, _ = wstats(xt, np.ones(len(xt)))
print(f"unweighted template sd (standardiser reference): {np.array2string(sd_uw, precision=3)}")

# ---- flow g=0 samples ------------------------------------------------------
cond = jnp.concatenate([jnp.zeros(2), SX])
xs = np.asarray(flow.sample(jax.random.key(1), (N_SAMPLE,), condition=cond))
print(f"flow samples: {xs.shape[0]:,}\n")

names = ["log10Mf", "Mr/Mf", "M1/Mr", "M2/Mr"]
mt, st, kt_s, kt_k = wstats(xt, w)
ms, ss, sk_s, sk_k = wstats(xs, np.ones(len(xs)))
print(f"{'coord':>8} | {'sd_tmpl':>8} {'sd_flow':>8} {'sd_f/sd_t':>9} | "
      f"{'kurt_tmpl':>9} {'kurt_flow':>9}   (>1 sd ratio = flow broader = over-smoothed)")
for i, nm in enumerate(names):
    print(f"{nm:>8} | {st[i]:8.3f} {ss[i]:8.3f} {ss[i]/st[i]:9.3f} | "
          f"{kt_k[i]:9.3f} {sk_k[i]:9.3f}")


# ---- per-cell ellipticity spread + observed Qf/Qa, Rf/Ra -------------------
d = np.load(NPZ)
tg = d["targets_p"].astype(float); ps = d["pqr_sim_p"].astype(float); pf = d["pqr_p"].astype(float)
okv = np.isfinite(ps).all(1) & (ps[:, 0] > 1e-10) & np.isfinite(pf).all(1) & (pf[:, 0] > 1e-10)
gfl = np.log10(tg[:, 0]); gsz = tg[:, 1] / tg[:, 0]


def qr_ratio(m):
    def tot(p):
        P, Q1, R11 = p[m, 0], p[m, 1], p[m, 3]
        return np.sum(Q1 / P), np.sum((Q1 / P) ** 2 - R11 / P)
    Qa, Ra = tot(ps); Qf, Rf = tot(pf)
    return (Qf / Qa if abs(Qa) > 1e-12 else np.nan,
            Rf / Ra if abs(Ra) > 1e-12 else np.nan)


tfl = np.log10(mo[:, 0]); tsz = mo[:, 1] / mo[:, 0]         # template flux/size (raw)
sfl = xs[:, 0] * std[0] + mean[0]; ssz = xs[:, 1] * std[1] + mean[1]   # flow-sample flux/size

FEDGES = np.linspace(np.log10(1500.0), np.log10(90000.0), 7)
SEDGES = np.linspace(2.2, 3.5, 5)
FLUX = 0.5 * (FEDGES[:-1] + FEDGES[1:]); SIZE = 0.5 * (SEDGES[:-1] + SEDGES[1:])

print(f"\n{'log Mf':>6} {'Mr/Mf':>6} | {'sd_e1_t':>7} {'sd_e1_f':>7} {'f/t':>5} | "
      f"{'Qf/Qa':>6} {'Rf/Ra':>6} {'n_tmpl':>7}   (Mr/Mf small = large galaxy)")
for i, fc in enumerate(FLUX):
    for j, sc in enumerate(SIZE):
        fl_lo, fl_hi = FEDGES[i], FEDGES[i + 1]; sz_lo, sz_hi = SEDGES[j], SEDGES[j + 1]
        tm = (tfl >= fl_lo) & (tfl < fl_hi) & (tsz >= sz_lo) & (tsz < sz_hi)
        sm = (sfl >= fl_lo) & (sfl < fl_hi) & (ssz >= sz_lo) & (ssz < sz_hi)
        gm = okv & (gfl >= fl_lo) & (gfl < fl_hi) & (gsz >= sz_lo) & (gsz < sz_hi)
        if tm.sum() < 200 or sm.sum() < 200 or gm.sum() < 30:
            continue
        _, sd_t, _, _ = wstats(xt[tm, 2:3], w[tm])          # weighted template e1 spread
        sd_f = xs[sm, 2].std()
        qf, rf = qr_ratio(gm)
        print(f"{fc:6.2f} {sc:6.2f} | {sd_t[0]:7.3f} {sd_f:7.3f} {sd_f/sd_t[0]:5.2f} | "
              f"{qf:6.3f} {rf:6.3f} {int(tm.sum()):7d}")
