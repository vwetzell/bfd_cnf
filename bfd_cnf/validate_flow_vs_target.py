"""
validate_flow_vs_target.py
==========================
Quantitative companion to plot_corner.py.  Samples the trained prior at the physical
reference centroid scale (log_scale = prior_sigmax_log_scale_mean, e=0, g=0), inverts to
raw moments, and prints p05/p50/p95 of [log10 M0, MR/M0, M1/MR, M2/MR] alongside the
at-centre summary templates (the success target).  Success = flow quantiles track summary,
especially the log10 M0 bright tail (p95) and the MR/M0 + ellipticity widths.

Run:  python -m bfd_cnf.validate_flow_vs_target
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .config import (
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    prior_sigmax_log_scale_mean,
)
from .data import load_data, load_summary_moments
from .models.flows import build_flows
from .training import load_models

QS = (0.05, 0.50, 0.95)
N = 1_000_000


def _marg_raw(M):
    return np.column_stack([
        np.log10(np.abs(M[:, 0])),
        M[:, 1] / M[:, 0],
        M[:, 2] / M[:, 1],
        M[:, 3] / M[:, 1],
    ])


def main() -> None:
    if not (os.path.exists(PRIOR_FLOW_PATH) and os.path.exists(Q_FLOW_PATH)):
        raise FileNotFoundError("Trained flow weights not found; train first.")

    data = load_data()
    key = data["key"]
    raw2standard = data["raw2standard"]

    prior_flow, q_flow = build_flows(key, latent_dim=4, cond_dim=16)
    prior, _ = load_models(prior_flow, q_flow, PRIOR_FLOW_PATH, Q_FLOW_PATH)

    ls = float(prior_sigmax_log_scale_mean)
    cond = jnp.array([0.0, 0.0, ls, 0.0, 0.0])
    key, sub = jr.split(key)
    z = prior.sample(sub, sample_shape=(N,), condition=cond)
    raw, _ = jax.vmap(raw2standard.inverse_and_log_det)(z)
    raw = np.asarray(raw)
    flow = _marg_raw(raw)
    flow = flow[np.all(np.isfinite(flow), axis=1)]

    summ = _marg_raw(np.asarray(load_summary_moments()))
    summ = summ[np.all(np.isfinite(summ), axis=1)]

    names = ["log10 M0", "MR/M0", "M1/MR", "M2/MR"]
    print(f"\nprior @ log_scale={ls:.2f}, e=0, g=0   (p05/p50/p95)")
    print(f"{'':10s}{'FLOW (prior)':>26s}{'TARGET summary(at-centre)':>30s}")
    for j, nm in enumerate(names):
        f = np.quantile(flow[:, j], QS)
        s = np.quantile(summ[:, j], QS)
        print(f"{nm:10s}  {f[0]:7.3f}/{f[1]:6.3f}/{f[2]:6.3f}"
              f"      {s[0]:7.3f}/{s[1]:6.3f}/{s[2]:6.3f}")


if __name__ == "__main__":
    main()
