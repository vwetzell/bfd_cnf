"""
plot_centering_target_vs_model.py
=================================
Show, side by side, that the ELBO's centring-marginalised **target** carries a
clear e2 dipole (mean shift of M2/Mr) while the **trained prior** does not learn
it — isolating the gap as a training/optimisation issue, not an architecture or
target one.

* Target: the centroid-likelihood-weighted population mean ⟨M2/Mr⟩, restricted to
  templates near a (log10 Mf, Mr/Mf) slice so it matches the model evaluation.
  This is exactly what the prior is trained to reproduce (FlowMath Step 4).
* Model: density-weighted ⟨M2/Mr⟩ from the trained prior on a grid at the same
  slice (reused from plot_centering_dipole_diagnostic).

Run from the repo root::

    python -m bfd_cnf.plot_centering_target_vs_model
"""

from __future__ import annotations

import math
import os

import numpy as np
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bfd_cnf.config import PLOTS_DIR, key as base_key
from bfd_cnf.data import load_data
from bfd_cnf.models.flows import _batch_log_L_X
from bfd_cnf.plot_centering_dipole_diagnostic import (
    build_grid,
    load_prior,
    make_density_fn,
)
from bfd_cnf.plot_shear_derivs_compare import load_standardiser

LOG10MF = math.log10(2000.0)   # 3.30
MRMF = 3.3
LS = 13.3
DLG = 0.10                      # half-width of the log10 Mf window
DMR = 0.30                      # half-width of the Mr/Mf window
E2S = np.linspace(0.0, 0.2, 9)


def data_target_curve():
    print("Loading data (28M templates)...")
    data = load_data(key=base_key)
    M = np.asarray(data["moments_jnp"])
    X = np.asarray(data["centroid_moments_jnp"])
    log10mf = np.log10(M[:, 0])
    mrmf = M[:, 1] / M[:, 0]
    m2mr = M[:, 3] / M[:, 1]
    mask = (np.abs(log10mf - LOG10MF) < DLG) & (np.abs(mrmf - MRMF) < DMR)
    print(f"templates in slice: {int(mask.sum()):,}")
    Xs = jnp.asarray(X[mask])
    m2 = m2mr[mask]
    out = []
    for e2 in E2S:
        logL = np.asarray(_batch_log_L_X(Xs, jnp.array([LS, 0.0, e2])))
        w = np.exp(logL - logL.max())
        out.append(float((w @ m2) / w.sum()))
    return np.array(out)


def model_curve():
    mean, std = load_standardiser()
    prior = load_prior()
    prob_v = make_density_fn(prior)
    m1, m2, m1_flat, m2_flat, x_std = build_grid(LOG10MF, MRMF, 101, 0.5, mean, std)
    out = []
    for e2 in E2S:
        p = np.asarray(prob_v(x_std, 0.0, 0.0, LS, 0.0, e2), np.float64)
        out.append(float((p @ m2_flat) / p.sum()))
    return np.array(out)


def main():
    tgt = data_target_curve()
    mdl = model_curve()
    tgt0 = tgt - tgt[0]
    mdl0 = mdl - mdl[0]
    s_t = np.polyfit(E2S, tgt0, 1)[0]
    s_m = np.polyfit(E2S, mdl0, 1)[0]
    print(f"data-target slope d<M2>/de2 = {s_t:+.4e}")
    print(f"trained-model slope d<M2>/de2 = {s_m:+.4e}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(E2S, tgt0, "s-", color="C2",
            label=rf"ELBO target (data, weighted by $L(X|C_X)$), slope={s_t:+.2e}")
    ax.plot(E2S, mdl0, "o-", color="C0",
            label=rf"trained prior (after dipole-fix), slope={s_m:+.2e}")
    ax.axhline(0.0, color="grey", lw=1)
    ax.set_xlabel(r"$e_2$ (centroid-covariance ellipticity)", fontsize=14)
    ax.set_ylabel(r"$\Delta\langle M_2/M_r\rangle$", fontsize=14)
    ax.set_title(
        rf"Centering dipole: trained prior vs ELBO target"
        rf"  ($\log_{{10}}M_f\approx{LOG10MF:.2g}$, $M_r/M_f\approx{MRMF:.2g}$, "
        rf"$\log\,\mathrm{{scale}}={LS:.3g}$, $e_1=0$)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    out = os.path.join(PLOTS_DIR, f"centering_target_vs_model_lgmf{LOG10MF:.2f}_mrmf{MRMF:.2f}_ls{LS:.1f}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
