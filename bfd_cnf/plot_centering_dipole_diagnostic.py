"""
plot_centering_dipole_diagnostic.py
===================================
Read-only diagnostic that tests whether the trained prior flow represents the
**dipole** (mean-shift) signature that marginalizing over the target's centroid
uncertainty ``C_X`` should imprint on the ellipticity distribution (M1, M2).

Physics (see ``FlowMath.tex`` Step 4 and ``selection_term_analysis.md``):
the prior ``p(M | g, C_X)`` is trained to be the template population marginalised
over the unknown centroid offset, weighting each copy at centroid moment ``X`` by
``L(X|C_X)=N(X;0,C_X)``.  The centroid moment ``X`` is spin-1; measuring the
ellipticity moments about an offset center injects a spin-2 term *quadratic* in
the offset, so marginalising ``X ~ N(0,C_X)`` does **not** average to zero:

    E[ΔM1, ΔM2] = κ · tr(C_X) · (e1, e2)     (leading order, O(e))

i.e. an elliptical centering uncertainty rigidly **shifts** the (M1,M2) cloud
(a dipole in ΔP).  The isotropic broadening (monopole) is only O(e²) at fixed
``λ = ½ log det C_X``.

Both shear ``g`` and centering ellipticity ``e`` are spin-2 additive shifts of
(M1, M2).  ``ExplicitPolyLast`` implements ``g`` with an additive term
(``M1,M2 ← M1,M2 − c·(g1,g2)``); ``SigmaXCouplingLayer`` implements ``e`` with the
purely *linear* map ``A = s·I + c·E``, which can scale/stretch but cannot
translate.  This script checks that asymmetry directly:

  * Figure 1 (decisive): ``Δ⟨M2⟩`` vs ``g2`` (at e=0) should have a clear slope;
    ``Δ⟨M2⟩`` vs ``e2`` (at g=0) should be ~flat in the current model.
  * Figure 2 (structure): the e2=e_max ``ΔP`` plane decomposed into azimuthal
    harmonics (m=0/1/2), shown next to a reference dipole ``−δ·∂P0/∂M2``.

Like ``plot_shear_derivs_compare``, this only loads and evaluates the trained
prior flow (weights + cached standardiser); the FITS table is not needed.

Run from the repo root::

    python -m bfd_cnf.plot_centering_dipole_diagnostic
"""

from __future__ import annotations

import argparse
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bfd_cnf.config import (
    PLOTS_DIR,
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    e_max,
    key as base_key,
)
from bfd_cnf.models.flows import build_flows
from bfd_cnf.training import load_models
from bfd_cnf.plot_shear_derivs_compare import load_standardiser


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose whether the prior represents the centering dipole.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log10mf", type=float, default=math.log10(2000.0),
                   help="log10 of the flux moment Mf at which to slice.")
    p.add_argument("--mrmf", type=float, default=3.3, help="Size ratio Mr/Mf.")
    p.add_argument("--log-scale", type=float, default=13.3,
                   help="Centering-bias log-noise-scale 0.5*log det(C_X).")
    p.add_argument("--n", type=int, default=101, help="Grid resolution per axis.")
    p.add_argument("--m-range", type=float, default=0.5,
                   help="Half-width of the M1/Mr, M2/Mr axes.")
    p.add_argument("--g2-max", type=float, default=0.04,
                   help="Max |g2| for the shear sweep (linear, so any small value works).")
    p.add_argument("--n-sweep", type=int, default=9,
                   help="Number of points in each parameter sweep.")
    p.add_argument("--stats", type=str, default=None,
                   help="Standardiser npz (pass the newtmpl stats for new-set flows).")
    p.add_argument("--prior", type=str, default=None, help="Override prior flow path.")
    p.add_argument("--q", type=str, default=None, help="Override q flow path.")
    return p.parse_args()


def load_prior(prior_path=PRIOR_FLOW_PATH, q_path=Q_FLOW_PATH):
    """Load the trained prior flow (mirrors plot_shear_derivs_compare)."""
    if not (os.path.exists(prior_path) and os.path.exists(q_path)):
        raise FileNotFoundError(
            f"Trained flow weights not found:\n  {prior_path}\n  {q_path}"
        )
    from bfd_cnf.models.bijections import load_stats
    prior_flow, q_flow = build_flows(
        base_key, latent_dim=4, cond_dim=16, raw2standard=load_stats(prior_path)
    )
    prior_trained, _ = load_models(prior_flow, q_flow, prior_path, q_path)
    return prior_trained


def build_grid(log10mf, mrmf, n, r, mean, std):
    """Return flat (M1/Mr, M2/Mr) coordinates and the standardised grid points.

    Coordinates are laid out so a ``(n, n)`` reshape gives ``[M2 index, M1 index]``
    (rows = M2/Mr, cols = M1/Mr) — the standard imshow orientation.
    """
    m1 = np.linspace(-r, r, n)
    m2 = np.linspace(-r, r, n)
    M1g, M2g = np.meshgrid(m1, m2)  # 'xy': M1 varies along cols, M2 along rows
    m1_flat = M1g.ravel().astype(np.float64)
    m2_flat = M2g.ravel().astype(np.float64)
    x_raw = np.stack(
        [
            np.full(m1_flat.size, log10mf, np.float32),
            np.full(m1_flat.size, mrmf, np.float32),
            m1_flat.astype(np.float32),
            m2_flat.astype(np.float32),
        ],
        axis=1,
    )
    x_std = jnp.asarray((x_raw - mean) / std)
    return m1, m2, m1_flat, m2_flat, x_std


def make_density_fn(prior_trained):
    """Return a jitted ``(x_std, g1, g2, log_scale, e1, e2) -> P`` evaluator."""
    def prob(x_row, g1, g2, log_scale, e1, e2):
        condition = jnp.array([g1, g2, log_scale, e1, e2], dtype=jnp.float32)
        return jnp.exp(prior_trained.log_prob(x_row, condition=condition))

    in_axes = (0, None, None, None, None, None)
    return jax.jit(jax.vmap(prob, in_axes=in_axes))


def multipole_fractions(dP_flat, m1_flat, m2_flat, r_max, n_bins=40):
    """Fraction of radial-weighted |ΔP| power in azimuthal harmonics m=0,1,2."""
    r = np.sqrt(m1_flat**2 + m2_flat**2)
    th = np.arctan2(m2_flat, m1_flat)
    bins = np.linspace(0.0, r_max, n_bins + 1)
    which = np.digitize(r, bins) - 1
    powers = {0: 0.0, 1: 0.0, 2: 0.0}
    for k in range(n_bins):
        msk = which == k
        if msk.sum() < 8:
            continue
        rk = 0.5 * (bins[k] + bins[k + 1])
        w = rk * (bins[k + 1] - bins[k])  # radial area weight ~ r dr
        v = dP_flat[msk]
        tk = th[msk]
        a0 = v.mean()
        a1 = (v * np.exp(-1j * tk)).mean()
        a2 = (v * np.exp(-2j * tk)).mean()
        powers[0] += (a0**2) * w
        powers[1] += (np.abs(a1) ** 2) * w
        powers[2] += (np.abs(a2) ** 2) * w
    total = sum(powers.values()) + 1e-300
    return {m: powers[m] / total for m in powers}


def main() -> None:
    args = parse_args()
    log10mf = args.log10mf
    mrmf = args.mrmf
    ls = args.log_scale

    mean, std = load_standardiser(args.stats)
    print("Loading flow...")
    from bfd_cnf.config import PRIOR_FLOW_PATH as _PP, Q_FLOW_PATH as _QP
    prior_trained = load_prior(args.prior or _PP, args.q or _QP)
    prob_v = make_density_fn(prior_trained)

    m1, m2, m1_flat, m2_flat, x_std = build_grid(
        log10mf, mrmf, args.n, args.m_range, mean, std
    )

    def density(g1, g2, e1, e2):
        return np.asarray(prob_v(x_std, g1, g2, ls, e1, e2), np.float64)

    def mean_m(g1, g2, e1, e2):
        p = density(g1, g2, e1, e2)
        s = p.sum()
        return (p @ m1_flat) / s, (p @ m2_flat) / s

    # ── Sweeps: Δ⟨M2⟩ vs e2 (g=0) and vs g2 (e=0) ───────────────────────────
    e2s = np.linspace(0.0, e_max, args.n_sweep)
    g2s = np.linspace(0.0, args.g2_max, args.n_sweep)
    print(f"Sweeping e2 in [0, {e_max}] (g=0) and g2 in [0, {args.g2_max}] (e=0)...")
    m2_vs_e2 = np.array([mean_m(0.0, 0.0, 0.0, e2)[1] for e2 in e2s])
    m2_vs_g2 = np.array([mean_m(0.0, g2, 0.0, 0.0)[1] for g2 in g2s])
    d_e2 = m2_vs_e2 - m2_vs_e2[0]
    d_g2 = m2_vs_g2 - m2_vs_g2[0]
    slope_e2 = np.polyfit(e2s, d_e2, 1)[0]
    slope_g2 = np.polyfit(g2s, d_g2, 1)[0]
    print(f"  d<M2>/de2 = {slope_e2:+.4e}   (expected nonzero from physics)")
    print(f"  d<M2>/dg2 = {slope_g2:+.4e}   (ExplicitPolyLast additive Q-term)")
    if abs(slope_g2) > 0:
        print(f"  ratio |slope_e2 / slope_g2| = {abs(slope_e2/slope_g2):.3e}")

    # ── Figure 1 ────────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(g2s, d_g2, "o-", color="C3",
             label=rf"$\Delta\langle M_2\rangle$ vs $g_2$ (e=0), slope={slope_g2:+.2e}")
    ax1.plot(e2s, d_e2, "s-", color="C0",
             label=rf"$\Delta\langle M_2\rangle$ vs $e_2$ (g=0), slope={slope_e2:+.2e}")
    ax1.axhline(0.0, color="grey", lw=1)
    ax1.set_xlabel(r"parameter value ($g_2$ or $e_2$)", fontsize=14)
    ax1.set_ylabel(r"$\Delta\langle M_2/M_r\rangle$ (density-weighted mean shift)",
                   fontsize=14)
    ax1.set_title(
        rf"Mean ellipticity shift: shear vs centering ellipticity"
        rf"  ($\log_{{10}}M_f={log10mf:.3g}$, $M_r/M_f={mrmf:.3g}$, "
        rf"$\log\,\mathrm{{scale}}={ls:.3g}$)",
        fontsize=12,
    )
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    out1 = os.path.join(
        PLOTS_DIR,
        f"centering_dipole_meanM2_vs_e2_g2_lgmf{log10mf:.2f}_mrmf{mrmf:.2f}_ls{ls:.1f}.png",
    )
    fig1.tight_layout()
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved to {out1}")

    # ── Figure 2: ΔP plane + reference dipole + multipole bars ───────────────
    n = args.n
    P0 = density(0.0, 0.0, 0.0, 0.0).reshape(n, n)        # [M2, M1]
    Pe = density(0.0, 0.0, 0.0, e_max).reshape(n, n)
    dP = Pe - P0

    dm2 = m2[1] - m2[0]
    dP0_dm2 = np.gradient(P0, dm2, axis=0)                # ∂P0/∂M2
    # reference dipole = +M2 shift of P0 (matched to dP amplitude for display only)
    amp = np.abs(dP).max() / (np.abs(dP0_dm2).max() + 1e-300)
    dP_ref = -amp * dP0_dm2

    frac_model = multipole_fractions(dP.ravel(), m1_flat, m2_flat, args.m_range)
    frac_ref = multipole_fractions(dP_ref.ravel(), m1_flat, m2_flat, args.m_range)
    print("Multipole power fractions (m=0 monopole, m=1 dipole, m=2 quadrupole):")
    print(f"  model ΔP : m0={frac_model[0]:.2f}  m1={frac_model[1]:.2f}  m2={frac_model[2]:.2f}")
    print(f"  ref dipole: m0={frac_ref[0]:.2f}  m1={frac_ref[1]:.2f}  m2={frac_ref[2]:.2f}")

    extent = (m1[0], m1[-1], m2[0], m2[-1])
    vmax = np.abs(dP).max()
    refmax = np.abs(dP_ref).max()
    fig2, ax2 = plt.subplots(1, 3, figsize=(18, 5.2))

    im0 = ax2[0].imshow(dP, cmap="RdBu", origin="lower", extent=extent,
                        vmin=-vmax, vmax=vmax, aspect="auto", interpolation="none")
    ax2[0].set_title(rf"model $\Delta P = P(e_2={e_max:.2g}) - P(0)$", fontsize=13)
    plt.colorbar(im0, ax=ax2[0])

    im1 = ax2[1].imshow(dP_ref, cmap="RdBu", origin="lower", extent=extent,
                        vmin=-refmax, vmax=refmax, aspect="auto", interpolation="none")
    ax2[1].set_title(r"reference dipole $-\delta\,\partial P_0/\partial M_2$ (structure only)",
                     fontsize=13)
    plt.colorbar(im1, ax=ax2[1])

    for a in ax2[:2]:
        a.set_xlabel(r"$M_1/M_r$", fontsize=13)
        a.set_ylabel(r"$M_2/M_r$", fontsize=13)
        a.axvline(0.0, color="grey", lw=1)
        a.axhline(0.0, color="grey", lw=1)

    ms = [0, 1, 2]
    width = 0.38
    ax2[2].bar([m - width / 2 for m in ms], [frac_model[m] for m in ms],
               width, color="C0", label="model ΔP")
    ax2[2].bar([m + width / 2 for m in ms], [frac_ref[m] for m in ms],
               width, color="C3", label="reference dipole")
    ax2[2].set_xticks(ms)
    ax2[2].set_xticklabels(["m=0\nmonopole", "m=1\ndipole", "m=2\nquadrupole"])
    ax2[2].set_ylabel("fraction of radial-weighted power", fontsize=12)
    ax2[2].set_title("azimuthal power decomposition", fontsize=13)
    ax2[2].legend(fontsize=11)
    ax2[2].grid(alpha=0.3, axis="y")

    fig2.suptitle(
        rf"Centering-ellipticity response structure "
        rf"($\log_{{10}}M_f={log10mf:.3g}$, $M_r/M_f={mrmf:.3g}$, "
        rf"$\log\,\mathrm{{scale}}={ls:.3g}$, $e_1=0$)",
        fontsize=14,
    )
    out2 = os.path.join(
        PLOTS_DIR,
        f"centering_dipole_multipole_lgmf{log10mf:.2f}_mrmf{mrmf:.2f}_ls{ls:.1f}_e2{e_max:.2f}.png",
    )
    fig2.tight_layout()
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved to {out2}")


if __name__ == "__main__":
    main()
