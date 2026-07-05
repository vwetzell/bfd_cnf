#!/usr/bin/env python
"""Target log_scale histogram + fixed-log_scale m sweep + the two failure curves."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID = "data/pqr_grid_independent_FULL_mf3000_90000_newtmpl.npz"
RANGE = (10.5, 13.0)
# fixed-log_scale m sweep (dev/run_nll_fixedls_sweep.sh; 10k targets, nll flow, e=0)
LS_M = np.array([10.5, 11.0, 11.41, 11.94, 12.5, 13.0])
M    = np.array([0.1045, 0.0997, 0.0969, 0.0892, 0.0829, 0.0781])
# measured failure curves (dev/sx_grid_quadrature.py, dev/sx_batch_ess.py)
LS_V = np.array([9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 11.98, 12.5, 13.0])
VAR_EFF = np.array([1.304, 1.080, 1.026, 1.021, 1.032, 1.049, 1.048, 0.990, 0.840])
LS_E = np.array([10.5, 11.0, 11.5, 11.98, 12.5, 13.0])
ESS_N = np.array([100, 167, 274, 438, 687, 977]) / 2048.0

def main():
    ls = np.load(GRID, allow_pickle=True)["sx_conds_p"][:, 0]
    med = np.median(ls)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.hist(ls, bins=200, range=(9.8, 13.2), density=True,
            color="0.6", alpha=0.5, label="target log_scale density")
    ax.axvspan(9.8, 11.0, color="red", alpha=0.05)
    ax.axvspan(12.5, 13.2, color="orange", alpha=0.07)
    for x, t, c in [(RANGE[0], "train range", "k"), (RANGE[1], "train range", "k"),
                    (med, f"target median={med:.2f}", "purple")]:
        ax.axvline(x, color=c, ls="--", lw=1.0)
        ax.text(x, ax.get_ylim()[1]*0.97, t, rotation=90, va="top", ha="right",
                fontsize=8, color=c)
    ax.set_xlabel("log_scale = 0.5·logdet(C_X)")
    ax.set_ylabel("target density")
    ax.set_xlim(9.8, 13.2)

    # right axis 1: failure curves (var_eff/true & ESS/batch)
    ax2 = ax.twinx()
    ax2.plot(LS_V, VAR_EFF, "o-", color="darkorange", label="var_eff/true (integral fidelity)")
    ax2.plot(LS_E, ESS_N, "s-", color="steelblue", label="ESS/batch (SNIS efficiency)")
    ax2.axhline(1.0, color="darkorange", lw=0.5, ls=":")
    ax2.set_ylabel("var_eff/true   &   ESS/batch", color="0.3")
    ax2.set_ylim(0, 1.45)

    # right axis 2 (offset): m sweep
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 58))
    ax3.plot(LS_M, M, "D-", color="crimson", lw=2, label="m (fixed-log_scale sweep)")
    m_med = np.interp(med, LS_M, M)
    ax3.plot([med], [m_med], "*", color="purple", ms=16, zorder=5)
    ax3.annotate(f"m≈{m_med:+.3f}", (med, m_med), textcoords="offset points",
                 xytext=(8, 8), fontsize=10, color="purple")
    ax3.set_ylabel("multiplicative bias  m", color="crimson")
    ax3.tick_params(axis="y", colors="crimson")
    ax3.set_ylim(0.0, 0.13)

    handles = sum([a.get_legend_handles_labels()[0] for a in (ax, ax2, ax3)], [])
    labels  = sum([a.get_legend_handles_labels()[1] for a in (ax, ax2, ax3)], [])
    ax.legend(handles, labels, loc="upper right", fontsize=9)
    ax.set_title("Target C_X distribution with m sweep and centroid-grid failure curves")
    out = "plots/target_logscale_with_m.png"
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}  (median log_scale={med:.3f}, m_at_median≈{m_med:+.4f})")

if __name__ == "__main__":
    main()
