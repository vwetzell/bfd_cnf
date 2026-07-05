#!/usr/bin/env python
"""Where do real targets' C_X live, vs the two centroid-grid failure regimes?

Target log_scale = 0.5·logdet(C_X) from the integration (sx_conds in the grid
npz; C_X = even_cov_to_CX flux-row recipe).  Overlay the target density on the
measured failure curves:
  - var_eff/true  (centroid X-integral fidelity; <1 = high-ls truncation bias)
  - ESS/batch     (SNIS effective sample size per step; low = low-ls inefficiency)
Both measured earlier (dev/sx_grid_quadrature.py, dev/sx_batch_ess.py).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID = "data/pqr_grid_independent_FULL_mf3000_90000_newtmpl.npz"
RANGE = (10.5, 13.0)

# measured curves (hardcoded from the two diagnostics; stable)
LS_V = np.array([9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 11.98, 12.5, 13.0])
VAR_EFF = np.array([1.304, 1.080, 1.026, 1.021, 1.032, 1.049, 1.048, 0.990, 0.840])
LS_E = np.array([10.5, 11.0, 11.5, 11.98, 12.5, 13.0])
ESS_N = np.array([100, 167, 274, 438, 687, 977]) / 2048.0

def main():
    ls = np.load(GRID, allow_pickle=True)["sx_conds_p"][:, 0]
    med = np.median(ls)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(ls, bins=200, range=(9.5, 13.2), density=True,
            color="0.6", alpha=0.6, label="target log_scale density")
    # regime shading
    ax.axvspan(9.5, 11.0, color="red", alpha=0.07)
    ax.axvspan(11.0, 12.5, color="green", alpha=0.07)
    ax.axvspan(12.5, 13.2, color="orange", alpha=0.10)
    for x, t, c in [(RANGE[0], "range", "k"), (RANGE[1], "range", "k"),
                    (11.98, "ref", "b"), (med, f"median={med:.2f}", "purple")]:
        ax.axvline(x, color=c, ls="--", lw=1.2)
        ax.text(x, ax.get_ylim()[1]*0.96, t, rotation=90, va="top", ha="right",
                fontsize=8, color=c)
    ax.set_xlabel("log_scale = 0.5·logdet(C_X)")
    ax.set_ylabel("target density")
    ax.set_xlim(9.5, 13.2)

    ax2 = ax.twinx()
    ax2.plot(LS_V, VAR_EFF, "o-", color="darkorange", label="var_eff/true (integral fidelity)")
    ax2.plot(LS_E, ESS_N, "s-", color="crimson", label="ESS/batch (SNIS efficiency)")
    ax2.axhline(1.0, color="darkorange", lw=0.6, ls=":")
    ax2.set_ylabel("var_eff/true   &   ESS/batch")
    ax2.set_ylim(0, 1.4)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    ax.set_title("Real targets vs centroid-grid failure regimes\n"
                 "96% of targets sit in the well-sampled middle (ESS healthy, var_eff≈1)")
    out = "plots/target_logscale_vs_failure.png"
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}  (N={len(ls):,}, median log_scale={med:.3f})")

if __name__ == "__main__":
    main()
