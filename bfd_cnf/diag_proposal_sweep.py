"""
diag_proposal_sweep.py
======================
Diagnose whether the RQMC importance proposal under-covers the integrand for
the grid targets — i.e. whether the "proposed points miss some of the
probability" for a given galaxy.

Two complementary read-outs, swept over ``hessian_scale`` (the multiplier on the
Laplace proposal covariance; production default 3.0):

1. **Stability of P/Q/R and the recovered shear / m-bias.** If widening the
   proposal leaves them unchanged, the proposal already covers the integrand;
   if they drift (especially P rising, m moving), the narrower proposal was
   clipping mass.
2. **Effective sample size** ``ESS = (Σc)²/Σc²`` per target (c = per-point
   contribution to P), out of ``n_points``. ``ESS ≪ n_points`` or a large
   single-point weight share = a peaked / under-covering proposal.

Run::

    python -m bfd_cnf.diag_proposal_sweep --n-targets 4000
"""

from __future__ import annotations

import argparse
import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import GRID_M_PATH, GRID_P_PATH, PLOTS_DIR, key as _base_key
from .inference import integrate_grid_pqr, load_grid_data
from .integrate_grid import load_prior_flow, load_raw2standard
from .statistics import pqr2g, pqr2multbias


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-targets", type=int, default=4000)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--n-replicates", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[2.0, 3.0, 4.0, 5.0, 6.0])
    ap.add_argument("--configs", type=str, nargs="+", default=None,
                    help="Explicit configs 'NPTSxNREPxHS' (e.g. 1024x16x3 65536x8x5). "
                         "Overrides --scales; batch-size auto-set to bound memory.")
    ap.add_argument("--out", default=os.path.join(PLOTS_DIR, "proposal_ess_sweep.png"))
    args = ap.parse_args()

    # Build the list of (n_points, n_replicates, hessian_scale, batch_size) configs.
    def auto_bs(npts):                              # keep ~2.1M points/flow-eval
        return int(max(16, min(512, 2_097_152 // npts)))
    if args.configs:
        configs = []
        for c in args.configs:
            npts_s, nrep_s, hs_s = c.lower().split("x")
            npts, nrep = int(npts_s), int(nrep_s)
            configs.append((npts, nrep, float(hs_s), auto_bs(npts)))
    else:
        configs = [(args.n_points, args.n_replicates, hs, args.batch_size)
                   for hs in args.scales]

    raw2standard = load_raw2standard()
    key, k_flow = jr.split(_base_key)
    prior_flow = load_prior_flow(k_flow)
    print("Loading grid...")
    joined_grid = load_grid_data(GRID_P_PATH, GRID_M_PATH)
    key, k_int = jr.split(key)          # same k_int for every config => same targets

    rows = []
    print(f"\nproposal convergence: n_targets={args.n_targets}\n")
    hdr = (f"{'npts':>7s} {'nrep':>4s} {'hs':>4s} | "
           f"{'g1(+)':>8s} {'g1(-)':>8s} {'g2(+)':>8s} {'g2(-)':>8s} "
           f"{'m':>8s} | {'medESS':>7s} {'ESS/N':>6s} {'effN':>7s} "
           f"{'%ESS<50':>7s} {'medMaxW':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for npts, nrep, hs, bs in configs:
        res = integrate_grid_pqr(
            joined_grid, raw2standard, prior_flow, key=k_int,
            n_targets=args.n_targets, n_points=npts,
            n_replicates=nrep, batch_size=bs,
            hessian_scale=hs, return_ess=True, verbose=False,
        )
        pqr_p, pqr_m = res["pqr_p"], res["pqr_m"]
        gp = np.asarray(pqr2g(jnp.asarray(pqr_p)))
        gm = np.asarray(pqr2g(jnp.asarray(pqr_m)))
        m = float(pqr2multbias(jnp.concatenate([pqr_p, pqr_m], axis=-1)))
        ess = np.concatenate([np.asarray(res["ess_p"]), np.asarray(res["ess_m"])])
        maxw = np.concatenate([np.asarray(res["maxw_p"]), np.asarray(res["maxw_m"])])
        ess = ess[np.isfinite(ess)]; maxw = maxw[np.isfinite(maxw)]
        medess = float(np.median(ess))
        # effective samples/target combining points across replicates:
        effN = medess * nrep
        row = dict(npts=npts, nrep=nrep, hs=hs, g1p=gp[0], g1m=gm[0],
                   g2p=gp[1], g2m=gm[1], m=m, medess=medess, essN=medess / npts,
                   effN=effN, frac_lt50=float(np.mean(ess < 50)),
                   medmaxw=float(np.median(maxw)),
                   label=f"{npts}x{nrep}\nhs{hs:g}")
        rows.append(row)
        print(f"{npts:7d} {nrep:4d} {hs:4.1f} | "
              f"{gp[0]:+8.4f} {gm[0]:+8.4f} {gp[1]:+8.4f} {gm[1]:+8.4f} "
              f"{m:+8.4f} | {medess:7.0f} {medess/npts:6.3f} {effN:7.0f} "
              f"{100*row['frac_lt50']:6.1f}% {row['medmaxw']:7.3f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    x = list(range(len(rows)))
    labels = [r["label"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].axhline(+0.02, color="0.7", ls=":"); ax[0].axhline(-0.02, color="0.7", ls=":")
    ax[0].plot(x, [r["g1p"] for r in rows], "o-", color="red", label=r"$g_1$(+)")
    ax[0].plot(x, [r["g1m"] for r in rows], "o-", color="blue", label=r"$g_1$(-)")
    ax[0].plot(x, [r["m"] for r in rows], "s--", color="k", label="m (bias)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_xlabel("config (n_points x n_replicates, hessian_scale)")
    ax[0].set_ylabel("recovered shear / m")
    ax[0].set_title("Shear & m-bias across configs")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(x, [r["effN"] for r in rows], "o-", color="purple",
               label="effective samples (medESS x nrep)")
    for xi, r in zip(x, rows):
        ax[1].annotate(f"ESS/N={r['essN']:.3f}", (xi, r["effN"]),
                       textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_xlabel("config (n_points x n_replicates, hessian_scale)")
    ax[1].set_ylabel("effective samples / target (log)")
    ax[1].set_title("Proposal effective sample size")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    fig.suptitle("Importance-proposal convergence (same flow & targets)", fontsize=12)
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(args.out, dpi=140); plt.close(fig)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
