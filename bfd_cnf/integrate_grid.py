"""
integrate_grid.py
=================
Standalone CLI for flow-based RQMC PQR integration over the BFD shear
simulation grid targets (``merged_masked_bfd_grid_{p,m}.npy``), using the
trained prior flow (``prior_flow_xy.eqx``).

This is the lightweight counterpart to :mod:`run` — it only loads the trained
prior flow and integrates the grid; it does **not** (re)train.  The per-target
PSF/noise condition Σ_X is built from the 2×2 ``[2:4, 2:4]`` sub-block of each
target's moment covariance via :func:`bfd_cnf.models.flows.cx_to_sx_cond`.

Usage
-----
    python -m bfd_cnf.integrate_grid [--n-targets N] [--n-points 1024]
        [--n-replicates 16] [--batch-size 512] [--out pqr_grid.npz]
        [--stats-file PATH] [--rebuild-stats]

The first run reads the (large) FITS template table once to recover the
standardiser statistics the flow was trained with, then caches them to
``--stats-file`` so subsequent runs start quickly.  Use ``--rebuild-stats`` to
force a refresh.
"""

from __future__ import annotations

import argparse
import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .config import (
    DATA_DIR,
    GRID_M_PATH,
    GRID_P_PATH,
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    key as _base_key,
    target_flux_min,
)
from .inference import integrate_grid_pqr, load_grid_data
from .models.bijections import RawMomentStandardize
from .models.flows import build_flows
from .statistics import bootstrap_total_mult_bias, pqr2g, pqr2multbias
from .training import load_models

DEFAULT_STATS_FILE = os.path.join(DATA_DIR, "raw2standard_stats.npz")


def load_raw2standard(
    stats_file: str = DEFAULT_STATS_FILE, rebuild: bool = False
) -> RawMomentStandardize:
    """Return the :class:`RawMomentStandardize` matching the trained flow.

    The flow must be standardised with exactly the ``(mean, std)`` it was trained
    with.  Those come from :func:`bfd_cnf.data.load_data`, which reads the
    multi-GB FITS template table.  To avoid that cost on every integration run we
    cache the two 4-vectors to ``stats_file`` and reuse them.

    Parameters
    ----------
    stats_file : str, optional
        Path to the cached ``.npz`` of ``mean`` / ``std``.
    rebuild : bool, optional
        Force rebuilding from the FITS even if the cache exists.

    Returns
    -------
    RawMomentStandardize
    """
    if (not rebuild) and os.path.exists(stats_file):
        d = np.load(stats_file)
        print(f"Loaded standardiser stats from {stats_file}")
        return RawMomentStandardize(
            mean=jnp.asarray(d["mean"]), std=jnp.asarray(d["std"])
        )

    print("Rebuilding standardiser stats from the FITS template table "
          "(this reads the large FITS once)...")
    from .data import load_data

    data = load_data(key=_base_key)
    np.savez(
        stats_file,
        mean=np.asarray(data["data_mean"]),
        std=np.asarray(data["data_std"]),
    )
    print(f"Saved standardiser stats to {stats_file}")
    return data["raw2standard"]


def load_prior_flow(key, prior_path: str = PRIOR_FLOW_PATH, q_path: str = Q_FLOW_PATH):
    """Build the flow architecture (config defaults) and load trained weights.

    Only the prior flow is needed for PQR integration; the q flow is built and
    deserialised because :func:`bfd_cnf.training.load_models` expects both
    template pytrees, but it is otherwise unused here.
    """
    prior_flow, q_flow = build_flows(key, latent_dim=4, cond_dim=16)
    prior_trained, _ = load_models(prior_flow, q_flow, prior_path, q_path)
    return prior_trained


def main() -> None:
    """Parse CLI args, run the grid integration, report and save results."""
    ap = argparse.ArgumentParser(
        description="Flow-based RQMC PQR integration over BFD grid targets."
    )
    ap.add_argument(
        "--n-targets", type=int, default=None,
        help="Subsample this many selected targets (default: all selected).",
    )
    ap.add_argument(
        "--flux-min", type=float, default=target_flux_min,
        help=f"Minimum template flux Mf for the selection cut "
             f"(default config target_flux_min={target_flux_min}).",
    )
    ap.add_argument("--n-points", type=int, default=2**10,
                    help="RQMC quadrature points per replicate (default 1024).")
    ap.add_argument("--n-replicates", type=int, default=16,
                    help="Independent RQMC replicates for SE (default 16).")
    ap.add_argument("--batch-size", type=int, default=512,
                    help="Templates processed per batch (default 512).")
    ap.add_argument("--out", type=str, default=os.path.join(DATA_DIR, "pqr_grid.npz"),
                    help="Output .npz path for the PQR arrays (default data/pqr_grid.npz).")
    ap.add_argument("--stats-file", type=str, default=DEFAULT_STATS_FILE,
                    help="Cache file for the standardiser (mean, std).")
    ap.add_argument("--rebuild-stats", action="store_true",
                    help="Rebuild the standardiser stats from the FITS.")
    ap.add_argument(
        "--fixed-log-scale", type=float, default=None,
        help="Diagnostic: use a single fixed Sigma_X log_scale for ALL targets "
             "instead of the per-target covariance-derived Sigma_X.",
    )
    ap.add_argument("--fixed-e1", type=float, default=0.0,
                    help="e1 used with --fixed-log-scale (default 0).")
    ap.add_argument("--fixed-e2", type=float, default=0.0,
                    help="e2 used with --fixed-log-scale (default 0).")
    args = ap.parse_args()

    fixed_sx_cond = None
    if args.fixed_log_scale is not None:
        fixed_sx_cond = (args.fixed_log_scale, args.fixed_e1, args.fixed_e2)

    key = _base_key

    raw2standard = load_raw2standard(args.stats_file, rebuild=args.rebuild_stats)

    key, k_flow = jr.split(key)
    prior_flow = load_prior_flow(k_flow)

    print("Loading galaxy grid data...")
    joined_grid = load_grid_data(GRID_P_PATH, GRID_M_PATH)

    key, k_int = jr.split(key)
    res = integrate_grid_pqr(
        joined_grid,
        raw2standard,
        prior_flow,
        key=k_int,
        n_targets=args.n_targets,
        flux_min=args.flux_min,
        n_points=args.n_points,
        n_replicates=args.n_replicates,
        batch_size=args.batch_size,
        fixed_sx_cond=fixed_sx_cond,
    )

    pqr_p, pqr_m = res["pqr_p"], res["pqr_m"]

    # Flow vs analytic-BFD shear recovery on the same targets.
    g_p, g_m = pqr2g(pqr_p), pqr2g(pqr_m)
    m_flow = pqr2multbias(jnp.concatenate([pqr_p, pqr_m], axis=-1))

    g_p_sim, g_m_sim = pqr2g(res["pqr_sim_p"]), pqr2g(res["pqr_sim_m"])
    m_sim = pqr2multbias(jnp.concatenate([res["pqr_sim_p"], res["pqr_sim_m"]], axis=-1))

    print("\n=== Shear recovery (expect g(+) ~ +0.02, g(-) ~ -0.02) ===")
    print(f"flow  g(+) = {np.asarray(g_p)}   g(-) = {np.asarray(g_m)}")
    print(f"BFD   g(+) = {np.asarray(g_p_sim)}   g(-) = {np.asarray(g_m_sim)}")
    print(f"flow  multiplicative bias m = {float(m_flow):+.4f}")
    print(f"BFD   multiplicative bias m = {float(m_sim):+.4f}")

    stats = bootstrap_total_mult_bias(
        pqr_p, pqr_m, n_boot=5000, delta_g=0.04, key=jr.PRNGKey(42)
    )
    print("\n=== Bootstrap multiplicative bias (flow) ===")
    print(
        f"m point={float(stats['m_point']):+.4f}  mean={float(stats['m_mean']):+.4f}  "
        f"std={float(stats['m_std']):.4f}  "
        f"16-84%=[{float(stats['m_p16']):+.4f}, {float(stats['m_p84']):+.4f}]  "
        f"n={stats['n_used']}"
    )

    np.savez(
        args.out,
        ids=np.asarray(res["ids"]),
        pqr_p=np.asarray(pqr_p),
        pqr_m=np.asarray(pqr_m),
        pqr_sim_p=np.asarray(res["pqr_sim_p"]),
        pqr_sim_m=np.asarray(res["pqr_sim_m"]),
        sx_conds_p=np.asarray(res["sx_conds_p"]),
        sx_conds_m=np.asarray(res["sx_conds_m"]),
        targets_p=np.asarray(res["targets_p"]),
        targets_m=np.asarray(res["targets_m"]),
    )
    print(f"\nSaved PQR results to {args.out}  "
          "(columns [P, Q1, Q2, R11, R22, R12]).")


if __name__ == "__main__":
    main()
