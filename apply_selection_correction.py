"""
apply_selection_correction.py
=============================
Apply the BFD-2016 flux-limit **selection correction** to an already-integrated
grid catalogue and report the shear / multiplicative bias *with and without* it.

The selection term ``P_sel(g)`` (the probability a galaxy from the sheared population
lands in the flux band ``[f_min, f_max]`` under its measurement noise) is computed from
the trained prior flow, swept over a histogram of the targets' flux-moment noise
``σ_f,i = sqrt(Σ_i[0,0])`` (see ``bfd_cnf.inference.selection_pqr_binned``), then
subtracted **per object** from each arm's PQR log-totals (``statistics.apply_selection``
and the selection-aware ``bootstrap_independent_mult_bias``).

σ_f is not stored in the integrated npz (it only has ``pqr_*``/``targets_*``/``ids_*``),
so it is sourced from the SOURCE grid catalogues (``GRID_P_PATH``/``GRID_M_PATH``, which
carry ``covariance``), matched to the integrated targets by ``id``.  ``--sigma-f`` gives
a single representative value as a fallback.

Run:
    python apply_selection_correction.py --in data/pqr_grid.npz \
        --source-p data/grid_p.npz --source-m data/grid_m.npz --f-min 1500 --f-max 90000
    python apply_selection_correction.py --in data/pqr_grid.npz --sigma-f 220
"""

from __future__ import annotations

import argparse

import numpy as np


def _sigma_f_for_ids(source_cat, ids):
    """Per-target σ_f = sqrt(Σ[0,0]) from a source catalogue, matched by ``id``."""
    from bfd_cnf.inference import unpack_packed_cov

    src_id = np.asarray(source_cat["id"])
    sig = np.sqrt(unpack_packed_cov(np.asarray(source_cat["covariance"]))[:, 0, 0])
    order = np.argsort(src_id)
    pos = np.searchsorted(src_id[order], ids)
    matched = order[np.clip(pos, 0, len(order) - 1)]
    if not np.array_equal(src_id[matched], ids):
        raise ValueError("some integrated ids are absent from the source catalogue")
    return sig[matched].astype(np.float64)


def _bins_from_sigma_f(sigma_f, n_bins):
    """Compact non-empty σ_f bins: (rep_sigma_f, counts, bin_idx) row-aligned to input."""
    sigma_f = np.asarray(sigma_f, dtype=np.float64)
    if sigma_f.size == 0:
        raise ValueError("no targets to bin")
    edges = np.histogram_bin_edges(sigma_f, bins=n_bins)
    bin_idx = np.clip(np.digitize(sigma_f, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    sums = np.bincount(bin_idx, weights=sigma_f, minlength=n_bins)
    rep = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    present = counts > 0
    remap = np.full(n_bins, -1, dtype=int)
    remap[present] = np.arange(int(present.sum()))
    return rep[present], counts[present], remap[bin_idx]


def _selection_terms(flow, r2s, f_min, f_max, sigma_f, n_bins, n_samples, key, single):
    """Per-object selection log-totals (sel_qtot (N,2), sel_rtot (N,2,2)) for one arm."""
    from bfd_cnf.inference import selection_pqr_binned

    if single is not None:
        rep = np.array([float(single)])
        counts = np.array([float(sigma_f.size)])
        bin_idx = np.zeros(sigma_f.size, dtype=int)
    else:
        rep, counts, bin_idx = _bins_from_sigma_f(sigma_f, n_bins)
    out = selection_pqr_binned(
        flow, r2s, f_min, f_max, rep, counts, n_samples=n_samples, key=key
    )
    q_tot_b = np.asarray(out["q_tot_b"])
    r_tot_b = np.asarray(out["r_tot_b"])
    return q_tot_b[bin_idx], r_tot_b[bin_idx], rep, counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--in",
        dest="inp",
        default="data/pqr_grid.npz",
        help="integrated grid npz (pqr_p/pqr_m/ids_p/ids_m).",
    )
    ap.add_argument(
        "--source-p", default=None, help="source +arm npz (default GRID_P_PATH)."
    )
    ap.add_argument(
        "--source-m", default=None, help="source -arm npz (default GRID_M_PATH)."
    )
    ap.add_argument(
        "--prior",
        default=None,
        help="prior flow .eqx (default config.PRIOR_FLOW_PATH).",
    )
    ap.add_argument(
        "--q-flow", default=None, help="q flow .eqx (default config.Q_FLOW_PATH)."
    )
    ap.add_argument(
        "--f-min",
        type=float,
        default=None,
        help="flux band lower edge (default config.target_flux_min).",
    )
    ap.add_argument(
        "--f-max", type=float, default=90000.0, help="flux band upper edge."
    )
    ap.add_argument(
        "--sigma-f",
        type=float,
        default=None,
        help="single representative σ_f (skip the source catalogue / histogram).",
    )
    ap.add_argument(
        "--n-bins", type=int, default=16, help="σ_f histogram bins (default 16)."
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=2**16,
        help="flow draws for the selection term (default 65536).",
    )
    ap.add_argument(
        "--n-boot", type=int, default=2000, help="bootstrap resamples (default 2000)."
    )
    ap.add_argument(
        "--delta-g", type=float, default=0.04, help="g_plus - g_minus (default 0.04)."
    )
    args = ap.parse_args()

    import jax.random as jr

    from bfd_cnf.config import (
        GRID_M_PATH,
        GRID_P_PATH,
        PRIOR_FLOW_PATH,
        Q_FLOW_PATH,
        target_flux_min,
    )
    from bfd_cnf.config import (
        key as base_key,
    )
    from bfd_cnf.integrate_grid import load_prior_flow
    from bfd_cnf.models.bijections import load_stats
    from bfd_cnf.statistics import (
        apply_selection,
        bootstrap_independent_mult_bias,
        pqr2g,
    )

    f_min = target_flux_min if args.f_min is None else args.f_min
    f_max = args.f_max

    d = np.load(args.inp)
    pqr_p = np.asarray(d["pqr_p"], dtype=np.float64)
    pqr_m = np.asarray(d["pqr_m"], dtype=np.float64)

    # Cut the input catalogue to the same flux band used for the selection term:
    # keep targets whose measured flux moment Mf (targets_*[:, 0]) is in (f_min, f_max).
    ids_p = np.asarray(d["ids_p"])
    ids_m = np.asarray(d["ids_m"])
    mf_p = np.asarray(d["targets_p"])[:, 0]
    mf_m = np.asarray(d["targets_m"])[:, 0]
    keep_p = (mf_p > f_min) & (mf_p < f_max)
    keep_m = (mf_m > f_min) & (mf_m < f_max)
    pqr_p, ids_p = pqr_p[keep_p], ids_p[keep_p]
    pqr_m, ids_m = pqr_m[keep_m], ids_m[keep_m]

    prior_path = args.prior or PRIOR_FLOW_PATH
    q_path = args.q_flow or Q_FLOW_PATH
    r2s = load_stats(prior_path)
    flow = load_prior_flow(base_key, prior_path, q_path)
    kp, km = jr.split(base_key)

    if args.sigma_f is not None:
        sig_p = np.zeros(pqr_p.shape[0])
        sig_m = np.zeros(pqr_m.shape[0])
    else:
        src_p = np.load(args.source_p or GRID_P_PATH)
        src_m = np.load(args.source_m or GRID_M_PATH)
        sig_p = _sigma_f_for_ids(src_p, ids_p)
        sig_m = _sigma_f_for_ids(src_m, ids_m)

    selq_p, selr_p, rep_p, cnt_p = _selection_terms(
        flow, r2s, f_min, f_max, sig_p, args.n_bins, args.n_samples, kp, args.sigma_f
    )
    selq_m, selr_m, rep_m, cnt_m = _selection_terms(
        flow, r2s, f_min, f_max, sig_m, args.n_bins, args.n_samples, km, args.sigma_f
    )

    # Point shear per arm (with vs without correction) and the selection-aware
    # bootstrap, which also yields the additive bias c1, c2 and their errors.
    g_p0, g_m0 = pqr2g(pqr_p), pqr2g(pqr_m)
    g_p1 = apply_selection(pqr_p, selq_p, selr_p)
    g_m1 = apply_selection(pqr_m, selq_m, selr_m)

    b0 = bootstrap_independent_mult_bias(
        pqr_p, pqr_m, n_boot=args.n_boot, delta_g=args.delta_g
    )
    b1 = bootstrap_independent_mult_bias(
        pqr_p,
        pqr_m,
        n_boot=args.n_boot,
        delta_g=args.delta_g,
        sel_qtot_p=selq_p,
        sel_rtot_p=selr_p,
        sel_qtot_m=selq_m,
        sel_rtot_m=selr_m,
    )

    def _row(b):  # "m ± σ   c1 ± σ   c2 ± σ" from a bootstrap dict
        return (
            f"m ={float(b['m_point']):+.5f} ± {float(b['m_std']):.5f}   "
            f"c1={float(b['c1_point']):+.5f} ± {float(b['c1_std']):.5f}   "
            f"c2={float(b['c2_point']):+.5f} ± {float(b['c2_std']):.5f}"
        )

    print(
        f"flux band: {f_min:g} < Mf < {f_max:g}   n=({pqr_p.shape[0]}, {pqr_m.shape[0]})"
    )
    src = "single σ_f" if args.sigma_f is not None else "source-catalogue histogram"
    print(
        f"σ_f source: {src}   bins=({rep_p.size}, {rep_m.size})   "
        f"σ_f range +arm=[{rep_p.min():.1f}, {rep_p.max():.1f}]"
    )
    print(
        f"  g(+) uncorr=({g_p0[0]:+.5f},{g_p0[1]:+.5f})  +sel=({g_p1[0]:+.5f},{g_p1[1]:+.5f})"
    )
    print(
        f"  g(-) uncorr=({g_m0[0]:+.5f},{g_m0[1]:+.5f})  +sel=({g_m1[0]:+.5f},{g_m1[1]:+.5f})"
    )
    print(f"  uncorrected  {_row(b0)}")
    print(f"  + selection  {_row(b1)}")
    print(
        f"  Δ (selection)  m ={float(b1['m_point'] - b0['m_point']):+.5f}         "
        f"   c1={float(b1['c1_point'] - b0['c1_point']):+.5f}            "
        f"   c2={float(b1['c2_point'] - b0['c2_point']):+.5f}"
    )


if __name__ == "__main__":
    main()
