"""
analyze_mbias_split.py
======================
Localise the residual multiplicative bias ``m`` from a saved PQR grid
integration (``data/pqr_grid_*.npz``, the independent-ensemble output of
``integrate_grid.py`` / ``run.py``) by splitting it across target properties:
centroid-ellipticity ``|e|``, ``log_scale`` (½ log det C_X), and flux ``Mf``.

The +shear and -shear catalogues are independent injection realisations (not
ring pairs), with different lengths and their own ``sx_conds_p/m`` and
``targets_p/m``, so every split below bins the two sides independently on
their own properties before combining them into
:func:`bfd_cnf.statistics.bootstrap_independent_mult_bias`.  For any such pair
of subsets the multiplicative bias is

    m_subset = (g1(+) - g1(-)) / Δg - 1,   Δg = 0.04

with ``g(±)`` the maximum-likelihood shear from each subset's *summed* PQR
(:func:`bfd_cnf.statistics.pqr2g`).  Because ``m`` is a ratio of sums it is not
additive across bins, so we report two complementary views:

  * **per-bin m** — the bias carried by each population (with bootstrap error);
  * **cumulative |e|-cut m(|e|<c)** — the directly actionable "does excluding
    the extrapolating tail move the total m toward zero?" curve.

We also rank +shear targets by leverage (|Q1|/P, the per-object contribution
to the shear numerator) to see whether a handful of extrapolating objects
dominate.

Run::

    python -m bfd_cnf.analyze_mbias_split --in data/pqr_grid_independent.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from .config import PLOTS_DIR, log_scale_range, e_max
from .statistics import bootstrap_independent_mult_bias


def _subset_m(pqr_p, pqr_m, mask_p, mask_m, n_boot=800, key=jr.PRNGKey(0)):
    """Point + bootstrap multiplicative bias on the masked subsets (independent arms)."""
    if mask_p.sum() < 5 or mask_m.sum() < 5:
        return dict(m_point=np.nan, m_std=np.nan, n_p=int(mask_p.sum()), n_m=int(mask_m.sum()))
    s = bootstrap_independent_mult_bias(
        jnp.asarray(pqr_p[mask_p]), jnp.asarray(pqr_m[mask_m]),
        n_boot=n_boot, delta_g=0.04, key=key,
    )
    return dict(
        m_point=float(s["m_point"]), m_std=float(s["m_std"]),
        n_p=int(mask_p.sum()), n_m=int(mask_m.sum()),
    )


def _binned(pqr_p, pqr_m, x_p, x_m, edges, key):
    """m per bin defined by ``edges`` on each side's own per-target scalar."""
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask_p = (x_p >= lo) & (x_p < hi)
        mask_m = (x_m >= lo) & (x_m < hi)
        r = _subset_m(pqr_p, pqr_m, mask_p, mask_m, key=jr.fold_in(key, i))
        r.update(lo=lo, hi=hi, mid=0.5 * (lo + hi))
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_grid_independent.npz")
    ap.add_argument("--out", default=os.path.join(PLOTS_DIR, "mbias_split.png"))
    args = ap.parse_args()

    d = np.load(args.inp)
    pqr_p, pqr_m = d["pqr_p"], d["pqr_m"]
    sx_p, tp = d["sx_conds_p"], d["targets_p"]
    sx_m, tm = d["sx_conds_m"], d["targets_m"]
    emag_p = np.sqrt(sx_p[:, 1] ** 2 + sx_p[:, 2] ** 2)
    emag_m = np.sqrt(sx_m[:, 1] ** 2 + sx_m[:, 2] ** 2)
    ls_p, ls_m = sx_p[:, 0], sx_m[:, 0]
    log10mf_p = np.log10(np.abs(tp[:, 0]))
    log10mf_m = np.log10(np.abs(tm[:, 0]))

    # per-arm validity mask (matches bootstrap_independent_mult_bias internals)
    valid_p = np.all(np.isfinite(pqr_p), axis=1) & (pqr_p[:, 0] > 0)
    valid_m = np.all(np.isfinite(pqr_m), axis=1) & (pqr_m[:, 0] > 0)
    print(f"valid +shear targets: {valid_p.sum()} / {len(valid_p)}")
    print(f"valid -shear targets: {valid_m.sum()} / {len(valid_m)}")
    key = jr.PRNGKey(42)

    # ---- overall + in/out of trained range ----
    all_m = _subset_m(pqr_p, pqr_m, valid_p, valid_m, key=key)
    ls_lo, ls_hi = log_scale_range
    inrange_p = valid_p & (emag_p <= e_max) & (ls_p >= ls_lo) & (ls_p <= ls_hi)
    inrange_m = valid_m & (emag_m <= e_max) & (ls_m >= ls_lo) & (ls_m <= ls_hi)
    outrange_p = valid_p & ~((emag_p <= e_max) & (ls_p >= ls_lo) & (ls_p <= ls_hi))
    outrange_m = valid_m & ~((emag_m <= e_max) & (ls_m >= ls_lo) & (ls_m <= ls_hi))
    in_m = _subset_m(pqr_p, pqr_m, inrange_p, inrange_m, key=jr.fold_in(key, 1))
    out_m = _subset_m(pqr_p, pqr_m, outrange_p, outrange_m, key=jr.fold_in(key, 2))
    print(f"\nALL          m = {all_m['m_point']:+.4f} ± {all_m['m_std']:.4f}  "
          f"(n_p={all_m['n_p']}, n_m={all_m['n_m']})")
    print(f"IN-range     m = {in_m['m_point']:+.4f} ± {in_m['m_std']:.4f}  "
          f"(n_p={in_m['n_p']}, n_m={in_m['n_m']})  "
          f"[|e|<={e_max}, {ls_lo}<=log_scale<={ls_hi}]")
    print(f"OUT-of-range m = {out_m['m_point']:+.4f} ± {out_m['m_std']:.4f}  "
          f"(n_p={out_m['n_p']}, n_m={out_m['n_m']})")

    # ---- per-bin splits ----
    e_edges = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, 0.21]
    ls_edges = [10.4, 11.0, 11.25, 11.5, 11.75, 12.0, 12.5, 13.0, 14.8]
    mf_edges = [3.25, 3.5, 3.65, 3.8, 4.0, 4.25, 5.0]
    e_rows = _binned(pqr_p[valid_p], pqr_m[valid_m], emag_p[valid_p], emag_m[valid_m],
                      e_edges, jr.fold_in(key, 10))
    ls_rows = _binned(pqr_p[valid_p], pqr_m[valid_m], ls_p[valid_p], ls_m[valid_m],
                       ls_edges, jr.fold_in(key, 20))
    mf_rows = _binned(pqr_p[valid_p], pqr_m[valid_m], log10mf_p[valid_p], log10mf_m[valid_m],
                       mf_edges, jr.fold_in(key, 30))

    def _print_rows(title, rows, fmt):
        print(f"\n--- m by {title} ---")
        for r in rows:
            if np.isfinite(r["m_point"]):
                print(f"  {fmt.format(r['lo'], r['hi']):>16s}  "
                      f"m={r['m_point']:+.4f} ± {r['m_std']:.4f}  n_p={r['n_p']} n_m={r['n_m']}")
    _print_rows("|e|", e_rows, "[{:.3f},{:.3f})")
    _print_rows("log_scale", ls_rows, "[{:.2f},{:.2f})")
    _print_rows("log10 Mf", mf_rows, "[{:.2f},{:.2f})")

    # ---- cumulative |e|<c cut ----
    cuts = [0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.21]
    print("\n--- cumulative m(|e| < c) ---")
    cum = []
    for j, c in enumerate(cuts):
        mask_p = valid_p & (emag_p < c)
        mask_m = valid_m & (emag_m < c)
        r = _subset_m(pqr_p, pqr_m, mask_p, mask_m, key=jr.fold_in(key, 40 + j))
        cum.append((c, r))
        print(f"  |e|<{c:.3f}:  m={r['m_point']:+.4f} ± {r['m_std']:.4f}  "
              f"(n_p={r['n_p']}, {100*r['n_p']/valid_p.sum():.2f}% of valid +shear)")

    # ---- leverage: which +shear targets dominate the numerator |Q1|/P ----
    lev = np.where(valid_p, np.abs(pqr_p[:, 1]) / np.maximum(pqr_p[:, 0], 1e-30), 0.0)
    top = np.argsort(lev)[::-1][:50]
    print("\n--- top-50 leverage (|Q1|/P, +shear) target properties ---")
    print(f"  median |e| of top-50  = {np.median(emag_p[top]):.4f}  (vs all median {np.median(emag_p[valid_p]):.4f})")
    print(f"  median log_scale top50= {np.median(ls_p[top]):.3f}   (vs all {np.median(ls_p[valid_p]):.3f})")
    print(f"  frac of top-50 with |e|>{e_max} = {100*np.mean(emag_p[top]>e_max):.1f}%  "
          f"(vs {100*np.mean(emag_p[valid_p]>e_max):.2f}% overall)")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    def _plot_bins(a, rows, xlabel, vlines=()):
        x = [r["mid"] for r in rows if np.isfinite(r["m_point"])]
        y = [r["m_point"] for r in rows if np.isfinite(r["m_point"])]
        e = [r["m_std"] for r in rows if np.isfinite(r["m_point"])]
        a.errorbar(x, y, yerr=e, fmt="o-", capsize=3)
        a.axhline(0, color="k", lw=1)
        a.axhline(all_m["m_point"], color="C3", ls="--", lw=1, label=f"all m={all_m['m_point']:+.3f}")
        for v in vlines:
            a.axvline(v, color="green", ls=":", lw=1.2)
        a.set_xlabel(xlabel); a.set_ylabel("m"); a.grid(alpha=0.3); a.legend(fontsize=8)

    _plot_bins(ax[0, 0], e_rows, "centroid |e|  (green: e_max)", vlines=(e_max,))
    ax[0, 0].set_title("m vs |e|")
    _plot_bins(ax[0, 1], ls_rows, "log_scale  (green: trained range)", vlines=(ls_lo, ls_hi))
    ax[0, 1].set_title("m vs log_scale")
    _plot_bins(ax[1, 0], mf_rows, "log10 Mf (flux)")
    ax[1, 0].set_title("m vs flux")

    cx = [c for c, _ in cum]; cy = [r["m_point"] for _, r in cum]; ce = [r["m_std"] for _, r in cum]
    ax[1, 1].errorbar(cx, cy, yerr=ce, fmt="o-", capsize=3)
    ax[1, 1].axhline(0, color="k", lw=1)
    ax[1, 1].axvline(e_max, color="green", ls=":", lw=1.2, label=f"e_max={e_max}")
    ax[1, 1].set_xlabel("|e| cut c"); ax[1, 1].set_ylabel("m(|e| < c)")
    ax[1, 1].set_title("cumulative m vs |e| cut"); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"Residual multiplicative bias localisation — {os.path.basename(args.inp)}", fontsize=13)
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(args.out, dpi=130); plt.close(fig)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
