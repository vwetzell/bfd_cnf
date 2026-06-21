"""
analyze_mbias_split.py
======================
Localise the residual multiplicative bias ``m`` from a saved PQR grid
integration (``data/pqr_grid_*.npz``) by splitting it across target
properties: centroid-ellipticity ``|e|``, ``log_scale`` (½ log det C_X), and
flux ``Mf``.

For any subset of targets the multiplicative bias is

    m_subset = (g1(+) - g1(-)) / Δg - 1,   Δg = 0.04

with ``g(±)`` the maximum-likelihood shear from the subset's *summed* PQR
(:func:`bfd_cnf.statistics.pqr2g`).  Because ``m`` is a ratio of sums it is not
additive across bins, so we report two complementary views:

  * **per-bin m** — the bias carried by each population (with bootstrap error);
  * **cumulative |e|-cut m(|e|<c)** — the directly actionable "does excluding
    the extrapolating tail move the total m toward zero?" curve.

We also rank targets by leverage (|Q1|/P, the per-object contribution to the
shear numerator) to see whether a handful of extrapolating objects dominate.

Run::

    python -m bfd_cnf.analyze_mbias_split --in data/pqr_grid_100k_adapt240k.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from .config import PLOTS_DIR, log_scale_range, e_max
from .statistics import bootstrap_total_mult_bias


def _subset_m(pqr_p, pqr_m, mask, n_boot=800, key=jr.PRNGKey(0)):
    """Point + bootstrap multiplicative bias on the masked subset."""
    if mask.sum() < 5:
        return dict(m_point=np.nan, m_std=np.nan, n=int(mask.sum()))
    s = bootstrap_total_mult_bias(
        jnp.asarray(pqr_p[mask]), jnp.asarray(pqr_m[mask]),
        n_boot=n_boot, delta_g=0.04, key=key,
    )
    return dict(m_point=float(s["m_point"]), m_std=float(s["m_std"]), n=int(mask.sum()))


def _binned(pqr_p, pqr_m, x, edges, key):
    """m per bin defined by ``edges`` on the per-target scalar ``x``."""
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (x >= lo) & (x < hi)
        r = _subset_m(pqr_p, pqr_m, mask, key=jr.fold_in(key, i))
        r.update(lo=lo, hi=hi, mid=0.5 * (lo + hi))
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/pqr_grid_100k_adapt240k.npz")
    ap.add_argument("--out", default=os.path.join(PLOTS_DIR, "mbias_split.png"))
    args = ap.parse_args()

    d = np.load(args.inp)
    pqr_p, pqr_m = d["pqr_p"], d["pqr_m"]
    sx, tp = d["sx_conds_p"], d["targets_p"]
    emag = np.sqrt(sx[:, 1] ** 2 + sx[:, 2] ** 2)
    ls = sx[:, 0]
    log10mf = np.log10(np.abs(tp[:, 0]))

    # shared validity mask (matches bootstrap_total_mult_bias internals)
    valid = (
        np.all(np.isfinite(pqr_p), axis=1) & np.all(np.isfinite(pqr_m), axis=1)
        & (pqr_p[:, 0] > 0) & (pqr_m[:, 0] > 0)
    )
    print(f"valid targets: {valid.sum()} / {len(valid)}")
    key = jr.PRNGKey(42)

    # ---- overall + in/out of trained range ----
    all_m = _subset_m(pqr_p, pqr_m, valid, key=key)
    ls_lo, ls_hi = log_scale_range
    inrange = valid & (emag <= e_max) & (ls >= ls_lo) & (ls <= ls_hi)
    outrange = valid & ~((emag <= e_max) & (ls >= ls_lo) & (ls <= ls_hi))
    in_m = _subset_m(pqr_p, pqr_m, inrange, key=jr.fold_in(key, 1))
    out_m = _subset_m(pqr_p, pqr_m, outrange, key=jr.fold_in(key, 2))
    print(f"\nALL          m = {all_m['m_point']:+.4f} ± {all_m['m_std']:.4f}  (n={all_m['n']})")
    print(f"IN-range     m = {in_m['m_point']:+.4f} ± {in_m['m_std']:.4f}  (n={in_m['n']})  "
          f"[|e|<={e_max}, {ls_lo}<=log_scale<={ls_hi}]")
    print(f"OUT-of-range m = {out_m['m_point']:+.4f} ± {out_m['m_std']:.4f}  (n={out_m['n']})")

    # ---- per-bin splits ----
    e_edges = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, 0.21]
    ls_edges = [10.4, 11.0, 11.25, 11.5, 11.75, 12.0, 12.5, 13.0, 14.8]
    mf_edges = [3.25, 3.5, 3.65, 3.8, 4.0, 4.25, 5.0]
    e_rows = _binned(pqr_p[valid], pqr_m[valid], emag[valid], e_edges, jr.fold_in(key, 10))
    ls_rows = _binned(pqr_p[valid], pqr_m[valid], ls[valid], ls_edges, jr.fold_in(key, 20))
    mf_rows = _binned(pqr_p[valid], pqr_m[valid], log10mf[valid], mf_edges, jr.fold_in(key, 30))

    def _print_rows(title, rows, fmt):
        print(f"\n--- m by {title} ---")
        for r in rows:
            if np.isfinite(r["m_point"]):
                print(f"  {fmt.format(r['lo'], r['hi']):>16s}  "
                      f"m={r['m_point']:+.4f} ± {r['m_std']:.4f}  n={r['n']}")
    _print_rows("|e|", e_rows, "[{:.3f},{:.3f})")
    _print_rows("log_scale", ls_rows, "[{:.2f},{:.2f})")
    _print_rows("log10 Mf", mf_rows, "[{:.2f},{:.2f})")

    # ---- cumulative |e|<c cut ----
    cuts = [0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.21]
    print("\n--- cumulative m(|e| < c) ---")
    cum = []
    for j, c in enumerate(cuts):
        mask = valid & (emag < c)
        r = _subset_m(pqr_p, pqr_m, mask, key=jr.fold_in(key, 40 + j))
        cum.append((c, r))
        print(f"  |e|<{c:.3f}:  m={r['m_point']:+.4f} ± {r['m_std']:.4f}  "
              f"(n={r['n']}, {100*r['n']/valid.sum():.2f}% of valid)")

    # ---- leverage: which targets dominate the +shear numerator |Q1|/P ----
    lev = np.where(valid, np.abs(pqr_p[:, 1]) / np.maximum(pqr_p[:, 0], 1e-30), 0.0)
    top = np.argsort(lev)[::-1][:50]
    print("\n--- top-50 leverage (|Q1|/P, +shear) target properties ---")
    print(f"  median |e| of top-50  = {np.median(emag[top]):.4f}  (vs all median {np.median(emag[valid]):.4f})")
    print(f"  median log_scale top50= {np.median(ls[top]):.3f}   (vs all {np.median(ls[valid]):.3f})")
    print(f"  frac of top-50 with |e|>{e_max} = {100*np.mean(emag[top]>e_max):.1f}%  "
          f"(vs {100*np.mean(emag[valid]>e_max):.2f}% overall)")

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
