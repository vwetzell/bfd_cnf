"""How much does the corrected m1 move when one window boundary is nudged?

Four panels, one per boundary of the nominal window
(`2500 < Mf < 50000`, `2.2 < Mr/Mf < 3.2`); in each, that boundary is scanned
over `+/- 2%`, `+/- 4%` while the other three are held nominal.  A boundary the
answer is sensitive to at the percent level is a boundary the analysis cannot
place by eye.

Two error bars, and they answer different questions:

  * the ABSOLUTE bar on each point is the galaxy bootstrap and the selection
    correction's own block bootstrap added in quadrature, i.e. "what is m1 in
    this window";
  * the DRIFT against nominal is measured PAIRED -- one galaxy resample and one
    prior-block resample evaluated at all five windows of a panel -- because the
    windows are overlapping subsets of one target catalog and one prior sample,
    so their absolute bars are heavily correlated and their quadrature
    difference is the wrong yardstick for a shift.  `dev/size_offline.py` makes
    the same distinction for nested size ceilings.

Eq. (40)'s terms come from the FLOW (`dev/size_bands.prior_draw`), the same
prior the targets' Q and R came from.

    PYTHONPATH=. python dev/window_shift.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

import bias                                              # noqa: E402
from bias import load_cov, window_mask                   # noqa: E402
from dev.size_bands import (per_template, prior_draw,    # noqa: E402
                            stencil_moments)

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
PQR = os.environ.get("PQR", "dev/pqr_200k.npz")
G = float(os.environ.get("G", 0.02))
NBLOCK = int(os.environ.get("NBLOCK", 20))
BOOT = int(os.environ.get("BOOT", 400))
OUT = os.environ.get("OUT", "plots/window_shift.png")

NOMINAL = dict(flux_lo=2500.0, flux_hi=50000.0, size_lo=2.2, size_hi=3.2)
FRAC = [-0.04, -0.02, 0.0, 0.02, 0.04]
LABEL = {"flux_lo": "Mf floor", "flux_hi": "Mf ceiling",
         "size_lo": "Mr/Mf floor", "size_hi": "Mr/Mf ceiling"}


def window(**kw):
    w = dict(NOMINAL, **kw)
    return (w["size_lo"], w["size_hi"]), (w["flux_lo"], w["flux_hi"])


if __name__ == "__main__":
    d = np.load(PQR)
    ok = np.ones(len(d["plus_q"]), dtype=bool)
    for k in ("plus", "minus"):
        ok &= np.isfinite(d[f"{k}_q"]).all(1)
        ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    qp, rp = d["plus_q"][ok], d["plus_r"][ok]
    qm, rm = d["minus_q"][ok], d["minus_r"][ok]
    obs_p, obs_m = d["obs_plus"][ok], d["obs_minus"][ok]
    cov = load_cov(f"{D}/{bias.CATALOGS[POP]['zero']}.fits")
    # The nine stencil moment sets are window-INDEPENDENT, so the prior (the
    # flow, in float64) is pushed through once and every window is then just a
    # `window_prob` over the same array.  Without that, each window recompiles
    # the whole bijection stack and the accumulated CUBINs OOM this GPU.
    draw, z = prior_draw()
    ms = stencil_moments(draw, z)
    parts = np.array_split(np.arange(ms.shape[1]), NBLOCK)

    cases, seen = {}, {}
    for key in NOMINAL:
        cases[key] = []
        for f in FRAC:
            v = NOMINAL[key] * (1.0 + f)
            size, flux = window(**{key: v})
            tag = (size, flux)
            if tag not in seen:
                sp = window_mask(obs_p, size, flux)
                sm = window_mask(obs_m, size, flux)
                pt = per_template(ms, cov, size, flux)
                blocks = [tuple(x[p].mean(0) for x in pt) for p in parts]
                full = tuple(x.mean(0) for x in pt)
                top = (np.abs(pt[2][:, 0, 0]).max()
                       / (ms.shape[1] * abs(full[2][0, 0])))
                seen[tag] = (sp, sm, int((~sp).sum()), int((~sm).sum()),
                             blocks, full, top)
                print(f"  {LABEL[key]:>14s} {v:9.4g}: kept {sp.mean():6.2%}  "
                      f"R_s11 {full[2][0, 0]:+.4f}  top template {top:.1%}",
                      flush=True)
            cases[key].append((v, tag))

    ns = lambda t, s: ((s[2], *t), (s[3], *t))
    point = {tag: bias.bias(qp, rp, qm, rm, G, sel=(s[0], s[1]),
                            ns=ns(s[5], s))[0] for tag, s in seen.items()}

    # Paired bootstrap: ONE galaxy resample and ONE block resample per draw,
    # evaluated at every window.
    rng = np.random.default_rng(0)
    n = len(qp)
    draws = {tag: np.empty(BOOT) for tag in seen}
    for b in range(BOOT):
        idx = rng.integers(0, n, n)
        kk = rng.integers(0, NBLOCK, NBLOCK)
        qpb, rpb, qmb, rmb = qp[idx], rp[idx], qm[idx], rm[idx]
        for tag, s in seen.items():
            sp, sm = s[0][idx], s[1][idx]
            t = tuple(np.mean([s[4][i][j] for i in kk], axis=0) for j in range(3))
            draws[tag][b] = bias.bias(qpb, rpb, qmb, rmb, G, sel=(sp, sm),
                                      ns=((int((~sp).sum()), *t),
                                          (int((~sm).sum()), *t)))[0]
        if (b + 1) % 100 == 0:
            print(f"    bootstrap {b + 1}/{BOOT}", flush=True)

    nom_tag = cases["flux_lo"][2][1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey=True)
    print(f"\n{'boundary':>14s}{'value':>10s}{'kept':>8s}{'m1':>10s}"
          f"{'+/- abs':>9s}{'drift':>9s}{'+/- paired':>11s}")
    for ax, key in zip(axes.ravel(), NOMINAL):
        xs, ys, es, ds, dse = [], [], [], [], []
        for v, tag in cases[key]:
            s = seen[tag]
            xs.append(v)
            ys.append(point[tag])
            es.append(draws[tag].std(ddof=1))
            ds.append(point[tag] - point[nom_tag])
            dse.append((draws[tag] - draws[nom_tag]).std(ddof=1))
            print(f"{LABEL[key]:>14s}{v:10.4g}{s[0].mean():8.2%}"
                  f"{ys[-1]:+10.5f}{es[-1]:9.5f}{ds[-1]:+9.5f}{dse[-1]:11.5f}")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.axvline(NOMINAL[key], color="0.8", lw=0.8, ls="--")
        ax.axhspan(point[nom_tag] - draws[nom_tag].std(ddof=1),
                   point[nom_tag] + draws[nom_tag].std(ddof=1),
                   color="C0", alpha=0.12, lw=0)
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color="C0", ms=4, capsize=3,
                    lw=1.2, label="absolute (galaxy + selection MC)")
        # The inner bar is what actually answers "does moving this boundary
        # change the answer": the paired error on the shift against nominal.
        ax.errorbar(xs, ys, yerr=dse, fmt="none", color="C3", lw=3.0,
                    alpha=0.75, label="paired, vs nominal")
        ax.set_xticks(xs)
        ax.set_xlabel(f"{LABEL[key]}   (nominal {NOMINAL[key]:g}, $\\pm$4%)")
        ax.set_ylabel("corrected $m_1$")
        ax.set_title(LABEL[key])
    axes[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("Corrected $m_1$ vs each window boundary, others held nominal"
                 "\nnominal window: $2500 < M_f < 50000$, "
                 "$2.2 < M_r/M_f < 3.2$   (band = nominal $\\pm 1\\sigma$)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"\nwrote {OUT}")
