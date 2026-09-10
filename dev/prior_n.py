"""Is the residual the FLOW at all?  BFD's own template sum, paired, per cell.

`dev/edge_share.py` localises the flux-window residual by `Mr/Mf`, and this
answers the question that localisation raises: on the SAME targets, what does
BFD's eq. (35)-(36) sum over the 22.7M shifted copies give?  No flow, no
importance sampling, no gauge -- but the same prior population and the same
per-galaxy `dm_dg` / `d2m_dg2`.  Flow and templates are evaluated on identical
targets, so the PAIRED difference is far better determined than either error:
the population scatter that dominates both cancels.

    CELL=all  NTARG=40000 ESSMIN=1000 python dev/prior_n.py     # the headline
    CELL=edge FLUX_LO=1600 FLUX_HI=2941 python dev/prior_n.py   # by band

MEASURED (2026-09-02): +0.0040 +/- 0.0041 over the flux window, i.e. the
template sum reproduces the flow's -0.032 on those targets.  Per band it is
tighter still: 3.407-3.530 gives -0.1782 (templates) against -0.1767 (flow).

`ESSMIN` is not optional.  `C_M` is the same for every target, so a BRIGHT
target's noise ball is tiny against the template spacing and the copy sum is
starved -- median ESS 26 at `Mf >= 4778`, in the BULK as much as at the edge.
Below ~1000 the template numbers are noise (m1 = -7.9 +/- 5.4 in one cell).
That cut keeps the faint 37% of the flux window, and the paired difference is
a statement about those targets.

`NGAL` thins the copy catalog by galaxy, which was the original question here
(finite-prior error): it is NULL, and the Fisher ratio's flatness along the
flux axis in `dev/edge_share.py`'s census says the same.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bias                                             # noqa: E402
import band_tmpl                                        # noqa: E402
from band_tmpl import whitened_templates, template_pqr  # noqa: E402

PQR = os.environ.get("PQR", "dev/pqr_200k.npz")
POP = os.environ.get("POP", "bulgedisc_deep_v2")
G = float(os.environ.get("G", 0.02))
CELL = os.environ.get("CELL", "edge")
FLUX_LO = float(os.environ.get("FLUX_LO", 4778))
FLUX_HI = float(os.environ.get("FLUX_HI", 1e12))
NTARG = int(os.environ.get("NTARG", 400))
NGAL = [int(x) for x in os.environ.get("NGAL", "6250 12500 25000 50000 100000")
        .split()]
BOOT = int(os.environ.get("BOOT", 300))
SEED = int(os.environ.get("SEED", 0))
ESSMIN = float(os.environ.get("ESSMIN", 0))
CELLS = {"edge": (3.407, 3.530), "mid": (3.252, 3.407), "bulk": (0.0, 3.005),
         "all": (0.0, 9.0)}


def m1_boot(qp, rp, qm, rm, idx):
    """m1, its paired galaxy-bootstrap error, and the Fisher ratio.

    `SUM q^2 / SUM -r` is 1 for the density the targets were drawn from, needs
    no truth and no second run, and is the same statistic `dev/fisher_cross.py`
    cross-fits.  Here it is the PLAIN version -- for the flow at S >= 8192 the
    cross-fit says the Monte-Carlo inflation is < 0.3% outside the top 2%, and
    the template sum has no draw noise at all.
    """
    def val(i):
        return ((qp[i, 0].sum() - qm[i, 0].sum()) / (2 * G)
                / (0.5 * (-rp[i, 0, 0].sum() - rm[i, 0, 0].sum())) - 1)
    fish = (0.5 * (qp[:, 0] ** 2 + qm[:, 0] ** 2).sum()
            / (0.5 * (-rp[:, 0, 0] - rm[:, 0, 0]).sum()))
    return val(np.arange(len(qp))), np.std([val(i) for i in idx], ddof=1), fish


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    d = np.load(PQR)
    qr = {k: (d[f"{k}_q"], d[f"{k}_r"]) for k in ("plus", "minus")}
    _, sane = bias.sane_targets(qr)
    mom = d["moments"]
    size = mom[:, 1] / mom[:, 0]
    lo, hi = CELLS[CELL]
    keep = (sane & (mom[:, 0] >= FLUX_LO) & (mom[:, 0] < FLUX_HI)
            & (size >= lo) & (size < hi))
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(SEED)
    if len(idx) > NTARG:
        idx = np.sort(rng.choice(idx, NTARG, replace=False))
    print(f"cell {CELL}: Mr/Mf in [{lo}, {hi}), true Mf in "
          f"[{FLUX_LO:g}, {FLUX_HI:g}) -- "
          f"{keep.sum()} targets, using {len(idx)}")

    cov = bias.load_cov(f"{band_tmpl.D}/{bias.CATALOGS[POP]['plus']}.fits")
    a = np.linalg.inv(np.linalg.cholesky(cov))
    t = whitened_templates(cov)
    ngal_all = int(t["gal"].max()) + 1
    print(f"{len(t['gal'])} copies from {ngal_all} galaxies")

    boot = rng.integers(0, len(idx), (BOOT, len(idx)))
    fq, fr = d["plus_q"][idx], d["plus_r"][idx]
    fqm, frm = d["minus_q"][idx], d["minus_r"][idx]
    v, e, fi = m1_boot(fq, fr, fqm, frm, boot)
    print(f"\n{'N galaxies':>12s}{'copies':>11s}{'med ESS':>10s}"
          f"{'m1':>14s}{'+/-':>9s}{'SUM q^2 / SUM -r':>19s}")
    print(f"{'FLOW':>12s}{'':>11s}{'':>10s}{v:+14.5f}{e:9.5f}{fi:19.4f}")

    for n in NGAL:
        sub = t["gal"] < n if n < ngal_all else slice(None)
        tt = {k: jnp.asarray(vv[sub]) for k, vv in t.items() if k != "gal"}
        out = {}
        for arm in ("plus", "minus"):
            out[arm] = template_pqr(d[f"obs_{arm}"][idx], tt, a)
        qp, rp, ess_p = out["plus"]
        qm, rm, ess_m = out["minus"]
        good = (ess_p > ESSMIN) & (ess_m > ESSMIN)
        v, e, fi = m1_boot(qp[good], rp[good], qm[good], rm[good],
                           rng.integers(0, good.sum(), (BOOT, good.sum())))
        print(f"{min(n, ngal_all):12d}{len(tt['m']):11d}"
              f"{np.median(np.r_[ess_p, ess_m]):10.0f}{v:+14.5f}{e:9.5f}"
              f"{fi:19.4f}", flush=True)

        # PAIRED: same galaxies, same noise, same window -- only the estimator
        # of P(M|g) differs, so the population scatter that dominates both
        # errors above cancels and a 0.01 difference is resolvable.
        gi = np.flatnonzero(good)
        pb = rng.integers(0, len(gi), (BOOT, len(gi)))
        f_v = m1_boot(fq[gi], fr[gi], fqm[gi], frm[gi], pb[:1])[0]
        dd = [m1_boot(qp[gi][j], rp[gi][j], qm[gi][j], rm[gi][j], pb[:1])[0]
              - m1_boot(fq[gi][j], fr[gi][j], fqm[gi][j], frm[gi][j],
                        pb[:1])[0] for j in pb]
        print(f"{'':12s}{len(gi):11d} kept (ESS > {ESSMIN:g});  flow "
              f"{f_v:+.5f},  templates - flow = "
              f"{np.mean(dd):+.5f} +/- {np.std(dd, ddof=1):.5f}", flush=True)
