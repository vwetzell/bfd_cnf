"""Why does the corrected m1 climb at a TIGHT size ceiling?

At `Mr/Mf <= 3.0` the uncorrected m1 is +0.086 and the correction has to cancel
almost all of it, so the answer is a difference of large numbers.  This sizes
that directly:

  * LEVERAGE -- what fraction of the corrected denominator is the selection
    term, and how much m1 moves per 1% error in `R_s11`.  If the leverage is
    10x higher at 3.0 than at 3.45, a correction accurate to a few percent
    everywhere still lands far off at 3.0 and nothing else needs explaining.
  * CONVERGENCE -- is `R_s11` itself converged at that window, in the central
    difference `h` and in the number of prior draws?  `h = 0.02` is the size of
    the true shear, so a step-like `F` need not be in its linear regime.

Reuses `dev/size_offline.py`'s prior-draw builder, so `TERMS`, `FLOW`, `NDRAW`
select the same routes.
"""
import os, sys
import numpy as np, jax.numpy as jnp

import bias
from bias import window_mask, selection_terms, load_cov
from size_offline import prior_draw, D, POP, PQR, G, FLUX_LO, TERMS

HIS = [float(x) for x in os.environ.get("HIS", "9e9 3.45 3.30 3.20 3.00").split()]
FDS = [float(x) for x in os.environ.get("FDS", "0.04 0.02 0.01 0.005").split()]


def denominators(qp, rp, qm, rm, sp, sm, ns_p, ns_m):
    """(num, D_obs, D_sel) of the 11 component, averaged over the two arms."""
    num = (qp[sp, 0].sum() - qm[sm, 0].sum()) / (2 * G)
    d_obs = 0.5 * (-rp[sp, 0, 0].sum() + -rm[sm, 0, 0].sum())
    d_sel = 0.5 * sum(n * (qs[0] ** 2 / (1 - ps) ** 2 + rs[0, 0] / (1 - ps))
                      for n, ps, qs, rs in (ns_p, ns_m))
    return num, d_obs, d_sel


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
    draw, z = prior_draw()
    flux = (FLUX_LO, 1e9)
    print(f"selection terms from the {TERMS}, flux >= {FLUX_LO:g}\n")

    print("LEVERAGE -- how much of the denominator the correction supplies, "
          "and\nwhat a 1% error in R_s11 costs in m1\n")
    print(f"{'Mr/Mf hi':>9s}{'kept':>7s}{'D_obs':>12s}{'D_sel':>12s}"
          f"{'D_sel/D':>9s}{'Q_s1':>11s}{'m1':>10s}{'dm1/1% R_s':>12s}")
    for hi in HIS:
        size = (-np.inf, hi)
        sp, sm = window_mask(obs_p, size, flux), window_mask(obs_m, size, flux)
        ps, qs, rs, _ = selection_terms(draw, z, cov, size, flux, fd=0.02)
        ns_p = (int((~sp).sum()), ps, qs, rs)
        ns_m = (int((~sm).sum()), ps, qs, rs)
        num, d_obs, d_sel = denominators(qp, rp, qm, rm, sp, sm, ns_p, ns_m)
        m1 = bias.bias(qp, rp, qm, rm, G, sel=(sp, sm), ns=(ns_p, ns_m))[0]
        # d m1 / d (R_s11 -> 1.01 R_s11), by the same 2x2 solve
        rs2 = rs.copy(); rs2[0, 0] *= 1.01
        m1b = bias.bias(qp, rp, qm, rm, G, sel=(sp, sm),
                        ns=((ns_p[0], ps, qs, rs2), (ns_m[0], ps, qs, rs2)))[0]
        print(f"{hi:9.2f}{sp.mean():7.1%}{d_obs:12.4g}{d_sel:12.4g}"
              f"{d_sel / (d_obs + d_sel):9.1%}{qs[0]:+11.2e}{m1:+10.4f}"
              f"{m1b - m1:+12.5f}", flush=True)

    print(f"\nCONVERGENCE of R_s11 in the central-difference h")
    print(f"{'Mr/Mf hi':>9s}" + "".join(f"{f'h={h}':>13s}" for h in FDS)
          + f"{'jvp (no fd)':>14s}")
    for hi in HIS:
        size = (-np.inf, hi)
        vals = [selection_terms(draw, z, cov, size, flux, fd=h)[2][0, 0]
                for h in FDS]
        jv = selection_terms(draw, z, cov, size, flux, fd=None)[2][0, 0]
        print(f"{hi:9.2f}" + "".join(f"{v:13.4f}" for v in vals)
              + f"{jv:14.4f}", flush=True)
