"""Do the flow's eq.-(40) terms actually disagree with the templates', or is it
Monte-Carlo noise?

`R_s` is a boundary flux -- only draws within a noise width of the window edge
contribute -- so a narrow window uses a few percent of the sample and its `R_s`
carries a correspondingly large error.  On the nominal window that is 17% for
BOTH routes at their default sample sizes, which is the same size as the
apparent flow-vs-template gap.  Before anything is "fixed", the comparison has
to be made at an error small enough to resolve it.

The asymmetry that matters: the templates' error is set by the CATALOG (1M
galaxies, and there is no more), while the flow's is set by the number of prior
DRAWS, which is free.  So this raises the flow's draw count and reports the
difference in sigma.

Both stencils are window-INDEPENDENT, so each prior is pushed through once and
every window is a `window_prob` over the same array.

    PYTHONPATH=. NDRAW=8388608 python dev/rs_agree.py
"""
import os
import sys

import numpy as np

import bias
from bias import load_cov, window_mask
import dev.size_bands as sb

D = sb.D
BOOT = int(os.environ.get("BOOT", 200))
TMPL = os.environ.get("TMPL", f"{D}/moments_bulgedisc_v2_1M.fits")

INF = float("inf")
# A LADDER: each entry adds ONE boundary to the one above it.  The corrected
# m1 has to be the same all the way down -- that is the whole content of
# eq. (45)-(46).  Where it stops being the same, that boundary's correction is
# what is wrong.
WINDOWS = [
    ("Mf>=1600", (-INF, INF), (1600.0, 1e9)),
    ("Mf>=2500", (-INF, INF), (2500.0, 1e9)),
    ("Mf 2500-50000", (-INF, INF), (2500.0, 50000.0)),
    ("+ size >2.2", (2.2, INF), (2500.0, 50000.0)),
    ("nominal (+ <3.2)", (2.2, 3.2), (2500.0, 50000.0)),
    ("size 2.2-3.2 only", (2.2, 3.2), (1600.0, 1e9)),
]


def terms(ms, cov, size, flux, nblock):
    """(P_s, Q_s, R_s), the block sem of R_s11, and the top draw's share."""
    pt = sb.per_template(ms, cov, size, flux)
    full = tuple(v.mean(0) for v in pt)
    b = np.array([pt[2][k][:, 0, 0].mean()
                  for k in np.array_split(np.arange(len(pt[0])), nblock)])
    top = np.abs(pt[2][:, 0, 0]).max() / (len(pt[0]) * abs(full[2][0, 0]))
    blocks = [tuple(v[k].mean(0) for v in pt)
              for k in np.array_split(np.arange(len(pt[0])), nblock)]
    return full, b.std(ddof=1) / np.sqrt(len(b)), top, blocks


if __name__ == "__main__":
    d = np.load(sb.PQR)
    ok = np.ones(len(d["plus_q"]), dtype=bool)
    for k in ("plus", "minus"):
        ok &= np.isfinite(d[f"{k}_q"]).all(1)
        ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    qp, rp = d["plus_q"][ok], d["plus_r"][ok]
    qm, rm = d["minus_q"][ok], d["minus_r"][ok]
    obs_p, obs_m = d["obs_plus"][ok], d["obs_minus"][ok]
    cov = load_cov(f"{D}/{bias.CATALOGS[sb.POP]['zero']}.fits")

    stencils = {}
    sb.TERMS = "flow"
    draw, z = sb.prior_draw()
    print(f"flow: {len(z)} draws", flush=True)
    stencils["flow"] = sb.stencil_moments(draw, z)
    del draw, z
    sb.TERMS, sb.TMPL = "templates", TMPL
    draw, z = sb.prior_draw()
    print(f"templates: {len(z)} draws", flush=True)
    stencils["templates"] = sb.stencil_moments(draw, z)
    del draw, z

    print(f"\n{'window':>20s}{'kept':>7s}"
          f"{'R_s11 flow':>13s}{'R_s11 tmpl':>13s}{'diff':>9s}{'sigma':>7s}"
          f"{'top f/t':>13s}{'m1 flow':>21s}{'m1 tmpl':>21s}")
    for name, size, flux in WINDOWS:
        sp = window_mask(obs_p, size, flux)
        sm = window_mask(obs_m, size, flux)
        n_p, n_m = int((~sp).sum()), int((~sm).sum())
        row, out = {}, {}
        for route, ms in stencils.items():
            full, sem, top, blocks = terms(ms, cov, size, flux, sb.NBLOCK)
            ns = ((n_p, *full), (n_m, *full))
            m1 = bias.bias(qp, rp, qm, rm, sb.G, sel=(sp, sm), ns=ns)[0]
            dm = bias.bootstrap(qp, rp, qm, rm, sb.G, BOOT, 0,
                                sel=(sp, sm), ns=ns)[0]
            ds = sb.selection_error(blocks, sp, sm, qp, rp, qm, rm)
            row[route] = (full[2][0, 0], sem, top)
            out[route] = (m1, dm, ds)
        df = row["flow"][0] - row["templates"][0]
        sd = np.hypot(row["flow"][1], row["templates"][1])
        print(f"{name:>20s}{sp.mean():7.1%}"
              f"{row['flow'][0]:+8.4f}{row['flow'][1]:5.4f}"
              f"{row['templates'][0]:+8.4f}{row['templates'][1]:5.4f}"
              f"{df:+9.4f}{abs(df) / sd:7.1f}"
              f"{row['flow'][2]:6.1%}{row['templates'][2]:7.1%}"
              + "".join(f"{out[r][0]:+9.4f} {out[r][1]:.4f} {out[r][2]:.4f}"
                        for r in ("flow", "templates")), flush=True)
    print("\nR_s11 columns are value then block sem; m1 columns are "
          "value, galaxy bootstrap, selection MC")
