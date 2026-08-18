"""Per-partial-derivative comparison of the layer's dm/dg against bfd's.

`shear.check` reports `RMS residual / RMS truth` after averaging over the target
axis AND the g axis, so it collapses the ten partials d m_i / d g_a into five
numbers and cannot see direction-specific error.  It also gives one number per
moment with no reference, which is uninterpretable on its own: `shear.py
scatter` measures a Var[Q|m] FLOOR of 2.09% (Mf), 20.98% (Mr), 33.38% (Mc) and
1.43% (spin-2), so a 27% residual on Mr is at the information limit while a 27%
residual on Mf is 13x above it.

This splits g1 from g2 and decomposes each partial into the two things that
matter separately:

    pred  =  slope * truth  +  scatter

  * `slope`, the regression through the origin <pred.truth>/<truth^2>.  A
    deterministic transport can only carry E[dm/dg | m], so it SHOULD shrink the
    response toward its conditional mean -- but a slope far from 1 is a
    systematic error, not the floor.
  * `1 - corr^2`, the share of the response the layer does not explain at all.
    This is what should approach the Var[Q|m] floor if the fit is as good as the
    moments allow.

Splitting them matters because `RMS resid / RMS truth` mixes the two:
    (resid/truth)^2 = (1 - slope)^2 + scatter^2/truth^2
so a fit that is perfectly calibrated but noisy and one that is quiet but
systematically half the right size can report the same number.

An indexing self-check runs first.  bfd's response is spin-2, so d Mf/d g1 must
correlate with e1 and d Mf/d g2 with e2; if the (2, 5) axes were transposed or
the moment order differed, that correlation would land in the wrong place.

    python dev/check_dmdg_components.py --flow flows/shear_cfo.eqx

ANSWER (2026-08-17), chart-first ordering, 20000 bulgedisc templates.

INDEXING VERIFIED, so `dm_dg`'s chart composition is doing the right thing:
corr(dMf/dg1, e1) = +0.960 against corr(dMf/dg1, e2) = -0.012, and the mirror
for g2 -- clean spin-2 structure on bfd's truth, no transpose, no moment-order
mismatch.  Independently, four of five collapsed columns land at their Var[Q|m]
floors; a broken change of variables could not produce that pattern.

`shear.check` HIDES THE LARGEST DEFECT.  M1's two partials differ by 27x in
magnitude, and averaging over g buries the small one:

    d M1/d g1 (diagonal)      RMS truth 0.3669   resid  1.9%   floor 1.4%
    d M1/d g2 (off-diagonal)  RMS truth 0.0134   resid 38.6%   floor 1.4%
    collapsed (what check prints)                resid  2.30%

So the off-diagonal spin-2 response is 27x above its floor and invisible in the
reported number.

Mf AMPLIFIES: slope 1.197.  A deterministic transport carries E[Q|m], whose
variance is SMALLER than Q's, so slope <= 1 is forced.  Mr (0.895), Mc (0.731)
and the diagonal spin-2 (0.999) all obey it; Mf does not, and no conditional-mean
argument produces amplification.  A real defect, in the one column the chain
reorder moved off its floor.

THE NLL DOES CONVERGE TO WHAT IT IS TRAINED ON -- for the component that carries
the signal.  With the supervision off:

    d M1/d g1  slope 1.002  corr 0.997  resid  7.6%    RMS truth 0.367
    d Mf/d g   slope 0.563  corr 0.684  resid   74%    RMS truth 0.056
    d Mr/d g   slope 0.320  corr 0.502  resid   87%    RMS truth 0.083
    d Mc/d g   slope 0.114  corr 0.280  resid   97%    RMS truth 0.102

The dominant spin-2 response comes out essentially exact; only the spin-0
components, 3.5-6.5x smaller in the data, fail.  "The response is not
identifiable from the likelihood" was too broad a reading of the collapsed
numbers.

WHY the NLL misses spin-0, measured on bfd's truth alone (no flow):

    partial        mean          RMS       mean/RMS   corr(e1)
    dMf/dg1     +2.3e-04     5.64e-02       0.004      +0.960
    dMr/dg1     +2.6e-04     8.40e-02       0.003      +0.837
    dMc/dg1     +2.4e-04     1.03e-01       0.002      +0.643
    dM1/dg1     -2.9e-01     3.66e-01      -0.780      -0.003

The population MEAN of every spin-0 partial is ~0.3% of its RMS -- the spin-0
response is odd in e (it enters through p1 = Re(e* g)), so over an isotropic
population it averages away.  The spin-2 diagonal has mean/RMS = -0.78, a
coherent shift.  So the population DENSITY feels spin-2 at O(g) and spin-0 only
at O(g^2), while at fixed m the ellipticity is fixed and E[dm/dg | m] is very
much nonzero (corr 0.64-0.96 with e).  That is the whole gap: the likelihood
sees the population, the supervision sees the template.

Confirmed by the g_max ladder (NLL only; `shear.py train --g-max`), where the
share explained rises monotonically for exactly the starved components:

    corr        g_max 0.15    0.40    0.80
    dMf/dg1        0.706     0.788   0.859
    dMr/dg1        0.520     0.594   0.712
    dMc/dg1        0.311     0.406   0.628
    dM1/dg1        0.997     0.997   0.997   (saturated -- the control)

but NOT a usable fix: slopes peak near g_max 0.4 and collapse by 0.8 (Mf 0.587
-> 0.599 -> 0.274), and the off-diagonal spin-2 slope goes NEGATIVE there.
`lens()` is a second-order Taylor model, so |g| = 0.8 is far outside the regime
where the training data are lensing at all.

OUTLIERS MATTER, and for the off-diagonal spin-2 they are not outliers (2026-08-17,
per-partial supervision at g_max 0.02).  The top 1% of templates by squared
residual carry 37-59% of it on most partials.  Trimming them:

    partial      resid   trimmed 1%   floor
    dMf/dg1       7.4%       7.1%     2.1%
    dMr/dg1      27.4%      21.5%    21.0%   <- AT floor once trimmed
    dM1/dg1       3.0%       2.3%     1.4%
    dM1/dg2      35.9%      25.6%       --   (slope 0.861 -> 0.941)
    dMc/dg1      42.1%      34.7%    33.4%   <- AT floor once trimmed

So Mr and Mc are FINISHED, not "near" their floors; the excess was a heavy tail.

What the tail IS, for dM1/dg2 (worst 1% vs rest): Mr/Mf 2.15 vs 3.32, |e| 0.101
vs 0.025, log10 Mf 3.47 vs 3.71, and true |dM1/dg2|/Mr 0.0355 vs 0.0015.  Large,
faint, 4x more elliptical galaxies whose true off-diagonal response is 24x
typical -- which is physics, since that partial is generated by the B e^2 gbar
terms and scales as |e|^2.  The 173x max/median is the genuine dynamic range of
the partial, not bad data.  The layer under-fits high-|e| galaxies; |e|^2 is
already an input to the coefficient net, so the information is there.

Note the trap: BAND = 3.4 upweights Mr/Mf near the POINT-SOURCE end, but 0% of
these templates are above Mr/Mf 3.5 against 19% of the rest.  `--band` would
upweight the opposite end from where this problem lives.

BATCH IS NOT THE CONSTRAINT once the supervision is per-partial.  Over 16x
(1024 -> 16384) nothing moves except the diagonal spin-2 (3.0% -> 2.4%):
everything else is already at its floor.

Separate open thread: the g1g2 CROSS second derivatives are uniformly poor
(slope 0.67-0.90, resid 43-55%) while the g1g1 and g2g2 diagonals are fine
(2.1%, 6.2%, 11.5%).  No Var[R|m] floor has been measured, so there is no
reference for how bad that actually is.
"""
import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402
from models.shear import dm_dg                       # noqa: E402

LAB = ["Mf", "Mr", "M1", "M2", "Mc"]


def measure_floor(m, q_true):
    """Var[Q|m] on THIS subset, via `shear.response_scatter`.

    Recomputed rather than tabulated, because any cut that changes the
    population changes the floor too -- and in the same direction as the
    residual, so a fixed table flatters a windowed fit.  Measured: the
    Mr/Mf >= 2.2 window moves Mr's floor 20.98% -> 16.23% and Mc's
    33.38% -> 27.07%, which is most of the improvement the window appears to
    buy.  Comparing a windowed residual against an unwindowed floor is how one
    concludes a column is "finished" when it is 1.3x above its real limit.

    NOTE the spin-2 entry is `response_scatter`'s COMBINED (A, B) number, so it
    is not a per-partial floor.  The diagonal partial is generated by A alone
    (de = A g gives dM1/dg1 = A, dM1/dg2 = 0) and the off-diagonal only by the
    higher structures, so it is the wrong reference for the off-diagonal and is
    printed there only as a placeholder.
    """
    import io, contextlib, re
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        shear_top.response_scatter(m, q_true)
    out = {}
    for line in buf.getvalue().splitlines():
        mm = re.match(r"\s*(spin-2 \(A, B\)|spin-0 (Mf|Mr|Mc))\s+[\d.]+%\s+([\d.]+)%", line)
        if mm:
            key = "AB" if mm.group(1).startswith("spin-2") else mm.group(2)
            out[key] = float(mm.group(3)) / 100.0
    return {"Mf": out.get("Mf"), "Mr": out.get("Mr"), "Mc": out.get("Mc"),
            "M1": out.get("AB"), "M2": out.get("AB")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear_cfo.eqx")
    p.add_argument("--data", default="../bfd_cnf_imsims/data/moments.fits")
    p.add_argument("-n", type=int, default=20000)
    p.add_argument("--size-window", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="keep only templates with LO <= Mr/Mf <= HI. The real "
                        "analysis sets a size window on TARGETS, so the response "
                        "outside it is never used; 2.2 has been the lower bound. "
                        "This restricts the COMPARISON only -- it does not cut "
                        "the training set, which is a separate decision.")
    p.add_argument("--flux-window", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="same, on log10 Mf.")
    a = p.parse_args()

    m, q_true, r_true = (np.asarray(x[:a.n], np.float64)
                         for x in shear_top.load(a.data))
    flow = bulk.build_flow(jr.key(0), shear_top.load(a.data)[0], shear=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    layer, chart = shear_top._shear_layer(flow), shear_top._chart(flow)

    q, r = jax.vmap(dm_dg, in_axes=(None, 0, None))(
        layer, jnp.asarray(m, jnp.float32), chart)
    q = np.asarray(q, np.float64)                     # (n, 2, 5)
    r = np.asarray(r, np.float64)                     # (n, 3, 5)
    s = np.stack([m[:, 0], m[:, 1], m[:, 1], m[:, 1], m[:, 4]], -1)
    n = len(m)

    r_mf, lgf = m[:, 1] / m[:, 0], np.log10(m[:, 0])
    print(f"{a.flow}: {len(m)} templates, q {q.shape}, truth {q_true.shape}")
    print(f"  Mr/Mf   : " + "  ".join(
        f"p{p_}={np.percentile(r_mf, p_):.3f}" for p_ in (0, 1, 5, 50, 95, 99, 100)))
    print(f"  log10 Mf: " + "  ".join(
        f"p{p_}={np.percentile(lgf, p_):.3f}" for p_ in (0, 1, 5, 50, 95, 99, 100)))
    keep = np.ones(len(m), bool)
    if a.size_window:
        keep &= (r_mf >= a.size_window[0]) & (r_mf <= a.size_window[1])
    if a.flux_window:
        keep &= (lgf >= a.flux_window[0]) & (lgf <= a.flux_window[1])
    if a.size_window or a.flux_window:
        print(f"  WINDOW keeps {keep.sum()} of {len(m)} ({keep.mean():.1%})")
        m, q, r, q_true, r_true, s = (x[keep] for x in
                                      (m, q, r, q_true, r_true, s))

    # ---- indexing self-check: the response is spin-2, so dMf/dg1 ~ e1.
    e1, e2 = m[:, 2] / m[:, 1], m[:, 3] / m[:, 1]
    print("\nindexing check -- bfd truth only, no flow involved:")
    for ai, an in ((0, "g1"), (1, "g2")):
        c1 = np.corrcoef(q_true[:, ai, 0] / s[:, 0], e1)[0, 1]
        c2 = np.corrcoef(q_true[:, ai, 0] / s[:, 0], e2)[0, 1]
        print(f"  corr(dMf/d{an}, e1) = {c1:+.3f}   corr(dMf/d{an}, e2) = {c2:+.3f}"
              f"   {'OK' if abs(c1 if ai == 0 else c2) > abs(c2 if ai == 0 else c1) else '<-- SWAPPED?'}")

    FLOOR = measure_floor(m, q_true)
    print("  Var[Q|m] floor measured on THIS subset: " + "  ".join(
        f"{k}={v:.2%}" for k, v in FLOOR.items() if k != "M2"))

    def table(name, pred, truth, comps, first_order=True):
        print(f"\n=== {name} ===")
        print(f"  {'':10s} {'RMS truth/s':>12s} {'slope':>8s} {'corr':>7s} "
              f"{'unexpl':>8s} {'floor':>7s} {'resid/truth':>12s}")
        for ai, an in enumerate(comps):
            for i, lab in enumerate(LAB):
                t = truth[:, ai, i] / s[:, i]
                p_ = pred[:, ai, i] / s[:, i]
                tt = np.mean(t * t)
                slope = np.mean(p_ * t) / tt
                corr = np.mean(p_ * t) / np.sqrt(np.mean(p_ * p_) * tt)
                unexpl = np.sqrt(max(0.0, 1.0 - corr ** 2))
                resid = np.sqrt(np.mean((p_ - t) ** 2) / tt)
                # The floor is FIRST ORDER only -- `shear.py scatter` measures
                # Var[Q|m], not Var[R|m] -- so it is not a reference for the
                # second-order block and is neither printed nor flagged there.
                fl = (FLOOR.get(lab) or np.nan) if first_order else np.nan
                flag = ("" if not first_order or resid < 1.6 * fl
                        else "   <-- above floor")
                print(f"  {an+' '+lab:10s} {np.sqrt(tt):12.4g} {slope:8.3f} "
                      f"{corr:7.3f} {unexpl:8.1%} "
                      f"{(f'{fl:7.1%}' if first_order else '      -')} "
                      f"{resid:12.1%}{flag}")

    # ---- how much of each number is a handful of templates?
    print("\n=== outlier sensitivity, first order ===")
    print("  RMS is a squared-error statistic, so a few templates can set it.")
    print(f"  {'':10s} {'top0.1% sq':>11s} {'top1% sq':>9s} {'|t|max/med':>11s} "
          f"{'slope':>8s} {'slope99':>8s} {'resid':>8s} {'resid99':>8s}")
    for ai, an in enumerate(["g1", "g2"]):
        for i, lab in enumerate(LAB):
            tr = q_true[:, ai, i] / s[:, i]
            pr = q[:, ai, i] / s[:, i]
            d2 = (pr - tr) ** 2
            o = np.sort(d2)[::-1]
            top01 = o[:max(1, n // 1000)].sum() / o.sum()
            top1 = o[:max(1, n // 100)].sum() / o.sum()
            tmax = np.abs(tr).max() / np.median(np.abs(tr))
            slope = np.mean(pr * tr) / np.mean(tr * tr)
            resid = np.sqrt(np.mean(d2) / np.mean(tr * tr))
            # drop the worst 1% BY RESIDUAL and recompute
            keep = d2 <= np.quantile(d2, 0.99)
            s99 = np.mean(pr[keep] * tr[keep]) / np.mean(tr[keep] ** 2)
            r99 = np.sqrt(np.mean(d2[keep]) / np.mean(tr[keep] ** 2))
            print(f"  {an+' '+lab:10s} {top01:11.1%} {top1:9.1%} {tmax:11.1f} "
                  f"{slope:8.3f} {s99:8.3f} {resid:8.1%} {r99:8.1%}")

    table("dm/dg, per partial", q, q_true, ["g1", "g2"])
    table("d2m/dg2, per partial", r, r_true, ["g1g1", "g1g2", "g2g2"],
          first_order=False)
    print("\n  slope   : <pred.truth>/<truth^2>; 1 = right size, <1 = shrunk.")
    print("            A transport carries E[Q|m], which has LESS variance than")
    print("            Q, so slope <= 1 is forced.  Above 1 is a real defect.")
    print("  unexpl  : sqrt(1-corr^2), the share not explained at all")
    print("  floor   : Var[Q|m] from `shear.py scatter` (first order only)")
    print("  resid   : what `shear.check` prints, but split by g component")


if __name__ == "__main__":
    main()
