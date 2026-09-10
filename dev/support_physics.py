"""Is the flow's out-of-support problem a PHYSICS problem or a fitting problem?

Two questions, both answerable from the catalog's own exact derivatives and the
target covariance, with no flow involved for parts 1-2.

1. **Does the shear response vanish at the point-source limit, as it must?**
   A point source is shear-invariant: shearing a delta gives a delta, so the
   PSF-convolved image is unchanged up to the magnification of the flux.  Every
   SHAPE ratio (Mr/Mf, Mc/Mr, |e|) therefore has zero shear derivative there,
   and d ln(Mr/Mf)/dg1 must vanish proportionally to (1 - u),
   u = Mr/(POINT_SOURCE Mf).  If the catalog agrees, the chart's 1/(1-u)
   amplification is cancelled by physics and a bounded chart is safe.  If it
   does not, either the catalog's derivatives are wrong near the edge or the
   population never gets close enough to see it.

2. **Are the draws that the flow answers garbage for actually OUTSIDE the
   physical support?**  If they are, the right value is exactly zero and no
   amount of flow tail-fitting is the answer -- an analytic indicator is.  If
   they are inside, the flow is just wrong in-support and the support is a red
   herring.

3. With the flow: is `poisoned` (log p < -1e4) the same set as `out of the
   analytic support`?  That is the one that decides between the two fixes.

    PYTHONPATH=. python dev/support_physics.py
"""
import os
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias                                                      # noqa: E402
import bulk                                                      # noqa: E402
import shear as sm                                               # noqa: E402
from models.bijections import POINT_SOURCE, POINT_SOURCE_MC      # noqa: E402

D = "../bfd_cnf_imsims/data"
FLOW = os.environ.get("FLOW", "flows/bulk_v3.eqx")
N = 20000


def part1(m, dmdg):
    """d ln(Mr/Mf)/dg1 against (1 - u).  Physics demands it -> 0."""
    size = m[:, 1] / m[:, 0]
    u = size / POINT_SOURCE
    one_minus_u = 1.0 - u

    # the three shape ratios whose response must vanish at a point source
    dlnsize = dmdg[:, 0, 1] / m[:, 1] - dmdg[:, 0, 0] / m[:, 0]
    dlnconc = dmdg[:, 0, 4] / m[:, 4] - dmdg[:, 0, 1] / m[:, 1]
    # d(M1/Mr)/dg1 -- the ellipticity response, which for a point source is 0
    de1 = dmdg[:, 0, 2] / m[:, 1] - m[:, 2] * dmdg[:, 0, 1] / m[:, 1] ** 2

    print("1. does the shear response vanish at the point-source limit?")
    print(f"   catalog max Mr/Mf = {size.max():.5f}, POINT_SOURCE = {POINT_SOURCE}"
          f"  -> min (1-u) = {one_minus_u.min():.5f}\n")
    print(f"   {'(1-u) band':>16s} {'n':>7s} {'<1-u>':>9s} "
          f"{'|dln(Mr/Mf)/dg|':>16s} {'ratio to <1-u>':>15s} "
          f"{'|d(M1/Mr)/dg|':>14s} {'|dln(Mc/Mr)/dg|':>16s}")
    qs = np.percentile(one_minus_u, [0, 1, 5, 20, 50, 80, 100])
    for lo, hi in zip(qs[:-1], qs[1:]):
        b = (one_minus_u >= lo) & (one_minus_u < hi)
        if b.sum() < 20:
            continue
        mu = one_minus_u[b].mean()
        a = np.abs(dlnsize[b]).mean()
        print(f"   {lo:7.4f}-{hi:7.4f} {int(b.sum()):7d} {mu:9.4f} "
              f"{a:16.5f} {a / mu:15.2f} {np.abs(de1[b]).mean():14.5f} "
              f"{np.abs(dlnconc[b]).mean():16.5f}")
    # If the response is proportional to (1-u) the ratio column is FLAT; if the
    # response is constant the ratio column grows like 1/(1-u).
    inb = one_minus_u > 0
    sl = np.polyfit(np.log(one_minus_u[inb]),
                    np.log(np.abs(dlnsize[inb]) + 1e-30), 1)[0]
    print(f"\n   log|dln(Mr/Mf)/dg1| vs log(1-u): slope {sl:+.3f}"
          f"   (physics predicts +1.00, a non-vanishing response predicts 0.00)")


def part2(M, cov):
    """What fraction of kernel draws leaves the analytic support?"""
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(0)
    S = 2048
    a = np.zeros(5)
    a[0], a[1] = POINT_SOURCE, -1.0
    t = M @ a / np.sqrt(a @ cov @ a)

    emax = None
    print("\n2. do kernel draws leave the analytic support?  (per target band,\n"
          "   'band' = (PS*Mf - Mr)/sigma of the TARGET; small = near the edge)")
    print(f"   {'band':>12s} {'n':>5s} {'Mr/Mf>PS':>10s} {'Mc/Mr>PSmc':>11s} "
          f"{'Mf<=0':>8s} {'Mr<=0':>8s} {'any':>8s}")
    for lo, hi, name in ((-9, 1, "< 1 sigma"), (1, 3, "1-3"), (3, 99, "> 3")):
        idx = np.where((t >= lo) & (t < hi))[0][:200]
        if len(idx) == 0:
            continue
        v = np.zeros(5)
        for i in idx:
            d = M[i] + rng.standard_normal((S, 5)) @ L.T
            size = d[:, 1] / np.where(d[:, 0] > 0, d[:, 0], np.nan)
            conc = d[:, 4] / np.where(d[:, 1] > 0, d[:, 1], np.nan)
            c = [np.nanmean(size > POINT_SOURCE), np.nanmean(conc > POINT_SOURCE_MC),
                 np.mean(d[:, 0] <= 0), np.mean(d[:, 1] <= 0)]
            bad = ((size > POINT_SOURCE) | (conc > POINT_SOURCE_MC)
                   | (d[:, 0] <= 0) | (d[:, 1] <= 0))
            v += np.array(c + [np.mean(bad)])
        v /= len(idx)
        print(f"   {name:>12s} {len(idx):5d} {v[0]:10.2%} {v[1]:11.2%} "
              f"{v[2]:8.2%} {v[3]:8.2%} {v[4]:8.2%}")
    return t, L


def part3(m, M, t, L):
    """Is 'poisoned' the same set as 'out of support'?"""
    flow = eqx.tree_deserialise_leaves(FLOW, bulk.build_flow(jr.key(0), m))
    rng = np.random.default_rng(0)
    S = 2048
    print("\n3. poisoned (log p < -1e4) vs out-of-support, same draws:")
    print(f"   {'band':>12s} {'poisoned':>10s} {'out-of-supp':>12s} "
          f"{'pois & in-supp':>15s} {'out & not pois':>15s}")
    for lo, hi, name in ((-9, 1, "< 1 sigma"), (1, 3, "1-3"), (3, 99, "> 3")):
        idx = np.where((t >= lo) & (t < hi))[0][:40]
        if len(idx) == 0:
            continue
        acc = np.zeros(4)
        for i in idx:
            d = M[i] + rng.standard_normal((S, 5)) @ L.T
            lp = np.asarray(jax.vmap(flow.log_prob)(jnp.asarray(d, jnp.float32)))
            size = d[:, 1] / np.where(d[:, 0] > 0, d[:, 0], np.nan)
            conc = d[:, 4] / np.where(d[:, 1] > 0, d[:, 1], np.nan)
            out = ((size > POINT_SOURCE) | (conc > POINT_SOURCE_MC)
                   | (d[:, 0] <= 0) | (d[:, 1] <= 0))
            out = np.nan_to_num(out, nan=True).astype(bool)
            pois = (lp < -1e4) | ~np.isfinite(lp)
            acc += [pois.mean(), out.mean(), (pois & ~out).mean(),
                    (out & ~pois).mean()]
        acc /= len(idx)
        print(f"   {name:>12s} {acc[0]:10.2%} {acc[1]:12.2%} {acc[2]:15.2%} "
              f"{acc[3]:15.2%}")


def main():
    m, dmdg, _ = sm.load(f"{D}/moments_bulgedisc_v3.fits")
    m = np.asarray(m, np.float64)
    dmdg = np.asarray(dmdg, np.float64)
    tg = fitsio.read(f"{D}/targets_v3_g0_200k.fits", rows=np.arange(N))
    M = np.asarray(tg["moments"], np.float64)
    cov = np.asarray(bias.load_cov(f"{D}/targets_v3_g0_200k.fits"), np.float64)
    if cov.ndim == 3:
        cov = cov[0]

    part1(m, dmdg)
    t, L = part2(M, cov)
    part3(m, M, t, L)


if __name__ == "__main__":
    main()
