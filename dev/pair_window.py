"""What matched target pairs can say about the size-ceiling failure.

The three target arms are the SAME galaxies under the SAME noise field, so a
target's window membership in the +g and -g arms is a matched pair.  Three
things follow that the unpaired error bars cannot show:

1. **The ceiling jump's PAIRED significance.**  `+size>2.2` and `nominal` are
   nested subsets of one catalog, so their absolute bars are ~90% common and
   the quadrature difference is the wrong yardstick.  One galaxy resample
   evaluated at both windows gives the right one.

2. **The crossing census.**  Only targets that change membership between arms
   carry any selection signal at all.  Counting them bounds what a
   count-based estimate of Q_s/R_s could ever resolve at this catalog size --
   i.e. whether the arms can validate eq. (40) directly, or only the prior can.

3. **Where the ceiling's removed targets sit.**  Binned on the g = 0 arm (so
   the binning is independent of both arms' noise and of g), the removed
   population's share of SUM q and SUM -R says what the correction is being
   asked to reinstate.

    PYTHONPATH=. python dev/pair_window.py
"""
import os

import numpy as np

import bias
from bias import window_mask

PQR = os.environ.get("PQR", "dev/pqr_v3_s3_20k.npz")
G = float(os.environ.get("G", 0.02))
BOOT = int(os.environ.get("BOOT", 2000))
FLUX = (2500.0, 50000.0)
INF = float("inf")


def m1_uncorrected(qp, rp, qm, rm, sp, sm):
    """The windowed estimate with NO selection terms -- the quantity whose
    window-to-window change the correction is supposed to remove."""
    return bias.bias(qp[sp], rp[sp], qm[sm], rm[sm], G)[0]


def main():
    d = np.load(PQR)
    ok = np.ones(len(d["plus_q"]), dtype=bool)
    for k in ("plus", "minus"):
        ok &= np.isfinite(d[f"{k}_q"]).all(1)
        ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    qp, rp = d["plus_q"][ok], d["plus_r"][ok]
    qm, rm = d["minus_q"][ok], d["minus_r"][ok]
    op, om, m0 = d["obs_plus"][ok], d["obs_minus"][ok], d["moments"][ok]
    n = len(qp)
    print(f"{n} targets with finite Q, R in both arms\n")

    floor = ((2.2, INF), FLUX)
    nomin = ((2.2, 3.2), FLUX)

    # --- 2. crossing census, per boundary -------------------------------
    print("crossing census (targets whose window membership differs between "
          "the +g and -g arms):")
    for lab, (size, flux) in (("size floor only", floor),
                              ("nominal", nomin)):
        sp, sm = window_mask(op, size, flux), window_mask(om, size, flux)
        cross = sp ^ sm
        net = int(sp.sum()) - int(sm.sum())
        print(f"  {lab:16s} kept +g {sp.sum():5d}  -g {sm.sum():5d}  "
              f"net {net:+4d}   crossers {cross.sum():4d} "
              f"({cross.mean():.2%})")
    # What a count-based Q_s / R_s could resolve.  P_s(g) from the arms is a
    # binomial proportion; its FIRST difference is the whole content of Q_s and
    # its SECOND difference of R_s, in units of targets.
    sp, sm = window_mask(op, *nomin[::-1][::-1]), None
    sp = window_mask(op, *nomin)
    sm = window_mask(om, *nomin)
    q_s, r_s = -1.638e-04, 0.0797       # nominal-window flow values
    print(f"\n  eq.(40) predicts, in TARGETS out of {n}:")
    print(f"    Q_s * 2g  -> a +g/-g count difference of "
          f"{q_s * 2 * G * n:+.2f}   (observed {int(sp.sum()) - int(sm.sum()):+d})")
    print(f"    R_s * g^2 -> a second difference of {r_s * G ** 2 * n:+.3f}")
    print("    => the arms cannot measure either at this catalog size; the "
          "prior is the only route.")

    # --- 1. paired significance of the ceiling jump ----------------------
    pts, boots = {}, {}
    for lab, (size, flux) in (("size floor only", floor), ("nominal", nomin)):
        sp, sm = window_mask(op, size, flux), window_mask(om, size, flux)
        pts[lab] = m1_uncorrected(qp, rp, qm, rm, sp, sm)
        boots[lab] = (sp, sm, np.empty(BOOT))
    rng = np.random.default_rng(0)
    for b in range(BOOT):
        idx = rng.integers(0, n, n)
        for lab in boots:
            sp, sm, arr = boots[lab]
            arr[b] = m1_uncorrected(qp[idx], rp[idx], qm[idx], rm[idx],
                                    sp[idx], sm[idx])
    a, c = boots["size floor only"][2], boots["nominal"][2]
    print(f"\nUNCORRECTED m1, same targets, galaxy bootstrap {BOOT} draws:")
    for lab in ("size floor only", "nominal"):
        print(f"  {lab:16s} {pts[lab]:+.5f} +/- {boots[lab][2].std(ddof=1):.5f}")
    dif = pts["nominal"] - pts["size floor only"]
    print(f"  PAIRED difference {dif:+.5f} +/- {(c - a).std(ddof=1):.5f}"
          f"   ({abs(dif) / (c - a).std(ddof=1):.1f} sigma)")
    print("  (the correction has to cancel exactly this, and instead it "
          "over-cancels)")

    # --- 3. what the ceiling removes, binned on the g = 0 arm ------------
    print("\nwhat the size ceiling removes, binned on the g=0 arm's Mr/Mf "
          "(no selection term: the binning is independent of g and of both "
          "arms' noise):")
    sp, sm = window_mask(op, *floor), window_mask(om, *floor)
    rp_sum, rm_sum = -rp[sp].sum(0), -rm[sm].sum(0)
    bp = np.linalg.solve(rp_sum, qp[sp].T)[0] / (2 * G)
    bm = np.linalg.solve(rm_sum, -qm[sm].T)[0] / (2 * G)
    size0 = m0[:, 1] / m0[:, 0]
    edges = [0.0, 2.8, 3.0, 3.1, 3.2, 3.3, 3.5, 9.0]
    # b_i sums to 1 + m1; the target's share w_i of the denominator is what
    # makes the contribution additive (SUM w = 1 supplies the "-1").
    print(f"  {'Mr/Mf band':>14s} {'targets':>8s} {'share -R11':>12s}"
          f" {'contribution':>13s}")
    removed = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inb = (size0 >= lo) & (size0 < hi)
        cp, cm = inb[sp], inb[sm]
        wr = (-rp[sp][cp][:, 0, 0].sum() - rm[sm][cm][:, 0, 0].sum()) \
            / (rp_sum[0, 0] + rm_sum[0, 0])
        contrib = bp[cp].sum() + bm[cm].sum() - wr
        if lo >= 3.2:
            removed += contrib
        star = "  <- REMOVED by the 3.2 ceiling" if lo >= 3.2 else ""
        print(f"  {lo:6.2f}-{hi:5.2f} {int(cp.sum() + cm.sum()):8d} "
              f"{wr:12.2%} {contrib:+13.5f}{star}")
    print(f"  {'':14s} {'':8s} {'total':>12s} "
          f"{pts['size floor only']:+13.5f}")
    print(f"\n  the three removed bands carry {removed:+.5f} of that "
          f"{pts['size floor only']:+.5f} "
          f"({removed / pts['size floor only']:.0%}) on "
          f"{(size0[sp] >= 3.2).mean():.0%} of the targets")

    # How much of the ceiling's raw selection effect does eq.(45)-(46) remove?
    CORR = {"size floor only": -0.0242, "nominal": +0.0071}   # ladder, 2^23
    raw = dif
    left = CORR["nominal"] - CORR["size floor only"]
    print(f"\neq.(45)-(46) vs the effect it has to cancel:")
    print(f"  raw (uncorrected) ceiling effect  {raw:+.4f} +/- "
          f"{(c - a).std(ddof=1):.4f}")
    print(f"  left after correction             {left:+.4f}")
    print(f"  => the correction captures {1 - left / raw:.0%} of it and "
          f"misses {left:+.4f}")


if __name__ == "__main__":
    main()
