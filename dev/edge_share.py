"""Where does the bias LIVE, without cutting anything the estimator has to
put back?

Every per-band m1 in this repo is a cut on the observed moments, so it needs
eq. (40) -- and `HANDOFF.md` 2026-09-02 (night, cont. 3/4) shows an `Mr/Mf`
CEILING breaks that correction's window-independence by 0.035.  So a per-band
m1 is not measurable with today's machinery, and "cut the bad band away" is not
available either.

A per-band CONTRIBUTION is.  `ghat = R^-1 SUM_i q_i` is linear in the targets
at fixed `R`, so with the window held FIXED (nothing cut, nothing reinstated)
each target's share of the ONE estimate is exact:

    m1 = SUM_i b_i,
    b_i = [(R_p^-1 q_i^p)_0 - (R_m^-1 q_i^m)_0] / 2g  -  w_i,

with `w_i` the target's share `d_i / D` of the denominator, so `SUM w = 1`
reproduces the "-1".  Summing `b_i` over a band answers "how much of the total
does this population put there" -- which is what "the bias is driven by X"
should mean, and unlike a per-band m1 it is additive and correction-free.

`CUT` picks the window:

  * `zero` (default) -- the flux floor on the G = 0 ARM's moments.  These are
    NOT the truth: on an image-noise catalog `bias.py`'s `truth = m["zero"]` is
    a third noisy realization, labelled "for binning only" (`save_pqr`'s
    docstring calling it clean is wrong for that case).  What it IS, and what
    matters here, is INDEPENDENT of both `+g` and `-g` arms' noise and
    independent of `g` -- so binning on it induces no correlation with the
    estimator being decomposed, and no selection term arises at all.  Binning
    on an arm's OWN moments does: it flips the `Mr/Mf < 3.005` band from an
    implied m1 of -0.005 to +0.109.
  * `obs` -- each arm's own observed `Mf`, i.e. the production window.  Then
    the selection term is real and is one GLOBAL row (`Q_s ~ 7e-4` is zero to
    isotropy, so it enters only the denominator); `CORR` supplies the corrected
    total for that window and `D_sel` is recovered from the two.

Bins are reported over both the g = 0 arm's and the selected arm's `Mr/Mf`.

READ NO PER-BAND NUMBER AS A BIAS.  `E[q] = 0` and `E[q q^T] = E[-R]` hold only
over the WHOLE population, so any subpopulation's m1 is nonzero for a perfectly
correct estimator -- and BFD's own template sum gives the same per-band numbers
as the flow (`dev/prior_n.py`).  The contributions are additive and sum to the
one number that IS a measurement; the implied-m1 column is printed only to say
how concentrated a band's share is, never as that band's bias.
"""
import os
import numpy as np

import bias

PQR = os.environ.get("PQR", "dev/pqr_200k.npz")
G = float(os.environ.get("G", 0.02))
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
CUT = os.environ.get("CUT", "zero")            # zero | obs
CORR = float(os.environ.get("CORR", -0.01737))  # corrected m1, CUT=obs only
BOOT = int(os.environ.get("BOOT", 400))
SIZE = [float(x) for x in os.environ.get(
    "SIZE", "0 3.005 3.252 3.407 3.530 9").split()]


def parts(d, sane):
    """(b, w, zero, obs, tid) with the two ARMS STACKED.

    Stacking is what lets `CUT=obs` keep the production estimator bit for bit:
    the window is on each arm's OWN observed moments, so a target can be
    selected in one arm and not the other and lands in different bins in the
    two.  `b` and `w` still sum to `m1` and to 1.
    """
    b, w, tru, obs, tid = [], [], [], [], []
    for arm, sgn in (("plus", +1), ("minus", -1)):
        q, r, o = d[f"{arm}_q"], d[f"{arm}_r"], d[f"obs_{arm}"]
        sel = sane & ((d["moments"] if CUT == "zero" else o)[:, 0] >= FLUX_LO)
        q, r = q[sel], r[sel]
        b.append(np.linalg.solve(-r.sum(0), sgn * q.T)[0] / (2 * G))
        w.append(0.5 * -r[:, 0, 0])
        tru.append(d["moments"][sel])
        obs.append(o[sel])
        tid.append(np.flatnonzero(sel))
    w = np.concatenate(w)
    return (np.concatenate(b) - w / w.sum(), w / w.sum(),
            np.concatenate(tru), np.concatenate(obs), np.concatenate(tid))


if __name__ == "__main__":
    d = np.load(PQR)
    qr = {k: (d[f"{k}_q"], d[f"{k}_r"]) for k in ("plus", "minus")}
    _, sane = bias.sane_targets(qr)
    b, w, tru, obs, tid = parts(d, sane)
    ntarg = len(sane)

    # u = SUM b, c = (SUM b - s) / (1 + s) with s = D_sel / D_obs
    u = b.sum()
    s = 0.0 if CUT == "zero" else (u - CORR) / (1 + CORR)
    b, w = b / (1 + s), w / (1 + s)
    print(f"{len(b)} (arm, target) rows, {CUT} Mf >= {FLUX_LO:g}, "
          f"{PQR}\nm1 = {b.sum():+.5f}" +
          ("" if CUT == "zero" else
           f" (uncorrected {u:+.5f}, D_sel/D_obs = {s:+.4f})"))

    rng = np.random.default_rng(0)
    idx = rng.integers(0, ntarg, (BOOT, ntarg))

    def table(name, v, edges):
        print(f"\n{name}")
        print(f"{'bin':>14s}{'rows':>9s}{'share of D':>12s}"
              f"{'contribution':>14s}{'+/-':>9s}{'implied m1':>12s}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v < hi)
            # bootstrap over GALAXIES: both arms move together, so the +/-
            # pairing that cancels shape noise stays cancelled
            per = np.bincount(tid, np.where(m, b, 0.0), ntarg)
            err = np.std(per[idx].sum(1), ddof=1)
            print(f"{lo:6.3f}-{hi:<7.3f}{m.sum():9d}{w[m].sum():12.1%}"
                  f"{b[m].sum():+14.5f}{err:9.5f}"
                  f"{b[m].sum() / max(w[m].sum(), 1e-12):+12.4f}")
        if s:
            print(f"{'selection':>14s}{'':>9s}{-s / (1 + s):12.1%}"
                  f"{s / (1 + s):+14.5f}")

    table("by g=0-arm Mr/Mf", tru[:, 1] / tru[:, 0], SIZE)
    table("by SELECTED-ARM Mr/Mf", obs[:, 1] / obs[:, 0], SIZE)
    fq = [0] + list(np.percentile(tru[:, 0], [20, 40, 60, 80])) + [np.inf]
    table("by g=0-arm Mf quintile", tru[:, 0], fq)
