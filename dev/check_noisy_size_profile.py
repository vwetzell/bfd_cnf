"""Does the noiseless size-axis structure in m1 survive the noisy pipeline?

`dev/check_size_oscillation.py` found a real multi-lobe m1(Mr/Mf) on the
NOISELESS path (exact per-galaxy derivatives, no kernel MC).  Open question 2
of HANDOFF.md: does it survive once the flow is integrated under C_M?  It is
not a foregone conclusion -- [[convolution-reaches-past-the-cut]] says the
noisy ramp is NOT the noiseless one, so the convolution can create structure
of its own, and [[ess-starvation-is-the-resolution-edge]] says the MC error is
worst at exactly the resolution edge the structure lives at.

Reads a `bias.py --save-pqr` npz.  Bin on the TRUE Mr/Mf from the g=0 catalog,
never the noisy one: `--n-targets` takes a plain prefix slice, so npz row i is
catalog row i, and the noiseless moment is known exactly.  That avoids the
regression proxy `dev/check_bias_flatness.noise_free` needs on the image-noise
catalogs, and it avoids binning on a noisy quantity -- which slides m1 by
0.055 across the plausible cut range all by itself
([[no-safe-noisy-resolution-cut]]).

The headline number is the PEAK-minus-TROUGH contrast, not the profile: the
windows come from the noiseless run, so this is a pre-registered two-sample
test with far more power than eyeballing 60 noisy bins.

    python dev/check_noisy_size_profile.py --pqr <run>.npz
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import shear as shear_top                            # noqa: E402

G, NBOOT = 0.02, 400

# Windows placed from the NOISELESS equal-width profile (HANDOFF.md section 2),
# fixed before this run was looked at.  The reference m1 beside each is the
# window's ACTUAL noiseless value, recomputed over the window (not the single
# extreme bin quoted in the handoff) from dev/bias_flux_size_cache.npz -- a
# window averages over a lobe, so it is always the milder number.
#
# The compact end (Mr/Mf ~ 1.2-2.0) is deliberately NOT here.  Aggregated it
# returns m1 = -0.44: that range holds the handful of targets with a
# near-singular 2x2 R whose bootstrap sd blows up to O(1), which
# `check_size_oscillation` filters from its figure for the same reason.  It is
# a broken estimator, not a signal, and no noisy run can say anything about it.
WINDOWS = {
    "peak   (noiseless +0.0221)": (3.10, 3.20),
    "trough (noiseless -0.0062)": (3.32, 3.42),
    "bump   (noiseless +0.0021)": (3.40, 3.46),
}
NOISELESS_CONTRAST = 0.02831        # +/- 0.00038, peak minus trough


def m1_of(d, sel):
    return bias.bias(d["plus_q"][sel], d["plus_r"][sel],
                     d["minus_q"][sel], d["minus_r"][sel], G)[0]


def boot(d, sel, rng, n=NBOOT):
    idx = np.flatnonzero(sel)
    return np.array([m1_of(d, rng.choice(idx, len(idx))) for _ in range(n)])


# The broad hump the C_M integration creates: it is NOT the noiseless lobes
# (which survive at 21%, see [[noiseless-oscillation-does-not-survive-cm]]) and
# NOT domain truncation (0.00-0.02% of kernel draws leave the chart over this
# range; truncation only bites above 3.5).  Top minus base, so a run that merely
# shifts m1 overall does not register as a hump.
HUMP_TOP, HUMP_BASE = (3.15, 3.40), (2.00, 2.60)


def _hump(d, y, rng, label="HUMP"):
    st = (y >= HUMP_TOP[0]) & (y < HUMP_TOP[1])
    sb = (y >= HUMP_BASE[0]) & (y < HUMP_BASE[1])
    if st.sum() < 200 or sb.sum() < 200:
        return None
    it, ib = np.flatnonzero(st), np.flatnonzero(sb)
    v = m1_of(d, st) - m1_of(d, sb)
    e = np.std([m1_of(d, rng.choice(it, len(it)))
                - m1_of(d, rng.choice(ib, len(ib))) for _ in range(NBOOT)])
    print(f"\n  {label}  m1[3.15,3.40) - m1[2.00,2.60) = {v:+.4f} +/- {e:.4f}   "
          f"(n={int(st.sum())}, {int(sb.sum())})")
    return v, e


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pqr", required=True)
    p.add_argument("--targets",
                   default="../bfd_cnf_imsims/data/targets_g0_1M.fits")
    p.add_argument("--nbin", type=int, default=8)
    p.add_argument("--control", default=None,
                   help="a dev/check_selfconsistency_noisy.py --save-pqr npz. "
                        "Binning on the TRUE Mr/Mf is a hidden-variable "
                        "selection and biases per-bin m1 even for a perfect "
                        "density, so the honest number is real MINUS control.")
    a = p.parse_args()

    d = np.load(a.pqr)
    n = len(d["plus_q"])
    y = np.asarray(shear_top.load(a.targets)[0][:n], np.float64)
    y = y[:, 1] / y[:, 0]
    rng = np.random.default_rng(0)
    print(f"{a.pqr}: {n} targets, true Mr/Mf in "
          f"[{y.min():.3f}, {y.max():.3f}]")

    all_ = np.ones(n, bool)
    print(f"\n  overall m1 = {m1_of(d, all_):+.4f} "
          f"+/- {boot(d, all_, rng).std():.4f}")

    print(f"\n=== 1. profile, {a.nbin} quantile bins of the TRUE Mr/Mf ===")
    ed = np.quantile(y, np.linspace(0, 1, a.nbin + 1))
    ed[0], ed[-1] = -np.inf, np.inf
    for k in range(a.nbin):
        s = (y >= ed[k]) & (y < ed[k + 1])
        if s.sum() < 100:
            continue
        v, e = m1_of(d, s), boot(d, s, rng).std()
        print(f"  Mr/Mf < {min(ed[k+1], y.max()):6.3f}  n={int(s.sum()):6d}  "
              f"m1 = {v:+.4f} +/- {e:.4f}  " + "*" * int(round(abs(v) / 0.005)))

    print("\n=== 2. pre-registered windows from the noiseless profile ===")
    draws = {}
    for lab, (lo, hi) in WINDOWS.items():
        s = (y >= lo) & (y < hi)
        if s.sum() < 100:
            print(f"  {lab}: only {int(s.sum())} targets, skipped")
            continue
        draws[lab] = boot(d, s, rng)
        print(f"  {lab}  n={int(s.sum()):6d}  m1 = {m1_of(d, s):+.4f} "
              f"+/- {draws[lab].std():.4f}")

    # The contrast, bootstrapped jointly so the two windows' shared targets and
    # shared draws do not get counted as independent error.
    pk, tr = "peak   (noiseless +0.0221)", "trough (noiseless -0.0062)"
    if pk in draws and tr in draws:
        sp = (y >= WINDOWS[pk][0]) & (y < WINDOWS[pk][1])
        st = (y >= WINDOWS[tr][0]) & (y < WINDOWS[tr][1])
        ip, it = np.flatnonzero(sp), np.flatnonzero(st)
        c = np.array([m1_of(d, rng.choice(ip, len(ip)))
                      - m1_of(d, rng.choice(it, len(it))) for _ in range(NBOOT)])
        obs = m1_of(d, sp) - m1_of(d, st)
        _hump(d, y, rng)
        print(f"\n  PEAK - TROUGH = {obs:+.4f} +/- {c.std():.4f}  "
              f"({abs(obs) / c.std():.1f} sigma from zero)")
        print(f"  same contrast, noiseless: {NOISELESS_CONTRAST:+.4f} +/- 0.0004")
        print(f"  ratio noisy/noiseless   : {obs / NOISELESS_CONTRAST:+.2f}  "
              f"({abs(obs - NOISELESS_CONTRAST) / c.std():.1f} sigma from "
              f"unattenuated)")

    if a.control:
        corrected(d, y, a.control, rng)


def corrected(d, y, path, rng):
    """real minus control, on shared fixed bin edges.

    The control's targets are flow draws, not catalog rows, so the two runs
    cannot share bin MEMBERSHIP -- only bin EDGES.  Fixed edges in Mr/Mf (not
    per-run quantiles) are what makes the subtraction meaningful.
    """
    z = np.load(path)
    dc = {k: z[k] for k in ("plus_q", "plus_r", "minus_q", "minus_r")}
    yc, keep = z["y"], z["keep"]
    print(f"\n=== 3. real MINUS control ({path}) ===")

    ed = np.array([2.00, 2.60, 3.00, 3.15, 3.40, 3.55, 3.70])
    print(f"  {'Mr/Mf bin':>15s} {'real':>18s} {'control':>18s} {'corrected':>18s}")
    for k in range(len(ed) - 1):
        s = (y >= ed[k]) & (y < ed[k + 1])
        sc = keep & (yc >= ed[k]) & (yc < ed[k + 1])
        if s.sum() < 200 or sc.sum() < 200:
            continue
        i, ic = np.flatnonzero(s), np.flatnonzero(sc)
        v, vc = m1_of(d, s), m1_of(dc, sc)
        e = np.std([m1_of(d, rng.choice(i, len(i))) for _ in range(NBOOT)])
        ec = np.std([m1_of(dc, rng.choice(ic, len(ic))) for _ in range(NBOOT)])
        print(f"  [{ed[k]:.2f},{ed[k+1]:.2f})  {v:+8.4f}+/-{e:.4f} "
              f"{vc:+8.4f}+/-{ec:.4f} {v-vc:+8.4f}+/-{np.hypot(e,ec):.4f}")

    hr = _hump(d, y, rng, "HUMP real   ")
    hc = _hump(dc, yc[keep] if keep.all() else yc, rng, "HUMP control")
    if hr and hc:
        print(f"\n  HUMP CORRECTED = {hr[0]-hc[0]:+.4f} +/- "
              f"{np.hypot(hr[1], hc[1]):.4f}   <- the flow's own contribution")


if __name__ == "__main__":
    main()
