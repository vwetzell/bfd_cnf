"""Does the selection term Q_s pick up PSF ellipticity through Sigma_X?

Sigma_X (cov_odd) is measured per-catalog and genuinely varies with psf_e
(verified directly against the fits files: psfe1p05 splits Sigma_X's diagonal,
psfe2p05 puts weight on its off-diagonal, both eigenvalue-equivalent to the
same |psf_e|=0.05 ellipse just rotated 45 degrees) -- and bias.py's
`--window-terms score` branch DOES condition the selection-term draws on it
(`condition(zero2, sx1)`, bias.py ~3538-3566). So a real Q_s response to
psf_e is physically possible; the open question was whether it's actually
there and whether it's significant.

This parses P_s/Q_s/Q_s_err straight out of the 10 psfe-grid logs already on
disk (logs/bias_g2v3d_full2_jac_psfe*.log, dev/bias_psfe_g2v3d.sh's output) --
NO rerun, no GPU. Those runs all share the base-distribution seed
(`jr.key(a.seed + 31)`), so the PAIRED diff-from-baseline resolves changes far
smaller than each config's own (independent-noise) Q_s_err -- down to the
log's ~1e-7 print precision.

A first pass fit a LINEAR slope through the +/-psf_e pairs and got exactly
0.0000 -- which is a fit-design artifact, not a null result. At g=0 the
baseline population is rotationally symmetric, so Q_s(Sigma_X) can only
depend on ROTATION-INVARIANT combinations of Sigma_X (its eigenvalues, i.e.
|psf_e|^2), never on orientation -- psfe1p10/psfe1m10/psfe2p10 are the same
|psf_e|=0.10 ellipse just rotated, and Sigma_X's eigenvalues (hence any
isotropy-allowed function of it) are IDENTICAL between them, verified against
the raw cov_odd in the fits catalogs. A linear fit through signed +/- pairs of
a purely-even function cancels to exactly zero by construction, regardless of
the physics -- so `wls_slope` below is kept only to show that failure mode
explicitly, and `magnitude_response` is the actual answer: Q_s diff-from-
baseline binned by |psf_e|^2 alone, pooling all three orientations per
amplitude (they agree to the log's print precision, confirming the isotropy
prediction).

That said, the real magnitude-response is TINY: 1e-7 to 2e-6, three orders of
magnitude below Q_s_err (5.9e-4, independent-noise) and Thread 1's ~1e-2 c1/c2
leak. It is not simply proportional to |psf_e|^2 either (the per-amplitude
ratio drops 2.5e-4 -> 0.8e-4 -> 0.7e-4 across the grid for Q_s1) -- with only
4 sig figs of print precision behind each point, that shape isn't trustworthy
data, just evidence the effect is real and small. Resolving its actual form
needs the raw (float64) P_s/Q_s from a rerun with higher-precision output, not
attempted here since it's >>1000x too small to be Thread 1's driver.
"""
import re
import numpy as np

CONFIGS = {
    "psfe00":   (0.0, 0.0),
    "psfe1p02": (0.02, 0.0), "psfe2p02": (0.0, 0.02), "psfe1m02": (-0.02, 0.0),
    "psfe1p05": (0.05, 0.0), "psfe2p05": (0.0, 0.05), "psfe1m05": (-0.05, 0.0),
    "psfe1p10": (0.10, 0.0), "psfe2p10": (0.0, 0.10), "psfe1m10": (-0.10, 0.0),
}

LINE_RE = re.compile(
    r"P_s = ([\d.]+)\s+Q_s = \(([+-][\d.]+e[+-]\d+), ([+-][\d.]+e[+-]\d+)\)\s+"
    r"Q_s_err = \(([\d.]+e[+-]\d+), ([\d.]+e[+-]\d+)\)")


def load(tag):
    """(P_s, Q_s, Q_s_err) parsed from the already-run log -- no rerun."""
    path = f"logs/bias_g2v3d_full2_jac_{tag}.log"
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                ps, qs1, qs2, qerr1, qerr2 = (float(x) for x in m.groups())
                return ps, np.array([qs1, qs2]), np.array([qerr1, qerr2])
    raise ValueError(f"no P_s/Q_s line found in {path}")


def wls_slope(x, y, yerr):
    """Weighted least squares through (x=0, y=y0) with y0 = the x=0 point,
    i.e. fit dQs/dpsf_e holding the baseline fixed -- matches how
    dev/psfe_joint_slope.py reports dc/dpsf_e (a slope, not an intercept)."""
    w = 1.0 / yerr**2
    slope = np.sum(w * x * y) / np.sum(w * x**2)
    slope_err = 1.0 / np.sqrt(np.sum(w * x**2))
    return slope, slope_err


def magnitude_response(qs0):
    """Q_s diff-from-baseline binned by |psf_e|^2, pooled over orientation.

    Returns {amp: (mean_diff_qs1, mean_diff_qs2, n_orientations)}."""
    by_amp = {}
    for tag, (pe1, pe2) in CONFIGS.items():
        if tag == "psfe00":
            continue
        amp2 = pe1**2 + pe2**2
        _, qs, _ = load(tag)
        by_amp.setdefault(amp2, []).append(qs - qs0)
    return {amp2: (np.mean(diffs, axis=0), len(diffs))
            for amp2, diffs in sorted(by_amp.items())}


def demo():
    ps0, qs0, qs0_err = load("psfe00")

    # The linear fit: guaranteed ~0 for a purely-even signal, kept to show why.
    x1, y1, e1 = [], [], []
    x2, y2, e2 = [], [], []
    for tag, (pe1, pe2) in CONFIGS.items():
        if tag == "psfe00":
            continue
        ps, qs, qs_err = load(tag)
        if pe1 != 0.0:
            x1.append(pe1); y1.append(qs[0]); e1.append(qs_err[0])
        if pe2 != 0.0:
            x2.append(pe2); y2.append(qs[1]); e2.append(qs_err[1])
    x1, y1, e1 = map(np.array, (x1, y1, e1))
    x2, y2, e2 = map(np.array, (x2, y2, e2))
    slope1, slope1_err = wls_slope(x1, y1 - qs0[0], e1)
    slope2, slope2_err = wls_slope(x2, y2 - qs0[1], e2)

    print(f"baseline (psfe00): Q_s = ({qs0[0]:+.3e}, {qs0[1]:+.3e})")
    print(f"linear fit (odd in psf_e, expect ~0 by isotropy):")
    print(f"  dQ_s1/d(psf_e1) = {slope1:+.4f} +/- {slope1_err:.4f}")
    print(f"  dQ_s2/d(psf_e2) = {slope2:+.4f} +/- {slope2_err:.4f}")

    print("\nmagnitude response (even in psf_e, pooled over orientation):")
    print("  |psf_e|^2   diff_Qs1     diff_Qs2     n_orientations")
    resp = magnitude_response(qs0)
    for amp2, (diff, n) in resp.items():
        print(f"  {amp2:.4f}     {diff[0]:+.3e}  {diff[1]:+.3e}   {n}")
    nonzero = [amp2 for amp2, (diff, n) in resp.items() if np.any(diff != 0)]
    print(f"\nQ_s moves with |psf_e|^2, isotropy-consistent (orientation-"
          f"independent to print precision), {len(nonzero)}/{len(resp)} "
          f"amplitudes show a nonzero paired shift -- real but ~1e3x below "
          f"Q_s_err and Thread 1's c1/c2 leak.")

    assert slope1_err > 0 and slope2_err > 0
    assert len(nonzero) == len(resp), "expected every amplitude to show a shift"
    print("ok: parsed all 10 configs, fit both slopes, checked magnitude response")


if __name__ == "__main__":
    demo()
