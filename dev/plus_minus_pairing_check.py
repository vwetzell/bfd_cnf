"""How much does the +g/-g antithetic pairing (same galaxy, same noise
realization, sheared both ways) buy over treating the two arms as
independent samples?

bias.py's own `bootstrap()` already does the right thing: one resampled
index applied to BOTH arms, so a galaxy's shape noise appears identically in
its plus and minus rows and cancels in (gp - gm)/(2g). This script also
builds the WRONG-on-purpose comparison -- an independent index per arm,
which breaks that per-galaxy correspondence -- to quantify the gap.

Usage: python dev/plus_minus_pairing_check.py [pqr_path]
"""
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import bias as B

PATH = sys.argv[1] if len(sys.argv) > 1 else "pqr/g2v3d_sigmaxblock_multiscale_500k.npz"
SIZE, FLUX = (2.2, 3.2), (2500.0, 50000.0)
G = 0.02
N_BOOT = 400

# The 500k run's own in-run selection terms (2^24 draws), from
# logs/bias_g2v3d_sigmaxblock_multiscale_500k.log -- reused here rather than
# recomputed, since they depend only on the flow/Sigma_X/window, not on which
# bootstrap variant is being tested.
PS = 0.2335
QS = np.array([-1.482e-04, -2.238e-04])
RS = np.array([[0.05911, 0.00501], [0.00501, 0.06259]])
N_OUT_P, N_OUT_M = 381380.0, 381443.0


def unpaired_bootstrap(qp, rp, qm, rm, sel=None, ns=None, n=N_BOOT, seed=0):
    """Same as bias.py's bootstrap(), except plus and minus arms are
    resampled with INDEPENDENT random indices -- breaking the per-galaxy
    correspondence the antithetic pairing relies on."""
    rng = np.random.default_rng(seed)
    sel_p, sel_m = B._split_per_arm(sel)
    ns_p, ns_m = B._split_per_arm(ns)
    idx0_p = np.arange(len(qp)) if sel_p is None else np.flatnonzero(sel_p)
    idx0_m = np.arange(len(qm)) if sel_m is None else np.flatnonzero(sel_m)
    out = []
    for _ in range(n):
        ip = rng.choice(idx0_p, len(idx0_p))
        im = rng.choice(idx0_m, len(idx0_m))
        gp = B.ghat(qp[ip], rp[ip], None, ns_p, None)
        gm = B.ghat(qm[im], rm[im], None, ns_m, None)
        out.append(((gp[0] - gm[0]) / (2 * G) - 1, *(0.5 * (gp + gm))))
    return np.std(np.array(out), axis=0)


def main():
    d = np.load(PATH)
    qp, rp, qm, rm = d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"]
    obs_p, obs_m = d["obs_plus"], d["obs_minus"]
    sel = (B.window_mask(obs_p, SIZE, FLUX), B.window_mask(obs_m, SIZE, FLUX))
    ns = ((N_OUT_P, PS, QS, RS), (N_OUT_M, PS, QS, RS))
    weights = d["prefilter_w"] if "prefilter_w" in d.files else None

    print(f"{PATH}: {len(qp)} rows, {sel[0].sum()}/{sel[1].sum()} in-window (plus/minus)\n")

    for label, s, n_ in [("unwindowed, full population", None, None),
                         ("windowed, uncorrected", sel, None),
                         ("windowed, corrected", sel, ns)]:
        point = B.bias(qp, rp, qm, rm, G, sel=s, ns=n_)
        paired = B.bootstrap(qp, rp, qm, rm, G, N_BOOT, 0, sel=s, ns=n_,
                             weights=weights if n_ is not None else None)
        unpaired = unpaired_bootstrap(qp, rp, qm, rm, s, n_)
        print(f"{label}:")
        print(f"  point:    m1={point[0]:+.5f}  c1={point[1]:+.3e}  c2={point[2]:+.3e}")
        print(f"  paired:   {paired}")
        print(f"  unpaired: {unpaired}")
        print(f"  unpaired/paired ratio: {unpaired/paired}\n")


if __name__ == "__main__":
    main()
