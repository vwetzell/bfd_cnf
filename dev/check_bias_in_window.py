"""m1 inside a target selection window, with the window applied AFTER the
integration.

Every target's P, Q, R is computed on the full catalog -- so no cut touches the
C_M integral or the flow -- and only then are the out-of-window rows dropped
from the eq. (45)-(46) sums.  That is `bias.bias(..., sel=)`; this script is
the mask plus a bootstrap.

The cut is on the catalog's own **unsheared** moments (`bias.py --save-pqr`'s
`moments` column), so it is identical in the +g and -g arms and cannot make the
two select different galaxies.  It is still a selection: the estimator carries
the FULL population's prior, so eq. (40)/(46)'s non-selection terms are owed
and are not implemented.  Run `dev/check_selfconsistency_noisy.py --save-pqr`
and pass its npz as `--control` to measure what that costs on its own -- the
control's prior is exact by construction, so whatever m1 it shows inside the
same window is the selection term, not the flow.

    python bias.py --flow flows/centroid_e2e.eqx --pop bulgedisc_noisy \\
        --samples 8192 --n-targets 40000 --save-pqr dev/pqr_noisy.npz
    python dev/check_bias_in_window.py dev/pqr_noisy.npz --control dev/pqr_ctrl.npz
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

import bias                                            # noqa: E402

NBOOT = 300


def load(path):
    """(cut_moments_per_arm, [qp, rp, qm, rm]) from a --save-pqr npz.

    Prefers `obs_plus`/`obs_minus` -- each arm's OWN observed moments, which is
    the only cut the paper's eq. (40)/(45)-(46) can correct.  Falls back to the
    shared unsheared `moments` for npz files written before those columns
    existed (and for the control's, whose targets are flow draws): that cut is
    g-independent, so it carries no boundary term to correct, but it selects on
    a variable the estimator cannot see and its error is a WRONG PRIOR instead
    -- which nothing here removes.  The printed label says which you got.
    """
    d = np.load(path)
    keep = d.get("keep")
    qr = [d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"]]
    if "obs_plus" in d:
        m, kind = (d["obs_plus"], d["obs_minus"]), "own M"
    else:
        m, kind = (d["moments"], d["moments"]), "g=0 twin"
    if keep is not None:                               # control npz carries one
        m, qr = tuple(a[keep] for a in m), [a[keep] for a in qr]
    return m, qr, kind


def report(label, m, qr, size, flux, g, rng):
    sel = tuple(bias.window_mask(a, size, flux) for a in m)
    n = len(qr[0])
    for name, s in (("all targets", (np.ones(n, bool),) * 2), ("in window", sel)):
        v = bias.bias(*qr, g, sel=s)
        e = bias.bootstrap(*qr, g, NBOOT, 0, sel=s)
        print(f"  {label:16s} {name:11s} n={int(s[0].sum()):6d} "
              f"({s[0].mean():5.1%})  m1 = {v[0]:+.5f} +/- {e[0]:.5f}   "
              f"c1 = {v[1]:+.2e} +/- {e[1]:.1e}   "
              f"c2 = {v[2]:+.2e} +/- {e[2]:.1e}")
    return sel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pqr", help="--save-pqr npz from bias.py")
    p.add_argument("--control", help="--save-pqr npz from "
                                     "check_selfconsistency_noisy.py")
    p.add_argument("--size", type=float, nargs=2, default=(2.2, 3.2),
                   metavar=("LO", "HI"), help="Mr/Mf window")
    p.add_argument("--flux", type=float, nargs=2, default=(2500.0, 50000.0),
                   metavar=("LO", "HI"), help="Mf window")
    p.add_argument("--g", type=float, default=0.02)
    a = p.parse_args()

    rng = np.random.default_rng(0)
    print(f"window: {a.size[0]} < Mr/Mf < {a.size[1]}, "
          f"{a.flux[0]:g} < Mf < {a.flux[1]:g}  (applied after the integration)\n")
    m, qr, kind = load(a.pqr)
    report(f"run [{kind}]", m, qr, a.size, a.flux, a.g, rng)
    if a.control:
        print()
        mc, qc, kc = load(a.control)
        report(f"control [{kc}]", mc, qc, a.size, a.flux, a.g, rng)
    print("\n  These are UNCORRECTED. eq. (40)/(45)-(46)'s non-selection term "
          "needs P(s|g),\n  which needs the flow -- so the corrected number "
          "comes from bias.py itself:\n"
          "    python bias.py ... --window-size LO HI --window-flux LO HI\n"
          "  It only applies to an 'own M' cut; on a 'g=0 twin' cut the error "
          "is a wrong\n  prior, not a boundary term, and the correction makes "
          "it worse.")


if __name__ == "__main__":
    main()
