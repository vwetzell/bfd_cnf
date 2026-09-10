"""Apply `dev/rs_fd_check.py`'s bounded, from-scratch P_s/Q_s/R_s to an
existing integrated PQR catalog and report the corrected m1.

This is the same eq. (45)-(46) correction `bias.py`/`dev/window_scan.py`
already apply -- `ghat` is affine in `(P_s, Q_s, R_s)`, so re-quoting a saved
run's `m1` under a DIFFERENT selection-term estimate needs no target
integration, just `bias.bias(..., ns=(N_ns, P_s, Q_s, R_s))` per arm.  Only
the selection terms themselves are new here: they come from
`rs_fd_check.fd_selection_terms` (finite-differencing the BOUNDED window
probability `F`, never a per-draw score/Hessian of the flow's density), at
N=2^24 draws instead of 2^20, for a tighter bar.

`bias.window_mask` and `bias.bias` are reused -- plain, deterministic
utilities (a box test in observed-moment space; a closed-form solve for the
shear given Q, R, N_ns) that this session found no bugs in, unlike the
selection-term estimators.

    python -u dev/rs_fd_apply.py --pqr pqr/g2v3e_s0.npz pqr/g2v3e_s0_seed4242.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B  # noqa: E402
from rs_fd_check import load_flow_f64, fd_selection_terms, se  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pqr", nargs="+", required=True,
                   help="pqr/*.npz files to re-solve; all must share the "
                        "same population/flow the selection terms are built "
                        "from")
    p.add_argument("--pop", default="gauss2_v3e", choices=sorted(B.CATALOGS))
    p.add_argument("--flow", default="flows/centroid_g2v3d.eqx")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(2500.0, 50000.0))
    p.add_argument("--n", type=int, default=1 << 24, help="prior draws")
    p.add_argument("--h", type=float, default=0.02)
    p.add_argument("--g", type=float, default=0.02, help="matches `--h`: "
                   "ghat's own operating point")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--boot", type=int, default=200)
    a = p.parse_args()

    flow, cov, sigma_x = load_flow_f64(a.pop, a.flow, a.data_dir)
    print(f"computing selection terms: {a.n} draws, h={a.h} ...", flush=True)
    grand, per_chunk = fd_selection_terms(
        flow, cov, sigma_x, a.window_size, a.window_flux, a.n, a.h, a.chunk,
        a.seed)
    P_s = grand["P_s"]
    Q_s = np.array([grand["Q_s1"], grand["Q_s2"]])
    R_s = np.array([[grand["R_s11"], grand["R_s12"]],
                    [grand["R_s12"], grand["R_s22"]]])

    print(f"\nP_s = {P_s:+.5e} +/- {se(per_chunk, 'P_s'):.2e}")
    print(f"Q_s = ({Q_s[0]:+.4e}, {Q_s[1]:+.4e})  +/- "
          f"({se(per_chunk, 'Q_s1'):.2e}, {se(per_chunk, 'Q_s2'):.2e})")
    print(f"R_s = [[{R_s[0,0]:+.4e}, {R_s[0,1]:+.4e}], "
          f"[{R_s[1,0]:+.4e}, {R_s[1,1]:+.4e}]]")
    print(f"R_s11 - R_s22 = {grand['iso_diff']:+.4e}  "
          f"({abs(grand['iso_diff']) / se(per_chunk, 'iso_diff'):.1f} sigma "
          f"from 0)\n")

    for pqr_path in a.pqr:
        print(f"{'=' * 90}\n{pqr_path}")
        d = np.load(pqr_path)
        qp, rp, qm, rm = d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"]
        obs_p, obs_m = d["obs_plus"], d["obs_minus"]
        if "prefilter_w" not in d.files:
            raise SystemExit(
                f"{pqr_path} has no `prefilter_w` -- N_ns would be "
                f"under-counted for any pre-filtered run. See "
                f"`dev/window_scan.py`'s own guard for why this is fatal, "
                f"not a warning.")
        pw = d["prefilter_w"]

        sp = B.window_mask(obs_p, a.window_size, a.window_flux)
        sm = B.window_mask(obs_m, a.window_size, a.window_flux)
        n_ns_p, n_ns_m = float(pw[~sp].sum()), float(pw[~sm].sum())
        ns = ((n_ns_p, P_s, Q_s, R_s), (n_ns_m, P_s, Q_s, R_s))

        m1u, _, _ = B.bias(qp, rp, qm, rm, a.g, sel=(sp, sm))
        m1c, c1c, c2c = B.bias(qp, rp, qm, rm, a.g, sel=(sp, sm), ns=ns)

        # Galaxy bootstrap for the error bar, same recipe `window_scan.py`
        # uses: resample targets, recompute the window mask and N_ns on the
        # SAME resample (N_ns is data, not a fixed constant).
        rng = np.random.default_rng(a.seed)
        n_all = len(qp)
        boots = np.empty((a.boot, 3))
        for b in range(a.boot):
            idx = rng.choice(n_all, n_all)
            spi, smi = sp[idx], sm[idx]
            pwi = pw[idx]
            boots[b] = B.bias(
                qp[idx], rp[idx], qm[idx], rm[idx], a.g, sel=(spi, smi),
                ns=((float(pwi[~spi].sum()), P_s, Q_s, R_s),
                    (float(pwi[~smi].sum()), P_s, Q_s, R_s)))
        m1_err, c1_err, c2_err = boots.std(axis=0, ddof=1)

        print(f"  n_in = {int(sp.sum())} / {n_all}   "
              f"N_ns = ({n_ns_p:.1f}, {n_ns_m:.1f})")
        print(f"  uncorrected m1 = {m1u:+.5f}")
        print(f"  corrected   m1 = {m1c:+.5f} +/- {m1_err:.5f} (galaxy)")
        print(f"  c1 = {c1c:+.3e} +/- {c1_err:.2e}   "
              f"c2 = {c2c:+.3e} +/- {c2_err:.2e}")


if __name__ == "__main__":
    main()
