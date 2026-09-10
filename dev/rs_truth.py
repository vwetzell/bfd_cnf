"""Is the `R_s` spike the FLOW being wrong, or the true density really doing that?

`dev/rs_census.py` localises `R_s11`'s heavy tail on gauss2 to a narrow band at
`Mr/Mf ~ 3.31-3.42`, `Mc/Mr ~ 6.10-6.26` -- ~92% of the way to both chart
ceilings -- where the flow's `R11` and `Q1^2` both reach ~4e5 against a
population median of order 1.  gauss2 is the ONLY population that can settle
what that means, because `truth.pqr` gives the exact `Q = d log P/dg` and
`R = d2 log P/dg2` in closed form at any moment vector.

Three outcomes, three different fixes:

  * exact Q, R spike too  -> the integrand really is heavy-tailed there, `R_s`
    has no effective sample size, and the estimator (not the flow) needs
    rethinking;
  * exact Q, R are ordinary -> the FLOW is wrong on that band; it is a density
    defect to be trained or bounded away, and [[mc-mr-ceiling-bound]] is the
    nearest existing lever;
  * `truth.folded` fires on the leaders -> the moment map is non-injective
    there, `theta_of_m` is returning the wrong root, and neither the flow nor
    the estimator is at fault -- the CHART is.

Run: python dev/rs_truth.py --leaders logs/rs_leaders_g2v3.npz
"""
import argparse
import os
import sys

import fitsio
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B  # noqa: E402
import truth  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--leaders", default="logs/rs_leaders_g2v3.npz")
    p.add_argument("--pop", default="gauss2_v3")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--n", type=int, default=64, help="leaders to evaluate")
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(2500.0, 50000.0))
    a = p.parse_args()

    z = np.load(a.leaders)
    m = z["moments"][:a.n]
    fr11, fq1sq = z["r11"][:a.n], z["q1sq"][:a.n]

    targets = f"{a.data_dir}/{B.CATALOGS[a.pop]['zero']}.fits"
    cov = B.load_cov(targets)
    F = np.asarray(B.window_prob(jnp.asarray(m), cov, a.window_size,
                                 a.window_flux), dtype=np.float64)
    safe = np.where(F > 1e-12, F, np.nan)
    r11_flow, q1_flow_sq = fr11 / safe, fq1sq / safe

    # The exact answer at the SAME moment vectors.
    q_ex, r_ex = truth.pqr_batch(jnp.asarray(m, dtype=jnp.float64))
    q_ex, r_ex = np.asarray(q_ex), np.asarray(r_ex)
    r11_ex, q1sq_ex = r_ex[:, 0, 0], q_ex[:, 0] ** 2

    # `nopre` = NO PREIMAGE, and it is NOT `truth.folded` -- which takes the
    # GENERATING theta and a flow draw has
    # none.  (The column was called `fold`, which invites exactly that
    # misreading; renamed 2026-09-06.)  The usable test on a draw is whether
    # the Newton solve lands on a
    # preimage at all, and whether that preimage is inside the population box
    # (log_p_theta finite).  A draw that fails either is one the flow invented.
    th, resid = truth.analytic.theta_of_m_batch(jnp.asarray(m, dtype=jnp.float64))
    resid = np.asarray(resid)
    lpt = np.asarray(truth.log_p_theta(th))
    nopre = ~np.isfinite(lpt)
    print(f"{len(m)} leaders from {a.leaders}")
    print(f"  Newton residual: median {np.median(resid):.2e}  "
          f"max {resid.max():.2e}   (unsolved > 1e-10: {(resid > 1e-10).sum()})")
    print(f"  preimage OUTSIDE the population box (log_p_theta = -inf): "
          f"{nopre.sum()}/{len(m)} ({nopre.mean():.0%})\n")

    print(f"  {'Mr/Mf':>7} {'Mc/Mr':>7} {'Mf':>10} {'F':>7} "
          f"{'R11 flow':>11} {'R11 exact':>11} {'Q1^2 flow':>11} "
          f"{'Q1^2 exact':>11} {'nopre':>5}")
    for i in range(min(15, len(m))):
        print(f"  {m[i, 1] / m[i, 0]:>7.4f} {m[i, 4] / m[i, 1]:>7.4f} "
              f"{m[i, 0]:>10.4g} {F[i]:>7.4f} {r11_flow[i]:>11.4g} "
              f"{r11_ex[i]:>11.4g} {q1_flow_sq[i]:>11.4g} "
              f"{q1sq_ex[i]:>11.4g} {str(bool(nopre[i])):>5}")

    ok = np.isfinite(r11_flow) & np.isfinite(r11_ex) & ~nopre
    if ok.sum():
        print(f"\n  over the {ok.sum()} leaders WITH a valid preimage with F > 0:")
        print(f"    median |R11|   flow {np.median(np.abs(r11_flow[ok])):.4g}"
              f"   exact {np.median(np.abs(r11_ex[ok])):.4g}")
        print(f"    median  Q1^2   flow {np.median(q1_flow_sq[ok]):.4g}"
              f"   exact {np.median(q1sq_ex[ok]):.4g}")
        print(f"    max    |R11|   flow {np.max(np.abs(r11_flow[ok])):.4g}"
              f"   exact {np.max(np.abs(r11_ex[ok])):.4g}")


if __name__ == "__main__":
    main()
