"""Formal (delta-method / sandwich) uncertainty on (m1, c1, c2) vs. the
empirical bootstrap, for a saved PQR run.

`ghat` solves the LINEAR normal equation R @ g = Q per arm, with
R = -SUM r (treated as a fixed plug-in Hessian) and Q = SUM q (a sum of N
iid per-galaxy contributions). The standard M-estimator/delta-method
variance is then

    Var(g) = R^-1 Var(Q) R^-T,   Var(Q) = SUM_i (q_i - qbar)(q_i - qbar)^T

-- the usual "Var of a sum of iid terms" plug-in, no resampling. Plus and
minus arms share the same galaxy per row (the antithetic design), so their
per-galaxy q vectors are stacked into one 4-vector [q_p_i, q_m_i] (zeroed
in whichever arm didn't select that galaxy) so the cross-arm covariance the
paired bootstrap captures is captured here too, then propagated through the
LINEAR map (gp, gm) -> (m1, c1, c2) exactly (no further approximation,
since bias() is linear in gp, gm given R held fixed).

This treats R (and, for the corrected case, P_s/Q_s/R_s) as known constants,
i.e. it captures ONLY the finite-galaxy-catalog sampling variance -- the same
thing the bootstrap estimates. Where they disagree, the bootstrap is the
more trustworthy number: it makes no Gaussian/CLT assumption, so it doesn't
miss heavy-tailed contributions (e.g. the R_s11 concentration this codebase
has repeatedly found in this estimator) the way a leading-order sandwich
formula can.

Usage: python dev/formal_vs_bootstrap.py [pqr_path]
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

PS = 0.2335
QS = np.array([-1.482e-04, -2.238e-04])
RS = np.array([[0.05911, 0.00501], [0.00501, 0.06259]])
N_OUT_P, N_OUT_M = 381380.0, 381443.0


def formal_error(qp, rp, qm, rm, sel=None, ns=None):
    sel_p, sel_m = B._split_per_arm(sel)
    mp = np.ones(len(qp), bool) if sel_p is None else np.asarray(sel_p)
    mm = np.ones(len(qm), bool) if sel_m is None else np.asarray(sel_m)

    Rp = -rp[mp].sum(0)
    Rm = -rm[mm].sum(0)
    if ns is not None:
        ns_p, ns_m = B._split_per_arm(ns)
        n_p, ps_, qs_, rs_ = ns_p
        Rp = Rp + n_p * (np.outer(qs_, qs_) / (1 - ps_) ** 2 + rs_ / (1 - ps_))
        n_m, ps_, qs_, rs_ = ns_m
        Rm = Rm + n_m * (np.outer(qs_, qs_) / (1 - ps_) ** 2 + rs_ / (1 - ps_))

    x_p = np.where(mp[:, None], qp, 0.0)
    x_m = np.where(mm[:, None], qm, 0.0)
    x = np.concatenate([x_p, x_m], axis=1)          # (N, 4)
    xc = x - x.mean(0)
    Sigma = xc.T @ xc                               # Var(SUM x_i), (4,4)

    Jp, Jm = np.linalg.inv(Rp), np.linalg.inv(Rm)
    J = np.zeros((4, 4))
    J[:2, :2], J[2:, 2:] = Jp, Jm
    cov_g = J @ Sigma @ J.T                          # Cov([gp, gm]), (4,4)

    v_m1 = np.array([1 / (2 * G), 0, -1 / (2 * G), 0])
    v_c1 = np.array([0.5, 0, 0.5, 0])
    v_c2 = np.array([0, 0.5, 0, 0.5])
    return np.array([np.sqrt(v @ cov_g @ v) for v in (v_m1, v_c1, v_c2)])


def main():
    d = np.load(PATH)
    qp, rp, qm, rm = d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"]
    obs_p, obs_m = d["obs_plus"], d["obs_minus"]
    sel = (B.window_mask(obs_p, SIZE, FLUX), B.window_mask(obs_m, SIZE, FLUX))
    ns = ((N_OUT_P, PS, QS, RS), (N_OUT_M, PS, QS, RS))
    weights = d["prefilter_w"] if "prefilter_w" in d.files else None

    for label, s, n_ in [("unwindowed, full population", None, None),
                         ("windowed, uncorrected", sel, None),
                         ("windowed, corrected", sel, n_ if False else ns)]:
        point = B.bias(qp, rp, qm, rm, G, sel=s, ns=n_)
        boot = B.bootstrap(qp, rp, qm, rm, G, N_BOOT, 0, sel=s, ns=n_,
                           weights=weights if n_ is not None else None)
        formal = formal_error(qp, rp, qm, rm, sel=s, ns=n_)
        print(f"{label}:")
        print(f"  point:  m1={point[0]:+.5f}  c1={point[1]:+.3e}  c2={point[2]:+.3e}")
        print(f"  bootstrap: m1={boot[0]:.5f}  c1={boot[1]:.3e}  c2={boot[2]:.3e}")
        print(f"  formal:    m1={formal[0]:.5f}  c1={formal[1]:.3e}  c2={formal[2]:.3e}")
        print(f"  formal/bootstrap ratio: {formal/boot}\n")


if __name__ == "__main__":
    main()
