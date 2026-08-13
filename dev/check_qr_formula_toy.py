"""check_qr_formula_toy.py
==========================
Purest possible unit test of the Q_tot/R_tot/g_hat aggregation formula (BFD
eq 16-17), with a closed-form toy P(m|g) so A(m), B(m), P(m) are all exactly
known by hand -- zero dependence on the real flow, moments, or any trained
net. If this doesn't recover g_true, the bug is in the aggregation/estimator
code itself, not in template physics or what a net learned.

Toy model: 2D m (matching g's dimensionality, so the Fisher matrix is
non-degenerate -- an earlier scalar-m version had a RANK-1, singular Fisher
matrix, which is a defect in that toy problem, not a test of the formula),
P(m|g) = N(m; mu0 + g, Sigma0) -- shear shifts the mean directly. Closed form
(standard multivariate-Gaussian score/Hessian identities):
    A(m) = dP/dg|0   = P0(m) * Sigma0^{-1} (m - mu0)
    B(m) = d2P/dg2|0 = P0(m) * [Sigma0^{-1}(m-mu0)(m-mu0)^T Sigma0^{-1} - Sigma0^{-1}]
Generate m_i ~ N(mu0 + g_true, Sigma0) for a KNOWN g_true, form
Q_tot=mean(A_i/P_i), R_tot=mean(A_iA_i^T/P_i^2 - B_i/P_i), g_hat=solve(R_tot,Q_tot).

Usage:
    python dev/check_qr_formula_toy.py
"""
from __future__ import annotations

import numpy as np


def main() -> None:
    rng = np.random.default_rng(0)
    mu0 = np.array([0.0, 0.0])
    Sigma0 = np.array([[1.0, 0.3], [0.3, 0.8]])
    Sigma0_inv = np.linalg.inv(Sigma0)
    g_true = np.array([0.02, 0.01])
    n = 2_000_000

    m = rng.multivariate_normal(mu0 + g_true, Sigma0, size=n)

    def P0(m):
        d = m - mu0
        quad = np.einsum("bi,ij,bj->b", d, Sigma0_inv, d)
        norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma0)))
        return norm * np.exp(-0.5 * quad)

    P = P0(m)
    d = m - mu0  # (n,2)
    s = np.einsum("ij,bj->bi", Sigma0_inv, d)  # Sigma0^{-1}(m-mu0), (n,2)
    A = P[:, None] * s  # (n,2)
    B = P[:, None, None] * (
        np.einsum("bi,bj->bij", s, s) - Sigma0_inv[None, :, :]
    )  # (n,2,2)

    Q_tot = np.mean(A / P[:, None], axis=0)
    R_tot = np.mean(
        np.einsum("bi,bj->bij", A, A) / (P ** 2)[:, None, None] - B / P[:, None, None],
        axis=0,
    )
    g_hat = np.linalg.solve(R_tot, Q_tot)

    print(f"g_true = {g_true}")
    print(f"g_hat  = {g_hat}")
    print(f"relative error = {(g_hat - g_true) / g_true}")


if __name__ == "__main__":
    main()
