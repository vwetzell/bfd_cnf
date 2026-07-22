"""Self-check for the bfd-backed PQR helpers in bfd_cnf.statistics.

Run: python tests/test_pqr_bfd.py
Checks (assert-based, no framework) that routing all Q/R math through ``bfd``
reproduces the flow-convention formulas — i.e. the [R11,R22,R12] <-> [R11,R12,R22]
column swap is handled in exactly one place and correctly:
  1. to_bfd is an involution and swaps the last two columns.
  2. split_qr assembles R with flow semantics: R=[[R11,R12],[R12,R22]].
  3. qr_log_totals matches the reference per-object Q_tot, R_tot.
  4. pqr2g matches the reference summed ML shear g=(ΣR_tot)⁻¹ΣQ_tot.
  5. clipR round-trips (no-op when eigenvalues are already in range).
"""

import numpy as np

from bfd_cnf.statistics import to_bfd, from_bfd, split_qr, qr_log_totals, pqr2g, clipR


def _ref_terms(pqr):
    """Independent reference for flow order [P,Q1,Q2,R11,R22,R12]."""
    P = pqr[:, 0]
    Q = pqr[:, 1:3]
    R = np.empty((pqr.shape[0], 2, 2))
    R[:, 0, 0] = pqr[:, 3]
    R[:, 1, 1] = pqr[:, 4]
    R[:, 0, 1] = R[:, 1, 0] = pqr[:, 5]
    Q_tot = Q / P[:, None]
    R_tot = np.einsum("ni,nj->nij", Q, Q) / P[:, None, None] ** 2 - R / P[:, None, None]
    return Q_tot, R_tot


def main() -> None:
    rng = np.random.default_rng(0)
    n = 500
    # P in [0.5,1.5] so nothing is masked; Q,R arbitrary.
    pqr = np.column_stack([
        rng.uniform(0.5, 1.5, n),      # P
        rng.normal(size=(n, 5)),       # Q1,Q2,R11,R22,R12
    ])

    # 1. involution + column swap
    assert np.array_equal(to_bfd(to_bfd(pqr)), pqr)
    b = to_bfd(pqr)
    assert np.array_equal(b[:, [3, 4, 5]], pqr[:, [3, 5, 4]])  # R11, R12<-R12, R22<-R22
    assert from_bfd is to_bfd

    # 2. split_qr has flow semantics
    P, Q, R = split_qr(pqr)
    assert np.allclose(P, pqr[:, 0])
    assert np.allclose(Q, pqr[:, 1:3])
    assert np.allclose(R[:, 0, 0], pqr[:, 3])   # R11
    assert np.allclose(R[:, 1, 1], pqr[:, 4])   # R22
    assert np.allclose(R[:, 0, 1], pqr[:, 5])   # R12
    assert np.allclose(R[:, 1, 0], pqr[:, 5])

    # 3. qr_log_totals vs reference
    q_ref, r_ref = _ref_terms(pqr)
    q, r = qr_log_totals(pqr)
    assert np.allclose(q, q_ref), np.abs(q - q_ref).max()
    assert np.allclose(r, r_ref), np.abs(r - r_ref).max()

    # 4. pqr2g vs reference summed solve
    g_ref = np.linalg.solve(r_ref.sum(0), q_ref.sum(0))
    g = np.asarray(pqr2g(pqr))
    assert np.allclose(g, g_ref), np.abs(g - g_ref).max()

    # 5. clipR is a no-op (round-trips flow order) when bounds don't bind.
    #    Build R_tot with eigenvalues safely inside [-100, 100].
    same = clipR(pqr, lower=-1e6, upper=1e6)
    assert np.allclose(same, pqr), np.abs(same - pqr).max()

    print("OK: all bfd PQR interop checks passed")


if __name__ == "__main__":
    main()
