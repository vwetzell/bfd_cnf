"""Self-checks for the flux-limit selection PQR term and its application.

Run: python tests/test_selection_pqr.py   (or via pytest)

The trained flow is replaced by an **affine-Gaussian** stub exposing
``.bijection.transform(z, condition) = A@z + b0 + B@g`` (g = condition[:2]), so the
prior ``p_theta(x|g)`` is exactly Gaussian and the flux-only selection depends only on
``x_f`` (dim 0).  That makes ``P_sel(g)`` a 1-D integral we can compute independently by
quadrature, pinning every sign / normalisation in ``selection_pqr``.  The remaining
checks guard the σ_f-bin weighting, the ``apply_selection`` no-op, and — crucially —
that the bootstrap subtracts the selection term *per object, inside the resample*.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax  # noqa: E402  (JAX_PLATFORMS must be set before import)
import jax.numpy as jnp
import jax.random as jr

# The package runs in float32 (config.py asserts x64 is disabled); match it here.

from bfd_cnf.inference import selection_pqr, selection_pqr_binned
from bfd_cnf.statistics import (
    pqr2g,
    qr_log_totals,
    apply_selection,
    bootstrap_independent_mult_bias,
)

KEY = jr.PRNGKey(0)

# Affine-Gaussian stub: x = A z + b0 + B g  ⇒  p(x|g) = N(b0 + B g, A Aᵀ).
_A = jnp.diag(jnp.array([1.0, 0.8, 0.9, 0.7]))
_B0 = jnp.zeros(4)
_B = jnp.array([[0.30, 0.10], [0.05, -0.04], [0.0, 0.0], [0.0, 0.0]])
_MEAN = jnp.array([3.5, 0.0, 0.0, 0.0])   # log10(Mf) mean etc.
_STD = jnp.array([0.25, 1.0, 1.0, 1.0])


class _AffineBij:
    def transform(self, z, condition):
        g = condition[:2]
        return _A @ z + _B0 + _B @ g


class _AffineFlow:
    bijection = _AffineBij()


class _R2S:
    mean = _MEAN
    std = _STD


FLOW = _AffineFlow()
R2S = _R2S()


def _P_sel_truth(g, f_min, f_max, sigma_f, n_grid=4000):
    """Independent 1-D quadrature of P_sel(g) for the affine-Gaussian prior."""
    mu_f = float(_B0[0] + _B[0, 0] * g[0] + _B[0, 1] * g[1])
    sig_f = float(_A[0, 0])                      # sqrt((A Aᵀ)[0,0])
    x0 = np.linspace(mu_f - 8 * sig_f, mu_f + 8 * sig_f, n_grid)
    mf = 10.0 ** (x0 * float(_STD[0]) + float(_MEAN[0]))
    from scipy.special import ndtr as _ndtr  # match jax.scipy ndtr
    s = _ndtr((f_max - mf) / sigma_f) - _ndtr((f_min - mf) / sigma_f)
    w = np.exp(-0.5 * ((x0 - mu_f) / sig_f) ** 2) / (np.sqrt(2 * np.pi) * sig_f)
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(s * w, x0))


def _truth_pqr(f_min, f_max, sigma_f, eps=1e-3):
    """[P,Q1,Q2,R11,R22,R12] by finite-differencing the quadrature P_sel(g) at g=0."""
    P = _P_sel_truth([0, 0], f_min, f_max, sigma_f)

    def Pf(g1, g2):
        return _P_sel_truth([g1, g2], f_min, f_max, sigma_f)

    q1 = (Pf(eps, 0) - Pf(-eps, 0)) / (2 * eps)
    q2 = (Pf(0, eps) - Pf(0, -eps)) / (2 * eps)
    r11 = (Pf(eps, 0) - 2 * P + Pf(-eps, 0)) / eps**2
    r22 = (Pf(0, eps) - 2 * P + Pf(0, -eps)) / eps**2
    r12 = (Pf(eps, eps) - Pf(eps, -eps) - Pf(-eps, eps) + Pf(-eps, -eps)) / (4 * eps**2)
    return np.array([P, q1, q2, r11, r22, r12])


def test_selection_pqr_matches_quadrature():
    """selection_pqr matches the 1-D-quadrature ground truth at two σ_f."""
    for sigma_f in (300.0, 800.0):
        got = np.asarray(
            selection_pqr(FLOW, R2S, 1500.0, 9000.0, sigma_f,
                          n_samples=2**16, key=KEY)
        )
        ref = _truth_pqr(1500.0, 9000.0, sigma_f)
        assert np.isclose(got[0], ref[0], atol=3e-3), (got[0], ref[0])
        assert np.allclose(got[1:3], ref[1:3], atol=3e-3, rtol=0.05), (got[1:3], ref[1:3])
        # R is 2nd-order (FD-truncated truth + MC); looser tolerance.
        assert np.allclose(got[3:], ref[3:], atol=0.05, rtol=0.1), (got[3:], ref[3:])


def test_full_band_is_noop():
    """Selecting everything (f_min≈0, f_max huge) ⇒ P_sel≈1, Q≈0, R≈0."""
    got = np.asarray(
        selection_pqr(FLOW, R2S, 0.0, 1e12, 300.0, n_samples=2**16, key=KEY)
    )
    assert np.isclose(got[0], 1.0, atol=1e-3), got[0]
    assert np.allclose(got[1:], 0.0, atol=1e-3), got[1:]


def test_binned_weighting():
    """selection_pqr_binned's q_tot_sel is the count-weighted sum of per-bin totals."""
    out = selection_pqr_binned(
        FLOW, R2S, 1500.0, 9000.0,
        sigma_f_bins=np.array([300.0, 600.0]),
        counts=np.array([3.0, 5.0]),
        n_samples=2**15, key=KEY,
    )
    q_b, r_b = qr_log_totals(out["pqr_bins"])
    assert np.allclose(out["q_tot_b"], q_b)
    assert np.allclose(out["q_tot_sel"], 3.0 * q_b[0] + 5.0 * q_b[1])
    assert np.allclose(out["r_tot_sel"], 3.0 * r_b[0] + 5.0 * r_b[1])


def _make_arm(n, seed):
    rng = np.random.default_rng(seed)
    P = np.exp(0.5 * rng.standard_normal(n)) + 0.1
    Q = 0.2 * rng.standard_normal((n, 2))
    R = 0.1 * rng.standard_normal((n, 3))
    return np.column_stack([P, Q, R])


def test_apply_selection_noop_and_shift():
    """apply_selection with no term == pqr2g; a nonzero term shifts ĝ."""
    p = _make_arm(300, 1)
    assert np.allclose(apply_selection(p), pqr2g(p), atol=1e-10)
    n = p.shape[0]
    sel_q = np.tile(np.array([0.05, -0.03]), (n, 1))
    sel_r = np.zeros((n, 2, 2))
    g0 = apply_selection(p)
    g1 = apply_selection(p, sel_q, sel_r)
    assert not np.allclose(g0, g1)


def test_bootstrap_selection_is_per_object():
    """Bootstrap subtracts the selection term per object, INSIDE the resample.

    Set each object's selection Q_tot to its own measured Q_tot ⇒ net Q_tot = 0 for
    every object ⇒ every resample gives ĝ = 0 ⇒ m = -1 with zero spread.  A fixed
    (outside-resample) subtraction could not make every resample's Q-sum vanish.
    """
    p = _make_arm(200, 1)
    m = _make_arm(220, 2)
    Qp, _ = qr_log_totals(p)
    Qm, _ = qr_log_totals(m)
    zp = np.zeros((p.shape[0], 2, 2))
    zm = np.zeros((m.shape[0], 2, 2))

    # No-op consistency: sel=None matches the plain per-arm mult bias.
    none = bootstrap_independent_mult_bias(p, m, n_boot=100, key=KEY)
    m_ref = (pqr2g(p)[0] - pqr2g(m)[0]) / 0.04 - 1.0
    assert np.isclose(float(none["m_point"]), m_ref, atol=1e-6)

    # Point consistency with apply_selection under a real per-object term.
    selq_p = np.tile(np.array([0.05, -0.02]), (p.shape[0], 1))
    selq_m = np.tile(np.array([0.05, -0.02]), (m.shape[0], 1))
    out = bootstrap_independent_mult_bias(
        p, m, n_boot=50, key=KEY,
        sel_qtot_p=selq_p, sel_rtot_p=zp, sel_qtot_m=selq_m, sel_rtot_m=zm,
    )
    g_p = apply_selection(p, selq_p, zp)
    g_m = apply_selection(m, selq_m, zm)
    assert np.isclose(float(out["m_point"]), (g_p[0] - g_m[0]) / 0.04 - 1.0, atol=1e-6)

    # Per-object, inside-resample: net Q_tot ≈ 0 ⇒ every resample ĝ ≈ 0 ⇒ m ≈ -1 with a
    # spread collapsed to float32 noise — orders below the uncorrected spread.  A fixed
    # (outside-resample) subtraction could not make each resample's Q-sum vanish.
    zero = bootstrap_independent_mult_bias(
        p, m, n_boot=100, key=KEY,
        sel_qtot_p=Qp, sel_rtot_p=zp, sel_qtot_m=Qm, sel_rtot_m=zm,
    )
    assert np.isclose(float(zero["m_point"]), -1.0, atol=1e-3)
    assert float(zero["m_std"]) < 1e-3
    assert float(zero["m_std"]) < 0.01 * float(none["m_std"])


if __name__ == "__main__":
    test_selection_pqr_matches_quadrature()
    test_full_band_is_noop()
    test_binned_weighting()
    test_apply_selection_noop_and_shift()
    test_bootstrap_selection_is_per_object()
    print("OK: selection PQR term, binning, apply, and per-object bootstrap")
