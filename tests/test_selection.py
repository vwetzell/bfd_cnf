"""BFD's selection terms (Bernstein et al. 2016, eq. 40, 45, 46).

Run: `python -m tests.test_selection` or pytest.  Everything here is
synthetic -- no flow, no FITS file -- so it runs in seconds on CPU.

These are exact-quadrature / exact-derivative checks (`window_prob` against a
brute-force Monte Carlo integral, `selection_terms`'s autodiff against finite
differences of the SAME data), so they run in float64: the estimator's own
float32 storage would swamp a 1e-3 relative tolerance with roundoff rather
than testing the thing the tolerance is meant to catch.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp        # noqa: E402  -- must follow the x64 switch
import numpy as np             # noqa: E402

from bias import bias, selection_terms, window_mask, window_prob  # noqa: E402

# A window and a moment covariance with a real (Mf, Mr) correlation, shared by
# tests 1 and 2.  Mf, Mr on the same order as a real target (~1e3), rho = 0.6
# so the quadrature's `sc = sr sqrt(1 - rho^2)` term actually does something.
SF, SR, RHO = 50.0, 30.0, 0.6
COV = np.eye(5)
COV[0, 0], COV[1, 1] = SF ** 2, SR ** 2
COV[0, 1] = COV[1, 0] = RHO * SF * SR
COV[2, 2] = COV[3, 3] = COV[4, 4] = 100.0   # unused by window_prob; just PD

SIZE = (2.0, 3.0)          # Mr/Mf
FLUX = (800.0, 1600.0)     # Mf

# ~20 rows spanning deep inside, deep outside, and right at each of the four
# edges (two from the flux cut, two from the size cut) -- the other three
# moment slots are irrelevant to `window_prob` and left at zero.
_MF = [1200, 800, 1600, 1200, 1200, 1200, 1200, 200, 5000, 810, 1590,
       1000, 1000, 900, 1500, 700, 1700, 950, 1050, 1300]
_RATIO = [2.5, 2.5, 2.5, 2.0, 3.0, 1.0, 5.0, 2.5, 2.5, 2.05, 2.95,
          1.99, 3.01, 2.2, 2.8, 2.5, 2.5, 2.01, 2.99, 2.6]
ROWS = np.zeros((len(_MF), 5))
ROWS[:, 0] = _MF
ROWS[:, 1] = np.array(_MF) * np.array(_RATIO)


def test_window_prob_matches_brute_force():
    """`window_prob(m, COV, ...)` against a 200k-sample Monte Carlo estimate
    of the same probability, for every row above."""
    n_mc = 200_000
    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(COV)
    p_true = np.asarray(window_prob(jnp.asarray(ROWS), jnp.asarray(COV), SIZE, FLUX))

    for i, row in enumerate(ROWS):
        samp = row + rng.standard_normal((n_mc, 5)) @ L.T
        p_hat = window_mask(samp, SIZE, FLUX).mean()
        se = np.sqrt(p_hat * (1 - p_hat) / n_mc)
        tol = 4 * se + 2e-3   # MC error plus a small floor for p near 0 or 1
        assert abs(p_true[i] - p_hat) < tol, (i, row, p_true[i], p_hat, tol)


def test_window_prob_gradient():
    """Central finite differences of `window_prob` w.r.t. Mf and Mr against
    `jax.grad`, on rows where the gradient is not ~0 (deep inside/outside the
    window it decays toward the float64 finite-difference noise floor faster
    than the signal, which is a finite-difference artifact, not a bug)."""
    h_mf, h_mr = 0.02 * SF, 0.02 * SR
    cov = jnp.asarray(COV)

    def f(m):
        return window_prob(m, cov, SIZE, FLUX)

    n_checked = 0
    for row in ROWS:
        m0 = jnp.asarray(row)
        grad = np.asarray(jax.grad(f)(m0))
        for k, h in ((0, h_mf), (1, h_mr)):
            if abs(grad[k]) < 1e-4:
                continue
            fd = (float(f(m0.at[k].add(h))) - float(f(m0.at[k].add(-h)))) / (2 * h)
            rel = abs(grad[k] - fd) / abs(grad[k])
            assert rel < 1e-3, (row, k, grad[k], fd, rel)
            n_checked += 1
    assert n_checked > 10, "too many rows filtered out; test too weak"


# A trivial, analytically differentiable model for test 3: `draw` just shifts
# a fixed base sample linearly in g, so `selection_terms`'s autodiff has a
# closed-form finite-difference check that needs no flow.
V1 = np.array([120.0, -60.0, 5.0, 5.0, 30.0])
V2 = np.array([-40.0, 90.0, 5.0, 5.0, -20.0])


def test_selection_terms_self_consistent():
    """`selection_terms`'s (Q_s, R_s) against central finite differences of
    P_s recomputed at +/-h with the SAME z -- checks the autodiff, not the
    Monte Carlo convergence (which test 1 already covers)."""
    rng = np.random.default_rng(3)
    n = 80_000
    L = np.linalg.cholesky(COV)
    z_mean = np.array([950.0, 2280.0, 0.0, 0.0, 0.0])
    z = jnp.asarray(z_mean + rng.standard_normal((n, 5)) @ L.T)

    draw = lambda g, zz: zz + g[0] * jnp.asarray(V1) + g[1] * jnp.asarray(V2)
    ps, qs, rs, qs_err = selection_terms(draw, z, COV, SIZE, FLUX, batch=16_000)

    def p_direct(g):
        return float(jnp.mean(window_prob(draw(jnp.asarray(g), z), jnp.asarray(COV),
                                          SIZE, FLUX)))

    h = 2e-3
    f0 = p_direct([0.0, 0.0])
    q_fd = np.array([(p_direct([h, 0]) - p_direct([-h, 0])) / (2 * h),
                     (p_direct([0, h]) - p_direct([0, -h])) / (2 * h)])
    assert np.max(np.abs(qs - q_fd) / np.abs(q_fd)) < 1e-3, (qs, q_fd)

    r_fd = np.zeros((2, 2))
    r_fd[0, 0] = (p_direct([h, 0]) - 2 * f0 + p_direct([-h, 0])) / h ** 2
    r_fd[1, 1] = (p_direct([0, h]) - 2 * f0 + p_direct([0, -h])) / h ** 2
    r_fd[0, 1] = r_fd[1, 0] = (p_direct([h, h]) - p_direct([h, -h])
                              - p_direct([-h, h]) + p_direct([-h, -h])) / (4 * h ** 2)
    assert np.max(np.abs(rs - r_fd) / np.abs(r_fd)) < 1e-3, (rs, r_fd)
    assert np.all(qs_err > 0), "multiple chunks must give a nonzero SEM"


def test_selection_correction_removes_the_bias():
    """The identity that is the whole point: a fully analytic 5-D linear-
    Gaussian model, where q_i, r_i are known exactly, shows that eq. (45)-(46)
    restores the unselected estimator's unbiasedness.

    `z ~ N(mu, S)` (the latent galaxy), `m(g) = z + g0 v1 + g1 v2` (its shear
    response), observed `M = m(g) + n` with `n ~ N(0, C)`.  Then exactly
    `P(M|g) = N(M; mu + g.v, S + C)`, so with `A = inv(S + C)`:

        q_a = v_a . A . (M - mu)          r_ab = -(v_a . A . v_b)

    `r` is the same for every galaxy (a Gaussian's curvature does not depend
    on the data) and `q` is linear in M, so both are exact BFD Q, R without
    ever building a flow.  The +g/-g arms share the SAME z and the SAME noise
    draw n, which is what `bias.py` itself does and is what cancels the shape
    noise -- without it N would need to be far larger for either assertion to
    hold reliably.
    """
    MU = np.array([5000.0, 15000.0, -100.0, 80.0, 6000.0])
    V1 = np.array([200.0, 1500.0, 20.0, 10.0, 400.0])       # dM/dg1
    V2 = np.array([10.0, 30.0, 400.0, 10.0, 20.0])          # dM/dg2

    A_S = np.diag([800.0, 3000.0, 200.0, 200.0, 2000.0])
    A_S[1, 0] = 900.0                                        # S not (0,1)-diagonal
    S = A_S @ A_S.T
    A_C = np.diag([400.0, 1500.0, 150.0, 150.0, 1000.0])
    A_C[1, 0] = 300.0
    C = A_C @ A_C.T

    A = np.linalg.inv(S + C)
    V = np.stack([V1, V2])          # (2, 5)
    r_const = -(V @ A @ V.T)        # (2, 2), same for every galaxy

    g = 0.02
    n_gal = 300_000
    rng = np.random.default_rng(1)
    L_S = np.linalg.cholesky(S)
    z = MU + rng.standard_normal((n_gal, 5)) @ L_S.T
    L_C = np.linalg.cholesky(C)
    noise = rng.standard_normal((n_gal, 5)) @ L_C.T

    m_plus = z + g * V1 + noise
    m_minus = z - g * V1 + noise

    def qr(M):
        q = (M - MU) @ A.T @ V.T                     # (n, 2)
        r = np.broadcast_to(r_const, (len(M), 2, 2))
        return q, r

    qp, rp = qr(m_plus)
    qm, rm = qr(m_minus)

    # A wide flux-only window (no size cut: SIZE = (-inf, inf) exercises the
    # infinite-bound path too) -- wide enough to keep ~90% of the population,
    # which keeps the correction's own sampling noise well under the 0.01
    # tolerance, but still narrow enough that the tails it truncates carry
    # enough of the shear response to bias the uncorrected estimator past 0.05.
    size = (-np.inf, np.inf)
    flux = (3500.0, 6500.0)
    sp = window_mask(m_plus, size, flux)
    sm = window_mask(m_minus, size, flux)

    # P_s, Q_s, R_s are a property of the MODEL at g = 0, not of which arm is
    # being corrected, so one `selection_terms` call serves both arms; only
    # N_ns (each arm's own excluded count) differs between them.
    draw = lambda gg, zz: zz + gg[0] * jnp.asarray(V1) + gg[1] * jnp.asarray(V2)
    z_big = jnp.asarray(MU) + jax.random.normal(
        jax.random.key(101), (300_000, 5)) @ jnp.asarray(L_S.T)
    ps, qs, rs, qs_err = selection_terms(draw, z_big, C, size, flux)

    ns_p = (int((~sp).sum()), ps, qs, rs)
    ns_m = (int((~sm).sum()), ps, qs, rs)

    m1_uncorrected = bias(qp, rp, qm, rm, g, sel=(sp, sm))[0]
    m1_corrected = bias(qp, rp, qm, rm, g, sel=(sp, sm), ns=(ns_p, ns_m))[0]

    assert abs(m1_uncorrected) > 0.05, m1_uncorrected
    assert abs(m1_corrected) < 0.01, m1_corrected


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
