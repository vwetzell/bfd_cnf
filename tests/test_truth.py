"""Self-checks for truth.py -- exact P(m|g), Q and R for the gauss2 population.

Run: pytest -q tests/test_truth.py  (or python -m tests.test_truth)

Builds a small gauss2 population directly from `imsims.sim` /
`imsims.analytic` (moments only, no images) and checks:

  1. log_prob(m, 0) == log_p0(m) exactly (the two Jacobians in the formula
     coincide at g = 0).
  2. to_coords inverts sim._gauss2_m_of_t, the chart P0 is defined in.
  3. log_prob is a proper change of variables under a real shear, checked
     against an INDEPENDENT reconstruction (not truth.py's own call chain).
  4. pqr matches central finite differences of log_prob in g.
  5. The end-to-end BFD estimator recovers an injected shear from exact Q, R.
  6. The batched entry points match per-target loops.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, ".")
sys.path.insert(0, "../bfd_cnf_imsims")

import truth                                    # noqa: E402
from imsims import analytic, sim                # noqa: E402

N = 60
SEED = 0


def _population():
    """thetas and unsheared moments for a small gauss2 population."""
    pop = sim.sample_population_gauss2(N, np.random.default_rng(SEED))
    thetas = jax.vmap(analytic.pack)(
        jnp.asarray(pop["flux"]), jnp.asarray(pop["sigma"]),
        jnp.asarray(pop["bulge_ratio"]), jnp.asarray(pop["e1"]),
        jnp.asarray(pop["e2"]))
    m = jax.vmap(analytic.moments)(thetas)
    return thetas, np.asarray(m)


THETAS, M = _population()


def test_log_prob_at_zero_shear_is_log_p0():
    zero = jnp.zeros(2)
    for m_i in M:
        a = float(truth.log_prob(jnp.asarray(m_i), zero))
        b = float(truth.log_p0(jnp.asarray(m_i)))
        assert np.isclose(a, b, rtol=1e-10), (m_i, a, b)


def test_to_coords_matches_sim():
    """`to_coords` must invert `sim._gauss2_m_of_t`, NOT match
    `bulk.to_coords`.

    `log_p0` evaluates a Gaussian at `sim.GAUSS2_MU`/`GAUSS2_COV`, and those
    live in sim's chart, whose slot 2 is the raw ratio `Mc/Mr`.  This test used
    to assert agreement with `bulk.to_coords`, which uses
    `logit(Mc/(POINT_SOURCE_MC Mr))` there -- so it actively enforced the bug
    that made `truth.py` unusable as ground truth (HANDOFF.md, 2026-08-28).
    Round-tripping through sim's own inverse is the invariant that matters.
    """
    from imsims import sim as _sim

    t = np.asarray(jax.vmap(truth.to_coords)(jnp.asarray(M)))
    np.testing.assert_allclose(_sim._gauss2_m_of_t(t), M, rtol=1e-10)

    # and it must NOT equal bulk's, which is the chart P0 is not in
    import bulk
    assert not np.allclose(t, bulk.to_coords(M), rtol=1e-3)


def test_log_prob_is_a_proper_change_of_variables():
    g = jnp.array([0.03, -0.02])
    zero = jnp.zeros(2)

    for theta in THETAS:
        m_g = analytic.moments(theta, g)
        got = float(truth.log_prob(m_g, g))

        # Independent reconstruction: the KNOWN theta (not truth's Newton
        # solve), and the Jacobian ratio from a fresh jacfwd of
        # analytic.moments, not anything internal to truth.log_prob.
        m0 = analytic.moments(theta, zero)
        j0 = jax.jacfwd(lambda th: analytic.moments(th, zero))(theta)
        jg = jax.jacfwd(lambda th: analytic.moments(th, g))(theta)
        _, ld0 = jnp.linalg.slogdet(j0)
        _, ldg = jnp.linalg.slogdet(jg)
        want = float(truth.log_p0(m0) + ld0 - ldg)

        assert np.isclose(got, want, rtol=1e-10), (got, want)


def _fd_grad_hess(m, h=1e-3):
    """Central finite differences of truth.log_prob(m, .) in g."""
    m = jnp.asarray(m)

    def f(g):
        return float(truth.log_prob(m, jnp.asarray(g)))

    grad = np.zeros(2)
    for i in range(2):
        gp, gm = [0.0, 0.0], [0.0, 0.0]
        gp[i], gm[i] = h, -h
        grad[i] = (f(gp) - f(gm)) / (2 * h)

    f0 = f([0.0, 0.0])
    hess = np.zeros((2, 2))
    for i in range(2):
        gp, gm = [0.0, 0.0], [0.0, 0.0]
        gp[i], gm[i] = h, -h
        hess[i, i] = (f(gp) - 2 * f0 + f(gm)) / h ** 2
    gpp, gpm, gmp, gmm = [h, h], [h, -h], [-h, h], [-h, -h]
    hess[0, 1] = hess[1, 0] = (f(gpp) - f(gpm) - f(gmp) + f(gmm)) / (4 * h * h)
    return grad, hess


def test_pqr_matches_finite_differences():
    """A pure rtol=1e-5 check on Q is not achievable at h=1e-3: central
    differences carry an O(h^2) truncation floor of ~1-2e-5 in ABSOLUTE terms
    here (verified separately by re-running at h=3e-4/1e-4/3e-5 and watching
    the residual shrink as h^2, i.e. pure truncation, not a bug), and some
    targets' Q components are themselves O(0.5) -- below that floor -- so a
    relative-only check fails on exactly those.  The atol values below are
    sized to that measured floor (with margin), not loosened past it."""
    order = np.argsort(M[:, 1] / M[:, 0])            # spread over Mr/Mf
    idx = order[np.linspace(0, N - 1, 6).astype(int)]

    for i in idx:
        m_i = jnp.asarray(M[i])
        q, r = truth.pqr(m_i)
        q, r = np.asarray(q), np.asarray(r)
        q_fd, r_fd = _fd_grad_hess(m_i)

        np.testing.assert_allclose(q, q_fd, rtol=1e-5, atol=3e-5)
        np.testing.assert_allclose(r, r_fd, rtol=1e-3, atol=1e-4)


def test_pqr_recovers_the_input_shear():
    """The formalism end to end: exact Q, R must return the shear put in.

    PAIRED, as `bias.bias` does it -- the same galaxies lensed +g and -g, and
    half their difference.  A single arm does NOT work at any population size
    a test can afford: the ensemble ghat carries shape noise an order of
    magnitude larger than the signal (measured, at n=60: a g=0 control arm
    gives ghat = (-0.006, 0.026) against a g_true of 0.02, and the offset
    changes sign seed to seed).  Pairing cancels it exactly, because both arms
    are the SAME galaxies -- which is the whole reason bias.py pairs.

    Both shear directions are checked: a g2-only shear once tripped a solver
    failure that a g1-only test could not see.
    """
    def ghat(g):
        m_g = jax.vmap(lambda th: analytic.moments(th, jnp.asarray(g)))(THETAS)
        q, r = truth.pqr_batch(m_g)
        return -np.linalg.solve(np.asarray(r).sum(0), np.asarray(q).sum(0))

    # atol from measurement, not from taste: the paired residual is 1.5e-5 at
    # this n = 60 and 1.6e-6 at n = 400, i.e. it is still finite-sample and
    # shrinking, not a floor.  1e-4 covers seed-to-seed variation at n = 60
    # while staying ~100x below any bias worth caring about (m1 ~ 1e-2), so a
    # real break in the formalism still fails this loudly.
    for g_true in [(0.02, 0.0), (0.0, 0.02)]:
        plus, minus = ghat(g_true), ghat((-g_true[0], -g_true[1]))
        np.testing.assert_allclose((plus - minus) / 2.0, np.asarray(g_true),
                                   atol=1e-4)


def test_batch_matches_single():
    """rtol=1e-9, not 1e-12: N=60 is under `pqr_batch`'s default batch_size
    (512), so `lax.map` runs the whole population through ONE `vmap`, whose
    batched Cholesky/solve kernels sum in a different order than the
    sequential per-target loop -- ordinary float64 non-associativity, not a
    bug (worst observed 9.2e-12, on the two highest-curvature targets; 1e-9
    still holds the two to far tighter than float32 round-off while giving
    it room)."""
    g = jnp.array([0.01, -0.03])
    ms = jnp.asarray(M)
    gs = jnp.broadcast_to(g, ms.shape[:-1] + (2,))

    lp_batch = np.asarray(truth.log_prob_batch(ms, gs))
    lp_loop = np.array([float(truth.log_prob(ms[i], g)) for i in range(N)])
    np.testing.assert_allclose(lp_batch, lp_loop, rtol=1e-9)

    q_batch, r_batch = truth.pqr_batch(ms)
    q_batch, r_batch = np.asarray(q_batch), np.asarray(r_batch)
    q_loop, r_loop = [], []
    for i in range(N):
        q_i, r_i = truth.pqr(ms[i])
        q_loop.append(np.asarray(q_i))
        r_loop.append(np.asarray(r_i))
    np.testing.assert_allclose(q_batch, np.stack(q_loop), rtol=1e-9)
    np.testing.assert_allclose(r_batch, np.stack(r_loop), rtol=1e-9)


if __name__ == "__main__":
    test_log_prob_at_zero_shear_is_log_p0()
    test_to_coords_matches_sim()
    test_log_prob_is_a_proper_change_of_variables()
    test_pqr_matches_finite_differences()
    test_pqr_recovers_the_input_shear()
    test_batch_matches_single()
    print("OK: truth.py self-checks passed")
