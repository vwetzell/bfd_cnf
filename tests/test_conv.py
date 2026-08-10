"""The noise integral, against a case where it can be done in closed form.
`python -m tests.test_conv`.

`bias.log_conv` estimates the paper's eq. (38) convolution

    P(M|g) = INT dm P(m|g) L(M - m),    L = N(., C_M)

by Monte Carlo over L.  Substitute a Gaussian prior for the flow and the
convolution is another Gaussian, with C_M simply added to its covariance -- so
P, Q = dP/dg and R = d2P/dg2 all have exact values to check the estimator, and
its derivatives, against.  Anything that gets the kernel width, the sign of the
offset or the normalisation wrong fails here.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from bias import kernel_draws, log_conv  # noqa: E402  (after the x64 flag)

# A prior N(mu(g), S0) in raw moment space, with mu linear in g.  mu0 sits far
# from the Mf, Mr > 0 edges so log_conv's domain mask never fires and the
# comparison is against the unmasked integral.
MU0 = np.array([6.0e3, 2.2e4, -1.5e2, 1.2e2, 1.5e5])
DMU = np.array([[30.0, 400.0, 4.4e3, 10.0, 2.0e3],       # d mu / d g1
                [-20.0, 250.0, 15.0, 4.1e3, -1.5e3]])    # d mu / d g2
_A = np.diag([80.0, 340.0, 240.0, 240.0, 2900.0])
_A[1, 0], _A[4, 1], _A[3, 2] = 90.0, 700.0, -40.0
S0 = _A @ _A.T                       # prior covariance
# The noise kernel, narrow next to the prior as a real C_M is (the imsims
# targets sit at S/N ~ 50).  Wider kernels give the estimator a heavy-tailed
# weight distribution and the tolerances below would have to grow like 1/ESS.
COV = 0.02 * S0 + np.diag(np.diag(S0)) * 0.01


class GaussPrior:
    """Stand-in for the flow: P(m|g) = N(m; MU0 + DMU.g, S0)."""

    def log_prob(self, x, condition):
        d = x - (MU0 + condition @ DMU)
        return (-0.5 * jnp.einsum("...i,ij,...j->...", d, jnp.linalg.inv(S0), d)
                - 0.5 * jnp.linalg.slogdet(2 * jnp.pi * S0)[1])


def exact(M, g):
    """log N(M; MU0 + DMU.g, S0 + COV) -- the convolution, done analytically."""
    d = M - (MU0 + g @ DMU)
    S = S0 + COV
    return (-0.5 * d @ np.linalg.solve(S, d)
            - 0.5 * np.linalg.slogdet(2 * np.pi * S)[1])


def test_convolution_and_its_shear_derivatives():
    M = MU0 + np.array([2.0e2, -8.0e2, 3.0e2, -2.0e2, 5.0e3])   # a target
    eps = kernel_draws(COV, 1, 400_000, seed=7)[0]
    f = lambda g: log_conv(GaussPrior(), jnp.asarray(M), jnp.asarray(eps), g)

    zero = jnp.zeros(2)
    logP = float(f(zero))
    q = np.asarray(jax.grad(f)(zero))
    r = np.asarray(jax.hessian(f)(zero))

    # Exact values.  R is constant for a Gaussian, so it is the sharpest test.
    A = np.linalg.inv(S0 + COV)
    q_true = DMU @ A @ (M - MU0)
    r_true = -DMU @ A @ DMU.T

    # Tolerances are ~2x the Monte Carlo scatter measured over seeds at this S;
    # every structural way of getting the convolution wrong misses by far more.
    assert abs(logP - exact(M, np.zeros(2))) < 6e-3, (logP, exact(M, np.zeros(2)))
    assert np.max(np.abs(q - q_true) / np.abs(q_true)) < 3e-3, (q, q_true)
    assert np.max(np.abs(r - r_true) / np.abs(r_true)) < 4e-3, (r, r_true)

    # ...and at a shear well away from zero, where mu(g) has moved.
    g = jnp.array([0.05, -0.03])
    assert abs(float(f(g)) - exact(M, np.asarray(g))) < 6e-3


def test_zero_kernel_is_the_point_evaluation():
    """C_M -> 0 must reduce to evaluating the prior at the target itself."""
    M = MU0 + np.array([1.0e2, 5.0e2, -2.0e2, 1.0e2, -3.0e3])
    eps = kernel_draws(1e-12 * COV, 1, 64, seed=3)[0]
    prior = GaussPrior()
    got = log_conv(prior, jnp.asarray(M), jnp.asarray(eps), jnp.zeros(2))
    want = prior.log_prob(jnp.asarray(M), jnp.zeros(2))
    assert abs(float(got) - float(want)) < 1e-8


def test_out_of_domain_draws_get_zero_weight_not_nan():
    """Draws at Mf <= 0 or Mr <= 0 are off the prior's chart; they must drop out
    of the sum without poisoning the value or the gradient."""
    M = np.array([50.0, 40.0, 5.0, -5.0, 1.0e3])          # right at the edge
    eps = kernel_draws(COV, 1, 4096, seed=11)[0]
    x = M + np.asarray(eps)
    assert ((x[:, 0] <= 0) | (x[:, 1] <= 0)).mean() > 0.2, "test point too safe"

    f = lambda g: log_conv(GaussPrior(), jnp.asarray(M), jnp.asarray(eps), g)
    assert np.isfinite(float(f(jnp.zeros(2))))
    assert np.all(np.isfinite(np.asarray(jax.grad(f)(jnp.zeros(2)))))
    assert np.all(np.isfinite(np.asarray(jax.hessian(f)(jnp.zeros(2)))))

    # And the value is the masked sum: the same draws, restricted by hand.
    ok = (x[:, 0] > 0) & (x[:, 1] > 0)
    lp = np.asarray(GaussPrior().log_prob(jnp.asarray(x[ok]), jnp.zeros(2)))
    want = jax.scipy.special.logsumexp(lp) - np.log(len(x))
    assert abs(float(f(jnp.zeros(2))) - float(want)) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
