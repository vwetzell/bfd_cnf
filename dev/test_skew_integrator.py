"""Does integrand SKEWNESS bias the production PQR estimator?

Feeds a synthetic prior with a KNOWN, dial-able skewness through the production
integrator (rqmc_integrate_pqr_jax: x-space mode-find + symmetric Gaussian Laplace
proposal + hard +/-5.6 sigma Halton clip) and compares P/Q/R against an independent
brute-force ground truth (kernel-sampling MC: no proposal, no clip => unbiased).

Synthetic prior (unnormalised; both estimators use the SAME p so normalisation is
irrelevant):
    mu(g)      = b0 + B g
    log p(x|g) = -0.5 (x-mu)^T Pinv (x-mu)  +  lam * sum_i ((x-mu)_i / s_i)^3
The cubic term is the skew; lam is the knob; shear shifts the skew CENTRE mu(g),
so the shear response itself becomes skew-dependent -- the hypothesised mechanism.

lam=0 is exactly Gaussian (estimator must be exact). If production drifts from the
brute-force truth as lam grows, skewness biases the integrator.

Run:  PYTHONPATH=. JAX_PLATFORMS=cuda python dev/test_skew_integrator.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from bfd_cnf.inference import rqmc_integrate_pqr_jax

D = 4
rng = np.random.default_rng(0)

# Prior shape (precision Pinv), shear loading B, skew per-dim scales s.
A = np.eye(D) + 0.3 * rng.standard_normal((D, D))
prior_cov = (A @ A.T).astype(np.float32)
Pinv = jnp.asarray(np.linalg.inv(prior_cov))
b0 = jnp.asarray([0.5, -0.2, 0.1, -0.3], dtype=jnp.float32)
B = np.zeros((D, 2), np.float32)
B[0, 0] = 0.7; B[1, 1] = 0.5; B[2, 0] = -0.2; B[3, 1] = 0.4
B = jnp.asarray(B)
s = jnp.asarray(np.sqrt(np.diag(prior_cov)), dtype=jnp.float32)  # skew scales


def make_logp(lam: float):
    lam = jnp.float32(lam)

    def logp(x, g1, g2):  # x: (D,), scalars g
        mu = b0 + B @ jnp.array([g1, g2])
        d = x - mu
        quad = -0.5 * d @ (Pinv @ d)
        skew = lam * jnp.sum((d / s) ** 3)
        return quad + skew

    return logp


def make_flow_fns(lam: float):
    """Return (log_flow_fn, flow_prob_and_derivs_fn) in the production interface."""
    logp = make_logp(lam)
    dg1 = jax.grad(logp, argnums=1)
    dg2 = jax.grad(logp, argnums=2)
    dg11 = jax.grad(dg1, argnums=1)
    dg22 = jax.grad(dg2, argnums=2)
    dg12 = jax.grad(dg1, argnums=2)

    def log_flow_fn(x):  # x: (1, D) -> (1,)
        return logp(x[0], 0.0, 0.0)[None]

    def flow_prob_and_derivs_fn(xb):  # xb: (N, D) -> 6 x (N,)
        def one(xi):
            return (logp(xi, 0.0, 0.0), dg1(xi, 0.0, 0.0), dg2(xi, 0.0, 0.0),
                    dg11(xi, 0.0, 0.0), dg22(xi, 0.0, 0.0), dg12(xi, 0.0, 0.0))
        return jax.vmap(one)(xb)

    return log_flow_fn, flow_prob_and_derivs_fn


def ground_truth(lam, M, Sigma, n=4_000_000, seed=1):
    """Unbiased kernel-sampling MC: x~N(M,Sigma), P=E[p], Q=E[p*dlogp], R=E[p*(d2+dd)]."""
    logp = make_logp(lam)
    dg1 = jax.grad(logp, argnums=1); dg2 = jax.grad(logp, argnums=2)
    dg11 = jax.grad(dg1, argnums=1); dg22 = jax.grad(dg2, argnums=2)
    dg12 = jax.grad(dg1, argnums=2)
    L = jnp.linalg.cholesky(Sigma)
    z = jr.normal(jr.PRNGKey(seed), (n, D))
    x = M + z @ L.T

    def terms(xi):
        lp = logp(xi, 0.0, 0.0)
        p = jnp.exp(lp)
        q1, q2 = dg1(xi, 0.0, 0.0), dg2(xi, 0.0, 0.0)
        r11 = dg11(xi, 0.0, 0.0) + q1 * q1
        r22 = dg22(xi, 0.0, 0.0) + q2 * q2
        r12 = dg12(xi, 0.0, 0.0) + q1 * q2
        return jnp.array([p, p * q1, p * q2, p * r11, p * r22, p * r12])

    # chunk to bound memory
    out = jnp.zeros(6)
    cs = 200_000
    for i in range(0, n, cs):
        out = out + jnp.sum(jax.vmap(terms)(x[i:i + cs]), axis=0)
    P, Q1, Q2, R11, R22, R12 = out / n
    return dict(P=float(P), Q=np.array([Q1, Q2]),
                R=np.array([[R11, R12], [R12, R22]]))


def g_from_QR(Q, R):
    return np.linalg.solve(np.asarray(R), np.asarray(Q))


def run(M, Sigma, lam, n_points, hessian_scale, seed=12345):
    lf, fpd = make_flow_fns(lam)
    res = rqmc_integrate_pqr_jax(
        lf, fpd, jnp.asarray(M), jnp.asarray(Sigma), jr.PRNGKey(seed),
        n_points=n_points, n_replicates=16,
        hessian_scale=hessian_scale, adapt_proposal=True,
    )
    P, _, Q1, _, Q2, _, R11, _, R22, _, R12, _, _, _ = res
    return dict(P=float(P), Q=np.array([float(Q1), float(Q2)]),
                R=np.array([[float(R11), float(R12)], [float(R12), float(R22)]]))


def main():
    # Two regimes mirroring the real flux dependence: broad (faint) vs narrow (bright) kernel.
    base = 0.25 * (np.eye(D) + 0.1 * (np.ones((D, D)) - np.eye(D)))
    base = 0.5 * (base + base.T)
    regimes = {
        "broad/faint  Sigma=1.0x": jnp.asarray((1.0 * base).astype(np.float32)),
        "narrow/bright Sigma=0.15x": jnp.asarray((0.15 * base).astype(np.float32)),
    }
    g0 = np.array([0.02, -0.015], np.float32)
    mu0 = np.asarray(b0) + np.asarray(B) @ g0

    for rname, Sigma in regimes.items():
        # Target offset a few kernel-sigmas off the prior mean => real slope.
        M = jnp.asarray((mu0 + np.array([0.8, -0.6, 0.4, 0.5], np.float32)
                         * np.sqrt(np.diag(np.asarray(Sigma)))).astype(np.float32))
        print("=" * 78)
        print(f"REGIME: {rname}")
        print("=" * 78)
        print(f"{'lam':>5} {'np':>6} {'hs':>3} | {'P relerr':>9} {'Q relerr':>9} "
              f"{'R relerr':>9} | {'g_true':>16} {'g_prod':>16} {'|dg|':>8}")
        for lam in (0.0, 0.05, 0.10, 0.20):
            gt = ground_truth(lam, M, Sigma)
            g_t = g_from_QR(gt["Q"], gt["R"])
            for n_points in (1024, 4096):
                for hs in (3.0, 5.0):
                    pr = run(M, Sigma, lam, n_points, hs)
                    g_p = g_from_QR(pr["Q"], pr["R"])
                    pe = abs(pr["P"] - gt["P"]) / abs(gt["P"])
                    qe = np.max(np.abs(pr["Q"] - gt["Q"])) / (np.max(np.abs(gt["Q"])) + 1e-30)
                    re = np.max(np.abs(pr["R"] - gt["R"])) / (np.max(np.abs(gt["R"])) + 1e-30)
                    dg = np.linalg.norm(g_p - g_t)
                    print(f"{lam:>5.2f} {n_points:>6} {hs:>3.0f} | {pe:>9.2e} {qe:>9.2e} "
                          f"{re:>9.2e} | [{g_t[0]:+.4f},{g_t[1]:+.4f}] "
                          f"[{g_p[0]:+.4f},{g_p[1]:+.4f}] {dg:>8.2e}")
        print()


if __name__ == "__main__":
    main()
