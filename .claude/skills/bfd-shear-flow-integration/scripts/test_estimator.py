"""Self-test for reference_estimator.py.

With an affine 'flow' x = A z + b0 + B g, the prior is Gaussian
p_theta(x|g) = N(b0 + B g, A A^T), so the target integral is closed form:

    P(g)  = N(M; b0 + B g, C),   C = A A^T + Sigma
    logP  = -d/2 log2pi - 1/2 log|C| - 1/2 (M-b)^T C^{-1} (M-b),  b = b0 + B g
    Q     = ∇_g P  = P * B^T C^{-1} (M - b)
    R     = ∇²_g P = P * ( -B^T C^{-1} B + (grad logP)(grad logP)^T )

We deliberately offset M from the prior mean so the integrand peak in z is NOT
at z=0 and the kernel sees real prior slope -- the regime the estimator targets.

Run:  python scripts/test_estimator.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import jax.numpy as jnp
from reference_estimator import (
    estimate_target, make_affine_forward, find_mode, psis_khat,
)

np.random.seed(0)
d = 4

# --- random but well-conditioned affine flow + kernel ---
A = np.eye(d) + 0.3 * np.random.randn(d, d)
prior_cov = A @ A.T
b0 = np.array([0.5, -0.2, 0.1, -0.3])
B = np.zeros((d, 2)); B[0, 0] = 0.7; B[1, 1] = 0.5; B[2, 0] = -0.2; B[3, 1] = 0.4
Sigma = 0.25 * (np.eye(d) + 0.1 * (np.ones((d, d)) - np.eye(d)))
Sigma = 0.5 * (Sigma + Sigma.T)

g0 = np.array([0.02, -0.015])
b = b0 + B @ g0
# offset the target several kernel-sigmas from the prior mean to create slope
M = b + np.array([0.8, -0.6, 0.4, 0.5])

C = prior_cov + Sigma
Cinv = np.linalg.inv(C)
sign, logdetC = np.linalg.slogdet(C)
diff = M - b
logP_true = -0.5 * d * np.log(2 * np.pi) - 0.5 * logdetC - 0.5 * diff @ Cinv @ diff
P_true = np.exp(logP_true)
grad_logP_true = B.T @ Cinv @ diff
Q_true = P_true * grad_logP_true
hess_logP_true = -B.T @ Cinv @ B
R_true = P_true * (hess_logP_true + np.outer(grad_logP_true, grad_logP_true))

forward = make_affine_forward(A, b0, B)

# analytic z-mode: (I + A^T Sinv A) z = A^T Sinv (M - b)
Sinv = np.linalg.inv(Sigma)
z_star_true = np.linalg.solve(np.eye(d) + A.T @ Sinv @ A, A.T @ Sinv @ (M - b))

print("=" * 64)
print("Affine-flow self-test (analytic ground truth)")
print("=" * 64)

res = estimate_target(forward, g0, M, Sigma, S=2048, nu=4.0, alpha=0.8,
                      inflate=1.3, antithetic=True, seed=12345)

def relerr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-300)

print(f"\nmode z*   est : {np.round(res['z_star'], 4)}")
print(f"mode z*   true: {np.round(z_star_true, 4)}   (relerr {relerr(res['z_star'], z_star_true):.2e})")

print(f"\nlog P  est={res['logP']:.6f}  true={logP_true:.6f}  diff={abs(res['logP']-logP_true):.2e}")
print(f"P      est={res['P']:.6e}  true={P_true:.6e}  relerr={abs(res['P']-P_true)/P_true:.2e}")
print(f"\nQ est : {np.round(res['Q'], 6)}")
print(f"Q true: {np.round(Q_true, 6)}   relerr={relerr(res['Q'], Q_true):.2e}")
print(f"\nR est :\n{np.round(res['R'], 6)}")
print(f"R true:\n{np.round(R_true, 6)}   relerr={relerr(res['R'], R_true):.2e}")
print(f"\nk-hat = {res['khat']:.3f}   ESS = {res['ess']:.0f} / 2048")

# --- assertions (realistic Monte Carlo tolerances) ---
# P and Q (1st-order) converge fast; R (2nd derivative) is the noisiest quantity.
ok = True
ok &= relerr(res['z_star'], z_star_true) < 1e-3      # mode is a deterministic solve
ok &= abs(res['P'] - P_true) / P_true < 1e-2
ok &= relerr(res['Q'], Q_true) < 1e-2
ok &= relerr(res['R'], R_true) < 2e-1                 # Hessian: loose, variance-dominated
ok &= res['khat'] < 0.5
print("\n" + ("PASS: estimator matches analytic P/Q/R and k-hat is healthy"
              if ok else "FAIL: see deviations above"))

# --- convergence: R error must shrink ~1/sqrt(S) (=> variance, not bias) ---
print("\n" + "-" * 64)
print("Convergence of R relerr with sample count (confirms unbiasedness):")
for S in (1024, 4096, 16384):
    r = estimate_target(forward, g0, M, Sigma, S=S, nu=4.0, alpha=0.8,
                        inflate=1.3, antithetic=True, seed=12345)
    print(f"  S={S:6d}:  P relerr={abs(r['P']-P_true)/P_true:.2e}   "
          f"R relerr={relerr(r['R'], R_true):.2e}   k-hat={r['khat']:.3f}")

# --- show the diagnostic catches a deliberately bad (too-narrow) proposal ---
print("\n" + "-" * 64)
print("Sanity: a too-narrow / mismatched proposal should raise k-hat")
bad = estimate_target(forward, g0, M, Sigma, S=2048, nu=50.0, alpha=1.0,
                      inflate=0.25, antithetic=False, seed=7)
print(f"narrow proposal: k-hat = {bad['khat']:.3f}  ESS = {bad['ess']:.0f}  "
      f"(P relerr = {abs(bad['P']-P_true)/P_true:.2e})")

sys.exit(0 if ok else 1)
