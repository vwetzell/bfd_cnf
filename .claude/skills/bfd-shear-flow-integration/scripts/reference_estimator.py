"""Reference implementation of the z-space defensive-IS shear-recovery estimator.

This is a *verified scaffold*, not a drop-in. It depends only on jax + numpy +
scipy so Claude Code can run scripts/test_estimator.py and confirm the core math
before wiring it to a real FlowJax conditional flow.

What it computes, per target, for a conditional normalizing-flow prior
p_theta(x | g) with base z ~ N(0, I) and forward map x = T_g(z):

    P(g) = ∫ p_theta(x | g) N(x; M, Sigma) dx
         = E_{z ~ N(0,I)} [ N(T_g(z); M, Sigma) ]          (z-space identity, no Jacobian)

estimated by importance sampling in z with a defensive Student-t proposal q,

    P_hat = (1/S) Σ_s  φ(z_s) k(T_g(z_s)) / q(z_s),   z_s ~ q,

where φ = N(0,I), k = N(·; M, Sigma). Q = ∇_g P and R = ∇²_g P fall out of
autodiff of log P at FIXED proposal and FIXED z_s (common random numbers).

KEY INVARIANTS (violate these and you lose unbiasedness):
  * q must be a *normalized* density -> use standard weights w = φ k / q,
    NOT self-normalized weights. Self-normalization re-introduces the
    partition-function bias you removed at training time.
  * The proposal q and the sample points z_s do NOT depend on g. Build them once
    at a reference g (typically g = 0), then hold fixed while differentiating in g.
    This is what makes Q, R low-variance and mutually consistent.
  * The defensive component is the base φ(z) = N(0,I) itself: pushing those draws
    through the flow recovers ordinary prior sampling, so the mixture tail
    provably dominates the prior tail and bounds the weights.

Replace `affine_forward` with your flow's base->data forward map:
    forward_map = lambda z, g: flow.bijection.transform(z, condition=g)
(confirm direction with a round-trip check: inverse(transform(z)) ≈ z).
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, gammaln
from jax.scipy.linalg import solve_triangular
from scipy.stats import qmc, chi2, norm as sp_norm

jax.config.update("jax_enable_x64", True)  # moment integrals want float64

# --------------------------------------------------------------------------- #
# Densities (all log-space, all vmapped over the leading sample axis)
# --------------------------------------------------------------------------- #

def log_base(z):
    """log N(z; 0, I), batched over leading axis."""
    d = z.shape[-1]
    return -0.5 * d * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(z ** 2, axis=-1)


def _log_gauss_one(x, mu, chol):
    d = x.shape[0]
    y = solve_triangular(chol, x - mu, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol)))
    return -0.5 * d * jnp.log(2 * jnp.pi) - 0.5 * logdet - 0.5 * jnp.dot(y, y)


log_gauss = jax.vmap(_log_gauss_one, in_axes=(0, None, None))


def _log_mvt_one(z, mu, chol, nu):
    d = z.shape[0]
    y = solve_triangular(chol, z - mu, lower=True)
    maha = jnp.dot(y, y)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol)))
    return (
        gammaln((nu + d) / 2) - gammaln(nu / 2)
        - 0.5 * d * jnp.log(nu * jnp.pi) - 0.5 * logdet
        - 0.5 * (nu + d) * jnp.log1p(maha / nu)
    )


log_mvt = jax.vmap(_log_mvt_one, in_axes=(0, None, None, None))


def proposal_logpdf(z, mu, chol, nu, alpha):
    """log of defensive mixture q = alpha * t_nu(mu, ΣΣ) + (1-alpha) * N(0, I)."""
    lt = log_mvt(z, mu, chol, nu)
    lb = log_base(z)
    return jnp.logaddexp(jnp.log(alpha) + lt, jnp.log1p(-alpha) + lb)


# --------------------------------------------------------------------------- #
# Mode finding in z-space (Adam ascent on the log integrand)
# --------------------------------------------------------------------------- #

def find_mode(forward_map, g, M, Sigma_inv, z0, n_steps=400, lr=5e-2,
              b1=0.9, b2=0.999, eps=1e-8, clip=10.0):
    """Maximize  log φ(z) + log k(T_g(z))  via Adam. Returns z*.

    Gradient ascent in z; the integrand's mode is generally NOT at z=0 because
    the prior slope pushes it off the measured moments (peak != M)."""
    Sinv = jnp.asarray(Sigma_inv)

    def neg_logpost(z):
        diff = forward_map(z, g) - M
        return 0.5 * jnp.dot(z, z) + 0.5 * diff @ Sinv @ diff

    grad = jax.grad(neg_logpost)

    def step(carry, _):
        z, m, v, t = carry
        gr = grad(z)
        gr = jnp.clip(gr, -clip, clip)
        t = t + 1
        m = b1 * m + (1 - b1) * gr
        v = b2 * v + (1 - b2) * gr ** 2
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        z = z - lr * mhat / (jnp.sqrt(vhat) + eps)   # descent on neg_logpost
        return (z, m, v, t), neg_logpost(z)

    init = (z0, jnp.zeros_like(z0), jnp.zeros_like(z0), 0)
    (z_star, *_), traj = jax.lax.scan(step, init, None, length=n_steps)
    return z_star, traj


def laplace_chol(forward_map, g, M, Sigma_inv, z_star, inflate=1.0, jitter=1e-6):
    """Cholesky of the Laplace covariance Σ_prop = inflate * H^{-1} at z*."""
    Sinv = jnp.asarray(Sigma_inv)

    def neg_logpost(z):
        diff = forward_map(z, g) - M
        return 0.5 * jnp.dot(z, z) + 0.5 * diff @ Sinv @ diff

    H = jax.hessian(neg_logpost)(z_star)
    d = z_star.shape[0]
    H = 0.5 * (H + H.T) + jitter * jnp.eye(d)        # symmetrize + regularize
    cov = inflate * jnp.linalg.inv(H)
    cov = 0.5 * (cov + cov.T) + jitter * jnp.eye(d)
    return jnp.linalg.cholesky(cov)


# --------------------------------------------------------------------------- #
# RQMC sampling of the defensive mixture (deterministic component split)
# --------------------------------------------------------------------------- #

def sample_mixture_rqmc(key_int, z_star, chol, nu, alpha, S, antithetic=True):
    """Draw S points from q via scrambled Sobol. Returns z (S, d) as a numpy array.

    Deterministic split: S_t = round(alpha*S) from the t-component, the rest from
    the base N(0,I). Every point's *mixture* density is used in the denominator
    (mixture sampling => unbiased). Antithetic reflection of the Gaussian seed
    about z* cancels the leading first-order slope variance.
    """
    z_star = np.asarray(z_star, float)
    chol = np.asarray(chol, float)
    d = z_star.shape[0]
    S_t = int(round(alpha * S))
    S_b = S - S_t

    def sobol(dim, n, seed):
        # request a power-of-2 block (Sobol balance) then slice to n
        m = max(1, int(np.ceil(np.log2(max(n, 2)))))
        pts = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(m)[:n]
        return np.clip(pts, 1e-10, 1 - 1e-10)

    # --- t-component: d Gaussian dims + 1 chi-square mixing dim ---
    if antithetic:
        n_seed = (S_t + 1) // 2
        u_t = sobol(d + 1, n_seed, key_int)
        zeta = sp_norm.ppf(u_t[:, :d])
        zeta = np.vstack([zeta, -zeta])[:S_t]          # reflect about z*
        chi_u = np.concatenate([u_t[:, d], u_t[:, d]])[:S_t]
        w_chi = chi2.ppf(chi_u, df=nu)
    else:
        u_t = sobol(d + 1, S_t, key_int)
        zeta = sp_norm.ppf(u_t[:, :d])
        w_chi = chi2.ppf(u_t[:, d], df=nu)

    scale = np.sqrt(nu / w_chi)[:, None]
    z_t = z_star[None, :] + scale * (zeta @ chol.T)

    # --- base component: d Gaussian dims ---
    if S_b > 0:
        z_b = sp_norm.ppf(sobol(d, S_b, key_int + 1))
        z = np.vstack([z_t, z_b])
    else:
        z = z_t
    return z


# --------------------------------------------------------------------------- #
# Estimator: log P(g), and Q, R by autodiff in g at fixed proposal / samples
# --------------------------------------------------------------------------- #

def make_logP(forward_map, z_samples, log_phi, log_q, M, Sigma_chol):
    """Return logP(g): a differentiable scalar fn of g, with everything else fixed."""
    z_samples = jnp.asarray(z_samples)
    log_phi = jnp.asarray(log_phi)
    log_q = jnp.asarray(log_q)
    S = z_samples.shape[0]

    def logP(g):
        x = jax.vmap(lambda z: forward_map(z, g))(z_samples)
        lk = log_gauss(x, M, Sigma_chol)
        lw = log_phi + lk - log_q              # log importance weights
        return logsumexp(lw) - jnp.log(S)

    return logP


def estimate_target(forward_map, g0, M, Sigma, S=1024, nu=4.0, alpha=0.8,
                    inflate=1.2, antithetic=True, seed=0, z_init=None):
    """Full per-target pipeline at conditioning g0. Returns a results dict with
    P, Q (grad), R (hessian), k-hat, and ESS."""
    M = jnp.asarray(M, float)
    Sigma = jnp.asarray(Sigma, float)
    d = M.shape[0]
    Sigma_chol = jnp.linalg.cholesky(Sigma)
    Sigma_inv = jnp.linalg.inv(Sigma)
    g0 = jnp.asarray(g0, float)
    if z_init is None:
        z_init = jnp.zeros(d)

    z_star, _ = find_mode(forward_map, g0, M, Sigma_inv, z_init)
    chol = laplace_chol(forward_map, g0, M, Sigma_inv, z_star, inflate=inflate)

    z = sample_mixture_rqmc(seed, z_star, chol, nu, alpha, S, antithetic)
    z = jnp.asarray(z)
    log_phi = log_base(z)
    log_q = proposal_logpdf(z, z_star, chol, nu, alpha)

    logP_fn = make_logP(forward_map, z, log_phi, log_q, M, Sigma_chol)

    logP, Q = jax.value_and_grad(logP_fn)(g0)        # ∇_g log P
    H_log = jax.hessian(logP_fn)(g0)                 # ∇²_g log P
    P = jnp.exp(logP)
    Q_P = P * Q                                      # ∇_g P
    R_P = P * (H_log + jnp.outer(Q, Q))             # ∇²_g P

    # diagnostics from the weights at g0
    x = jax.vmap(lambda zz: forward_map(zz, g0))(z)
    lw = log_phi + log_gauss(x, M, Sigma_chol) - log_q
    lw_np = np.asarray(lw)
    khat = psis_khat(lw_np)
    w = np.exp(lw_np - lw_np.max())
    ess = (w.sum() ** 2) / np.sum(w ** 2)

    return dict(logP=float(logP), P=float(P),
                Q=np.asarray(Q_P), R=np.asarray(R_P),
                grad_logP=np.asarray(Q), hess_logP=np.asarray(H_log),
                khat=float(khat), ess=float(ess), z_star=np.asarray(z_star))


# --------------------------------------------------------------------------- #
# PSIS k-hat (Zhang & Stephens GPD fit; mirrors the standard PSIS diagnostic)
# --------------------------------------------------------------------------- #

def _gpd_fit(x):
    """Generalized-Pareto shape/scale via the Zhang-Stephens estimator.
    x: 1-D ascending positive exceedances."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    m = 30 + int(np.sqrt(n))
    b = 1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b /= 3 * x[int(n / 4 + 0.5) - 1]
    b += 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    ll = n * (np.log(-b / k) - k - 1)
    w = 1 / np.exp(ll - ll[:, None]).sum(axis=1)
    w /= w.sum()
    b_post = np.sum(b * w)
    k_post = np.log1p(-b_post * x).mean()
    k_post = (n * k_post + 10 * 0.5) / (n + 10)   # weakly-informative prior on k
    return k_post


def psis_khat(log_weights):
    """Pareto-k diagnostic for a set of log importance weights.
    k < 0.5 reliable; 0.5-0.7 marginal; > 0.7 the proposal is inadequate."""
    lw = np.asarray(log_weights, float)
    S = lw.size
    M = int(min(0.2 * S, 3 * np.sqrt(S)))
    if M < 5:
        return np.nan
    order = np.argsort(lw)
    tail = lw[order[-M:]]
    cutoff = lw[order[-M - 1]]
    exceed = np.exp(tail) - np.exp(cutoff)
    exceed = exceed[exceed > 0]
    if exceed.size < 5:
        return np.nan
    return _gpd_fit(exceed)


# --------------------------------------------------------------------------- #
# Affine "flow" stub for testing only (x = A z + b0 + B g). Replace in real use.
# --------------------------------------------------------------------------- #

def make_affine_forward(A, b0, B):
    A = jnp.asarray(A); b0 = jnp.asarray(b0); B = jnp.asarray(B)
    def forward(z, g):
        return A @ z + b0 + B @ g
    return forward
