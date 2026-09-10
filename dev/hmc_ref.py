"""A converged Q, R reference: sample the posterior instead of importance-
weighting it.

`bias.py` estimates Q and R by importance sampling, and `dev/gauge.py` showed
that answer swings by 1.28 in m1 on the PROPOSAL alone.  Both arms are unbiased,
so neither is converged and there is nothing to check against.

Drop the proposal.  Q and R are both expectations under ONE measure -- the
target's posterior over its latent moment,

    pi(m) = L(M - m) p(m | g=0) / P(M | 0)

    Q = E_pi[d_g log p]      R = E_pi[d2_g log p] + Var_pi[d_g log p]      [P]
    Q = E_pi[d_g log L]      R = E_pi[d2_g log L] + Var_pi[d_g log L]      [K]

with the gauge-K derivatives taken along `Psi_g` (see `dev/gauge.py`).  Both
lines are exact at g = 0, where the two measures coincide, so ONE set of pi
samples gives both -- and **they must agree**.  That agreement is the
convergence certificate no estimator in this repo has ever had.

Sampling pi is easy in the flow's BASE coordinates.  With m = T(z) the flow's
generative map, p(m)dm = N(z; 0, I)dz, so

    pi(z) ~ N(z; 0, I) * L(M - T(z))

-- a standard normal times a Gaussian likelihood.  No importance weights, no
ESS, no log-dets, and no domain mask: every z maps to an in-domain moment by
construction, which is exactly what the bounded chart buys.

Targets are independent, so all of them are sampled as one block-diagonal NUTS
target; the mass matrix adapts per coordinate.

Also reports the Hill tail index of the gauge-P score under pi.  If it is below
2 then `Var_pi[d_g log p]` does not exist, gauge P's R is not merely noisy but
UNESTIMABLE at any draw count, and gauge K is mandatory rather than preferable.
"""
import os
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
import numpyro, numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

from bias import condition
from dev.gauge import prep, make_psi, D

NT_POOL = int(os.environ.get("NT_POOL", 2000))   # targets to draw the faint cut from
NREF = int(os.environ.get("NREF", 200))          # targets carried into the reference
WARMUP = int(os.environ.get("WARMUP", 1000))
DRAWS = int(os.environ.get("DRAWS", 4000))
CHAINS = int(os.environ.get("CHAINS", 2))
IS_CHUNK = int(os.environ.get("IS_CHUNK", 4096))


def hill(x, frac=0.05):
    """Tail index of |x|: alpha < 2 means the VARIANCE of x does not exist."""
    x = np.sort(np.abs(x[np.isfinite(x)]))[::-1]
    k = max(10, int(frac * len(x)))
    if len(x) <= k + 1 or x[k] <= 0:
        return np.nan
    return 1.0 / np.mean(np.log(x[:k] / x[k]))


def sample_posterior(flow, M, sx, cov, seed=0):
    """NUTS on pi(z) ~ N(z;0,I) L(M - T(z)), all targets in one block target."""
    n = len(M)
    Cinv = jnp.asarray(np.linalg.inv(cov), jnp.float32)
    Mj = jnp.asarray(M, jnp.float32)
    sxj = jnp.asarray(sx, jnp.float32)
    zero = jnp.zeros(2)
    # base -> data at g = 0.  `flow.bijection` is `Invert(Chain(...))`, so its
    # `transform` runs the generative direction.
    gen = lambda z, s: flow.bijection.transform(z, condition(zero, s))

    def model():
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n, 5]).to_event(2))
        d = Mj - jax.vmap(gen)(z, sxj)
        numpyro.factor("lik", -0.5 * jnp.einsum("ni,ij,nj->", d, Cinv, d))

    mcmc = MCMC(NUTS(model, target_accept_prob=0.85), num_warmup=WARMUP,
                num_samples=DRAWS, num_chains=CHAINS, chain_method="vectorized",
                progress_bar=False)
    mcmc.run(jax.random.key(seed))
    z = mcmc.get_samples()["z"]                       # (chains*draws, n, 5)
    ex = summary(mcmc.get_samples(group_by_chain=True))["z"]
    return z, np.asarray(ex["n_eff"]).min(-1), np.asarray(ex["r_hat"]).max(-1)


def per_target(flow, psi, cov, z, M, sx):
    Cinv = jnp.asarray(np.linalg.inv(cov), jnp.float32)
    zero = jnp.zeros(2)
    logL = lambda off: -0.5 * jnp.einsum("...i,ij,...j->...", off, Cinv, off)
    gen = lambda zz, s: flow.bijection.transform(zz, condition(zero, s))

    @eqx.filter_jit
    def one(z_i, M_i, s_i):
        m = jax.vmap(gen, in_axes=(0, None))(z_i, s_i)          # (S, 5) pi draws
        fp = lambda x: (lambda g: flow.log_prob(x, condition=condition(g, s_i)))
        fk = lambda x: (lambda g: logL(M_i - psi(x, g, s_i)))

        def pair(x):
            return (jax.grad(fp(x))(zero), jax.hessian(fp(x))(zero),
                    jax.grad(fk(x))(zero), jax.hessian(fk(x))(zero))

        gP, hP, gK, hK = jax.vmap(pair)(m)
        # `Psi`'s Jacobian runs through `_chart_spin0_jac`'s 1/(1-u), which
        # diverges as a draw approaches the point-source ceiling.  Under pi
        # these are REAL posterior draws, not the zero-weight ones gauge.py
        # could discard, so the dropped fraction is reported, not assumed.
        fin = (jnp.isfinite(gK).all(-1)
               & jnp.isfinite(hK).reshape(len(gK), -1).all(-1))
        # Zero the VALUES too: a zero weight times an inf is still a NaN.
        gK = jnp.where(fin[:, None], gK, 0.0)
        hK = jnp.where(fin[:, None, None], hK, 0.0)
        w = fin / jnp.maximum(fin.sum(), 1)
        mean = lambda x: jnp.tensordot(w, x, axes=1)
        cov = lambda x, mu: mean((x - mu)[:, :, None] * (x - mu)[:, None, :])
        sP = (gP.mean(0), hP.mean(0), jnp.cov(gP.T))
        muK = mean(gK)
        sK = (muK, mean(hK), cov(gK, muK))
        return sP, sK, gP[:, 0], 1.0 - fin.mean()

    out = []
    for i in range(0, len(M)):
        out.append(one(jnp.asarray(z[:, i], jnp.float32),
                       jnp.asarray(M[i], jnp.float32),
                       jnp.asarray(sx[i], jnp.float32)))
    tail = np.array([hill(np.asarray(o[2])) for o in out])
    drop = np.array([float(o[3]) for o in out])
    pack = lambda k: [np.stack([np.asarray(o[k][j], np.float64) for o in out])
                      for j in range(3)]
    return pack(0), pack(1), tail, drop


def show(name, Q, A, B):
    """Means AND medians: the gauge-P mean is dominated by a handful of draws,
    which is the whole point, so a mean alone would hide the comparison."""
    R = A + B
    q1, a, b, r = Q[:, 0], A[:, 0, 0], B[:, 0, 0], R[:, 0, 0]
    print(f"{name:>30s}{a.mean():12.3e}{b.mean():12.3e}{r.mean():12.3e}"
          f"{np.median(r):12.3e}{(q1**2).sum()/(-r.sum()):10.3f}{np.median(q1):12.3e}")


if __name__ == "__main__":
    import sys
    pop = sys.argv[1] if len(sys.argv) > 1 else "bulgedisc_deep_v2"
    fp = sys.argv[2] if len(sys.argv) > 2 else "flows/centroid_bulgedisc_v2.eqx"
    gain = float(sys.argv[3]) if len(sys.argv) > 3 else 1.4043

    import dev.gauge as G
    G.NT = NT_POOL
    flow, m, sx, cov = prep(pop, fp, gain)
    psi = make_psi(flow)
    # The faint quintile is where alpha moved m1 by 1.28; take the reference
    # targets from there.
    faint = np.argsort(m[:, 0])[:NREF]
    m, sx = m[faint], sx[faint]
    print(f"{NREF} faintest of {NT_POOL}: Mf {m[:,0].min():.0f}-{m[:,0].max():.0f}",
          flush=True)

    cached = "dev/hmc_ref.npz" if os.environ.get("REUSE") else None
    if cached and os.path.exists(cached):
        d = np.load(cached)
        z, n_eff, r_hat = d["z"], d["n_eff"], d["r_hat"]
        print("reusing chains from " + cached, flush=True)
    else:
        z, n_eff, r_hat = sample_posterior(flow, m, sx, cov)
    print(f"NUTS: {z.shape[0]} draws x {len(m)} targets; "
          f"min n_eff {n_eff.min():.0f} (median {np.median(n_eff):.0f}), "
          f"max r_hat {r_hat.max():.3f}", flush=True)

    (QP, AP, BP), (QK, AK, BK), tail, drop = per_target(flow, psi, cov, z, m, sx)
    print(f"gauge-K pi draws dropped for a non-finite dPsi/dg: "
          f"median {np.median(drop):.2e}, max {drop.max():.2e}", flush=True)
    print(f"\n{'':>30s}{'mean A':>12s}{'mean B':>12s}{'mean R':>12s}"
          f"{'med R':>12s}{'Fisher R':>10s}{'med Q1':>12s}")
    show("HMC reference, gauge P", QP, AP, BP)
    show("HMC reference, gauge K", QK, AK, BK)
    dr = np.abs((AK + BK)[:, 0, 0] / (AP + BP)[:, 0, 0] - 1)
    dq = np.abs(QK[:, 0] / QP[:, 0] - 1)
    print(f"\ngauge agreement under pi (the convergence certificate):"
          f"  median |dR/R| = {np.median(dr):.3e},  median |dQ/Q| = {np.median(dq):.3e}")
    print(f"Hill tail index of gauge-P score under pi: median {np.nanmedian(tail):.2f}, "
          f"min {np.nanmin(tail):.2f}  (< 2 => Var_pi[d_g log p] does not exist)")
    np.savez("dev/hmc_ref.npz", m=m, sx=sx, QP=QP, AP=AP, BP=BP,
             QK=QK, AK=AK, BK=BK, tail=tail, drop=drop,
             n_eff=n_eff, r_hat=r_hat, z=np.asarray(z, np.float32))
    # What bias.py actually consumes, on these same targets, against the
    # reference: gauge P + a mixture proposal, at both alphas.
    Rref, Qref = (AK + BK)[:, 0, 0], QK[:, 0]
    G.CHUNK, G.TBATCH = IS_CHUNK, 8
    for a in (0.5, 1.0):
        G.ALPHA = a
        (Qi, Ai, Bi), _, _, _ = G.decompose(flow, psi, m, sx, cov)
        show(f"IS gauge P, alpha={a}", Qi, Ai, Bi)
        ri = (Ai + Bi)[:, 0, 0]
        print(f"{'':>30s}vs reference: median |dR/R| = "
              f"{np.median(np.abs(ri / Rref - 1)):.3e}, "
              f"sign disagrees on {(np.sign(ri) != np.sign(Rref)).mean():.1%} "
              f"of targets, median |dQ/Q| = "
              f"{np.median(np.abs(Qi[:, 0] / Qref - 1)):.3e}", flush=True)
    print("wrote dev/hmc_ref.npz", flush=True)
