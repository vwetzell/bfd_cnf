"""Is there ONE gauge that works at both ends, or only two that each work at one?

Gauge P and gauge K are the endpoints (v = 0 and v = s) of a one-parameter
family of changes of variables inside the SAME integral: move the prior by
(1-lambda) of the shear and the kernel by lambda.  Writing the exact velocity
field v = lambda*s into the continuity equation, the per-draw score is

    d_g l  =  (1 - lambda) X_P  +  lambda X_K

exactly -- a linear blend, with `E_pi[X_P] = E_pi[X_K] = Q` for every lambda.
So lambda is a CONTROL VARIATE, and the minimum-variance choice is the usual

    lambda* = (Var X_P - Cov(X_P, X_K)) / (Var X_P + Var X_K - 2 Cov)

`B = Var_pi[score]` is the term that blows up -- gauge P's at the faint end,
gauge K's at the bright end -- so this asks the design question directly:
**does the blend's minimum sit well below BOTH endpoints, everywhere?**

If yes, one adaptive estimator covers the catalog and the regime switch is
unnecessary.  If the minimum just tracks whichever endpoint is better, the
family buys nothing and the honest answer is two estimators.

This is post-processing of the per-draw scores only -- no new estimator, no
retrain.  It costs one `dev/gauge.py` decompose.
"""
import os
import numpy as np, jax, jax.numpy as jnp, equinox as eqx

from bias import (mixture_draws, in_domain, condition, safe_point, make_psi)
import dev.gauge as G
from dev.gauge import prep

NT = int(os.environ.get("NT", 400))
CHUNK = int(os.environ.get("CHUNK", 4096))
ALPHA = float(os.environ.get("ALPHA", 0.5))
TBATCH = 4


def scores(flow, psi, m, sx, cov):
    """Per-target (pi-weighted) Var of the blend at lambda = 0, 1 and lambda*."""
    cinv = jnp.asarray(np.linalg.inv(cov), jnp.float32)
    zero = jnp.zeros(2)

    @eqx.filter_jit
    def one(m_i, d_i, lw_i, sxi, ok_i):
        d = jnp.where(ok_i[:, None], d_i, safe_point(m_i))
        fp = lambda x: (lambda g: flow.log_prob(x, condition=condition(g, sxi)))
        lp0, xP = jax.vmap(lambda x: (fp(x)(zero), jax.grad(fp(x))(zero)))(d)
        # X_K = d_g log L(M - Psi_g(u)) = r0 . Cinv . dPsi/dg
        r0 = m_i - d
        dpsi = jax.vmap(lambda x: jax.jacfwd(psi, argnums=1)(x, zero, sxi))(d)
        xK = jnp.einsum("si,ij,sja->sa", r0, cinv, dpsi)

        u = jnp.where(ok_i, lp0 + lw_i, -jnp.inf)
        fin = jnp.isfinite(xK).all(-1) & jnp.isfinite(xP).all(-1)
        u = jnp.where(fin, u, -jnp.inf)
        pi = jax.nn.softmax(u)
        xP = jnp.where(fin[:, None], xP, 0.0)
        xK = jnp.where(fin[:, None], xK, 0.0)

        # g1 component only; the two directions behave alike by symmetry.
        # Zero-weight draws must be dropped BEFORE the squares: gauge P's score
        # reaches 1e15 on draws whose pi is exactly 0, and 0 * inf -- or the
        # float32 square itself, which overflows past |a| ~ 1.8e19 -- turns
        # the whole target into a NaN.  Kept draws peak near 1e4.  They
        # contribute nothing to any expectation, so this changes no estimand.
        keep = pi > 0
        pi = jnp.where(keep, pi, 0.0)
        a = jnp.where(keep, xP[:, 0], 0.0)
        b = jnp.where(keep, xK[:, 0], 0.0)
        ma, mb = pi @ a, pi @ b
        vaa = pi @ jnp.where(keep, (a - ma) ** 2, 0.0)
        vbb = pi @ jnp.where(keep, (b - mb) ** 2, 0.0)
        vab = pi @ jnp.where(keep, (a - ma) * (b - mb), 0.0)
        den = vaa + vbb - 2 * vab
        lam = jnp.where(den > 0, (vaa - vab) / jnp.where(den > 0, den, 1.0), 0.0)
        vmin = vaa + lam * lam * den - 2 * lam * (vaa - vab)
        # Q must agree between the endpoints -- that is what makes lambda free.
        return vaa, vbb, vmin, lam, ma, mb, 1.0 / jnp.sum(pi ** 2)

    out = []
    for i in range(0, len(m), TBATCH):
        m_b, sx_b = m[i:i + TBATCH], sx[i:i + TBATCH]
        d, lw = mixture_draws(flow, m_b, cov, CHUNK, ALPHA, 104729 * (i // TBATCH),
                              batch=len(m_b), sigma_x=sx_b)
        ok = in_domain(jnp.asarray(d))
        out.append([np.asarray(v, np.float64) for v in jax.vmap(one)(
            jnp.asarray(m_b, jnp.float32), jnp.asarray(d, jnp.float32),
            jnp.asarray(lw, jnp.float32), jnp.asarray(sx_b, jnp.float32), ok)])
        if i % (TBATCH * 20) == 0:
            print(f"  {i}/{len(m)}", flush=True)
    return [np.concatenate([o[j] for o in out]) for j in range(7)]


if __name__ == "__main__":
    import sys
    pop = sys.argv[1] if len(sys.argv) > 1 else "bulgedisc_deep_v2"
    fp = sys.argv[2] if len(sys.argv) > 2 else "flows/centroid_bulgedisc_v2.eqx"
    gain = float(sys.argv[3]) if len(sys.argv) > 3 else 1.4043

    G.NT = 2000
    flow, m, sx, cov = prep(pop, fp, gain)
    keep = np.linspace(0, len(m) - 1, NT).astype(int)     # spread over all fluxes
    m, sx = m[keep], sx[keep]
    # `pqr_streamed` does not peel this chain's (5,)-conditioned centroid layer,
    # so the draws it evaluates are RAW -- match that here.
    psi = make_psi(flow, peeled=False)

    vP, vK, vmin, lam, qP, qK, ess = scores(flow, psi, m, sx, cov)
    ed = np.percentile(m[:, 0], [0, 20, 40, 60, 80, 100])
    print(f"\n{'Mf quintile':>12s}{'B lam=0 (P)':>14s}{'B lam=1 (K)':>14s}"
          f"{'B at lam*':>12s}{'med lam*':>10s}{'gain vs best':>14s}{'dQ P-vs-K':>11s}")
    for k in range(5):
        s = (m[:, 0] >= ed[k]) & (m[:, 0] <= ed[k + 1])
        best = np.minimum(vP[s], vK[s])
        print(f"{'q'+str(k+1):>12s}{np.median(vP[s]):14.3e}{np.median(vK[s]):14.3e}"
              f"{np.median(vmin[s]):12.3e}{np.median(lam[s]):10.3f}"
              f"{np.median(best / np.maximum(vmin[s], 1e-30)):14.2f}"
              f"{np.median(np.abs(qK[s] / qP[s] - 1)):11.2e}")
    print(f"\nlambda* percentiles: " +
          " ".join(f"p{p}={np.percentile(lam, p):.3f}" for p in (5, 25, 50, 75, 95)))
    print(f"median ESS {np.median(ess):.0f}")
