"""Closed loop: the flow IS the truth, so any bias left is numerics.

`catalogs`: draw a synthetic target population from the flow itself at
g = +/-0.02 (and 0), common base draw and common noise across arms (paired,
exactly like the rendered catalogs), noise from the real targets' C_M.  Written
in the real targets' FITS layout and registered in bias.py as pop
`gauss2_v4n_closed`, so the production pipeline runs on it unchanged.  Prior,
likelihood and selection are exact by construction: a nonzero m1/c there is
target integration, selection-term MC, Newton step or finite N.

`selection`: ground-truth P_s, R_s for the SAME flow by a common-random-number
finite difference in g of P_s(g) = E_{m~P(.|g)}[F(m)], F = bias.window_prob.
F is bounded, so this has finite variance at any draw count -- unlike the
pathwise estimator -- and it differentiates the sample, not the density, so it
is independent of the score estimator's weighting.  Two step sizes expose the
O(h^2) truncation.
"""
import argparse
import os
import sys
sys.path.insert(0, ".")

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import bias as B
import bulk
import shear

D = "../bfd_cnf_imsims/data"
# C_M only: moment noise does not depend on the galaxy.  REAL_POP must be fixed-noise, round-PSF.
REAL = B.CATALOGS[__import__("os").environ.get("REAL_POP", "gauss2_v4n")]
# the flow's own prior (chart constants are rebuilt from it); TRAIN_POP=bulgedisc_g2n for that chain
TRAIN = __import__("os").environ.get("TRAIN_POP", "gauss2_v4n")
# target noise / 0.93: C_M and Sigma_X both scale as noise^2 (DEPTH=sn8 in dev/bdg2n.sh)
S2 = float(__import__("os").environ.get("NOISE_SCALE", "1")) ** 2


def load_flow(path):
    m = np.asarray(shear.load(f"{D}/{B.TRAIN_DATA[TRAIN]}")[0])[:20000]
    return bulk.load_flow(path, m, shear=True, centroid=True)   # adapter flows too


def sampler(flow, sx):
    @eqx.filter_jit
    def f(z, g):
        return jax.vmap(lambda zi: flow.bijection.transform(zi, B.condition(g, sx)))(z)
    return f


def catalogs(a):
    flow = load_flow(a.flow)
    real = {k: fitsio.read(f"{D}/{v}.fits", ext=1) for k, v in REAL.items()}
    hdr = fitsio.read_header(f"{D}/{REAL['plus']}.fits", ext=1)
    sx = jnp.asarray(real["plus"]["cov_odd"][0], jnp.float32)
    cm = B.load_cov(f"{D}/{REAL['plus']}.fits")
    draw = sampler(flow, sx)
    kz, kn = jr.split(jr.key(a.seed))
    z = flow.base_dist.sample(kz, (a.n,))
    rng = np.random.default_rng(a.seed)
    if a.jacobian:
        # The paper's P(M|g) = J(M) INT p_flow(m|g)/J(m) N(M; m, C) dm: per galaxy the
        # noise is N(n) J(m+n)/J(m) (normalised: E[J(m+n)] = J(m) as Tr(BC) = 0).
        # Rejection from N with bound KMAX, candidates and uniforms common to the
        # arms so they stay paired.  ponytail: KMAX clips ratios > 3 (faint only; counted).
        cand = rng.multivariate_normal(np.zeros(5), cm, (a.n, 48)).astype(np.float32)
        u = rng.random((a.n, 48), dtype=np.float32)
    else:
        noise = rng.multivariate_normal(np.zeros(5), cm, a.n)
    for arm, g1 in (("plus", 0.02), ("minus", -0.02), ("zero", 0.0)):
        g = jnp.array([g1, 0.0])
        m = np.concatenate([np.asarray(draw(z[i:i + a.batch], g), np.float64)
                            for i in range(0, a.n, a.batch)])
        if a.jacobian:
            jac = lambda d: 0.25 * (d[..., 1] ** 2 - d[..., 2] ** 2 - d[..., 3] ** 2)
            r = np.maximum(jac(m[:, None] + cand) / jac(m)[:, None], 0.0)
            r = np.where(np.isfinite(r), r, 1.0)
            ok = u < r / 3.0
            k = ok.argmax(1)
            print(f"  {arm}: ratio > 3 in {(r > 3).mean():.2e} of candidates; "
                  f"{(~ok.any(1)).sum()} rows with no acceptance (kept candidate 0)")
            noise = cand[np.arange(a.n), k]
        m = m + noise
        bad = ~np.isfinite(m).all(1)
        m[bad] = 0.0
        out = np.zeros(a.n, dtype=real[arm].dtype)
        out["moments"] = m
        for col in ("cov", "cov_odd", "nda"):
            out[col] = real[arm][col][0]
        out["badcenter"] = bad   # non-finite flow draws: dropped by bias.py in all arms
        h = {k: hdr[k] for k in ("PIXSCALE", "WTSIGMA", "PSFSIGMA", "NOISESIG",
                                 "PSFE1", "PSFE2", "POPKIND", "IMGNOISE", "SIG_XY")}
        h.update(G1=g1, G2=0.0, SEED=a.seed, CLOSEDLP=os.path.basename(a.flow), NPOP=a.n,
                 CLOSEDJ=a.jacobian)
        path = f"{D}/{B.CATALOGS[a.pop][arm]}.fits"
        fitsio.write(path, out, header=h, clobber=True)
        print(f"{arm:5s} g1={g1:+.2f}  wrote {path}  ({bad.sum()} non-finite draws)")


def selection(a):
    flow = load_flow(a.flow)
    sx = jnp.asarray(S2 * fitsio.read(f"{D}/{REAL['plus']}.fits", rows=[0])["cov_odd"][0], jnp.float32)
    cm = jnp.asarray(S2 * B.load_cov(f"{D}/{REAL['plus']}.fits"), jnp.float32)
    draw = sampler(flow, sx)
    size, flux = tuple(a.size), tuple(a.flux)

    # (g, weight) stencils: R11, R22 central second differences, R12 cross.
    def stencils(h):
        e1, e2 = np.array([h, 0.0]), np.array([0.0, h])
        return {"R11": [(e1, 1), (-e1, 1), (0 * e1, -2)],
                "R22": [(e2, 1), (-e2, 1), (0 * e1, -2)],
                "R12": [(e1 + e2, .25), (-e1 - e2, .25), (e1 - e2, -.25), (e2 - e1, -.25)]}

    gs = {tuple(g) for h in a.h for st in stencils(h).values() for g, _ in st}
    F = eqx.filter_jit(lambda m: B.window_prob(m, cm, size, flux))
    key = jr.key(a.seed)
    n_done, n = 0, 1 << a.log2_draws
    # per-draw second differences, accumulated (sum, sum sq) in float64
    acc = {(h, k): [0.0, 0.0] for h in a.h for k in ("R11", "R22", "R12")}
    p0 = [0.0, 0.0]
    while n_done < n:
        key, k = jr.split(key)
        z = flow.base_dist.sample(k, (a.batch,))
        Fg = {g: np.asarray(F(draw(z, jnp.asarray(g, jnp.float32))), np.float64) for g in gs}
        Fg = {g: np.where(np.isfinite(v), v, 0.0) for g, v in Fg.items()}
        f0 = Fg[(0.0, 0.0)]
        p0[0] += f0.sum(); p0[1] += (f0 ** 2).sum()
        for h in a.h:
            for name, st in stencils(h).items():
                d = sum(w * Fg[tuple(g)] for g, w in st) / h ** 2
                acc[(h, name)][0] += d.sum(); acc[(h, name)][1] += (d ** 2).sum()
        n_done += a.batch
        if n_done % (1 << 22) == 0:
            print(f"  {n_done} draws", flush=True)
    mean = lambda s: (s[0] / n_done, np.sqrt(max(s[1] / n_done - (s[0] / n_done) ** 2, 0) / n_done))
    P, Pe = mean(p0)
    print(f"window size {size} flux {flux}, 2^{a.log2_draws} CRN draws")
    print(f"  P_s = {P:.5f} +/- {Pe:.5f}")
    for h in a.h:
        row = "  ".join(f"{k} {mean(acc[(h, k)])[0]:+.5f} +/- {mean(acc[(h, k)])[1]:.5f}"
                        for k in ("R11", "R22", "R12"))
        print(f"  h={h:.3f}: {row}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["catalogs", "selection"])
    p.add_argument("--flow", default="flows/centroid_g2v4n_K.eqx")
    p.add_argument("--n", type=int, default=65372, help="population size (real NPOP)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pop", default="gauss2_v4n_closed", help="bias.py CATALOGS entry to write")
    p.add_argument("--jacobian", action="store_true", help="noise tilted by J(M)/J(m) (paper eq. pMsG2)")
    p.add_argument("--batch", type=int, default=65536)
    p.add_argument("--log2-draws", type=int, default=24)
    p.add_argument("--h", type=float, nargs="+", default=[0.02, 0.05])
    p.add_argument("--size", type=float, nargs=2, default=[2.2, 3.2])
    p.add_argument("--flux", type=float, nargs=2, default=[3000, 20000])
    a = p.parse_args()
    catalogs(a) if a.cmd == "catalogs" else selection(a)


if __name__ == "__main__":
    main()
