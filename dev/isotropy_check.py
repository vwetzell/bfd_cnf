"""Is `Q_s != 0` a real anisotropy, or is it MC noise in the estimator?

`Q_s = E[F(m) Q(m)]` over prior draws at `g = 0`.  The window `F` cuts on `Mf`
and `Mr/Mf`, both spin-0, and `Sigma_X` is isotropic on disk (`C00 = C11`,
`C01 = 4.5e-13`), so IF the flow's density is rotation-invariant then `Q_s = 0`
exactly, and likewise `R_s11 = R_s22`, `R_s12 = 0`.

Rather than argue that from the layer definitions, measure it.  A sky rotation
by `theta` rotates the spin-2 pair `(M1, M2)` by `2 theta` and leaves
`(Mf, Mr, Mc)` alone.  Take `theta = 45 deg`, i.e. `(M1, M2) -> (-M2, M1)`,
which is exact in floating point.  Equivariance of the whole chain means

    log P(rot m) = log P(m)          and     Q(rot m) = Rot(90 deg) Q(m)

per draw.  If that holds, then summing `F Q` over the four rotations
`k * 90 deg` gives exactly zero draw by draw -- so `Q_s` is zero as an
INTEGRAL, the sample value is pure MC scatter, and symmetrising is a free
variance reduction that makes it zero as an ESTIMATE too.

Also prints the Hill tail index of the per-draw `F Q`, which is what says
whether the quoted `Q_s_err` means anything.

    python -u dev/isotropy_check.py --draws 65536
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

import bias as B
import bulk


def hill(x, k=None):
    """Hill tail index of |x|; < 1 no finite mean, < 2 no finite variance."""
    v = np.sort(np.abs(np.asarray(x, dtype=np.float64)))[::-1]
    v = v[v > 0]
    k = max(10, len(v) // 100) if k is None else min(k, len(v) - 1)
    return float(1.0 / np.mean(np.log(v[:k]) - np.log(v[k])))


def rot_m(m, k):
    """Sky rotation by `k * 45 deg`: the spin-2 pair turns by `k * 90 deg`."""
    m1, m2 = m[:, 2], m[:, 3]
    p = [(m1, m2), (-m2, m1), (-m1, -m2), (m2, -m1)][k % 4]
    return m.at[:, 2].set(p[0]).at[:, 3].set(p[1])


def rot_g(q, k):
    """The matching rotation on a spin-2 vector in the shear plane."""
    q1, q2 = q[:, 0], q[:, 1]
    p = [(q1, q2), (-q2, q1), (-q1, -q2), (q2, -q1)][k % 4]
    return jnp.stack(p, axis=-1)


def rot_sigma(sx, k):
    """`Sigma_X = [C00, C01, C11]` under the SAME `k * 45 deg` sky rotation.

    `X` is a position, i.e. spin-1, so `Sigma_X -> R Sigma_X R^T` with `R` the
    ORDINARY rotation by `k * 45 deg` -- not the doubled angle the spin-2
    moments turn through.  Holding `Sigma_X` fixed while rotating the draws (as
    this script originally did) is only valid while it is isotropic; with an
    elliptical PSF it is not, and the joint rotation is the equivariance the
    chain actually has to satisfy.

    At `k = 1` this maps the `e2p` value `[12676.665, 1084.720, 12676.665]`
    onto `[11591.944, 0, 13761.385]`, which is exactly the rendered `e1m`
    `Sigma_X` -- so the identity is checkable against an independent catalog.
    """
    if sx is None:
        return None
    a, b, d = (float(x) for x in np.asarray(sx))
    c, s = np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)
    return jnp.asarray([c * c * a - 2 * c * s * b + s * s * d,
                        c * s * (a - d) + (c * c - s * s) * b,
                        s * s * a + 2 * c * s * b + c * c * d],
                       dtype=jnp.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_v10.eqx")
    p.add_argument("--pop", default="bulgedisc_v3")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--draws", type=int, default=65536)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(2500.0, 50000.0))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-centroid", action="store_true",
                   help="build a shear-only chain (hypothesis 4)")
    p.add_argument("--sigma-x", type=float, nargs=3, default=None,
                   metavar=("C00", "C01", "C11"),
                   help="override Sigma_X; use the e2p value 12676.665 "
                        "1084.720 12676.665 to exercise the C01 path")
    p.add_argument("--rotate-sigma", action="store_true",
                   help="rotate Sigma_X WITH the draws (mandatory once it is "
                        "anisotropic -- holding it fixed is not the symmetry)")
    a = p.parse_args()

    import fitsio
    train = f"{a.data_dir}/moments_{a.pop}.fits"
    targets = f"{a.data_dir}/targets_{a.pop.replace('bulgedisc_', '')}_g0_200k.fits"
    cov = B.load_cov(targets)
    sigma_x = None if a.no_centroid else jnp.asarray(
        np.asarray(a.sigma_x if a.sigma_x else
                   fitsio.read(targets)["cov_odd"][0], dtype=np.float64))

    m_train = bulk.load_moments(train)
    flow = bulk.build_flow(jr.key(a.seed), m_train[:int(0.9 * len(m_train))],
                           shear=True, centroid=not a.no_centroid)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    flow = bulk.SupportedFlow(flow, eps=0.0)

    size, flux = tuple(a.window_size), tuple(a.window_flux)
    print(f"flow {a.flow}  centroid={not a.no_centroid}\n"
          f"window: size {size}, flux {flux}\n{a.draws} prior draws")
    print(f"Sigma_X = {None if sigma_x is None else np.asarray(sigma_x)}   "
          f"rotate-sigma = {a.rotate_sigma}")
    if a.rotate_sigma:
        for k in range(1, 4):
            print(f"  at {k * 45:>3d} deg: {np.asarray(rot_sigma(sigma_x, k))}")
    elif sigma_x is not None and (sigma_x[0] != sigma_x[2]
                                  or abs(sigma_x[1]) > 1e-6):
        print("  WARNING: Sigma_X is anisotropic and is being held FIXED "
              "across the rotations.\n  That is not a symmetry of the chain; "
              "pass --rotate-sigma.")
    print()

    z = flow.base_dist.sample(jr.key(a.seed + 31), (a.draws,)).astype(jnp.float64)
    m0 = jax.vmap(lambda z1: flow.bijection.transform(
        z1, B.condition(jnp.zeros(2), sigma_x)))(z)

    zero = jnp.zeros(2)

    def logp_and_q(mm, sx):
        def one(m_i):
            f = lambda g: flow.log_prob(m_i, condition=B.condition(g, sx))
            return jax.value_and_grad(f)(zero)
        return jax.vmap(one)(mm)

    step = eqx.filter_jit(logp_and_q)

    lps, qs, Fs = [], [], []
    for k in range(4):
        sx_k = rot_sigma(sigma_x, k) if a.rotate_sigma else sigma_x
        lp_k, q_k = [], []
        for i in range(0, len(m0), a.batch):
            mm = rot_m(m0[i:i + a.batch], k)
            lp, q = step(mm, sx_k)
            lp_k.append(np.asarray(lp)), q_k.append(np.asarray(q))
            if k == 0:
                Fs.append(np.asarray(B.window_prob(mm, cov, size, flux)))
        lps.append(np.concatenate(lp_k)), qs.append(np.concatenate(q_k))
    F = np.concatenate(Fs)

    ok = np.isfinite(F) & np.all([np.isfinite(l) for l in lps], axis=0) \
        & np.all([np.isfinite(q).all(-1) for q in qs], axis=0)
    print(f"finite in all four rotations: {ok.sum()}/{len(ok)}")
    lps = [l[ok] for l in lps]
    qs = [np.asarray(q)[ok] for q in qs]
    F = F[ok]

    # --- equivariance, per draw --------------------------------------------
    # The MAX is the wrong statistic here: the flow's tail has a handful of
    # numerically ill-conditioned draws, and they carry no window weight.
    # Report the distribution, and the F-weighted version that Q_s actually
    # integrates.
    print(f"\n{'rotation':>10s}{'|dlogP| p50':>13s}{'p99':>10s}{'max':>10s}"
          f"{'|dQ|/|Q| p50':>14s}{'p99':>10s}{'max':>10s}{'F-wtd |dQ|':>13s}")
    nq = np.linalg.norm(qs[0], axis=-1)
    for k in range(1, 4):
        dl = np.abs(lps[k] - lps[0])
        dq = np.linalg.norm(qs[k] - np.asarray(rot_g(jnp.asarray(qs[0]), k)),
                            axis=-1)
        rel = dq / np.maximum(nq, 1e-300)
        print(f"{k * 90:>7d}deg{np.median(dl):13.2e}"
              f"{np.percentile(dl, 99):10.2e}{dl.max():10.2e}"
              f"{np.median(rel):14.2e}{np.percentile(rel, 99):10.2e}"
              f"{rel.max():10.2e}{(F * dq).sum() / F.sum():13.2e}")
    print(f"(|Q| median {np.median(nq):.3g}.  Equivariance at the level the "
          f"window weights\n means the chain is isotropic and Q_s = 0 as an "
          f"integral.)")

    # --- Q_s: naive vs symmetrised -----------------------------------------
    n = len(F)
    fq = F[:, None] * qs[0]
    q_naive = fq.mean(0)
    sem_iid = fq.std(0, ddof=1) / np.sqrt(n)
    chunks = np.stack([fq[i:i + a.batch].mean(0)
                       for i in range(0, n, a.batch)])
    sem_chunk = chunks.std(0, ddof=1) / np.sqrt(len(chunks))
    fq_sym = np.mean([F[:, None] * q for q in qs], axis=0)
    print(f"\nQ_s naive       = ({q_naive[0]:+.4e}, {q_naive[1]:+.4e})")
    print(f"  SEM iid       = ({sem_iid[0]:.2e}, {sem_iid[1]:.2e})   "
          f"z = ({q_naive[0] / sem_iid[0]:+.2f}, {q_naive[1] / sem_iid[1]:+.2f})")
    print(f"  SEM chunk     = ({sem_chunk[0]:.2e}, {sem_chunk[1]:.2e})   "
          f"z = ({q_naive[0] / sem_chunk[0]:+.2f}, "
          f"{q_naive[1] / sem_chunk[1]:+.2f})")
    print(f"Q_s symmetrised = ({fq_sym.mean(0)[0]:+.4e}, "
          f"{fq_sym.mean(0)[1]:+.4e})   max per-draw "
          f"{np.abs(fq_sym).max():.2e}")

    # --- tails of the integrand --------------------------------------------
    print(f"\nHill index of |F Q|: q1 {hill(fq[:, 0]):.2f}  q2 "
          f"{hill(fq[:, 1]):.2f}   (< 2 => the SEM is meaningless)")
    print(f"top draw's share of |Q_s2|: "
          f"{np.abs(fq[:, 1]).max() / (n * abs(q_naive[1])):.1%}")


if __name__ == "__main__":
    main()
