"""Compare the three estimators of eq. (40)'s selection terms, side by side.

`P_s`, `Q_s` and `R_s` depend only on the flow, `C_M` and the window -- not on
the targets -- so there is no reason to pay for a full `bias.py` run (20k
targets x 8192 convolution draws, ~40 min) to compare them.  This does the
selection terms alone, in minutes.

    pathwise   `--window-terms flow`, autodiff.  Differentiates the SAMPLE:
               d2F/dg2 = F''(m)(dm/dg)^2 + ..., and F is a noise-smoothed step,
               so the per-draw term goes as Mf^2/sigma^2.  Hill index ~0.75,
               i.e. no finite mean -- it gets WORSE with more draws.
    fd         `--window-terms flow --window-fd H`.  Central differences of F,
               which is bounded in [0, 1], so the mean exists.  Costs an
               O(H^2) bias and H must not be shrunk (that walks back toward the
               divergence).
    score      `--window-terms score`.  Differentiates the DENSITY instead:
               Q_s = E[F Q], R_s = E[F (R + Q Q^T)] over prior draws at g = 0.
               F carries no g at all, so there is no sigma in the integrand and
               nothing to diverge.  Exact, and no step size.

Also reports the concentration (top draw's share of R_s11) and a Hill tail
index for each, which is what actually distinguishes them.

    python -u dev/window_terms_compare.py --flow flows/centroid_v10.eqx
"""
from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")
# x64 is switched on INSIDE main, after the checkpoint is loaded: enabling it
# here would make `build_flow` produce float64 leaves and the float32 file on
# disk would then fail equinox's dtype check.  `bias.py` does the same.

import bias as B
import bulk


def hill(x, k=None):
    """Hill tail-index estimate of |x|.  < 1 means the mean does not exist,
    < 2 means the variance does not.  `k` defaults to the top 1%."""
    v = np.sort(np.abs(np.asarray(x, dtype=np.float64)))[::-1]
    v = v[v > 0]
    k = max(10, len(v) // 100) if k is None else k
    k = min(k, len(v) - 1)
    return float(1.0 / np.mean(np.log(v[:k]) - np.log(v[k])))


def per_draw_r00(draw, z, cov, size, flux, fd, batch=16384):
    """Per-draw d2F/dg1^2 (or its central difference), for the tail diagnostics."""
    zero = jnp.zeros(2)

    def exact(zc):
        e0 = jnp.array([1.0, 0.0])
        fv = lambda g: B.window_prob(draw(g, zc), cov, size, flux)
        d1 = lambda g: jax.jvp(fv, (g,), (e0,))[1]
        return jax.jvp(d1, (zero,), (e0,))[1]

    def diff(zc):
        P = lambda a: B.window_prob(draw(jnp.array([a, 0.0]), zc), cov, size, flux)
        return (P(fd) - 2.0 * P(0.0) + P(-fd)) / fd ** 2

    f = eqx.filter_jit(diff if fd else exact)
    out = [np.asarray(f(jnp.asarray(z[i:i + batch])))
           for i in range(0, len(z), batch)]
    v = np.concatenate(out)
    return v[np.isfinite(v)]


def per_draw_r00_score(flow, m, cov, size, flux, sigma_x, batch=4096):
    """Per-draw F (R + Q Q^T)_11, the score-function integrand."""
    zero = jnp.zeros(2)
    e0 = jnp.array([1.0, 0.0])

    def one(m_i):
        f = lambda g: flow.log_prob(m_i, condition=B.condition(g, sigma_x))
        vg = jax.value_and_grad(f)
        (_, q), (_, h0) = jax.jvp(vg, (zero,), (e0,))
        return q[0], h0[0]

    g1 = eqx.filter_jit(jax.vmap(one))
    out = []
    for i in range(0, len(m), batch):
        mm = jnp.asarray(m[i:i + batch])
        q0, r00 = g1(mm)
        F = B.window_prob(mm, cov, size, flux)
        out.append(np.asarray(F * (r00 + q0 * q0)))
    v = np.concatenate(out)
    return v[np.isfinite(v)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_v10.eqx")
    p.add_argument("--pop", default="bulgedisc_v3")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--train-data", default=None)
    p.add_argument("--draws", type=int, default=262144)
    p.add_argument("--fd", type=float, default=0.02)
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(2500.0, 50000.0))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    train = a.train_data or f"{a.data_dir}/moments_{a.pop}.fits"
    targets = f"{a.data_dir}/targets_{a.pop.replace('bulgedisc_', '')}_g0_200k.fits"
    import fitsio
    cov = B.load_cov(targets)
    sigma_x = jnp.asarray(
        np.asarray(fitsio.read(targets)["cov_odd"], dtype=np.float64)[0])

    m_train = bulk.load_moments(train)
    flow = bulk.build_flow(jr.key(a.seed), m_train[:int(0.9 * len(m_train))],
                           shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    flow = bulk.SupportedFlow(flow, eps=0.0)

    size, flux = tuple(a.window_size), tuple(a.window_flux)
    print(f"flow {a.flow}\nwindow: size {size} (Mr/Mf), flux {flux} (Mf)\n"
          f"{a.draws} prior draws, fd step {a.fd}\n")

    z = flow.base_dist.sample(jr.key(a.seed + 31), (a.draws,)).astype(jnp.float64)
    draw = lambda g, zz: jax.vmap(
        lambda z1: flow.bijection.transform(z1, B.condition(g, sigma_x)))(zz)
    m0 = draw(jnp.zeros(2), z)

    rows = []
    for name, fd in (("pathwise (autodiff)", None), (f"fd = {a.fd}", a.fd)):
        ps, qs, rs, qs_err = B.selection_terms(draw, z, cov, size, flux, fd=fd)
        v = per_draw_r00(draw, z, cov, size, flux, fd)
        rows.append((name, ps, qs, qs_err, rs, v))

    ps, qs, rs, qs_err = B.selection_terms_score(flow, m0, cov, size, flux,
                                                 sigma_x=sigma_x)
    v = per_draw_r00_score(flow, m0, cov, size, flux, sigma_x)
    rows.append(("score-function", ps, qs, qs_err, rs, v))

    print(f"\n{'estimator':22s}{'P_s':>9s}{'R_s11':>11s}{'R_s22':>11s}"
          f"{'Q_s1/err':>11s}{'top share':>11s}{'Hill':>8s}")
    for name, ps, qs, qs_err, rs, v in rows:
        share = np.abs(v).max() / (len(v) * abs(rs[0, 0])) if rs[0, 0] else np.inf
        z1 = qs[0] / qs_err[0] if qs_err[0] else np.nan
        print(f"{name:22s}{ps:9.4f}{rs[0, 0]:11.4f}{rs[1, 1]:11.4f}"
              f"{z1:11.2f}{share:11.1%}{hill(v):8.2f}")
    print("\nQ_s1/err: isotropy forces Q_s = 0, so |z| >> 1 means something is "
          "wrong.\ntop share: one draw's share of R_s11 -- a sample mean with no "
          "effective n.\nHill: tail index of the per-draw integrand; < 1 means "
          "NO FINITE MEAN, < 2 no\n      finite variance.  This is the number "
          "that separates the three.")


if __name__ == "__main__":
    main()
