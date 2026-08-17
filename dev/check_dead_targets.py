"""Why does the Mc/Mr ceiling leave 15.7% of noisy targets with no weighted draw?

`bias.pqr_streamed` reports "N targets had no draw with any weight" and gives
them Q = R = 0, so they silently drop out of the eq. (45)-(46) sums.  That count
is 0 on the old chart and ~3100 of 20000 on the new one, at alpha 1.0 and 0.5
alike, and it moves overall m1 by ~+0.019 in the self-consistency control
([[noisy-binned-m1-is-half-methodology]]).

Two candidate mechanisms, which want opposite fixes:

  DOMAIN   the target's noisy M sits past a ceiling, so the kernel centred on
           it puts (almost) every draw off-chart and `in_domain` masks them.
           Fix: a proposal that is not centred on M, or an explicit cut.
  NUMERIC  the draws are on-chart but `log_prob` underflows -- the logit chart
           computes log p_base(z) + log|det J| as two large opposite terms, and
           near a ceiling both diverge, so float32 cancellation loses the finite
           answer.  Fix: the chart's conditioning, not the proposal.

This separates them: for each target, the fraction of draws `in_domain`, and
the best (max) log_prob over the draws that ARE in domain.

    python dev/check_dead_targets.py
"""
import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402
from models.bijections import (POINT_SOURCE, POINT_SOURCE_MC,  # noqa: E402
                               in_domain)

DATA = "../bfd_cnf_imsims/data"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--train-data", default=f"{DATA}/moments.fits")
    p.add_argument("--targets", default=f"{DATA}/targets_g0_1M.fits")
    p.add_argument("-n", type=int, default=20000)
    p.add_argument("--draws", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deriv-n", type=int, default=2000)
    p.add_argument("--deriv-draws", type=int, default=2048)
    a = p.parse_args()

    m_train = shear_top.load(a.train_data)[0]
    m_train = m_train[:int(0.9 * len(m_train))]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)

    m = np.asarray(shear_top.load(a.targets)[0][:a.n], np.float64)
    cov = bias.load_cov(a.targets)
    M = bias.add_noise(m, cov, a.seed)          # the noisy target, as bias.main
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(a.seed)
    dr = M[:, None, :] + rng.standard_normal((a.n, a.draws, 5)) @ L.T

    dom = np.asarray(in_domain(dr))                       # (n, draws)
    frac_dom = dom.mean(1)

    g = jnp.zeros(2)
    flat = dr.reshape(-1, 5)
    dflat = dom.reshape(-1)
    # Chunked: the (n x draws, 5) stack is millions of points and the shear
    # layer's per-point jacfwd log-det will not fit in one vmap.
    ev = eqx.filter_jit(jax.vmap(lambda x: flow.log_prob(x, condition=g)))
    out = np.empty(len(flat), np.float64)
    step = 200_000
    for i in range(0, len(flat), step):
        sl = slice(i, min(i + step, len(flat)))
        safe = jnp.where(jnp.asarray(dflat[sl])[:, None],
                         jnp.asarray(flat[sl], dtype=jnp.float32),
                         jnp.asarray(m[0], dtype=jnp.float32))
        out[sl] = np.asarray(ev(safe), np.float64)
    lp = np.where(dflat, out, -np.inf).reshape(a.n, a.draws)
    best = np.nanmax(np.where(np.isfinite(lp), lp, -np.inf), axis=1)

    dead_dom = frac_dom == 0.0
    dead_num = (~dead_dom) & ~np.isfinite(best)
    alive = np.isfinite(best)
    print(f"{a.flow}: {a.n} targets x {a.draws} draws\n")
    print(f"  dead, ALL draws off-chart (DOMAIN) : {100*dead_dom.mean():6.2f}%")
    print(f"  dead, on-chart but no finite log_p (NUMERIC): {100*dead_num.mean():6.2f}%")
    print(f"  alive                              : {100*alive.mean():6.2f}%")
    print(f"\n  draws off-chart overall: {100*(1-dom.mean()):.2f}%")

    y = m[:, 1] / m[:, 0]
    c = m[:, 4] / m[:, 1]
    Y = M[:, 1] / M[:, 0]
    C = M[:, 4] / M[:, 1]
    print(f"\n  ceilings: Mr/Mf < {POINT_SOURCE:.4f}, Mc/Mr < {POINT_SOURCE_MC:.4f}")
    for lab, s in (("alive", alive), ("dead(domain)", dead_dom),
                   ("dead(numeric)", dead_num)):
        if s.sum() < 5:
            print(f"  {lab:14s} n={int(s.sum()):6d}  --")
            continue
        print(f"  {lab:14s} n={int(s.sum()):6d}  "
              f"true Mr/Mf {np.median(y[s]):.3f}  true Mc/Mr {np.median(c[s]):.3f}  "
              f"| NOISY Mr/Mf {np.median(Y[s]):.3f} (over {100*np.mean(Y[s]>=POINT_SOURCE):5.1f}%)  "
              f"NOISY Mc/Mr {np.median(C[s]):.3f} (over {100*np.mean(C[s]>=POINT_SOURCE_MC):5.1f}%)")

    # --- the real test -------------------------------------------------------
    # `pqr_streamed` kills a chunk unless la AND df AND d2f are all finite, so a
    # target dies from non-finite g-DERIVATIVES, not a vanishing density.
    print(f"\n=== g-derivatives, exactly as pqr_streamed forms them "
          f"({a.deriv_n} targets, {a.deriv_draws} draws) ===")
    nb, zero = a.deriv_n, jnp.zeros(2)

    def one(m_i, draws_i, log_wt_i):
        f = lambda gg: bias.log_conv_is(flow, m_i, draws_i, log_wt_i, gg)
        vg = jax.value_and_grad(f)
        (val, df), (_, h0) = jax.jvp(vg, (zero,), (jnp.array([1.0, 0.0]),))
        _, (_, h1) = jax.jvp(vg, (zero,), (jnp.array([0.0, 1.0]),))
        return val, df, jnp.stack([h0, h1], axis=-1)

    run = eqx.filter_jit(jax.vmap(one))
    # Batch exactly as `pqr_streamed` does (64 targets x `chunk` draws); the
    # whole (nb x draws) stack in one vmap is hundreds of GB.
    vs, ds, hs = [], [], []
    for i in range(0, nb, 64):
        mb = jnp.asarray(M[i:i + 64], dtype=jnp.float32)
        d_, lw_ = bias.mixture_draws(flow, mb, cov, a.deriv_draws, 1.0,
                                     a.seed, batch=len(mb), sigma_x=None)
        v_, g_, h_ = run(mb, jnp.asarray(d_, jnp.float32),
                         jnp.asarray(lw_, jnp.float32))
        vs.append(np.asarray(v_, np.float64))
        ds.append(np.asarray(g_, np.float64))
        hs.append(np.asarray(h_, np.float64))
    val, df, d2f = np.concatenate(vs), np.concatenate(ds), np.concatenate(hs)
    nb = len(val)
    okv = np.isfinite(val)
    okd = np.isfinite(df).all(1)
    okh = np.isfinite(d2f).reshape(nb, -1).all(1)
    print(f"  val non-finite : {100*(~okv).mean():6.2f}%")
    print(f"  grad non-finite: {100*(~okd).mean():6.2f}%")
    print(f"  hess non-finite: {100*(~okh).mean():6.2f}%")
    print(f"  chunk REJECTED (any of the three): {100*(~(okv&okd&okh)).mean():6.2f}%")
    bad = ~(okv & okd & okh)
    if bad.sum() > 2:
        print(f"\n  rejected targets: median true Mr/Mf {np.median(y[:nb][bad]):.3f} "
              f"(alive {np.median(y[:nb][~bad]):.3f}), "
              f"true Mc/Mr {np.median(c[:nb][bad]):.3f} "
              f"(alive {np.median(c[:nb][~bad]):.3f})")
        print(f"  rejected: noisy Mc/(rc* Mr) median "
              f"{np.median(C[:nb][bad]/POINT_SOURCE_MC):.4f}, "
              f"frac of their draws off-chart "
              f"{np.mean(1-dom[:nb][bad].mean(1)):.4f}")

    # Of the targets that die, how far past a ceiling is their own M?
    if dead_dom.sum() > 5:
        d = dead_dom
        over1 = M[d, 1] / (POINT_SOURCE * M[d, 0])
        over2 = M[d, 4] / (POINT_SOURCE_MC * M[d, 1])
        print(f"\n  dead(domain) targets, own M as a fraction of each ceiling:")
        print(f"    Mr/(r* Mf) median {np.median(over1):.4f}  q90 {np.quantile(over1,0.9):.4f}")
        print(f"    Mc/(rc* Mr) median {np.median(over2):.4f}  q90 {np.quantile(over2,0.9):.4f}")


if __name__ == "__main__":
    main()
