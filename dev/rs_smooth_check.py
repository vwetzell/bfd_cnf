"""Does convolving the flow's density with C_M (eq. 38's own kernel) tame the
selection-term score/curvature spikes `dev/rs_census.py` found, WITHOUT
disturbing the bulk where the flow already agrees with templates?

Context: `dev/rs_census.py --pop gauss2_v3e --flow flows/centroid_g2v3d.eqx`
found 10 leading draws for `R_s11` that sit WELL INSIDE both `in_support` and
the selection window (Mr/Mf 2.25-2.94 against a 3.69 ceiling and a (2.2,3.2)
window), yet carry F(R11+Q1^2) up to 1.5e5 against a pilot median of ~80 --
so it is not a manifold-support leak, it is the flow's fitted density itself
being unboundedly sharp in a region with real training data.

`bias.log_conv_is` already computes `P(M_i|g) = INT P(m|g) L(M_i - m) dm`,
the paper's own eq. (38) continuum convolution -- the SAME object that makes
BFD's template-sum KDE provably bounded (bandwidth = C_M), just evaluated on
the flow via importance-sampled kernel draws instead of a template sum.

Two checks, both against the RAW (point-evaluated `flow.log_prob`) estimator:

  1. LEADERS -- the 10 draws that dominate the raw `R_s11` sum.  Does the
     smoothed `F*(R11+Q1^2)` collapse toward template-sum scale (O(1e3-1e4)),
     and is that collapse a real signal or still MC noise at the sample counts
     tried?  `--reps` independent kernel-draw seeds per `--samples` value give
     a mean +/- std per leader instead of one point estimate.
  2. BULK -- a plain random sample from the flow's own prior (not
     cherry-picked), where the raw estimator is presumably already sane.  If
     smoothing moves these as much as it moves the leaders, it is not
     selective and the fix is not safe to deploy; if it leaves them close to
     raw (within the smoothed estimator's own MC error), the fix only bites
     where the flow is actually pathological.

    python -u dev/rs_smooth_check.py --leaders <rs_census --save output.npz>
"""
import argparse
import os
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402


def _smoothed(flow, m, cov, sigma_x_n, samples, seed, batch=None):
    """(q, r) from `pqr_full` under the eq. (38) convolution, pure-kernel
    proposal (`alpha=1`, so `log_wt` is identically zero and `L` cancels).

    `pqr_full` vmaps ONE forward-over-reverse Hessian call over its whole
    `batch` of targets, each carrying `samples` inner kernel draws -- so the
    cost is NOT `batch * samples` flat evals, it is `batch` Hessians of a
    function whose primal already costs `samples` flow calls, and reverse-mode
    retains intermediates for all of them.  `bias._score_guard`'s own default
    Hessian batch is 4096 at `samples=1` (its docstring: a 65536-wide one
    OOMs at 11.7 GiB); `131072 // samples` (mirroring `mixture_draws`'s flat
    budget) still OOM'd here at samples=64 (18.5 GiB).  Scale the SAME 4096
    budget down by `samples` instead.
    """
    if batch is None:
        batch = max(1, 4096 // samples)
    offsets = B.kernel_draws(cov, len(m), samples, seed)      # (n, samples, 5)
    draws = m[:, None, :] + offsets
    log_wt = jnp.zeros(draws.shape[:2])
    _, q, r = B.pqr_full(flow, m, draws=draws, log_wt=log_wt, sigma_x=sigma_x_n,
                         batch=batch)
    return q, r


def _draw_prior(flow, n, seed, sigma_x, chunk=16384):
    """Plain draws from the flow's own prior at g=0 -- same recipe
    `window_scan.py`/`rs_census.py` use, base_dist -> bijection at g=0."""
    zs = flow.base_dist.sample(jr.key(seed), (n,)).astype(jnp.float64)
    tr = eqx.filter_jit(jax.vmap(lambda z1: flow.bijection.transform(
        z1, B.condition(jnp.zeros(2), jnp.asarray(sigma_x)))))
    m = jnp.concatenate([tr(zs[i:i + chunk]) for i in range(0, n, chunk)])
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--leaders", required=True,
                   help="npz from `rs_census.py --save`")
    p.add_argument("--pop", default="gauss2_v3e", choices=sorted(B.CATALOGS))
    p.add_argument("--flow", required=True)
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(2500.0, 50000.0))
    p.add_argument("--top", type=int, default=10, help="leading draws to check")
    p.add_argument("--samples", type=int, nargs="*", default=[16, 64, 256],
                   help="kernel-draw counts to try, alpha=1 (pure kernel)")
    p.add_argument("--reps", type=int, default=5,
                   help="independent kernel-draw seeds per `samples`, for a "
                        "mean +/- std instead of one point estimate")
    p.add_argument("--bulk-n", type=int, default=20000,
                   help="plain (non-cherry-picked) prior draws for the "
                        "control check")
    p.add_argument("--bulk-samples", type=int, default=64,
                   help="kernel-draw count used for the bulk control")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    z = np.load(a.leaders)
    print(f"{a.leaders}: {len(z['moments'])} retained leaders, "
          f"checking the top {a.top}\n")

    cat = B.CATALOGS[a.pop]
    targets = f"{a.data_dir}/{cat['zero']}.fits"
    cov = B.load_cov(targets)
    sigma_x = np.asarray(fitsio.read(targets)["cov_odd"], dtype=np.float64)[0]

    # Flow checkpoints are stored float32: build/deserialise BEFORE enabling
    # x64 (`eqx.tree_deserialise_leaves` asserts dtype against the template),
    # then flip x64 on and upcast -- same order `rs_census.py`/`window_scan.py`
    # use.  Every float64 array in this script (including `m_top` below) has
    # to be built AFTER this point, or it silently stays float32 with only a
    # warning (bit this script once already).
    m_train = shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    flow = bulk.SupportedFlow(flow, eps=0.0, support=True)

    m_top = jnp.asarray(z["moments"][:a.top], dtype=jnp.float64)

    fprob = jax.jit(lambda mm: B.window_prob(mm, cov, a.window_size, a.window_flux))

    # ------------------------------------------------------------------ #
    # 1. LEADERS: raw vs smoothed, with a mean +/- std over `--reps` seeds.
    # ------------------------------------------------------------------ #
    sigma_x_top = jnp.tile(jnp.asarray(sigma_x), (a.top, 1))
    F_top = np.asarray(fprob(m_top))
    _, q_raw, r_raw = B.pqr_full(flow, m_top, sigma_x=sigma_x_top)
    v_raw = F_top * (r_raw[:, 0, 0] + q_raw[:, 0] ** 2)

    print("=== LEADERS (cherry-picked, dominate the raw R_s11 sum) ===")
    print(f"{'Mf':>10} {'Mr/Mf':>7} | {'raw F*v':>11} |"
          + "".join(f"   S={s:<4d} mean +/- std   " for s in a.samples))
    smoothed = {}   # samples -> (reps, top) array of F*v
    for s in a.samples:
        vals = []
        for rep in range(a.reps):
            q_s, r_s = _smoothed(flow, m_top, cov, sigma_x_top, s,
                                 a.seed + 1000 * s + rep)
            vals.append(np.asarray(F_top * (r_s[:, 0, 0] + q_s[:, 0] ** 2)))
        smoothed[s] = np.stack(vals)   # (reps, top)

    for i in range(a.top):
        Mf, Mr = m_top[i, 0], m_top[i, 1]
        line = (f"{float(Mf):10.4g} {float(Mr / Mf):7.4f} | "
                f"{v_raw[i]:11.4g} |")
        for s in a.samples:
            mu, sd = smoothed[s][:, i].mean(), smoothed[s][:, i].std(ddof=1)
            line += f" {mu:10.4g} +/- {sd:9.3g}"
        print(line)

    print(f"\n  raw sum F*v over top {a.top}: {v_raw.sum():.4g}")
    for s in a.samples:
        tot = smoothed[s].sum(axis=1)      # per-rep total, for the sum's own error
        print(f"  samples={s:<4d} smoothed sum: {tot.mean():.4g} +/- "
              f"{tot.std(ddof=1):.4g}  ({a.reps} reps)")

    # ------------------------------------------------------------------ #
    # 2. BULK CONTROL: a plain (uncherry-picked) draw from the flow's prior.
    #    If smoothing is selective, this should move little relative to its
    #    own MC scatter; if it moves as much as the leaders, the fix is not
    #    safe to deploy as-is.
    # ------------------------------------------------------------------ #
    print(f"\n=== BULK CONTROL ({a.bulk_n} plain prior draws, unfiltered) ===")
    m_bulk = _draw_prior(flow, a.bulk_n, a.seed + 7, sigma_x)
    ok = np.asarray(jnp.isfinite(m_bulk).all(-1))
    m_bulk = m_bulk[ok]
    sigma_x_bulk = jnp.tile(jnp.asarray(sigma_x), (len(m_bulk), 1))
    F_bulk = np.asarray(fprob(m_bulk))

    _, q_raw_b, r_raw_b = B.pqr_full(flow, m_bulk, sigma_x=sigma_x_bulk)
    okb = np.asarray(jnp.isfinite(q_raw_b).all(-1) & jnp.isfinite(r_raw_b).all((-1, -2)))

    # One COMMON mask across raw and every smoothed rep, so per-draw values
    # line up -- boolean-indexing each rep separately and then truncating to
    # the shortest would silently compare DIFFERENT draws across reps.
    okc = okb.copy()
    q_reps, r_reps = [], []
    for rep in range(a.reps):
        q_s, r_s = _smoothed(flow, m_bulk, cov, sigma_x_bulk, a.bulk_samples,
                             a.seed + 5000 + rep)
        okc &= np.asarray(jnp.isfinite(q_s).all(-1) & jnp.isfinite(r_s).all((-1, -2)))
        q_reps.append(q_s); r_reps.append(r_s)

    Fc = np.asarray(F_bulk)[okc]
    v_raw_common = Fc * (np.asarray(r_raw_b[:, 0, 0])[okc]
                         + np.asarray(q_raw_b[:, 0])[okc] ** 2)
    vals_b = np.stack([Fc * (np.asarray(r_s[:, 0, 0])[okc]
                             + np.asarray(q_s[:, 0])[okc] ** 2)
                       for q_s, r_s in zip(q_reps, r_reps)])   # (reps, n_common)

    print(f"  {ok.sum()}/{a.bulk_n} finite draws kept, {okb.sum()} finite raw Q/R, "
          f"{okc.sum()} common to raw and every smoothed rep")
    print(f"  raw:      sum F*v = {v_raw_common.sum():.4g}   "
          f"mean = {v_raw_common.mean():.4g}")
    tot_b = vals_b.sum(axis=1)
    print(f"  smoothed: sum F*v = {tot_b.mean():.4g} +/- {tot_b.std(ddof=1):.4g}   "
          f"mean = {(tot_b.mean() / okc.sum()):.4g}   ({a.reps} reps, "
          f"samples={a.bulk_samples})")
    delta = tot_b.mean() - v_raw_common.sum()
    print(f"  smoothed - raw = {delta:+.4g}  ({abs(delta) / (tot_b.std(ddof=1) + 1e-30):.2f}"
          f" sigma of the smoothed estimator's own scatter)")
    corr = np.corrcoef(v_raw_common, vals_b.mean(axis=0))[0, 1]
    print(f"  per-draw correlation(raw, smoothed mean) = {corr:.4f}  "
          f"(near 1: smoothing tracks raw in the bulk; near 0: it does not)")


if __name__ == "__main__":
    main()
