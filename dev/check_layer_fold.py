"""Is the centroid layer a bijection at the Sigma_X where the +0.006 lives?

HANDOFF's last surviving candidate.  `unmarginalize` (data -> base) is closed
form and is what `log_prob` uses; `marginalize` (base -> data) is a 3-pass fixed
point and is what `sample` -- and the self-consistency control's own target
generation -- uses.  ~36 of 4000 rows were seen never to converge at ANY
`n_iter`, which is the signature of a genuine FOLD (non-injectivity) rather than
slow convergence, and the `--n-iter` A/B is null against a fold of any size.

A fold fits the two shape facts the control's Sigma_X ladder produced: it needs
the layer (dead at Sigma_X = 0) and its effect can turn over as Sigma_X grows if
the density smooths faster than the fold region does.  So:

    round trip |unmarginalize(marginalize(y)) - y| on the control's own targets
    at each Sigma_X scale -- does the failing fraction trace the same bump?

Two direct fold detectors go with it, since a round-trip residual alone cannot
tell non-injectivity from a lazy iteration:

    * `n_iter = 30` against 3.  What is left is not iteration count.
    * sign and size of det d(unmarginalize)/dx.  A sign flip is a PROOF of a
      fold (the map is orientation-reversing there); a near-zero det is the
      fold's boundary, where the log-det in the importance weight blows up.

Both are run on the control's targets AND on the same targets plus one C_M draw,
because the estimator does not only evaluate the layer at the target: it
unmarginalises every kernel draw, and those reach much further out.

    python dev/check_layer_fold.py [-n 20000]
"""
import argparse
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402
from models.bijections import in_domain              # noqa: E402
from models.centroid import CentroidMarginalize      # noqa: E402

DATA = "../bfd_cnf_imsims/data"
SCALES = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]


def diagnose(layer, y, sx, n_iter):
    """Round-trip error and det of the forward map, per row.

    The iteration is `CentroidMarginalize.marginalize` written out, so `n_iter`
    is a plain argument instead of a static field needing tree surgery.
    """
    def one(yy):
        u = lambda xx: layer.unmarginalize(xx, sx)
        x = yy
        for _ in range(n_iter):
            x = yy - (u(x) - x)
        sign, logdet = jnp.linalg.slogdet(jax.jacfwd(u)(x))
        # M1, M2 pass through zero, so scale them by Mr rather than themselves.
        scale = jnp.abs(yy[jnp.array([0, 1, 1, 1, 4])])
        return jnp.max(jnp.abs(u(x) - yy) / scale), sign, logdet
    return jax.vmap(one)(y)


def report(tag, rel, sign, logdet):
    rel, sign, logdet = (np.asarray(v, np.float64) for v in (rel, sign, logdet))
    bad = ~np.isfinite(rel)
    rel = np.where(bad, np.inf, rel)
    print(f"    {tag:12s} rel p50 {np.median(rel):.2e}  p99 "
          f"{np.quantile(rel, 0.99):.2e}  max {rel.max():.2e}   "
          f"frac>1e-6 {np.mean(rel > 1e-6):.5f}  frac>1e-4 "
          f"{np.mean(rel > 1e-4):.5f}   det<=0 {np.mean(sign <= 0):.5f}  "
          f"min logdet {np.nanmin(logdet):+.3f}  nonfinite {bad.mean():.5f}")


def weight_share(flow, layer, a, sx0):
    """Does the fold carry any of Phat?

    The estimator never INVERTS the layer on a draw -- `centroid_transform` runs
    the closed-form forward map -- so a fold reaches the answer only through the
    density: on a folded branch the pushforward is missing its other preimages,
    and `unmarginalize`'s log-det is then not the whole change of variables.
    This measures the share of each target's Phat carried by draws that are
    inside the fold (det <= 0) or off the principal branch (the fixed-point
    inverse of their image does not come back to them).  Same shape of test as
    dev/check_safe_point_seam.py, which is how the mask seam was shown inert.
    """
    rows = fitsio.read(a.cov_from)[:4 * a.nw]
    rows = rows[~rows["badcenter"]][:a.nw]
    m_raw = np.asarray(rows["moments"], dtype=np.float64)
    sigma_x = np.asarray(rows["cov_odd"], dtype=np.float64) * a.wscale
    cov = bias.load_cov(a.cov_from)
    flow_g = bias.split_centroid(flow)[0]

    tb = max(1, 262144 // a.samples)
    x, log_wt = bias.mixture_draws(flow, m_raw, cov, a.samples, a.alpha, a.seed,
                                   batch=tb, sigma_x=sigma_x)
    z, ld = bias.centroid_transform(layer, x, sigma_x, to_host=False)
    ok_z = np.asarray(in_domain(z))

    def flags(xx, sx):
        u = lambda w: layer.unmarginalize(w, sx)
        sign = jnp.linalg.slogdet(jax.jacfwd(u)(xx))[0]
        y = u(xx)
        w = y
        for _ in range(30):
            w = y - (u(w) - w)
        scale = jnp.abs(xx[jnp.array([0, 1, 1, 1, 4])])
        return sign <= 0, jnp.max(jnp.abs(w - xx) / scale) > 1e-4
    fj = eqx.filter_jit(jax.vmap(jax.vmap(flags, (0, None)), (0, 0)))
    parts = [fj(jnp.asarray(x[i:i + tb]), jnp.asarray(sigma_x[i:i + tb]))
             for i in range(0, len(m_raw), tb)]
    neg = np.concatenate([np.asarray(p[0]) for p in parts])
    offbranch = np.concatenate([np.asarray(p[1]) for p in parts])

    one = eqx.filter_jit(jax.vmap(lambda zz, sx: flow_g.log_prob(
        zz, condition=bias.condition(jnp.zeros(2), sx))))
    sx_j = jnp.asarray(sigma_x, jnp.float32)
    lp = np.concatenate([np.asarray(one(z[i:i + tb], sx_j[i:i + tb]), np.float64)
                         for i in range(0, len(m_raw), tb)])
    lw = np.where(ok_z, lp + np.asarray(log_wt, np.float64)
                  + np.asarray(ld, np.float64), -np.inf)
    # A draw the peeled flow scores as NaN carries no weight either -- same
    # convention as `pqr_streamed`'s mask.
    bad = ~np.isfinite(lw) & ~np.isneginf(lw)
    print(f"    ({bad.mean():.2e} of draws non-finite in log w, zeroed)")
    lw = np.where(np.isfinite(lw), lw, -np.inf)
    w = np.where(np.isfinite(lw), np.exp(lw - lw.max(1, keepdims=True)), 0.0)
    tot = np.maximum(w.sum(1), 1e-300)
    print(f"\n  Phat weight carried by folded draws "
          f"({a.nw} deep targets x {a.samples} draws, Sigma_X x{a.wscale:g}, "
          f"alpha {a.alpha}):")
    for tag, f in (("det <= 0", neg), ("off-branch", offbranch),
                   ("either", neg | offbranch)):
        s = (w * f).sum(1) / tot
        print(f"    {tag:11s} draws {f.mean():.2e}   share of Phat: "
              f"median {np.median(s):.2e}  p90 {np.percentile(s, 90):.2e}  "
              f"max {s.max():.2e}   targets over 1e-3: {np.mean(s > 1e-3):.4f}"
              f"  over 1e-2: {np.mean(s > 1e-2):.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_deep.eqx")
    p.add_argument("--cov-from", default=f"{DATA}/targets_deep_g0_200k.fits")
    p.add_argument("--train-data", default=f"{DATA}/moments.fits")
    p.add_argument("-n", type=int, default=20000)
    p.add_argument("--g", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--nw", type=int, default=200,
                   help="targets for the Phat weight-share section (0 skips it)")
    p.add_argument("--samples", type=int, default=8192)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--wscale", type=float, default=1.0,
                   help="Sigma_X scale for the weight-share section")
    a = p.parse_args()

    m_train = shear_top.load(a.train_data)[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    # As in dev/check_float32_bias.py: x64 only AFTER the load, or the float32
    # checkpoint will not deserialise.  The round trip is 1e-4-ish, so float32
    # would be measuring its own roundoff.
    jax.config.update("jax_enable_x64", True)
    arr, static = eqx.partition(flow, eqx.is_inexact_array)
    flow = eqx.combine(jax.tree.map(lambda x: x.astype(jnp.float64), arr), static)
    # BY TYPE: bijections[0] is the chart since 16ece5c, so this SILENTLY
    # took RawMomentStandardize and called `unmarginalize` on it.
    layer = next(b for b in flow.bijection.bijection.bijections
                 if isinstance(b, CentroidMarginalize))

    sx0 = np.asarray(fitsio.read(a.cov_from)["cov_odd"][0], dtype=np.float64)
    cov = bias.load_cov(a.cov_from)
    z = flow.base_dist.sample(jr.key(a.seed), (a.n,))

    for s in SCALES:
        sx = jnp.asarray(sx0 * s)
        cond = bias.condition(jnp.asarray([a.g, 0.0]), sx)
        y = np.asarray(jax.vmap(lambda zz: flow.bijection.transform(zz, cond))(z),
                       dtype=np.float64)
        ok = np.isfinite(y).all(1) & np.asarray(in_domain(y))
        y = y[ok]
        # One C_M draw per target: where the kernel actually evaluates the layer.
        M = bias.add_noise(y, cov, a.seed + 1)
        print(f"  Sigma_X x{s:<5g}  n = {len(y)} ({int((~ok).sum())} dropped)")
        for tag, pts in (("targets", y), ("targets+C_M", M)):
            for n_iter in (3, 30):
                report(f"{tag} it{n_iter}", *diagnose(layer, jnp.asarray(pts),
                                                      sx, n_iter))

    if a.nw:
        weight_share(flow, layer, a, sx0)


if __name__ == "__main__":
    main()
