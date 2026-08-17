"""Three Sigma_X-shaped candidates for the centroid control's bump, measured.

Everything here is DERIVED OR MEASURED from the flow and the catalogs.  Nothing
is quoted from a comment -- the margins, the layer's push and the scale at which
they cross are all recomputed, because the previous pass leaned on
`centroid.py`'s "0.34% / 1.2%" comment and that is exactly how one confuses
oneself.

The object to explain (HANDOFF 2026-08-17): with the debiased estimator the
centroid self-consistency control reads +0.0019 at Sigma_X x0, rises to +0.0095
at x0.3, and falls to +0.0031 by x3.  A bump.  `--sigma-x-scale` moves Sigma_X
while C_M, the chart and the flow stay fixed, so the ladder scans a RATIO, and a
peak means the layer's displacement crosses a fixed scale near x0.3.

PART 1  How close does the population actually sit to each chart ceiling?
        u = Mr/(r* Mf) and v = Mc/(rc* Mr) both live in (0,1); the margin is
        1 - u and 1 - v.  Measured on the deep targets and on flow draws.

PART 2  How hard does the centroid layer push, per unit Sigma_X?
        `CentroidMarginalize.transform` is the data -> base ("un-marginalise")
        direction, and un-marginalising moves Mr/Mf and Mc/Mr UP.  Measured as
        the fractional change in u and v at each Sigma_X scale.

        PART 1 / PART 2 then GIVE the crossing scale, rather than assuming it.

PART 3  The seam.  `bias.log_conv_is` masks draws with `in_domain`, evaluated on
        the post-centroid but PRE-SHEAR moments -- while the chart itself lives
        in `RawMomentStandardize`, which the measured chain puts AFTER
        `ShearResponse` in the data -> base order.  So the quantity that has to
        be on-chart is `ShearResponse.transform(x, g)`, not `x`, and the
        difference is g-DEPENDENT.  Measured: how much of each target's Phat
        sits on draws where the two disagree, as a function of Sigma_X.

PART 4  The proposal's density.  `bias._mixture_chunk` draws its flow component
        with `flow.sample` (base -> data, which runs the centroid layer's 3-pass
        fixed-point `marginalize`) and then evaluates that draw's proposal
        density with `flow.log_prob` (data -> base, the exact closed-form
        `unmarginalize`).  Those agree only if the two maps are exact inverses.
        Where they are not, `log_q` is not the density the draw came from, the
        importance weights are wrong, and the estimator is biased in a way that
        is DRAW-COUNT INDEPENDENT -- so nothing the jackknife fixed touches it.

        Measured directly, without needing the fixed point to be inverted: draw
        z from the base, push it through `transform_and_log_det` to get x and
        log|det dx/dz|, so the TRUE log-density of x is
        `base.log_prob(z) - logdet`, and compare with `flow.log_prob(x)`.

    python dev/check_ceiling_and_proposal.py [-n 400] [--samples 2048]
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

import bias                                                    # noqa: E402
import bulk                                                    # noqa: E402
import shear as shear_top                                      # noqa: E402
from models.bijections import (POINT_SOURCE, POINT_SOURCE_MC,  # noqa: E402
                               in_domain)

DATA = "../bfd_cnf_imsims/data"
SCALES = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)


def uv(m):
    """The two chart coordinates that carry ceilings, both in (0, 1)."""
    u = m[..., 1] / (POINT_SOURCE * m[..., 0])
    v = m[..., 4] / (POINT_SOURCE_MC * m[..., 1])
    return u, v


def pct(x, q):
    return np.percentile(np.asarray(x, np.float64), q)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_deep.eqx")
    p.add_argument("--cov-from", default=f"{DATA}/targets_deep_g0_200k.fits")
    p.add_argument("--train-data", default=f"{DATA}/moments.fits")
    p.add_argument("-n", type=int, default=400)
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--g", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    m_train = shear_top.load(a.train_data)[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    chain = flow.bijection.bijection.bijections
    layer, shear_layer = chain[0], chain[1]
    print(f"chain[0] = {type(layer).__name__}, "
          f"chain[1] = {type(shear_layer).__name__}   "
          f"(data -> base order; the chart is chain[2], AFTER shear)")

    sx1 = np.asarray(fitsio.read(a.cov_from)["cov_odd"][:1], dtype=np.float64)
    cov = bias.load_cov(a.cov_from)
    tgt = np.asarray(fitsio.read(a.cov_from)["moments"][:20000], dtype=np.float64)

    # ---------------------------------------------------------------- PART 1
    print("\n=== PART 1: measured margin to each chart ceiling ===")
    print(f"  ceilings: POINT_SOURCE = {POINT_SOURCE}, "
          f"POINT_SOURCE_MC = {POINT_SOURCE_MC}")
    for name, m in (("deep targets (noisy)", tgt),):
        ok = np.asarray(in_domain(jnp.asarray(m)))
        u, v = uv(m[ok])
        print(f"  {name}: {ok.sum()} in-domain of {len(m)}")
        for lab, x in (("u = Mr/(r* Mf)", u), ("v = Mc/(rc* Mr)", v)):
            print(f"    {lab}:  median {np.median(x):.4f}  p99 {pct(x,99):.4f}  "
                  f"max {x.max():.6f}   -> margin 1-max = {1-x.max():.5f}")

    z = flow.base_dist.sample(jr.key(a.seed), (20000,))
    cond0 = bias.condition(jnp.zeros(2), jnp.asarray(sx1[0]))
    lat = np.asarray(jax.vmap(lambda zz: flow.bijection.transform(zz, cond0))(z),
                     np.float64)
    lok = np.asarray(in_domain(jnp.asarray(lat)))
    ul, vl = uv(lat[lok])
    print(f"  flow draws at the deep Sigma_X: {lok.sum()} in-domain of 20000")
    for lab, x in (("u", ul), ("v", vl)):
        print(f"    {lab}:  median {np.median(x):.4f}  p99 {pct(x,99):.4f}  "
              f"max {x.max():.6f}   -> margin 1-max = {1-x.max():.5f}")

    # ---------------------------------------------------------------- PART 2
    print("\n=== PART 2: measured push of the centroid layer, per Sigma_X ===")
    print("  un-marginalise the SAME latent moments at each scale and measure")
    print("  the fractional rise in u and v (positive = toward the ceiling)")
    base = lat[lok][:5000]
    ub, vb = uv(base)
    push = {}
    for s in SCALES:
        sxs = np.repeat(sx1 * s, len(base), axis=0)
        c = jax.vmap(bias.condition, in_axes=(None, 0))(
            jnp.zeros(2), jnp.asarray(sxs, dtype=jnp.float32))
        out = np.asarray(jax.vmap(layer.transform)(
            jnp.asarray(base, jnp.float32), c), np.float64)
        uo, vo = uv(out)
        du, dv = np.median(uo / ub - 1.0), np.median(vo / vb - 1.0)
        push[s] = (du, dv)
        print(f"    x{s:<5}  median du/u = {du:+.5f}   median dv/v = {dv:+.5f}"
              f"   frac off-chart after push = "
              f"{1 - float(np.mean(np.asarray(in_domain(jnp.asarray(out))))):.5f}")

    print("\n  DERIVED crossing scale (push == margin), from the numbers above:")
    for lab, i, marg in (("u", 0, 1 - ul.max()), ("v", 1, 1 - vl.max())):
        p1 = push[1.0][i]
        if p1 > 0:
            print(f"    {lab}: margin {marg:.5f} / push-at-x1 {p1:.5f} "
                  f"-> crossing at Sigma_X x{marg / p1:.3f}")
        else:
            print(f"    {lab}: push at x1 is {p1:+.5f} (not toward the ceiling)")

    # -------------------------------------------------------------- PART 3, 4
    print(f"\n=== PART 3/4: per-Sigma_X seam and proposal-density scan "
          f"({a.n} targets, S = {a.samples}, alpha = {a.alpha}) ===")
    m_t = tgt[:a.n]
    hdr = (f"  {'Sigma_X':>8} {'masked':>9} {'shearflip':>10} {'flipPhat':>10} "
           f"{'|dlogq|p50':>11} {'|dlogq|p99':>11} {'dlogq*w':>10}")
    print(hdr)
    for s in SCALES:
        sxs = np.repeat(sx1 * s, a.n, axis=0)
        d, lw = bias.mixture_draws(flow, m_t, cov, a.samples, a.alpha,
                                   a.seed + 100, sigma_x=sxs)
        # peel: the estimator evaluates the peeled flow on these
        pk, ld = bias.centroid_transform(layer, d, sxs)
        pk = np.asarray(pk, np.float64)
        lw = np.asarray(lw, np.float64) + np.asarray(ld, np.float64)

        ok_pre = np.asarray(in_domain(jnp.asarray(pk)))
        # what the chart actually sees is the POST-shear point
        cg = jax.vmap(bias.condition, in_axes=(None, 0))(
            jnp.array([a.g, 0.0]), jnp.asarray(sxs, dtype=jnp.float32))
        cgb = jnp.broadcast_to(cg[:, None, :], pk.shape[:-1] + cg.shape[-1:])
        post = jax.vmap(jax.vmap(shear_layer.transform))(
            jnp.asarray(pk, jnp.float32), cgb)
        ok_post = np.asarray(in_domain(post))
        flip = ok_pre & ~ok_post          # counted by the mask, rejected by the chart

        # Phat weight share, using the estimator's own weights at g = 0
        rest, _ = bias.split_centroid(flow)
        c0 = jax.vmap(bias.condition, in_axes=(None, 0))(
            jnp.zeros(2), jnp.asarray(sxs, dtype=jnp.float32))
        c0b = jnp.broadcast_to(c0[:, None, :], pk.shape[:-1] + c0.shape[-1:])
        dummy = np.asarray(bias.safe_point(jnp.asarray(m_t)), np.float64)
        lp = np.asarray(rest.log_prob(jnp.where(ok_pre[..., None], pk,
                                                jnp.asarray(dummy)[:, None, :]),
                                      condition=c0b), np.float64)
        lwt = np.where(ok_pre, lp + lw, -np.inf)
        mx = lwt.max(1, keepdims=True)
        w = np.where(np.isfinite(lwt), np.exp(lwt - mx), 0.0)
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
        flip_share = float(np.mean((w * flip).sum(1)))

        # ---- PART 4 on the same Sigma_X: sampler density vs log_prob density
        nz = 4000
        zz = flow.base_dist.sample(jr.key(a.seed + 1), (nz,))
        cz = bias.condition(jnp.zeros(2), jnp.asarray(sx1[0] * s))
        xz, ldf = jax.vmap(lambda q: flow.bijection.transform_and_log_det(q, cz))(zz)
        true_lp = np.asarray(flow.base_dist.log_prob(zz), np.float64) \
            - np.asarray(ldf, np.float64)
        dom = np.asarray(in_domain(xz))
        assumed = np.asarray(flow.log_prob(xz, condition=cz), np.float64)
        dl = np.abs(assumed - true_lp)[dom & np.isfinite(assumed)]

        print(f"  {'x'+str(s):>8} {1-ok_pre.mean():9.5f} {flip.mean():10.6f} "
              f"{flip_share:10.6f} {np.median(dl):11.3e} {pct(dl,99):11.3e} "
              f"{np.mean(dl):10.3e}")

    print("\n  masked     = fraction of draws in_domain() rejects (pre-shear)")
    print("  shearflip  = fraction in-domain PRE-shear but off-chart POST-shear")
    print("  flipPhat   = share of each target's Phat carried by those, averaged")
    print("  |dlogq|    = |flow.log_prob(x) - true log-density of x| for flow draws")
    print("\n  candidate 1 predicts flipPhat / masked peaks near x0.3;")
    print("  candidate 3 predicts |dlogq| is large and rises with Sigma_X.")


if __name__ == "__main__":
    main()
