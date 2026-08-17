"""Is the centroid control's +0.006 a float32 cancellation in Q and R?

`pqr_streamed` merges chunks in float64 "because `c_over_a` is rebuilt as
`hess + (B/A)(B/A)^T` and `hess` was itself computed as `C/A - (B/A)(B/A)^T`, so
at low ESS that is a cancellation undone" -- and its own table puts the float32
merge at 1.7% on R.  But only the MERGE was moved to float64: the per-chunk
`f, df, d2f` still come out of the jitted flow in float32.

A float32 error there would be draw-count independent, g-independent, sharper
where the density is sharper, and worse with the centroid layer's extra
curvature -- which is the (layer activity) x (density sharpness) product the
Sigma_X ladder traced out: 0 at Sigma_X = 0, +0.007 at x0.3-1, back to 0 at x3.

This runs the SAME targets and the SAME draws through the identical estimator
twice, once in float32 and once in float64, in ONE chunk so no merge is
involved.  The draw VALUES are shared, so the comparison is exactly paired and
the difference needs no bootstrap: any shift is precision, full stop.

    python dev/check_float32_bias.py [-n 1000] [--samples 4096]
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

import bias                                            # noqa: E402
import bulk                                            # noqa: E402
import shear as shear_top                              # noqa: E402
from models.bijections import in_domain                # noqa: E402

DATA = "../bfd_cnf_imsims/data"


def qr_at(flow_g, m_z, draws, lw, sigma_x, dtype):
    """`pqr_streamed.one`, verbatim, at a chosen precision and in one chunk."""
    cast = lambda x: jnp.asarray(x, dtype=dtype)
    zero = jnp.zeros(2, dtype=dtype)

    def one(m_i, d_i, lw_i, sx_i):
        f = lambda g: bias.log_conv_is(flow_g, m_i, d_i, lw_i,
                                       bias.condition(g, sx_i))
        vg = jax.value_and_grad(f)
        (_, df), (_, h0) = jax.jvp(vg, (zero,), (jnp.array([1.0, 0.0], dtype),))
        _, (_, h1) = jax.jvp(vg, (zero,), (jnp.array([0.0, 1.0], dtype),))
        return df, jnp.stack([h0, h1], axis=-1)

    return eqx.filter_jit(jax.vmap(one))(
        cast(m_z), cast(draws), cast(lw), cast(sigma_x))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_deep.eqx")
    p.add_argument("--cov-from", default=f"{DATA}/targets_deep_g0_200k.fits")
    p.add_argument("--train-data", default=f"{DATA}/moments.fits")
    p.add_argument("-n", type=int, default=1000)
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--g", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    m_train = shear_top.load(a.train_data)[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    # x64 goes on only AFTER the load: `build_flow` would otherwise construct
    # float64 leaves and the float32 checkpoint on disk would not deserialise.
    jax.config.update("jax_enable_x64", True)
    # float64 params for the fp64 arm; the fp32 arm casts back down inside qr_at.
    arr, static = eqx.partition(flow, eqx.is_inexact_array)
    flow64 = eqx.combine(jax.tree.map(lambda x: x.astype(jnp.float64), arr), static)
    # BOTH flows, not just both input dtypes: casting the inputs alone leaves
    # float64 parameters to promote every op straight back up.
    flow_g64, layer = bias.split_centroid(flow64)
    flow_g32 = bias.split_centroid(flow)[0]
    peeled = {jnp.float32: flow_g32, jnp.float64: flow_g64}

    sigma_x = np.repeat(np.asarray(fitsio.read(a.cov_from)["cov_odd"][:1],
                                   dtype=np.float64), a.n, axis=0)
    cov = bias.load_cov(a.cov_from)

    # The control's targets, exactly as check_selfconsistency_noisy builds them.
    z = flow64.base_dist.sample(jr.key(a.seed), (a.n,))
    cond = lambda g: bias.condition(jnp.asarray(g), jnp.asarray(sigma_x[0]))
    draw = lambda g: np.asarray(
        jax.vmap(lambda zz: flow64.bijection.transform(zz, cond(g)))(z),
        dtype=np.float64)
    mp, mm = draw([a.g, 0.0]), draw([-a.g, 0.0])
    ok = (np.isfinite(mp).all(1) & np.isfinite(mm).all(1)
          & np.asarray(in_domain(mp) & in_domain(mm)))
    mp, mm, sigma_x = mp[ok], mm[ok], sigma_x[ok]
    print(f"{int((~ok).sum())} of {a.n} draws dropped")

    out = {}
    for name, m in (("plus", bias.add_noise(mp, cov, a.seed + 1)),
                    ("minus", bias.add_noise(mm, cov, a.seed + 1))):
        # Draws and weights are generated ONCE and shared by both precisions --
        # that is the pairing.  One chunk, so no merge enters the comparison.
        d, lw = bias.mixture_draws(flow64, m, cov, a.samples, a.alpha,
                                   a.seed + 100, batch=a.batch, sigma_x=sigma_x)
        dz, ld = bias.centroid_transform(layer, d, sigma_x)
        m_z = bias.centroid_transform(layer, m, sigma_x)[0]
        lw = np.asarray(lw, np.float64) + np.asarray(ld, np.float64)
        for dt in (jnp.float32, jnp.float64):
            q, r = [], []
            for i in range(0, len(m), a.batch):
                s = slice(i, i + a.batch)
                qq, rr = qr_at(peeled[dt], m_z[s], dz[s], lw[s], sigma_x[s], dt)
                q.append(np.asarray(qq, np.float64))
                r.append(np.asarray(rr, np.float64))
            out[name, dt.__name__] = (np.concatenate(q), np.concatenate(r))
        print(f"  {name} arm done")

    fin = np.ones(len(mp), bool)
    for q, r in out.values():
        fin &= np.isfinite(q).all(1) & np.isfinite(r).reshape(len(r), -1).all(1)
    print(f"  {int(fin.sum())} of {len(mp)} finite in all four passes")

    for k in ("plus", "minus"):
        q32, r32 = out[k, "float32"]
        q64, r64 = out[k, "float64"]
        dq = np.abs(q32 - q64)[fin] / np.maximum(np.abs(q64)[fin], 1e-300)
        rn = np.linalg.norm(r64.reshape(len(r64), -1), axis=1)[fin]
        dr = np.linalg.norm((r32 - r64).reshape(len(r32), -1), axis=1)[fin] / rn
        print(f"  {k}: |dQ|/|Q| median {np.median(dq):.2e} p99 "
              f"{np.percentile(dq, 99):.2e} | |dR|/|R| median {np.median(dr):.2e} "
              f"p99 {np.percentile(dr, 99):.2e}")

    got = {}
    for dt in ("float32", "float64"):
        qp, rp = out["plus", dt]
        qm, rm = out["minus", dt]
        got[dt] = bias.bias(qp[fin], rp[fin], qm[fin], rm[fin], a.g)[0]
        print(f"  m1 ({dt:>7}) = {got[dt]:+.8f}")
    print(f"  m1 shift from precision = {got['float32'] - got['float64']:+.2e}  "
          f"(the +0.006 to explain is 6e-3)")
    print("\n  the two m1 share targets AND draws, so their difference is "
          "precision alone")


if __name__ == "__main__":
    main()
