"""Fast proxy for "did a candidate bulk.py fix reduce the flow's worst
training-artifact spike": draw N unconditional samples from a checkpoint and
compare the flow's own (Q, R) against `truth.pqr_sigma`'s EXACT analytic
(Q, R) at the same point (g = 0).  Seconds, not the full 16M-draw health
gate -- meant to be run after every candidate fix, not instead of the gate.

ONLY meaningful for gauss2-family populations: `truth.py`'s analytic model
(`theta_of_m`, `log_p_theta`, ...) assumes the gauss2 generative model
specifically, so a non-gauss2 `--pop` has no truth to compare against.

    python dev/flow_vs_truth_check.py --flow flows/shear_g2v4_jac5.eqx
"""
from __future__ import annotations

import argparse
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402
# NOT imported at module scope: `truth.py` flips `jax_enable_x64` on AT IMPORT
# TIME (its own comment: "exactness is the whole point"), and checkpoints on
# disk are float32 -- `eqx.tree_deserialise_leaves` below needs the template
# flow's dtype to match the file's, so the flow has to be built and loaded
# BEFORE x64 is turned on, exactly as `bias.py`'s own `import truth` (inside
# its function, after the real flow has already been loaded) does.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", required=True)
    p.add_argument("--pop", default="gauss2_v4")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--stage", choices=["bulk", "shear", "centroid"],
                   default="shear")
    p.add_argument("--log2-draws", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=10,
                   help="how many worst-ratio draws to print")
    a = p.parse_args()

    m_train = shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=a.stage != "bulk",
                           centroid=a.stage == "centroid")
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    import truth  # noqa: E402 -- see the top-of-file comment on WHY this is lazy

    if a.stage == "centroid":
        cat = B.CATALOGS[a.pop]
        targets = f"{a.data_dir}/{cat['zero']}.fits"
        sigma_x = jnp.asarray(
            np.asarray(fitsio.read(targets)["cov_odd"], dtype=np.float64)[0])
    else:
        sigma_x = jnp.zeros(3)
    flow_sigma_x = sigma_x if a.stage == "centroid" else None
    print(f"flow {a.flow}   pop {a.pop}   stage {a.stage}")
    print(f"Sigma_X = {np.asarray(sigma_x)}")

    zero, e0, e1 = jnp.zeros(2), jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])

    def one(m_i):
        f = lambda g: flow.log_prob(m_i, condition=B.condition(g, flow_sigma_x))
        vg = jax.value_and_grad(f)
        (_, q), lin = jax.linearize(vg, zero)
        _, h0 = lin(e0)
        _, h1 = lin(e1)
        return q, jnp.stack([h0, h1], axis=-1)

    flow_qr = eqx.filter_jit(jax.vmap(one))
    truth_qr = eqx.filter_jit(jax.vmap(lambda m_i: truth.pqr_sigma(m_i, sigma_x)))

    n = 1 << a.log2_draws
    print(f"{n} draws (2^{a.log2_draws}), seed {a.seed}", flush=True)
    zs = flow.base_dist.sample(jr.key(a.seed + 31), (n,)).astype(jnp.float64)
    tr = eqx.filter_jit(jax.vmap(lambda z1: flow.bijection.transform(
        z1, B.condition(jnp.zeros(2), flow_sigma_x))))
    m0 = np.concatenate([np.asarray(tr(zs[i:i + 16384]))
                         for i in range(0, n, 16384)])
    del zs

    # Q/R here cost a `jax.hessian`/`jax.linearize` per point (unlike the
    # plain transform above), so batch it -- an unchunked call over 2^16+
    # draws at once is what OOM'd a 16 GB card. 4096 matches
    # `dev/leader_corner_plot.py`'s default `--batch`.
    # 4096 OOM'd at the centroid stage with a multi-scale SigmaXBlockLayer
    # (2026-09-22, g2v4n: 37 GiB requested on a 16 GB card) -- that layer's
    # per-point Hessian is heavier than the shear-only flows this default was
    # tuned against, so shrink it here rather than raise a CLI flag nobody
    # will remember to pass at the stage that needs it.
    QR_CHUNK = 512
    qf_l, rf_l, qt_l, rt_l = [], [], [], []
    for i in range(0, n, QR_CHUNK):
        mm = jnp.asarray(m0[i:i + QR_CHUNK])
        qf, rf = flow_qr(mm)
        qt, rt = truth_qr(mm)
        qf_l.append(np.asarray(qf)); rf_l.append(np.asarray(rf))
        qt_l.append(np.asarray(qt)); rt_l.append(np.asarray(rt))
    q_f, r_f = np.concatenate(qf_l), np.concatenate(rf_l)
    q_t, r_t = np.concatenate(qt_l), np.concatenate(rt_l)

    ok = (np.isfinite(q_f).all(-1) & np.isfinite(r_f).reshape(n, -1).all(-1) &
          np.isfinite(q_t).all(-1) & np.isfinite(r_t).reshape(n, -1).all(-1))
    n_drop = n - int(ok.sum())
    print(f"dropped {n_drop}/{n} non-finite draws (flow or truth side)")

    m0, r_f, r_t = m0[ok], r_f[ok], r_t[ok]
    r11_f, r11_t = r_f[:, 0, 0], r_t[:, 0, 0]
    # r11_t == 0 is a measure-zero event for a continuous analytic model, but
    # guard the divide anyway rather than trust that on a finite draw set.
    ratio = np.where(r11_t != 0, np.abs(r11_f / r11_t), np.inf)

    order = np.argsort(ratio)[::-1][:a.top]
    t = bulk.to_coords(m0[order])
    print(f"\nworst {a.top} |R11_flow / R11_truth| draws "
          f"(Mf, Mr/Mf, M1/Mr, M2/Mr):")
    for k, i in enumerate(order):
        print(f"  ratio={ratio[i]:10.4g}   Mf={m0[i,0]:10.4g}  "
              f"Mr/Mf={t[k,1]:7.4f}  M1/Mr={t[k,3]:+7.4f}  "
              f"M2/Mr={t[k,4]:+7.4f}")

    pct = [1, 5, 25, 50, 75, 95, 99, 100]
    vals = np.percentile(ratio[np.isfinite(ratio)], pct)
    print(f"\n|R11_flow / R11_truth| percentiles over {len(ratio)} kept draws:")
    for p_, v in zip(pct, vals):
        print(f"  p{p_:>3d}  {v:.4g}")


if __name__ == "__main__":
    main()
