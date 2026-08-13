"""
check_flow_vs_truth.py
======================
Difference a trained flow against the EXACT density, on the gauss2 population.

Every other diagnostic in this repo can only ask whether a measured bias is
zero.  This one asks where the flow's density is wrong, because on gauss2 the
answer is known in closed form (`truth.py`): the moments were drawn from a
chosen density and the galaxy solved for, so P(m|g), Q and R have no
marginalisation in them.

Three questions, in order of how directly they bear on the m1 ramp:

1. Is the DENSITY wrong, and where?  `log flow - log truth`, binned in Mr/Mf
   and log10 Mf.  Only the SPREAD of this is meaningful, not its level: the
   population is P0 restricted to the reachable moments and renormalised, so
   truth carries an unknown additive -log Z (see truth.py) and the flow carries
   its own normalisation.  The mean is subtracted and reported separately.

2. Is the RESPONSE wrong, and where?  Flow Q and R against exact Q and R, per
   target and in the same bins.  A flow can have a good density and a bad
   dP/dg; this separates them.

3. What does that cost in m1 and c?  The bias from exact Q, R -- which must
   come out at zero to machine precision, and so tests `bias.ghat`'s
   aggregation with no flow involved at all -- against the bias from the flow's
   Q, R over the identical targets.  The difference is the flow's contribution,
   with the population's shape noise cancelled by the +g/-g pairing.

Run:
    python -m dev.check_flow_vs_truth --flow flows/shear_gauss2.eqx
"""

from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)

import bias
import bulk
import shear
import truth


def over_targets(fn, m, batch=4096):
    """Map `fn` over targets in float64, batched.

    Deliberately NOT `bias._over_targets`, which casts its inputs to float32 --
    correct there, because the production estimator runs an f32 flow on device.
    Here the flow has been promoted to f64 (see `main`), so f32 inputs would
    make the evaluation mixed-precision, and the whole point of this script is
    that the only error left is the flow's own.
    """
    out = [jax.jit(jax.vmap(fn))(jnp.asarray(m[i:i + batch], dtype=jnp.float64))
           for i in range(0, len(m), batch)]
    return jax.tree.map(lambda *a: np.concatenate([np.asarray(x) for x in a]),
                        *out) if len(out) > 1 else jax.tree.map(np.asarray, out[0])


def binned(x, y, edges, label):
    """Mean and scatter of `y` in bins of `x`, as printable rows."""
    idx = np.digitize(x, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() < 2:
            continue
        rows.append((f"{edges[b]:.3g}-{edges[b + 1]:.3g}", int(sel.sum()),
                     float(np.mean(y[sel])), float(np.std(y[sel]))))
    print(f"\n  by {label}:")
    print(f"    {'bin':>16s} {'n':>7s} {'mean':>12s} {'sd':>12s}")
    for name, n, mu, sd in rows:
        print(f"    {name:>16s} {n:7d} {mu:12.5f} {sd:12.5f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", required=True)
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--train-data", default=None,
                   help="catalog the flow's standardisation was fixed on; "
                        "defaults to the gauss2 zero-shear catalog")
    p.add_argument("--pop", default="gauss2")
    p.add_argument("--g", type=float, default=0.02)
    p.add_argument("--n-targets", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    cat = bias.CATALOGS[a.pop]
    path = lambda v: f"{a.data_dir}/{v}.fits"
    train_data = a.train_data or path(cat["zero"])

    # Rebuild the flow exactly as `shear.py train` did: RawMomentStandardize is
    # fixed by the training split, so the same 90% has to go in here.  This
    # mirrors bias.py's main, and getting it wrong silently offsets everything.
    m_train = shear.load(train_data)[0]
    m_train = m_train[:int(0.9 * len(m_train))]
    # Build and load with x64 OFF, i.e. in exactly the configuration that
    # trained and serialised the checkpoint.  `truth` turns x64 on globally, so
    # otherwise `build_flow` hands back f64 (and int64) leaves and equinox
    # refuses the f32/int32 checkpoint -- for the floats AND for the integer
    # `permutation` leaves, which is why casting only the inexact arrays is not
    # enough.  Toggling the flag is the one thing that gets every leaf right.
    jax.config.update("jax_enable_x64", False)
    try:
        flow = bulk.build_flow(jr.key(a.seed), m_train, shear=True, centroid=False)
        flow = eqx.tree_deserialise_leaves(a.flow, flow)
    finally:
        jax.config.update("jax_enable_x64", True)
    cast = lambda dt: (lambda x: x.astype(dt) if eqx.is_inexact_array(x) else x)
    # Promote the flow to float64.  `truth` enables x64 globally (it has to --
    # it is the reference), and a float32 flow under x64 is MIXED precision:
    # its f32 weights meet an f64 condition vector and promote unpredictably.
    # Making it uniformly f64 is both consistent and the right thing to
    # measure, since the question here is what the flow LEARNED, not how much
    # float32 noise its arithmetic carries.  Q and R will therefore differ
    # slightly from a production `bias.py` run, which stays in f32 on device.
    flow = jax.tree.map(cast(jnp.float64), flow)

    m0 = shear.load(path(cat["zero"]))[0][:a.n_targets]
    ratio = m0[:, 1] / m0[:, 0]
    logmf = np.log10(m0[:, 0])
    r_edges = np.quantile(ratio, np.linspace(0, 1, 9))
    f_edges = np.quantile(logmf, np.linspace(0, 1, 6))

    # --- 1. the density -----------------------------------------------------
    zero = jnp.zeros(2, dtype=jnp.float64)
    flow_lp = lambda m_i: flow.log_prob(m_i, condition=zero)
    flow_pqr = lambda m_i: (jax.grad(lambda g: flow.log_prob(m_i, condition=g))(zero),
                            jax.hessian(lambda g: flow.log_prob(m_i, condition=g))(zero))

    lp_flow = np.asarray(over_targets(flow_lp, m0))
    lp_true = np.asarray(truth.log_prob_batch(jnp.asarray(m0), jnp.zeros(2)))
    ok = np.isfinite(lp_flow) & np.isfinite(lp_true)
    d = lp_flow - lp_true
    offset = float(np.mean(d[ok]))
    print(f"\n=== 1. density: log P_flow - log P_truth  ({int(ok.sum())} targets) ===")
    print(f"  constant offset (log Z + flow normalisation): {offset:.4f}")
    print(f"  residual scatter about it:                    {np.std(d[ok]):.5f} nats")
    binned(ratio[ok], d[ok] - offset, r_edges, "Mr/Mf")
    binned(logmf[ok], d[ok] - offset, f_edges, "log10 Mf")

    # --- 2. the response ----------------------------------------------------
    q_flow, r_flow = over_targets(flow_pqr, m0)
    q_true, r_true = truth.pqr_batch(jnp.asarray(m0))
    q_true, r_true = np.asarray(q_true), np.asarray(r_true)
    fin = (np.isfinite(q_flow).all(1) & np.isfinite(q_true).all(1)
           & np.isfinite(r_flow).reshape(len(m0), -1).all(1))
    # Scale-free: Q has units of 1/g, and its magnitude varies by orders of
    # magnitude across the population, so an absolute residual would just
    # re-plot the flux distribution.
    dq = np.linalg.norm(q_flow - q_true, axis=1) / np.linalg.norm(q_true, axis=1)
    dr = (np.linalg.norm((r_flow - r_true).reshape(len(m0), -1), axis=1)
          / np.linalg.norm(r_true.reshape(len(m0), -1), axis=1))
    print(f"\n=== 2. response: |Q_flow - Q_true| / |Q_true| ===")
    print(f"  median {np.median(dq[fin]):.4f}   90th pct {np.quantile(dq[fin], 0.9):.4f}")
    binned(ratio[fin], dq[fin], r_edges, "Mr/Mf")
    print(f"\n=== 2b. curvature: |R_flow - R_true| / |R_true| ===")
    print(f"  median {np.median(dr[fin]):.4f}   90th pct {np.quantile(dr[fin], 0.9):.4f}")
    binned(ratio[fin], dr[fin], r_edges, "Mr/Mf")

    # --- 3. the bias --------------------------------------------------------
    mp = shear.load(path(cat["plus"]))[0][:a.n_targets]
    mm = shear.load(path(cat["minus"]))[0][:a.n_targets]
    qtp, rtp = truth.pqr_batch(jnp.asarray(mp))
    qtm, rtm = truth.pqr_batch(jnp.asarray(mm))
    qfp, rfp = over_targets(flow_pqr, mp)
    qfm, rfm = over_targets(flow_pqr, mm)
    sel = np.ones(len(mp), dtype=bool)
    for arr in (qtp, qtm, qfp, qfm):
        sel &= np.isfinite(np.asarray(arr)).all(1)

    b_true = bias.bias(np.asarray(qtp), np.asarray(rtp),
                       np.asarray(qtm), np.asarray(rtm), a.g, sel)
    b_flow = bias.bias(qfp, rfp, qfm, rfm, a.g, sel)
    print(f"\n=== 3. bias over {int(sel.sum())} paired targets, g = {a.g} ===")
    print(f"  from EXACT Q,R : m1 = {b_true[0]:+.3e}  c1 = {b_true[1]:+.3e}  "
          f"c2 = {b_true[2]:+.3e}   <- must be ~0; tests the aggregation alone")
    print(f"  from FLOW  Q,R : m1 = {b_flow[0]:+.3e}  c1 = {b_flow[1]:+.3e}  "
          f"c2 = {b_flow[2]:+.3e}   <- the flow's own contribution")


if __name__ == "__main__":
    main()
