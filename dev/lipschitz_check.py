"""How much do Q, R (the score-estimator's per-draw response, which is what
selection_terms_score integrates) actually change over a small step in moment
space -- near the known R_s leader versus between ordinary real templates in
the same locus?

Answers: is the leader's local sensitivity (|Q,R change| / |moment-space
step| to its nearest real template) genuinely anomalous relative to how much
Q, R normally vary between adjacent real templates in this Mr/Mf~3.0-3.3
band -- and is the real-to-real distribution's own tail already large enough
(from the legitimate point-source-ceiling steepness) that a flat Lipschitz
cap couldn't tell the two apart.

Run: python dev/lipschitz_check.py
"""
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402

LEADER = np.array([10001.8, 31896.3, -5789.59, 436.861, 195488.0])
POP = "gauss2_v3d"
FLOW = "flows/shear_g2v3d_full2.eqx"
DATA_DIR = "../bfd_cnf_imsims/data"
N_REAL_SAMPLE = 400


def main():
    m_train = shear.load(f"{DATA_DIR}/{B.TRAIN_DATA[POP]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=False)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)

    zero, e0, e1 = jnp.zeros(2), jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])

    def one(m_i):
        f = lambda g: flow.log_prob(m_i, condition=B.condition(g, None))
        vg = jax.value_and_grad(f)
        (_, q), lin = jax.linearize(vg, zero)
        _, h0 = lin(e0)
        _, h1 = lin(e1)
        return q, jnp.stack([h0, h1], axis=-1)

    one_jit = eqx.filter_jit(one)

    def qr_vec(m_i):
        q, r = one_jit(jnp.asarray(m_i))
        return np.concatenate([np.asarray(q).ravel(), np.asarray(r).ravel()])

    # Standardised shape-invariant space, same convention as
    # build_support_density, for the nearest-neighbour distance metric.
    inv = B._shape_invariants(m_train)
    ok = np.isfinite(inv).all(-1)
    inv, m_train = inv[ok], np.asarray(m_train)[ok]
    scale = inv.std(0)
    tree = cKDTree(inv / scale)

    # --- the leader vs its nearest REAL template ---
    leader_inv = B._shape_invariants(LEADER[None, :])[0] / scale
    d_leader, j_leader = tree.query(leader_inv, k=1)
    neighbor = m_train[j_leader]
    qr_leader, qr_neighbor = qr_vec(LEADER), qr_vec(neighbor)
    dq_lead = np.linalg.norm(qr_leader - qr_neighbor)
    lip_leader = dq_lead / d_leader
    print(f"leader -> nearest real template: distance {d_leader:.4f}  "
          f"|Q,R change| {dq_lead:.6g}  Lipschitz ratio {lip_leader:.6g}")

    # --- real-to-real baseline in the same Mr/Mf band ---
    ratio = m_train[:, 1] / m_train[:, 0]
    band = (ratio > 2.7) & (ratio < 3.5)
    idx_band = np.where(band)[0]
    rng = np.random.default_rng(0)
    sample = rng.choice(idx_band, size=min(N_REAL_SAMPLE, len(idx_band)),
                        replace=False)

    lips, dists = [], []
    for i in sample:
        d2, j2 = tree.query(inv[i] / scale, k=2)   # k=0 is self
        d, j = d2[1], j2[1]
        qa, qb = qr_vec(m_train[i]), qr_vec(m_train[j])
        dq = np.linalg.norm(qa - qb)
        lips.append(dq / d)
        dists.append(d)
    lips = np.array(lips)

    print(f"\nreal-to-real baseline, {len(lips)} pairs in Mr/Mf(2.7,3.5) band:")
    print(f"  Lipschitz ratio: p50 {np.percentile(lips,50):.4g}  "
          f"p90 {np.percentile(lips,90):.4g}  p99 {np.percentile(lips,99):.4g}  "
          f"max {lips.max():.4g}")
    print(f"\nleader's ratio ({lip_leader:.4g}) is "
          f"{lip_leader/np.median(lips):.1f}x the real-to-real median, "
          f"{lip_leader/lips.max():.2f}x the real-to-real MAX")
    pct = 100.0 * np.mean(lips < lip_leader)
    print(f"leader's ratio sits at/above the {pct:.2f} percentile of "
          "real-to-real local sensitivity")


if __name__ == "__main__":
    main()
