"""Overfitting predicts: local Q/R steepness (Lipschitz ratio) is high
specifically where training data is SPARSE (the flow was weakly constrained
there). A structural/architectural cause (log-det Jacobian reconciling a
smooth base with the target geometry) predicts no such correlation --
steepness shows up independent of local data density.

Tests this directly: for real templates in the leader's Mr/Mf band, correlate
each point's local training density (k-NN distance in shape-invariant space,
same metric as build_support_density) against its local Lipschitz ratio
(from lipschitz_check.py's definition). Also reports where the leader itself
and its nearest real template sit on the density axis.
"""
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

sys.path.insert(0, ".")
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402

LEADER = np.array([10001.8, 31896.3, -5789.59, 436.861, 195488.0])
POP = "gauss2_v3d"
FLOW = "flows/shear_g2v3d_full2.eqx"
DATA_DIR = "../bfd_cnf_imsims/data"
N_SAMPLE = 600


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

    inv = B._shape_invariants(m_train)
    ok = np.isfinite(inv).all(-1)
    inv, m_train = inv[ok], np.asarray(m_train)[ok]
    scale = inv.std(0)
    tree = cKDTree(inv / scale)

    ratio = m_train[:, 1] / m_train[:, 0]
    band = (ratio > 2.7) & (ratio < 3.5)
    idx_band = np.where(band)[0]
    rng = np.random.default_rng(0)
    sample = rng.choice(idx_band, size=min(N_SAMPLE, len(idx_band)), replace=False)

    # Local training density proxy: distance to the 10th-nearest OTHER real
    # template (k=10, matches build_support_density's k). Larger = sparser.
    density_dist, _ = tree.query(inv[sample] / scale, k=11)
    density_dist = density_dist[:, -1]   # 10th neighbour, excluding self

    lips = []
    for i in sample:
        d2, j2 = tree.query(inv[i] / scale, k=2)
        d, j = d2[1], j2[1]
        qa, qb = qr_vec(m_train[i]), qr_vec(m_train[j])
        lips.append(np.linalg.norm(qa - qb) / d)
    lips = np.array(lips)

    rho, p = spearmanr(density_dist, lips)
    print(f"n={len(sample)} real templates in Mr/Mf(2.7,3.5) band")
    print(f"Spearman corr(local sparsity, Lipschitz ratio) = {rho:+.4f}  (p={p:.3g})")
    print("  positive & significant -> overfitting-consistent (steep where data is sparse)")
    print("  near zero / not significant -> structural (steep independent of local density)")

    # Where does the leader's own density sit relative to the band?
    leader_inv = B._shape_invariants(LEADER[None, :])[0] / scale
    d_leader10, _ = tree.query(leader_inv, k=11)
    d_leader10 = d_leader10[-1]
    pct = 100.0 * np.mean(density_dist < d_leader10)
    print(f"\nleader's local 10-NN distance: {d_leader10:.4f} "
          f"(band's {pct:.1f} percentile of sparsity -- "
          f"{'SPARSE' if pct > 80 else 'DENSE' if pct < 20 else 'TYPICAL'} spot)")
    print(f"band 10-NN distance: p10={np.percentile(density_dist,10):.4f} "
          f"p50={np.percentile(density_dist,50):.4f} "
          f"p90={np.percentile(density_dist,90):.4f}")


if __name__ == "__main__":
    main()
