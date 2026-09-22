"""Is the leader's bend-layer net_out outlier (`bend_layer_check.py`: 100th
percentile of 300 in-band controls) explained by local training-data
sparsity in the net's OWN 4-d input space -- [z0, z1, z2, log1p(q)-W0], the
exact vector `Spin2CouplingLayer._h` conditions on -- rather than the 2-3d
shape-invariant space `build_support_density`/`lipschitz_vs_density.py`
already checked (and found NOT explanatory: leader was steep despite normal
density there, ruling out overfitting in that space).

Mirrors `lipschitz_vs_density.py`'s method (Spearman rho of local k-NN
density against the steepness metric) but in the net's actual input space,
which is the more direct test of "is this an extrapolation gap."

Run: python dev/bend_conditioner_density_check.py
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
N_SAMPLE = 3000
K_NN = 10
_W0 = 1.0986122886681098


def build():
    m_train = shear.load(f"{DATA_DIR}/{B.TRAIN_DATA[POP]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=False)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    return flow, np.asarray(m_train)


def conditioner(bijections, m_i):
    """u = [z0, z1, z2, log1p(q)-W0], the bend net's actual 4-d input."""
    x = jnp.asarray(m_i)
    for b in bijections[:2]:            # RawMomentStandardize, ShearResponse
        c = jnp.zeros(2) if b.cond_shape is not None else None
        x = b.transform_and_log_det(x, c)[0]
    x = bijections[2].spin0.transform_and_log_det(x)[0]
    q = x[3] ** 2 + x[4] ** 2
    return jnp.concatenate([x[:3], (jnp.log1p(q) - _W0)[None]])


def main():
    flow, m_train = build()
    bijections = flow.bijection.bijection.bijections
    net = bijections[2].spin2.net

    ratio = m_train[:, 1] / m_train[:, 0]
    band = (ratio > 2.7) & (ratio < 3.5)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(band)[0], size=N_SAMPLE, replace=False)

    cond_jit = eqx.filter_jit(lambda m: conditioner(bijections, m))
    U = np.stack([np.asarray(cond_jit(m_train[i])) for i in idx])
    u_leader = np.asarray(cond_jit(LEADER))

    net_jit = eqx.filter_jit(lambda u: net(u)[0])
    net_out = np.array([float(net_jit(u)) for u in U])
    net_out_leader = float(net_jit(u_leader))

    scale = U.std(0)
    tree = cKDTree(U / scale)
    d10, _ = tree.query(U / scale, k=K_NN + 1)
    density = d10[:, -1]                        # distance to 10th neighbor
    d_leader, _ = tree.query(u_leader / scale, k=K_NN)
    density_leader = d_leader[-1]

    rho, p = spearmanr(density, np.abs(net_out))
    print(f"Spearman(local 10-NN distance, |net_out|) = {rho:.3f} (p={p:.2g}) "
          f"across {N_SAMPLE} in-band real templates")
    print(f"  (negative would mean net_out grows in DENSE regions, like the "
          f"Lipschitz-vs-density finding -- positive/near-zero means sparse "
          f"regions are where net_out is largest, i.e. classic extrapolation)")

    dpct = 100.0 * np.mean(density < density_leader)
    npct = 100.0 * np.mean(np.abs(net_out) < abs(net_out_leader))
    print(f"\nleader: local 10-NN distance = {density_leader:.4g} "
          f"(percentile {dpct:.1f} of {N_SAMPLE} real templates' own density)")
    print(f"leader: |net_out| = {abs(net_out_leader):.4g} "
          f"(percentile {npct:.1f})")
    print(f"real templates' |net_out|: p50={np.percentile(np.abs(net_out),50):.4g} "
          f"p99={np.percentile(np.abs(net_out),99):.4g} "
          f"max={np.abs(net_out).max():.4g}")

    # Conditional check: among real templates at density >= leader's own
    # density (i.e. at least as sparse), is the leader still the outlier?
    sparser = density >= density_leader
    print(f"\namong the {sparser.sum()} real templates AT LEAST AS SPARSE as "
          f"the leader: |net_out| p50={np.percentile(np.abs(net_out[sparser]),50):.4g} "
          f"max={np.abs(net_out[sparser]).max():.4g} "
          f"(leader={abs(net_out_leader):.4g})")


if __name__ == "__main__":
    main()
