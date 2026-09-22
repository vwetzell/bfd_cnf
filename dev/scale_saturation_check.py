"""Is any layer's internal scale factor saturated (sigmoid pinned near 0 or 1)
at the R_s leader, in a way that's anomalous vs. real templates in the same
Mr/Mf band?

Every affine/coupling layer in the bulk stack (Spin0AutoregressiveLayer's
per-step log-scale, Spin2CouplingLayer's stretch) squashes a raw MLP output
through a sigmoid via `_bounded_log_scale`. sigmoid(raw) near 0 or 1 means the
scale is pinned to its bound -- and per that helper's own docstring, "it is a
sigmoid, so a coefficient that reaches it cannot come back": once saturated, a
small input change can no longer move the scale (flat), but a network whose
raw pre-sigmoid output is highly sensitive to input right at the edge of the
transition can still make the SQUASHED scale swing fast just before pinning.
This walks the leader and a control sample through all 8 bulk layers,
recording the sigmoid activation (0..1) at each internal scale, and reports
where each sits.
"""
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402
from models.bijections import EquivariantAutoregressiveLayer, Permute  # noqa: E402
from models.shear import ShearResponse  # noqa: E402

LEADER = np.array([10001.8, 31896.3, -5789.59, 436.861, 195488.0])
POP = "gauss2_v3d"
FLOW = "flows/shear_g2v3d_full2.eqx"
DATA_DIR = "../bfd_cnf_imsims/data"
N_SAMPLE = 300


def walk(flow_bij, x):
    """Push x (post-chart, post-shear-at-g=0 coordinate) through the bulk
    stack, returning per-layer (spin0_frac1, spin0_frac2, spin2_frac) sigmoid
    activations, each in (0, 1); near 0/1 = saturated."""
    fracs = []
    for b in flow_bij:
        if isinstance(b, EquivariantAutoregressiveLayer):
            spin0, spin2 = b.spin0, b.spin2
            # ls0 is position-independent (log_scale0 is a free scalar, not a
            # function of x) so it can't be "saturated near this point" --
            # only ls1, ls2 (and the spin2 stretch) depend on x.
            z0 = (x[0] - spin0.loc0) * jnp.exp(-_bounded(spin0.log_scale0))
            out1 = spin0.net_ls1(jnp.array([z0]))
            loc1, pre1 = out1[0], out1[1]
            z1 = (x[1] - loc1) * jnp.exp(-_bounded(pre1))
            out2 = spin0.net_ls2(jnp.array([z0, z1]))
            loc2, pre2 = out2[0], out2[1]
            z2 = (x[2] - loc2) * jnp.exp(-_bounded(pre2))
            x = x.at[0].set(z0).at[1].set(z1).at[2].set(z2)

            q = x[3] ** 2 + x[4] ** 2
            u = x[:3] if not spin2.bend else jnp.concatenate(
                [x[:3], (jnp.log1p(q) - 1.0986122886681098)[None]])
            pre_s2 = spin2.net(u)[0]
            frac_s2 = jax.nn.sigmoid(pre_s2)
            s = jnp.exp(_bounded_wide(pre_s2))
            x = x.at[3].set(x[3] * s).at[4].set(x[4] * s)

            fracs.append((float(jax.nn.sigmoid(pre1)), float(jax.nn.sigmoid(pre2)),
                          float(frac_s2)))
        elif isinstance(b, Permute):
            x = x[b.permutation]
    return fracs


def _bounded(pre, lo=1e-2, hi=100.0):
    return jnp.log(lo) + (jnp.log(hi) - jnp.log(lo)) * jax.nn.sigmoid(pre)


def _bounded_wide(pre, lo=1e-4, hi=1e4):
    return (jnp.log(lo) + (jnp.log(hi) - jnp.log(lo)) * jax.nn.sigmoid(pre)) * 0.5


def satur_score(fracs):
    """Max distance-from-0.5-normalised saturation across all sub-scales in
    the stack: 0 = dead centre, 1 = fully pinned to a sigmoid extreme."""
    vals = np.array(fracs).ravel()
    return np.abs(vals - 0.5).max() * 2


def main():
    m_train = shear.load(f"{DATA_DIR}/{B.TRAIN_DATA[POP]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=False)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)

    bij = flow.bijection.bijection.bijections
    chart = bij[0]

    def to_post_shear(m_row):
        m_row = jnp.asarray(m_row, dtype=jnp.float64)
        z = chart.transform(m_row, None)
        return z   # shear at g=0 is the identity, so z is what the bulk sees

    leader_x = to_post_shear(LEADER)
    fracs_leader = walk(bij, leader_x)
    print("leader per-layer sigmoid activations (spin0_1, spin0_2, spin2):")
    for i, f in enumerate(fracs_leader):
        print(f"  layer {i}: {f[0]:.5f} {f[1]:.5f} {f[2]:.5f}")
    print(f"leader saturation score (0=centred,1=pinned): {satur_score(fracs_leader):.5f}")

    ratio = np.asarray(m_train)[:, 1] / np.asarray(m_train)[:, 0]
    band = (ratio > 2.7) & (ratio < 3.5)
    idx = np.where(band)[0]
    rng = np.random.default_rng(0)
    sample = rng.choice(idx, size=N_SAMPLE, replace=False)
    m_train64 = np.asarray(m_train, dtype=np.float64)

    scores = []
    for i in sample:
        x = to_post_shear(m_train64[i])
        scores.append(satur_score(walk(bij, x)))
    scores = np.array(scores)
    pct = 100.0 * np.mean(scores < satur_score(fracs_leader))
    print(f"\ncontrol ({len(sample)} real templates, Mr/Mf(2.7,3.5)) saturation score:")
    print(f"  p50={np.median(scores):.5f} p90={np.percentile(scores,90):.5f} "
          f"p99={np.percentile(scores,99):.5f} max={scores.max():.5f}")
    print(f"leader sits at the {pct:.2f} percentile of control saturation")


if __name__ == "__main__":
    main()
