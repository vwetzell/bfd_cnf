"""Does the flow's OWN log p(m) flag the known gauss2 R_s leader as anomalous?

Answers a specific question raised in review: instead of the exact
Newton-preimage test (gauss2-only, needs a closed-form generative model),
threshold on log_prob(m) from the trained flow itself -- population-agnostic,
deployable anywhere. `[[retrain-ruled-out-logp-is-right]]` already measured
(on an older checkpoint) that the flow's log P matches the true density to
0.01 nats even where R inverts catastrophically -- i.e. density magnitude and
score roughness are decoupled under NLL training. This re-measures that claim
directly on the CURRENT flow, at the CURRENT known leader
(Mf=10001.8, Mr=31896.3, M1=-5789.59, M2=436.861, Mc=195488,
Mr/Mf=3.1890, Mc/Mr=6.1289 -- Newton residual 2.9e-19, log_p_theta=-inf,
R11 flow -103647 vs exact 0.213), against the log_prob DISTRIBUTION over the
real training catalog.

Run: python dev/leader_logp_check.py --flow flows/centroid_g2v3.eqx
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

LEADER = np.array([10001.8, 31896.3, -5789.59, 436.861, 195488.0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pop", default="gauss2_v3d", choices=sorted(B.CATALOGS))
    p.add_argument("--flow", default="flows/shear_g2v3d_full2.eqx")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    a = p.parse_args()

    m_train = shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0]
    # Matches the exact setup that found this leader last session
    # (rs_hull_leader_check.py): no centroid layer, sigma_x=None.
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=False)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)

    zero = jnp.zeros(2)
    cond = B.condition(zero, None)
    lp_fn = eqx.filter_jit(jax.vmap(
        lambda m: flow.log_prob(m, condition=cond)))

    m_train64 = np.asarray(m_train, dtype=np.float64)
    lp_train = np.concatenate([
        np.asarray(lp_fn(jnp.asarray(m_train64[i:i + 16384])))
        for i in range(0, len(m_train64), 16384)])
    finite = np.isfinite(lp_train)
    print(f"{a.pop}: {len(m_train64)} training templates, "
          f"{finite.mean():.2%} finite log_prob")
    lp_train = lp_train[finite]
    m_train64 = m_train64[finite]

    lp_leader = float(np.asarray(lp_fn(jnp.asarray(LEADER[None, :]))[0]))
    pct = 100.0 * np.mean(lp_train < lp_leader)
    print(f"\nleader log_prob = {lp_leader:.4f}")
    print(f"training catalog log_prob: mean {lp_train.mean():.4f}  "
          f"median {np.median(lp_train):.4f}  "
          f"p1/p50/p99 = {np.percentile(lp_train, [1, 50, 99])}")
    print(f"leader sits at the {pct:.2f} percentile of the training "
          f"catalog's own log_prob distribution")

    # Apples-to-apples: restrict to real templates in the SAME Mr/Mf sliver
    # the leader sits in (3.1890 +/- 1%), since density legitimately varies
    # with size/flux -- a global percentile could hide a local anomaly.
    ratio = m_train64[:, 1] / m_train64[:, 0]
    near = np.abs(ratio - 3.1890) < 0.03189
    if near.sum() >= 20:
        lp_near = lp_train[near]
        pct_near = 100.0 * np.mean(lp_near < lp_leader)
        print(f"\nrestricted to {near.sum()} real templates with "
              f"Mr/Mf in [3.157, 3.221] (leader's own band):")
        print(f"  local log_prob mean {lp_near.mean():.4f}  "
              f"median {np.median(lp_near):.4f}")
        print(f"  leader sits at the {pct_near:.2f} percentile locally")
    else:
        print(f"\nonly {near.sum()} real templates in the leader's Mr/Mf "
              "band -- too few for a local percentile")


if __name__ == "__main__":
    main()
