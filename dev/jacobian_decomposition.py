"""Layer-by-layer Jacobian condition number, at the R_s leader vs control
points, extending fold_check.py's SVD approach from the analytic truth map to
the TRAINED flow's own data->base chain.

`lipschitz_check.py` found the leader's full composed Q/R response is ~1470x
steeper than the real-to-real median in-band, with no single layer's scale
factor saturated (`coeff_ceiling_check.py`, `scale_saturation_check.py`). This
walks `flow.bijection.bijection.bijections` (data -> base order: raw2standard,
shear, then 8x[EquivariantAutoregressiveLayer, Permute]) one layer at a time,
taking the LOCAL Jacobian dy/dx of each layer's `transform_and_log_det` at the
point actually flowing through it (g=0), and reports each layer's own
condition number plus the running CUMULATIVE product Jacobian's condition
number -- to see which layer(s) contribute the amplification.

Run: python dev/jacobian_decomposition.py
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

LEADER = np.array([10001.8, 31896.3, -5789.59, 436.861, 195488.0])
POP = "gauss2_v3d"
FLOW = "flows/shear_g2v3d_full2.eqx"
DATA_DIR = "../bfd_cnf_imsims/data"
N_CONTROL = 20


def build():
    m_train = shear.load(f"{DATA_DIR}/{B.TRAIN_DATA[POP]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=False)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    return flow, np.asarray(m_train)


def layer_svals(bijections, cond, x0):
    """Per-layer and cumulative singular values of dy/dx, walking the chain."""
    x = jnp.asarray(x0)
    cum = jnp.eye(x.shape[0])
    per_layer, running = [], []
    for b in bijections:
        c = cond if b.cond_shape is not None else None
        f = (lambda z: b.transform_and_log_det(z, c)[0])
        J = jax.jacfwd(f)(x)
        per_layer.append(np.asarray(jnp.linalg.svd(J, compute_uv=False)))
        cum = J @ cum
        running.append(np.asarray(jnp.linalg.svd(cum, compute_uv=False)))
        x = f(x)
    return per_layer, running


def cond_no(sv):
    return sv[0] / sv[-1]


def main():
    flow, m_train = build()
    bijections = flow.bijection.bijection.bijections
    names = [type(b).__name__ for b in bijections]
    zero_g = jnp.zeros(2)

    ratio = m_train[:, 1] / m_train[:, 0]
    band = (ratio > 2.7) & (ratio < 3.5)
    rng = np.random.default_rng(0)
    control_idx = rng.choice(np.where(band)[0], size=N_CONTROL, replace=False)

    def report(label, x0):
        per_layer, running = layer_svals(bijections, zero_g, x0)
        return np.array([cond_no(sv) for sv in per_layer]), \
            np.array([cond_no(sv) for sv in running])

    lead_per, lead_cum = report("leader", LEADER)
    ctrl_per = np.stack([report("ctrl", m_train[i])[0] for i in control_idx])
    ctrl_cum = np.stack([report("ctrl", m_train[i])[1] for i in control_idx])

    print(f"{'layer':28s} {'leader cond':>14s} {'ctrl p50':>12s} "
          f"{'ctrl max':>12s} {'leader/ctrl_p50':>16s}")
    for i, name in enumerate(names):
        c50, cmax = np.percentile(ctrl_per[:, i], 50), ctrl_per[:, i].max()
        print(f"{i:2d} {name:25s} {lead_per[i]:14.4g} {c50:12.4g} "
              f"{cmax:12.4g} {lead_per[i]/max(c50,1e-12):16.2f}x")

    print(f"\n{'cumulative through layer':28s} {'leader cond':>14s} "
          f"{'ctrl p50':>12s} {'ctrl max':>12s} {'leader/ctrl_p50':>16s}")
    for i, name in enumerate(names):
        c50, cmax = np.percentile(ctrl_cum[:, i], 50), ctrl_cum[:, i].max()
        print(f"{i:2d} thru {name:19s} {lead_cum[i]:14.4g} {c50:12.4g} "
              f"{cmax:12.4g} {lead_cum[i]/max(c50,1e-12):16.2f}x")


if __name__ == "__main__":
    main()
