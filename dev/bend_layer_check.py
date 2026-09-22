"""Is the leader's |e|^2 (== q) at the FIRST bulk layer's input (the one
layer with `bend=True`, `models/bijections.py` EquivariantAutoregressiveLayer)
unusual, and does that explain `jacobian_decomposition.py`'s finding that this
layer alone is ~87x steeper for the leader than for real templates in-band?
The bend itself lives in `Spin2CouplingLayer` (`layer.spin2`), the second
sublayer of `EquivariantAutoregressiveLayer`.

`_h(x, q) = _bounded_log_scale(net([x0,x1,x2, log1p(q)-W0]), 1e-4, 1e4) * 0.5`
is a sigmoid, so it is FLATTEST at saturation (0/1) and STEEPEST mid-sigmoid
(0.5) -- `scale_saturation_check.py` already ruled out saturation as the
leader's issue everywhere else, so check the same thing for this layer's own
sigmoid, plus the `1/(1+q)` factor in d(log1p(q))/dq, which blows up for
small q and is a mechanism unique to the bend layer's 4th input.

Run: python dev/bend_layer_check.py
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
N_CONTROL = 300


def build():
    m_train = shear.load(f"{DATA_DIR}/{B.TRAIN_DATA[POP]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=False)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    return flow, np.asarray(m_train)


def bend_layer_input(bijections, m_i):
    """x at the bend sublayer's own input: after raw2standard + shear(g=0)
    + the spin0 sublayer of the same EquivariantAutoregressiveLayer (spin0
    runs first inside it and touches slots 0-2, which is what `_h` reads)."""
    x = jnp.asarray(m_i)
    for b in bijections[:2]:            # RawMomentStandardize, ShearResponse
        c = jnp.zeros(2) if b.cond_shape is not None else None
        x = b.transform_and_log_det(x, c)[0]
    x = bijections[2].spin0.transform_and_log_det(x)[0]
    return x


def diagnostics(layer, x):
    q = x[3] ** 2 + x[4] ** 2
    u_raw = jnp.concatenate([x[:3], (jnp.log1p(q) - 1.0986122886681098)[None]])
    net_out = layer.net(u_raw)[0]
    sig = jax.nn.sigmoid(net_out)
    h, dh = jax.value_and_grad(lambda t: layer._h(x, t))(q)
    return dict(q=float(q), net_out=float(net_out), sigmoid=float(sig),
                dh_dq=float(dh), inv_1pq=float(1.0 / (1.0 + q)))


def main():
    flow, m_train = build()
    bijections = flow.bijection.bijection.bijections
    layer = bijections[2].spin2

    lead_x = bend_layer_input(bijections, LEADER)
    lead = diagnostics(layer, lead_x)

    ratio = m_train[:, 1] / m_train[:, 0]
    band = (ratio > 2.7) & (ratio < 3.5)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(band)[0], size=N_CONTROL, replace=False)
    ctrl = [diagnostics(layer, bend_layer_input(bijections, m_train[i]))
            for i in idx]

    print(f"leader: {lead}\n")
    for k in lead:
        vals = np.array([c[k] for c in ctrl])
        p = 100.0 * np.mean(vals < lead[k])
        print(f"{k:10s} leader={lead[k]:12.5g}  ctrl p50={np.percentile(vals,50):12.5g} "
              f"p1={np.percentile(vals,1):12.5g} p99={np.percentile(vals,99):12.5g} "
              f"max_abs={np.max(np.abs(vals)):12.5g}  leader_pctile={p:5.1f}")


if __name__ == "__main__":
    main()
