"""Does the leader's Lipschitz-anomalous score trace to _COEFF_MAX_SPIN0
saturation / _edge_factor's point-source taper?

_edge_factor tapers the shear response to exactly zero at Mr/Mf=POINT_SOURCE
(3.6926); _Coeffs is free to grow (up to +/-1e4 for the spin-0 block) to
compensate. If the coefficient net saturates near the ceiling, small changes
in ratio can produce large changes in coeffs -> large changes in Q,R -> the
steep Lipschitz ratio measured for the leader (Mr/Mf=3.1890).

This pulls _Coeffs and _edge_factor straight out of the trained flow's
ShearResponse layer and evaluates them on a ratio sweep plus the leader
itself and its real-template neighbours.
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
from models.shear import ShearResponse, _invariants, _edge_factor, POINT_SOURCE  # noqa: E402
from paramax import unwrap  # noqa: E402

LEADER = np.array([10001.8, 31896.3, -5789.59, 436.861, 195488.0])
POP = "gauss2_v3d"
FLOW = "flows/shear_g2v3d_full2.eqx"
DATA_DIR = "../bfd_cnf_imsims/data"


def main():
    m_train = shear.load(f"{DATA_DIR}/{B.TRAIN_DATA[POP]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=False)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)

    bij = flow.bijection.bijection.bijections
    chart = bij[0]
    shear_layer = next(b for b in bij if isinstance(b, ShearResponse))
    print(f"POINT_SOURCE = {POINT_SOURCE:.4f}")

    def to_z(m_batch):
        m_batch = jnp.asarray(m_batch, dtype=jnp.float64)
        return jax.vmap(lambda m: chart.transform(m, None))(m_batch)

    def coeffs_and_edge(z_row):
        f, a, b, q, _ = _invariants(z_row)
        c = shear_layer.coeffs(f, a, b, q)
        s = _edge_factor(z_row, unwrap(shear_layer.chart_loc), unwrap(shear_layer.chart_scale))
        return c, s

    cne = eqx.filter_jit(jax.vmap(coeffs_and_edge))

    # --- leader ---
    z_leader = to_z(LEADER[None, :])
    c_lead, s_lead = cne(z_leader)
    c_lead, s_lead = np.asarray(c_lead[0]), float(s_lead[0])
    ratio_lead = LEADER[1] / LEADER[0]
    print(f"\nleader: Mr/Mf={ratio_lead:.4f}  edge_factor s={s_lead:.4f}  "
          f"(1 - Mr/Mf/POINT_SOURCE = {1 - ratio_lead/POINT_SOURCE:.4f})")
    print(f"leader raw coeffs (spin-0 block, bound=1e4): {c_lead[:9]}")
    print(f"  max |spin-0 coeff| = {np.abs(c_lead[:9]).max():.4g}  "
          f"({100*np.abs(c_lead[:9]).max()/1e4:.2f}% of _COEFF_MAX_SPIN0)")
    print(f"leader raw coeffs (spin-2 block, bound=12): {c_lead[9:]}")
    print(f"  max |spin-2 coeff| = {np.abs(c_lead[9:]).max():.4g}  "
          f"({100*np.abs(c_lead[9:]).max()/12:.2f}% of _COEFF_MAX)")

    # --- sensitivity: d(coeffs)/d(ratio) near the leader, via a small sweep
    # in Mr holding Mf, Mc/Mr fixed at the leader's own values ---
    mf, mr = LEADER[0], LEADER[1]
    conc = LEADER[4] / LEADER[1]
    d_ratio = np.linspace(-0.02, 0.02, 41) * (POINT_SOURCE - ratio_lead)
    ratios = ratio_lead + d_ratio
    mrs = ratios * mf
    ms = np.stack([np.full_like(mrs, mf), mrs, np.full_like(mrs, LEADER[2]),
                   np.full_like(mrs, LEADER[3]), mrs * conc], axis=-1)
    zs = to_z(ms)
    cs, ss = cne(zs)
    cs = np.asarray(cs)
    dc = np.linalg.norm(np.diff(cs, axis=0), axis=-1)
    dr = np.diff(ratios)
    slope = dc / np.abs(dr)
    print(f"\nlocal d|coeffs|/d(ratio) sweep around the leader "
          f"(ratio {ratios[0]:.4f} to {ratios[-1]:.4f}, POINT_SOURCE={POINT_SOURCE:.4f}):")
    print(f"  slope: min {slope.min():.4g}  median {np.median(slope):.4g}  "
          f"max {slope.max():.4g}  (max at ratio={ratios[np.argmax(dc)]:.4f})")

    # --- same sweep far from the ceiling, e.g. ratio ~ 2.0, for contrast ---
    ratio0 = 2.0
    d_ratio2 = np.linspace(-0.02, 0.02, 41) * (POINT_SOURCE - ratio0)
    ratios2 = ratio0 + d_ratio2
    mrs2 = ratios2 * mf
    ms2 = np.stack([np.full_like(mrs2, mf), mrs2, np.full_like(mrs2, LEADER[2]),
                    np.full_like(mrs2, LEADER[3]), mrs2 * conc], axis=-1)
    zs2 = to_z(ms2)
    cs2, ss2 = cne(zs2)
    cs2 = np.asarray(cs2)
    dc2 = np.linalg.norm(np.diff(cs2, axis=0), axis=-1)
    dr2 = np.diff(ratios2)
    slope2 = dc2 / np.abs(dr2)
    print(f"\nsame sweep width, centered at ratio={ratio0} (well below ceiling):")
    print(f"  slope: min {slope2.min():.4g}  median {np.median(slope2):.4g}  "
          f"max {slope2.max():.4g}")
    print(f"\nleader-region slope is {np.median(slope)/np.median(slope2):.1f}x "
          "the away-from-ceiling slope")


if __name__ == "__main__":
    main()
