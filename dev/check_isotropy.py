"""Where does the flow pick up a preferred direction on the sky?

`dev/check_noiseless_shear_recovery.py` measures g1 - g2 = -0.0074 +/- 0.0006 in
the top Mr/Mf octile and ~1e-4 in the other six.  That should be impossible: the
spin structure is a hard architectural constraint, so the density must be
EXACTLY invariant under a frame rotation, not approximately.

Rotating the frame by an angle theta sends (M1, M2) -> R(theta) (M1, M2) and
leaves the three spin-0 moments alone.  Under that rotation:

  * log P(m) is invariant                        -- tests the density
  * |Q| and the eigenvalues of R are invariant   -- tests the shear response

All three are convention-free, so nothing here depends on getting a sign or a
transpose right.  Each is reported for the bulk alone and for the full flow, so
a leak can be attributed to the g-independent density or to ShearResponse.

    python dev/check_isotropy.py [--flow flows/shear.eqx] [--n 20000]
"""
import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flowjax.bijections import Chain, Invert
from flowjax.distributions import Transformed

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402
from models.shear import ShearResponse               # noqa: E402

DATA = "../bfd_cnf_imsims/data/moments.fits"
NBINS = 8
THETAS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]) * np.pi


def rotate(m, theta):
    """Frame rotation: the spin-2 pair turns by theta, spin-0 is untouched."""
    c, s = np.cos(theta), np.sin(theta)
    out = m.copy()
    out[:, 2] = c * m[:, 2] - s * m[:, 3]
    out[:, 3] = s * m[:, 2] + c * m[:, 3]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--n", type=int, default=20000)
    a = p.parse_args()

    m, dm, d2m = (np.asarray(v, np.float64)[:a.n]
                  for v in shear_top.load(DATA))
    n90 = int(0.9 * len(shear_top.load(DATA)[0]))
    flow = bulk.build_flow(jr.key(0), np.asarray(shear_top.load(DATA)[0],
                                                 np.float64)[:n90], shear=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    # BY TYPE: the chart is bijections[0] since 16ece5c, so the old
    # `assert isinstance(bij[0], ShearResponse)` could never pass.
    bij = flow.bijection.bijection.bijections
    assert any(isinstance(b, ShearResponse) for b in bij), "no ShearResponse"
    bulk_flow = Transformed(flow.base_dist, Invert(Chain(
        [b for b in bij if not isinstance(b, ShearResponse)]).merge_chains()))
    print(f"{a.flow} on {len(m)} galaxies, {len(THETAS)} frame rotations\n")

    lp_bulk = eqx.filter_jit(jax.vmap(bulk_flow.log_prob))
    lp_full = eqx.filter_jit(lambda x: flow.log_prob(x, condition=jnp.zeros(2)))

    L_b, L_f, QN, RT = [], [], [], []
    for th in THETAS:
        mr = jnp.asarray(rotate(m, th))
        L_b.append(np.asarray(lp_bulk(mr), np.float64))
        L_f.append(np.asarray(jax.vmap(lp_full)(mr), np.float64))
        q, r = (np.asarray(v, np.float64) for v in bias.pqr(flow, rotate(m, th)))
        QN.append(np.linalg.norm(q, axis=1))
        # trace and determinant of R: the two rotation invariants of a 2x2
        RT.append(np.stack([r[:, 0, 0] + r[:, 1, 1],
                            r[:, 0, 0] * r[:, 1, 1] - r[:, 0, 1] * r[:, 1, 0]], 1))
    L_b, L_f, QN, RT = (np.array(v) for v in (L_b, L_f, QN, RT))

    def drift(v):
        """Peak-to-peak across rotations, per galaxy, relative to its own scale."""
        return np.ptp(v, axis=0) / (np.abs(v).mean(0) + 1e-30)

    r_ = m[:, 1] / m[:, 0]
    qs = np.quantile(r_, np.linspace(0, 1, NBINS + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    print(f"median relative spread over the {len(THETAS)} rotations "
          f"(0 = exactly isotropic)")
    print(f"{'<Mr/Mf>':>9s} {'log P bulk':>12s} {'log P full':>12s} "
          f"{'|Q|':>12s} {'tr R':>12s} {'det R':>12s}")
    cols = [np.abs(np.ptp(L_b, 0)), np.abs(np.ptp(L_f, 0)),
            drift(QN), drift(RT[:, :, 0]), drift(RT[:, :, 1])]
    for k in range(NBINS):
        s = (r_ >= qs[k]) & (r_ < qs[k + 1])
        print(f"{r_[s].mean():9.3f}" +
              "".join(f"{np.median(c[s]):12.3e}" for c in cols))
    print(f"{'ALL':>9s}" + "".join(f"{np.median(c):12.3e}" for c in cols))
    print("\n(log P columns are absolute nats; |Q|, tr R, det R are relative)")


if __name__ == "__main__":
    main()
