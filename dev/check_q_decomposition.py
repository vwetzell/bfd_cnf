"""Which half of Q carries the Mr/Mf ramp -- the bulk score, or the response's
divergence?  And is Var[dm/dg | m] resolution-dependent?

For the stack base -> bulk -> shear(g) -> data,

    log P(m|g) = log p_bulk(T_g(m)) + log|det dT_g/dm|,     T_g = unshear

and since T_0 = identity, differentiating at g = 0 gives

    Q  =  grad log p_bulk . u  +  div u,        u = dT_g/dg |_0

so Q is built from TWO derivative objects.  Neither is in any loss:

  * `bulk.train` minimises NLL, which constrains p, not grad log p.
  * `shear.train`'s deriv_weight term is `_velocity_mse`, which compares the
    layer's response to `dm_dg` POINTWISE -- it constrains u, not du/dm, hence
    not div u.

That is the structural reason bulk nll, shear nll and velocity mse all sat flat
across a 10x range in RMS(m1) in `dev/sweep_capacity.py`: they measure the two
things that ARE constrained, and Q needs the two that are not.

This prints, per Mr/Mf octile, the two terms side by side with the m1 ramp, and
the share of the first-order response that the moments do not determine at all
(`shear.response_scatter`'s statistic, binned -- it is only ever reported
globally, and a resolution-dependent ramp needs it resolved in Mr/Mf).  That
last one is a FLOOR: a deterministic transport layer carries E[dm/dg | m] and
nothing else, so Var[dm/dg | m] is bias no amount of capacity can remove.

    python dev/check_q_decomposition.py [--flow flows/shear.eqx] [--n 100000]
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


def split_shear(flow):
    """Peel ShearResponse off the front, leaving the g-independent bulk.

    Same pattern as `bias.split_centroid`: in data -> base order the chain is
    [raw2standard, centroid?, shear, *bulk] since 16ece5c made the chart
    data-adjacent, so the shear layer is located BY TYPE -- it is no longer
    `bijections[0]`, and asserting that it was is what made this script die.
    Everything else is the density whose SCORE the first term of Q needs.
    """
    bij = flow.bijection.bijection.bijections
    shear = [b for b in bij if isinstance(b, ShearResponse)]
    assert len(shear) == 1, f"expected one ShearResponse, got {len(shear)}"
    rest = Invert(Chain([b for b in bij
                         if not isinstance(b, ShearResponse)]).merge_chains())
    return Transformed(flow.base_dist, rest), shear[0]


def q_terms(bulk_flow, layer, m_i):
    """(score term, divergence term, u) for one galaxy, each (2,) or (5, 2)."""
    zero = jnp.zeros(2)
    a_of = lambda x: jax.jacfwd(layer.unshear, argnums=1)(x, zero)   # (5, 2)
    u = a_of(m_i)
    # jacfwd(a_of)[i, a, j] = d u[i, a] / d m[j]; the divergence is its trace
    # over the two moment indices, one per shear component.
    div = jnp.einsum("iai->a", jax.jacfwd(a_of)(m_i))
    score = jax.grad(lambda x: bulk_flow.log_prob(x))(m_i)           # (5,)
    return score @ u, div, u


def response_scatter_binned(m, q_true, sel_list, deg=4):
    """`shear.response_scatter`'s spin-2 statistic, evaluated per bin.

    RMS of what a degree-`deg` regression of the exact response coefficients on
    (Mr/Mf, Mc Mf/Mr^2, |e|^2) CANNOT explain, over RMS of the response itself.
    The fit is global and the residual is then binned, so every bin is judged
    against the same conditional mean -- which is what a transport layer would
    be carrying.
    """
    A, B, w4 = shear_top._spin2_AB(m, q_true)
    Mf, Mr, Mc = m[:, 0], m[:, 1], m[:, 4]
    e_sq = (m[:, 2] ** 2 + m[:, 3] ** 2) / Mr ** 2
    basis = shear_top._poly(np.stack([Mr / Mf, Mc * Mf / (Mr * Mr), e_sq], -1), deg)
    ones = np.ones(len(m))
    terms = [(A, ones), (B, w4)]
    resid = [shear_top._wls_resid(basis, c, wt) for c, wt in terms]
    out = []
    for sel in sel_list:
        num = sum(np.mean((d ** 2 * wt)[sel]) for d, (_, wt) in zip(resid, terms))
        den = sum(np.mean((c ** 2 * wt)[sel]) for c, wt in terms)
        out.append(float(np.sqrt(num / den)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--batch", type=int, default=2000)
    p.add_argument("--g", type=float, default=0.02)
    a = p.parse_args()

    m, dm, d2m = (np.asarray(v, np.float64)[:a.n]
                  for v in shear_top.load(DATA))
    n90 = int(0.9 * len(m))
    flow = bulk.build_flow(jr.key(0), m[:n90], shear=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    bulk_flow, layer = split_shear(flow)
    print(f"{a.flow} on {len(m)} noiseless galaxies\n")

    batched = eqx.filter_jit(jax.vmap(lambda x: q_terms(bulk_flow, layer, x)))
    sc, dv, uu = [], [], []
    for i in range(0, len(m), a.batch):
        s, d, u = batched(jnp.asarray(m[i:i + a.batch], dtype=jnp.float32))
        sc.append(np.asarray(s, np.float64))
        dv.append(np.asarray(d, np.float64))
        uu.append(np.asarray(u, np.float64))
    score_t, div_t, u_flow = np.concatenate(sc), np.concatenate(dv), np.concatenate(uu)

    # The decomposition has to reproduce the Q that `bias.pqr` gets by
    # differentiating the whole log_prob.  If it does not, nothing below means
    # anything, so this is checked before any of it is printed.
    q_ref = np.asarray(bias.pqr(flow, m)[0], np.float64)
    q_dec = score_t + div_t
    rel = np.abs(q_dec - q_ref) / (np.abs(q_ref) + 1e-6)
    print(f"decomposition vs bias.pqr Q: median rel err {np.median(rel):.2e}, "
          f"p99 {np.percentile(rel, 99):.2e}")
    if np.median(rel) > 1e-2:
        print("  !! decomposition does not reproduce Q -- read no further")

    # u is d(unshear)/dg, the INVERSE map, so it should track -dm_dg.
    fit = np.polyfit(dm[:, 0, 2], u_flow[:, 2, 0], 1)
    print(f"u[:,2,0] vs dm_dg[:,0,2]: slope {fit[0]:+.4f} (expect ~-1)\n")

    r = m[:, 1] / m[:, 0]
    rng = np.random.default_rng(0)
    qp, rp = (np.asarray(v, np.float64) for v in bias.pqr(flow, m + a.g * dm[:, 0, :]
                                                         + 0.5 * a.g ** 2 * d2m[:, 0, :]))
    qm, rm = (np.asarray(v, np.float64) for v in bias.pqr(flow, m - a.g * dm[:, 0, :]
                                                         + 0.5 * a.g ** 2 * d2m[:, 0, :]))
    m1 = lambda s: bias.bias(qp, rp, qm, rm, a.g, s)[0]

    q = np.quantile(r, np.linspace(0, 1, NBINS + 1))
    q[0], q[-1] = -np.inf, np.inf
    sels = [(r >= q[k]) & (r < q[k + 1]) for k in range(NBINS)]
    scat = response_scatter_binned(m, dm, sels)

    print(f"{'<Mr/Mf>':>8s} {'m1':>9s} | {'<score.u>':>10s} {'sd':>9s} "
          f"| {'<div u>':>10s} {'sd':>9s} | {'|div|/|Q|':>9s} | {'Var[u|m]':>9s}")
    for k, sel in enumerate(sels):
        s1, d1 = score_t[sel, 0], div_t[sel, 0]
        frac = np.mean(np.abs(d1)) / np.mean(np.abs(s1 + d1))
        print(f"{r[sel].mean():8.3f} {m1(sel):+9.4f} | {s1.mean():10.3f} "
              f"{s1.std():9.3f} | {d1.mean():10.3f} {d1.std():9.3f} | "
              f"{frac:9.3f} | {scat[k]:8.2%}")

    print(f"\nwhole sample: m1 {m1(np.ones(len(m), bool)):+.4f}, "
          f"<score.u> {score_t[:, 0].mean():.3f}, <div u> {div_t[:, 0].mean():.3f}, "
          f"Var[u|m] {response_scatter_binned(m, dm, [slice(None)])[0]:.2%}")


if __name__ == "__main__":
    main()
