"""Is the score-contracted response error irreducible, or is the layer underfitting?

`dev/sweep_score_weight.py` put a term `mean((score . (q - q_true))^2)` directly
in the loss and swept its weight.  It barely moved: 390 -> 376 at weight 0.03,
and 376 again at weight 0.1, while the m1 ramp got monotonically WORSE (RMS
0.0130 -> 0.0186 -> 0.0325).  So ~96% of that error is unreachable by training
the shear layer.  Two things would explain it:

  (a) Var[Q|m] -- templates sharing m genuinely respond differently, and a
      DETERMINISTIC transport layer can only ever carry E[dm/dg | m].  Then the
      floor is physical, no loss or architecture removes it, and it is the
      multiplicative-bias floor `shear.py`'s docstring warns about.
  (b) The layer cannot represent E[dm/dg | m] well enough -- an architecture or
      objective problem, and the floor is an artefact.

These are separable without training anything.  `models.shear.response` is
LINEAR in the five first-order coefficients (a_Mf, a_Mr, a_Mc, A, B), and by
flux-blindness each is a function of the three invariants (r, k, q) alone, so
the best response the class can express is one weighted least-squares solve.
It must be fitted in the metric being REPORTED: an earlier version regressed the
coefficients in `shear.response_scatter`'s fractional metric and returned a
"floor" above the trained layer's error, which is impossible for a lower bound.

Measured (flows/shear.eqx, 100k): layer 1.42, floor 0.84, so 59% of the
first-order error is irreducible Var[Q|m] and 41% is reachable -- and the
reachable part is flat through the middle and jumps 4-6x in the top two Mr/Mf
octiles.  The last table tests whether the irreducible part IS the ramp, by
comparing Var[Q|m]/E[Q|m]^2 -- `shear.py`'s predicted bias -- against m1 bin by
bin.  It is not: the ratio rises monotonically 0.0046 -> 0.0159 while m1
oscillates and changes sign.  Right magnitude, wrong shape.

    python dev/check_response_floor.py [--flow flows/shear.eqx] [--n 100000]
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
from models.shear import ShearResponse, dm_dg        # noqa: E402

DATA = "../bfd_cnf_imsims/data/moments.fits"
NBINS = 8
G = 0.02


def bulk_score(flow, m, batch=2000):
    """grad log p_bulk at each m, with the shear layer peeled off.

    Q = score . u + div u, so this is the weight Q puts on a response error --
    the whole reason a 26% error on Mr can cost more than a 2% error on M1.
    """
    bij = flow.bijection.bijection.bijections
    assert isinstance(bij[0], ShearResponse), f"expected shear first, got {type(bij[0])}"
    bulk_flow = Transformed(flow.base_dist,
                            Invert(Chain(list(bij[1:])).merge_chains()))
    g = eqx.filter_jit(jax.vmap(jax.grad(lambda x: bulk_flow.log_prob(x))))
    return np.concatenate([np.asarray(g(m[i:i + batch]), np.float64)
                           for i in range(0, len(m), batch)])


def design(m, score):
    """d(score . dm/dg[a]) / d(coefficient p), shape (n, 2, 5).

    `models.shear.response` is LINEAR in the five first-order coefficients
    (a_Mf, a_Mr, a_Mc, A, B) -- the spin-0 exponential only matters at second
    order -- so the whole representable class is the column span of this tensor.
    That is what makes the floor a least-squares solve rather than a training
    run.  Spin-0 responses are e-parallel; spin-2 is d1 = A + B e^2,
    d2 = i (A - B e^2).
    """
    Mf, Mr, Mc = m[:, 0], m[:, 1], m[:, 4]
    e = (m[:, 2] + 1j * m[:, 3]) / Mr
    e2, s = e * e, score
    D = np.zeros((len(m), 2, 5))
    for a, ea in enumerate((e.real, e.imag)):
        D[:, a, 0] = s[:, 0] * Mf * ea
        D[:, a, 1] = s[:, 1] * Mr * ea
        D[:, a, 2] = s[:, 4] * Mc * ea
    D[:, 0, 3] = s[:, 2] * Mr
    D[:, 1, 3] = s[:, 3] * Mr
    D[:, 0, 4] = Mr * (s[:, 2] * e2.real + s[:, 3] * e2.imag)
    D[:, 1, 4] = Mr * (s[:, 2] * e2.imag - s[:, 3] * e2.real)
    return D


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--deg", type=int, default=4)
    a = p.parse_args()

    m, dm, d2m = (np.asarray(v, np.float64)[:a.n]
                  for v in shear_top.load(DATA))
    n90 = int(0.9 * len(m))
    flow = bulk.build_flow(jr.key(0), m[:n90], shear=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    print(f"{a.flow} on {len(m)} noiseless galaxies, poly degree {a.deg}\n")

    score = bulk_score(flow, jnp.asarray(m))
    Mf, Mr, Mc = m[:, 0], m[:, 1], m[:, 4]
    e_sq = (m[:, 2] ** 2 + m[:, 3] ** 2) / Mr ** 2

    # The floor: minimise the SAME quantity being reported, over every response
    # a layer of this class can express.  A previous version regressed the
    # coefficients in `response_scatter`'s fractional metric instead and got a
    # "floor" ABOVE the trained layer's error -- a fit is only a lower bound in
    # the metric it minimises.
    D = design(m, score)                                             # (n, 2, 5)
    basis = shear_top._poly(
        np.stack([Mr / Mf, Mc * Mf / (Mr * Mr), e_sq], -1), a.deg)   # (n, nb)
    y = np.einsum("bm,bam->ba", score, dm)                           # (n, 2)
    # c_p(m) = basis . beta_p, so score.q = sum_p D_p (basis . beta_p) is linear
    # in the stacked beta -- one lstsq over (n*2) rows, 5*nb columns.
    X = (D[:, :, :, None] * basis[:, None, None, :]).reshape(len(m) * 2, -1)
    beta = np.linalg.lstsq(X, y.reshape(-1), rcond=None)[0]
    d_fit = (X @ beta - y.reshape(-1)).reshape(len(m), 2)

    layer = shear_top._shear_layer(flow)
    q_lay = np.asarray(eqx.filter_jit(jax.vmap(dm_dg, in_axes=(None, 0)))(
        layer, jnp.asarray(m))[0], np.float64)
    d_lay = np.einsum("bm,bam->ba", score, q_lay - dm)

    print(f"\nfirst-order score-contracted error, mean((score.dq)^2):")
    print(f"  rms(Q1) for scale    {np.sqrt(np.mean(y[:, 0] ** 2)):10.2f}")
    print(f"  trained layer        {np.mean(d_lay ** 2):10.2f}")
    print(f"  best in class        {np.mean(d_fit ** 2):10.2f}"
          f"   <- Var[Q|m] floor")
    print(f"  ratio                {np.mean(d_fit ** 2) / np.mean(d_lay ** 2):10.3f}")

    # The layer is fitted to the SECOND derivative too, and `_score_mse` summed
    # both; report the split so the 390 is attributable.
    r_lay = np.asarray(eqx.filter_jit(jax.vmap(dm_dg, in_axes=(None, 0)))(
        layer, jnp.asarray(m))[1], np.float64)
    d2 = np.einsum("bm,bam->ba", score, r_lay - d2m)
    print(f"\n  (second order, layer {np.mean(d2 ** 2):.1f} -- the two summed "
          f"are what the sweep drove)")

    # The payoff test.  A deterministic transport carries E[Q|m] and supplies
    # ZERO Var[Q|m], so the density it produces is too narrow in g by exactly
    # the floor above.  That enters R through the QQ^T/P^2 term, and `shear.py`'s
    # docstring predicts a multiplicative bias ~ Var[Q|m]/E[Q|m]^2 that does NOT
    # shrink with g.  If that is the ramp, this ratio tracks m1 bin by bin.
    qp, rp = (np.asarray(v, np.float64) for v in
              bias.pqr(flow, m + G * dm[:, 0, :] + 0.5 * G ** 2 * d2m[:, 0, :]))
    qm, rm = (np.asarray(v, np.float64) for v in
              bias.pqr(flow, m - G * dm[:, 0, :] + 0.5 * G ** 2 * d2m[:, 0, :]))
    m1 = lambda s: bias.bias(qp, rp, qm, rm, G, s)[0]

    r = Mr / Mf
    qs = np.quantile(r, np.linspace(0, 1, NBINS + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    print(f"\n{'<Mr/Mf>':>9s} {'layer':>8s} {'floor':>8s} {'<Q1^2>':>9s} "
          f"{'floor/<Q1^2>':>13s} {'m1':>9s}")
    for k in range(NBINS):
        s = (r >= qs[k]) & (r < qs[k + 1])
        lo, fl = np.mean(d_lay[s] ** 2), np.mean(d_fit[s] ** 2)
        q2 = np.mean(y[s, 0] ** 2)
        print(f"{r[s].mean():9.3f} {lo:8.2f} {fl:8.2f} {q2:9.2f} "
              f"{fl / q2:13.4f} {m1(s):+9.4f}")


if __name__ == "__main__":
    main()
