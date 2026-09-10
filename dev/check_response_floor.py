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
from models.shear import (ShearResponse, dm_dg,      # noqa: E402
                          project_to_physics, unwrap)

DATA = "../bfd_cnf_imsims/data/moments.fits"
NBINS = 8
G = 0.02


def bulk_score(flow, m, batch=2000):
    """grad log p_bulk at each m, with the shear layer peeled off.

    Q = score . u + div u, so this is the weight Q puts on a response error --
    the whole reason a 26% error on Mr can cost more than a 2% error on M1.
    """
    bij = [b for b in flow.bijection.bijection.bijections
           if not isinstance(b, ShearResponse)]
    assert len(bij) < len(flow.bijection.bijection.bijections), "no ShearResponse"
    bulk_flow = Transformed(flow.base_dist,
                            Invert(Chain(bij).merge_chains()))
    g = eqx.filter_jit(jax.vmap(jax.grad(lambda x: bulk_flow.log_prob(x))))
    return np.concatenate([np.asarray(g(m[i:i + batch]), np.float64)
                           for i in range(0, len(m), batch)])


# The five coefficient slots that survive to FIRST order in g: `p2` and `p3` in
# `project_to_physics` are both O(g^2), so of the 3x3 spin-0 block only column 0
# contributes, and of (A, B, mu, nu, rho) only A and B do.
_FIRST_ORDER = np.array([0, 3, 6, 9, 10])


def _dmdg_of_coeffs(c5, m, layer, chart):
    """The layer's generative dm/dg (2, 5) with the first-order coefficients
    replaced by `c5`, everything else zeroed.

    The layer no longer acts on raw moments: it shears in the chart's
    standardised z and `dm_dg` composes the chart on both sides.  The map stays
    LINEAR in the coefficients through that composition (the chain rule is), so
    the representable class is still a column span -- but the columns have to be
    taken through the real chart, not the old hand-written raw-moment formula.
    """
    full = jnp.zeros(14).at[_FIRST_ORDER].set(c5)
    z = chart.transform(m)
    Q = project_to_physics(full, z, jnp.zeros(2), unwrap(layer.e_scale),
                           unwrap(layer.chart_loc), unwrap(layer.chart_scale))[0]
    # `shear` (base -> data) is the generative direction; to first order it is
    # the negative of `unshear`'s, which is what `project_to_physics` builds.
    return jax.jacfwd(lambda g: chart.inverse(z - Q @ g))(jnp.zeros(2)).T


def design(m, layer, chart, batch=2000):
    """d(dm/dg[a, i]) / d(coefficient p), shape (n, 2, 5, 5)."""
    d = eqx.filter_jit(jax.vmap(
        lambda mi: jax.jacobian(_dmdg_of_coeffs)(
            jnp.zeros(5), mi, layer, chart)))                        # (2, 5, 5)
    return np.concatenate([np.asarray(d(m[i:i + batch]), np.float64)
                           for i in range(0, len(m), batch)])


def _solve(rows, basis, y, chunk=5000):
    """Least squares of `y` on the span of `rows[..., p] * basis`, by normal
    equations -- the per-moment fit has n*2*5 rows, too many to form densely.

    `rows` is (n, ..., 5) and `y` is (n, ...); returns the residual, shaped
    like `y`.
    """
    nb = basis.shape[1]
    xtx, xty = np.zeros((5 * nb, 5 * nb)), np.zeros(5 * nb)
    for i in range(0, len(y), chunk):
        r, b = rows[i:i + chunk], basis[i:i + chunk]
        # broadcast the basis over every non-galaxy axis of `rows`
        shape = (1,) * (r.ndim - 2)
        X = (r[..., None] * b.reshape(len(b), *shape, 1, nb)).reshape(-1, 5 * nb)
        xtx += X.T @ X
        xty += X.T @ y[i:i + chunk].reshape(-1)
    beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    out = np.empty_like(y)
    for i in range(0, len(y), chunk):
        r, b = rows[i:i + chunk], basis[i:i + chunk]
        shape = (1,) * (r.ndim - 2)
        X = (r[..., None] * b.reshape(len(b), *shape, 1, nb)).reshape(-1, 5 * nb)
        out[i:i + chunk] = (X @ beta).reshape(y[i:i + chunk].shape) - y[i:i + chunk]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--data", default=DATA)
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--deg", type=int, default=4)
    a = p.parse_args()

    m, dm, d2m = (np.asarray(v, np.float64)[:a.n]
                  for v in shear_top.load(a.data))
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
    layer, chart = shear_top._shear_layer(flow), shear_top._chart(flow)
    D = design(jnp.asarray(m), layer, chart)                     # (n, 2, 5, 5)
    # Regressed on the layer's OWN conditioning inputs (`models.shear._invariants`
    # -- z0, z1, z2, |e|^2), not the old flux-blind (r, k, q): flux is an input
    # now, so a flux-blind basis would understate what the class can express.
    z = np.asarray(eqx.filter_jit(jax.vmap(chart.transform))(jnp.asarray(m)),
                   np.float64)
    basis = shear_top._poly(
        np.stack([z[:, 0], z[:, 1], z[:, 2],
                  z[:, 3] ** 2 + z[:, 4] ** 2], -1), a.deg)          # (n, nb)
    y = np.einsum("bm,bam->ba", score, dm)                           # (n, 2)
    # c_p(m) = basis . beta_p, so score.q = sum_p D_p (basis . beta_p) is linear
    # in the stacked beta -- one lstsq over (n*2) rows, 5*nb columns.
    d_fit = _solve(np.einsum("bm,bamp->bap", score, D), basis, y)


    q_lay = np.asarray(eqx.filter_jit(jax.vmap(dm_dg, in_axes=(None, 0, None)))(
        layer, jnp.asarray(m), chart)[0], np.float64)
    d_lay = np.einsum("bm,bam->ba", score, q_lay - dm)

    print(f"\nfirst-order score-contracted error, mean((score.dq)^2):")
    print(f"  rms(Q1) for scale    {np.sqrt(np.mean(y[:, 0] ** 2)):10.2f}")
    print(f"  trained layer        {np.mean(d_lay ** 2):10.2f}")
    print(f"  best in class        {np.mean(d_fit ** 2):10.2f}"
          f"   <- Var[Q|m] floor")
    print(f"  ratio                {np.mean(d_fit ** 2) / np.mean(d_lay ** 2):10.3f}")

    # Which moment's response error the score actually charges for.  The
    # contraction mixes all five, so attribute by leave-one-out: zero moment i's
    # error and see how much of mean((score.dq)^2) goes away.  Cross terms are
    # real, so the shares need not sum to 1.
    dq = q_lay - dm
    # The (b) question.  The CONTRACTED floor above is reachable only by letting
    # Mr and Mc cancel, which is the exploit `--score-weight` was rewritten to
    # close.  `pm floor` is the same class with the same shared coefficients but
    # scored PER MOMENT, i.e. cancellation not allowed.  If the layer sits at it,
    # the parameterisation cannot express E[dm/dg|m] on the score-heavy tail and
    # no objective will fix it; if it sits well above, training is still leaving
    # something.
    sy = score[:, None, :] * dm                                   # (n, 2, 5)
    # Solved in the training term's own per-moment normalisation, so the shared
    # coefficients trade the five moments off the way training does; the residual
    # is scaled back for reporting.
    ns = np.sqrt(np.mean(sy ** 2, axis=(0, 1)))                   # (5,)
    pm_fit = ns * _solve(score[:, None, :, None] * D / ns[:, None],
                         basis, sy / ns)
    pm_lay = score[:, None, :] * dq

    print(f"\n  score-contracted error by moment "
          f"(total {np.mean(d_lay ** 2):.2f}):")
    print(f"    {'':3s} {'alone':>8s} {'drop-one':>9s} {'pm floor':>9s} "
          f"{'layer/floor':>12s}")
    for i, nm in enumerate(["Mf", "Mr", "M1", "M2", "Mc"]):
        keep = np.ones(5)
        keep[i] = 0.0
        alone = np.mean(pm_lay[:, :, i] ** 2)
        d_wo = np.einsum("bm,bam,m->ba", score, dq, keep)
        fl = np.mean(pm_fit[:, :, i] ** 2)
        print(f"    {nm:3s} {alone:8.2f} {np.mean(d_wo ** 2):9.2f} {fl:9.2f} "
              f"{alone / fl:12.1f}")

    # The layer is fitted to the SECOND derivative too, and `_score_mse` summed
    # both; report the split so the 390 is attributable.
    r_lay = np.asarray(eqx.filter_jit(jax.vmap(dm_dg, in_axes=(None, 0, None)))(
        layer, jnp.asarray(m), chart)[1], np.float64)
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

    for name, r in (("Mr/Mf", Mr / Mf), ("Mf", Mf)):
        qs = np.quantile(r, np.linspace(0, 1, NBINS + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        print(f"\n{'<' + name + '>':>10s} {'layer':>8s} {'floor':>8s} "
              f"{'<Q1^2>':>9s} {'floor/<Q1^2>':>13s} {'m1':>9s}")
        for k in range(NBINS):
            s = (r >= qs[k]) & (r < qs[k + 1])
            lo, fl = np.mean(d_lay[s] ** 2), np.mean(d_fit[s] ** 2)
            q2 = np.mean(y[s, 0] ** 2)
            print(f"{r[s].mean():10.4g} {lo:8.2f} {fl:8.2f} {q2:9.2f} "
                  f"{fl / q2:13.4f} {m1(s):+9.4f}")


if __name__ == "__main__":
    main()
