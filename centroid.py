"""
centroid.py
===========
Phase 3, now `check`-only: validate `models.centroid.CentroidMarginalize` (a
closed-form, zero-parameter transport -- see its module docstring) against a
catalog of shifted template copies.  `train` is kept only for its warm-start
graft (loading bulk+shear from `--init`); the centroid layer itself has
nothing left to fit by likelihood.

The validation set is not the galaxies -- it is their *copies*.
`imsims.copies` replicates every template galaxy over a grid of coordinate
origins u and records the moments m(u) and first moments X(u) of each, which is
the left-hand side of the paper's eq. (36).  The weight of a copy under a target
whose first moments have covariance Sigma_X is

    w(u) = d2u . |J(u)| . N(X(u); 0, Sigma_X)

so the marginalised prior is the w-weighted distribution of the m(u).  That is
what `check` compares `CentroidMarginalize`'s analytic transport against.

Why draw rather than weight
---------------------------
Copy weights span e^-8 across a galaxy's grid, so a flat batch of copies has an
effective sample size of ~13% of its length -- the classic reason this needed
self-normalised importance sampling before.  It does not need it here.  The
weights of ONE galaxy's copies sum to that galaxy's detection probability, and
`imsims` measures that as 1.00000 for every galaxy when sn_min = 0.  So drawing
galaxies uniformly and then one copy each in proportion to w is exact
stratification at full ESS, with no importance weights anywhere.

That equality is a property of the no-selection case, not a general one.  Turn on
a flux cut and the per-galaxy sum becomes P(s|G) < 1 and genuinely varies between
galaxies; `CopySampler` would then have to carry it as a per-galaxy weight
instead of normalising it away.  `tests/test_centroid.py::test_sampler_is_the
_weighted_distribution` and imsims' own
`test_copy_weights_sum_to_the_detection_probability` are what license it today.

Staging
-------
Sigma_X is the same for every galaxy in the current sims (fixed noise level,
circular PSF), so this stage asks only whether the layer can represent the
marginalisation at all -- the conditioning is on a constant.  `--sigma-scale`
reweights the same catalog to a different Sigma_X without regenerating it, which
is what the catalog's factor-1.4 grid margin is for, and is the cheap way to see
the layer's Sigma_X dependence before the sims grow a varying noise level and
then an elliptical PSF.

Usage:
    python centroid.py train --copies ../bfd_cnf_imsims/data/copies_bulgedisc.fits
    python centroid.py check --copies ../bfd_cnf_imsims/data/copies_bulgedisc.fits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import bulk
from models.bijections import in_domain, safe_point
import shear
from models.centroid import dm_dsigma
from models.shear import G_MAX
from shear import sample_g

# The copy-weight convention lives in imsims and must not be duplicated here: it
# is the one place the |J| choice of eq. (35) vs (36) is made, and two copies of
# that would drift.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bfd_cnf_imsims"))
try:
    from imsims.copies import log_weights
except ImportError as exc:                                  # pragma: no cover
    raise SystemExit("centroid.py needs ../bfd_cnf_imsims on the path: "
                     f"{exc}") from None

MOMENT_LABELS = ["Mf", "Mr", "M1", "M2", "Mc"]


def load_copies(path):
    """Read an `imsims.copies` catalog -> (copies, galaxies)."""
    with fitsio.FITS(path) as f:
        return f["COPIES"].read(), f["GALAXIES"].read()


class CopySampler:
    """Draw one shifted copy per galaxy, in proportion to its eq.-36 weight.

    Galaxies are drawn in proportion to the SUM of their copy weights, not
    uniformly.  That sum is the galaxy's detection probability, and it is not
    always 1: on a shallow catalog it is 1.00000 for every galaxy, but at
    noise_sigma = 2.73 the faintest ~1% fall to 0.98 and the worst to 0.69,
    because their centroid distribution reaches the `det J <= 0` boundary that
    `makeTemplates` cuts on -- the paper's positive-Jacobian assumption (sec.
    5.3) failing at low S/N, not a grid that is too small (none of them come
    near `XY_MAX`).  Those galaxies really are less likely to be detected at all,
    so normalising the sum away would over-weight exactly the objects the
    formalism represents worst.

    Within a galaxy the copies are then drawn by weight, which is exact
    stratification at full ESS -- the point of doing it this way rather than
    self-normalised importance weighting over a flat batch of copies, where the
    e^-8 spread of the weights leaves an ESS of ~13% of the batch.

    Copies must be grouped by `gal`, which `imsims.copies` writes them as.
    """

    def __init__(self, copies, sigma_x):
        w = np.exp(log_weights(copies, sigma_x))
        self.m = np.ascontiguousarray(copies["moments"], dtype=np.float64)
        # Each copy's OWN exact shear derivatives, when the catalog carries them.
        # A copy is the parent measured about a shifted origin, so the parent's
        # derivatives do NOT transfer -- these have to come from the catalog, and
        # `imsims.copies` now stores them (bfd's `makeTemplates` computed them all
        # along; only slot D0 used to be kept).  Without them the layer can only
        # be trained at g = 0.
        names = copies.dtype.names
        self.dm = (np.ascontiguousarray(copies["dm_dg"], dtype=np.float64)
                   if "dm_dg" in names else None)
        self.d2m = (np.ascontiguousarray(copies["d2m_dg2"], dtype=np.float64)
                    if "d2m_dg2" in names else None)
        # One running cumulative sum serves every galaxy: within a group the CDF
        # is just cw shifted by the group's base, so a single searchsorted over
        # the whole array does the per-galaxy draw for the entire batch at once.
        self.cw = np.cumsum(w)
        gal = copies["gal"]
        # FITS hands back big-endian ints, which JAX will not accept.
        self.ids = np.unique(gal).astype(np.int64)
        self.starts = np.searchsorted(gal, self.ids, side="left")
        self.ends = np.searchsorted(gal, self.ids, side="right")
        self.base = self.cw[self.starts] - w[self.starts]
        self.total = self.cw[self.ends - 1] - self.base
        # P(s|G) as a galaxy-selection probability -- see the class docstring.
        self._gal_cdf = np.cumsum(self.total)
        self._gal_cdf /= self._gal_cdf[-1]

    @property
    def n_galaxies(self):
        return len(self.ids)

    def ess(self):
        """Effective sample size per galaxy -- the paper's own sec.-2.5 diagnostic."""
        w = np.diff(np.concatenate([[0.0], self.cw]))
        num = np.add.reduceat(w, self.starts) ** 2
        den = np.add.reduceat(w * w, self.starts)
        return num / den

    def draw(self, rng, batch):
        """`batch` galaxies drawn by detection probability, one copy each by weight.

        Returns (copy moments, galaxy row indices, dm/dg, d2m/dg2) -- the second
        so the caller can line the batch up with per-galaxy quantities, and the
        last two so it can LENS the drawn copies.  The derivatives are None on a
        catalog rendered before `imsims.copies` began storing them.
        """
        gi = np.searchsorted(self._gal_cdf, rng.random(batch))
        gi = np.clip(gi, 0, len(self.ids) - 1)
        target = self.base[gi] + rng.random(batch) * self.total[gi]
        idx = np.searchsorted(self.cw, target, side="left")
        # Guard the group edges against float error in the cumulative sum.
        idx = np.clip(idx, self.starts[gi], self.ends[gi] - 1)
        return (self.m[idx], self.ids[gi],
                None if self.dm is None else self.dm[idx],
                None if self.d2m is None else self.d2m[idx])

    @property
    def has_derivatives(self):
        return self.dm is not None and self.d2m is not None


def _centroid_layer(flow):
    """The CentroidMarginalize inside the built flow.

    Located BY TYPE: the chart is bijections[0] now, so an index here would
    return the chart and train it instead of this layer.
    """
    from models.centroid import CentroidMarginalize
    for b in flow.bijection.bijection.bijections:
        if isinstance(b, CentroidMarginalize):
            return b
    raise ValueError("no CentroidMarginalize in this flow")


def _chart(flow):
    """The RawMomentStandardize, which `dm_dsigma` needs to convert its z-space
    shift back to the raw moments the copy catalog is measured in."""
    return flow.bijection.bijection.bijections[0]


def _trainable(flow):
    """Filter spec selecting only the centroid layer.

    Same reasoning as `shear._trainable`: the Sigma_X dependence is a small part
    of the total likelihood, so a free bulk will absorb batch noise and drown it.
    """
    spec = jax.tree.map(lambda _: False, flow)
    return eqx.tree_at(
        lambda f: _centroid_layer(f), spec,
        replace=jax.tree.map(eqx.is_inexact_array, _centroid_layer(flow)))


def condition(sigma_x, batch, g=(0.0, 0.0)):
    """The chain's condition vector [g1, g2, C00, C01, C11], tiled over a batch.

    Training used to run only at g = 0, because the copies were unlensed and
    there was nothing to lens them with.  There is now: `imsims.copies` stores
    each copy's own exact dm/dg and d2m/dg2, so `train` samples g over the shear
    disc and lenses the drawn copies by their own derivatives, exactly as
    `shear.train` does with templates.  That is what gives the layer's
    g-conditioned coefficients (`models/centroid.g_invariants`) a gradient --
    at g = 0 their inputs are identically zero and they never move.
    """
    row = jnp.concatenate([jnp.asarray(g, dtype=jnp.float32),
                           jnp.asarray(sigma_x, dtype=jnp.float32)])
    return jnp.tile(row, (batch, 1))


def condition_batch(sigma_x, g):
    """Per-row ``[g1, g2, C00, C01, C11]`` for a batch with its own shears."""
    return jnp.concatenate(
        [jnp.asarray(g, dtype=jnp.float32),
         jnp.broadcast_to(jnp.asarray(sigma_x, dtype=jnp.float32),
                          (len(g), 3))], axis=-1)


def train(flow, sampler, sigma_x, key, steps=4000, batch=1024,
          lr=3e-3, g_max=G_MAX):
    """Likelihood only -- see `shear.train` for the reasoning.

    `models/centroid.py`'s `CentroidMarginalize` is now a closed-form,
    zero-parameter transport (see its module docstring) -- there is nothing
    left in the centroid layer for this loop to optimise.  Kept, rather than
    deleted, because `main()`'s warm-start graft (loading bulk+shear from
    `--init`) still runs through this function's caller; this just skips the
    now-pointless gradient loop instead of spending `steps` NLL evaluations
    on an empty parameter set.

    `g_max` is retired along with the layer's (now nonexistent)
    shear-conditioned coefficients; kept as an accepted-but-unused argument
    only so existing call sites do not need to change.
    """
    # `_trainable`'s boolean spec does not use paramax's documented
    # `is_leaf=isinstance(..., NonTrainable)` convention, so `mean`/`std`
    # still show up as nominal leaves of `params` below even though they are
    # frozen -- `NonTrainable.unwrap()` applies `stop_gradient` internally,
    # so their gradient (and Adam's update from it) is exactly zero either
    # way.  Checked here with the CORRECT `is_leaf`, so this guard actually
    # detects "nothing real to train" instead of never firing.
    from paramax.wrappers import NonTrainable
    has_free_params = any(jax.tree.leaves(jax.tree.map(
        eqx.is_inexact_array, _centroid_layer(flow),
        is_leaf=lambda x: isinstance(x, NonTrainable))))
    if not has_free_params:
        print("centroid layer has no trainable parameters (closed-form "
              "transport) -- nothing to train.")
        return flow
    if g_max and not sampler.has_derivatives:
        raise ValueError(
            "this copies catalog carries no per-copy dm_dg/d2m_dg2, so its "
            "copies cannot be lensed. Regenerate it with imsims.copies (which "
            "now stores them), or pass --g-max 0 to train at g = 0 as before.")
    cond0 = condition(sigma_x, batch)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(optax.cosine_decay_schedule(lr, steps)))
    params, static = eqx.partition(flow, _trainable(flow))
    state = opt.init(params)
    rng = np.random.default_rng(int(jr.randint(key, (), 0, 2**30)))

    @eqx.filter_jit
    def step(params, state, x, cond):
        def loss_fn(p):
            model = eqx.combine(p, static)
            # The chart is now the FIRST thing applied, so nothing downstream
            # of it -- this layer included -- can carry a point across a
            # point-source ceiling.  The only domain question left is whether
            # the raw copy itself is on-chart, which is a property of the data
            # and does not depend on the layer at all.  (Under the old ordering
            # this had to un-marginalise first and test the RESULT, because
            # un-marginalising pushed Mr/Mf and Mc/Mr up toward their ceilings.)
            #
            # Masking the RESULT is still not enough: `jnp.where` differentiates
            # the -inf branch and hands back NaN, which took out an entire 8000
            # step deep run at step 144.  So substitute the INPUT and drop those
            # rows from the mean.
            ok = in_domain(x)
            x_safe = jnp.where(ok[:, None], x, safe_point(x))
            lp = model.log_prob(x_safe, condition=cond)

            # A SECOND, independent failure mode: an off-centre copy can carry
            # every chart coordinate in a perfectly ordinary range (in_domain
            # true, no coordinate more than a few sigma out) while the
            # COMBINATION sits off the bulk flow's training manifold in a
            # direction its autoregressive stack extrapolates catastrophically
            # -- measured on copies_bulgedisc_deep.fits: one such row's z was
            # [-1.2, -2.8, -3.3, -0.9, -2.4] (nothing extreme) yet its |z|
            # after the bulk's 8 layers reached 3e5, and log_prob -4.4e10, on
            # the very FIRST batch, with the flow's INITIAL (pre-training)
            # parameters -- so this is not a training instability, it is a
            # standing numerical cliff in the bulk stack that ordinary
            # moments.fits batches apparently never sample and copies
            # (off-centre measurements) sometimes do.  `in_domain` cannot see
            # it because it is a property of the TRAINED FLOW's response to a
            # coordinate, not of the raw moments.  A normal row's nll is
            # O(10-100); flag anything past 1e4 as this same kind of poison
            # and substitute it the same way `in_domain` violations are, via a
            # second evaluation so `jnp.where` cannot leak its gradient
            # (exactly the mechanism the comment above already warns about).
            sane = jax.lax.stop_gradient(lp > -1e4)
            ok = ok & sane
            x_safe = jnp.where(ok[:, None], x_safe, safe_point(x_safe))
            lp = model.log_prob(x_safe, condition=cond)

            nll = -jnp.sum(jnp.where(ok, lp, 0.0)) / jnp.maximum(jnp.sum(ok), 1)
            return nll, (nll, 1.0 - jnp.mean(ok))

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = opt.update(grads, state, params)
        return eqx.apply_updates(params, updates), state, aux

    for i in range(steps):
        x, _, dm, d2m = sampler.draw(rng, batch)
        if g_max:
            # Antithetic +g/-g, as `shear._antithetic` does: the spin-0 response
            # is odd in e and cancels at O(g) over an isotropic population, so
            # pairing the batch removes that cancellation's variance instead of
            # waiting for it to average out.
            gk, key = jr.split(key)
            half = sample_g(gk, batch // 2, g_max)
            g = jnp.concatenate([half, -half])
            x = shear.lens(jnp.asarray(x), jnp.asarray(dm), jnp.asarray(d2m), g)
            cond = condition_batch(sigma_x, g)
        else:
            x, cond = jnp.asarray(x), cond0
        params, state, (nll, off) = step(params, state, x, cond)
        # `off` is the fraction of copies the layer maps out of the chart, and
        # it is the diagnostic for the guard above: those copies contribute no
        # gradient, so nothing stops the layer pushing MORE of them out.  It
        # measured ~1e-4 and flat at the deep depth; a rising trend means the
        # map needs bounding rather than masking (see models/centroid.py).
        if i % 250 == 0 or i == steps - 1:
            print(f"step {i:5d}  nll {nll:.4f}  off-chart {off:.2e}")
    return eqx.combine(params, static)


def weighted_copy_mean(copies, galaxies, sigma_x):
    """Per-galaxy w-weighted mean of the copy moments -- the layer's target.

    This is what the marginalisation does to each galaxy, straight from the
    catalog and with no flow involved.
    """
    w = np.exp(log_weights(copies, sigma_x))
    n = len(galaxies)
    den = np.bincount(copies["gal"], weights=w, minlength=n)
    num = np.stack([np.bincount(copies["gal"], weights=w * copies["moments"][:, j],
                                minlength=n) for j in range(5)], axis=1)
    keep = den > 0
    return num[keep] / den[keep, None], keep


def check(flow, copies, galaxies, sigma_x, n=4000):
    """Compare the layer's own shift with the catalog's weighted copy means.

    Diagnostic only, exactly as `shear.check` is: the layer is a transport that
    matches DENSITIES, so it is not obliged to reproduce any single galaxy's
    marginalisation -- only the conditional mean of it, which is what the columns
    below measure.  The scatter that no deterministic transport can carry is the
    `Var[.|m]` floor of the module docstring.
    """
    target, keep = weighted_copy_mean(copies, galaxies, sigma_x)
    m0 = galaxies["moments"][keep][:n]
    target = target[:n]
    layer, chart = _centroid_layer(flow), _chart(flow)
    sx = jnp.asarray(sigma_x, dtype=jnp.float32)
    pred = np.asarray(jax.vmap(dm_dsigma, in_axes=(None, 0, None, None))(
        layer, jnp.asarray(m0, dtype=jnp.float32), sx, chart))
    truth = target - m0

    # Normalise by each moment's own magnitude; the spin-2 pair by Mr, since M1
    # and M2 pass through zero (same convention as shear._scale).
    scale = np.stack([m0[:, 0], m0[:, 1], m0[:, 1], m0[:, 1], m0[:, 4]], axis=1)
    print(f"  {'':4s} {'layer shift':>13s} {'catalog shift':>14s} {'resid':>10s}")
    for j, lab in enumerate(MOMENT_LABELS):
        p, t = pred[:, j] / scale[:, j], truth[:, j] / scale[:, j]
        print(f"  {lab:4s} {p.mean():+13.3e} {t.mean():+14.3e} "
              f"{np.sqrt(np.mean((p - t) ** 2)):10.3e}")

    # The number that decides whether this was worth doing: how much of the
    # ellipticity amplification the layer reproduces.
    def e(m):
        return (m[:, 2] + 1j * m[:, 3]) / m[:, 1]

    e0 = e(m0)
    resp = lambda mm: ((e(mm) * np.conj(e0)).real.sum()
                       / (e0 * np.conj(e0)).real.sum() - 1.0)
    print(f"  ellipticity response  catalog {resp(target):+.4e}   "
          f"layer {resp(m0 + pred):+.4e}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "check"])
    p.add_argument("--copies", default="../bfd_cnf_imsims/data/copies_bulgedisc.fits")
    p.add_argument("--flow", default="flows/centroid.eqx")
    p.add_argument("--init", default="flows/shear.eqx",
                   help="shear checkpoint to warm-start bulk+shear from")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--sigma-scale", type=float, default=1.0,
                   help="rescale Sigma_X by this factor squared; the catalog "
                        "grid is valid over roughly [1/1.4, 1.4]")
    p.add_argument("--g-max", type=float, default=G_MAX,
                   help="radius of the shear disc the drawn copies are lensed\n"
                        "over, using each copy's OWN exact derivatives. Must be\n"
                        "> 0 for the layer's g-conditioned coefficients to get\n"
                        "any gradient -- their inputs vanish at g = 0. Set 0 for\n"
                        "the old g = 0 behaviour, or for a catalog with no\n"
                        "per-copy derivatives.")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    copies, galaxies = load_copies(a.copies)
    sigma_x = galaxies["cov_odd"][0] * a.sigma_scale**2
    sampler = CopySampler(copies, sigma_x)
    ess = sampler.ess()
    print(f"{sampler.n_galaxies} galaxies, {len(copies)} copies, "
          f"Sigma_X = {np.round(sigma_x, 1)}")
    print(f"copies/galaxy ESS: median {np.median(ess):.0f}, "
          f"5th pct {np.percentile(ess, 5):.0f}")

    # Standardise the bulk against the unmarginalised galaxy moments, which is
    # the population the frozen bulk was trained on.
    m_train = np.asarray(galaxies["moments"], dtype=np.float64)
    flow = bulk.build_flow(jr.key(a.seed), m_train, shear=True, centroid=True)

    if a.mode == "train":
        if a.init:
            prior = eqx.tree_deserialise_leaves(
                a.init, bulk.build_flow(jr.key(a.seed), m_train, shear=True))
            pb = prior.bijection.bijection.bijections
            # prior is [raw2standard, shear, *bulk]; this chain inserts the
            # centroid layer after the chart, so it is
            # [raw2standard, centroid, shear, *bulk].  Graft the chart and the
            # bulk wholesale, and the shear layer's PARAMETERS rather than the
            # layer itself -- the prior's copy carries cond_shape (2,) and this
            # chain needs (5,).
            flow = eqx.tree_at(lambda f: f.bijection.bijection.bijections[0],
                               flow, pb[0])
            flow = eqx.tree_at(lambda f: f.bijection.bijection.bijections[3:],
                               flow, pb[2:])
            flow = eqx.tree_at(lambda f: f.bijection.bijection.bijections[2].coeffs,
                               flow, pb[1].coeffs)
            # Both grafted layers' frozen chart copies must match the chart
            # they now sit behind -- `bulk.train` moves the chart's mean/std,
            # so what `build_flow` copied from the raw sample is stale.  This
            # used to fix only the centroid layer; leaving the SHEAR layer
            # stale was the whole of its dm/dg error peak at Mr/Mf ~ 3.1.
            # It also re-points the centroid layer's own frozen COPY of this
            # shear layer (`models.centroid.shear_delta_invariants`) at the
            # just-grafted one -- otherwise it would go on delensing with the
            # PRE-graft (freshly initialised) shear coefficients instead of
            # the warm-started ones the chain itself now uses.  Passing `m_train`
            # also resyncs the shear layer's coefficient-net whitening stats
            # (`coeffs.u_mean/.u_white`), stale by the same mechanism.
            flow = bulk.sync_chart_constants(flow, m_train=m_train)
            print(f"warm started bulk + shear from {a.init}")
        flow = train(flow, sampler, sigma_x,
                     jr.key(a.seed + 1), steps=a.steps, batch=a.batch,
                     g_max=a.g_max)
        eqx.tree_serialise_leaves(a.flow, flow)
        print(f"wrote {a.flow}")
    else:
        flow = eqx.tree_deserialise_leaves(a.flow, flow)
    check(flow, copies, galaxies, sigma_x)


if __name__ == "__main__":
    main()
