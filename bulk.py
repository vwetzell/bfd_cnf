"""
bulk.py
=======
Phase 1: learn p(m) for the imsims Gaussian-galaxy population, using only the
*bulk* (unconditional) layers of the bfd_cnf flow.  m is the five even moments
[Mf, Mr, M1, M2, Mc].

The end goal is P(m | g, Sigma_X) of Bernstein et al. 2016 (MNRAS 459, 4467),
where m = [Mf, Mr, M1, M2] are the four even moments, g the shear and Sigma_X the
covariance of the odd/centroid moments the prior is marginalised over.  That flow
is built as

    base -> bulk -> shear(g) -> Sigma_X -> data

(`models.bijections.new_masked_autoregressive_flow`).  Here we build only the
first arrow and its data-space coordinate change, i.e. the same stack with the
two conditional layers removed:

    RawMomentStandardize  o  N x [EquivariantAutoregressiveLayer, Permute]

so the conditional layers can be slotted in later without retraining the bulk
from scratch.  Sigma_X is homoscedastic in the phase-1 sims (a pure noise
covariance under a fixed PSF and noise level), so conditioning on it would be
learning a constant -- which is exactly why it is left out for now.

Usage:
    python bulk.py train  --data ../bfd_cnf_imsims/data/moments.fits
    python bulk.py corner --data ../bfd_cnf_imsims/data/moments.fits
"""

from __future__ import annotations

import argparse

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

# XLA picks matmul precision PER PROCESS, and on this flow that is worth ~1
# nat in the log-densities ([[tf32-breaks-paired-runs]]; `bias.py` has forced
# this since f1f3c92).  Training left unpinned is worse than imprecise, it is
# not comparable: two processes at the SAME seed landed at even log-dets of
# 0.019 and 1.430 -- different basins, not scatter -- which silently voided a
# paired experiment on 2026-08-20.  Costs ~7%.
jax.config.update("jax_default_matmul_precision", "highest")

from flowjax.bijections import Chain, Invert, Permute
from flowjax.distributions import MultivariateNormal, Transformed
from paramax import non_trainable

from models.bijections import (EquivariantAutoregressiveLayer,
                               RawMomentStandardize, SigmaXBlockLayer,
                               in_support)
from models.centroid import CentroidMarginalize  # noqa: F401 -- kept importable, see build_flow
from models.shear import ShearResponse

# All six permutations of the three spin-0 slots.  These are the orderings each
# autoregressive layer should SEE -- see `_spin0_increments` for why that is not
# the same as the permutation inserted between layers.
_SPIN0_PERMS = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [2, 1, 0], [1, 0, 2]]


def _spin0_increments(n_layers):
    """Permutations to insert BETWEEN layers so the cumulative ordering walks
    `_SPIN0_PERMS`.

    `Permute` sits after each autoregressive layer, so the orderings COMPOSE:
    what layer i sees is perm_0 . perm_1 . ... . perm_{i-1}, not perm_i.
    Inserting `_SPIN0_PERMS[i]` directly -- which this did -- therefore does not
    cycle the orderings at all.  Measured over the 8 layers it gave

        z0 (log10 Mf) the autoregressive head in 4 of 8 layers, against 2 each
        for z1 and z2, and presented only 4 of the 6 orderings,

    so the flux coordinate spent half the stack modelled unconditionally and
    some pairwise dependences were only ever seen in one direction -- both of
    which the design comment says must not happen.  The first inserted
    permutation was also the identity, so layers 0 and 1 saw the same ordering.

    `Permute` maps y[k] = x[perm[k]], so to go from ordering `cur` to `nxt` the
    increment is `cur^-1 . nxt`.
    """
    out = []
    for i in range(n_layers):
        cur, nxt = _SPIN0_PERMS[i % 6], _SPIN0_PERMS[(i + 1) % 6]
        pos = {v: k for k, v in enumerate(cur)}
        out.append([pos[v] for v in nxt])
    return out

LAYERS = 8
NN_WIDTH = 64
NN_DEPTH = 2

# The flow works in t = [log10(Mf), logit(Mr/Mf / r*), logit(Mc/Mr / rc*),
# M1/Mr, M2/Mr] (RawMomentStandardize), standardised by the training set's own
# mean/std -- spin-0 first, spin-2 last.  Label the corner plot in it.
#
# Slots 1 and 2 are the BARE RATIOS.  They were logits once, and these labels
# went on saying so after `_forward_transform` dropped them -- so a reader
# comparing a plotted 3.15 against a catalog's Mr/Mf of 3.15 was told the axis
# was a logit.  Read `RawMomentStandardize._forward_transform`, not this
# comment, if they ever disagree again: that function IS the chart.
COORD_LABELS = [r"$\log_{10}M_f$", r"$M_r / M_f$", r"$M_c / M_r$",
                r"$M_1 / M_r$", r"$M_2 / M_r$"]
LABEL_FONTSIZE, TICK_LABELSIZE = 34, 24
# Mr/Mf ceiling: a point source, i.e. the PSF itself.  Anything at or above it is
# unresolved and carries no shape information -- the old repo drew it as the
# "stellar locus", and it is the same number here (same weight function).  It is
# now a hard boundary of the flow's chart, so it is defined next to the chart.
# Mc/Mr ceiling, the identical point-source argument one moment order up --
# see models.bijections.POINT_SOURCE_MC.
from models.bijections import POINT_SOURCE, POINT_SOURCE_MC  # noqa: E402,F401
from models.shear import _Q_LOC, _Q_SCALE  # noqa: E402

# The target selection the real analysis applies, recorded here (not in
# bias.py, which shear.py cannot import without a cycle) so it is not
# re-invented per script.  A size window in Mr/Mf and a flux window in Mf, as
# on sky.  These are NOT yet wired into `bias.main` -- eq. (40)/(46)'s
# selection terms are owed before a cut may be applied to a bias measurement
# -- but they are what `dev/check_dmdg_components.py` restricts its
# comparison to, since the response outside the window is never used.
#
# Measured on bulgedisc: the size window keeps 76.4% (4.9% below 2.2, 18.7%
# above 3.5) and the flux window 94.9% standalone but only 3.1% more inside the
# size window -- the two are largely redundant, because the faint galaxies are
# mostly the small-Mr/Mf ones already cut (median Mf 2548 below Mr/Mf = 2.2
# against 5202 above).  Both together keep 74.1%.
SIZE_WINDOW = (2.2, 3.5)          # Mr/Mf
FLUX_WINDOW = (2500.0, 50000.0)   # Mf


def load_moments(path):
    """Read the five even moments [Mf, Mr, M1, M2, Mc] from an imsims catalog."""
    return np.asarray(fitsio.read(path)["moments"], dtype=np.float64)


def to_coords(m, flux_sas=None):
    """Raw moments -> the flow's transformed coordinates t (for plotting).

    Mirrors `RawMomentStandardize._forward_transform`, including slots 1 and
    2's logits: this is what fixes the standardisation's mean/std, so the two
    must not drift apart.
    """
    f = np.log10(m[:, 0])
    if flux_sas is not None:
        mu, sig, a, b = flux_sas
        f = np.sinh((np.arcsinh((f - mu) / sig) - a) / b)
    return np.stack([f, m[:, 1] / m[:, 0], m[:, 4] / m[:, 1],
                     m[:, 2] / m[:, 1], m[:, 3] / m[:, 1]], axis=-1)


def coeff_stats(t, mean=None, std=None):
    """(mean, W) whitening the ShearResponse coefficient net's four inputs.

    `t` is the chart output for the training set.  The net sees
    `[z0, z1, z2, (|e|^2 - 2)/2]` in STANDARDISED coordinates, so reproduce that
    here and return the mean and inverse Cholesky factor of its covariance.
    See `models.shear._Coeffs.__call__` for why: those inputs have condition
    number ~2600 as they stand, with z1 and z2 correlated at 0.997.

    `mean`/`std` default to `t`'s own (matching `build_flow`'s freshly
    initialised chart, whose `RawMomentStandardize.mean/std` ARE `t.mean(0)`/
    `t.std(0)` at that point).  Pass the chart's TRAINED mean/std instead to
    recompute against the distribution `_Coeffs` actually sees post-training
    -- `RawMomentStandardize.mean/std` are trainable and move substantially
    (see `sync_chart_constants`), so the whitening built here at init goes
    stale exactly the way `chart_loc`/`chart_scale`/`e_scale` do.
    """
    mean = t.mean(0) if mean is None else np.asarray(mean)
    std = t.std(0) if std is None else np.asarray(std)
    z = (t - mean) / std
    q = z[:, 3] ** 2 + z[:, 4] ** 2
    u = np.stack([z[:, 0], z[:, 1], z[:, 2], (q - _Q_LOC) / _Q_SCALE], axis=-1)
    cov = np.cov(u.T) + 1e-8 * np.eye(4)
    return u.mean(0), np.linalg.inv(np.linalg.cholesky(cov))


def build_flow(key, m_train, layers=LAYERS, nn_width=NN_WIDTH, nn_depth=NN_DEPTH,
               shear=False, centroid=False,
               flux_sas=None):
    """Bulk flow standardised against `m_train`; conditioned on g and/or Sigma_X.

    The generative stack is ``base -> bulk -> shear(g) -> centroid(Sigma_X) ->
    data``, and data -> base the chain is
    ``[raw2standard, centroid?, shear, *bulk]``.

    Since 16ece5c made the chart data-adjacent the conditional layers act in
    STANDARDISED coordinates, not raw moment space as this said before.  What
    keeps their coefficients comparable to the physics -- bfd's dm/dg for
    shear, the weighted copy means for centroid -- is that the supervision
    composes the chart back in (`shear.dm_dg(layer, m, chart)`), not the
    layers' own coordinates.  Centroid comes last because it happens last -- a galaxy is lensed
    on the sky and only then measured about a centroid somebody had to guess.
    """
    # Loud, not silent: slot 1's chart is undefined at or above the ceiling, so
    # a training set containing such a row would produce NaNs several layers
    # away from the cause.  Noisy moments DO reach up there -- they must be fed
    # through the convolution as latent draws, never straight into `log_prob`.
    t = to_coords(m_train, flux_sas)
    raw2standard = RawMomentStandardize(mean=t.mean(0), std=t.std(0),
                                        flux_sas=flux_sas)
    _u_stats = coeff_stats(t)

    key, k_shear, k_centroid = jr.split(key, 3)
    keys = jr.split(key, layers)
    _incs = _spin0_increments(layers)
    bulk = []
    for i, k in enumerate(keys):
        # Exactly ONE layer gets the spin-2 radial bend, the last one: the tail
        # index composes multiplicatively across layers, so a stack of them
        # bounds nothing (see `Spin2CouplingLayer`).  Last rather than first
        # because it acts on coordinates already close to the standard normal
        # base, where the map it has to make is the most predictable.
        bulk.append(EquivariantAutoregressiveLayer(
            k, nn_width, nn_depth, jax.nn.silu, bend=(i == 0)))
        # Cycle through all six orderings of the three SPIN-0 coordinates, so no
        # one of them is permanently the unconditional head of the autoregression
        # and every pairwise dependence gets modelled in both directions.  The
        # spin-2 pair (3, 4) is never touched: swapping M1/M2 would rotate the
        # shape by 45 degrees and destroy the equivariance.
        bulk.append(Permute(jnp.array(_incs[i] + [3, 4])))

    # Chain.transform runs in list order and maps data -> base, so the
    # data-adjacent layer comes first; Invert flips it for sampling.
    #
    # `raw2standard` is FIRST.  It is the fixed reparametrisation that brings raw
    # moments into a shape the flow can learn -- unbounded, roughly Gaussian --
    # and it is where the chart's point-source ceilings live.  Putting it at the
    # data boundary means the chart is enforced exactly ONCE, and no map
    # downstream of it can push a point off-chart.  Under the previous ordering
    # the conditional layers acted on raw moments BEFORE the chart, so a draw
    # that `bias.log_conv_is` had passed as in-domain could still be carried
    # across a ceiling by the shear layer -- a g-DEPENDENT seam.  That is gone by
    # construction here.
    #
    # After it, generative order still reads base -> bulk -> shear(g) ->
    # centroid(Sigma_X) -> data, so centroid is generatively last and hence
    # first among the conditional layers in this data -> base list.
    #
    # The cost: the conditional layers now respond in z, not in raw moments, so
    # their coefficients are no longer directly comparable to bfd's dm/dg.  See
    # models/shear.py's header.
    #
    # With both on, the chain carries one condition vector for the pair:
    # [g1, g2, C00, C01, C11].  Shear reads the first two, centroid the last three.
    cond = 5 if (shear and centroid) else None
    shear_layer = (ShearResponse(
        k_shear, cond_dim=cond or 2,
        # The chart's EFFECTIVE spin-2 std, not slot 3's own.
        # `RawMomentStandardize._effective` symmetrises the pair to
        # sqrt((s3^2 + s4^2)/2) -- that is what z3 and z4 are actually
        # divided by, so anything else here makes `response`'s "physical
        # units" wrong by the ratio.  It was `t.std(0)[3]`, which is 0.67%
        # off on bulgedisc and 1.8% off on a 1000-row subsample;
        # `dev/reparam_paired.py`'s constant-coefficient check is what
        # turned it up.
        e_scale=float(np.sqrt(0.5 * (t.std(0)[3] ** 2 + t.std(0)[4] ** 2))),
        u_mean=_u_stats[0], u_white=_u_stats[1],
        chart_loc=t.mean(0)[:3], chart_scale=t.std(0)[:3])
        if shear else None)
    # SigmaXBlockLayer (models/bijections.py) replaces CentroidMarginalize here:
    # a closed-form, invertible-by-construction block on the standardised
    # z-coordinates, conditioned on the same [g1, g2, C00, C01, C11] vector
    # (it converts C00/C01/C11 -> [log_scale, e1, e2] internally, see its
    # `_unpack`). `log_scale_mean`/`e_max` are calibrated to this population's
    # own Sigma_X (see `bulk.build_flow`'s docstring / HANDOFF for the
    # measurement) -- do not copy them to a different population without
    # re-checking `cov_odd`.
    # `size_loc` anchors net_size's multiplicative kappa at the chart's OWN
    # pre-standardisation mean/std of slot 1 (Mr/Mf logit), same `t` used to
    # build `raw2standard` a few lines above -- see `SigmaXCouplingLayer`'s
    # docstring ("c1 = mu1/sigma1 ... equals kappa x (Mr/Mf) on the
    # un-centred ratio").  Without it, `y1 = kappa*x1` is a pure scaling
    # through zero, and since `x1` is chart-standardised to zero population
    # mean, no constant kappa can produce a net population-level size shift
    # -- exactly why net_size never moved off its zero-shift plateau.
    head = ([SigmaXBlockLayer(k_centroid, full_cond_dim=cond or 3,
                              log_scale_mean=10.8, e_max=0.1,
                              size_loc=float(t.mean(0)[1] / t.std(0)[1]))]
            if centroid else []) + \
           ([shear_layer] if shear else [])
    bijection = Invert(Chain([raw2standard, *head, *bulk]).merge_chains())
    base = non_trainable(MultivariateNormal(jnp.zeros(5), jnp.eye(5)))
    return Transformed(base, bijection)


def chart_of(flow):
    """The `RawMomentStandardize` at the data boundary of a built flow.

    `build_flow` puts it first in the data -> base chain, and `Invert` wraps the
    chain, so it is `bijections[0]` whether or not the conditional layers are
    present.
    """
    return flow.bijection.bijection.bijections[0]


class SupportedFlow(eqx.Module):
    """A prior with an explicit physical support and a defensive density floor.

    Two independent corrections to what `flow.log_prob` returns, both of which
    exist because NLL training says NOTHING about regions with no training data,
    while the estimator's kernel draws visit them constantly:

    1. **Support indicator** (`models.bijections.in_support`).  The integration
       variable is a NOISELESS template moment, so the prior is exactly zero
       past the point-source bounds.  Measured: 99.9% of out-of-support draws
       were already being discarded as poisoned, so this barely moves the
       numbers -- what it buys is that the boundary is now the physical surface
       at an analytically known place, instead of wherever the flow's learned
       extrapolation happened to fall off a cliff.

    2. **Defensive floor**: `(1 - eps) P_flow + eps P_broad`, with `P_broad` a
       wide Gaussian in the chart's standardised coordinates -- "a galaxy could
       be anywhere physically allowed".  This is the part that carries weight.
       For targets far from the boundary, 30.7% of kernel draws were poisoned
       while comfortably INSIDE the support (0.08% outside): the flow answering
       `log p ~ -1e7` in supported regions it simply never saw in training.  The
       floor converts that into a controlled, smooth, tiny number.

       It also fixes the gradient, which is the part that actually reaches the
       shear estimate: where the flow's garbage dominates, `d(mixed)/d(lp)` is
       `(1-eps) e^lp / (...)`, which is ~0.  So a nonsense score contributes
       nothing to Q and R instead of contributing nonsense.

    This is a REGULARISED PRIOR, not an importance-sampling trick -- the
    estimand moves by O(eps), deliberately and computably, in exchange for
    removing values that were wrong by many orders of magnitude.  `eps` is a
    knob to be scanned, not a fitted parameter.

    `P_broad`'s normalisation is not corrected for the support truncation.  It
    does not need to be: an error of a factor `c` there is exactly equivalent to
    using `eps * c`, so it is absorbed into the knob.
    """

    flow: eqx.Module
    eps: float = eqx.field(static=True)
    broad_std: float = eqx.field(static=True)
    support: bool = eqx.field(static=True)

    def __init__(self, flow, eps=1e-3, broad_std=4.0, support=True):
        self.flow = flow
        self.eps = float(eps)
        self.broad_std = float(broad_std)
        self.support = bool(support)

    def __getattr__(self, name):
        # Forward `.bijection`, `.sample`, `.base_dist`, ... to the wrapped
        # flow.  Guarded against the fields themselves so unflattening, which
        # touches attributes before they are all set, cannot recurse.
        if name.startswith("__") or name in ("flow", "eps", "broad_std",
                                             "support"):
            raise AttributeError(name)
        return getattr(self.flow, name)

    def _log_broad(self, x):
        """log of a wide Gaussian on the chart's standardised coordinates,
        pushed back to raw-moment space by the chart's own log-det.

        flowjax's bijections assert their declared `shape`, so the chart has to
        be vmapped over leading axes by hand -- `flow.log_prob` does the same
        internally, which is why it accepts batches and this would not.
        """
        flat = x.reshape(-1, 5)
        z, lad = jax.vmap(chart_of(self.flow).transform_and_log_det)(flat)
        s = self.broad_std
        lb = (-0.5 * jnp.sum((z / s) ** 2, axis=-1)
              - 5.0 * jnp.log(s) - 2.5 * jnp.log(2.0 * jnp.pi) + lad)
        return lb.reshape(x.shape[:-1])

    def log_prob(self, x, condition=None):
        lp = self.flow.log_prob(x, condition=condition)
        # A non-finite flow value is garbage, not information; let the floor
        # carry those rows rather than propagating a NaN through the mixture.
        lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)
        if self.eps <= 0.0:
            mixed = lp
        else:
            mixed = jnp.logaddexp(jnp.log1p(-self.eps) + lp,
                                  jnp.log(self.eps) + self._log_broad(x))
        if not self.support:
            return mixed
        return jnp.where(in_support(x), mixed, -jnp.inf)


def sync_chart_constants(flow, m_train=None):
    """Re-point every layer's FROZEN copy of the chart statistics at the chart
    it is actually sitting behind.  Call this after ANY warm-start graft.

    `build_flow` hands `ShearResponse` and `CentroidMarginalize` a copy of the
    raw sample's mean/std, but `RawMomentStandardize.mean/std` are TRAINABLE and
    `bulk.train` moves them -- on gauss2 the spin-2 std went 0.0402 -> 0.1566
    (3.9x) and log10(Mf)'s 0.296 -> 0.591.  So the moment a `--init` graft drops
    a trained chart in underneath a freshly built layer, the layer's copy
    describes a chart that is no longer there.

    For `ShearResponse` that is not cosmetic.  `_chart_spin0_jac` reconstructs
    logit(u) = chart_scale*z1 + chart_loc to build the 1/(1-u) size Jacobian,
    and `response` reconstructs the physical e = e_scale*(z3 + i z4); with stale
    constants both are wrong, in a way that VARIES ALONG z1, i.e. along Mr/Mf.
    Measured on gauss2 at 20k steps: that mismatch is the entire dm/dg error
    peak at Mr/Mf ~ 3.1 (bin frac_resid 6.36% -> 0.90%, flat everywhere after),
    and the derivative MSE improves 22x (8.9e-3 -> 4.1e-4).  It was NOT the
    NLL competing with the derivative term (identical peak with the NLL weight
    at 0), not the parameterisation, and not Var[Q|m] (measured floor 0.001,
    7x below the peak).

    `centroid.py` already did this for its own layer -- the same two lines, for
    the same reason -- and that is now here instead.

    `m_train` (the same raw moments the flow was/will be trained on) additionally
    resyncs `ShearResponse.coeffs.u_mean/.u_white`, the whitening stats for
    `_Coeffs`'s four net inputs.  Those are computed once by `coeff_stats` from
    the chart's PRE-TRAINING statistics and never touched by the loop below,
    even though `_Coeffs.__call__` whitens live against the chart's current
    (trained) mean/std -- so left unsynced, the whitening silently reverts to
    the ill-conditioned state (correlation 0.997, condition number ~2625) it
    exists to fix, for the entire duration of every `--init` warm start.
    Omit `m_train` to leave them as-is (matches the old behaviour).

    `CentroidMarginalize` has no coefficients of its own to graft or resync
    (see its module docstring) -- only its frozen chart `mean`/`std` need
    fixing here, same as any other layer downstream of the chart.
    """
    bij = flow.bijection.bijection.bijections
    chart = bij[0]
    e_scale = jnp.sqrt(0.5 * (chart.std[3] ** 2 + chart.std[4] ** 2))
    u_stats = None
    if m_train is not None:
        t = to_coords(np.asarray(m_train, dtype=np.float64))
        u_stats = coeff_stats(t, np.asarray(chart.mean), np.asarray(chart.std))
    for i, b in enumerate(bij):
        if isinstance(b, ShearResponse):
            getters = [lambda f, i=i: f.bijection.bijection.bijections[i].e_scale,
                       lambda f, i=i: f.bijection.bijection.bijections[i].chart_loc,
                       lambda f, i=i: f.bijection.bijection.bijections[i].chart_scale]
            values = [non_trainable(e_scale), non_trainable(chart.mean[:3]),
                      non_trainable(chart.std[:3])]
            if u_stats is not None:
                getters += [lambda f, i=i: f.bijection.bijection.bijections[i].coeffs.u_mean,
                            lambda f, i=i: f.bijection.bijection.bijections[i].coeffs.u_white]
                values += [non_trainable(jnp.asarray(u_stats[0], jnp.float32)),
                           non_trainable(jnp.asarray(u_stats[1], jnp.float32))]
            flow = eqx.tree_at(lambda f: [g(f) for g in getters], flow, values)
    # SigmaXBlockLayer (build_flow's current centroid layer) has no frozen
    # chart mean/std of its own -- it operates on already-standardised z, so
    # there is nothing to resync for it and this loop is a no-op on a flow
    # built after this port.  It still matters for a warm-start graft of an
    # OLDER flow that still carries a CentroidMarginalize layer.
    for i, b in enumerate(flow.bijection.bijection.bijections):
        if isinstance(b, CentroidMarginalize):
            flow = eqx.tree_at(
                lambda f, i=i: [f.bijection.bijection.bijections[i].mean,
                                f.bijection.bijection.bijections[i].std],
                flow, [non_trainable(chart.mean), non_trainable(chart.std)])
    return flow


def train(flow, m_train, key, steps=4000, batch=1024, lr=1e-3):
    # The power-law flux gives log10(Mf) a long tail, so an outlier batch can blow
    # the NLL up mid-training and never recover -- clip, then decay the step size.
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(optax.cosine_decay_schedule(lr, steps)))
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    state = opt.init(params)
    data = jnp.asarray(m_train)

    @eqx.filter_jit
    def step(params, state, idx):
        def nll(p):
            return -jnp.mean(eqx.combine(p, static).log_prob(data[idx]))
        loss, grads = jax.value_and_grad(nll)(params)
        updates, state = opt.update(grads, state, params)
        return eqx.apply_updates(params, updates), state, loss

    for i in range(steps):
        key, sk = jr.split(key)
        idx = jr.randint(sk, (batch,), 0, data.shape[0])
        params, state, loss = step(params, state, idx)
        if i % 500 == 0 or i == steps - 1:
            print(f"step {i:5d}  nll {loss:.4f}")
    return eqx.combine(params, static)


def _split(m, frac=0.9):
    n = int(frac * len(m))
    return m[:n], m[n:]


def _flux_sas(s):
    """Parse "mu,sig,a,b" for `--flux-sas`; see `models.bijections.sas`."""
    v = tuple(float(x) for x in s.split(","))
    if len(v) != 4:
        raise argparse.ArgumentTypeError("--flux-sas needs mu,sig,a,b")
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "corner"])
    p.add_argument("--data", default="../bfd_cnf_imsims/data/moments.fits")
    p.add_argument("--flow", default="flows/bulk.eqx")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flux-sas", type=_flux_sas, default=None,
                   help="fitted sinh-arcsinh warp of the flux axis as \"mu,sig,a,b\"; omit for the plain log10 this chart has always used. Gaussianises log10 Mf (skew 1.78 -> 0 on bulgedisc_v2). MUST match across bulk/shear/centroid/bias or the charts disagree.")
    p.add_argument("--out", default="plots/bulk_corner.png")
    a = p.parse_args()

    m = load_moments(a.data)
    m_train, m_val = _split(m)
    key = jr.key(a.seed)
    k_build, k_train, k_sample = jr.split(key, 3)
    flow = build_flow(k_build, m_train, flux_sas=a.flux_sas)

    if a.mode == "train":
        flow = train(flow, m_train, k_train, steps=a.steps)
        print(f"val nll {-jnp.mean(flow.log_prob(jnp.asarray(m_val))):.4f}")
        eqx.tree_serialise_leaves(a.flow, flow)
        print(f"wrote {a.flow}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import corner
    import matplotlib.pyplot as plt

    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    d = to_coords(m)
    s = to_coords(np.asarray(flow.sample(k_sample, (len(m),))))

    # Percentile ranges from the DATA so both sets share axes even if the flow
    # puts mass somewhere the data has none -- that mismatch is the thing to see.
    plot_range = [np.percentile(d[:, i], [0.05, 99.95]) for i in range(5)]
    # Zoom out on the three spin-0 axes: all are bounded by the population's own
    # cuts, so padding puts the point-source line and the empty margin beyond each
    # edge in frame -- that is where a flow leaking mass off the support shows up.
    for i in (0, 1, 2):
        plot_range[i] += 0.25 * np.ptp(plot_range[i]) * np.array([-1.0, 1.0])
    plot_range = [tuple(r) for r in plot_range]
    style = dict(labels=COORD_LABELS, bins=500, range=plot_range, smooth=5.0,
                 plot_density=False, plot_contours=True, fill_contours=False,
                 label_kwargs={"fontsize": LABEL_FONTSIZE})

    fig = plt.figure(figsize=(16, 16))
    corner.corner(d, fig=fig, color="tab:blue", plot_datapoints=True, smooth1d=1.0,
                  hist_kwargs={"label": "imsims moments"}, **style)
    corner.corner(s, fig=fig, color="tab:orange", plot_datapoints=False,
                  hist_kwargs={"label": "Prior flow"}, **style)

    # corner lays the panels out row-major in an N x N grid, so panel (i, j)
    # is axes[i*N + j] and only j <= i exists.  Derive the reference-line
    # positions from that rather than hard-coding indices for one N.
    n_c = len(COORD_LABELS)
    axs = fig.axes

    def mark(coord, value, **kw):
        """Draw `value` on every panel where `coord` is an axis."""
        for row in range(n_c):
            for col in range(row + 1):
                ax = axs[row * n_c + col]
                if col == coord:
                    ax.axvline(value, lw=1, color="tab:red", zorder=500, **kw)
                    kw = {}          # label once
                elif row == coord and row != col:
                    ax.axhline(value, lw=1, color="tab:red", zorder=500)

    mark(1, POINT_SOURCE, label="Point source")   # Mr/Mf unresolved ceiling
    mark(3, 0.0)                                  # M1/Mr
    mark(4, 0.0)                                  # M2/Mr
    axs[n_c + 1].legend(bbox_to_anchor=(0.0, 1.0), loc="lower left", fontsize=16)
    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)

    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
