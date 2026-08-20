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
from flowjax.bijections import Chain, Invert, Permute
from flowjax.distributions import MultivariateNormal, Transformed
from paramax import non_trainable

from models.bijections import EquivariantAutoregressiveLayer, RawMomentStandardize
from models.centroid import CentroidMarginalize
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

# The flow works in t = [log10(Mf), Mr/Mf, Mc/Mr, M1/Mr, M2/Mr]
# (RawMomentStandardize), standardised by the training set's own mean/std --
# spin-0 first, spin-2 last.  Label the corner plot in it.
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

# The target selection the real analysis applies, recorded here (not in
# bias.py, which shear.py cannot import without a cycle) so it is not
# re-invented per script.  A size window in Mr/Mf and a flux window in Mf, as
# on sky.  These are NOT yet wired into `bias.main` -- eq. (40)/(46)'s
# selection terms are owed before a cut may be applied to a bias measurement
# -- but they are what `dev/check_dmdg_components.py` restricts its
# comparison to and what `shear.band_weight` upweights, since the response
# outside the window is never used.
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


def to_coords(m):
    """Raw moments -> the flow's transformed coordinates t (for plotting).

    Mirrors `RawMomentStandardize._forward_transform`, including slots 1 and
    2's logits: this is what fixes the standardisation's mean/std, so the two
    must not drift apart.
    """
    u = m[:, 1] / (POINT_SOURCE * m[:, 0])
    v = m[:, 4] / (POINT_SOURCE_MC * m[:, 1])
    return np.stack([np.log10(m[:, 0]), np.log(u) - np.log1p(-u),
                     np.log(v) - np.log1p(-v), m[:, 2] / m[:, 1], m[:, 3] / m[:, 1]],
                    axis=-1)


def build_flow(key, m_train, layers=LAYERS, nn_width=NN_WIDTH, nn_depth=NN_DEPTH,
               shear=False, centroid=False):
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
    bad = int((m_train[:, 1] >= POINT_SOURCE * m_train[:, 0]).sum())
    if bad:
        raise ValueError(
            f"{bad} training moments are at or above the point-source ceiling "
            f"Mr/Mf = {POINT_SOURCE}, where the flow's chart is undefined. "
            "The bound is on the LATENT moment; if these are noisy "
            "measurements they need a deconvolving objective, not this one.")
    t = to_coords(m_train)
    raw2standard = RawMomentStandardize(mean=t.mean(0), std=t.std(0))

    key, k_shear, k_centroid = jr.split(key, 3)
    keys = jr.split(key, layers)
    _incs = _spin0_increments(layers)
    bulk = []
    for i, k in enumerate(keys):
        bulk.append(EquivariantAutoregressiveLayer(k, nn_width, nn_depth, jax.nn.silu))
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
    head = ([CentroidMarginalize(k_centroid, cond_dim=cond or 3,
                                 mean=t.mean(0), std=t.std(0))]
            if centroid else []) + \
           ([ShearResponse(k_shear, cond_dim=cond or 2,
                           e_scale=float(t.std(0)[3]))] if shear else [])
    bijection = Invert(Chain([raw2standard, *head, *bulk]).merge_chains())
    base = non_trainable(MultivariateNormal(jnp.zeros(5), jnp.eye(5)))
    return Transformed(base, bijection)


def z_of(flow):
    """The same density in the flow's standardized coordinates z = T(m).

    Data -> base order is [chart, *bulk] (`build_flow`'s comment on
    `flow.bijection.bijection.bijections`), so peeling off `bijections[0]`
    -- the `RawMomentStandardize` chart -- leaves exactly the bulk chain
    acting on z instead of m.  This is `shear.bulk_of` one layer further in:
    that function peels the shear layer to expose the bulk at g = 0, this one
    peels the chart to expose the bulk in the coordinates it actually models.
    """
    bij = flow.bijection.bijection.bijections
    return Transformed(flow.base_dist,
                       Invert(Chain(list(bij[1:])).merge_chains()))


def _hyvarinen(log_prob, z, coords=(0, 1, 2, 3, 4)):
    """sum_j E[ 0.5 (d_j log p)^2 + d_jj log p ] over `coords`, rows of `z`.

    This is the (implicit) Hyvarinen score-matching objective: its expectation
    over the TRUE data density equals, up to an additive constant the model
    cannot move, 0.5 E[|grad log p_model - grad log p_true|^2] -- so minimising
    it fits the model's SCORE without ever needing the true density's score.
    `log_prob` maps a single (5,) point to a scalar; `z` is (n, 5).

    It is summed over coordinates and the integration by parts that produces it
    is done one coordinate at a time, so a SUBSET of `coords` is still a
    consistent objective -- for those partials of the score, and only those.
    That subsetting is the point rather than an economy: on this population the
    full sum is dominated by `Mc/Mr` (about -8600 of -10200, because Mc is
    nearly determined by the other moments so its conditional is razor thin),
    which would spend the whole term on a direction the shear estimator never
    looks along.  See `SCORE_COORDS`.

    Each coordinate costs one Hessian-vector product (`jvp` of `grad` along a
    basis vector), not a full Hessian: with the default `coords` that is the
    same work, but at one coordinate it is ~5x cheaper, which is the difference
    between a 20 minute training run and an 11 hour one.
    """
    def per_row(z_i):
        def one(j):
            e = jnp.zeros(z.shape[-1]).at[j].set(1.0)
            g, hv = jax.jvp(jax.grad(log_prob), (z_i,), (e,))
            return 0.5 * g[j] ** 2 + hv[j]
        return sum(one(j) for j in coords)

    return jnp.mean(jax.vmap(per_row)(z))


# The resolution direction, logit(Mr/Mf).  The ramp that motivates this whole
# term lives in Mr/Mf, and the convolution that carries it reaches through the
# 3.5-3.976 band of exactly this coordinate; the other four either dominate the
# objective for irrelevant reasons (Mc/Mr, above) or are already small.
SCORE_COORDS = (1,)


def score_match(flow, m, coords=SCORE_COORDS):
    """The bulk's score-matching term at raw moments `m` ((n, 5) array).

    Maximum likelihood constrains p but not grad log p, yet the downstream
    estimator only ever touches the SCORE -- `Q = grad log p . u + div u` --
    so this term is what actually supervises the quantity that reaches the
    bias (mirrors `shear.py`'s `_velocity_mse` first-order score term one
    stage upstream).

    Evaluated in the chart's standardized coordinates z = T(m), not raw
    moment space: raw scores are ~1e-4 and the term would underflow float32.
    `z` is wrapped in `stop_gradient` -- the chart's mean/std are trainable
    (`bulk.train` partitions on `eqx.is_inexact_array`), so without this the
    optimiser could shrink the term by moving the data's coordinates instead
    of by fixing the score, which is not the objective.
    """
    chart = flow.bijection.bijection.bijections[0]
    z, _ = jax.vmap(chart.transform_and_log_det)(m)
    z = jax.lax.stop_gradient(z)
    return _hyvarinen(z_of(flow).log_prob, z, coords)


def train(flow, m_train, key, steps=4000, batch=1024, lr=1e-3, score_weight=0.0,
          score_coords=SCORE_COORDS):
    # The power-law flux gives log10(Mf) a long tail, so an outlier batch can blow
    # the NLL up mid-training and never recover -- clip, then decay the step size.
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(optax.cosine_decay_schedule(lr, steps)))
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    state = opt.init(params)
    data = jnp.asarray(m_train)

    # Guarded outside the jit so the score term is never traced, let alone
    # computed, when it is off -- the default path stays bit-identical to
    # what it was before this option existed.
    if score_weight:
        @eqx.filter_jit
        def step(params, state, idx):
            def loss_fn(p):
                model = eqx.combine(p, static)
                x = data[idx]
                nll = -jnp.mean(model.log_prob(x))
                score = score_match(model, x, score_coords)
                return nll + score_weight * score, (nll, score)
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, state = opt.update(grads, state, params)
            return eqx.apply_updates(params, updates), state, aux
    else:
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
        params, state, aux = step(params, state, idx)
        if i % 500 == 0 or i == steps - 1:
            if score_weight:
                nll, score = aux
                print(f"step {i:5d}  nll {nll:.4f}  score {score:.4f}")
            else:
                print(f"step {i:5d}  nll {aux:.4f}")
    return eqx.combine(params, static)


def _split(m, frac=0.9):
    n = int(frac * len(m))
    return m[:n], m[n:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "corner"])
    p.add_argument("--data", default="../bfd_cnf_imsims/data/moments.fits")
    p.add_argument("--flow", default="flows/bulk.eqx")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--score-weight", type=float, default=0.0,
                   help="weight on the Hyvarinen score-matching term, which "
                        "supervises grad log p directly -- the quantity the "
                        "downstream estimator actually uses; 0 = off. The term "
                        "is ~-1500 against an NLL of ~41, so useful weights "
                        "are 1e-3 and below")
    p.add_argument("--score-coords", default=",".join(map(str, SCORE_COORDS)),
                   help="standardized coordinates the score term supervises, "
                        "comma separated: 0 log10(Mf), 1 logit(Mr/Mf), "
                        "2 Mc/Mr, 3-4 the spin-2 pair")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="plots/bulk_corner.png")
    a = p.parse_args()

    m = load_moments(a.data)
    m_train, m_val = _split(m)
    key = jr.key(a.seed)
    k_build, k_train, k_sample = jr.split(key, 3)
    flow = build_flow(k_build, m_train)

    if a.mode == "train":
        flow = train(flow, m_train, k_train, steps=a.steps,
                     score_weight=a.score_weight,
                     score_coords=tuple(int(c) for c in a.score_coords.split(",")))
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
