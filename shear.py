"""
shear.py
========
Phase 2: learn the shear conditioning of P(m | g, Sigma_X) by likelihood.

The flow LEARNS the g dependence rather than being handed a response map, because
for a real population the shear response is not a function of m at all: two
galaxies with identical moments have different higher-order structure and
respond differently.  What BFD needs is the density P(m|g) (paper eq. 10-13),
so the density is what the loss targets.

Training data is the sheared template population.  bfd computes each template's
shear derivatives analytically (paper appendix C) at high S/N, and imsims stores
them, so a template's lensed moments are exact:

    m_i(g) = m_i + Q_i g + 1/2 g . R_i . g

Draw a g per template, push it through, and maximise log P(m_i(g) | g).  When
templates that share m carry different (Q, R), those samples land in different
places and the flow learns the resulting spread.  Low-S/N target validation
comes later.

Why the analytic derivatives also enter the loss
------------------------------------------------
Pure likelihood is correct but statistically starved: the whole g dependence is
worth ~0.3 nats/galaxy against a ~35 nat total, and it is carried almost
entirely by the spin-2 moments, so the flux/size response and every second
derivative are barely identified (measured: 40-70% error, versus 0.1-1% when
fitted to the derivatives directly).

The resolution is that the two signals estimate the SAME object.  The density
P(m|g) evolves by the continuity equation, so the transport velocity a flow
layer needs is exactly

    v(m) = E[ dm/dg | m ],

the response averaged over whatever hidden structure galaxies sharing m have.
An L2 fit of the layer's velocity to the per-galaxy dm/dg converges to that
conditional mean -- it does NOT assume the response is a function of m.  So the
derivative term is a low-variance estimator of the very transport the likelihood
is trying to find, and adding it costs no generality at first order.

At second order they do differ: the g^2 term of the density feels both E[R|m]
and the *spread* Var[Q|m], and a deterministic transport can only supply the
first.  The resulting multiplicative bias is ~Var[Q|m]/E[Q|m]^2, and note that
it does NOT shrink with the shear -- the missing term and the term it competes
with in R are both O(g^2), so the ratio is the same at g = 0.02 as at 0.1.
Measured on a bulge+disc population, the spin-2 response scatter at fixed
[Mr/Mf, |e|^2] is 8.1% (m ~ 7e-3, well over the 1e-3 target); adding Mc to the
moment vector drops it to 2.0% (m ~ 4e-4).  That is why Mc is modelled.  What
is left would need a stochastic layer; keeping the likelihood term in the loss
is what will expose it when the sims get richer.

Usage:
    python shear.py train  --data ../bfd_cnf_imsims/data/moments.fits
    python shear.py derivs --data ../bfd_cnf_imsims/data/moments.fits
    python shear.py check  --data ../bfd_cnf_imsims/data/moments.fits
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
from flowjax.bijections import Chain, Invert
from flowjax.distributions import Transformed

import bulk
from models.shear import ShearResponse, dm_dg

# Second-order in g is the model (paper sec. 5.5), so training over a range wider
# than any real shear costs nothing and pins the quadratic term down properly.
G_MAX = 0.15


def load(path):
    """Moments and their exact shear derivatives, as float64 arrays."""
    t = fitsio.read(path)
    f64 = lambda a: np.asarray(a, dtype=np.float64)
    return f64(t["moments"]), f64(t["dm_dg"]), f64(t["d2m_dg2"])


def load_labels(path):
    """The population's `red` flag if the catalog has one, else None.

    Only the two-type Sersic catalogs carry it; it is a diagnostic label, never
    an input to the flow -- nothing observable tells a target which type it is.
    """
    try:
        pop = fitsio.read(path, ext="POPULATION")
    except OSError:
        return None
    return np.asarray(pop["red"], dtype=bool) if "red" in pop.dtype.names else None


def lens(m, q, r, g):
    """Exact lensed moments of a template: m + Q.g + 1/2 g.R.g (batched)."""
    g1, g2 = g[:, 0], g[:, 1]
    quad = jnp.stack([0.5 * g1 * g1, g1 * g2, 0.5 * g2 * g2], axis=-1)
    return m + jnp.einsum("bi,bim->bm", g, q) + jnp.einsum("bi,bim->bm", quad, r)


def sample_g(key, n, g_max=G_MAX):
    """Uniform over the disc |g| <= g_max, so no direction is preferred."""
    k1, k2 = jr.split(key)
    ang = jr.uniform(k1, (n,), maxval=2 * jnp.pi)
    rad = g_max * jnp.sqrt(jr.uniform(k2, (n,)))
    return jnp.stack([rad * jnp.cos(ang), rad * jnp.sin(ang)], axis=-1)


def _trainable(flow, bulk_frozen):
    """Filter spec selecting what the optimiser may move.

    Freezing the bulk matters more than it looks.  The g dependence is worth
    ~0.3 nats/galaxy against a ~35 nat total, so if the bulk is free it will
    happily absorb batch noise and drown the signal the shear layer needs.
    """
    if not bulk_frozen:
        return eqx.is_inexact_array
    spec = jax.tree.map(lambda _: False, flow)
    # Located BY TYPE, not by index: the chart is bijections[0] now and the
    # conditional layers follow it, so an index here silently froze the shear
    # layer and trained the chart instead.
    return eqx.tree_at(
        lambda f: _shear_layer(f), spec,
        replace=jax.tree.map(eqx.is_inexact_array, _shear_layer(flow)))


def _scale(m):
    """Per-moment normalisation [Mf, Mr, Mr, Mr, Mc].  Each moment is divided by
    its own magnitude, so the residuals are fractional and flux-blind -- except
    the spin-2 pair, which is divided by Mr because M1 and M2 pass through zero."""
    return jnp.stack([m[:, 0], m[:, 1], m[:, 1], m[:, 1], m[:, 4]], axis=-1)


BAND = 3.4          # where the response fit starts failing


def band_weight(m, k, edge=BAND, width=0.08):
    """Per-galaxy weights that upweight the poorly-fitted resolution band by `k`.

    The response supervision is a REGRESSION against bfd's exact per-galaxy
    `dm_dg`, so reweighting it is free of the objection that would sink a
    reweighted NLL: it moves where the fit is accurate, not what it converges
    to.  The density term is deliberately left unweighted.

    NOT an inverse-density weight.  The band is ~45% of the population, not a
    sparse tail, so equalising the Mr/Mf histogram would DE-weight it.  The
    reason to upweight it is that the REACHABLE (above the Var[Q|m] floor) part
    of the response error is 0.14-0.38 rms through the middle octiles and
    1.16/1.78 in the top two -- the fit is worst exactly where every noisy
    target's convolution integral has to be evaluated.

    Smooth in `Mr/Mf` rather than a step, so the layer is not asked to learn a
    discontinuity in its own loss; renormalised to mean 1 so `deriv_weight` and
    `score_weight` keep their meaning.
    """
    r = np.asarray(m[:, 1] / m[:, 0], dtype=np.float64)
    w = 1.0 + (k - 1.0) / (1.0 + np.exp(-(r - edge) / width))
    return jnp.asarray(w / w.mean())


def _velocity_mse(layer, chart, m, q_true, r_true, score=None, wt=None):
    """L2 distance between the layer's transport velocity and the templates'
    own shear derivatives, normalised by `_scale` so it is
    dimensionless, flux-blind, and weighs every galaxy equally.  Its minimiser
    is E[dm/dg | m] -- see the module docstring.

    Returns `(mse, first)`.  `first` is the same first-order residual contracted
    with the FROZEN bulk's score, which is the combination that actually reaches
    the bias: Q = score . u + div u, so a component with a small response but a
    large score matters to Q while `_scale` -- which divides by the response --
    barely weighs it.  Measured, the Mr and Mc columns each contribute ~15 to
    Q1's error and cancel to 1.8, a cancellation nothing in the loss enforces.
    Zero unless a score is supplied.
    """
    q, r = jax.vmap(dm_dg, in_axes=(None, 0, None))(layer, m, chart)
    s = _scale(m)[:, None, :]
    w = 1.0 if wt is None else wt[:, None, None]
    mse = (jnp.mean(w * ((q - q_true) / s) ** 2)
           + jnp.mean(w * ((r - r_true) / s) ** 2))
    if score is None:
        return mse, jnp.zeros(())
    # q is (batch, shear, moment); the score contracts the moment index.
    w2 = 1.0 if wt is None else wt[:, None]
    return mse, jnp.mean(w2 * jnp.einsum("bm,bam->ba", score, q - q_true) ** 2)


def bulk_of(flow):
    """The g-independent density behind the shear layer, whose score drives Q.

    Data -> base order is [raw2standard, shear, *bulk], so what has to come out
    is the SHEAR layer -- and the chart, which is now in front of it, has to
    stay.  Dropping by type rather than by position, since the position moved.
    """
    keep = [b for b in flow.bijection.bijection.bijections
            if not isinstance(b, ShearResponse)]
    return Transformed(flow.base_dist,
                       Invert(Chain(keep).merge_chains()))


def bulk_score(flow, m, chunk=10000):
    """grad log p_bulk at each m, from the frozen bulk (constant during training)."""
    g = eqx.filter_jit(jax.vmap(jax.grad(bulk_of(flow).log_prob)))
    return jnp.concatenate([g(m[i:i + chunk]) for i in range(0, len(m), chunk)])


def train(flow, data, key, steps=6000, batch=1024, lr=3e-3, bulk_frozen=True,
          deriv_weight=0.0, varq=0.0, score_weight=0.0, band=1.0):
    """Likelihood only by default.

    `deriv_weight` used to be 1e4, regressing the layer's response against bfd's
    exact per-template dm/dg.  Turned OFF on 2026-08-17 for two reasons.

    First, it kept being wrong in ways nothing caught.  The chain reorder broke
    it silently -- the layer responds in standardised coordinates and the target
    is in raw moments, so for one commit the loss compared incommensurate
    quantities at weight 1e4 and no test failed.  `dm_dsigma` had the identical
    bug.  A term that dominates the loss and cannot be checked by the suite is a
    liability whatever it buys.

    Second, and more useful: while the layer is FITTED to bfd's dm/dg, the
    `check` comparison against bfd's dm/dg is not a validation of anything --
    it is a training diagnostic.  With this off it becomes an independent
    held-out test of whether the layer learned the right response from the
    density alone.

    The cost is convergence.  The g dependence is worth ~0.3 nats/galaxy against
    a ~35 nat total, so on the NLL alone the signal is a small part of a noisy
    objective.  Note the failure mode if it does not converge: that is a
    VARIANCE floor, not a rate, so the knob is `batch`, not `steps`.
    """
    m, q, r = (jnp.asarray(a) for a in data)
    wt = band_weight(data[0], band) if band != 1.0 else None
    score = None
    if score_weight:
        if not bulk_frozen:
            raise ValueError("the score term uses the frozen bulk's score, "
                             "which would otherwise move under the optimiser")
        score = bulk_score(flow, m)
    # The velocity term's target and the LENSED MOMENTS are different objects.
    # `lens` must keep the exact physical d2m_dg2 -- those moments are what the
    # templates really have -- while the velocity term is fitted to E[R_t|m]
    # plus the Var[Q|m] offset (varq.py).  Only the latter moves.
    r_tgt = r
    if varq:
        if not bulk_frozen:
            raise ValueError("the Var[Q|m] offset is built from the frozen "
                             "bulk's score, so the bulk must be frozen")
        import varq as varq_mod
        r_tgt = r + varq * varq_mod.target_offset(flow, m, q)
        print(f"added {varq:g} x the Var[Q|m] offset to the second-order target")
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(optax.cosine_decay_schedule(lr, steps)))
    params, static = eqx.partition(flow, _trainable(flow, bulk_frozen))
    state = opt.init(params)

    @eqx.filter_jit
    def step(params, state, idx, gkey):
        # Antithetic pairing: the SAME template at +g and -g.  The population
        # scatter, which is what swamps the shear signal, is common to the pair
        # and cancels in the gradient; only the g dependence survives.
        half = sample_g(gkey, idx.shape[0] // 2)
        g = jnp.concatenate([half, -half])
        two = lambda a: jnp.concatenate([a[idx], a[idx]])[: g.shape[0]]
        x = lens(two(m), two(q), two(r), g)

        def loss_fn(p):
            model = eqx.combine(p, static)
            nll = -jnp.mean(model.log_prob(x, condition=g))
            if not (deriv_weight or score_weight):
                # Guarded outside the traced branch, as bulk.train does with its
                # score term: with the supervision off, `dm_dg` -- and with it
                # the chart composition -- is not part of training at all,
                # rather than being computed and multiplied by zero.
                return nll, (nll, jnp.zeros(()), jnp.zeros(()))
            mse, first = _velocity_mse(_shear_layer(model), _chart(model),
                                       m[idx], q[idx],
                                       r_tgt[idx],
                                       None if score is None else score[idx],
                                       None if wt is None else wt[idx])
            return (nll + deriv_weight * mse + score_weight * first,
                    (nll, mse, first))

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = opt.update(grads, state, params)
        return eqx.apply_updates(params, updates), state, aux

    for i in range(steps):
        key, sk, gk = jr.split(key, 3)
        params, state, (nll, mse, first) = step(
            params, state, jr.randint(sk, (batch // 2,), 0, m.shape[0]), gk)
        if i % 500 == 0 or i == steps - 1:
            print(f"step {i:5d}  nll {nll:.4f}  velocity mse {mse:.3e}"
                  f"  score term {first:.3e}")
    return eqx.combine(params, static)


def val_nll(flow, data, key, n=20000):
    m, q, r = (jnp.asarray(a[:n]) for a in data)
    g = sample_g(key, m.shape[0])
    return float(-jnp.mean(flow.log_prob(lens(m, q, r, g), condition=g)))


def _shear_layer(flow):
    """The ShearResponse inside the built flow.

    The chain is [raw2standard, shear, *bulk] data -> base (or
    [raw2standard, centroid, shear, *bulk] with the centroid layer), so the
    chart is index 0 and the shear layer follows it.
    """
    for b in flow.bijection.bijection.bijections:
        if isinstance(b, ShearResponse):
            return b
    raise ValueError("no ShearResponse in this flow")


def _chart(flow):
    """The RawMomentStandardize, which `dm_dg` needs to convert dz/dg to dm/dg."""
    return flow.bijection.bijection.bijections[0]


def check(flow, data, n=4000, labels=None):
    """Compare the layer's own dm/dg at g=0 with bfd's exact values.

    Diagnostic only.  The layer is a transport that matches DENSITIES, so it is
    under no obligation to reproduce any individual template's response; it is
    meaningful only to the extent that the response really is a function of m,
    which `response_scatter` measures independently.
    """
    m, q_true, r_true = (jnp.asarray(a[:n]) for a in data)
    layer, chart = _shear_layer(flow), _chart(flow)
    q, r = jax.vmap(dm_dg, in_axes=(None, 0, None))(layer, m, chart)
    s = _scale(m)[:, None, :]
    groups = [("", slice(None))]
    if labels is not None:
        groups += [(" red", labels[:n]), (" blue", ~labels[:n])]
    print("\nlayer dm/dg vs bfd truth: RMS residual / RMS truth")
    print(f"{'':15s}" + "".join(f"{n_:>10s}"
                                for n_ in ["Mf", "Mr", "M1", "M2", "Mc"]))
    for name, pred, truth in (("dm/dg", q, q_true), ("d2m/dg2", r, r_true)):
        d = np.asarray((pred - truth) / s)
        t = np.asarray(truth / s)
        for tag, sel in groups:
            frac = (np.sqrt(np.mean(d[sel] ** 2, axis=(0, 1)))
                    / np.sqrt(np.mean(t[sel] ** 2, axis=(0, 1))))
            print(f"{name + tag:15s}" + "".join(f"{v:10.2%}" for v in frac))


def _spin2_AB(m, q_true):
    """The exact spin-2 response coefficients A, B of each template, from bfd.

    At g = 0 the equivariant response of `models.shear` is

        d(M1 + i M2)/dg1 = Mr (A + B e^2),   d(M1 + i M2)/dg2 = i Mr (A - B e^2)

    with A and B real, so both follow from `dm_dg` by inversion -- no fitting.
    Together they carry the entire first-order spin-2 response.
    """
    Mr = m[:, 1]
    e = (m[:, 2] + 1j * m[:, 3]) / Mr
    d1 = (q_true[:, 0, 2] + 1j * q_true[:, 0, 3]) / Mr
    d2 = (q_true[:, 1, 2] + 1j * q_true[:, 1, 3]) / Mr
    A, Be2 = 0.5 * (d1 - 1j * d2), 0.5 * (d1 + 1j * d2)
    # Both are real by parity; if they are not, the spin decomposition is wrong.
    assert np.abs(A.imag).max() < 1e-6 * np.abs(A.real).max()
    e_sq = np.abs(e) ** 2
    # B is Be2/e^2, which blows up as e -> 0.  Carry it as the pair (B, |e|^4)
    # instead: |e|^4 is exactly the weight B enters every later average with.
    with np.errstate(invalid="ignore", divide="ignore"):
        B = np.where(e_sq > 0, (Be2 / np.where(e_sq > 0, e * e, 1.0)).real, 0.0)
    return A.real, B, e_sq**2


def _spin0_a(m, q_true):
    """The exact first-order spin-0 response coefficient of Mf, Mr and Mc.

    The spin-0 structure at first order is dX/dg = X a_X Re(e* g), i.e.
    (dX/dg1, dX/dg2) = X a_X (e1, e2), so a_X inverts out of `dm_dg` with |e|^2
    as its natural weight -- a round galaxy has no first-order spin-0 response
    at all, and nothing to say about a_X.
    """
    Mr = m[:, 1]
    e = np.stack([m[:, 2] / Mr, m[:, 3] / Mr], axis=-1)
    e_sq = (e**2).sum(-1)
    X = m[:, [0, 1, 4]]
    num = np.einsum("bi,bij->bj", e, q_true[:, :, [0, 1, 4]])
    return num / (X * e_sq[:, None]), e_sq


def _poly(x, deg):
    """All monomials in the columns of `x` up to total degree `deg`."""
    import itertools

    x = (x - x.mean(0)) / x.std(0)
    cols = [np.ones(len(x))]
    for d in range(1, deg + 1):
        for p in itertools.combinations_with_replacement(range(x.shape[1]), d):
            cols.append(np.prod(x[:, p], axis=1))
    return np.stack(cols, axis=-1)


def _wls_resid(basis, y, w):
    """Weighted least-squares residual of y on basis."""
    s = np.sqrt(w)
    beta = np.linalg.lstsq(basis * s[:, None], y * s, rcond=None)[0]
    return y - basis @ beta


def response_scatter(m, q_true, labels=None, deg=4):
    """The share of the first-order shear response that the moments do not fix.

    A transport layer makes dm/dg a deterministic function of m, so it can only
    ever carry E[dm/dg | m].  What a flexible regression of the exact response
    coefficients on the flux-blind invariants CANNOT explain is Var[Q|m] -- the
    piece no deterministic layer can represent, and the floor on the resulting
    multiplicative bias (see the module docstring: the bias is ~Var[Q|m]/E[Q|m]^2
    and does not shrink with g).  Running it with and without the concentration
    k = Mc Mf/Mr^2 is what says whether modelling Mc is enough for a population,
    and it is a property of the population alone -- no trained flow involved.
    """
    A, B, w4 = _spin2_AB(m, q_true)
    a0, e_sq = _spin0_a(m, q_true)
    Mf, Mr, Mc = m[:, 0], m[:, 1], m[:, 4]
    r, k, q = Mr / Mf, Mc * Mf / (Mr * Mr), e_sq
    ones = np.ones(len(m))

    # (label, [(coefficient, its weight in the response), ...])
    parts = [("spin-2 (A, B)", [(A, ones), (B, w4)])]
    parts += [(f"spin-0 {n}", [(a0[:, i], e_sq)])
              for i, n in enumerate(["Mf", "Mr", "Mc"])]

    def frac(resid, terms, sel=slice(None)):
        num = sum(np.mean((d**2 * wt)[sel]) for d, (_, wt) in zip(resid, terms))
        den = sum(np.mean((c**2 * wt)[sel]) for c, wt in terms)
        return np.sqrt(num / den)

    bases = {name: _poly(np.stack(cols, axis=-1), deg)
             for name, cols in (("no Mc", (r, q)), ("with Mc", (r, k, q)))}
    print("\nfirst-order response scatter at fixed moments: "
          f"RMS unexplained / RMS response, {len(m)} templates")
    print("invariants regressed on: Mr/Mf, |e|^2 (no Mc) and + Mc Mf/Mr^2")
    header = f"{'component':18s}{'no Mc':>10s}{'with Mc':>10s}"
    if labels is not None:
        header += f"{'  red':>10s}{'  blue':>10s}   (with Mc)"
    print(header)
    for name, terms in parts:
        resid = {b: [_wls_resid(bases[b], c, wt) for c, wt in terms]
                 for b in bases}
        row = (f"{name:18s}{frac(resid['no Mc'], terms):10.2%}"
               f"{frac(resid['with Mc'], terms):10.2%}")
        if labels is not None:
            row += (f"{frac(resid['with Mc'], terms, labels):10.2%}"
                    f"{frac(resid['with Mc'], terms, ~labels):10.2%}")
        print(row)


def derivs_plot(flow, log10mf, mrmf, mcmr, m_range, n, out):
    """P and its shear derivatives over the (M1/Mr, M2/Mr) plane at fixed Mf, Mr/Mf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Mf = 10.0**log10mf
    Mr = mrmf * Mf
    Mc = mcmr * Mr
    ax1 = np.linspace(-m_range, m_range, n)
    E1, E2 = np.meshgrid(ax1, ax1, indexing="ij")
    x = jnp.asarray(np.stack([np.full(E1.size, Mf), np.full(E1.size, Mr),
                              Mr * E1.ravel(), Mr * E2.ravel(),
                              np.full(E1.size, Mc)], axis=-1))

    def prob(row, g1, g2):
        return jnp.exp(flow.log_prob(row, condition=jnp.array([g1, g2])))

    ax = (0, None, None)
    g0 = (0.0, 0.0)
    P = np.asarray(jax.vmap(prob, ax)(x, *g0))
    d1 = np.asarray(jax.vmap(jax.grad(prob, 1), ax)(x, *g0))
    d2 = np.asarray(jax.vmap(jax.grad(prob, 2), ax)(x, *g0))
    H = jax.vmap(jax.hessian(prob, argnums=(1, 2)), ax)(x, *g0)
    h11, h12, h22 = (np.asarray(H[0][0]), np.asarray(H[0][1]), np.asarray(H[1][1]))

    # rows index M2/Mr (y), columns M1/Mr (x), to match the imshow extent
    plane = lambda a: a.reshape(n, n).T
    dmax = 0.5 * max(np.abs(d1).max(), np.abs(d2).max())
    hmax = 0.5 * max(np.abs(h11).max(), np.abs(h22).max())

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    panels = [
        (axes[0, 0], P, "Blues", None, r"$P$"),
        (axes[0, 1], d1, "RdBu", dmax, r"$\frac{\partial P}{\partial g_1}$"),
        (axes[0, 2], d2, "RdBu", dmax, r"$\frac{\partial P}{\partial g_2}$"),
        (axes[1, 0], h11, "RdBu", hmax, r"$\frac{\partial^2 P}{\partial g_1^2}$"),
        (axes[1, 1], h22, "RdBu", hmax, r"$\frac{\partial^2 P}{\partial g_2^2}$"),
        (axes[1, 2], h12, "RdBu", 0.5 * np.abs(h12).max(),
         r"$\frac{\partial^2 P}{\partial g_1 \partial g_2}$"),
    ]
    kw = dict(aspect="auto", origin="lower", interpolation="none",
              extent=(-m_range, m_range, -m_range, m_range))
    for a, field, cmap, absmax, label in panels:
        k = dict(kw, cmap=cmap)
        if absmax is not None:
            k.update(vmin=-absmax, vmax=absmax)
        im = a.imshow(plane(field), **k)
        a.set_xlabel(r"$M_1/M_r$", fontsize=20)
        a.set_ylabel(r"$M_2/M_r$", fontsize=20)
        a.axvline(0.0, color="grey", lw=1)
        a.axhline(0.0, color="grey", lw=1)
        plt.colorbar(im, ax=a).set_label(label=label, size=20)
    fig.suptitle(rf"$\log_{{10}}M_f = {log10mf:.3g}$ ($M_f = {Mf:.0f}$), "
                 rf"$M_r/M_f = {mrmf:.3g}$, $M_c/M_r = {mcmr:.3g}$, "
                 rf"$g = (0, 0)$", fontsize=22)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "check", "derivs", "scatter"])
    p.add_argument("--data", default="../bfd_cnf_imsims/data/moments.fits")
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--init", default="flows/bulk.eqx",
                   help="bulk checkpoint to warm-start from (train only)")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=1024,
                   help="templates per step (antithetic: batch//2 templates, "
                        "each at +g and -g). With the response supervision off "
                        "the g signal is a small part of a noisy NLL, and that "
                        "is a VARIANCE floor -- 4x the steps moves nothing, so "
                        "this is the knob that matters.")
    p.add_argument("--lr", type=float, default=3e-3,
                   help="shear-stage learning rate. It has never been tuned "
                        "UPWARD, and it moved the old noiseless metric 5x -- "
                        "more than every architectural axis combined.")
    p.add_argument("--deriv-weight", type=float, default=0.0,
                   help="weight on the velocity-matching term relative to the NLL")
    p.add_argument("--varq", type=float, default=0.0,
                   help="scale on the Var[Q|m] offset to the second-order "
                        "target (0 = off, 1 = the derived value)")
    p.add_argument("--band", type=float, default=1.0,
                   help="upweight the derivative supervision above Mr/Mf ~ 3.4 "
                        "by this factor -- the band every noisy target's "
                        "convolution integrates through, and where the "
                        "reachable response error is 4-6x larger (the NLL "
                        "stays unweighted). 1.0 = off")
    p.add_argument("--score-weight", type=float, default=0.0,
                   help="weight on the score-contracted first-order residual "
                        "(the combination that reaches Q); 0 = off")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log10mf", type=float, default=3.6)
    p.add_argument("--mrmf", type=float, default=3.3)
    p.add_argument("--mcmr", type=float, default=None,
                   help="Mc/Mr slice (default: the catalog median at this Mr/Mf). "
                        "Mc is nearly determined by the other moments, so an "
                        "off-locus slice lands where the population has no "
                        "support and P collapses to zero.")
    p.add_argument("--m-range", type=float, default=0.25)
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--out", default="plots/shear_derivs.png")
    a = p.parse_args()

    data = load(a.data)
    if a.mode == "scatter":
        # A property of the population, not of any trained flow -- no checkpoint.
        response_scatter(data[0], data[1], load_labels(a.data))
        return

    n_train = int(0.9 * len(data[0]))
    train_set = tuple(x[:n_train] for x in data)
    val_set = tuple(x[n_train:] for x in data)
    labels = load_labels(a.data)
    val_labels = None if labels is None else labels[n_train:]

    flow = bulk.build_flow(jr.key(a.seed), train_set[0], shear=True)

    if a.mode == "train":
        # Warm start the bulk from phase 1: it already knows p(m) at g=0, so the
        # optimiser only has to find the g dependence.
        if a.init:
            bulk_only = eqx.tree_deserialise_leaves(
                a.init, bulk.build_flow(jr.key(a.seed), train_set[0]))
            # `flow` is [raw2standard, shear, *bulk]; `bulk_only` is
            # [raw2standard, *bulk].  Graft every non-shear slot, in order.
            keep = [i for i, b in enumerate(flow.bijection.bijection.bijections)
                    if not isinstance(b, ShearResponse)]
            flow = eqx.tree_at(
                lambda f: [f.bijection.bijection.bijections[i] for i in keep],
                flow, list(bulk_only.bijection.bijection.bijections))
            print(f"warm started bulk from {a.init}")
        flow = train(flow, train_set, jr.key(a.seed + 1), steps=a.steps,
                     batch=a.batch, lr=a.lr,
                     deriv_weight=a.deriv_weight, varq=a.varq,
                     score_weight=a.score_weight, band=a.band)
        print(f"val nll {val_nll(flow, val_set, jr.key(99)):.4f}")
        eqx.tree_serialise_leaves(a.flow, flow)
        print(f"wrote {a.flow}")
        check(flow, val_set, labels=val_labels)
        return

    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    if a.mode == "check":
        print(f"val nll {val_nll(flow, val_set, jr.key(99)):.4f}")
        check(flow, val_set, labels=val_labels)
    else:
        mcmr = a.mcmr
        if mcmr is None:
            M = data[0]
            near = np.abs(M[:, 1] / M[:, 0] - a.mrmf) < 0.05
            mcmr = float(np.median((M[:, 4] / M[:, 1])[near]))
            print(f"Mc/Mr slice from the catalog at Mr/Mf={a.mrmf}: {mcmr:.4f} "
                  f"({near.sum()} galaxies)")
        derivs_plot(flow, a.log10mf, a.mrmf, mcmr, a.m_range, a.n, a.out)


if __name__ == "__main__":
    main()
