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

import bulk

# XLA picks matmul precision PER PROCESS, and on this flow that is worth ~1
# nat in the log-densities ([[tf32-breaks-paired-runs]]; `bias.py` has forced
# this since f1f3c92).  Training left unpinned is worse than imprecise, it is
# not comparable: two processes at the SAME seed landed at even log-dets of
# 0.019 and 1.430 -- different basins, not scatter -- which silently voided a
# paired experiment on 2026-08-20.  Costs ~7%.
jax.config.update("jax_default_matmul_precision", "highest")

from models.shear import G_MAX, ShearResponse, dm_dg

# Second-order in g is the model (paper sec. 5.5), so training over a range wider
# than any real shear costs nothing and pins the quadratic term down properly.
# Re-exported from models.shear, which is where the layers can see it.

# Training steps fused into one `lax.scan` per dispatch; also the print interval.
REPORT = 500


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


def _antithetic(idx, gkey, g_max):
    """`(g, take)` for one antithetic batch over the templates `idx`.

    Template `take[k]` appears twice: at `+g[k]` and, `len(idx)` rows later, at
    `-g[k]`.  That pairing is the whole point -- the population scatter is what
    swamps the shear signal, it is common to the two rows of a pair, and it
    cancels in the gradient, leaving only the g dependence.

    Was inlined in `train`'s `step` and WRONG from at least 2026-08 until
    2026-08-20: it drew `idx.shape[0] // 2` shears, which made `g` half the
    length of the duplicated template list, so the trailing `[: g.shape[0]]`
    sliced the second copy off again.  The duplication was a no-op, `+g` and
    `-g` landed on DIFFERENT templates, no scatter cancelled, and the effective
    batch was half what `--batch` asked for.  `tests/test_shear.py::
    test_antithetic_pairing` is what stops it coming back.
    """
    g_half = sample_g(gkey, idx.shape[0], g_max)
    return jnp.concatenate([g_half, -g_half]), jnp.concatenate([idx, idx])


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


def partial_norms(m, q_true, r_true):
    """Per-partial RMS of the truth: (2, 5) for Q, (3, 5) for R.

    dM1/dg1 has RMS ~0.37 against dM1/dg2's ~0.01 -- a factor ~27, 750x squared
    -- so an unnormalised mean over the (2, 5)/(3, 5) blocks supervises the ten
    Q partials and fifteen (three independent, g1g1/g1g2/g2g2) R partials
    wildly unevenly.  Dividing each by its own RMS over the batch it is
    measured on makes `train`'s derivative term a mean of equally-weighted
    residual/RMS pieces, which is exactly what `shear.check` and
    `dev/check_dmdg_components.py` report.
    """
    s = _scale(m)[:, None, :]
    return (jnp.sqrt(jnp.mean((q_true / s) ** 2, axis=0)),
            jnp.sqrt(jnp.mean((r_true / s) ** 2, axis=0)))


def train(flow, data, key, steps=6000, batch=1024, lr=3e-3, bulk_frozen=True,
          g_max=G_MAX, deriv_weight=0.0):
    """Likelihood only by default: the g dependence is learned from P(m|g) alone.

    The cost is convergence.  The g dependence is worth ~0.3 nats/galaxy against
    a ~35 nat total, so on the NLL alone the signal is a small part of a noisy
    objective.  For the spin-2 (coherent, O(g)) part of the response that is
    merely slow: NLL alone recovers it (`dev/spin0_selfconsistent_levers.py`,
    `logs/spin0_levers.log`: M1/M2 alpha 0.96-0.97).  For the spin-0
    (flux/size/concentration) part it is not a rate, it is a floor: that
    response is ODD in the shape e, so it cancels in the population density at
    O(g) and only appears at O(g^2), ~400x weaker than the spin-2 signal at
    g_max=0.02 (measured Fisher information ~0.011/galaxy).  The same
    self-consistency control -- target exactly representable, bulk exactly
    correct, Var[Q|m] = 0 by construction, so every other explanation is
    excluded -- shows NLL alone recovering Mc at alpha ~ -0.03 (i.e. not at
    all), and shows every convergence knob failing to fix it: g_max 0.02 -> 0.10
    (25x more spin-0 signal, by the O(g^2) scaling) makes EVERY coefficient
    worse, including the previously-fine spin-2 ones; batch 1024 -> 65536 at
    fixed steps monotonically worsens the dm/dg residual against bfd truth
    (`logs/batch_probe_fixed/`) while held-out NLL does not move.  So
    `deriv_weight` is not a fine-tuning knob in the sense the rest of this
    project forbids one -- there is no value to search for that trades one
    metric against another.  It regresses the layer's own (dm/dg, d2m/dg2)
    (`models.shear.dm_dg`) onto bfd's exact per-template derivative, which is
    never itself fit to anything (an analytic derivative of a known moment
    integral, `bfd.MomentCalculator.getTemplate`, appendix C) -- the same
    target NLL alone cannot see because the population density it maximises
    is, to the precision that matters here, insensitive to it.  `deriv_weight
    = 1e4` is what makes the term (order 1e-2 to 1) comparable to the ~35 nat
    NLL; the self-consistency control at that weight recovers every
    coefficient to alpha = 1.00.
    """
    m, q, r = (jnp.asarray(a) for a in data)
    nq, nr = partial_norms(m, q, r) if deriv_weight else (None, None)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(optax.cosine_decay_schedule(lr, steps)))
    params, static = eqx.partition(flow, _trainable(flow, bulk_frozen))
    state = opt.init(params)

    def one(carry, _):
        params, state, key = carry
        key, sk, gk = jr.split(key, 3)
        idx = jr.randint(sk, (batch // 2,), 0, m.shape[0])
        g, take = _antithetic(idx, gk, g_max)
        x = lens(m[take], q[take], r[take], g)

        def loss_fn(p):
            model = eqx.combine(p, static)
            nll = -jnp.mean(model.log_prob(x, condition=g))
            if not deriv_weight:
                return nll, nll
            layer, chart = _shear_layer(model), _chart(model)
            qq, rr = jax.vmap(dm_dg, in_axes=(None, 0, None))(layer, m[idx], chart)
            s = _scale(m[idx])[:, None, :]
            mse = (jnp.mean(((qq - q[idx]) / s / nq) ** 2)
                   + jnp.mean(((rr - r[idx]) / s / nr) ** 2))
            return nll + deriv_weight * mse, nll

        (loss, nll), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = opt.update(grads, state, params)
        return (eqx.apply_updates(params, updates), state, key), (loss, nll)

    # One `lax.scan` per REPORT steps rather than one dispatch per step.  A
    # batch-1024 pass through this flow is ~1e8 FLOPs -- microseconds of real
    # compute -- against ~10 ms of Python dispatch and kernel-launch latency per
    # step, which held the GPU at 33-43% SM.  Scanning moves the whole inner
    # loop inside one jit, so the launches happen once per chunk instead of once
    # per step.  The RNG stream is unchanged: same `jr.split(key, 3)` in the same
    # order, so this is a speed change and not a numerical one.
    @eqx.filter_jit
    def run(params, state, key, n):
        return jax.lax.scan(one, (params, state, key), None, length=n)

    done = 0
    while done < steps:
        n = min(REPORT, steps - done)
        (params, state, key), (losses, nlls) = run(params, state, key, n)
        done += n
        if deriv_weight:
            print(f"step {done - 1:6d}  loss {float(losses[-1]):.4f}  "
                  f"nll {float(nlls[-1]):.4f}", flush=True)
        else:
            print(f"step {done - 1:6d}  nll {float(losses[-1]):.4f}", flush=True)
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
    p.add_argument("--g-max", type=float, default=G_MAX,
                   help="training shear disc radius. The spin-0 response is "
                        "odd in e, so it averages out of the population density "
                        "at O(g) and only shows up at O(g^2); the spin-2 "
                        "response is coherent and shows at O(g). Raising this "
                        "grows the spin-0 signal as g^2 against the spin-2 g, "
                        "which is the test of whether that is why NLL-only "
                        "training recovers spin-2 and not spin-0.")
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
                   help="weight on regressing the layer's own dm/dg, d2m/dg2 "
                        "onto bfd's exact per-template derivative (never "
                        "itself fit to anything); 0 = off, pure NLL. Not a "
                        "value to search: NLL alone provably cannot identify "
                        "the spin-0 (flux/size/concentration) response at any "
                        "batch/steps/g_max (dev/spin0_selfconsistent_levers.py, "
                        "logs/spin0_levers.log, logs/batch_probe_fixed/) "
                        "because it is O(g^2) against spin-2's O(g) on an "
                        "isotropic population; 1e4 is what makes the term "
                        "comparable to the ~35 nat NLL and recovers alpha = "
                        "1.00 in that self-consistency control.")
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
                     batch=a.batch, lr=a.lr, g_max=a.g_max,
                     deriv_weight=a.deriv_weight)
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
