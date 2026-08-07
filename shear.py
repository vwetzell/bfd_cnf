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
first.  That gap is real for populations whose response scatters at fixed m
(it vanishes on this Gaussian test bed, where the response is a function of the
invariants to a few percent).  Closing it needs a stochastic layer; keeping the
likelihood term in the loss is what will expose it when the sims get richer.

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
from models.shear import dm_dg

# Second-order in g is the model (paper sec. 5.5), so training over a range wider
# than any real shear costs nothing and pins the quadratic term down properly.
G_MAX = 0.15


def load(path):
    """Moments and their exact shear derivatives, as float64 arrays."""
    t = fitsio.read(path)
    f64 = lambda a: np.asarray(a, dtype=np.float64)
    return f64(t["moments"][:, :4]), f64(t["dm_dg"]), f64(t["d2m_dg2"])


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
    return eqx.tree_at(
        lambda f: f.bijection.bijection.bijections[0], spec,
        replace=jax.tree.map(eqx.is_inexact_array, _shear_layer(flow)))


def _velocity_mse(layer, m, q_true, r_true):
    """L2 distance between the layer's transport velocity and the templates'
    own shear derivatives, normalised by [Mf, Mr, Mr, Mr] so it is
    dimensionless, flux-blind, and weighs every galaxy equally.  Its minimiser
    is E[dm/dg | m] -- see the module docstring."""
    q, r = jax.vmap(dm_dg, in_axes=(None, 0))(layer, m)
    s = jnp.stack([m[:, 0], m[:, 1], m[:, 1], m[:, 1]], axis=-1)[:, None, :]
    return jnp.mean(((q - q_true) / s) ** 2) + jnp.mean(((r - r_true) / s) ** 2)


def train(flow, data, key, steps=6000, batch=1024, lr=3e-3, bulk_frozen=True,
          deriv_weight=1e3):
    m, q, r = (jnp.asarray(a) for a in data)
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
            mse = _velocity_mse(_shear_layer(model), m[idx], q[idx], r[idx])
            return nll + deriv_weight * mse, (nll, mse)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = opt.update(grads, state, params)
        return eqx.apply_updates(params, updates), state, aux

    for i in range(steps):
        key, sk, gk = jr.split(key, 3)
        params, state, (nll, mse) = step(
            params, state, jr.randint(sk, (batch // 2,), 0, m.shape[0]), gk)
        if i % 500 == 0 or i == steps - 1:
            print(f"step {i:5d}  nll {nll:.4f}  velocity mse {mse:.3e}")
    return eqx.combine(params, static)


def val_nll(flow, data, key, n=20000):
    m, q, r = (jnp.asarray(a[:n]) for a in data)
    g = sample_g(key, m.shape[0])
    return float(-jnp.mean(flow.log_prob(lens(m, q, r, g), condition=g)))


def _shear_layer(flow):
    """The ShearResponse sitting data-adjacent inside the built flow."""
    return flow.bijection.bijection.bijections[0]


def check(flow, data, n=4000):
    """Compare the layer's own dm/dg at g=0 with bfd's exact values.

    Diagnostic only.  The layer is a transport that matches DENSITIES, so it is
    under no obligation to reproduce any individual template's response; on this
    Gaussian population the two coincide because the response happens to be a
    function of m, and that is what makes the comparison meaningful here.
    """
    m, q_true, r_true = (jnp.asarray(a[:n]) for a in data)
    layer = _shear_layer(flow)
    q, r = jax.vmap(dm_dg, in_axes=(None, 0))(layer, m)
    s = jnp.stack([m[:, 0], m[:, 1], m[:, 1], m[:, 1]], axis=-1)[:, None, :]
    print("\nlayer dm/dg vs bfd truth: RMS residual / RMS truth")
    print(f"{'':10s}" + "".join(f"{n_:>10s}" for n_ in ["Mf", "Mr", "M1", "M2"]))
    for name, pred, truth in (("dm/dg", q, q_true), ("d2m/dg2", r, r_true)):
        d = np.asarray((pred - truth) / s)
        t = np.asarray(truth / s)
        frac = (np.sqrt(np.mean(d**2, axis=(0, 1)))
                / np.sqrt(np.mean(t**2, axis=(0, 1))))
        print(f"{name:10s}" + "".join(f"{v:10.2%}" for v in frac))


def derivs_plot(flow, log10mf, mrmf, m_range, n, out):
    """P and its shear derivatives over the (M1/Mr, M2/Mr) plane at fixed Mf, Mr/Mf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Mf = 10.0**log10mf
    Mr = mrmf * Mf
    ax1 = np.linspace(-m_range, m_range, n)
    E1, E2 = np.meshgrid(ax1, ax1, indexing="ij")
    x = jnp.asarray(np.stack([np.full(E1.size, Mf), np.full(E1.size, Mr),
                              Mr * E1.ravel(), Mr * E2.ravel()], axis=-1))

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
                 rf"$M_r/M_f = {mrmf:.3g}$, $g = (0, 0)$", fontsize=22)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "check", "derivs"])
    p.add_argument("--data", default="../bfd_cnf_imsims/data/moments.fits")
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--init", default="flows/bulk.eqx",
                   help="bulk checkpoint to warm-start from (train only)")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log10mf", type=float, default=3.6)
    p.add_argument("--mrmf", type=float, default=3.3)
    p.add_argument("--m-range", type=float, default=0.25)
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--out", default="plots/shear_derivs.png")
    a = p.parse_args()

    data = load(a.data)
    n_train = int(0.9 * len(data[0]))
    train_set = tuple(x[:n_train] for x in data)
    val_set = tuple(x[n_train:] for x in data)

    flow = bulk.build_flow(jr.key(a.seed), train_set[0], shear=True)

    if a.mode == "train":
        # Warm start the bulk from phase 1: it already knows p(m) at g=0, so the
        # optimiser only has to find the g dependence.
        if a.init:
            bulk_only = eqx.tree_deserialise_leaves(
                a.init, bulk.build_flow(jr.key(a.seed), train_set[0]))
            flow = eqx.tree_at(
                lambda f: f.bijection.bijection.bijections[1:], flow,
                bulk_only.bijection.bijection.bijections)
            print(f"warm started bulk from {a.init}")
        flow = train(flow, train_set, jr.key(a.seed + 1), steps=a.steps)
        print(f"val nll {val_nll(flow, val_set, jr.key(99)):.4f}")
        eqx.tree_serialise_leaves(a.flow, flow)
        print(f"wrote {a.flow}")
        check(flow, val_set)
        return

    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    if a.mode == "check":
        print(f"val nll {val_nll(flow, val_set, jr.key(99)):.4f}")
        check(flow, val_set)
    else:
        derivs_plot(flow, a.log10mf, a.mrmf, a.m_range, a.n, a.out)


if __name__ == "__main__":
    main()
