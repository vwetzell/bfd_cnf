"""Fit the flow's density to NOISY targets through the noise kernel.

The self-consistency control (`dev/check_selfconsistency_noisy.py`) draws its
targets FROM the flow, so the flow is truth by construction: it measures the
estimator's own bias and is blind, by design, to the flow being wrong.  That
blind spot is where the residual bias lives.

This is the truth-free attack on it.  You never observe p(m), but you DO observe
M, and the model's prediction for it is the convolution the estimator already
computes,

    P(M | g=0) = INT dm p(m | g=0) L(M - m),      L = N(., C_M)

so the target catalog is itself a likelihood for the prior.  Maximising it moves
the density towards the one that actually generated the data -- no truth, no
simulation, no second catalog.

Why this does not eat the shear signal
--------------------------------------
Fit a prior to sheared targets naively and it absorbs the very signal you are
measuring.  The escape is that shear is a SPIN-2 distortion with a coherent
orientation, so rotating each target's (M1, M2) pair by its own random angle
destroys it while leaving every spin-0 quantity -- Mf, Mr, Mc, and hence the
size and concentration directions where this project's residual bias lives --
bit-for-bit unchanged.  What survives the symmetrisation is the population's
own |e| distribution, which shear moves only at O(g^2) ~ 4e-4 for an isotropic
population.  So the assumption is isotropy of intrinsic orientations, the
standard weak-lensing one, and NOT knowledge of g.

The rotation needs no change to C_M: it is exactly spin-2 isotropic here
(cov[2,2] == cov[3,3] to the bit, cross terms ~1e-12), so it is invariant.

What is fit
-----------
Only the bulk layers.  The shear layer and the standardisation are frozen: the
targets are symmetrised, so they carry no information about the g response, and
a free shear layer would fit noise with it.

`log (1/S) sum_s w_s` is a lower bound on `log P` (Jensen), tightening as S
grows -- the IWAE bound.  At the ESS this reaches (several hundred) the gap is
small next to the density error being chased, but it is a bound, so S is not a
free knob to economise on.

    python dev/refine_prior.py --flow flows/shear_gauss2.eqx --pop gauss2 \
        --noise-scale 2.73 --out flows/shear_gauss2_refined.eqx
"""
import argparse
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402
from models.centroid import CentroidMarginalize      # noqa: E402

DATA = "../bfd_cnf_imsims/data"
LOG_FLOOR = -80.0        # see `objective`; real per-target values run ~ -44


def symmetrise(m, key):
    """Rotate each row's spin-2 pair by its own random angle.

    (M1, M2) is spin-2, so a rotation by phi takes it through 2 phi; the spin-0
    entries are untouched.  Drawing one angle per target samples the
    orientation-averaged density rather than approximating it.
    """
    phi = jr.uniform(key, (len(m),), minval=0.0, maxval=2.0 * np.pi)
    c, s = jnp.cos(2 * phi), jnp.sin(2 * phi)
    m1, m2 = m[:, 2], m[:, 3]
    return m.at[:, 2].set(c * m1 - s * m2).at[:, 3].set(s * m1 + c * m2)


def trainable(flow):
    """Everything past the data-adjacent layers and the standardisation: the bulk.

    Data -> base the chain is [raw2standard, centroid?, shear, *bulk] since
    16ece5c, so the bulk starts at 2 or 3 depending on whether the centroid
    layer is there -- but the TEST for it must scan the chain, not look at
    bijections[0], which is now always the chart.  As written it silently took
    start = 2 for centroid flows and left the shear layer trainable.
    Freezing both conditional layers is the point: the targets are symmetrised,
    so they carry no information about the g response, and Sigma_X is not being
    re-fit here either.
    """
    bij = flow.bijection.bijection.bijections
    start = 3 if any(isinstance(b, CentroidMarginalize) for b in bij) else 2
    spec = jax.tree.map(lambda _: False, flow)
    return eqx.tree_at(
        lambda f: f.bijection.bijection.bijections[start:], spec,
        replace=[jax.tree.map(eqx.is_inexact_array, b) for b in bij[start:]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear_gauss2.eqx")
    p.add_argument("--out", default="flows/shear_gauss2_refined.eqx")
    p.add_argument("--pop", default="gauss2", choices=sorted(bias.CATALOGS))
    p.add_argument("--catalog", default="plus", choices=["plus", "minus", "zero"],
                   help="which arm to FIT on. Default is a sheared one, since "
                        "the point is that the fit does not need an unsheared "
                        "sample -- the symmetrisation handles it.")
    p.add_argument("--train-data", default=None)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--noise-seed", type=int, default=1)
    p.add_argument("-n", type=int, default=40000, help="targets to fit on")
    p.add_argument("--offset", type=int, default=20000,
                   help="skip this many rows first. The measurement uses the "
                        "catalog's PREFIX, and add_noise gives a galaxy the same "
                        "noise vector in all three arms, so fitting on the same "
                        "rows would let the prior learn that realization.")
    p.add_argument("--samples", type=int, default=1024,
                   help="kernel draws per target per step (the IWAE bound's S)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--symmetrise", action=argparse.BooleanOptionalAction,
                   default=True, help="--no-symmetrise is the control that shows "
                                      "the fit eating the shear signal")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    cat = bias.CATALOGS[a.pop]
    # Same switch `bias.main` reads: an IMGNOISE catalog already carries its
    # noise (put there before a real recenter()), so it must not be handed to
    # `add_noise`, and its prior is the centroid-marginalised one.
    img_noise = bool(fitsio.read_header(f"{DATA}/{cat['zero']}.fits",
                                        ext=1).get("IMGNOISE", False))
    if img_noise and a.noise_scale != 1.0:
        raise SystemExit("--noise-scale cannot deepen an IMGNOISE catalog")

    rows = fitsio.read(f"{DATA}/{cat[a.catalog]}.fits")[a.offset:a.offset + a.n]
    if img_noise:
        rows = rows[~rows["badcenter"]]
    m = np.asarray(rows["moments"], dtype=np.float64)
    cov = bias.load_cov(f"{DATA}/{cat['zero']}.fits") * a.noise_scale ** 2
    # Sigma_X conditions the centroid layer. Constant across these catalogs, and
    # isotropic, so the spin-2 rotation below leaves it invariant just as it
    # leaves C_M invariant.
    sigma_x = (jnp.asarray(rows["cov_odd"][0], dtype=jnp.float32)
               if img_noise else None)
    M = m if img_noise else bias.add_noise(m, cov, a.noise_seed)
    key = jr.key(a.seed)
    key, k_sym = jr.split(key)
    if a.symmetrise:
        M = np.asarray(symmetrise(jnp.asarray(M), k_sym), dtype=np.float64)
    # Small on purpose: validation runs the same (targets x samples) grid as a
    # training step, so a "held-out set" of thousands is a 26 GB allocation.
    n_val = min(256, len(M) // 5)
    M_val, M_fit = jnp.asarray(M[:n_val]), jnp.asarray(M[n_val:])
    print(f"fitting {len(M_fit)} targets from {cat[a.catalog]}[{a.offset}:], "
          f"noise x{a.noise_scale}, symmetrise={a.symmetrise}, "
          f"centroid={img_noise}, {n_val} held out")

    train_data = a.train_data or f"{DATA}/moments.fits"
    m_train = shear_top.load(train_data)[0]
    if not img_noise:
        m_train = m_train[:int(0.9 * len(m_train))]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=img_noise)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)

    L = jnp.asarray(np.linalg.cholesky(cov), dtype=jnp.float32)
    # The condition the whole fit runs at: g = 0 (the targets are symmetrised,
    # so they carry no g information) and, with the centroid layer, the targets'
    # own Sigma_X.
    zero = bias.condition(jnp.zeros(2), sigma_x)

    def objective(model, Mb, key):
        """-mean_i log (1/S) sum_s p(M_i + eps_s | g=0), eps ~ N(0, C_M).

        `bias.log_conv_is` at log_wt = 0 -- the kernel as its own proposal, the
        alpha = 1 estimator, with its in-domain mask and safe dummy already in
        it.  Antithetic in pairs, as `bias.kernel_draws` is, for the same
        reason: the prior varies nearly linearly across a narrow kernel.

        The centroid layer is PEELED first, exactly as `bias.pqr_streamed` does,
        and for a reason that is not about speed here: `in_domain` has to be
        tested on the point the chart actually sees.  Applied to the raw draws it
        admits 0.27% that the layer then maps off the chart -- their log_prob is
        -inf, which the value survives and the GRADIENT does not (170 of 172
        gradient leaves came back NaN).  The layer's log-det is frozen and
        g-independent, so it rides along as an importance weight.
        """
        half = jr.normal(key, (Mb.shape[0], a.samples // 2, 5)) @ L.T
        eps = jnp.concatenate([half, -half], axis=1)
        rest, layer = bias.split_centroid(model)
        if layer is None:
            lp = jax.vmap(lambda M_i, e: bias.log_conv(model, M_i, e, zero))(Mb, eps)
        else:
            tld = jax.vmap(layer.transform_and_log_det, in_axes=(0, None))
            z, ld = jax.vmap(tld, in_axes=(0, None))(Mb[:, None, :] + eps, zero)
            # A draw whose log-det is not finite is sent somewhere `in_domain`
            # rejects, so `log_conv_is` masks it with a CONSTANT -inf.  Leaving
            # it in would put a COMPUTED -inf in the logsumexp, whose gradient is
            # NaN -- the same distinction that matters in `safe_point`.  Mf = 0
            # is the cheapest such point.
            bad = ~jnp.isfinite(ld)
            z = jnp.where(bad[..., None], jnp.zeros_like(z), z)
            ld = jnp.where(bad, 0.0, ld)
            lp = jax.vmap(lambda M_i, d, w: bias.log_conv_is(rest, M_i, d, w,
                                                             zero))(Mb, z, ld)
        # A target every one of whose draws lands off-support gets -inf, and
        # d(-inf)/dtheta is NaN -- one such target kills the run (it happens at
        # the deep depth, where the kernel is wide next to a support with hard
        # edges).  The floor is a constant far below any real per-target value
        # (~e^-44), so it changes nothing that is finite and contributes exactly
        # zero gradient for the ones it catches: they leave the fit rather than
        # poison it.  `LOG_FLOOR` deliberately does NOT rescue a starved
        # PROPOSAL -- if the count is more than a trickle, raise --samples.
        return -jnp.mean(jnp.logaddexp(lp, LOG_FLOOR))

    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(optax.cosine_decay_schedule(a.lr, a.steps)))
    params, static = eqx.partition(flow, trainable(flow))
    state = opt.init(params)

    @eqx.filter_jit
    def step(params, state, idx, key):
        loss, grads = jax.value_and_grad(
            lambda p: objective(eqx.combine(p, static), M_fit[idx], key))(params)
        updates, state = opt.update(grads, state, params)
        return eqx.apply_updates(params, updates), state, loss

    @eqx.filter_jit
    def validate(params, key):
        return objective(eqx.combine(params, static), M_val, key)

    rng = np.random.default_rng(a.seed)
    for i in range(a.steps):
        key, k_b = jr.split(key)
        idx = jnp.asarray(rng.integers(0, len(M_fit), a.batch))
        params, state, loss = step(params, state, idx, k_b)
        if i % 100 == 0 or i == a.steps - 1:
            # A FIXED key for validation.  The objective is a Monte Carlo
            # estimate whose scatter over 256 targets is ~0.1 nat -- larger than
            # the improvement being watched for -- so with fresh draws each call
            # the curve is unreadable.  Held fixed, a change is the parameters.
            print(f"step {i:5d}  fit {float(loss):.4f}  "
                  f"val {float(validate(params, jr.key(12345))):.4f}", flush=True)

    eqx.tree_serialise_leaves(a.out, eqx.combine(params, static))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
