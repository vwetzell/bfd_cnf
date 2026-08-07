"""
bulk.py
=======
Phase 1: learn p(m) for the imsims Gaussian-galaxy population, using only the
*bulk* (unconditional) layers of the bfd_cnf flow.

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
from models.shear import ShearResponse

LAYERS = 8
NN_WIDTH = 64
NN_DEPTH = 2

# The flow works in t = [log10(Mf), Mr/Mf, M1/Mr, M2/Mr] (RawMomentStandardize),
# standardised by the training set's own mean/std.  Label the corner plot in it.
COORD_LABELS = [r"$\log_{10}M_f$", r"$M_r / M_f$", r"$M_1 / M_r$", r"$M_2 / M_r$"]
LABEL_FONTSIZE, TICK_LABELSIZE = 34, 24
# Mr/Mf ceiling: a point source, i.e. the PSF itself.  Anything at or above it is
# unresolved and carries no shape information -- the old repo drew it as the
# "stellar locus", and it is the same number here (same weight function).
POINT_SOURCE = 3.976167


def load_moments(path):
    """Read the four even moments [Mf, Mr, M1, M2] from an imsims catalog."""
    return np.asarray(fitsio.read(path)["moments"][:, :4], dtype=np.float64)


def to_coords(m):
    """Raw moments -> the flow's transformed coordinates t (for plotting)."""
    return np.stack([np.log10(m[:, 0]), m[:, 1] / m[:, 0],
                     m[:, 2] / m[:, 1], m[:, 3] / m[:, 1]], axis=-1)


def build_flow(key, m_train, layers=LAYERS, nn_width=NN_WIDTH, nn_depth=NN_DEPTH,
               shear=False):
    """Bulk flow standardised against `m_train`; with `shear`, conditioned on g.

    The generative stack is ``base -> bulk -> shear(g) -> data``, so the shear
    layer is data-adjacent and acts in raw moment space -- which is where shear
    physically acts, and where its coefficients are comparable to bfd's dm/dg.
    """
    t = to_coords(m_train)
    raw2standard = RawMomentStandardize(mean=t.mean(0), std=t.std(0))

    key, k_shear = jr.split(key)
    keys = jr.split(key, layers)
    bulk = []
    for i, k in enumerate(keys):
        bulk.append(EquivariantAutoregressiveLayer(k, nn_width, nn_depth, jax.nn.silu))
        # Alternate the spin-0 (flux/size) order so both directions get conditioned;
        # spin-2 is left alone -- swapping M1/M2 would break the equivariance.
        bulk.append(Permute(jnp.array([1, 0, 2, 3] if i % 2 else [0, 1, 2, 3])))

    # Chain.transform runs in list order and maps data -> base, so the
    # data-adjacent layers come first; Invert flips it for sampling.
    head = [ShearResponse(k_shear)] if shear else []
    bijection = Invert(Chain([*head, raw2standard, *bulk]).merge_chains())
    base = non_trainable(MultivariateNormal(jnp.zeros(4), jnp.eye(4)))
    return Transformed(base, bijection)


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "corner"])
    p.add_argument("--data", default="../bfd_cnf_imsims/data/moments.fits")
    p.add_argument("--flow", default="flows/bulk.eqx")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="plots/bulk_corner.png")
    a = p.parse_args()

    m = load_moments(a.data)
    m_train, m_val = _split(m)
    key = jr.key(a.seed)
    k_build, k_train, k_sample = jr.split(key, 3)
    flow = build_flow(k_build, m_train)

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
    plot_range = [np.percentile(d[:, i], [0.05, 99.95]) for i in range(4)]
    # Zoom out on flux and size: both are bounded by the population's own cuts, so
    # padding puts the point-source line and the empty margin beyond each edge in
    # frame -- that is where a flow leaking mass off the support would show up.
    for i in (0, 1):
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

    axs = fig.axes
    axs[5].axvline(POINT_SOURCE, lw=1, color="tab:red", label="Point source")
    axs[4].axhline(POINT_SOURCE, lw=1, color="tab:red")
    for i in (9, 13):
        axs[i].axvline(POINT_SOURCE, lw=1, color="tab:red", zorder=500)
    for i in (8, 12, 14):
        axs[i].axhline(0.0, lw=1, color="tab:red", zorder=500)
    for i in (10, 14, 15):
        axs[i].axvline(0.0, lw=1, color="tab:red", zorder=500)
    axs[5].legend(bbox_to_anchor=(0.0, 1.0), loc="lower left", fontsize=16)
    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)

    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
