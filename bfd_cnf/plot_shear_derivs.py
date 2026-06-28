"""
plot_shear_derivs.py
====================
Reproduce the "shear derivatives in the M1/Mr – M2/Mr plane" diagnostic from the
exploratory notebooks (``cNF_BFD_VI_custom_arc_xy*.ipynb``).

At a *fixed* flux ``log10(Mf)`` and size ratio ``Mr/Mf``, this sweeps a 2-D grid
over the ellipticity moments ``M1/Mr`` (x-axis) and ``M2/Mr`` (y-axis) and
evaluates the trained prior flow probability ``P`` together with its first and
second derivatives with respect to the applied shear ``(g1, g2)``:

    P,  ∂P/∂g1,  ∂P/∂g2,  ∂²P/∂g1²,  ∂²P/∂g2²,  ∂²P/∂g1∂g2

The derivatives are taken at ``g = (g1, g2)`` (default ``g = 0``) for a provided
PSF log-noise-scale and ellipticity ``(e1, e2)``, exactly the condition vector
``[g1, g2, log_scale, e1, e2]`` the prior flow was trained on.

Run from the repo root so the package import resolves::

    cd /home/vwetzell/gitrepos/bfd_cnf
    python -m bfd_cnf.plot_shear_derivs --log10mf 3.3 --mrmf 3.3 --log-scale 12 --e2 0.25

By default ``--log10mf`` is interpreted as log10(Mf); pass ``--mf`` instead to
give Mf in raw moment units.
"""

from __future__ import annotations

import argparse
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bfd_cnf.models.flows import build_flows
from bfd_cnf.training import load_models
from bfd_cnf.config import PLOTS_DIR, PRIOR_FLOW_PATH, Q_FLOW_PATH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot prior-flow shear derivatives in the M1/Mr – M2/Mr plane.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mf = p.add_mutually_exclusive_group()
    mf.add_argument(
        "--log10mf",
        type=float,
        default=math.log10(2000.0),
        help="log10 of the flux moment Mf at which to slice.",
    )
    mf.add_argument(
        "--mf",
        type=float,
        default=None,
        help="Flux moment Mf in raw units (overrides --log10mf).",
    )
    p.add_argument(
        "--mrmf", type=float, default=3.3, help="Size ratio Mr/Mf at which to slice."
    )
    p.add_argument(
        "--log-scale",
        type=float,
        default=12.0,
        help="PSF log-noise-scale  log_scale = 0.5 * log det(Sigma_X).",
    )
    p.add_argument("--e1", type=float, default=0.0, help="PSF ellipticity component e1.")
    p.add_argument("--e2", type=float, default=0.0, help="PSF ellipticity component e2.")
    p.add_argument(
        "--g1", type=float, default=0.0, help="Shear g1 at which derivatives are taken."
    )
    p.add_argument(
        "--g2", type=float, default=0.0, help="Shear g2 at which derivatives are taken."
    )
    p.add_argument(
        "--n", type=int, default=101, help="Grid resolution along each ellipticity axis."
    )
    p.add_argument(
        "--m-range",
        type=float,
        default=0.5,
        help="Half-width of the M1/Mr and M2/Mr axes: linspace(-r, r, n).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: auto-named from the slice parameters).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log10mf = math.log10(args.mf) if args.mf is not None else args.log10mf
    mf_raw = 10.0**log10mf
    mrmf = args.mrmf
    g1, g2 = args.g1, args.g2
    log_scale, e1, e2 = args.log_scale, args.e1, args.e2
    n = args.n
    r = args.m_range

    # ── Load data (for the standardisation) and the trained prior flow ──────────
    print("Loading standardiser from the flow's sidecar...")
    from bfd_cnf.models.bijections import load_stats
    _r2s = load_stats(PRIOR_FLOW_PATH)
    key = jax.random.key(0)
    data_mean = np.array(_r2s.mean)
    data_std = np.array(_r2s.std)

    print("Loading flow...")
    if not (os.path.exists(PRIOR_FLOW_PATH) and os.path.exists(Q_FLOW_PATH)):
        raise FileNotFoundError(
            "Trained flow weights not found:\n"
            f"  {PRIOR_FLOW_PATH}\n  {Q_FLOW_PATH}\n"
            "Train the flows first (e.g. `python -m bfd_cnf.train_from_scratch`). "
            "This script only loads and evaluates the prior flow."
        )
    from bfd_cnf.models.bijections import load_stats
    prior_flow, q_flow = build_flows(
        key, latent_dim=4, cond_dim=16, raw2standard=load_stats(PRIOR_FLOW_PATH)
    )
    prior_trained, _ = load_models(prior_flow, q_flow, PRIOR_FLOW_PATH, Q_FLOW_PATH)

    # ── Build the M1/Mr – M2/Mr grid at fixed (log10 Mf, Mr/Mf) ─────────────────
    # Transformed moment coordinates are [log10(Mf), Mr/Mf, M1/Mr, M2/Mr].
    m1mr = np.linspace(-r, r, n)  # x-axis
    m2mr = np.linspace(-r, r, n)  # y-axis
    n1 = n2 = 1
    x = (
        np.array(np.meshgrid([log10mf], [mrmf], m1mr, m2mr))
        .reshape(4, -1)
        .T.astype(np.float32)
    )
    # Standardise into the flow's latent input space.
    x = (x - data_mean) / data_std
    x = jnp.asarray(x)

    # ── P and shear derivatives of the prior flow ───────────────────────────────
    def prob(x_row, g1, g2, log_scale, e1, e2):
        condition = jnp.array([g1, g2, log_scale, e1, e2], dtype=jnp.float32)
        return jnp.exp(prior_trained.log_prob(x_row, condition=condition))

    # log_scale, e1, e2 are fixed across the grid → broadcast (in_axes None).
    in_axes = (0, None, None, None, None, None)
    prob_v = jax.vmap(prob, in_axes=in_axes)
    dg1_v = jax.vmap(jax.grad(prob, argnums=1), in_axes=in_axes)
    dg2_v = jax.vmap(jax.grad(prob, argnums=2), in_axes=in_axes)
    hess_v = jax.vmap(jax.hessian(prob, argnums=(1, 2)), in_axes=in_axes)

    print(f"Evaluating prior flow on {x.shape[0]} grid points...")
    probs = np.array(prob_v(x, g1, g2, log_scale, e1, e2), dtype=np.float32)
    dp_dg1 = np.array(dg1_v(x, g1, g2, log_scale, e1, e2), dtype=np.float32)
    dp_dg2 = np.array(dg2_v(x, g1, g2, log_scale, e1, e2), dtype=np.float32)
    hess = hess_v(x, g1, g2, log_scale, e1, e2)
    d2p_dg1_dg1 = np.array(hess[0][0], dtype=np.float32)
    d2p_dg2_dg2 = np.array(hess[1][1], dtype=np.float32)
    d2p_dg1_dg2 = np.array(hess[0][1], dtype=np.float32)

    # Reshape (n1, n2, n3, n4) and slice the single (log10mf, mrmf) point; transpose
    # so rows index M2/Mr (y) and columns index M1/Mr (x) to match imshow extent.
    def to_plane(a):
        return a.reshape((n1, n2, n, n))[0, 0, :, :].T

    p_plane = to_plane(probs)
    dp_dg1_plane = to_plane(dp_dg1)
    dp_dg2_plane = to_plane(dp_dg2)
    d2p_dg1_dg1_plane = to_plane(d2p_dg1_dg1)
    d2p_dg2_dg2_plane = to_plane(d2p_dg2_dg2)
    d2p_dg1_dg2_plane = to_plane(d2p_dg1_dg2)

    # ── Plot: 2×3 grid mirroring the notebook layout ────────────────────────────
    extent = (m1mr[0], m1mr[-1], m2mr[0], m2mr[-1])
    imshow_kw = dict(aspect="auto", origin="lower", extent=extent, interpolation="none")

    dp_absmax = 0.5 * max(
        np.max(np.abs(dp_dg1_plane)), np.max(np.abs(dp_dg2_plane))
    )
    d2p_absmax = 0.5 * max(
        np.max(np.abs(d2p_dg1_dg1_plane)), np.max(np.abs(d2p_dg2_dg2_plane))
    )
    d2p_cross_absmax = 0.5 * np.max(np.abs(d2p_dg1_dg2_plane))

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))

    panels = [
        (ax[0, 0], p_plane, "Blues", None, r"$P$"),
        (ax[0, 1], dp_dg1_plane, "RdBu", dp_absmax, r"$\frac{\partial P}{\partial g_1}$"),
        (ax[0, 2], dp_dg2_plane, "RdBu", dp_absmax, r"$\frac{\partial P}{\partial g_2}$"),
        (ax[1, 0], d2p_dg1_dg1_plane, "RdBu", d2p_absmax, r"$\frac{\partial^2 P}{\partial g_1^2}$"),
        (ax[1, 1], d2p_dg2_dg2_plane, "RdBu", d2p_absmax, r"$\frac{\partial^2 P}{\partial g_2^2}$"),
        (ax[1, 2], d2p_dg1_dg2_plane, "RdBu", d2p_cross_absmax, r"$\frac{\partial^2 P}{\partial g_1 \partial g_2}$"),
    ]
    for a, field, cmap, absmax, label in panels:
        kw = dict(imshow_kw, cmap=cmap)
        if absmax is not None:
            kw.update(vmin=-absmax, vmax=absmax)
        im = a.imshow(field, **kw)
        a.set_xlabel(r"$M_1/M_r$", fontsize=20)
        a.set_ylabel(r"$M_2/M_r$", fontsize=20)
        a.axvline(0.0, color="grey", lw=1)
        a.axhline(0.0, color="grey", lw=1)
        plt.colorbar(im, ax=a).set_label(label=label, size=20)

    fig.suptitle(
        rf"$\log_{{10}}M_f = {log10mf:.3g}$ ($M_f = {mf_raw:.0f}$), "
        rf"$M_r/M_f = {mrmf:.3g}$, $g = ({g1:.3g}, {g2:.3g})$, "
        rf"$\log\,\mathrm{{scale}} = {log_scale:.3g}$, $e = ({e1:.3g}, {e2:.3g})$",
        fontsize=22,
    )
    plt.tight_layout()

    out = args.out
    if out is None:
        out = os.path.join(
            PLOTS_DIR,
            f"shear_derivs_m1mr_m2mr_lgmf{log10mf:.2f}_mrmf{mrmf:.2f}"
            f"_ls{log_scale:.1f}_e1{e1:.2f}_e2{e2:.2f}.png",
        )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
