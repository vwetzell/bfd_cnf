"""
plot_shear_derivs_compare.py
============================
Recreate the :mod:`bfd_cnf.plot_shear_derivs` diagnostic for *several* Σ_X
ellipticity slices using a **shared color scale** across all of them, so the
panels can be compared by eye.

The single-slice :mod:`plot_shear_derivs` auto-scales every plot's colorbars
from its own data, so two slices that differ only in ``e2`` get different color
ranges and cannot be compared directly.  Here we evaluate all requested slices
first, take the colour limits over the *union* of their data, and render each
slice with those common limits.

Like :mod:`plot_shear_derivs`, this only loads and evaluates the trained prior
flow.  The standardiser ``(mean, std)`` is read from the cached
``data/raw2standard_stats.npz`` (written by ``integrate_grid.load_raw2standard``)
so the multi-GB FITS template table is not needed.

Run from the repo root::

    cd /home/vwetzell/gitrepos/bfd_cnf
    python -m bfd_cnf.plot_shear_derivs_compare --log-scale 13.3 --e2 0.0 0.2
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

from bfd_cnf.config import (
    DATA_DIR,
    PLOTS_DIR,
    PRIOR_FLOW_PATH,
    Q_FLOW_PATH,
    key as base_key,
)
from bfd_cnf.models.flows import build_flows
from bfd_cnf.training import load_models

STATS_FILE = os.path.join(DATA_DIR, "raw2standard_stats.npz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recreate shear-derivative slices with a shared color scale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mf = p.add_mutually_exclusive_group()
    mf.add_argument("--log10mf", type=float, default=math.log10(2000.0),
                    help="log10 of the flux moment Mf at which to slice.")
    mf.add_argument("--mf", type=float, default=None,
                    help="Flux moment Mf in raw units (overrides --log10mf).")
    p.add_argument("--mrmf", type=float, default=3.3, help="Size ratio Mr/Mf.")
    p.add_argument("--log-scale", type=float, default=13.3,
                   help="Σ_X log-noise-scale 0.5*log det(Sigma_X).")
    p.add_argument("--e1", type=float, default=0.0, help="Σ_X ellipticity e1.")
    p.add_argument("--e2", type=float, nargs="+", default=[0.0, 0.2],
                   help="One or more Σ_X e2 values; one PNG per value, shared scale.")
    p.add_argument("--g1", type=float, default=0.0, help="Shear g1 for derivatives.")
    p.add_argument("--g2", type=float, default=0.0, help="Shear g2 for derivatives.")
    p.add_argument("--n", type=int, default=101, help="Grid resolution per axis.")
    p.add_argument("--m-range", type=float, default=0.5,
                   help="Half-width of the M1/Mr, M2/Mr axes.")
    p.add_argument("--prior", default=None,
                   help="Prior-flow .eqx to evaluate (default: config canonical).")
    p.add_argument("--q", default=None,
                   help="Q-flow .eqx (default: config canonical).")
    p.add_argument("--stats", default=None,
                   help="raw2standard_stats npz (mean,std) the flow was trained with "
                        "(default: legacy data/raw2standard_stats.npz).")
    p.add_argument("--suffix", type=str, default="_shared",
                   help="Filename suffix marking the shared-scale recreation.")
    return p.parse_args()


def load_standardiser(
    prior_path: str, override: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(mean, std)`` the prior flow at ``prior_path`` was trained with.

    Prefers the flow's sidecar ``<prior_path>.stats.npz``; an explicit
    ``override`` (``--stats``) must agree with it or this raises.
    """
    from bfd_cnf.models.bijections import load_stats
    r2s = load_stats(prior_path, override=override)
    return np.asarray(r2s.mean, np.float32), np.asarray(r2s.std, np.float32)


def compute_planes(prior_trained, mean, std, *, log10mf, mrmf, log_scale,
                   e1, e2, g1, g2, n, r):
    """Evaluate P and its 1st/2nd shear derivatives on the M1/Mr - M2/Mr grid.

    Returns a dict of six (n, n) arrays keyed
    ``p, dg1, dg2, dg1dg1, dg2dg2, dg1dg2`` (rows = M2/Mr, cols = M1/Mr).
    """
    m1mr = np.linspace(-r, r, n)
    m2mr = np.linspace(-r, r, n)
    x = (
        np.array(np.meshgrid([log10mf], [mrmf], m1mr, m2mr))
        .reshape(4, -1)
        .T.astype(np.float32)
    )
    x = jnp.asarray((x - mean) / std)

    def prob(x_row, g1, g2, log_scale, e1, e2):
        condition = jnp.array([g1, g2, log_scale, e1, e2], dtype=jnp.float32)
        return jnp.exp(prior_trained.log_prob(x_row, condition=condition))

    in_axes = (0, None, None, None, None, None)
    prob_v = jax.vmap(prob, in_axes=in_axes)
    dg1_v = jax.vmap(jax.grad(prob, argnums=1), in_axes=in_axes)
    dg2_v = jax.vmap(jax.grad(prob, argnums=2), in_axes=in_axes)
    hess_v = jax.vmap(jax.hessian(prob, argnums=(1, 2)), in_axes=in_axes)

    args = (x, g1, g2, log_scale, e1, e2)
    probs = np.array(prob_v(*args), np.float32)
    dp_dg1 = np.array(dg1_v(*args), np.float32)
    dp_dg2 = np.array(dg2_v(*args), np.float32)
    hess = hess_v(*args)

    def to_plane(a):
        return a.reshape((n, n)).T

    return {
        "p": to_plane(probs),
        "dg1": to_plane(dp_dg1),
        "dg2": to_plane(dp_dg2),
        "dg1dg1": to_plane(np.array(hess[0][0], np.float32)),
        "dg2dg2": to_plane(np.array(hess[1][1], np.float32)),
        "dg1dg2": to_plane(np.array(hess[0][1], np.float32)),
    }, (m1mr, m2mr)


def render(planes, axes_coords, *, log10mf, mf_raw, mrmf, log_scale, e1, e2,
           g1, g2, p_vmax, dp_absmax, d2p_absmax, d2p_cross_absmax, out):
    """Render one 2x3 slice with externally supplied (shared) colour limits."""
    m1mr, m2mr = axes_coords
    extent = (m1mr[0], m1mr[-1], m2mr[0], m2mr[-1])
    imshow_kw = dict(aspect="auto", origin="lower", extent=extent, interpolation="none")

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    panels = [
        (ax[0, 0], planes["p"], "Blues", (0.0, p_vmax), r"$P$"),
        (ax[0, 1], planes["dg1"], "RdBu", (-dp_absmax, dp_absmax), r"$\frac{\partial P}{\partial g_1}$"),
        (ax[0, 2], planes["dg2"], "RdBu", (-dp_absmax, dp_absmax), r"$\frac{\partial P}{\partial g_2}$"),
        (ax[1, 0], planes["dg1dg1"], "RdBu", (-d2p_absmax, d2p_absmax), r"$\frac{\partial^2 P}{\partial g_1^2}$"),
        (ax[1, 1], planes["dg2dg2"], "RdBu", (-d2p_absmax, d2p_absmax), r"$\frac{\partial^2 P}{\partial g_2^2}$"),
        (ax[1, 2], planes["dg1dg2"], "RdBu", (-d2p_cross_absmax, d2p_cross_absmax), r"$\frac{\partial^2 P}{\partial g_1 \partial g_2}$"),
    ]
    for a, field, cmap, (vmin, vmax), label in panels:
        im = a.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kw)
        a.set_xlabel(r"$M_1/M_r$", fontsize=20)
        a.set_ylabel(r"$M_2/M_r$", fontsize=20)
        a.axvline(0.0, color="grey", lw=1)
        a.axhline(0.0, color="grey", lw=1)
        plt.colorbar(im, ax=a).set_label(label=label, size=20)

    fig.suptitle(
        rf"$\log_{{10}}M_f = {log10mf:.3g}$ ($M_f = {mf_raw:.0f}$), "
        rf"$M_r/M_f = {mrmf:.3g}$, $g = ({g1:.3g}, {g2:.3g})$, "
        rf"$\log\,\mathrm{{scale}} = {log_scale:.3g}$, $e = ({e1:.3g}, {e2:.3g})$ "
        rf"[shared color scale]",
        fontsize=22,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


def main() -> None:
    args = parse_args()
    log10mf = math.log10(args.mf) if args.mf is not None else args.log10mf
    mf_raw = 10.0**log10mf

    print("Loading flow...")
    prior_path = args.prior or PRIOR_FLOW_PATH
    q_path = args.q or Q_FLOW_PATH
    mean, std = load_standardiser(prior_path, args.stats)

    if not (os.path.exists(prior_path) and os.path.exists(q_path)):
        raise FileNotFoundError(f"Trained flow weights not found:\n  {prior_path}\n  {q_path}")
    prior_flow, q_flow = build_flows(base_key, latent_dim=4, cond_dim=16, prior_size_loc_c1=float(mean[1] / std[1]))
    prior_trained, _ = load_models(prior_flow, q_flow, prior_path, q_path)

    # ── Evaluate every requested e2 slice first ─────────────────────────────────
    results = []
    for e2 in args.e2:
        print(f"Evaluating prior flow for e2 = {e2:.3g} ...")
        planes, axes_coords = compute_planes(
            prior_trained, mean, std, log10mf=log10mf, mrmf=args.mrmf,
            log_scale=args.log_scale, e1=args.e1, e2=e2, g1=args.g1, g2=args.g2,
            n=args.n, r=args.m_range,
        )
        results.append((e2, planes, axes_coords))

    # ── Shared colour limits over the union of all slices ───────────────────────
    # Keep plot_shear_derivs' 0.5x saturation factor so faint structure stays visible.
    all_planes = [pl for _, pl, _ in results]
    p_vmax = max(np.max(pl["p"]) for pl in all_planes)
    dp_absmax = 0.5 * max(
        max(np.max(np.abs(pl["dg1"])), np.max(np.abs(pl["dg2"]))) for pl in all_planes
    )
    d2p_absmax = 0.5 * max(
        max(np.max(np.abs(pl["dg1dg1"])), np.max(np.abs(pl["dg2dg2"]))) for pl in all_planes
    )
    d2p_cross_absmax = 0.5 * max(np.max(np.abs(pl["dg1dg2"])) for pl in all_planes)

    # ── Render each slice with the shared limits ────────────────────────────────
    for e2, planes, axes_coords in results:
        out = os.path.join(
            PLOTS_DIR,
            f"shear_derivs_m1mr_m2mr_lgmf{log10mf:.2f}_mrmf{args.mrmf:.2f}"
            f"_ls{args.log_scale:.1f}_e1{args.e1:.2f}_e2{e2:.2f}{args.suffix}.png",
        )
        render(
            planes, axes_coords, log10mf=log10mf, mf_raw=mf_raw, mrmf=args.mrmf,
            log_scale=args.log_scale, e1=args.e1, e2=e2, g1=args.g1, g2=args.g2,
            p_vmax=p_vmax, dp_absmax=dp_absmax, d2p_absmax=d2p_absmax,
            d2p_cross_absmax=d2p_cross_absmax, out=out,
        )


if __name__ == "__main__":
    main()
