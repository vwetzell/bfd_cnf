"""
Corner plot of the template moment distribution vs. the trained prior flow.

One pipeline, two template sources (``--source``), both from
``load_training_dataset`` — the exact set the flow trains on:

  train    the EXACT training population (quality + derivative cuts, selection box),
           nda·N(X|C_X) weighted — NO per-copy detj, matching the nll Σ_X loss the
           flow was trained on.
  draw     1M templates drawn EXACTLY as training selects batches (``ds["weights"]``),
           optionally × nda. Data-only by default (the distribution the loss sees).

Shared across sources: g=0, e=0 flow sampling at ``--log-scale``; the
[log10 Mf, Mr/Mf, M1/Mr, M2/Mr] coordinates; count equalisation; and the
selection-box (green) + stellar-locus (red) overlays.

Pure load — never trains. Examples::

    python -m bfd_cnf.plot_corner --source train --log-scale 11.94
    python -m bfd_cnf.plot_corner --source draw
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import corner
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from bfd_cnf.config import PLOTS_DIR, PRIOR_FLOW_PATH, Q_FLOW_PATH

LABELS = [r"$\log_{10}M_f$", r"$M_r / M_f$", r"$M_1 / M_r$", r"$M_2 / M_r$"]
PLOT_RANGE = ((np.log10(500), np.log10(200000)), (1.2, 5.5), (-0.8, 0.8), (-0.8, 0.8))
LABEL_FONTSIZE, TICK_LABELSIZE = 34, 24
STELLAR = 3.976167  # Mr/Mf stellar locus reference

# Per-source default Σ_X log_scale to condition the prior on.
_DEFAULT_LOG_SCALE = {"train": 11.94, "draw": 11.94}


def to_coords(m):
    """Raw moments (N,4) [Mf,Mr,M1,M2] → corner coords, finite rows only."""
    m = np.asarray(m, dtype=np.float64)
    c = np.column_stack([
        np.log10(np.abs(m[:, 0])),
        m[:, 1] / m[:, 0],
        m[:, 2] / m[:, 1],
        m[:, 3] / m[:, 1],
    ])
    return c[np.all(np.isfinite(c), axis=1)]


def resample(m, w, n, label):
    """Subsample raw moments to n rows; weighted (with ESS print) or uniform."""
    if w is None:
        sel = np.random.default_rng(0).choice(len(m), size=min(n, len(m)), replace=False)
        return m[sel]
    w = np.asarray(w, dtype=np.float64)
    w[~np.isfinite(w)] = 0.0
    w /= w.sum()
    ess = 1.0 / float(np.sum(w ** 2))
    print(f"  {label} ESS = {ess:.0f} ({100.0 * ess / len(w):.2f}% of {len(w):,})")
    return m[np.random.default_rng(0).choice(len(w), size=n, replace=True, p=w)]


# ── Source loaders: each returns (raw_moments (N,4), weights (N,) or None, label) ──

def load_train(args):
    # Blue = the EXACT training population via load_training_dataset (same quality +
    # derivative cuts the flow saw), weighted by the per-copy objective weight
    # nda·N(X|C_X) — the ONLY weights the loss uses (BFD template weight nda, already
    # HT-corrected; and the centroid marginalisation L).  No per-copy detj, no box;
    # C_X isotropic at var_xy = exp(args._log_scale) (e=0), matching flows._log_L_X.
    from bfd_cnf.data import load_training_dataset
    ds = load_training_dataset()
    m = np.asarray(ds["moments_jnp"], dtype=np.float64)                  # (N,4) post cuts
    X = np.asarray(ds["centroid_moments_jnp"], dtype=np.float64)[:, :2]  # [MX,MY] at g=0
    nda = np.asarray(ds["nda"], dtype=np.float64)                        # BFD template weight (HT)
    var_xy = float(np.exp(args._log_scale))
    L_X = np.exp(-0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2) / var_xy)
    return m, nda * L_X, "Templates (training cuts, nda·N(X|C_X))"


def load_draw(args):
    # ds["weights"] is now the batch-sampling proposal ∝ nda (the loss samples ∝ this).
    from bfd_cnf.data import load_training_dataset
    ds = load_training_dataset()
    m = np.asarray(ds["moments_jnp"], dtype=np.float64)
    w = np.asarray(ds["weights"], dtype=np.float64)  # ∝ nda (batch-sampling proposal)
    return m, w, "training draw (∝ nda)"


LOADERS = {"train": load_train, "draw": load_draw}


def load_flow(args, source):
    """Load prior flow + raw2standard (pure load; never trains)."""
    import jax.numpy as jnp
    import jax.random as jr
    from bfd_cnf.models.flows import build_flows
    from bfd_cnf.training import load_models
    from bfd_cnf.models.bijections import load_stats
    prior_path, q_path = args.prior or PRIOR_FLOW_PATH, args.q or Q_FLOW_PATH
    if not (os.path.exists(prior_path) and os.path.exists(q_path)):
        raise FileNotFoundError(f"Trained flow weights not found:\n  {prior_path}\n  {q_path}")
    # Prefer the flow's sidecar; an explicit --stats must agree with it.
    raw2standard = load_stats(prior_path, override=args.stats)
    key = jr.key(0)
    prior_flow, q_flow = build_flows(key, latent_dim=4, cond_dim=16, raw2standard=raw2standard)
    prior_trained, _ = load_models(prior_flow, q_flow, prior_path, q_path)
    return prior_trained, raw2standard, key


def sample_flow(prior_trained, raw2standard, key, n, log_scale):
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    print(f"Sampling {n:,} flow points at g=0, log_scale={log_scale:.2f}, e=0 ...")
    cond = jnp.array([0.0, 0.0, log_scale, 0.0, 0.0])
    _, sk = jr.split(key)
    raw, _ = jax.vmap(raw2standard.inverse_and_log_det)(
        prior_trained.sample(sk, sample_shape=(n,), condition=cond))
    raw = np.asarray(jnp.clip(raw, jnp.array([1.0, 1e-3, -jnp.inf, -jnp.inf]),
                              jnp.array([jnp.inf] * 4)))
    return to_coords(raw)


def draw_overlay(fig):
    """Selection-box bounds (green) + stellar-locus reference (red) on the corner grid."""
    axs = fig.axes
    axs[0].axvline(np.log10(1500), lw=1, c="tab:green")
    axs[0].axvline(np.log10(90000), lw=1, c="tab:green")
    axs[5].axvline(2.2, label="Target Bounds", lw=1, c="tab:green")
    axs[5].axvline(3.5, lw=1, c="tab:green")
    for a in (10, 15):
        axs[a].axvline(-1.0, lw=1, c="tab:green"); axs[a].axvline(1.0, lw=1, c="tab:green")
    axs[5].axvline(STELLAR, color="tab:red", lw=1, label="Stars")
    axs[4].axhline(STELLAR, lw=1, color="tab:red")
    for a in (8, 12, 13, 14):
        axs[a].axhline(0.0, lw=1, color="tab:red", zorder=500)
    axs[9].axvline(STELLAR, lw=1, color="tab:red"); axs[9].axhline(0.0, lw=1, color="tab:red", zorder=500)
    axs[13].axvline(STELLAR, lw=1, color="tab:red", zorder=500)
    axs[14].axvline(0.0, lw=1, color="tab:red", zorder=500)
    axs[10].axvline(0.0, lw=1, color="tab:red", zorder=500)
    axs[15].axvline(0.0, lw=1, color="tab:red", zorder=500)

    def rect(ax, x0, y0, dx, dy):
        ax.add_patch(mpatches.Rectangle((x0, y0), dx, dy, linewidth=1,
                                        edgecolor="tab:green", facecolor="none", zorder=100))
    w = np.log10(90000) - np.log10(1500)
    rect(axs[4], np.log10(1500), 2.2, w, 3.5 - 2.2)
    rect(axs[8], np.log10(1500), -1.0, w, 2.0)
    rect(axs[9], 2.2, -1.0, 3.5 - 2.2, 2.0)
    rect(axs[12], np.log10(1500), -1.0, w, 2.0)
    rect(axs[13], 2.2, -1.0, 3.5 - 2.2, 2.0)
    axs[14].add_patch(mpatches.Circle((0.0, 0.0), 1.0, linewidth=1,
                                      edgecolor="tab:green", facecolor="none", zorder=100))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=list(LOADERS), default="train")
    ap.add_argument("--log-scale", type=float, default=None,
                    help="Σ_X log_scale to condition the prior on / weight the X-marginal at "
                         "(default: per-source).")
    ap.add_argument("--n-plot", type=int, default=1_000_000, help="points per distribution")
    ap.add_argument("--prior", default=None, help="prior flow .eqx (default: canonical)")
    ap.add_argument("--q", default=None, help="q flow .eqx (default: canonical)")
    ap.add_argument("--stats", default=None,
                    help="raw2standard stats npz override; default reads the flow's "
                         "sidecar <flow>.eqx.stats.npz.")
    ap.add_argument("--no-flow", action="store_true", help="data cloud only, no flow overlay.")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip selection-box + stellar-locus reference lines.")
    args = ap.parse_args()

    # draw is a data-only diagnostic by default (no flow, no selection overlay).
    if args.source == "draw":
        args.no_flow = True
        args.no_overlay = True

    # Resolve the conditioning/weighting log_scale.
    log_scale = float(args.log_scale) if args.log_scale is not None \
        else _DEFAULT_LOG_SCALE[args.source]
    args._log_scale = log_scale
    print(f"Source={args.source}  log_scale={log_scale:.3f}")

    # ── Blue reference ────────────────────────────────────────────────────────
    obs_m, obs_w, blue_label = LOADERS[args.source](args)
    obs = to_coords(resample(obs_m, obs_w, args.n_plot, blue_label))
    print(f"  {len(obs):,} template points")

    # ── Flow overlay ──────────────────────────────────────────────────────────
    flow = None
    if not args.no_flow:
        prior_trained, raw2standard, key = load_flow(args, args.source)
        flow = sample_flow(prior_trained, raw2standard, key, args.n_plot, log_scale)
        n_common = min(len(obs), len(flow))
        rr = np.random.default_rng(1)
        obs = obs[rr.choice(len(obs), n_common, replace=False)]
        flow = flow[rr.choice(len(flow), n_common, replace=False)]
        print(f"  equalised both to {n_common:,} points")

    # ── Corner plot ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 16))
    corner.corner(obs, labels=LABELS, bins=500, plot_datapoints=True, plot_density=False,
                  plot_contours=True, fill_contours=False, hist_kwargs={"label": blue_label},
                  smooth=5.0, smooth1d=1.0, color="tab:blue", range=PLOT_RANGE,
                  label_kwargs={"fontsize": LABEL_FONTSIZE}, fig=fig)
    if flow is not None:
        corner.corner(flow, labels=LABELS, bins=500, plot_datapoints=False, plot_density=False,
                      plot_contours=True, fill_contours=False, hist_kwargs={"label": "Prior Flow"},
                      smooth=5.0, color="tab:orange", range=PLOT_RANGE,
                      label_kwargs={"fontsize": LABEL_FONTSIZE}, fig=fig)
    if not args.no_overlay:
        draw_overlay(fig)
        fig.axes[5].legend(bbox_to_anchor=(0.0, 1.0), loc="lower left", fontsize=16)
    for ax in fig.axes:
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)

    out = os.path.join(PLOTS_DIR,
                       f"corner_{args.source}_g0_ls{log_scale:.1f}_e0"
                       f"{'_dataonly' if args.no_flow else ''}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
