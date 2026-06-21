"""
Corner plot of the template moment distribution vs. the trained prior flow.

One pipeline, four template sources (``--source``):

  summary  Deep-field summary templates (one row per galaxy, no shifted copies)
           from ``summary_templates_new.fits`` via ``load_summary_moments``.
           ``--weighted-joined`` instead uses the tmpl_t04_joined copies
           importance-resampled by nda × N(X;0,C_X) (the X-marginal the flow learned).
  new      The NEW template set (``templates_summary.fits`` + row-aligned
           ``templates.hdf5``), streamed in blocks, with the pipeline quality cuts
           and nda × N(X|C_X) [× detj] centroid weighting. ``--no-cut`` / ``--no-weight``.
  train    The huge ``templates_train.fits`` (~34 GB), streamed and detj × nda weighted.
  draw     1M templates drawn EXACTLY as training selects batches (``ds["weights"]``),
           optionally × detj × nda. Data-only by default (the distribution the loss sees).

Shared across sources: g=0, e=0 flow sampling at ``--log-scale``; the
[log10 Mf, Mr/Mf, M1/Mr, M2/Mr] coordinates; count equalisation; and the
selection-box (green) + stellar-locus (red) overlays.

Pure load — never trains. Examples::

    python -m bfd_cnf.plot_corner                       # summary vs canonical flow
    python -m bfd_cnf.plot_corner --source new --detj
    python -m bfd_cnf.plot_corner --source train --mf-cut 800
    python -m bfd_cnf.plot_corner --source draw --detj-nda-weight
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

# Per-source default Σ_X log_scale to condition the prior on.  None for "new" means
# "use the set's native scale 2·ln(sigmaXY) from the HDF5 metadata".
_DEFAULT_LOG_SCALE = {"summary": 13.27, "new": None, "train": 11.94, "draw": 11.94}


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
def load_summary(args):
    from bfd_cnf.data import load_data, load_summary_moments
    if args.weighted_joined:
        print("Building nda-weighted joined-template reference...")
        data = load_data()
        m = np.asarray(data["moments_jnp"], dtype=np.float64)
        X = np.asarray(data["centroid_moments_jnp"], dtype=np.float64)
        nda = np.asarray(data["nda"], dtype=np.float64)
        var_xy = float(np.exp(args._log_scale))
        log_w = np.log(np.maximum(nda, np.finfo(np.float64).tiny)) \
            - 0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2) / var_xy
        log_w -= log_w.max()
        return m, np.exp(log_w), "nda×N(X|C_X) joined"
    print("Loading deep-field summary templates...")
    sm = np.asarray(load_summary_moments())
    print(f"  {sm.shape[0]:,} summary templates after quality cuts")
    return sm, None, "Observed Templates"


def load_new(args):
    import bfd
    import fitsio
    import h5py
    from bfd_cnf.data import quality_cut_mask
    need_hdf5 = not args.no_weight
    print(f"Reading {args.summary}" + (f" + {args.hdf5}" if need_hdf5 else "") + " (blocked)...")
    m_keep, x_keep, nda_keep = [], [], []
    n_read = n_kept = 0
    fsum = fitsio.FITS(args.summary)
    hf = h5py.File(args.hdf5, "r") if need_hdf5 else None
    try:
        hsum = fsum[1]
        N = hsum.get_nrows()
        ds = hf["templates"] if need_hdf5 else None
        if ds is not None and ds.shape[0] != N:
            raise ValueError(f"row-count mismatch: summary {N} vs hdf5 {ds.shape[0]}")
        pool = N if args.pool <= 0 else min(args.pool, N)
        block = pool // 24
        for s in np.linspace(0, N - block, 24).astype(np.int64):
            s = int(s)
            rows = np.arange(s, s + block)
            mom = hsum.read_column("moments", rows=rows)[:, :4].astype(np.float64)
            if args.no_cut:
                m = np.all(np.isfinite(mom), axis=1)
            else:
                cov = bfd.MomentCovariance.bulkUnpack(
                    hsum.read_column("covariance", rows=rows))[:, :4, :4]
                m = quality_cut_mask(mom, cov)
            if need_hdf5:
                rec = ds[s:s + block]
                if not np.array_equal(rec["id"], hsum.read_column("id", rows=rows)):
                    raise ValueError(f"files not row-aligned in block at row {s}")
                x_keep.append(rec["derivs"][:, 5:7, 0].astype(np.float64)[m])
                nda_keep.append(rec["nda"].astype(np.float64)[m])
            m_keep.append(mom[m])
            n_read += len(mom)
            n_kept += int(m.sum())
    finally:
        fsum.close()
        if hf is not None:
            hf.close()
    m_sub = np.concatenate(m_keep, axis=0)
    print(f"  cut survival: {n_kept:,}/{n_read:,} = {100.0 * n_kept / n_read:.1f}%")
    label = "New Templates (" + ("raw" if args.no_cut else "cut")
    if args.no_weight:
        return m_sub, None, label + ")"
    X = np.concatenate(x_keep, axis=0)
    nda = np.concatenate(nda_keep, axis=0)
    var_xy = float(np.exp(args._log_scale))
    log_w = np.log(np.maximum(nda, np.finfo(np.float64).tiny)) \
        - 0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2) / var_xy
    if args.detj:
        detj = 0.25 * (m_sub[:, 1] ** 2 - m_sub[:, 2] ** 2 - m_sub[:, 3] ** 2)
        log_w += np.log(np.maximum(detj, np.finfo(np.float64).tiny))
        label += ", nda·detj-wtd)"
    else:
        label += ", nda-wtd)"
    log_w[~np.isfinite(log_w)] = -np.inf
    log_w -= np.nanmax(log_w)
    return m_sub, np.exp(log_w), label


def load_train(args):
    import fitsio
    h = fitsio.FITS(args.file)[1]
    N = h.get_nrows()
    keep_frac = min(1.0, args.n_sub / (N * 0.92))  # ~92% pass mf>800
    rng = np.random.default_rng(0)
    m_keep, nda_keep, n_read = [], [], 0
    for s in range(0, N, args.chunk):
        rows = np.arange(s, min(s + args.chunk, N))
        blk = h.read(rows=rows, columns=["moments", "nda"])
        m, nda = blk["moments"][:, :4].astype(np.float64), blk["nda"].astype(np.float64)
        sel = (m[:, 0] > args.mf_cut) & np.all(np.isfinite(m), axis=1) & (nda > 0)
        sel &= rng.random(len(m)) < keep_frac
        m_keep.append(m[sel]); nda_keep.append(nda[sel])
        n_read += len(rows)
        print(f"  {n_read:,}/{N:,} read, {sum(len(x) for x in m_keep):,} kept", end="\r")
    print()
    m = np.concatenate(m_keep); nda = np.concatenate(nda_keep)
    detj = 0.25 * (m[:, 1] ** 2 - m[:, 2] ** 2 - m[:, 3] ** 2)
    good = detj > 0
    return m[good], nda[good] * detj[good], f"Templates (Mf>{args.mf_cut:g}, detj×nda)"


def load_draw(args):
    from bfd_cnf.data import load_training_dataset
    ds = load_training_dataset()
    m = np.asarray(ds["moments_jnp"], dtype=np.float64)
    w = np.asarray(ds["weights"], dtype=np.float64)  # training selection distribution
    if args.detj_nda_weight:
        detj = np.clip(0.25 * (m[:, 1] ** 2 - m[:, 2] ** 2 - m[:, 3] ** 2), 0.0, None)
        w = w * detj * np.asarray(ds["nda"], dtype=np.float64)
        return m, w, "training draw × detj×nda"
    return m, w, "training draw"


LOADERS = {"summary": load_summary, "new": load_new, "train": load_train, "draw": load_draw}


def load_flow(args, source):
    """Load prior flow + raw2standard (pure load; never trains)."""
    import jax.numpy as jnp
    import jax.random as jr
    from bfd_cnf.models.flows import build_flows
    from bfd_cnf.training import load_models
    prior_path, q_path = args.prior or PRIOR_FLOW_PATH, args.q or Q_FLOW_PATH
    if not (os.path.exists(prior_path) and os.path.exists(q_path)):
        raise FileNotFoundError(f"Trained flow weights not found:\n  {prior_path}\n  {q_path}")
    if args.stats:
        from bfd_cnf.models.bijections import RawMomentStandardize
        s = np.load(args.stats)
        raw2standard = RawMomentStandardize(mean=jnp.asarray(s["mean"]), std=jnp.asarray(s["std"]))
        key = jr.key(0)
    else:
        from bfd_cnf.data import load_data
        data = load_data()
        key, raw2standard = data["key"], data["raw2standard"]
    prior_flow, q_flow = build_flows(key, latent_dim=4, cond_dim=16)
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
    ap.add_argument("--source", choices=list(LOADERS), default="summary")
    ap.add_argument("--log-scale", type=float, default=None,
                    help="Σ_X log_scale to condition the prior on / weight the X-marginal at "
                         "(default: per-source).")
    ap.add_argument("--n-plot", type=int, default=1_000_000, help="points per distribution")
    ap.add_argument("--prior", default=None, help="prior flow .eqx (default: canonical)")
    ap.add_argument("--q", default=None, help="q flow .eqx (default: canonical)")
    ap.add_argument("--stats", default=None,
                    help="raw2standard stats npz the flow was trained with (new-template flows); "
                         "default reads it via load_data (legacy set, canonical flow).")
    ap.add_argument("--no-flow", action="store_true", help="data cloud only, no flow overlay.")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip selection-box + stellar-locus reference lines.")
    # summary
    ap.add_argument("--weighted-joined", action="store_true",
                    help="[summary] nda×N(X|C_X)-resampled tmpl_t04_joined copies as the reference.")
    # new
    ap.add_argument("--summary", default="data/templates_summary.fits", help="[new] moments+cov FITS")
    ap.add_argument("--hdf5", default="data/templates.hdf5", help="[new] row-aligned derivs+nda HDF5")
    ap.add_argument("--pool", type=int, default=0, help="[new] rows to read (0 = full file)")
    ap.add_argument("--no-cut", action="store_true", help="[new] skip quality_cut_mask")
    ap.add_argument("--no-weight", action="store_true", help="[new] skip nda×N(X|C_X) weighting")
    ap.add_argument("--detj", action="store_true", help="[new] fold detj into the weight")
    # train
    ap.add_argument("--file", default="data/templates_train.fits", help="[train] templates_train.fits")
    ap.add_argument("--mf-cut", type=float, default=800.0, help="[train] keep Mf > this")
    ap.add_argument("--n-sub", type=int, default=3_000_000, help="[train] survivors before weighting")
    ap.add_argument("--chunk", type=int, default=5_000_000, help="[train] stream chunk size")
    # draw
    ap.add_argument("--detj-nda-weight", action="store_true",
                    help="[draw] weight drawn templates by detj×nda (the per-copy ELBO weight)")
    args = ap.parse_args()

    # draw is a data-only diagnostic by default (no flow, no selection overlay).
    if args.source == "draw":
        args.no_flow = True
        args.no_overlay = True

    # Resolve the conditioning/weighting log_scale.
    if args.log_scale is not None:
        log_scale = float(args.log_scale)
    elif args.source == "new":
        log_scale = _read_native_log_scale(args.hdf5)
    else:
        log_scale = _DEFAULT_LOG_SCALE[args.source]
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


def _read_native_log_scale(hdf5_path):
    """[new] 2·ln(sigmaXY) from the HDF5 column-meta blob, else 11.98 fallback."""
    import h5py
    import yaml
    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            meta_ds = f.get("templates.__table_column_meta__")
            if meta_ds is not None:
                blob = "\n".join(bytes(x).decode("latin1") for x in meta_ds[:])
                try:
                    for item in yaml.safe_load(blob).get("meta", []):
                        if isinstance(item, dict) and "sigmaXY" in item:
                            return 2.0 * float(np.log(item["sigmaXY"]))
                        if isinstance(item, (tuple, list)) and len(item) == 2 and item[0] == "sigmaXY":
                            return 2.0 * float(np.log(item[1]))
                except Exception:
                    pass
    return 11.98


if __name__ == "__main__":
    main()
