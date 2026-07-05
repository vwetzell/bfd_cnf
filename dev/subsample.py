"""Importance-subsample the massive template summary catalog to a transferable size.

Uniform random subsampling preserves the catalog's raw copy distribution, which has
*more* copies at large centroid shift [MX,MY] than near the centre (they tile a grid;
area grows with radius). But training weights every copy by nda * L(X|C_X) with
L = N([MX,MY]; 0, C_X) — so uniform sampling spends the row budget on far-shift copies
that L nearly zeros out, under-covering the near-centroid region that drives the loss.

NOTE: the keep-prob is p ∝ nda·detj·L(X|C_X*) — detj is a PROPOSAL factor, not a loss
weight.  detj = ¼(MR²−M1²−M2²) is the moment-vs-shift Jacobian: copies are born on a
uniform centroid-shift grid, so per unit MOMENT volume they pile up where detj is small
(the high-ellipticity tail).  detj in the keep-prob flattens that pile-up so the kept
COUNT distribution is faithful to moment space (keeps the unweighted standardizer and the
flow's learned density shape honest).  The loss still weights copies by nda·L only — NO
per-copy detj (HT removes the proposal in expectation; detj_target cancels in g, see
project_centroid_shear_response_fix).  Dropping detj here (2026-06-29) flooded the kept set
with high-|e| copies (ellipticity std 0.088→0.185) and drove grid m to −0.66 — restored.

Fix: keep each copy with probability p ∝ its training weight w = nda*L(X|C_X*), and store
nda -> nda/p (Horvitz-Thompson). This is unbiased for ANY p (the trained flow matches the
full file) because E[Σ_kept (nda/p)·f] = Σ_all nda·f for any per-copy f; p ∝ w just
minimises variance and concentrates rows where the weight is.

The summary FITS (dev/make_template_file.py) already carries `centroid`=[MX,MY] (moment
units, the X for L) and `nda`, plus moments/cov/dm_dg/d2m_dg2 — so this is standalone,
no HDF5 join, and the output keeps the same columns (directly loadable by data.py).

Streams the ~320 GB file in chunks; peak memory is one chunk. Run on the HPC node, then
transfer the output. Edit the constants below. `python dev/subsample.py --check` runs a
fast synthetic self-check of the unbiasedness instead of touching the big file.
"""
import sys

import numpy as np
import fitsio
import bfd

SRC = "/pscratch/sd/v/vwetzell/image_sims_grid_++_random_adjusted_density/templates_summary.fits"
OUT = "/pscratch/sd/v/vwetzell/image_sims_grid_++_random_adjusted_density/templates_train.fits"
KEEP_FRAC = 1.0 / 32     # target output size ≈ KEEP_FRAC * N (before the quality cut)
SIGMA_XY = 665.0         # keep-prob L width = WIDEST training C_X (σ=exp(log_scale_max/2),
                         # log_scale_max=13.0 → σ≈665).  The proposal is reused by the loss at every
                         # C_X in log_scale_range (10.5,13.0); making it ≥ the widest σ_t bounds the
                         # importance ratio L(σ_t)/L(σ*) to (σ*/σ_t)²≈12× at the narrow end instead of
                         # blowing up in the far-X tail at the wide end (~23% of draws have σ_t>500).
                         # Controls CENTROID coverage only; detj (per_copy_weight) handles the
                         # ellipticity tail independently.  σ=300 was known-good WITH detj (m=−0.039);
                         # 665 covers the wide-σ_t draws better.  Flip to 300 if 665 surprises.
P_MIN = 1e-6             # keep-prob floor: rescues the far tail the widest C_X (σ≈665) needs;
                         # caps nda inflation at 1/P_MIN. 0 = pure ∝w (drops the tail, biases wide C_X low)
CHUNK = 5_000_000        # rows per read
SEED = 18061998

# Inlined from bfd_cnf.data.quality_cut_mask: importing the package drags in jax
# (config/data import it at module top), and this is a pure-CPU numpy I/O job.
# Keep in sync with bfd_cnf/data.py if the cut changes.
template_flux_min = 800.0  # mirrors bfd_cnf/config.py


def quality_cut_mask(moments: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Boolean mask of templates passing the moment + covariance quality cuts.

    Raw moments ``[Mf, Mr, M1, M2]`` and their 4x4 covariances, numpy on CPU.
    """
    good_moments = moments[:, 0] / np.sqrt(cov[:, 0, 0]) > 5.0
    good_moments &= moments[:, 0] > template_flux_min
    good_moments &= moments[:, 1] > 0.0
    good_moments &= np.all(np.isfinite(moments), axis=1)
    good_moments &= (
        np.hypot(moments[:, 2] / moments[:, 1], moments[:, 3] / moments[:, 1]) < 0.99
    )

    m1mr = moments[:, 2] / moments[:, 1]
    m2mr = moments[:, 3] / moments[:, 1]

    m1mr_err = np.abs(m1mr) * np.sqrt(
        cov[:, 1, 1] / moments[:, 1] ** 2
        + cov[:, 2, 2] / moments[:, 2] ** 2
        - 2 * cov[:, 1, 2] / (moments[:, 1] * moments[:, 2])
    )
    m2mr_err = np.abs(m2mr) * np.sqrt(
        cov[:, 1, 1] / moments[:, 1] ** 2
        + cov[:, 3, 3] / moments[:, 3] ** 2
        - 2 * cov[:, 1, 3] / (moments[:, 1] * moments[:, 3])
    )

    mrmf = moments[:, 1] / moments[:, 0]
    mrmf_err = np.abs(mrmf) * np.sqrt(
        cov[:, 0, 0] / moments[:, 0] ** 2
        + cov[:, 1, 1] / moments[:, 1] ** 2
        - 2 * cov[:, 0, 1] / (moments[:, 0] * moments[:, 1])
    )

    good_cov = (np.hypot(m1mr, m2mr) + 1 * np.hypot(m1mr_err, m2mr_err)) < 0.95
    good_cov &= (mrmf - 5 * mrmf_err) < 3.976167
    good_cov &= (moments[:, 0]) > 1.0

    return good_moments & good_cov


def per_copy_weight(moments4: np.ndarray, centroid: np.ndarray, nda: np.ndarray,
                    sigma_xy: float = SIGMA_XY) -> np.ndarray:
    """Keep-prob weight w = nda * detj * L([MX,MY] | C_X*), C_X* = sigma_xy^2 I.

    detj = ¼(MR²−M1²−M2²) is the moment-vs-shift Jacobian (bfd.momentcalc:540).  It
    belongs in the PROPOSAL, not the loss: copies are born on a uniform centroid-shift
    grid, so their density per unit MOMENT volume ∝ 1/detj and piles up in the small-detj
    (high-ellipticity) tail; multiplying the keep-prob by detj flattens that so the kept
    COUNT distribution is faithful to moment space.  The loss still weights copies by nda·L
    only (HT correction nda→nda/p removes the proposal in expectation; detj_target cancels
    in g — project_centroid_shear_response_fix).  Negative detj (pre-quality-cut junk) is
    floored to 0 so those rows are never kept.
    """
    Mr, M1, M2 = moments4[:, 1], moments4[:, 2], moments4[:, 3]
    detj = np.maximum(0.25 * (Mr ** 2 - M1 ** 2 - M2 ** 2), 0.0)
    r2 = centroid[:, 0] ** 2 + centroid[:, 1] ** 2
    logL = -0.5 * r2 / sigma_xy ** 2 - np.log(2.0 * np.pi * sigma_xy ** 2)
    return nda * detj * np.exp(logL)


def subsample(src: str = SRC, out: str = OUT):
    fin = fitsio.FITS(src)
    h = fin[1]
    colnames = h.get_colnames()
    n_total = h.get_nrows()
    print(f"{n_total:,} rows; columns: {colnames}", flush=True)

    # Pass 1 (light): mean training weight over all rows -> sets the keep-prob scale.
    sum_w = 0.0
    for s in range(0, n_total, CHUNK):
        e = min(s + CHUNK, n_total)
        blk = h.read(rows=np.arange(s, e), columns=["moments", "centroid", "nda"])
        sum_w += float(per_copy_weight(
            blk["moments"][:, :4].astype(np.float64),
            blk["centroid"].astype(np.float64),
            blk["nda"].astype(np.float64),
        ).sum())
        print(f"  pass1 [{e:>12,}/{n_total:,}]", flush=True)
    mean_w = sum_w / n_total
    c = KEEP_FRAC / mean_w            # p_i = clip(c*w_i, P_MIN, 1)
    print(f"mean_w={mean_w:.4g}  scale c={c:.4g}", flush=True)

    # Pass 2 (full): quality cut, Bernoulli keep ∝ w, write kept rows with nda -> nda/p.
    rng = np.random.default_rng(SEED)
    hdr = [{"name": "SUBSAMP", "value": "HT-importance",
            "comment": "nda replaced by nda/p_inclusion (p ∝ nda*detj*L)"},
           {"name": "KEEPFRAC", "value": float(KEEP_FRAC), "comment": "target keep fraction"},
           {"name": "SIGMAXY", "value": float(SIGMA_XY), "comment": "central C_X sigma for L"}]
    fout = fitsio.FITS(out, "rw", clobber=True)
    written = False
    n_kept = 0
    try:
        for s in range(0, n_total, CHUNK):
            e = min(s + CHUNK, n_total)
            blk = h.read(rows=np.arange(s, e))
            mom = blk["moments"][:, :4].astype(np.float64)
            cov = bfd.MomentCovariance.bulkUnpack(blk["cov"])[:, :4, :4].astype(np.float64)
            w = per_copy_weight(mom, blk["centroid"].astype(np.float64),
                                blk["nda"].astype(np.float64))
            p = np.clip(c * w, P_MIN, 1.0)
            keep = quality_cut_mask(mom, cov) & (rng.random(len(blk)) < p)
            if not keep.any():
                print(f"  pass2 [{e:>12,}/{n_total:,}]  kept {n_kept:,}", flush=True)
                continue
            out_rec = blk[keep].copy()
            out_rec["nda"] = (blk["nda"][keep].astype(np.float64) / p[keep])  # Horvitz-Thompson
            if not written:
                fout.write(out_rec, header=hdr)
                written = True
            else:
                fout[-1].append(out_rec)
            n_kept += int(keep.sum())
            print(f"  pass2 [{e:>12,}/{n_total:,}]  kept {n_kept:,}", flush=True)
    finally:
        fin.close()
        fout.close()
    print(f"Done: wrote {n_kept:,} rows -> {out}")


def _check():
    """Synthetic self-check: the importance subsample preserves the L-weighted prior
    (the quantity the flow learns) for both a central and a wide C_X, unbiasedly."""
    rng = np.random.default_rng(0)
    N = 1_000_000
    nda = rng.uniform(0.5, 1.5, N)
    Mf = rng.uniform(2000.0, 50000.0, N)
    Mr = Mf * rng.uniform(0.3, 0.6, N)
    M1 = Mr * rng.uniform(-0.3, 0.3, N)
    M2 = Mr * rng.uniform(-0.3, 0.3, N)
    mom4 = np.stack([Mf, Mr, M1, M2], axis=1)
    centroid = rng.normal(0.0, 766.0, size=(N, 2))  # copies spread wider than σ_xy=400

    w = per_copy_weight(mom4, centroid, nda)
    p = np.clip((KEEP_FRAC / (w.sum() / N)) * w, P_MIN, 1.0)  # same rule as production
    keep = rng.random(N) < p
    nda_ht = nda[keep] / p[keep]

    # Check both ends of the training log_scale_range (σ=exp(log_scale/2)): the proposal
    # (σ=SIGMA_XY=widest) must keep BOTH unbiased.  Narrow σ=190 is where weights now run
    # up to (SIGMA_XY/σ)²≈12× — bounded, not the old ~19000× blow-up.
    r2 = centroid[:, 0] ** 2 + centroid[:, 1] ** 2
    for sigma, tol, label in [(190.0, 0.10, "narrow end"), (SIGMA_XY, 0.05, "wide end=σ_imp")]:
        L = np.exp(-0.5 * r2 / sigma ** 2) / (2 * np.pi * sigma ** 2)
        full = float((nda * L).sum())          # nda·L objective (no per-copy detj)
        sub = float((nda_ht * L[keep]).sum())
        ratio = sub / full
        print(f"  L-weighted prior ({label} σ={sigma}): sub/full = {ratio:.4f}")
        assert abs(ratio - 1.0) < tol, f"{label} biased: {ratio}"

    tot = float(nda_ht.sum()) / float(nda.sum())
    print(f"  total nda (HT identity, noisy): sub/full = {tot:.4f}")
    print(f"  kept {keep.sum():,}/{N:,};  mean |X| kept {np.sqrt(r2[keep]).mean():.0f} "
          f"vs all {np.sqrt(r2).mean():.0f}  (tail suppressed)")
    print("self-check OK")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    else:
        subsample()
