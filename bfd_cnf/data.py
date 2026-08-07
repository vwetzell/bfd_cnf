"""
data.py
=======
Data loading from the BFD FITS template table, moment extraction,
quality filtering, 2-D density histogram fitting in (log10 Mf, Mr/Mf) space,
importance weighting / resampling, and standardisation utilities.

The main entry point is ``load_training_dataset()`` (a thin wrapper over
``load_training_table()``), which runs the full pipeline and returns all key
data objects needed by the rest of the package.
"""

from __future__ import annotations

from typing import Any

import bfd
import fitsio
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr

from .config import (
    TRAIN_FITS_PATH,
    key as _initial_key,
    template_flux_min,
    template_flux_max,
)

# ---------------------------------------------------------------------------
# Moment-space helpers
# ---------------------------------------------------------------------------


def augment_moments_raw_jax(
    mu_raw: jax.Array,
    CM_raw: jax.Array,
    CA_raw: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the augmented noise covariance and BFD Jacobian base value.

    Parameters
    ----------
    mu_raw : jax.Array, shape (4,)
        Raw template moments ``[Mf, Mr, M1, M2]``.
    CM_raw : jax.Array, shape (4, 4)
        Measurement noise covariance in raw moment space.
    CA_raw : jax.Array, shape (4, 4)
        Augmentation noise covariance to be added to ``CM_raw``.

    Returns
    -------
    C_raw : jax.Array, shape (4, 4)
        Total noise covariance ``CM_raw + CA_raw``.
    trace_correction : jax.Array, scalar
        ``tr(B_RAW @ CA_raw)`` — the trace term needed for the augmented
        BFD Jacobian.
    J_bfd_base : jax.Array, scalar
        ``mu_raw^T B_RAW mu_raw`` — the BFD Jacobian evaluated at the
        template centroid (before augmentation).
    """
    from .config import B_RAW_JNP

    C_raw = CM_raw + CA_raw
    trace_correction = jnp.trace(B_RAW_JNP @ CA_raw)
    J_bfd_base = mu_raw @ B_RAW_JNP @ mu_raw
    return C_raw, trace_correction, J_bfd_base


def make_augmentation_noise_raw_jax(
    CM_raw: jax.Array,
    target_snr: float = 20.0,
    current_snr: float | None = None,
) -> jax.Array:
    """Build an additive noise covariance that degrades SNR to ``target_snr``.

    Parameters
    ----------
    CM_raw : jax.Array, shape (4, 4)
        Measurement noise covariance in raw moment space.
    target_snr : float, optional
        Desired flux SNR after augmentation.  Default is 20.
    current_snr : float or None, optional
        Current flux SNR of the template.  If ``None``, a fixed scale factor
        of 8 is used instead of computing it from the SNR ratio.

    Returns
    -------
    jax.Array, shape (4, 4)
        Augmentation noise covariance ``k * CM_raw`` where ``k`` is chosen so
        that the effective SNR equals ``target_snr``.
    """
    k = jnp.where(
        current_snr is None,
        8.0,
        jnp.maximum((current_snr / target_snr) ** 2 - 1.0, 0.0),
    )
    return k * CM_raw


def transform_dataset_to_standard(
    raw2standard_bijection: Any,
    data_y: jax.Array,
    data_Sigma: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Transform raw moments and their covariances into standardized coordinates.

    Applies ``raw2standard_bijection`` to ``data_y`` and propagates
    ``data_Sigma`` through the Jacobian of the transformation (including a
    second-order correction term).

    Parameters
    ----------
    raw2standard_bijection : RawMomentStandardize
        Bijection that maps raw moments ``[Mf, Mr, M1, M2]`` to standardized
        coordinates ``[(log10 Mf - mean)/std, ...]``.
    data_y : array-like, shape (N, 4)
        Raw template moments.
    data_Sigma : array-like, shape (N, 4, 4)
        Per-object raw-moment covariance matrices.

    Returns
    -------
    data_y_std : jax.Array, shape (N, 4)
        Moments in standardized coordinates.
    Sigma_std : jax.Array, shape (N, 4, 4)
        Propagated covariance matrices in standardized coordinates (first- plus
        second-order Jacobian correction).
    """
    data_y = jnp.asarray(data_y)
    data_Sigma = jnp.asarray(data_Sigma)

    transform_and_logdet = raw2standard_bijection.transform_and_log_det
    data_y_std = jax.vmap(transform_and_logdet)(data_y)[0]

    Mf = data_y[..., 0]
    Mr = data_y[..., 1]
    M1 = data_y[..., 2]
    M2 = data_y[..., 3]

    z = jnp.zeros_like(Mf)
    log10 = jnp.log(10.0)

    # Build J directly — one allocation of shape (N, 4, 4)
    J = jnp.stack(
        [
            jnp.stack([1.0 / (Mf * log10), z, z, z], axis=-1),
            jnp.stack([-Mr / Mf**2, 1.0 / Mf, z, z], axis=-1),
            jnp.stack([z, -M1 / Mr**2, 1.0 / Mr, z], axis=-1),
            jnp.stack([z, -M2 / Mr**2, z, 1.0 / Mr], axis=-1),
        ],
        axis=-2,
    )

    # Build H directly — one allocation of shape (N, 4, 4, 4)
    def row4(a, b, c, d):
        return jnp.stack([a, b, c, d], axis=-1)

    H0 = jnp.stack(
        [
            row4(-1.0 / (Mf**2 * log10), z, z, z),
            row4(z, z, z, z),
            row4(z, z, z, z),
            row4(z, z, z, z),
        ],
        axis=-2,
    )

    H1 = jnp.stack(
        [
            row4(2.0 * Mr / Mf**3, -1.0 / Mf**2, z, z),
            row4(-1.0 / Mf**2, z, z, z),
            row4(z, z, z, z),
            row4(z, z, z, z),
        ],
        axis=-2,
    )

    H2 = jnp.stack(
        [
            row4(z, z, z, z),
            row4(z, 2.0 * M1 / Mr**3, -1.0 / Mr**2, z),
            row4(z, -1.0 / Mr**2, z, z),
            row4(z, z, z, z),
        ],
        axis=-2,
    )

    H3 = jnp.stack(
        [
            row4(z, z, z, z),
            row4(z, 2.0 * M2 / Mr**3, z, -1.0 / Mr**2),
            row4(z, z, z, z),
            row4(z, -1.0 / Mr**2, z, z),
        ],
        axis=-2,
    )

    H = jnp.stack([H0, H1, H2, H3], axis=-3)

    Sigma1 = jnp.einsum("nij,njk,nlk->nil", J, data_Sigma, J)

    def second_order_term(Hn, Sig):
        tmp = jnp.einsum("iab,bc->iac", Hn, Sig)
        tmp = jnp.einsum("iac,jad->ijcd", tmp, Hn)
        return 0.5 * jnp.einsum("ijcd,dc->ij", tmp, Sig)

    Sigma2 = jax.vmap(second_order_term)(H, data_Sigma)
    Sigma_raw = Sigma1 + Sigma2

    std = raw2standard_bijection.std
    S = jnp.diag(1.0 / std)
    Sigma_std = jnp.einsum("ij,njk,kl->nil", S, Sigma_raw, S)

    return data_y_std, Sigma_std




# ---------------------------------------------------------------------------
# Quality cuts (shared between loaders)
# ---------------------------------------------------------------------------


def quality_cut_mask(moments: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Boolean mask of templates passing the moment + covariance quality cuts.

    Mirrors the moment- and covariance-level cuts applied inside
    :func:`load_training_table` so that auxiliary loaders (e.g. the chunked
    standardiser rebuild in ``backfill_stats``) select the same template
    population.  Operates on raw moments ``[Mf, Mr, M1, M2]`` and
    their 4×4 covariance matrices, in numpy on the CPU.

    Parameters
    ----------
    moments : np.ndarray, shape (N, 4)
        Raw template moments ``[Mf, Mr, M1, M2]``.
    cov : np.ndarray, shape (N, 4, 4)
        Per-object raw-moment covariance matrices.

    Returns
    -------
    np.ndarray, shape (N,)
        Boolean mask, ``True`` where the template passes every cut.
    """
    good_moments = moments[:, 0] / np.sqrt(cov[:, 0, 0]) > 5.0
    good_moments &= moments[:, 0] > template_flux_min
    good_moments &= moments[:, 0] < template_flux_max
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


def _finalize_dataset(
    moments_jnp: jax.Array,
    centroid_moments_jnp: jax.Array,
    cov_jnp: jax.Array,
    dm_dg_jnp: jax.Array,
    d2m_dg2_jnp: jax.Array,
    nda_jnp: jax.Array,
    key: jax.Array,
) -> dict[str, Any]:
    """Shared tail of the data loaders: flat-coverage importance weighting,
    resampling, and the standardisation bijection.

    Takes the already-filtered (quality- and derivative-cut) training arrays —
    identical in meaning regardless of which template file they came from — and
    builds the dict returned by :func:`load_training_table`.
    """
    # Import here to avoid circular imports
    from .models.bijections import RawMomentStandardize

    # -------------------------------------------------------------------
    # Initial resampled array (uniform bootstrap, for visualization / IS resampling)
    # -------------------------------------------------------------------
    key, subkey = jr.split(key)
    resampled_idx = jr.choice(
        subkey,
        np.arange(moments_jnp.shape[0]),
        shape=(moments_jnp.shape[0],),
        replace=True,
    )

    resampled = np.array(
        [
            jnp.log10(moments_jnp[:, 0]),
            (moments_jnp[:, 1] / moments_jnp[:, 0]),
            moments_jnp[:, 2] / moments_jnp[:, 1],
            moments_jnp[:, 3] / moments_jnp[:, 1],
        ]
    ).T[resampled_idx]

    # -------------------------------------------------------------------
    # Batch-sampling proposal = the BFD per-copy prior weight  nda·detj.
    # -------------------------------------------------------------------
    # The BFD template contribution is nda·detj·kernel: nda=sky_density·da is the copy's
    # shift-grid weight (momentcalc.py:556; the FITS column is already the Horvitz-Thompson
    # weight nda_orig/p for our subsample, dev/subsample.py), and detj=¼(Mr²−M1²−M2²) is the
    # |dX/dx| Jacobian carried in BFD's likelihood kernel (probabilities_jax.py:200,
    # detj_target).  Our flow inference kernel has NO detj, so it must live in the prior:
    # per-copy weight = nda·detj·L(X|C_X), giving Σ_copies ∝ base per galaxy (without detj the
    # prior collapses ∝ base/detj ∝ 1/Mf², starving high flux).  Sampling ∝ nda·detj makes the
    # proposal match the objective's static part ⇒ residual per-copy weight is L alone
    # (see make_nll_loss / make_elbo_loss).  No 1/p̂ flattening, no clip.
    detj = jnp.clip(
        0.25 * (moments_jnp[:, 1] ** 2 - moments_jnp[:, 2] ** 2 - moments_jnp[:, 3] ** 2),
        0.0, None,
    )
    w = jnp.asarray(nda_jnp) * detj
    weights = w / jnp.maximum(jnp.mean(w), jnp.finfo(jnp.float32).tiny)

    # Population bounds kept only as optional viz overlay hints (no longer gate training).
    x_lo, x_hi = float(jnp.log10(template_flux_min)), float(jnp.log10(500000))  # log10 Mf
    y_lo, y_hi = 0.5, 9.0  # Mr/Mf

    # -------------------------------------------------------------------
    # Standardisation bijection
    # -------------------------------------------------------------------
    moments_transformed = jnp.array(
        [
            jnp.log10(moments_jnp[:, 0]),
            moments_jnp[:, 1] / moments_jnp[:, 0],
            moments_jnp[:, 2] / moments_jnp[:, 1],
            moments_jnp[:, 3] / moments_jnp[:, 1],
        ]
    ).T

    data_mean = jnp.mean(moments_transformed, axis=0)
    data_mean = data_mean.at[2:].set(0.0)  # force mean M1 and M2 to be zero
    data_std = jnp.std(moments_transformed, axis=0)
    data_std = data_std.at[2:].set(jnp.mean(data_std[2:]))
    # Create transform with whitening
    raw2standard = RawMomentStandardize(mean=data_mean, std=data_std)

    return dict(
        moments_jnp=moments_jnp,
        centroid_moments_jnp=centroid_moments_jnp,
        cov_jnp=cov_jnp,
        dm_dg_jnp=dm_dg_jnp,
        d2m_dg2_jnp=d2m_dg2_jnp,
        nda=nda_jnp,
        weights=weights,
        resampled=resampled,
        x_lo=x_lo,
        x_hi=x_hi,
        y_lo=y_lo,
        y_hi=y_hi,
        data_mean=data_mean,
        data_std=data_std,
        raw2standard=raw2standard,
        key=key,
    )


def load_training_table(
    fits_path: str = TRAIN_FITS_PATH,
    key: jax.Array | None = None,
    subsample: int = 1,
) -> dict[str, Any]:
    """Load the pre-joined training table and run the full pipeline.

    The file stores the RAW BFD layout, so cov is bulkUnpacked and the
    derivative vectors are sliced/reshaped to trainer-native shapes:

        moments  (N, 5)      → keep [:, :4] = [Mf, Mr, M1, M2]
        cov      (N, 15)     packed → bulkUnpack → (N, 4, 4)
        dm_dg    (N, 2, 7)   [shear, moment] → (N, 4, 2)
        d2m_dg2  (N, 3, 7)   [g1g1,g1g2,g2g2 × moment] → (N, 4, 2, 2)
        centroid (N, 2)      [MX, MY]
        nda      (N,)

    The file is ~34 GB, so it is read in chunks and quality-cut per chunk (only
    survivors are kept in memory) rather than column-at-once.

    Applies the standard quality- and derivative-cuts and returns the dataset
    dict assembled by :func:`_finalize_dataset` (see it for the key descriptions).

    Parameters
    ----------
    fits_path : str, optional
        Path to the joined training table.  Defaults to ``config.TRAIN_FITS_PATH``.
    key : jax.Array or None, optional
        JAX PRNG key.  If ``None``, ``config.key`` is used.
    subsample : int, optional
        Keep roughly ``1/subsample`` of the rows at random before any cuts.
    """
    if key is None:
        key = _initial_key

    # The training table is ~34 GB / 72M rows in the RAW BFD layout: moments (N,5),
    # PACKED cov (N,15), dm_dg (N,2,7)=[shear,moment], d2m_dg2 (N,3,7)=[g1g1,g1g2,
    # g2g2 × moment], centroid (N,2), nda (N,).  A whole-column read both OOMs and
    # overflows fitsio's single huge-read path, so stream in contiguous chunks:
    # per chunk unpack cov, reshape derivs to trainer-native, quality-cut, and keep
    # ONLY survivors.  Peak memory is one chunk + the (much smaller) survivor pile.
    sub_rng = np.random.default_rng(int(jr.randint(key, (), 0, 2**31 - 1)))
    _CHUNK = 5_000_000
    m_l, x_l, cov_l, dm_l, d2_l, nda_l = [], [], [], [], [], []
    fits = fitsio.FITS(fits_path)
    try:
        h = fits[1]
        # HT-importance subsample (dev/subsample.py) stores nda as the Horvitz-Thompson
        # weight nda/p; its heavy tail IS the correction, so it must NOT be nda-clipped.
        # Surfaced as nda_is_ht so trainers can refuse to clip it (see converge_train).
        nda_is_ht = str(h.read_header().get("SUBSAMP", "")).strip().upper().startswith("HT")
        n_total = h.get_nrows()
        for s in range(0, n_total, _CHUNK):
            e = min(s + _CHUNK, n_total)
            blk = h.read(rows=np.arange(s, e),
                         columns=["moments", "cov", "dm_dg", "d2m_dg2",
                                  "centroid", "nda"])
            if subsample > 1:
                blk = blk[sub_rng.random(len(blk)) < 1.0 / subsample]
            mom = blk["moments"][:, :4].astype(np.float64)               # [Mf,Mr,M1,M2]
            cov = bfd.MomentCovariance.bulkUnpack(blk["cov"])[:, :4, :4].astype(np.float64)
            keep = quality_cut_mask(mom, cov)
            if not keep.any():
                continue
            raw_dm = blk["dm_dg"][keep].astype(np.float64)              # (n,2,7)
            raw_d2 = blk["d2m_dg2"][keep].astype(np.float64)           # (n,3,7)
            n_keep = int(keep.sum())
            # Keep even moments [Mf,Mr,M1,M2] AND the odd centroid pair [MX,MY]
            # (raw indices 5,6) so the loss can shear the centroid weight L(X|C_X)
            # with g — the BFD centroid shear response (probabilities_jax.py:197,238
            # use all 7 moments).  Drop Mc (index 4).  Layout: (n,6,2) = even4+odd2.
            sel = [0, 1, 2, 3, 5, 6]
            dm = np.stack([raw_dm[:, 0, sel], raw_dm[:, 1, sel]], axis=-1)  # (n,6,2)
            d2 = np.empty((n_keep, 6, 2, 2), np.float64)
            d2[..., 0, 0] = raw_d2[:, 0, sel]
            cross = raw_d2[:, 1, sel]
            d2[..., 0, 1] = cross
            d2[..., 1, 0] = cross
            d2[..., 1, 1] = raw_d2[:, 2, sel]
            m_l.append(mom[keep])
            cov_l.append(cov[keep])
            x_l.append(blk["centroid"][keep].astype(np.float64))         # (n,2)
            dm_l.append(dm)
            d2_l.append(d2)
            nda_l.append(blk["nda"][keep].astype(np.float64))            # (n,)
    finally:
        fits.close()

    moments = np.concatenate(m_l)
    cov = np.concatenate(cov_l)
    centroid = np.concatenate(x_l)
    dm_dg = np.concatenate(dm_l)
    d2m_dg2 = np.concatenate(d2_l)
    nda = np.concatenate(nda_l)

    # -------------------------------------------------------------------
    # Quality cuts on derivatives
    # -------------------------------------------------------------------
    good_dm_dg = (dm_dg >= np.percentile(dm_dg, 0.01, axis=0)) & (
        dm_dg <= np.percentile(dm_dg, 99.99, axis=0)
    )
    good_d2m_dg2 = (d2m_dg2 >= np.percentile(d2m_dg2, 0.01, axis=0)) & (
        d2m_dg2 <= np.percentile(d2m_dg2, 99.99, axis=0)
    )
    good_derivs = np.all(good_dm_dg, axis=(1, 2)) & np.all(good_d2m_dg2, axis=(1, 2, 3))

    moments = moments[good_derivs]
    centroid = centroid[good_derivs]
    cov = cov[good_derivs]
    dm_dg = dm_dg[good_derivs]
    d2m_dg2 = d2m_dg2[good_derivs]
    nda = nda[good_derivs]
    print(f"load_training_table: {moments.shape[0]} templates after quality + derivative cuts")

    # -------------------------------------------------------------------
    # Single GPU transfer — after all filtering is done
    # -------------------------------------------------------------------
    moments_jnp = jnp.array(moments)
    centroid_moments_jnp = jnp.array(centroid)
    cov_jnp = jnp.array(cov)
    nda_jnp = jnp.array(nda)
    dm_dg_jnp = jnp.array(dm_dg)
    d2m_dg2_jnp = jnp.array(d2m_dg2)

    ds = _finalize_dataset(
        moments_jnp,
        centroid_moments_jnp,
        cov_jnp,
        dm_dg_jnp,
        d2m_dg2_jnp,
        nda_jnp,
        key,
    )
    ds["nda_is_ht"] = nda_is_ht
    return ds


def load_training_dataset(
    key: jax.Array | None = None,
    subsample: int = 1,
) -> dict[str, Any]:
    """Load the training template set (``TRAIN_FITS_PATH`` via :func:`load_training_table`).

    Thin wrapper kept as the stable entry point for the training scripts
    (``train_from_scratch``, ``converge_train``, ``pretrain_prior``, ``run``).
    """
    print(f"Training set: {TRAIN_FITS_PATH}")
    return load_training_table(key=key, subsample=subsample)
