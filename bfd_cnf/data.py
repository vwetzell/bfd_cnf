"""
data.py
=======
Data loading from the BFD FITS template table, moment extraction,
quality filtering, 2-D density histogram fitting in (log10 Mf, Mr/Mf) space,
importance weighting / resampling, and standardisation utilities.

The main entry point is ``load_data()``, which runs the full pipeline and
returns all key data objects needed by the rest of the package.
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
    FITS_PATH,
    SUMMARY_FITS_PATH,
    key as _initial_key,
    target_flux_min,
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
    :func:`load_data` so that auxiliary loaders (e.g.
    :func:`load_summary_moments`, used for diagnostic corner plots) select the
    same template population.  Operates on raw moments ``[Mf, Mr, M1, M2]`` and
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
    good_moments &= moments[:, 0] > target_flux_min
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


def load_summary_moments(fits_path: str = SUMMARY_FITS_PATH) -> np.ndarray:
    """Load quality-cut raw moments from the deep-field summary-template table.

    Reads ``summary_templates_new.fits`` — the 1.37M-galaxy deep-field template
    library — and returns the raw moments ``[Mf, Mr, M1, M2]`` of the templates
    that pass :func:`quality_cut_mask`.  Unlike ``tmpl_t04_joined.fits``, this
    catalogue holds one row per template galaxy with no sub-pixel-shifted
    copies, so it is the appropriate source for diagnostic corner plots of the
    underlying moment distribution.

    Parameters
    ----------
    fits_path : str, optional
        Path to the summary-template FITS table.  Defaults to
        ``config.SUMMARY_FITS_PATH``.

    Returns
    -------
    np.ndarray, shape (N, 4)
        Quality-cut raw template moments ``[Mf, Mr, M1, M2]`` (float64).
    """
    with fitsio.FITS(fits_path) as fits:
        h = fits[1]
        moments = h.read_column("moments")[:, :4].astype(np.float64)
        cov = bfd.MomentCovariance.bulkUnpack(h.read_column("covariance"))[:, :4, :4]

    return moments[quality_cut_mask(moments, cov)]


# ---------------------------------------------------------------------------
# Main data-loading pipeline
# ---------------------------------------------------------------------------


def load_data(
    fits_path: str = FITS_PATH,
    key: jax.Array | None = None,
    subsample: int = 1,
) -> dict[str, Any]:
    """Load the BFD template table and run the full data-preparation pipeline.

    Reads the FITS file, applies quality cuts on moments and derivatives,
    fits a 2-D quadratic log-density in (log10 Mf, Mr/Mf) space, computes
    importance weights for approximately flat coverage inside the selection
    box, performs importance-weighted resampling, and builds the
    :class:`~bfd_cnf.models.bijections.RawMomentStandardize` bijection.

    Parameters
    ----------
    fits_path : str, optional
        Path to the BFD FITS template table.  Defaults to ``config.FITS_PATH``.
    key : jax.Array or None, optional
        JAX PRNG key.  If ``None``, ``config.key`` is used.
    subsample : int, optional
        Keep roughly ``1/subsample`` of the templates chosen at random before
        any quality cuts.  ``1`` (default) keeps all templates.

    Returns
    -------
    dict
        Dictionary with the following keys:

        moments_jnp : jax.Array, shape (N, 4)
            Filtered raw template moments ``[Mf, Mr, M1, M2]``.
        centroid_moments_jnp : jax.Array, shape (N, 2)
            True 1st-order centroid ("odd") moments ``[MX, MY]`` (cols 5,6 of the
            on-disk 7-vector ``[M0, MR, M1, M2, MC, MX, MY]``) for each template
            after quality filtering.  Used as ``data_X`` in the ELBO loss to compute
            the per-object centroid likelihood weight ``log N(X_G; 0, C_X)``.
            (Previously this erroneously held the 2nd-order ellipticity moments
            ``moments_jnp[:, 2:]``.)
        cov_jnp : jax.Array, shape (N, 4, 4)
            Per-object moment covariance matrices.
        dm_dg_jnp : jax.Array, shape (N, 4, 2)
            First shear derivatives of the moments.
        d2m_dg2_jnp : jax.Array, shape (N, 4, 2, 2)
            Second shear derivatives of the moments.
        weights : jax.Array, shape (N,)
            Importance weights (sum to 1) for flat (log10 Mf, Mr/Mf) coverage.
        resampled : np.ndarray, shape (N, 4)
            Importance-resampled data in
            ``[log10 Mf, Mr/Mf, M1/Mr, M2/Mr]`` coordinates.
        x_lo, x_hi : float
            Lower / upper bounds of the log10 Mf selection box.
        y_lo, y_hi : float
            Lower / upper bounds of the Mr/Mf selection box.
        a0..a5 : float
            Coefficients of the fitted quadratic log-density
            ``log p ≈ a0 + a1 x + a2 y + a3 x² + a4 y² + a5 xy``.
        data_mean : jax.Array, shape (4,)
            Empirical mean of the transformed moments used for standardisation.
        data_std : jax.Array, shape (4,)
            Empirical std of the transformed moments used for standardisation.
        raw2standard : RawMomentStandardize
            Bijection from raw to standardised coordinates.
        log_p_hat : callable
            Fitted quadratic log-density function ``(x, y) -> float``.
        key : jax.Array
            Updated PRNG key.
    """
    # Import here to avoid circular imports
    from .models.bijections import RawMomentStandardize

    if key is None:
        key = _initial_key

    # -------------------------------------------------------------------
    # Memory-frugal column-by-column read.
    #
    # The table is ~18 GB (N≈44M rows) and most of its columns are unused.
    # Reading the whole table with fitsio.read() and then casting every used
    # column to float64 while the table is still alive peaks near ~38 GB.
    # Instead we open the HDU and pull only the six columns we need one at a
    # time, immediately reducing each to its final shape/dtype and freeing the
    # raw column before reading the next.  The covariance is unpacked in chunks
    # so the full (N, 5, 5) intermediate is never materialised.  This keeps the
    # peak near ~17 GB while producing byte-identical arrays.
    # -------------------------------------------------------------------
    import gc as _gc

    fits = fitsio.FITS(fits_path)
    try:
        h = fits[1]

        # moments: (N, 7) f4 on disk = [M0, MR, M1, M2, MC, MX, MY] (5 evens + 2 odds,
        # per bfd.moment.Moment).  Keep the first 4 evens [M0, MR, M1, M2] as the modelled
        # moment vector, and the LAST two [MX, MY] (cols 5,6) as the true 1st-order
        # centroid ("odd") moments used for the L(X|C_X) centroid weight.  NOTE: this is
        # NOT moments[:, 2:4] (= the 2nd-order ellipticity moments M1, M2) — that earlier
        # slice was a moment-order bug.
        moments_full = h.read_column("moments")
        moments = moments_full[:, :4].astype(np.float64)
        centroid_moments = moments_full[:, 5:7].astype(np.float64)  # [MX, MY]
        del moments_full
        N_rows = moments.shape[0]

        # covariance: packed (N, 15) f8 → unpack to (N, 4, 4) in chunks so the
        # full (N, 5, 5) (~8.75 GB) intermediate is never materialised.
        cov_pkgd = h.read_column("covariance")
        cov = np.empty((N_rows, 4, 4), dtype=np.float64)
        _CHUNK = 4_000_000
        for _s in range(0, N_rows, _CHUNK):
            _e = min(_s + _CHUNK, N_rows)
            cov[_s:_e] = bfd.MomentCovariance.bulkUnpack(cov_pkgd[_s:_e])[:, :4, :4]
        del cov_pkgd
        _gc.collect()

        # first shear derivatives → (N, 4, 2); read straight into the slots so
        # no per-column float64 temporary survives.
        dm_dg = np.empty((N_rows, 4, 2), dtype=np.float64)
        dm_dg[..., 0] = h.read_column("moments_dg1")[:, :4]
        dm_dg[..., 1] = h.read_column("moments_dg2")[:, :4]

        # second shear derivatives → (N, 4, 2, 2), symmetric in the last two axes.
        d2m_dg2 = np.empty((N_rows, 4, 2, 2), dtype=np.float64)
        d2m_dg2[..., 0, 0] = h.read_column("moments_dg1_dg1")[:, :4]
        _cross = h.read_column("moments_dg1_dg2")[:, :4]
        d2m_dg2[..., 0, 1] = _cross
        d2m_dg2[..., 1, 0] = _cross
        del _cross
        d2m_dg2[..., 1, 1] = h.read_column("moments_dg2_dg2")[:, :4]
    finally:
        fits.close()
    _gc.collect()

    # -------------------------------------------------------------------
    # Optional subsampling (before quality cuts to maximise memory savings)
    # -------------------------------------------------------------------
    if subsample > 1:
        key, subkey = jr.split(key)
        n_total = moments.shape[0]
        n_keep = max(1, n_total // subsample)
        sub_idx = np.sort(
            np.array(jr.choice(subkey, n_total, shape=(n_keep,), replace=False))
        )
        moments = moments[sub_idx]
        centroid_moments = centroid_moments[sub_idx]
        cov = cov[sub_idx]
        dm_dg = dm_dg[sub_idx]
        d2m_dg2 = d2m_dg2[sub_idx]

    # -------------------------------------------------------------------
    # Quality cuts on moments + covariance
    # -------------------------------------------------------------------
    keep = quality_cut_mask(moments, cov)
    moments = moments[keep]
    centroid_moments = centroid_moments[keep]
    cov = cov[keep]
    dm_dg = dm_dg[keep]
    d2m_dg2 = d2m_dg2[keep]

    # -------------------------------------------------------------------
    # Quality cuts on derivatives (in numpy — keep everything on CPU)
    # -------------------------------------------------------------------
    good_dm_dg = (dm_dg >= np.percentile(dm_dg, 0.01, axis=0)) & (
        dm_dg <= np.percentile(dm_dg, 99.99, axis=0)
    )
    good_d2m_dg2 = (d2m_dg2 >= np.percentile(d2m_dg2, 0.01, axis=0)) & (
        d2m_dg2 <= np.percentile(d2m_dg2, 99.99, axis=0)
    )
    good_derivs = np.all(good_dm_dg, axis=(1, 2)) & np.all(good_d2m_dg2, axis=(1, 2, 3))

    moments = moments[good_derivs]
    centroid_moments = centroid_moments[good_derivs]
    cov = cov[good_derivs]
    dm_dg = dm_dg[good_derivs]
    d2m_dg2 = d2m_dg2[good_derivs]

    # -------------------------------------------------------------------
    # Single GPU transfer — after all filtering is done
    # -------------------------------------------------------------------
    moments_jnp = jnp.array(moments)
    centroid_moments_jnp = jnp.array(centroid_moments)
    cov_jnp = jnp.array(cov)
    del moments, centroid_moments, cov
    dm_dg_jnp = jnp.array(dm_dg)
    del dm_dg
    d2m_dg2_jnp = jnp.array(d2m_dg2)
    del d2m_dg2

    # -------------------------------------------------------------------
    # Initial resampled array (uniform, for 2-D density fitting)
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
    # 2-D density histogram fitting in (log10 Mf, Mr/Mf) space
    # -------------------------------------------------------------------
    # --- Rectangle bounds ---
    x_lo, x_hi = jnp.log10(1000), jnp.log10(500000)  # log10 Mf
    y_lo, y_hi = 0.5, 9.0  # Mr/Mf

    x = resampled[:, 0]
    y = resampled[:, 1]

    mask = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    x_box = x[mask]
    y_box = y[mask]

    # --- Build 2D histogram of log-density inside the box ---
    n_bins = 40
    counts, x_edges, y_edges = np.histogram2d(x_box, y_box, bins=n_bins, density=True)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    XX, YY = np.meshgrid(x_centers, y_centers, indexing="ij")

    # Flatten and mask out empty bins
    flat_counts = counts.ravel()
    valid = flat_counts > 0
    log_p = np.log(flat_counts[valid])
    X_flat = XX.ravel()[valid]
    Y_flat = YY.ravel()[valid]

    # --- Design matrix for quadratic: [1, x, y, x^2, y^2, x*y] ---
    def design_matrix(x, y):
        return np.column_stack(
            [
                np.ones_like(x),  # a0 (intercept)
                x,  # a1
                y,  # a2
                x**2,  # a3
                y**2,  # a4
                x * y,  # a5  <-- correlation term
            ]
        )

    A = design_matrix(X_flat, Y_flat)

    # --- Weighted least squares (weight by counts for stability) ---
    W = np.diag(flat_counts[valid])
    coeffs, _, _, _ = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ log_p, rcond=None)
    a0, a1, a2, a3, a4, a5 = coeffs

    print(
        "Fitted log p(x,y) = "
        f"{a0:.3f} + {a1:.3f}x + {a2:.3f}y "
        f"+ {a3:.3f}x^2 + {a4:.3f}y^2 + {a5:.3f}xy"
    )

    # --- Evaluate fitted log-density at all sample points ---
    def log_p_hat(x, y):
        """Quadratic approximation to log p(x,y) inside the box."""
        return a0 + a1 * x + a2 * y + a3 * x**2 + a4 * y**2 + a5 * x * y

    # --- Weights = 1/p_hat (inside box only) ---
    log_w = -log_p_hat(x, y)
    log_w -= log_w[mask].max()  # numerical stability: normalize in log space
    w = np.exp(log_w)
    w[~mask] = 0.0  # zero weight outside box

    # --- Clip to control variance (99th percentile) ---
    clip_val = np.percentile(w[mask], 99)
    w = np.minimum(w, clip_val)
    w /= w.sum()

    # --- Diagnostics ---
    n_eff = 1.0 / np.sum(w[mask] ** 2) / mask.sum()
    print(f"ESS fraction inside box: {n_eff:.3f}")

    # -------------------------------------------------------------------
    # Importance-weighted resampling
    # -------------------------------------------------------------------
    key, subkey = jr.split(key)
    resampled_idx = jr.choice(
        subkey,
        np.arange(resampled.shape[0]),
        shape=(resampled.shape[0],),
        replace=True,
        p=w,
    )

    resampled = np.array(
        [
            resampled[:, 0],
            resampled[:, 1],
            resampled[:, 2],
            resampled[:, 3],
        ]
    ).T[resampled_idx]

    # -------------------------------------------------------------------
    # Re-compute weights for training (on the full moments_jnp array)
    # -------------------------------------------------------------------
    x_full = jnp.log10(moments_jnp[:, 0])
    y_full = moments_jnp[:, 1] / moments_jnp[:, 0]

    mask_full = (
        (x_full >= x_lo) & (x_full <= x_hi) & (y_full >= y_lo) & (y_full <= y_hi)
    )

    log_w_full = -log_p_hat(x_full, y_full)
    log_w_full -= log_w_full[mask_full].max()
    w_full = jnp.exp(log_w_full)
    w_full = w_full.at[~mask_full].set(0.0)

    clip_val_full = jnp.percentile(w_full[mask_full], 99)
    w_full = jnp.minimum(w_full, clip_val_full)
    w_full /= w_full.sum()

    weights = jnp.asarray(w_full)

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
        weights=weights,
        resampled=resampled,
        x_lo=x_lo,
        x_hi=x_hi,
        y_lo=y_lo,
        y_hi=y_hi,
        a0=a0,
        a1=a1,
        a2=a2,
        a3=a3,
        a4=a4,
        a5=a5,
        data_mean=data_mean,
        data_std=data_std,
        raw2standard=raw2standard,
        key=key,
        log_p_hat=log_p_hat,
    )
