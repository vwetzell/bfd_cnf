"""Smoke test for the BFD nda (area/density) template weighting added to the flow loss.

Validates, on a small real slice of the template table (no 17 GB full read):
  1. all touched modules import (catches wiring/syntax errors);
  2. make_elbo_loss runs in BOTH branches (use_sx True/False) with nda;
  3. nda=None and nda=constant give IDENTICAL loss (the weighting is a clean no-op
     when uniform — i.e. the change is backward-compatible);
  4. nda=real (the FITS `weight` column) gives a finite loss that DIFFERS from the
     unweighted loss (the factor actually does something) and yields finite grads;
  5. the optional top-tail clip path runs.

Run:  python dev/smoke_nda_weight.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import fitsio
import bfd

from bfd_cnf.config import FITS_PATH
from bfd_cnf.data import quality_cut_mask
from bfd_cnf.models.bijections import RawMomentStandardize
from bfd_cnf.models.flows import build_flows, make_elbo_loss
from bfd_cnf.training import compute_std_stats

N_SLICE = 400_000          # rows read from disk (cheap vs the 44M full table)
BATCH = 256
NUM_SAMPLES = 2
N_SX = 2


def load_slice(n):
    """Read the first `n` rows and build the same arrays data.load_data would, incl. nda."""
    cols = [
        "moments", "covariance", "weight",
        "moments_dg1", "moments_dg2",
        "moments_dg1_dg1", "moments_dg1_dg2", "moments_dg2_dg2",
    ]
    d = fitsio.read(FITS_PATH, rows=np.arange(n), columns=cols, ext=1)

    moments = d["moments"][:, :4].astype(np.float64)
    centroid = d["moments"][:, 5:7].astype(np.float64)
    cov = bfd.MomentCovariance.bulkUnpack(d["covariance"])[:, :4, :4].astype(np.float64)
    nda = d["weight"].astype(np.float64)

    dm_dg = np.empty((n, 4, 2))
    dm_dg[..., 0] = d["moments_dg1"][:, :4]
    dm_dg[..., 1] = d["moments_dg2"][:, :4]
    d2 = np.empty((n, 4, 2, 2))
    d2[..., 0, 0] = d["moments_dg1_dg1"][:, :4]
    cross = d["moments_dg1_dg2"][:, :4]
    d2[..., 0, 1] = cross
    d2[..., 1, 0] = cross
    d2[..., 1, 1] = d["moments_dg2_dg2"][:, :4]

    # same filters data.load_data applies, carrying nda alongside (the bit we changed)
    keep = quality_cut_mask(moments, cov)
    moments, centroid, cov, dm_dg, d2, nda = (
        moments[keep], centroid[keep], cov[keep], dm_dg[keep], d2[keep], nda[keep]
    )
    gd = (dm_dg >= np.percentile(dm_dg, 0.01, axis=0)) & (dm_dg <= np.percentile(dm_dg, 99.99, axis=0))
    gd2 = (d2 >= np.percentile(d2, 0.01, axis=0)) & (d2 <= np.percentile(d2, 99.99, axis=0))
    good = np.all(gd, axis=(1, 2)) & np.all(gd2, axis=(1, 2, 3))
    moments, centroid, cov, dm_dg, d2, nda = (
        moments[good], centroid[good], cov[good], dm_dg[good], d2[good], nda[good]
    )

    assert nda.shape[0] == moments.shape[0], "nda lost alignment with moments!"
    assert np.all(np.isfinite(nda)) and np.all(nda > 0), "nda has non-finite/non-positive values!"
    print(f"  slice -> {moments.shape[0]} templates after cuts; "
          f"nda min/med/max = {nda.min():.3g}/{np.median(nda):.3g}/{nda.max():.3g}")
    return (
        jnp.asarray(moments), jnp.asarray(centroid), jnp.asarray(cov),
        jnp.asarray(dm_dg), jnp.asarray(d2), jnp.asarray(nda),
    )


def build_raw2standard(moments):
    mt = jnp.stack([
        jnp.log10(moments[:, 0]),
        moments[:, 1] / moments[:, 0],
        moments[:, 2] / moments[:, 1],
        moments[:, 3] / moments[:, 1],
    ], axis=1)
    mean = jnp.mean(mt, axis=0).at[2:].set(0.0)
    std = jnp.std(mt, axis=0)
    std = std.at[2:].set(jnp.mean(std[2:]))
    return RawMomentStandardize(mean=mean, std=std)


def make_loss(N, raw2standard, stats, *, nda, use_sx, clip=None):
    mld, sld, mo, so = stats
    return make_elbo_loss(
        N=N, batch_size=BATCH, num_samples=NUM_SAMPLES,
        weights=None, nda=nda, nda_clip_percentile=clip,
        log_scale_range=((10.5, 13.0) if use_sx else None),
        e_max=(0.05 if use_sx else 0.0), n_sx_train=N_SX, use_sx=use_sx,
        raw2standard=raw2standard, mean_log_diag=mld, std_log_diag=sld,
        mean_off=mo, std_off=so,
    )


def main():
    key = jax.random.key(0)
    moments, centroid, cov, dm_dg, d2, nda = load_slice(N_SLICE)
    N = moments.shape[0]
    raw2standard = build_raw2standard(moments)
    stats = compute_std_stats(raw2standard, moments, cov)

    key, ks = jax.random.split(key)
    model = build_flows(ks, latent_dim=4, cond_dim=16)
    data = (moments, cov, dm_dg, d2, centroid)
    ones = jnp.ones(N)

    eval_key = jax.random.key(42)  # SAME key across configs for fair comparison

    for use_sx in (True, False):
        tag = "Σ_X" if use_sx else "no-Σ_X"
        loss_none = float(make_loss(N, raw2standard, stats, nda=None, use_sx=use_sx)(model, *data, eval_key))
        loss_ones = float(make_loss(N, raw2standard, stats, nda=ones, use_sx=use_sx)(model, *data, eval_key))
        loss_real = float(make_loss(N, raw2standard, stats, nda=nda, use_sx=use_sx)(model, *data, eval_key))
        loss_clip = float(make_loss(N, raw2standard, stats, nda=nda, use_sx=use_sx, clip=99.9)(model, *data, eval_key))

        print(f"[{tag}] loss none={loss_none:.5f} ones={loss_ones:.5f} "
              f"real={loss_real:.5f} real+clip={loss_clip:.5f}")

        assert np.isfinite([loss_none, loss_ones, loss_real, loss_clip]).all(), f"{tag}: non-finite loss"
        # constant nda must be a perfect no-op
        assert abs(loss_ones - loss_none) < 1e-4, f"{tag}: constant nda changed the loss ({loss_ones} vs {loss_none})"
        # real nda must actually change the loss
        assert abs(loss_real - loss_none) > 1e-4, f"{tag}: real nda did NOT change the loss (factor inert?)"

        # gradients must be finite with the weighting on
        loss_fn = make_loss(N, raw2standard, stats, nda=nda, use_sx=use_sx, clip=99.9)
        lv, grads = eqx.filter_value_and_grad(lambda m: loss_fn(m, *data, eval_key))(model)
        leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_inexact_array(g)]
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves), f"{tag}: non-finite grad"
        print(f"[{tag}] grad step OK (finite), loss={float(lv):.5f}")

    print("\nSMOKE TEST PASSED ✅")


if __name__ == "__main__":
    main()
