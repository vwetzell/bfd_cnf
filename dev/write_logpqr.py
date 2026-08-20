"""Write a per-target logPQR column onto a copy of a target catalog.

Computes (log P, d logP/dg, d2 logP/dg2) at g = 0 for every row of a target
catalog -- noiseless (point evaluation) or noisy (the C_M integral, same as
`bias.py`) -- and packs it with `bfd.pqr.packPqr` into bfd's own linear PQR
layout (`bfd.pqr.PqrNoMu`, no magnification): [P, dg1, dg2, dg1dg1, dg1dg2,
dg2dg2].

THIS IS LOG PQR, NOT bfd's raw PQR.  bfd's own `PQR` column convention (see
`bfd/pqr.py`'s `sumPqr`) holds the *probability* P, not log P, and its
`logPqr()` converts raw -> log by dividing by p.  This project never forms
raw P -- log P runs O(-40) here, so exp(log P) is float32 noise, not a
number -- so what's written is already `logPqr`'s OUTPUT with no pre-image.
The column is named `logPQR` for exactly this reason: never feed it through
`bfd.pqr.logPqr()` again, and never assume it means what bfd's plain `PQR`
column means. `bfd.pqr.splitPqr`/`packPqr` themselves are representation-
agnostic (they just pack/unpack the linear <-> (p, q, r) form) and apply
unchanged either way.

Every input row is kept -- nothing is dropped, unlike `bias.main`. A row gets
`logPQR = NaN` (all 6 slots) if: its recentring didn't converge (IMGNOISE
catalogs' `badcenter`), or the computed value is non-finite, or (noisy only)
every draw was zero-weight. This script never streams: it builds the full
(n_targets, samples, 5) draw array in memory in one shot, like `bias.main`
does whenever `--samples <= --chunk`. That is the validated fix, not merely
the simple path -- this session found `pqr_streamed`'s chunk-merge silently
zeroes ~16% of targets on a population this project cares about (a float32
cancellation in the per-chunk Hessian reconstruction, worse at low per-chunk
ESS), while the unchunked path was clean to 0.02%. Pick `--batch` to fit
`--samples` x `--batch` draws in memory; there is no `--chunk` here on
purpose.

    python dev/write_logpqr.py --flow flows/shear.eqx \\
        --targets ../bfd_cnf_imsims/data/targets_g0_1M.fits \\
        --train-data ../bfd_cnf_imsims/data/moments.fits \\
        --out dev/targets_g0_1M_logpqr.fits
    python dev/write_logpqr.py --flow flows/centroid.eqx \\
        --targets ../bfd_cnf_imsims/data/targets_noisy_g0_200k.fits \\
        --train-data ../bfd_cnf_imsims/data/moments.fits \\
        --samples 8192 --out dev/targets_noisy_g0_200k_logpqr.fits
"""
import argparse
import sys

import equinox as eqx
import fitsio
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bfd.pqr                                        # noqa: E402
import bias                                            # noqa: E402
import bulk                                             # noqa: E402
import shear as shear_top                               # noqa: E402
from models.bijections import in_domain                 # noqa: E402

# FITS structural/table-layout keywords fitsio writes itself -- never forward
# these from the source header, or `fitsio.FITS.write` errors on the clash.
_STRUCTURAL_PREFIXES = ("XTENSION", "BITPIX", "NAXIS", "PCOUNT", "GCOUNT",
                        "TFIELDS", "TTYPE", "TFORM", "TDIM", "EXTNAME")


def _custom_header(path):
    h = fitsio.read_header(path, ext=1)
    return {k: h[k] for k in h.keys()
            if not any(k.startswith(p) for p in _STRUCTURAL_PREFIXES)}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flow", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--train-data", required=True,
                   help="catalog the flow was trained on -- sets the flow's "
                        "standardisation, exactly as bias.py's --train-data")
    p.add_argument("--centroid", action=argparse.BooleanOptionalAction, default=None,
                   help="defaults to the catalog's IMGNOISE header, as bias.py")
    p.add_argument("--samples", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--noise-seed", type=int, default=1)
    p.add_argument("--batch", type=int, default=None,
                   help="targets processed at once; batch x samples draws are "
                        "held at a time. Defaults to a 131072-draw budget over "
                        "samples, same convention as bias.py --batch-budget.")
    p.add_argument("--n-targets", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    img_noise = bool(fitsio.read_header(a.targets, ext=1).get("IMGNOISE", False))
    use_centroid = a.centroid if a.centroid is not None else img_noise
    if img_noise and a.noise_scale != 1.0:
        raise SystemExit(
            "--noise-scale on an IMGNOISE catalog would rescale the KERNEL "
            "without touching the noise already in the moments.")
    if img_noise and not a.samples:
        raise SystemExit(
            "these targets carry image noise, so P(M|g) is the prior "
            "CONVOLVED with C_M. Pass --samples > 0.")

    m_train_full = shear_top.load(a.train_data)[0]
    m_train = m_train_full if use_centroid else m_train_full[:int(0.9 * len(m_train_full))]
    flow = bulk.build_flow(jr.key(a.seed), m_train, shear=True, centroid=use_centroid)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    print(f"{a.flow} on {a.targets}"
          + (" (centroid layer on)" if use_centroid else ""))

    n = slice(None) if a.n_targets is None else slice(a.n_targets)
    rows = fitsio.read(a.targets)[n]
    m = np.asarray(rows["moments"], dtype=np.float64)
    N = len(m)

    keep = np.ones(N, dtype=bool)
    if img_noise:
        keep &= ~rows["badcenter"]
        print(f"  {int((~keep).sum())} rows with badcenter (recentring didn't "
              f"converge) -> NaN")

    sigma_x = np.asarray(rows["cov_odd"], dtype=np.float64) if use_centroid else None

    draws = log_wt = ok_raw = None
    batch = a.batch or max(1, 131_072 // max(a.samples, 1))
    if a.samples:
        cov = bias.load_cov(a.targets) * a.noise_scale ** 2
        if not img_noise:
            m = bias.add_noise(m, cov, a.noise_seed)
        draws, log_wt = bias.mixture_draws(
            flow, m, cov, a.samples, a.alpha, a.noise_seed + 1000, sigma_x=sigma_x)

    flow_g, layer = bias.split_centroid(flow)
    if layer is not None:
        # `ok` must be computed on the RAW draw and threaded through explicitly
        # -- `log_conv_is`'s own `in_domain` check is meaningless on peeled
        # (standardised) coordinates. Only do this when peeling actually
        # happens: `log_conv_is(ok=<given>)` substitutes `dummy =
        # zeros_like(draws[0])` for masked draws, which is a fine finite point
        # in standardised space but a chart SINGULARITY (Mf=0) in raw moment
        # space -- feeding it there reintroduced exactly the "computed -inf
        # differentiates to NaN" trap HANDOFF.md documents (0 * inf = NaN
        # survives `where`'s masking). Passing `ok=None` for the no-centroid
        # case keeps `log_conv_is`'s own safe_point(m_i) dummy instead.
        if draws is not None:
            ok_raw = in_domain(draws)
            draws, ld = bias.centroid_transform(layer, draws, sigma_x)
            log_wt = log_wt + ld
        m = bias.centroid_transform(layer, m, sigma_x)[0]

    p_val, q, r = bias.pqr_full(flow_g, m, draws, log_wt, batch=batch,
                                sigma_x=sigma_x, ok=ok_raw)

    dead = np.zeros(N, dtype=bool)
    if a.samples:
        # Should not occur -- this script never streams -- but cheap to check:
        # pqr_streamed's zero-weight targets are exact (0, 0, 0); a genuinely
        # computed value never lands there.
        dead = (p_val == 0.0) & (q == 0.0).all(1) & (r.reshape(N, -1) == 0.0).all(1)
        if dead.any():
            print(f"  {int(dead.sum())} rows with all-zero P,Q,R (unexpected "
                  f"in the unchunked path) -> NaN")

    bad = (~keep) | dead | ~np.isfinite(p_val) | ~np.isfinite(q).all(1) \
        | ~np.isfinite(r.reshape(N, -1)).all(1)
    print(f"  {int(bad.sum())} of {N} rows NaN-filled total")
    p_val = np.where(bad, np.nan, p_val)
    q = np.where(bad[:, None], np.nan, q)
    r = np.where(bad[:, None, None], np.nan, r)

    logpqr = bfd.pqr.packPqr(p_val, q, r)              # (N, 6) float64

    # Round-trip self-check on the finite rows.  packPqr keeps only ONE
    # off-diagonal entry of r (bfd.pqr.PqrNoMu.GETR picks r[0,1], not an
    # average of r[0,1]/r[1,0]), so this also symmetrises away the ~1e-5
    # float32 asymmetry between the Hessian's two independently-computed jvp
    # rows -- allclose, not exact equality, and only p/q (untouched by that)
    # are checked exactly.
    p2, q2, r2 = bfd.pqr.splitPqr(logpqr)
    fin = ~bad
    assert np.array_equal(p2[fin], p_val[fin])
    assert np.array_equal(q2[fin], q[fin])
    assert np.allclose(r2[fin], r[fin], rtol=1e-4, atol=1e-6)
    assert logpqr.shape == (N, bfd.pqr.PqrNoMu.NPQR)

    out_dtype = rows.dtype.descr + [("logPQR", "f8", (bfd.pqr.PqrNoMu.NPQR,))]
    out = np.empty(N, dtype=out_dtype)
    for name in rows.dtype.names:
        out[name] = rows[name]
    out["logPQR"] = logpqr

    header = _custom_header(a.targets)
    header.update({
        "PQRLOG": True,       # log P, d logP/dg, d2 logP/dg2 -- NOT bfd's raw PQR
        "PQRFLOW": a.flow,
        "PQRTRAIN": a.train_data,
        "PQRSAMP": a.samples,
        "PQRALPHA": a.alpha,
        "PQRNSCL": a.noise_scale,
        "PQRCEN": use_centroid,
    })
    with fitsio.FITS(a.out, "rw", clobber=True) as f:
        f.write(out, header=header)
    print(f"wrote {a.out}: {N} rows, logPQR packed as bfd.pqr.PqrNoMu "
          f"[P, dg1, dg2, dg1dg1, dg1dg2, dg2dg2] (log-space)")


if __name__ == "__main__":
    main()
