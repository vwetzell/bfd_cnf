"""Thread 1, closure-plan Step 1: does the PSFE leak survive with NO flow at
all, using BFD's own exact template-sum machinery (paper eq. 35-36)?

Full re-render of `copies_gauss2_fwd_g2v3d.fits` (24.3M rows) at each psf_e
config was judged too expensive (imsims/copies.py has no --psf-e1/--psf-e2
flag, and each config would need its own ~3.5GB, multi-hour render). Instead
(user's call, 2026-09-15): reuse the ONE existing psf_e=0 copies pool for
every config's template moments/derivatives, and inject the PSF's real
anisotropy through the two channels BFD's own eq. 35-36 actually carries it
in, using each config's OWN already-rendered, already-measured catalog:

  - Sigma_X (the copy weight `N(X_c; 0, Sigma_X)` in `log_weights`) -- pulled
    from that config's own `cov_odd`, the REAL measured centroid-noise
    covariance under that config's real anisotropic PSF (confirmed
    genuinely anisotropic on disk: e.g. psfe2p10 has cov_odd off-diagonal
    ~2085 against ~49668 on the diagonal).
  - C_M (the whitening covariance for both templates and the observed
    target) -- pulled from that config's own `cov` column, likewise real
    and PSF-dependent.

This does NOT capture a PSF-driven shift in the deconvolved template
moments themselves (that would need the actual re-render) -- only the two
weighting/whitening channels. If the leak reproduces through THESE channels
alone, that is already decisive: it means the leak is baked into BFD's own
Sigma_X/C_M-driven likelihood, not into anything the trained flow learned
or failed to learn (channels 2-3 of the Thread 1 closure plan, not channel
4). If it does NOT reproduce, the moment-shift channel (needing the real
re-render) is implicated instead.

No window, no selection term -- this is a pure c1/c2 (additive) leak check,
and PSFE_PROVENANCE.md's own finding is that the leak is additive (m1 flat).
`bias.bias(qp, rp, qm, rm)` on the WHOLE catalog (no window) gives (m1, c1,
c2) directly from per-target Q, R.

Usage: python dev/psfe_no_flow_check.py [tag ...]
  default tags: psfe00 psfe1p02 psfe1p05 psfe1p10 psfe1m02 psfe1m05 psfe1m10
                psfe2p02 psfe2p05 psfe2p10
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "bfd_cnf_imsims"))

import fitsio
import jax
import jax.numpy as jnp
import numpy as np

import bias as B
from imsims.copies import log_jacobian
from band_tmpl import template_pqr

NOJAC = os.environ.get("NOJAC", "0") == "1"


def log_weights(copies, sigma_x):
    """Same as imsims.copies.log_weights, but with a NOJAC escape hatch: the
    paper's eq. 35/36 J(M) prefactor is evaluated ONCE at the target's own
    OBSERVED moment, OUTSIDE the u-sum (a per-target constant that trivially
    cancels from Q/P, R/P). `imsims.copies.log_weights` instead evaluates
    |J| PER COPY (`log_jacobian(copies["moments"])`, varying with u) and
    folds it INSIDE the sum -- structurally different from the paper's
    formula, since a per-copy factor does not trivially cancel the way a
    single per-target constant does. NOJAC=1 drops that term to test whether
    it is contributing to the PSFE leak."""
    c00, c01, c11 = np.moveaxis(np.asarray(sigma_x, dtype=float), -1, 0)
    det = c00 * c11 - c01 * c01
    x, y = copies["xy"][..., 0], copies["xy"][..., 1]
    chisq = (c11 * x * x - 2.0 * c01 * x * y + c00 * y * y) / det
    out = -0.5 * chisq - 0.5 * np.log(det) - np.log(2.0 * np.pi) \
        + np.log(copies["da"])
    if not NOJAC:
        out = out + log_jacobian(copies["moments"])
    return out

D = "../bfd_cnf_imsims/data"
COPIES = os.environ.get("COPIES", f"{D}/copies_gauss2_fwd_g2v3d.fits")
CACHE_DIR = os.environ.get("CACHE_DIR", "dev/_psfe_no_flow_cache")
WMIN = 1e-4
ESSMIN = 1000.0
SIZE, FLUX = (2.2, 3.2), (1500.0, 20000.0)  # Thread-1 validated-null window
TAGS = sys.argv[1:] or ["psfe00", "psfe1p02", "psfe1p05", "psfe1p10",
                        "psfe1m02", "psfe1m05", "psfe1m10",
                        "psfe2p02", "psfe2p05", "psfe2p10"]

_raw_copies_cache = None


def _raw_copies():
    global _raw_copies_cache
    if _raw_copies_cache is None:
        print(f"reading {COPIES} ...", flush=True)
        _raw_copies_cache = fitsio.read(COPIES, ext="COPIES")
    return _raw_copies_cache


def whitened_templates_for(cov, sx):
    """Same recipe as band_tmpl.whitened_templates, but `sx` (Sigma_X) is
    passed in per-config instead of read from a fixed catalog, and `cov`
    whitens against the CALLER's C_M -- no shared on-disk cache, since the
    result differs per config."""
    c = _raw_copies()
    lw = log_weights(c, sx)
    keep = lw > lw.max() + np.log(WMIN)
    a = np.linalg.inv(np.linalg.cholesky(cov))
    m = np.asarray(c["moments"][keep], np.float64) @ a.T
    q = np.asarray(c["dm_dg"][keep], np.float64) @ a.T
    r = np.asarray(c["d2m_dg2"][keep], np.float64) @ a.T
    f32 = lambda x: np.ascontiguousarray(x, np.float32)
    return dict(
        logw=f32(lw[keep] - lw.max()),
        m2=f32((m * m).sum(-1)), m=f32(m), q=f32(q), r=f32(r),
        mq=f32(np.einsum("ka,kia->ki", m, q)),
        mr=f32(np.einsum("ka,kia->ki", m, r)),
        qq=f32(np.stack([(q[:, 0] * q[:, 0]).sum(-1),
                         (q[:, 0] * q[:, 1]).sum(-1),
                         (q[:, 1] * q[:, 1]).sum(-1)], -1)),
    )


def run(tag):
    pop = f"gauss2_v3d_{tag}"
    cat = B.CATALOGS[pop]
    path = lambda v: f"{D}/{v}.fits"
    hdr = fitsio.read_header(path(cat["zero"]), ext=1)
    psf_e1, psf_e2 = float(hdr["PSFE1"]), float(hdr["PSFE2"])

    cov = B.load_cov(path(cat["plus"]))
    zero_row = fitsio.read(path(cat["zero"]), rows=[0])
    sx = np.asarray(zero_row["cov_odd"][0], np.float64)

    t = {k: jnp.asarray(v) for k, v in whitened_templates_for(cov, sx).items()}
    a = np.linalg.inv(np.linalg.cholesky(cov))

    out = {}
    for arm in ("plus", "minus"):
        obs = fitsio.read(path(cat[arm]))["moments"]
        q, r, ess = template_pqr(obs, t, a)
        out[arm] = (q, r, ess, obs)
    qp, rp, essp, obsp = out["plus"]
    qm, rm, essm, obsm = out["minus"]
    # ESSMIN (band_tmpl.py's own convention) + the Thread-1 validated-null
    # window -- without these the estimator is dominated by the SAME
    # catastrophic leader-draw instability Thread 2 fixed for the flow (raw
    # unwindowed/unguarded m1 here was ~-1.05, nowhere near the true ~0; the
    # windowed/ESS-guarded run below is the only one worth trusting).
    good = (np.isfinite(qp).all(1) & np.isfinite(rp).reshape(len(qp), -1).all(1)
            & np.isfinite(qm).all(1) & np.isfinite(rm).reshape(len(qm), -1).all(1)
            & (essp > ESSMIN) & (essm > ESSMIN)
            & B.window_mask(obsp, SIZE, FLUX) & B.window_mask(obsm, SIZE, FLUX))
    m1, c1, c2 = B.bias(qp[good], rp[good], qm[good], rm[good])
    print(f"{tag:>10s}  psf_e=({psf_e1:+.3f},{psf_e2:+.3f})  "
          f"n={good.sum()}/{len(good)}  med ESS {np.median(np.r_[essp, essm]):.0f}"
          f"  m1={m1:+.4f}  c1={c1:+.5f}  c2={c2:+.5f}", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(f"{CACHE_DIR}/{tag}.npz", qp=qp[good], rp=rp[good],
             qm=qm[good], rm=rm[good], idx=np.flatnonzero(good),
             psf_e1=psf_e1, psf_e2=psf_e2)
    return psf_e1, psf_e2, c1, c2


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    rows = [run(tag) for tag in TAGS]
    e1 = np.array([r[0] for r in rows])
    e2 = np.array([r[1] for r in rows])
    c1 = np.array([r[2] for r in rows])
    c2 = np.array([r[3] for r in rows])
    sl = lambda y, x: (y * x).sum() / (x * x).sum() if (x * x).sum() else np.nan
    print(f"\nunweighted linear-through-origin slope (no error bars, see "
          f"psfe_joint_slope.py for that): dc1/d(psf_e1) = {sl(c1, e1):+.4f}, "
          f"dc2/d(psf_e2) = {sl(c2, e2):+.4f}")
