"""
dev/gauss2_aniso_cm_check.py
=============================
Item 6 of NEXT_SESSION_PSFE_FORMALISM_AUDIT.md: the paper's own Gauss-test
recipe (sec. 4.1, fully analytic, no rendering, no PSF at all), with an
anisotropic C_M injected DIRECTLY (`analytic.cov_M_aniso`) rather than via a
rendered elliptical PSF -- gauss2's galaxy has no PSF in it at all
(`analytic.moments` is the already-deconvolved limit), so this changes ONLY
C_M's shape and nothing about the galaxy population or its shear response.

The prior is `truth.BiasFlow` -- the EXACT analytic gauss2 density, not a
trained flow -- so there is no train/inference generalisation gap possible.
Items 1, 2 and 5 (NEXT_SESSION_PSFE_FORMALISM_AUDIT.md) narrowed the leak's
only channel to C_M's anisotropy and ruled out the IS/proposal machinery; if
THIS test is also a clean null, that pins the real leak entirely to the
trained flow's own density (the train/inference PSF gap), not the estimator.
If it leaks too, that is a major new finding: a real anisotropy bug survives
even with the exact prior and pure-kernel sampling.

Usage: python -m dev.gauss2_aniso_cm_check [--n 3000] [--samples 4096]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bfd_cnf_imsims"))

import truth                       # noqa: E402  (sets jax_enable_x64 on import)
import bias                        # noqa: E402
from imsims import analytic, sim   # noqa: E402

G = 0.02
PSF_E1 = 0.05   # matches the production PSFE grid's "psfe1p05" tag


def batch_theta(pop):
    flux, sigma, rho = pop["flux"], pop["sigma"], pop["bulge_ratio"]
    e1, e2 = pop["e1"], pop["e2"]
    logit_rho = np.log(rho) - np.log1p(-rho)
    return jnp.asarray(
        np.stack([np.log(flux), np.log(sigma), logit_rho, e1, e2], axis=-1))


def run_arm(m_true, cov, z, samples, seed, batch, chunk):
    """Noisy targets under `cov`, then exact-prior (Q, R) via pqr_streamed.

    `truth.BiasFlow` differentiates through a 30-step Newton solve PER DRAW
    (`analytic.theta_of_m`), unlike a trained flow's single forward pass --
    NEXT_SESSION.md documents an earlier attempt at this exact combination
    (bias.py --flow truth, noisy MC) being abandoned as impractically slow
    even at n=500 with --chunk 16 --batch-budget 64.  `batch`/`chunk` must
    stay small here for the same reason.
    """
    L = np.linalg.cholesky(np.asarray(cov, dtype=np.float64))
    m_noisy = jnp.asarray(np.asarray(m_true, dtype=np.float64) + z @ L.T)
    sigma_x = jnp.zeros((m_noisy.shape[0], 3))
    q, r = bias.pqr_streamed(truth.BiasFlow(), m_noisy, cov,
                             samples=samples, alpha=1.0, seed=seed,
                             sigma_x=sigma_x, batch=batch, chunk=chunk)
    return q, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    pop = sim.sample_population_gauss2_fwd(a.n, rng)
    theta = batch_theta(pop)

    m_plus = analytic.moments_batch(theta, g=(G, 0.0))
    m_minus = analytic.moments_batch(theta, g=(-G, 0.0))

    cov_iso = analytic.cov_M()
    cov_aniso = analytic.cov_M_aniso(psf_e1=PSF_E1, psf_e2=0.0)
    print(f"n={a.n} samples={a.samples} psf_e1={PSF_E1}")
    print(f"trace(cov_iso)={float(jnp.trace(cov_iso)):.4g}  "
          f"trace(cov_aniso)={float(jnp.trace(cov_aniso)):.4g}")

    # Common random numbers: same standard-normal draws scaled by each cov's
    # OWN Cholesky factor, so iso vs aniso differ only in C_M's shape, not in
    # which noise realization is used -- same pattern `bias.py` uses
    # throughout for paired comparisons.
    zp = rng.standard_normal((a.n, 5))
    zm = rng.standard_normal((a.n, 5))

    for label, cov in (("isotropic (control)", cov_iso),
                       ("ANISOTROPIC (psf_e1=%.2f)" % PSF_E1, cov_aniso)):
        qp, rp = run_arm(m_plus, cov, zp, a.samples, seed=1, batch=a.batch, chunk=a.chunk)
        qm, rm = run_arm(m_minus, cov, zm, a.samples, seed=2, batch=a.batch, chunk=a.chunk)
        m1, c1, c2 = bias.bias(qp, rp, qm, rm, g=G)
        em1, ec1, ec2 = bias.bootstrap(qp, rp, qm, rm, g=G, n=200, seed=3)
        print(f"{label:28s}  m1={float(m1):+.5f} +/- {em1:.5f}  "
              f"c1={float(c1):+.2e} +/- {ec1:.1e}  "
              f"c2={float(c2):+.2e} +/- {ec2:.1e}")


if __name__ == "__main__":
    main()
