"""Smoke test for rqmc_pqr_grid's bruteforce quadrature path.

Checks that quadrature points drawn N(mu_raw, CM_raw) in RAW moment space and
pushed pointwise through the exact (nonlinear) raw2standard.transform -- rather
than drawn directly from a linearised Gaussian in standardised space -- still
produce a finite, positive P with a sane effective sample size. Run:
python tests/test_rqmc_raw_space.py
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bfd_cnf.models.flows import build_flows
from bfd_cnf.models.bijections import RawMomentStandardize
from bfd_cnf.data import transform_dataset_to_standard
from bfd_cnf.inference import rqmc_pqr_grid


def _toy_setup():
    prior, _ = build_flows(
        jr.PRNGKey(0), latent_dim=4, cond_dim=16, prior_size_loc_c1=0.0
    )
    raw2standard = RawMomentStandardize(
        mean=jnp.array([2.0, 2.5, 0.0, 0.0]), std=jnp.array([0.3, 0.3, 0.1, 0.1])
    )
    mu_raw = jnp.array([[100.0, 250.0, 5.0, -3.0]])
    CM_raw = jnp.array(
        [[[4.0, 0.0, 0.0, 0.0],
          [0.0, 9.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]]]
    )
    mu_std, cov_std = transform_dataset_to_standard(raw2standard, mu_raw, CM_raw)
    return prior, raw2standard, mu_std, cov_std, mu_raw, CM_raw


def test_bruteforce_raw_space_quadrature_runs_and_is_sane():
    prior, raw2standard, mu_std, cov_std, mu_raw, CM_raw = _toy_setup()

    out = rqmc_pqr_grid(
        mu_std, cov_std, mu_raw, CM_raw,
        n_points=2**10, n_replicates=8, batch_size=1,
        raw2standard=raw2standard, prior_flow=prior,
        bruteforce=True, augment=False, return_ess=True,
        key=jr.PRNGKey(123),
    )
    (P, P_se, Q1, Q1_se, Q2, Q2_se, R11, R11_se, R22, R22_se, R12, R12_se,
     ess, maxw, skew) = out

    assert np.all(np.isfinite(np.asarray(P))), "P must be finite"
    assert np.all(np.asarray(P) > 0), "P (a probability density integral) must be positive"
    assert np.all(np.isfinite(np.asarray(Q1))) and np.all(np.isfinite(np.asarray(Q2)))
    assert np.all(np.isfinite(np.asarray(R11)))
    # ESS should be a meaningful fraction of n_points for an un-augmented kernel draw
    # (weights are all 1: jac_corr=0 with no augmentation), not collapsed to ~0.
    assert float(np.asarray(ess)[0]) > 2**10 * 0.5


if __name__ == "__main__":
    test_bruteforce_raw_space_quadrature_runs_and_is_sane()
    print("ok: bruteforce raw-space quadrature runs and gives sane P/Q/R/ESS")
