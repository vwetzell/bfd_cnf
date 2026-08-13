"""Control for the jit-hoist equivalence check: is the EAGER path reproducible?

The hoist comparison keeps reporting differences on draws that carry weight,
which would mean the jit changed the answer.  Before believing that, check the
null hypothesis: run the unchanged reference implementation twice in one
process and compare it to itself.  If ref-vs-ref moves by the same order as
ref-vs-new, the differences are XLA nondeterminism (autotuning picks kernels by
timing, so a contended or differently-warmed GPU can select different ones) and
say nothing about the hoist.
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import fitsio
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear                                         # noqa: E402
from dev.check_mixture_jit_equivalence import mixture_draws_reference  # noqa: E402

DATA = "../bfd_cnf_imsims/data"
SEED = 12345


def spread(w_a, w_b):
    """max |dlog_wt| over draws within 30 nats of the best, and over all."""
    lead = w_a - w_a.max(axis=1, keepdims=True)
    both_inf = ~np.isfinite(w_a) & ~np.isfinite(w_b)
    d = np.where(both_inf, 0.0, np.abs(w_a - w_b))
    return float(d[lead >= -30.0].max()), float(d.max())


def main():
    m_train = shear.load(f"{DATA}/moments.fits")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves("flows/centroid_deep.eqx", flow)

    rows = fitsio.read(f"{DATA}/targets_deep_g0_200k.fits")[:32]
    m = np.asarray(rows["moments"], dtype=np.float64)
    sigma_x = np.asarray(rows["cov_odd"], dtype=np.float64)
    cov = bias.load_cov(f"{DATA}/targets_deep_g0_200k.fits")

    for samples in (512, 8192):
        ref = lambda: mixture_draws_reference(flow, m, cov, samples, 0.5, SEED,
                                              sigma_x=sigma_x)
        new = lambda: bias.mixture_draws(flow, m, cov, samples, 0.5, SEED,
                                         sigma_x=sigma_x)
        d_a, w_a = ref()
        d_b, w_b = ref()          # SAME code, same seed, same process
        d_n, w_n = new()

        print(f"\nsamples = {samples}")
        print(f"  ref vs ref  draws bitwise equal: {np.array_equal(d_a, d_b)}"
              f"   max|dlog_wt| near/all = {spread(w_a, w_b)[0]:.3e} / "
              f"{spread(w_a, w_b)[1]:.3e}")
        print(f"  ref vs new  draws bitwise equal: {np.array_equal(d_a, d_n)}"
              f"   max|dlog_wt| near/all = {spread(w_a, w_n)[0]:.3e} / "
              f"{spread(w_a, w_n)[1]:.3e}")


if __name__ == "__main__":
    main()
