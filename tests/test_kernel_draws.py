"""`kernel_draws` moved from numpy to `jax.random`; the stream changed, so the
only thing that can be asserted is the DISTRIBUTION plus the two structural
properties the estimator actually relies on.

Antithetic pairing is load-bearing (it cancels the leading error term where the
prior varies linearly across the kernel) and has silently broken before, so it
is checked exactly, not statistically.  Per-target independence is equally
load-bearing: sharing one offset set across targets would correlate their Monte
Carlo errors so they stop averaging down in the eq. (45)-(46) ensemble sums.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import bias as B                      # noqa: E402

COV = np.diag([1.2e2, 9.0e1, 4.0e1, 4.0e1, 2.5e3]).astype(np.float64)
COV[0, 1] = COV[1, 0] = 25.0          # keep it non-diagonal


def test_antithetic_pairs_are_exact():
    eps = np.asarray(B.kernel_draws(COV, 3, 512, seed=0))
    assert eps.shape == (3, 512, 5)
    half, mirror = eps[:, :256], eps[:, 256:]
    assert np.array_equal(half, -mirror), "eps and -eps must both appear"
    # so the sample mean over the draw axis is exactly zero, not just close
    assert np.abs(eps.sum(axis=1)).max() < 1e-3


def test_covariance_matches_C_M():
    eps = np.asarray(B.kernel_draws(COV, 1, 1 << 17, seed=1))[0]
    emp = np.cov(eps, rowvar=False)
    # float32 draws at 131k samples: a few 1e-3 relative is the sampling floor
    rel = np.abs(emp - COV) / np.sqrt(np.outer(np.diag(COV), np.diag(COV)))
    assert rel.max() < 0.02, rel.max()


def test_targets_are_independent():
    eps = np.asarray(B.kernel_draws(COV, 2, 4096, seed=2))
    a, b = eps[0, :, 0], eps[1, :, 0]
    r = np.corrcoef(a, b)[0, 1]
    assert abs(r) < 0.05, r
    assert not np.array_equal(eps[0], eps[1])


def test_seed_changes_the_draws():
    a = np.asarray(B.kernel_draws(COV, 1, 256, seed=3))
    b = np.asarray(B.kernel_draws(COV, 1, 256, seed=4))
    assert not np.allclose(a, b)
    assert np.array_equal(a, np.asarray(B.kernel_draws(COV, 1, 256, seed=3)))


def test_empty_kernel_is_allowed():
    # alpha <= 0 gives n_k = 0; the shape must survive so the concat downstream
    # still lines up.
    assert np.asarray(B.kernel_draws(COV, 4, 0, seed=5)).shape == (4, 0, 5)


if __name__ == "__main__":
    for fn in (test_antithetic_pairs_are_exact, test_covariance_matches_C_M,
               test_targets_are_independent, test_seed_changes_the_draws,
               test_empty_kernel_is_allowed):
        fn(); print("ok", fn.__name__)
