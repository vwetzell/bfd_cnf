"""`_over_targets` pads its trailing chunk to one shape -- check it still
returns exactly the unpadded answer, and only for the real rows.

The padding exists to stop XLA compiling `one` a second time for the short
last chunk (measured at half of `pqr_streamed`'s compile cost).  It is only
safe because `vmap` is row-independent; the risk it carries is an off-by-one
in the pad index or the `[:k]` slice, which would silently return pad rows as
if they were galaxies.  That is what this asserts.
"""
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")           # run from the repo root, as the others do
import bias as B                  # noqa: E402


def _ref(one, arrays):
    """The obvious answer: apply `one` row by row, no batching at all."""
    n = len(arrays[0])
    rows = [one(*(None if a is None else jnp.asarray(a[i], jnp.float32)
                  for a in arrays)) for i in range(n)]
    return np.stack([np.asarray(r, np.float64) for r in rows])


def test_trailing_chunk_padded_matches_unbatched():
    # A row-dependent `one`, so any pad row leaking into the output moves it.
    one = lambda m_i: jnp.sum(m_i ** 2) + m_i[0]
    rng = np.random.default_rng(0)
    m = rng.normal(size=(47, 5))
    ref = _ref(one, (m,))
    for batch in (1, 5, 8, 16, 47, 64, 4096):   # divides, doesn't, exceeds n
        got = B._over_targets(one, (m,), batch)
        assert got.shape == (47,), (batch, got.shape)
        assert np.allclose(got, ref, rtol=0, atol=1e-5), batch


def test_batch_larger_than_catalog_does_not_inflate():
    """Regression: padding to `batch` rather than min(batch, n).

    `ess` calls `_over_targets` with n=1 and batch=20000, so padding the single
    row up to `batch` asked XLA for a 16 GiB allocation and OOMed.  The value
    checks above pass either way -- only the SIZE is wrong -- so reproduce the
    real shape: one row carrying a payload that is trivial at n=1 and absurd
    at n=20000.
    """
    one = lambda row: jnp.sum(row ** 2)
    m = np.random.default_rng(0).normal(size=(1, 20_000, 5))   # 400 KB as-is
    got = B._over_targets(one, (m,), batch=20_000)             # 8 GB if broken
    assert got.shape == (1,)
    assert np.isclose(got[0], float((m[0] ** 2).sum()), rtol=1e-5)


def test_none_argument_passes_through_unbatched():
    # vmap treats a None leaf as an empty pytree; padding must not disturb it.
    one = lambda m_i, unused: jnp.sum(m_i)
    m = np.random.default_rng(1).normal(size=(13, 5))
    got = B._over_targets(one, (m, None), batch=8)
    assert got.shape == (13,)
    assert np.allclose(got, m.sum(1), rtol=0, atol=1e-5)


if __name__ == "__main__":
    test_trailing_chunk_padded_matches_unbatched()
    test_none_argument_passes_through_unbatched()
    print("ok")
