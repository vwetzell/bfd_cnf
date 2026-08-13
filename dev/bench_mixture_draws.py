"""Is `mixture_draws` at alpha < 1 dispatch-bound rather than compute-bound?

The paired run sits at 8% GPU and 108% CPU -- one core pegged -- which is the
signature of eager op-by-op dispatch through a deep flow, not of a GPU doing
work.  `mixture_draws` calls `flow.sample` and `flow.log_prob` with no jit
around either, so every bijection in the chain (and every MLP inside its
coefficient net) is a separate dispatch from Python.

Times the two calls as they are now against the same work under one
`eqx.filter_jit`.  Run with the real run in flight: preallocation off and a
small workload, so it fits in what the other process left free.
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402

DATA = "../bfd_cnf_imsims/data"
N_TGT, N_DRAW, REPS = int(os.environ.get("NT", 32)), int(os.environ.get("ND", 4096)), 5


def timeit(fn, reps=REPS):
    fn()                                 # warm up / compile
    t = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t) / reps * 1e3


def main():
    m_train = shear.load(f"{DATA}/moments.fits")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves("flows/centroid_deep.eqx", flow)

    rows = __import__("fitsio").read(f"{DATA}/targets_deep_g0_200k.fits")[:N_TGT]
    m = jnp.asarray(np.asarray(rows["moments"], dtype=np.float32))
    sx = jnp.asarray(np.asarray(rows["cov_odd"], dtype=np.float32))
    cond = jax.vmap(bias.condition, in_axes=(None, 0))(jnp.zeros(2), sx)

    key = jr.key(0)
    x = jnp.tile(m[:, None, :], (1, N_DRAW, 1))
    cond_b = jnp.broadcast_to(cond[:, None, :], x.shape[:-1] + cond.shape[-1:])

    print(f"{N_TGT} targets x {N_DRAW} draws, mean of {REPS}\n")

    t_s = timeit(lambda: jnp.moveaxis(
        flow.sample(key, (N_DRAW,), condition=cond), 0, 1))
    t_lp = timeit(lambda: flow.log_prob(x, condition=cond_b))
    print(f"  eager  flow.sample    {t_s:9.1f} ms")
    print(f"  eager  flow.log_prob  {t_lp:9.1f} ms")
    print(f"  eager  total          {t_s + t_lp:9.1f} ms")

    @eqx.filter_jit
    def both(flow, key, cond, x, cond_b):
        s = jnp.moveaxis(flow.sample(key, (N_DRAW,), condition=cond), 0, 1)
        return s, flow.log_prob(x, condition=cond_b)

    t_j = timeit(lambda: both(flow, key, cond, x, cond_b))
    print(f"  jitted both           {t_j:9.1f} ms   ->  {(t_s + t_lp) / t_j:.1f}x")

    # What the run actually spends per chunk, for scale: 3 catalogs x 2 chunks
    # x 624 target batches = 3744 mixture_draws calls per arm.
    print(f"\n  per arm at the eager cost: "
          f"{3744 * (t_s + t_lp) / 6e4:.0f} min just in these two calls")


if __name__ == "__main__":
    main()
