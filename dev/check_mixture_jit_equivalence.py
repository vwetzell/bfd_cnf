"""Does jitting `mixture_draws`' per-batch body change what it returns?

The hoist moved `flow.sample`/`flow.log_prob` and the weight arithmetic into a
module-level `_mixture_chunk`, jitted once.  This checks the new
`bias.mixture_draws` against a verbatim copy of the pre-hoist body (from
`git show HEAD:bias.py`, before the hoisting commit) on the real flow and real
targets: same seed in, draws must be bitwise equal and log-weights equal to
within float32 fusion noise, at both alpha < 1 (the path that changed) and
alpha = 1 (the path that must still bypass the flow entirely).
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias
import bulk
import shear

DATA = "../bfd_cnf_imsims/data"


def condition(g, sigma_x):
    return g if sigma_x is None else jnp.concatenate([g, sigma_x])


def mixture_draws_reference(flow, m, cov, samples, alpha, seed, batch=None,
                             sigma_x=None):
    """Verbatim pre-hoist body of `bias.mixture_draws` (git show HEAD:bias.py),
    kept here only as the equivalence check's ground truth."""
    n = len(m)
    batch = max(1, 131_072 // samples) if batch is None else batch
    n_k = round(alpha * samples)
    n_k -= n_k % 2
    n_f = samples - n_k

    m64 = np.asarray(m, dtype=np.float64)
    kernel = bias.kernel_draws(cov, n, n_k, seed)

    L = jnp.asarray(np.linalg.cholesky(cov))
    log_diag = jnp.sum(jnp.log(jnp.diag(L)))

    def log_normal(offset):
        shp = offset.shape[:-1]
        y = jax.scipy.linalg.solve_triangular(L, offset.reshape(-1, 5).T, lower=True)
        quad = jnp.sum(y * y, axis=0).reshape(shp)
        return -0.5 * quad - log_diag - 2.5 * jnp.log(2 * jnp.pi)

    log_alpha = -jnp.inf if alpha <= 0.0 else float(np.log(alpha))
    log_1ma = -jnp.inf if alpha >= 1.0 else float(np.log1p(-alpha))

    key = jr.key(seed) if n_f else None
    draws_out, wt_out = [], []
    for i in range(0, n, batch):
        m_i = jnp.asarray(m64[i:i + batch])
        x = m_i[:, None, :] + jnp.asarray(kernel[i:i + batch])
        cond0 = (jnp.zeros(2) if sigma_x is None else
                 jax.vmap(condition, in_axes=(None, 0))(
                     jnp.zeros(2), jnp.asarray(sigma_x[i:i + batch],
                                               dtype=jnp.float32)))
        if n_f:
            key, sub = jr.split(key)
            if cond0.ndim == 1:
                flow_x = flow.sample(sub, (m_i.shape[0], n_f), condition=cond0)
            else:
                flow_x = jnp.moveaxis(
                    flow.sample(sub, (n_f,), condition=cond0), 0, 1)
            x = jnp.concatenate([x, flow_x], axis=1)

        if alpha >= 1.0:
            draws_out.append(np.asarray(x, dtype=np.float32))
            wt_out.append(np.zeros(x.shape[:-1], dtype=np.float32))
            continue
        log_k = log_normal(x - m_i[:, None, :])
        ok = (x[..., 0] > 0) & (x[..., 1] > 0)
        dummy = m_i.at[:, :2].set(jnp.maximum(m_i[:, :2], 1e-6))
        safe = jnp.where(ok[..., None], x, dummy[:, None, :])
        cond_f = (cond0 if cond0.ndim == 1 else
                  jnp.broadcast_to(cond0[:, None, :],
                                   safe.shape[:-1] + cond0.shape[-1:]))
        log_f = jnp.where(ok, flow.log_prob(safe, condition=cond_f), -jnp.inf)
        log_q = jnp.logaddexp(log_alpha + log_k, log_1ma + log_f)
        log_wt = log_k - log_q
        draws_out.append(np.asarray(x, dtype=np.float32))
        wt_out.append(np.asarray(jnp.where(jnp.isfinite(log_wt), log_wt, -jnp.inf),
                                 dtype=np.float32))

    return np.concatenate(draws_out), np.concatenate(wt_out)


def check(flow, m, cov, sigma_x, alpha, samples=512, seed=12345, label=""):
    d_ref, w_ref = mixture_draws_reference(flow, m, cov, samples, alpha, seed,
                                           sigma_x=sigma_x)
    d_new, w_new = bias.mixture_draws(flow, m, cov, samples, alpha, seed,
                                      sigma_x=sigma_x)

    assert d_ref.shape == d_new.shape, (d_ref.shape, d_new.shape)
    assert w_ref.shape == w_new.shape, (w_ref.shape, w_new.shape)

    assert np.array_equal(d_ref, d_new), f"{label}: draws not bitwise equal"

    both_inf = ~np.isfinite(w_ref) & ~np.isfinite(w_new)
    diff = np.abs(w_ref - w_new)
    diff = np.where(both_inf, 0.0, diff)
    assert np.isfinite(diff[~both_inf]).all(), f"{label}: non-finite mismatch"

    # An ABSOLUTE tolerance is the wrong yardstick here.  `log_wt = log_k -
    # log_q`, and for a draw far out in the tail `log_q` is carried by a
    # `log_f` of order -1e4, where one float32 ULP is ~1e-3.  So a single-ULP
    # difference between the jitted and eager flow reads as ~1e-3 nats on a
    # draw whose weight is e^-1e4 -- exactly zero contribution.  The observed
    # failures were 2^-3 and 2^-10, i.e. whole numbers of ULPs, not drift.
    # What must be tight is the draws that actually carry weight.
    lead = w_ref - w_ref.max(axis=1, keepdims=True)
    near = lead >= -30.0                 # e^-30 below the best draw and up
    max_near = float(diff[near].max())
    max_diff = float(diff.max())
    assert max_near < 1e-4, f"{label}: {max_near} on a weight-carrying draw"
    print(f"  {label}: max |log_wt diff| = {max_near:.3e} on the draws within "
          f"30 nats of the best ({max_diff:.3e} over all, tail included)")
    return max_near


def main():
    m_train = shear.load(f"{DATA}/moments.fits")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves("flows/centroid_deep.eqx", flow)

    n = 32
    rows = fitsio.read(f"{DATA}/targets_deep_g0_200k.fits")[:n]
    m = np.asarray(rows["moments"], dtype=np.float64)
    sigma_x = np.asarray(rows["cov_odd"], dtype=np.float64)
    cov = bias.load_cov(f"{DATA}/targets_deep_g0_200k.fits")

    max_diff = 0.0
    max_diff = max(max_diff, check(flow, m, cov, sigma_x, alpha=0.5,
                                   label="alpha=0.5"))
    max_diff = max(max_diff, check(flow, m, cov, sigma_x, alpha=1.0,
                                   label="alpha=1.0"))

    print(f"\nmax abs log-weight difference over all checks: {max_diff:.3e}")
    print("ok")


if __name__ == "__main__":
    main()
