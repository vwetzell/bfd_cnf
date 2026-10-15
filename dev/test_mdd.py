"""Self-check of centroid.mdd_points against a brute-force U-statistic, and its null mean.  CPU, seconds.
Usage: JAX_PLATFORMS=cpu python dev/test_mdd.py"""
import sys
sys.path.insert(0, ".")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import centroid as C

rng = np.random.default_rng(0)
B = 300
z = rng.normal(size=(B, 5))
r = rng.normal(size=(B, 2)) + np.stack([0.3 * z[:, 0], 0 * z[:, 0]], 1) + 0.1
A = np.sqrt(((z[:, None] - z[None]) ** 2).sum(-1))
a, at = A.sum(1), A.sum()
At = A - a[:, None] / (B - 2) - a[None] / (B - 2) + at / ((B - 1) * (B - 2))
np.fill_diagonal(At, 0)
S, off = r @ r.T, ~np.eye(B, dtype=bool)
brute = -(At * S)[off].sum() / (B * (B - 3)) + at / (B * (B - 1)) * S[off].sum() / (B * (B - 1))
got = float(jnp.mean(C.mdd_points(jnp.asarray(r), C.mdd_dist(jnp.asarray(z)))))
assert abs(got - brute) < 1e-9 * abs(brute), (got, brute)
null = [float(jnp.mean(C.mdd_points(jnp.asarray(rng.normal(size=(B, 2))), C.mdd_dist(jnp.asarray(z)))))
        for _ in range(400)]
assert abs(np.mean(null)) < 4 * np.std(null) / 20, np.mean(null)
print(f"ok: brute {brute:.6e} code {got:.6e}; null mean {np.mean(null):+.1e} +/- {np.std(null) / 20:.1e}")
