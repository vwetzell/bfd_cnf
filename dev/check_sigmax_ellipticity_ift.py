"""Verify SigmaXBlockLayer's own-ellipticity extension to net_dipquad.

net_dipquad (D,c) now reads the galaxy's own ellipticity (x3,x4), closing a
diagnosed PSF-leak model-form gap (see the class docstring). That makes the
INVERSE's (x3,x4) recovery a genuine 2x2 implicit equation instead of one
closed-form step -- solved by Picard fixed-point iteration
(`_solve_ell_picard`) + an implicit-function-theorem `custom_jvp`
(`_solve_ell_ift`/`_solve_ell_ift_jvp`), mirroring how `_solve_x0`/
`_solve_x0_ift` already handle the class's OTHER implicit equation.

Two things can go wrong that a training run would not obviously surface:

  1. the Picard iteration might not have converged in `_ELL_FIXEDPOINT_STEPS`
     -- caught here by checking the round trip end to end (forward then
     inverse recovers x, and the two log-dets cancel);
  2. the 2x2 IFT solve in `_solve_ell_ift_jvp` might have a sign/transpose
     bug -- invisible to (1) because a WRONG gradient does not break the
     primal round trip at all, only training dynamics later. Caught here by
     comparing `jax.jvp` against finite differences directly on
     `inverse_and_log_det`, both through `y0` and through a condition
     component (`e1`, via `C01`).

    python dev/check_sigmax_ellipticity_ift.py
"""
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

sys.path.insert(0, ".")
from models.bijections import SigmaXBlockLayer

ROUNDTRIP_TOL = 1e-4
JVP_REL_TOL = 0.03  # "a few percent" per the spec


def make_layer(key):
    return SigmaXBlockLayer(key, full_cond_dim=5)


def randomize_params(layer, key, scale=0.05):
    """Perturb every inexact-array leaf away from zero-init so D,c are
    actually nonzero -- at zero-init the layer is the identity and both the
    Picard loop and the IFT are exercised trivially (0 iterations needed,
    dPhi/dx = 0)."""
    params, static = eqx.partition(layer, eqx.is_inexact_array)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jr.split(key, len(leaves))
    noisy = [l + scale * jr.normal(k, l.shape) for l, k in zip(leaves, keys)]
    params2 = jax.tree_util.tree_unflatten(treedef, noisy)
    return eqx.combine(params2, static)


def sample_x(key, n):
    k0, k3 = jr.split(key)
    xy = 0.3 * jr.normal(k0, (n, 3))
    ell = 0.05 * jr.normal(k3, (n, 2))
    return jnp.concatenate([xy, ell], axis=1)


def sample_condition(key, n, e_max):
    kg, kc00, kc11, kc01u = jr.split(key, 4)
    g = 0.05 * jr.normal(kg, (n, 2))
    C00 = 0.3 + 0.3 * jr.uniform(kc00, (n,))
    C11 = 0.3 + 0.3 * jr.uniform(kc11, (n,))
    # |C01| < sqrt(C00*C11) * f, f<1 chosen so the implied PSF ellipticity
    # stays within the layer's own e_max (see cx_to_sx_cond); f up to ~0.9*
    # e_max keeps a healthy spread without hugging the boundary singularity.
    f = (0.9 * e_max) * (2.0 * jr.uniform(kc01u, (n,)) - 1.0)
    C01 = f * jnp.sqrt(C00 * C11)
    return jnp.stack([g[:, 0], g[:, 1], C00, C01, C11], axis=1)


def main():
    key = jr.PRNGKey(0)
    k_layer, k_perturb, k_x, k_cond, k_jvp = jr.split(key, 5)
    layer = make_layer(k_layer)
    layer = randomize_params(layer, k_perturb, scale=0.05)
    e_max = layer._e_mag_sq_scale ** 0.5

    xs = sample_x(k_x, 50)
    conds = sample_condition(k_cond, 20, e_max)

    # vmap over the (x, condition) grid + jit -- eager per-pair calls (each
    # one a jacfwd + a 20-step Picard scan) were too slow to be a "quiet
    # check" at 50*20=1000 pairs.
    @jax.jit
    def _roundtrip(x, cond):
        y, ld_fwd = layer.transform_and_log_det(x, cond)
        xr, ld_inv = layer.inverse_and_log_det(y, cond)
        return xr, ld_fwd + ld_inv

    roundtrip_grid = jax.vmap(jax.vmap(_roundtrip, in_axes=(None, 0)), in_axes=(0, None))
    xr_grid, ld_sum_grid = roundtrip_grid(xs, conds)
    resid_grid = jnp.max(jnp.abs(xr_grid - xs[:, None, :]), axis=-1)
    max_resid = float(jnp.max(resid_grid))
    max_ld_resid = float(jnp.max(jnp.abs(ld_sum_grid)))
    n_checked = xs.shape[0] * conds.shape[0]

    print(f"round trip: {n_checked} (x, condition) pairs")
    print(f"  max |x_recovered - x|      = {max_resid:.3e}  (tol {ROUNDTRIP_TOL:.0e})")
    print(f"  max |log_det_fwd + log_det_inv| = {max_ld_resid:.3e}  (tol {ROUNDTRIP_TOL:.0e})")
    assert max_resid < ROUNDTRIP_TOL, (
        f"Picard fixed point ({SigmaXBlockLayer._ELL_FIXEDPOINT_STEPS} steps) "
        f"did not converge to {ROUNDTRIP_TOL:.0e}: got {max_resid:.3e}")
    assert max_ld_resid < ROUNDTRIP_TOL, (
        f"log-det round trip off by {max_ld_resid:.3e}, tol {ROUNDTRIP_TOL:.0e}")

    # --- custom_jvp finite-difference check -------------------------------
    # This is the step most likely to have a sign/transpose bug in the 2x2
    # IFT solve (`_solve_ell_ift_jvp`): a wrong gradient does not break the
    # round trip above at all (that only exercises the primal), only
    # training dynamics later -- so it needs its own direct check.
    jvp_keys = jr.split(k_jvp, 6)
    pairs = [(xs[i], conds[i % len(conds)]) for i in range(6)]
    eps = 1e-6
    max_rel_err_y0 = 0.0
    max_rel_err_e1 = 0.0
    for x, cond in pairs:
        y, _ = layer.transform_and_log_det(x, cond)

        def f_of_y(yy, cond=cond):
            return layer.inverse_and_log_det(yy, cond)[0]

        dy0 = jnp.zeros(5).at[0].set(1.0)
        fd = (f_of_y(y + eps * dy0) - f_of_y(y - eps * dy0)) / (2 * eps)
        _, jv = jax.jvp(f_of_y, (y,), (dy0,))
        rel = float(jnp.max(jnp.abs(fd - jv)) / (jnp.max(jnp.abs(jv)) + 1e-9))
        max_rel_err_y0 = max(max_rel_err_y0, rel)

        # Perturb a condition component that maps onto e1 (C01, index 3):
        # net_dipquad's dependence on e1 flows entirely through D,c inside
        # the Picard/IFT block, so this exercises the same solve.
        def f_of_cond(c, y=y):
            return layer.inverse_and_log_det(y, c)[0]

        dc = jnp.zeros(5).at[3].set(1.0)
        fdc = (f_of_cond(cond + eps * dc) - f_of_cond(cond - eps * dc)) / (2 * eps)
        _, jvc = jax.jvp(f_of_cond, (cond,), (dc,))
        relc = float(jnp.max(jnp.abs(fdc - jvc)) / (jnp.max(jnp.abs(jvc)) + 1e-9))
        max_rel_err_e1 = max(max_rel_err_e1, relc)

    print(f"jvp vs finite-diff (d/dy0):  max rel err = {max_rel_err_y0:.3e}  (tol {JVP_REL_TOL:.0%})")
    print(f"jvp vs finite-diff (d/dC01): max rel err = {max_rel_err_e1:.3e}  (tol {JVP_REL_TOL:.0%})")
    assert max_rel_err_y0 < JVP_REL_TOL, f"y0 jvp mismatch: {max_rel_err_y0:.3e}"
    assert max_rel_err_e1 < JVP_REL_TOL, f"C01/e1 jvp mismatch: {max_rel_err_e1:.3e}"

    print("PASS")


if __name__ == "__main__":
    main()
