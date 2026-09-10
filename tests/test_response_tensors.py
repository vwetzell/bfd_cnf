"""`ShearResponse.response_tensors` against differentiating `unshear`.
`python -m tests.test_response_tensors`.

`response` is `z + Q @ g + 0.5 g.R.g` and `project_to_physics` builds `Q`, `R`
from `z` alone -- its `g` argument is dead.  So `Q` and `R` ARE `unshear`'s
first and second g-derivatives at g = 0, and `shear` used to recover them with
`jacfwd(unshear, argnums=1)` and its double, differentiating through a function
that already contains a `jacfwd` and a `hessian`.

Three claims:

  1. `Q` equals `d unshear/dg` at g = 0, and `R` equals `d2 unshear/dg2`;
  2. `g` really is unused -- `project_to_physics` returns the same tensors for
     wildly different `g`, which is what licenses claim 1;
  3. `shear` still inverts `unshear` to the documented O(g^3).

An untrained layer is enough: all three are structural.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)

from paramax import unwrap  # noqa: E402

from models.shear import (  # noqa: E402
    ShearResponse, _invariants, project_to_physics)

KEY = jr.key(0)
LAYER = ShearResponse(KEY, nn_width=32, nn_depth=2, e_scale=0.3)
# Standardised coordinates, the space this layer acts in.
X = jr.normal(jr.key(1), (16, 5))
ZERO = jnp.zeros(2)


def test_tensors_are_the_g_derivatives():
    """Claim 1: the closed form and the autodiff agree."""
    got_q, got_r = jax.vmap(LAYER.response_tensors)(X)
    want_q = jax.vmap(
        lambda x: jax.jacfwd(LAYER.unshear, argnums=1)(x, ZERO))(X)
    want_r = jax.vmap(
        lambda x: jax.jacfwd(jax.jacfwd(LAYER.unshear, argnums=1),
                             argnums=1)(x, ZERO))(X)
    for name, got, want in (("Q", got_q, want_q), ("R", got_r, want_r)):
        err = np.max(np.abs(np.asarray(got - want)))
        scale = max(1.0, np.max(np.abs(np.asarray(want))))
        assert err / scale < 1e-10, (name, err, scale)


def test_g_argument_is_dead():
    """Claim 2: `project_to_physics` ignores `g`, which is what makes 1 hold."""
    coeffs = jax.vmap(lambda x: LAYER.coeffs(*_invariants(x)[:4]))(X)
    args = (unwrap(LAYER.e_scale), unwrap(LAYER.chart_loc),
            unwrap(LAYER.chart_scale))
    f = lambda g: jax.vmap(
        lambda c, x: project_to_physics(c, x, g, *args))(coeffs, X)
    a_q, a_r = f(ZERO)
    b_q, b_r = f(jnp.array([0.3, -0.4]))
    assert np.array_equal(np.asarray(a_q), np.asarray(b_q))
    assert np.array_equal(np.asarray(a_r), np.asarray(b_r))


def test_shear_still_inverts_unshear():
    """Claim 3: the series inversion is unchanged, residual still O(g^3)."""
    prev = None
    for scale in (0.02, 0.01):
        g = jnp.array([scale, -0.5 * scale])
        rt = jax.vmap(lambda x: LAYER.unshear(LAYER.shear(x, g), g))(X)
        err = float(np.max(np.abs(np.asarray(rt - X))))
        if prev is not None:
            # Halving g must cut the residual by ~8x if it is genuinely cubic;
            # allow slack for the float floor.
            assert err < prev / 4.0, (err, prev)
        prev = err


if __name__ == "__main__":
    test_tensors_are_the_g_derivatives()
    test_g_argument_is_dead()
    test_shear_still_inverts_unshear()
    print("ok")
