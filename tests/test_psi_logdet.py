"""`make_psi_ld`'s chain-rule log-det against `slogdet(jacfwd(make_psi))`.
`python -m tests.test_psi_logdet`.

`log_conv_is_blend` needs `log|det dPsi_g/du|` per draw.  It used to get it by
differentiating the whole composition, which contains `ShearResponse.shear` --
itself three nested `jacfwd` calls -- so a 5-column Jacobian of it, inside a
forward-over-reverse Hessian in g, was 92% of a production run's wall clock.
`make_psi_ld` sums the layers' own log-dets instead.

Three claims, and the third is the one Q and R actually rest on:

  1. the POINT is unchanged -- this is a log-det change, not a map change;
  2. the log-det agrees at g = 0, where every draw is evaluated;
  3. its first and second g-derivatives at g = 0 agree, which is all that
     reaches Q and R (eq. 12-13).

They differ at O(g^3) by construction: `shear` inverts `unshear`'s g-series to
second order, while `inverse_and_log_det` reads the log-det off `unshear` at
the image point.  Nothing in the estimator sees that order, and claim 3 is
what says so.

An untrained flow is enough -- the identity is structural, not learned.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)

import bias  # noqa: E402
import bulk  # noqa: E402
from models.bijections import in_domain  # noqa: E402

SIGMA_X = jnp.asarray([1.155e4, 0.0, 1.155e4])   # the catalogs' own cov_odd


def _flow_and_points(centroid):
    key = jr.key(0)
    m = np.asarray(
        jr.uniform(key, (256, 5), minval=jnp.asarray([2e3, 6e3, -5e2, -5e2, 3e4]),
                   maxval=jnp.asarray([9e3, 2.4e4, 5e2, 5e2, 1.2e5])), np.float64)
    m = m[np.asarray(in_domain(jnp.asarray(m)))]
    assert len(m) > 32, len(m)
    flow = bulk.build_flow(key, m, shear=True, centroid=centroid)
    return flow, jnp.asarray(m[:16])


def test_psi_ld_matches_jacfwd():
    for centroid in (False, True):
        flow, x = _flow_and_points(centroid)
        psi = bias.make_psi(flow, peeled=False)
        psi_ld = bias.make_psi_ld(flow, peeled=False)
        sx = SIGMA_X if centroid else None

        old_ld = lambda u, g: jnp.linalg.slogdet(
            jax.jacfwd(psi, argnums=0)(u, g, sx))[1]

        for g in (jnp.zeros(2), jnp.asarray([0.02, -0.01])):
            y_new, ld_new = jax.vmap(psi_ld, in_axes=(0, None, None))(x, g, sx)
            y_old = jax.vmap(psi, in_axes=(0, None, None))(x, g, sx)
            ld_old = jax.vmap(old_ld, in_axes=(0, None))(x, g)

            # 1. the map itself is untouched.
            rel = np.max(np.abs(np.asarray(y_new - y_old))
                         / np.abs(np.asarray(y_old)))
            assert rel < 1e-12, (f"point centroid={centroid} g={g}", rel)

            # 2. the log-det agrees.  At g != 0 the O(g^3) series-inversion gap
            # is real and is allowed a loose bound; at g = 0 it is roundoff.
            tol = 1e-9 if not np.any(np.asarray(g)) else 1e-4
            err = np.max(np.abs(np.asarray(ld_new - ld_old)))
            assert err < tol, (f"logdet centroid={centroid} g={g}", err)


def test_psi_ld_g_derivatives_match():
    """Claim 3: the first two g-derivatives at g = 0 -- what Q and R consume.

    The ground truth here is FINITE DIFFERENCES of `old_ld`, not a second
    `jax.grad`/`jax.hessian` pass over it as in an earlier version of this
    test.  `centroid=True` puts `SigmaXBlockLayer`'s `_solve_x0` (its flux
    coefficient reads the galaxy's OWN flux, so inverting it takes a
    bisection, see `models/bijections.py`) in the composition, and that
    bisection is only DIFFERENTIABLE at all because of `_solve_x0_ift`'s
    `custom_jvp` (an implicit-function-theorem rule; the naive bisection's
    own gradient through `jax.lax.scan`'s boolean branching is exactly
    zero).  That rule is verified correct to first AND second order by
    direct finite-difference checks on `psi_ld` itself (not shown here).
    But `old_ld` here differentiates `psi` via an EXTRA, inner `jacfwd`
    (w.r.t. `u`, not `g`) before `old_ld`'s own `g`-derivative is taken --
    a cross-derivative of `_solve_x0_ift`'s rule w.r.t. a primal variable
    the rule was not itself differentiated with respect to, which plain
    `jax.grad`/`jax.hessian` of `old_ld` silently gets WRONG (measured
    ~1-2% relative on this population) rather than raising -- `custom_jvp`
    has no first-order support for that composition, only for differentiating
    directly w.r.t. the variable its own rule declares tangents for, which is
    exactly what `psi_ld`/`f_new` below does (matches finite differences to
    ~1e-8, see the same investigation).  Finite-differencing `old_ld` itself
    needs no such second pass -- every evaluation is a fixed-`g` VALUE -- so
    it stays a trustworthy, if slower, oracle.
    """
    for centroid in (False, True):
        flow, x = _flow_and_points(centroid)
        psi = bias.make_psi(flow, peeled=False)
        psi_ld = bias.make_psi_ld(flow, peeled=False)
        sx = SIGMA_X if centroid else None
        zero = jnp.zeros(2)
        eps = 1e-4

        for lam in (0.5, 1.0):
            def old_ld_all(g, psi=psi, lam=lam, sx=sx, x=x):
                return jax.vmap(lambda u: jnp.linalg.slogdet(
                    jax.jacfwd(psi, argnums=0)(u, lam * g, sx))[1])(x)

            f_new = lambda g, u: psi_ld(u, lam * g, sx)[1]

            e0, e1 = jnp.array([eps, 0.0]), jnp.array([0.0, eps])
            c = np.asarray(old_ld_all(zero))
            p0, m0 = np.asarray(old_ld_all(e0)), np.asarray(old_ld_all(-e0))
            p1, m1 = np.asarray(old_ld_all(e1)), np.asarray(old_ld_all(-e1))
            pp = np.asarray(old_ld_all(e0 + e1))
            pm = np.asarray(old_ld_all(e0 - e1))
            mp = np.asarray(old_ld_all(-e0 + e1))
            mm = np.asarray(old_ld_all(-e0 - e1))

            grad_old = np.stack([(p0 - m0) / (2 * eps),
                                  (p1 - m1) / (2 * eps)], axis=-1)
            h00 = (p0 - 2 * c + m0) / eps ** 2
            h11 = (p1 - 2 * c + m1) / eps ** 2
            h01 = (pp - pm - mp + mm) / (4 * eps ** 2)
            hess_old = np.stack([np.stack([h00, h01], axis=-1),
                                  np.stack([h01, h11], axis=-1)], axis=-2)

            for f_b, ref, order, tol in (
                    (jax.grad(f_new), grad_old, 1, 1e-4),
                    (jax.hessian(f_new), hess_old, 2, 1e-2)):
                b = np.asarray(jax.vmap(f_b, in_axes=(None, 0))(zero, x))
                err = np.max(np.abs(b - ref))
                scale = max(1.0, np.max(np.abs(ref)))
                assert err / scale < tol, (
                    f"d{order} centroid={centroid} lam={lam}", err, scale)


if __name__ == "__main__":
    test_psi_ld_matches_jacfwd()
    test_psi_ld_g_derivatives_match()
    print("ok")
