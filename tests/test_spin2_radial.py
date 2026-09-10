"""The spin-2 radial stretch: a scale with no shift, whose profile in |e| the
net learns.  What must hold is monotonicity -- a stretch that turns over folds
the plane and makes log|det| silently wrong -- plus the closed-form inverse,
equivariance, and the thing it exists for: under-dispersing |e|^2."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from models.bijections import Spin2CouplingLayer, check_monotone


def layer(seed=0, width=16, depth=2, bend=True):
    return Spin2CouplingLayer(jr.key(seed), width, depth, jax.nn.silu, bend=bend)


def points(n=256, seed=1, e_scale=1.2):
    k1, k2 = jr.split(jr.key(seed))
    spin0 = jr.normal(k1, (n, 3)) * 1.5
    spin2 = jr.normal(k2, (n, 2)) * e_scale
    return jnp.concatenate([spin0, spin2], axis=-1)


def test_log_det_matches_autodiff():
    """The closed-form log T'(q) is the real 5x5 log-det.

    FORWARD ONLY by autodiff.  The inverse is a bisection, whose output has
    zero gradient with respect to w (the bracket updates are comparisons), so
    `jacfwd` through it reports a collapsed Jacobian -- an artefact of the
    solver, not of the map.  Nothing differentiates through `sample`, so this
    costs nothing; the inverse's log-det is checked against the forward's
    below, which is the invariant that actually has to hold.
    """
    lay = layer()
    f = lambda v: lay.transform_and_log_det(v)[0]
    lad = jax.vmap(lambda v: lay.transform_and_log_det(v)[1])(points())
    num = jax.vmap(lambda v: jnp.linalg.slogdet(jax.jacfwd(f)(v))[1])(points())
    assert np.allclose(lad, num, atol=1e-6)


def test_inverse_log_det_is_the_forward_one_negated():
    """The invariant the bisection must satisfy: at x = inverse(y), the two
    log-dets are equal and opposite."""
    lay, y = layer(), points()
    x, lad_inv = jax.vmap(lambda v: lay.inverse_and_log_det(v))(y)
    lad_fwd = jax.vmap(lambda v: lay.transform_and_log_det(v)[1])(x)
    assert np.allclose(lad_inv, -lad_fwd, atol=1e-6)


def test_round_trip():
    lay, x = layer(), points()
    y = jax.vmap(lambda v: lay.transform_and_log_det(v)[0])(x)
    back = jax.vmap(lambda v: lay.inverse_and_log_det(v)[0])(y)
    assert np.allclose(back, x, atol=1e-6)


def test_is_a_pure_stretch():
    """No shift, and the direction of (M1, M2) never moves -- that is what
    makes the layer equivariant.  Only |e| may change."""
    lay, x = layer(), points()
    y = jax.vmap(lambda v: lay.transform_and_log_det(v)[0])(x)
    assert np.allclose(y[:, :3], x[:, :3], atol=1e-12)     # spin-0 untouched
    rx = np.linalg.norm(np.asarray(x[:, 3:]), axis=-1)
    ry = np.linalg.norm(np.asarray(y[:, 3:]), axis=-1)
    ux, uy = np.asarray(x[:, 3:]) / rx[:, None], np.asarray(y[:, 3:]) / ry[:, None]
    assert np.allclose(ux, uy, atol=1e-9)                  # direction preserved


@pytest.mark.parametrize("seed", range(6))
def test_does_not_fold_at_initialisation(seed):
    """Monotonicity is NOT enforced -- ellipticity does not physically fold, so
    the job is to keep the flow out of that region rather than contort the map.
    This checks the diagnostic that watches for it, and that a fresh net sits
    comfortably inside the valid region over a wide |e| sweep."""
    lay = layer(seed)
    x = points(2048, seed=seed + 10, e_scale=4.0)
    assert check_monotone(lay, x) == 0.0
    det = jax.vmap(lambda v: jnp.linalg.det(
        jax.jacfwd(lambda w: lay.transform_and_log_det(w)[0])(v)))(x)
    assert np.all(np.asarray(det) > 0)


def test_a_fold_is_loud_not_silent():
    """The failure mode we are choosing to tolerate must announce itself: a
    folded stretch makes log1p see an argument <= -1, so the log-det is NaN at
    the step it happens, not a quietly wrong density with two preimages."""
    lay = layer()
    # force a steeply decreasing h(q) by driving the q input's weight
    W = lay.net.layers[0].weight
    lay = eqx.tree_at(lambda l: l.net.layers[0].weight, lay,
                      W.at[:, 3].set(-60.0))
    x = points(512, e_scale=1.5)
    assert check_monotone(lay, x) > 0.0, "test did not actually induce a fold"
    lad = jax.vmap(lambda v: lay.transform_and_log_det(v)[1])(x)
    assert np.any(~np.isfinite(np.asarray(lad)))


def test_far_field_cannot_fold():
    """`_soft_clip` saturates the conditioner inputs, so h goes constant in q
    far outside the envelope, dh/dq -> 0 and T' -> s^2 > 0 on its own.  That is
    what confines folding to the region the data occupies."""
    lay = layer()
    far = jnp.array([[0.4, -0.9, 1.3, m, 0.0] for m in (1e3, 1e5, 1e9)])
    assert check_monotone(lay, far) == 0.0
    lad = jax.vmap(lambda v: lay.transform_and_log_det(v)[1])(far)
    assert np.all(np.isfinite(np.asarray(lad)))


def test_finite_and_smooth_at_zero_ellipticity():
    """q = 0 is 0/0 in T(q)/q; the continuation must keep value AND gradient."""
    lay = layer()
    x = jnp.array([0.3, -0.7, 1.1, 0.0, 0.0])
    y, lad = lay.transform_and_log_det(x)
    assert np.all(np.isfinite(y)) and np.isfinite(lad)
    assert np.allclose(y[3:], 0.0)          # a stretch fixes the origin
    assert np.all(np.isfinite(
        jax.jacfwd(lambda v: lay.transform_and_log_det(v)[0])(x)))


@pytest.mark.parametrize("mag", [1e2, 1e3, 1e5, 1e9])
def test_survives_the_flux_tail_in_float32(mag):
    """Intermediate coordinates reach ~1e5 on the flux tail and training is
    float32; one inf poisons the whole batch gradient.  Beyond the spline's
    interval the profile is the identity, so this is now structural rather
    than a numerical continuation -- an earlier expm1 form lost it by forming
    expm1(z)/z, which overflows at z ~ 88 while the product it feeds is ~1e31.
    """
    lay = jax.tree.map(
        lambda l: l.astype(jnp.float32) if eqx.is_inexact_array(l) else l, layer())
    x = jnp.concatenate([jnp.zeros((3, 3)),
                         jnp.array([[mag, 0.0], [0.0, -mag], [mag, mag]])],
                        axis=-1).astype(jnp.float32)
    for name in ("transform_and_log_det", "inverse_and_log_det"):
        v, lad = jax.vmap(lambda p, n=name: getattr(lay, n)(p))(x)
        assert np.all(np.isfinite(v)), f"{name} value at |e| = {mag}"
        assert np.all(np.isfinite(lad)), f"{name} log-det at |e| = {mag}"


def test_is_rotation_equivariant():
    lay, x = layer(), points(64)
    th = 0.7
    R = jnp.array([[jnp.cos(th), -jnp.sin(th)], [jnp.sin(th), jnp.cos(th)]])
    rot = lambda v: v.at[3:].set(R @ v[3:])
    f = jax.vmap(lambda v: lay.transform_and_log_det(v)[0])
    assert np.allclose(f(jax.vmap(rot)(x)), jax.vmap(rot)(f(x)), atol=1e-9)


def test_can_under_disperse_e_squared():
    """The whole point.  A pure scale -- what this layer was before -- gives
    var/mean^2 = 1 for |e|^2 identically and can only err on the OVER-dispersed
    side; bulgedisc_v3 wants ~0.565.  Sweep the weight on the q input (a
    stretch that shrinks with |e| compresses the tail) and require the
    reachable range to pass through the target."""
    e = jr.normal(jr.key(3), (60000, 2))
    base = jnp.concatenate([jnp.zeros((len(e), 3)), e], axis=-1)

    def driven(wq):
        lay = layer()
        W = lay.net.layers[0].weight
        lay = eqx.tree_at(
            lambda l: (l.net.layers[0].weight, l.net.layers[-1].weight),
            lay, (jnp.zeros_like(W).at[:, 3].set(wq),
                  jnp.ones_like(lay.net.layers[-1].weight) * 0.3))
        d = jax.vmap(lambda v: lay.inverse_and_log_det(v)[0])(base)
        q = np.asarray(d[:, 3] ** 2 + d[:, 4] ** 2)
        return q.var() / q.mean() ** 2

    # 0 -> ~1.01 (a pure scale, the old layer), 4 -> ~0.51; the target is
    # crossed in between, and the reachable floor is ~0.47.
    hi, lo = driven(0.0), driven(4.0)
    assert lo < 0.565 < hi, (lo, hi)


def test_only_the_first_layer_bends():
    """The bend composes multiplicatively across layers, and the spline's
    interval is a fixed range in |e|^2 that only matches the data-adjacent
    coordinate."""
    import bulk
    f = bulk.build_flow(jr.key(0), np.exp(np.random.default_rng(0).normal(
        size=(4000, 5)) * 0.1 + np.array([8.0, 7.0, 3.0, 3.0, 8.5])))
    bends = [b.spin2.bend for b in f.bijection.bijection.bijections
             if hasattr(b, "spin2")]
    assert sum(bends) == 1, bends
    assert bends[0], "the bend belongs to the first layer"


def test_unbent_layer_is_exactly_the_old_pure_scale():
    """The other seven must be numerically what they always were."""
    lay, x = layer(bend=False), points()
    y, lad = jax.vmap(lambda v: lay.transform_and_log_det(v))(x)
    s = jnp.exp(jax.vmap(lambda v: lay._h(v, 0.0))(x))
    assert np.allclose(y[:, 3:], x[:, 3:] * s[:, None], atol=1e-12)
    assert np.allclose(y[:, :3], x[:, :3], atol=1e-12)
    assert np.allclose(lad, 2.0 * jnp.log(s), atol=1e-12)
    # and it must not read q at all
    assert lay.net.layers[0].weight.shape[1] == 3
