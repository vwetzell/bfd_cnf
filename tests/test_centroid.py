"""Self-checks for models/centroid.py -- the centroid-marginalisation layer.

Run: `python -m tests.test_centroid` or pytest.

These check the symmetries the layer's form was DERIVED from, so a failure here
means the derivation or the code departed from it, not that a fit is poor.  The
quality of the fit is `centroid.py check`'s job, against the catalog.
"""

import jax

# These are exact-symmetry checks, so they run in float64.  The layer itself
# runs in float32, where `_tensor`'s spin-2 part loses most of its significance
# to cancellation (see the note there) -- testing an exact symmetry at float32
# roundoff would measure the roundoff, not the symmetry.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp        # noqa: E402  -- must follow the x64 switch
import jax.random as jr        # noqa: E402
import numpy as np             # noqa: E402

from models.bijections import POINT_SOURCE, POINT_SOURCE_MC
from models.centroid import (CentroidMarginalize, displacement_covariance,
                             raw_from_standard, response, _tensor)

# A representative bulge+disc galaxy and the imsims Sigma_X at the nominal noise.
M = jnp.array([57026.8, 135254.4, -9006.5, -8736.6, 658762.5])
SIGMA_X = jnp.array([15941.4, 0.0, 15941.4])

# The layer now acts on STANDARDISED coordinates, so the tests need the
# standardiser's constants -- the layer carries a frozen copy to rebuild a
# physical scale for T (see `raw_from_standard`).  These are representative of
# what `build_flow` measures on bulgedisc; the exact values do not matter to a
# symmetry test, but two things do: the spin-2 slots must share a std and have
# ZERO mean, which is the symmetrisation that keeps the chart isotropic.
MEAN = jnp.array([4.35, 0.95, 1.05, 0.0, 0.0])
STD = jnp.array([0.35, 0.75, 0.60, 0.040, 0.040])


def _to_z(m):
    """`bulk.to_coords` then standardise -- the inverse of `raw_from_standard`."""
    u = m[1] / (POINT_SOURCE * m[0])
    v = m[4] / (POINT_SOURCE_MC * m[1])
    c = jnp.stack([jnp.log10(m[0]), jnp.log(u) - jnp.log1p(-u),
                   jnp.log(v) - jnp.log1p(-v), m[2] / m[1], m[3] / m[1]])
    return (c - MEAN) / STD


Z = _to_z(M)
LAYER = CentroidMarginalize(jr.key(0), mean=MEAN, std=STD)


def _rotate(z, phi):
    """Rotate the frame by phi: spin-0 fixed, the spin-2 pair turns by 2 phi.

    In standardised coordinates the spin-2 pair is slots 3, 4 -- not the raw
    layout's 2, 3.
    """
    c, s = jnp.cos(2 * phi), jnp.sin(2 * phi)
    return z.at[3].set(c * z[3] - s * z[4]).at[4].set(s * z[3] + c * z[4])


def _rotate_sigma(sx, phi):
    """The same rotation acting on Sigma_X = [C00, C01, C11]."""
    c, s = jnp.cos(phi), jnp.sin(phi)
    r = jnp.array([[c, -s], [s, c]])
    full = jnp.array([[sx[0], sx[1]], [sx[1], sx[2]]])
    out = r @ full @ r.T
    return jnp.array([out[0, 0], out[0, 1], out[1, 1]])


def test_identity_at_zero_sigma():
    """No centroid uncertainty, no marginalisation.  Exact, not approximate."""
    out = LAYER.unmarginalize(Z, jnp.zeros(3))
    assert float(jnp.abs(out - Z).max()) == 0.0


def test_round_trip():
    """`marginalize` inverts `unmarginalize` to float precision.

    The fixed point contracts at the rate of T ~ 1e-3, so three passes is
    already exact; this is what licenses not doing a series inversion.
    """
    for scale in (1.0, 4.0, 16.0):
        sx = SIGMA_X * scale
        y, ld = LAYER.inverse_and_log_det(Z, sx)
        x, ld2 = LAYER.transform_and_log_det(y, sx)
        assert float(jnp.abs(x - Z).max()) < 1e-6, scale
        assert abs(float(ld + ld2)) < 1e-5, scale


def test_rotation_equivariance():
    """Rotating the galaxy AND Sigma_X rotates the answer.

    This is the whole basis of the layer's functional form: Sigma_X is a
    symmetric 2-tensor, so it splits into a spin-0 trace and a spin-2 traceless
    part, and only spin-covariant combinations of those with e may appear.
    """
    for phi in (0.3, 1.1, 2.7):
        a = _rotate(LAYER.unmarginalize(Z, SIGMA_X), phi)
        b = LAYER.unmarginalize(_rotate(Z, phi), _rotate_sigma(SIGMA_X, phi))
        assert float(jnp.abs(a - b).max()) < 1e-6, phi


def test_isotropic_sigma_x_leaves_no_preferred_direction():
    """With Sigma_X isotropic the spin-2 response is PARALLEL to e.

    Note t2 itself does not vanish: Sigma_u = sigma^2 J^-1 J^-T and J is
    elliptical, so an elliptical galaxy has an anisotropic centroid error --
    larger along the major axis, where the flux gradient is shallower.  But J
    depends only on the galaxy's own e, so t2 comes out along e, and so does
    every spin-2 structure built from it.  The marginalisation therefore rescales
    a galaxy's ellipticity without rotating it, and picks out no axis on the sky.

    That is the property worth pinning: a prior with a preferred direction reads
    out as additive shear, the same failure `RawMomentStandardize` was
    symmetrised to avoid.
    """
    _, t2 = _tensor(Z, SIGMA_X, MEAN, STD)
    e0_t = M[2] + 1j * M[3]
    assert abs(float((t2 * jnp.conj(e0_t)).imag
                     / (jnp.abs(t2) * jnp.abs(e0_t)))) < 1e-6

    out = LAYER.unmarginalize(Z, SIGMA_X)
    # The spin-2 slots are 3, 4 and carry zero mean, so z3 + i z4 is parallel to
    # the physical e; the response's shift must be parallel to it too.
    e0 = Z[3] + 1j * Z[4]
    de = (out[3] - Z[3]) + 1j * (out[4] - Z[4])
    # de is parallel to e: the cross product of the two vanishes.
    cross = float((de * jnp.conj(e0)).imag / (jnp.abs(de) * jnp.abs(e0) + 1e-30))
    assert abs(cross) < 1e-6, cross


def test_flux_homogeneity():
    """Moments are linear in the image, so scaling flux must scale the answer.

    Sigma_X is a NOISE covariance and does not scale with the galaxy, so this
    also pins that the layer reads Sigma_X only through the dimensionless
    T = (Mr/Mf) Sigma_u -- the reason its coefficient network has three inputs
    and not five.
    """
    for f in (0.1, 10.0, 1000.0):
        # Scaling flux at fixed shape shifts z0 by log10(f)/std0 and leaves the
        # other four alone, and leaves T unchanged only if Sigma_X scales with
        # the square of the flux.  So the whole map must COMMUTE with that shift.
        d = jnp.log10(f) / STD[0]
        a = LAYER.unmarginalize(Z, SIGMA_X).at[0].add(d)
        b = LAYER.unmarginalize(Z.at[0].add(d), SIGMA_X * f * f)
        assert float(jnp.abs(a - b).max()) < 1e-5, f


def test_displacement_covariance_matches_the_linearisation():
    """Sigma_u = J^-1 Sigma_X J^-T, with J bfd's own xyJacobian.

    Checked against an explicit build of J rather than the closed form, so a sign
    or transpose slip in either would show.
    """
    Mr, M1, M2 = float(M[1]), float(M[2]), float(M[3])
    j = -0.5 * np.array([[Mr + M1, M2], [M2, Mr - M1]])
    sx = np.array([[15941.4, 0.0], [0.0, 15941.4]])
    want = np.linalg.inv(j) @ sx @ np.linalg.inv(j).T
    got = np.asarray(displacement_covariance(M, SIGMA_X))
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-5, (got, want)


def test_response_is_first_order_in_sigma_x():
    """The shift is linear in Sigma_X where the expansion is valid.

    The marginalisation is second order in the displacement u and <u u> is linear
    in Sigma_X, so the leading response must be too.  Checked by halving Sigma_X
    and asking that the shift halves.
    """
    coeffs = LAYER.coeffs(Z[1], Z[2], Z[3] ** 2 + Z[4] ** 2)
    big = response(coeffs, Z, SIGMA_X * 1e-3, MEAN, STD) - Z
    small = response(coeffs, Z, SIGMA_X * 5e-4, MEAN, STD) - Z
    ratio = np.asarray(big / jnp.where(jnp.abs(small) > 0, small, jnp.inf))
    b = np.abs(np.asarray(big))
    # z is O(1) where raw moments were O(1e5), so the "is this slot actually
    # moving" cut has to be relative to the shift itself, not an absolute 1e-8.
    finite = np.isfinite(ratio) & (b > 1e-6 * b.max())
    assert finite.any(), (big, small)
    assert np.abs(ratio[finite] - 2.0).max() < 0.05, ratio[finite]


def test_saturates_instead_of_overflowing():
    """The unresolved tail must not produce inf.

    t0 reaches 0.10 on the imsims population -- 74x its median -- at the barely
    resolved end where the linearisation behind this layer has already failed.
    The map must stay finite and invertible there rather than returning inf and
    poisoning the whole batch.
    """
    for scale in (1e3, 1e6, 1e9):
        out = LAYER.unmarginalize(Z, SIGMA_X * scale)
        assert bool(jnp.all(jnp.isfinite(out))), scale
        # z is unbounded, so "stays positive" is not the statement any more --
        # what must hold is that the saturation caps the shift.
        assert float(jnp.abs(out - Z).max()) < 1.0, (scale, out - Z)


def test_gradients_are_finite_for_a_round_galaxy():
    """A round galaxy under isotropic Sigma_X is the commonest batch member and
    the one place a complex modulus would hand back NaN.  See `response`."""
    round_z = Z.at[3].set(0.0).at[4].set(0.0)
    g = jax.grad(lambda z: LAYER.unmarginalize(z, SIGMA_X).sum())(round_z)
    assert bool(jnp.all(jnp.isfinite(g))), g


def test_peeling_the_layer_off_is_exact_and_g_independent():
    """`bias.split_centroid` must reproduce the full flow's log_prob exactly.

    The whole point of the peel is that this layer is data-adjacent, so

        log p(x|g,S) = log p_rest(centroid(x,S)|g) + log|det d centroid/dx|

    with the second term independent of g.  That is what lets `bias.py` evaluate
    it once and fold it into an importance weight instead of dragging a 5x5
    jacfwd through a forward-over-reverse Hessian.  Both halves are checked: the
    identity holds, and the offset it introduces is the SAME at different g --
    so Q and R (eq. 12-13) are untouched.
    """
    import bulk
    from bias import centroid_transform, split_centroid

    m = np.array([[57026.8, 135254.4, -9006.5, -8736.6, 658762.5],
                  [12000.0, 40000.0, 900.0, -1500.0, 190000.0]])
    sx = np.tile(np.asarray(SIGMA_X), (2, 1))
    flow = bulk.build_flow(jr.key(0), m, shear=True, centroid=True)
    rest, layer = split_centroid(flow)
    assert layer is not None

    z, ld = centroid_transform(layer, m, sx)
    offsets, fulls = [], []
    for g in ([0.0, 0.0], [0.03, -0.02]):
        cond = jnp.tile(jnp.concatenate([jnp.array(g), jnp.asarray(SIGMA_X)]),
                        (2, 1))
        full = np.asarray(flow.log_prob(jnp.asarray(m), condition=cond))
        peeled = np.asarray(rest.log_prob(jnp.asarray(z), condition=cond)) + ld
        # `centroid_transform` casts to float32 to match the flow it feeds, and
        # the moments are ~1e5, so the identity is exact only to f32 -- a few
        # 1e-4 in a log-density of order tens.
        assert np.abs(full - peeled).max() < 1e-2, (g, full, peeled)
        offsets.append(full - peeled)
        fulls.append(full)

    # The claim that matters is not that the offset is constant to machine
    # precision, but that it carries no SHEAR signal: it must move by far less
    # than log_prob itself does between the two g, or Q and R would pick it up.
    d_offset = np.abs(offsets[0] - offsets[1]).max()
    d_full = np.abs(fulls[0] - fulls[1]).max()
    assert d_offset < 1e-3 * d_full, (d_offset, d_full)


def test_sampler_is_the_weighted_distribution():
    """`CopySampler` draws copies at exactly their eq.-36 weight.

    Built on a synthetic catalog with known weights rather than a real one, so it
    tests the sampler's arithmetic -- the shared cumulative sum, the per-galaxy
    bases, the edge clipping -- and not the physics, which imsims already checks.

    The expected frequency is the GLOBAL normalised weight, w_i / sum(all w).
    That is the whole content of drawing galaxies by their detection probability
    rather than uniformly: a galaxy whose copy weights sum to less than one is
    less likely to be detected at all, and must be sampled less often to match.
    """
    from centroid import CopySampler

    rng = np.random.default_rng(0)
    n_gal, per_gal = 40, 12
    copies = np.zeros(n_gal * per_gal,
                      dtype=[("gal", "i4"), ("moments", "f8", (5,)),
                             ("xy", "f8", (2,)), ("da", "f8")])
    copies["gal"] = np.repeat(np.arange(n_gal), per_gal)
    # Distinct first moments give distinct weights; tag each copy by its index so
    # the draw can be identified from the moments alone.
    copies["xy"] = rng.normal(0.0, 1.0, (copies.size, 2))
    copies["da"] = 1.0
    copies["moments"][:, 1] = 2.0        # Mr, so |J| > 0
    copies["moments"][:, 0] = np.arange(copies.size)

    sigma_x = np.array([1.0, 0.0, 1.0])
    want = np.exp(np.asarray(__import__("imsims.copies", fromlist=["x"])
                             .log_weights(copies, sigma_x)))
    sampler = CopySampler(copies, sigma_x)

    drawn, _ = sampler.draw(np.random.default_rng(1), 400000)
    counts = np.bincount(drawn[:, 0].astype(int), minlength=copies.size)
    # Galaxy by detection probability, then copy by weight, so the two
    # normalisations telescope into one global one.
    expect = want / want.sum()
    got = counts / counts.sum()
    assert np.abs(got - expect).max() < 5e-4, np.abs(got - expect).max()

    # And it must actually be sensitive to a galaxy carrying less total weight:
    # halve one galaxy's copies and its share of the draws must halve too.
    copies["da"][:per_gal] = 0.5
    s2 = CopySampler(copies, sigma_x)
    d2, _ = s2.draw(np.random.default_rng(2), 400000)
    share = (d2[:, 0] < per_gal).mean()
    w2 = np.exp(np.asarray(__import__("imsims.copies", fromlist=["x"])
                           .log_weights(copies, sigma_x)))
    assert abs(share - w2[:per_gal].sum() / w2.sum()) < 2e-3, share


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
