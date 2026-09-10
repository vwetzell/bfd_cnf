"""Self-checks for models/centroid.py -- the centroid-marginalisation layer.

Run: `python -m tests.test_centroid` or pytest.

These check the symmetries the layer's closed form was DERIVED from, and the
exactness the derivation claims, so a failure here means the derivation or the
code departed from it, not that a fit is poor -- there is no fit any more, see
`models/centroid.py`'s module docstring.  The quality of the underlying
Gaussian-profile-in-k approximation is `centroid.py check`'s job, against the
catalog.
"""

import jax

# These are exact-symmetry checks, so they run in float64.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp        # noqa: E402  -- must follow the x64 switch
import jax.random as jr        # noqa: E402
import numpy as np             # noqa: E402

from models.bijections import POINT_SOURCE, POINT_SOURCE_MC
from models.centroid import (CentroidMarginalize, displacement_covariance,
                             raw_from_standard, split_condition, _transport)

# A representative bulge+disc galaxy and the imsims Sigma_X at the nominal noise.
M = jnp.array([57026.8, 135254.4, -9006.5, -8736.6, 658762.5])
SIGMA_X = jnp.array([15941.4, 0.0, 15941.4])
# Genuinely anisotropic (nonzero C01), for the tests that must not collapse
# onto the isotropic special case where R and Sigma_u happen to commute.
SIGMA_X_ANISO = jnp.array([15941.4, 4000.0, 9000.4])

# The layer acts on STANDARDISED coordinates, so the tests need the
# standardiser's constants -- the layer carries a frozen copy to rebuild a
# physical scale for Sigma_u (see `raw_from_standard`).  These are
# representative of what `build_flow` measures on bulgedisc; the exact values
# do not matter to a symmetry test, but two things do: the spin-2 slots must
# share a std and have ZERO mean, which is the symmetrisation that keeps the
# chart isotropic.
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
LAYER = CentroidMarginalize(mean=MEAN, std=STD)


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


def _assert_rotation_equivariant(layer, z, sigma_x):
    """Rotating the galaxy AND Sigma_X must rotate `unmarginalize`'s answer."""
    for phi in (0.3, 1.1, 2.7):
        a = _rotate(layer.unmarginalize(z, sigma_x), phi)
        b = layer.unmarginalize(_rotate(z, phi), _rotate_sigma(sigma_x, phi))
        assert float(jnp.abs(a - b).max()) < 1e-8, phi


def test_identity_at_zero_sigma():
    """No centroid uncertainty, no marginalisation.  Exact within `_transport`
    itself (P = 0 makes every term in it trivially exact) -- but this layer
    now round-trips through RAW moment space (`raw_from_standard` then
    `standard_from_raw`), so machine roundoff (~1e-15 in float64) survives
    even at Sigma_X = 0, unlike the old architecture's pure z-space additive
    shift.  1e-10 is far above that roundoff and far below any real physics."""
    out = LAYER.unmarginalize(Z, jnp.zeros(3))
    assert float(jnp.abs(out - Z).max()) < 1e-10


def test_round_trip():
    """`marginalize` inverts `unmarginalize` to float precision.

    Both directions are now closed-form (see module docstring): `unmarginalize`
    solves R from the DATA point's own moments and damps by -Sigma_u,
    `marginalize` solves R from the BASE point and damps by +Sigma_u, and the
    two are exact inverses within the Gaussian-in-k ansatz -- no fixed-point
    iteration, so this should hold near machine precision, not just "small
    enough that 3 passes converged" as the old architecture's docstring said.
    Checked at an ANISOTROPIC Sigma_X, where R and Sigma_u do not commute and
    a same-sign matrix-ordering bug would show (see HANDOFF.md, 2026-08-27).
    """
    for scale in (1.0, 4.0, 16.0):
        for sx in (SIGMA_X * scale, SIGMA_X_ANISO * scale):
            y, ld = LAYER.inverse_and_log_det(Z, sx)
            x, ld2 = LAYER.transform_and_log_det(y, sx)
            assert float(jnp.abs(x - Z).max()) < 1e-5, (scale, sx)
            assert abs(float(ld + ld2)) < 1e-4, (scale, sx)


def test_rotation_equivariance():
    """Rotating the galaxy AND Sigma_X rotates the answer.

    This is the whole basis of the layer's functional form: Sigma_X is a
    symmetric 2-tensor, so it splits into a spin-0 trace and a spin-2 traceless
    part, and R (built the same way from the galaxy's own M1, M2) rotates the
    same way, so the whole transport commutes with a joint rotation.
    """
    _assert_rotation_equivariant(LAYER, Z, SIGMA_X_ANISO)


def test_equivariance_survives_asymmetric_stored_chart_constants():
    """The layer must symmetrise its OWN frozen chart, not trust what it was given.

    Every other rotation test here hands the layer a chart that is already
    isotropic (`MEAN`, `STD` above), so none of them can see the failure this
    guards: `bulk.build_flow` and `bulk.sync_chart_constants` both copy
    `chart.mean`/`chart.std` RAW, while `RawMomentStandardize` only ever reads
    `_effective()` -- and a trained checkpoint's raw spin-2 pair DRIFTS APART
    (1.00133 at init, 1.01458 on `centroid_v10.eqx`).  Left raw, the layer
    decoded M1 and M2 with two different scales and a nonzero mean, and its map
    was not rotation-equivariant: 4.3e-5 at p50 against a 8.6e-7 float32 floor.

    The stored values here are exaggerated (5% apart) so the assertion has
    headroom over the tolerance rather than sitting on it.
    """
    skew_mean = MEAN.at[3].set(0.004).at[4].set(-0.011)
    skew_std = STD.at[3].set(0.041).at[4].set(0.039)
    layer = CentroidMarginalize(mean=skew_mean, std=skew_std)

    # `chart()` is `_effective`: zero spin-2 mean, one shared spin-2 scale.
    mean, std = layer.chart()
    assert float(jnp.abs(mean[3:]).max()) == 0.0
    assert float(std[3]) == float(std[4])
    assert abs(float(std[3]) - float(jnp.sqrt(0.5 * (skew_std[3] ** 2
                                                     + skew_std[4] ** 2)))) < 1e-12
    # and the spin-0 slots are untouched.
    assert bool(jnp.array_equal(mean[:3], skew_mean[:3]))
    assert bool(jnp.array_equal(std[:3], skew_std[:3]))

    # The map is equivariant DESPITE the asymmetric stored constants, which is
    # the property the fix exists for.
    z = _to_z(M)
    _assert_rotation_equivariant(layer, z, SIGMA_X_ANISO)


def test_isotropic_sigma_x_leaves_no_preferred_direction():
    """With Sigma_X isotropic the spin-2 response is PARALLEL to e.

    Sigma_u = sigma^2 J^-1 J^-T does not itself vanish or go isotropic just
    because Sigma_X is isotropic -- J is elliptical, so an elliptical galaxy's
    centroid error is anisotropic, larger along the major axis where the flux
    gradient is shallower.  But J (and so R) depends only on the galaxy's own
    e, so R and Sigma_u SHARE that eigenbasis whenever Sigma_X is isotropic,
    and the marginalisation rescales the galaxy's ellipticity without
    rotating it -- the property `RawMomentStandardize` was symmetrised to
    protect, and the one place a matrix-ordering slip (`R @ inv(I+P)` vs.
    `inv(I+P) @ R`) would NOT show up, since the two coincide when R, Sigma_u
    commute (see `test_round_trip` for the anisotropic check that catches it).
    """
    sigma_u = displacement_covariance(M, SIGMA_X)
    t2 = jax.lax.complex(0.5 * (sigma_u[0, 0] - sigma_u[1, 1]), sigma_u[0, 1])
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

    Sigma_X is a NOISE covariance and does not scale with the galaxy: scaling
    Mf by f at fixed shape scales Mr, M1, M2, Mc by f too (same shape,
    f times brighter), under which R is invariant (both the numerator matrix
    and Mf scale by f) and Sigma_u is invariant only if Sigma_X scales as f^2
    (J scales by f, J^-1 by 1/f, so J^-1 Sigma_X J^-T needs Sigma_X ~ f^2 to
    stay fixed) -- so the whole transport must commute with that joint shift.
    """
    for f in (0.1, 10.0, 1000.0):
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


def test_transport_is_exact_for_a_gaussian_in_k():
    """The one case the ansatz claims exactness for, checked to machine precision.

    `models/centroid.py` models `W(k) I(k)` as Gaussian in k -- "exact for a
    single Gaussian profile times a Gaussian-shaped weight".  Then
    `W(k)I(k) = c exp(-1/2 k^T A k)` and bfd's kernels (1, k^2, kx^2-ky^2,
    2 kx ky, k^4) give, with `B = A^-1`,

        Mf = 2 pi c / sqrt(det A)
        Mr = Mf tr B,   M1 = Mf (B00 - B11),   M2 = 2 Mf B01
        Mc = Mf (3 B00^2 + 3 B11^2 + 2 B00 B11 + 4 B01^2)

    and damping by `exp(-1/2 k^T Sigma_u k)` is EXACTLY `A -> A + Sigma_u`, so
    the marginalised moments are the same formulas at `A + Sigma_u`.  No
    quadrature, no approximation -- ground truth in closed form.

    Worth pinning because the layer's measured 25-30% ellipticity-response
    undershoot on `copies_gauss2_deep` looked like it could be an algebra slip
    in the spin-2 branch (the `(I+P)^-1 R` vs `R (I+P)^-1` ordering, or a factor
    in `M1'`/`M2'`).  It is not: this passes at 1e-14, so the closed form is
    right and the whole undershoot is `W(k) I(k)` not being Gaussian -- which it
    is not even for a single-Gaussian GALAXY, since the weight is
    KBlackmanHarris rather than a Gaussian (HANDOFF.md, 2026-08-28).
    """
    rng = np.random.default_rng(0)

    def moments_from_A(A):
        B = np.linalg.inv(A)
        mf = 2 * np.pi / np.sqrt(np.linalg.det(A))
        return np.array([mf, mf * np.trace(B), mf * (B[0, 0] - B[1, 1]),
                         2 * mf * B[0, 1],
                         mf * (3 * B[0, 0] ** 2 + 3 * B[1, 1] ** 2
                               + 2 * B[0, 0] * B[1, 1] + 4 * B[0, 1] ** 2)])

    def rand_spd(scale):
        L = np.tril(rng.normal(size=(2, 2)))
        L[np.diag_indices(2)] = np.abs(L[np.diag_indices(2)]) + 0.6
        return scale * (L @ L.T)

    worst = 0.0
    for _ in range(100):
        A = rand_spd(rng.uniform(0.5, 3.0))
        m = moments_from_A(A)
        # `_transport` forms Sigma_u = J^-1 Sigma_X J^-T itself, so invert that
        # to hand it the Sigma_X producing the Sigma_u wanted.
        su = rand_spd(rng.uniform(0.002, 0.05) * np.trace(A) / 2)
        j = -0.5 * np.array([[m[1] + m[2], m[3]], [m[3], m[1] - m[2]]])
        sx = j @ su @ j.T
        want = moments_from_A(A + su)
        got = np.asarray(_transport(
            jnp.asarray(m), jnp.asarray([sx[0, 0], sx[0, 1], sx[1, 1]]), 1.0))
        worst = max(worst, float(np.abs(got / want - 1).max()))
    assert worst < 1e-10, worst


def test_transport_is_first_order_in_sigma_x_for_small_sigma():
    """The shift is linear in Sigma_X in the small-Sigma_X limit.

    The marginalisation is second order in the displacement u and <u u> is
    linear in Sigma_X, so the leading response must be too, and the closed
    form's own Taylor expansion (module docstring) confirms it -- checked by
    halving Sigma_X and asking that the shift halves.
    """
    big = LAYER.unmarginalize(Z, SIGMA_X * 1e-3) - Z
    small = LAYER.unmarginalize(Z, SIGMA_X * 5e-4) - Z
    ratio = np.asarray(big / jnp.where(jnp.abs(small) > 0, small, jnp.inf))
    b = np.abs(np.asarray(big))
    # z is O(1) where raw moments were O(1e5), so the "is this slot actually
    # moving" cut has to be relative to the shift itself, not an absolute 1e-8.
    finite = np.isfinite(ratio) & (b > 1e-6 * b.max())
    assert finite.any(), (big, small)
    assert np.abs(ratio[finite] - 2.0).max() < 0.05, ratio[finite]


def test_saturates_instead_of_overflowing():
    """The unresolved tail must not produce inf or a broken (non-invertible)
    transport.

    On the imsims population T reaches ~0.57 at the barely resolved end
    (HANDOFF.md, 2026-08-26); these scales push Sigma_X to 100x-10000x
    nominal, i.e. T up to ~50, ~100x past the largest T ever measured --
    generous stress margin, not the adversarial 1e9x the old (linear-in-T)
    architecture's own overflow risk needed.  Past this the DATA -> BASE
    direction can genuinely have no valid preimage (undoing that much
    marginalisation would need a base more resolved than the chart's own
    point-source ceiling allows) -- a real boundary of the closed form's
    domain, not a bug this test tries to paper over.
    """
    # Finiteness only -- NOT round-trip precision (that is `test_round_trip`'s
    # job, at scales where the safety floor is not engaged).  Once the floor
    # is actively clamping trace(P), `unmarginalize` is no longer an exact
    # analytic inverse of `marginalize` by construction: it is projecting an
    # otherwise-invalid preimage back into the domain the chart can express,
    # not solving the ansatz exactly any more.  That is the floor doing its
    # job, not a precision bug.
    for scale in (1e2, 1e3, 1e4):
        out = LAYER.unmarginalize(Z, SIGMA_X * scale)
        assert bool(jnp.all(jnp.isfinite(out))), scale


def test_gradients_are_finite_for_a_round_galaxy():
    """A round galaxy under isotropic Sigma_X is the commonest batch member,
    and the one place R and Sigma_u are simultaneously diagonalisable with a
    repeated eigenvalue -- the kind of degeneracy that can make some matrix
    decompositions NaN their gradient.  `_transport` never decomposes R or
    Sigma_u individually (only `jnp.trace`/`jnp.linalg.solve`/`det` on their
    product), so this should stay smooth."""
    round_z = Z.at[3].set(0.0).at[4].set(0.0)
    g = jax.grad(lambda z: LAYER.unmarginalize(z, SIGMA_X).sum())(round_z)
    assert bool(jnp.all(jnp.isfinite(g))), g


def test_peeling_the_layer_off_is_exact_for_a_standalone_layer():
    """`bias.split_centroid` must reproduce the full flow's log_prob exactly
    for a standalone (3,)-conditioned layer, and must REFUSE to peel a
    chained (5,)-conditioned one.

    The refusal is NOT because this layer reads g -- it doesn't (module
    docstring, "No shear-conditioning") -- it is an empirically-required
    safety restriction: peeling a chained layer reproduces `flow.log_prob`
    exactly (matches to float32 roundoff, checked directly), but `bias.py`'s
    actual Q/R -- computed via `jax.grad`/`jax.hessian` w.r.t. g THROUGH the
    reconstructed `rest` sub-chain -- come out wrong regardless of what this
    layer's transport computes (even with `_transport` forced to a literal
    identity).  Root cause not yet found -- see `bias.split_centroid`'s
    docstring and HANDOFF.md, 2026-08-27.
    """
    import bulk
    from bias import centroid_transform, split_centroid

    m = np.array([[57026.8, 135254.4, -9006.5, -8736.6, 658762.5],
                  [12000.0, 40000.0, 900.0, -1500.0, 190000.0]])
    sx = np.tile(np.asarray(SIGMA_X), (2, 1))
    rng = np.random.default_rng(0)
    mf = 10 ** rng.uniform(3.5, 4.5, 500)
    mr = mf * rng.uniform(2.0, 3.4, 500)
    pop = np.stack([mf, mr, mr * rng.normal(0, 0.05, 500),
                    mr * rng.normal(0, 0.05, 500),
                    mr * rng.uniform(2.0, 6.0, 500)], axis=-1)

    # Chained with shear: must be refused, even though this layer itself is
    # g-independent -- see docstring above.
    chained = bulk.build_flow(jr.key(0), pop, shear=True, centroid=True)
    rest, layer = split_centroid(chained)
    assert layer is None, "a chained (5,)-conditioned layer must not be peeled"
    assert rest is chained

    # Standalone: no shear layer, (3,) condition, peelable and exact.
    solo = bulk.build_flow(jr.key(0), pop, centroid=True)
    rest, layer = split_centroid(solo)
    assert layer is not None, "a (3,)-conditioned layer is still peelable"

    z, ld = centroid_transform(layer, m, sx)
    cond = jnp.tile(jnp.asarray(SIGMA_X), (2, 1))
    full = np.asarray(solo.log_prob(jnp.asarray(m), condition=cond))
    peeled = np.asarray(rest.log_prob(jnp.asarray(z), condition=cond)) + ld
    # `centroid_transform` casts to float32 to match the flow it feeds, and
    # the moments are ~1e5, so the identity is exact only to f32 -- a few
    # 1e-4 in a log-density of order tens.
    assert np.abs(full - peeled).max() < 1e-2, (full, peeled)


def test_split_condition():
    """The splitter must not mistake Sigma_X for a shear, on either width."""
    g3, s3 = split_condition(jnp.asarray(SIGMA_X))
    assert np.allclose(np.asarray(g3), 0.0)
    g5, s5 = split_condition(jnp.concatenate([jnp.array([0.03, -0.02]),
                                              jnp.asarray(SIGMA_X)]))
    assert np.allclose(np.asarray(g5), [0.03, -0.02])
    assert np.allclose(np.asarray(s3), np.asarray(s5))


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

    drawn, _, _, _ = sampler.draw(np.random.default_rng(1), 400000)
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
    d2, _, _, _ = s2.draw(np.random.default_rng(2), 400000)
    share = (d2[:, 0] < per_gal).mean()
    w2 = np.exp(np.asarray(__import__("imsims.copies", fromlist=["x"])
                           .log_weights(copies, sigma_x)))
    assert abs(share - w2[:per_gal].sum() / w2.sum()) < 2e-3, share


def test_spin2_coeff_is_a_no_op_at_one_and_is_linear_in_the_departure():
    """`c_spin2` moves dM1/dM2 through tr(Sigma_u) and dMr through its traceless
    part -- the two contractions of the SAME k^4 spin-2 bracket.

    1.0 is the Gaussian/Wick value, so it must reproduce the bare ansatz bit
    for bit.  Away from it the correction is first order by construction, so
    the shift must be exactly linear in (c - 1): a quadratic term would mean
    the bracket had been wired into the resummed backbone by mistake.
    """
    base = _transport(M, SIGMA_X, 1.0)
    assert float(jnp.abs(_transport(M, SIGMA_X, 1.0, 1.0) - base).max()) == 0.0

    d1 = _transport(M, SIGMA_X_ANISO, 1.0, 1.5) - _transport(M, SIGMA_X_ANISO, 1.0)
    d2 = _transport(M, SIGMA_X_ANISO, 1.0, 2.0) - _transport(M, SIGMA_X_ANISO, 1.0)
    assert float(jnp.abs(d2 - 2.0 * d1).max()) < 1e-6 * float(jnp.abs(M).max())
    # Mf carries no k^4 bracket at all, so it must not move.
    assert float(jnp.abs(d2[0]) / M[0]) < 1e-12
    # Mr and the spin-2 pair must all move, or the contraction is wired wrong.
    # Mr's leverage is ~35x weaker than the spin-2 pair's: it sees the bracket
    # only through Sigma_u's traceless part, which is |s|/s0 ~ 2|e| of the
    # trace the spin-2 channels see.
    for j in (1, 2, 3):
        assert float(jnp.abs(d2[j]) / jnp.abs(M[j])) > 1e-8, j


def test_spin2_coeff_is_real_so_the_layer_stays_parity_even():
    """A reflection must commute with the layer.

    `c_spin2` is real precisely so it cannot rotate the k^4 spin-2 moment away
    from the k^2 one; a complex coefficient would survive this only for a
    parity-violating population.  Reflect about the x axis: M2 -> -M2 and
    Sigma_X's off-diagonal flips.
    """
    flip = lambda z: z.at[4].multiply(-1.0)
    sx = jnp.array([SIGMA_X_ANISO[0], -SIGMA_X_ANISO[1], SIGMA_X_ANISO[2]])
    layer = CentroidMarginalize(mean=MEAN, std=STD)
    a = flip(layer.unmarginalize(Z, SIGMA_X_ANISO))
    b = layer.unmarginalize(flip(Z), sx)
    assert float(jnp.abs(a - b).max()) < 1e-8


def test_round_trip_survives_a_spin2_coefficient():
    """A departure from the Gaussian bracket leaves the layer invertible.

    The two directions were never EXACT inverses -- Sigma_u is re-solved from
    whichever point is handed in -- so what matters is that the correction adds
    nothing on top of the round-trip error the bare ansatz already carries.
    Checked at an anisotropic Sigma_X, where R and Sigma_u do not commute.
    """
    layer = CentroidMarginalize(mean=MEAN, std=STD)
    import equinox as _eqx
    for c in (0.5, 2.0):
        bumped = _eqx.tree_at(lambda l: l.coeff.layers[-1].bias, layer,
                              jnp.full((1,), c - 1.0))
        for sx in (SIGMA_X, SIGMA_X_ANISO):
            y, ld = bumped.inverse_and_log_det(Z, sx)
            x, ld2 = bumped.transform_and_log_det(y, sx)
            assert float(jnp.abs(x - Z).max()) < 1e-3, (c, sx)
            assert abs(float(ld + ld2)) < 1e-3, (c, sx)


def test_coeff_net_starts_at_the_gaussian_value():
    """Zeroed output layer => both coefficients are 1 exactly, whatever the
    hidden init."""
    layer = CentroidMarginalize(mean=MEAN, std=STD)
    m = raw_from_standard(Z, MEAN, STD)
    assert [float(v) for v in layer.bracket_coeffs(m, MEAN, STD)] == [1.0, 1.0]


def test_spin4_moves_only_the_spin2_channels():
    """`c_spin4` may touch M1 and M2 and nothing else.

    A trace cannot see a spin-4 moment, so `K` appears in neither dMf nor dMr.
    If it moves either, the bracket has been contracted with the wrong part of
    Sigma_u.  It also has to vanish on a ROUND galaxy, where `K` itself is 0.

    Mc is the one exception, and only at second order: it is corrected
    DIFFERENTIALLY through `mc_ansatz_out`, which is a function of M1_out and
    M2_out, so moving the spin-2 channels drags it by O(|e| dM1 / Mf).  That is
    the ansatz staying self-consistent, not a spin-4 bracket of its own, so it
    is bounded well below the spin-2 move rather than pinned at zero.
    """
    base = _transport(M, SIGMA_X_ANISO, 1.0)
    got = _transport(M, SIGMA_X_ANISO, 1.0, 1.0, 2.5)
    assert float(jnp.abs(got[0] - base[0]) / base[0]) < 1e-12
    assert float(jnp.abs(got[1] - base[1]) / base[1]) < 1e-12
    assert float(jnp.abs(got[4] - base[4]) / base[4]) < 1e-6
    for j in (2, 3):
        assert float(jnp.abs(got[j] - base[j]) / jnp.abs(M[j])) > 1e-9, j

    round_m = M.at[2].set(0.0).at[3].set(0.0)
    a = _transport(round_m, SIGMA_X_ANISO, 1.0)
    b = _transport(round_m, SIGMA_X_ANISO, 1.0, 1.0, 2.5)
    assert float(jnp.abs(a - b).max()) == 0.0


def test_spin4_leverage_is_e_squared_below_spin2():
    """`K`'s pull on the spin-2 channels is |e|^2 times `N`'s, up to O(1).

    This is why the two coefficients are near-degenerate on a circular-PSF
    population, and it is worth pinning: if a refactor ever gave `K` the
    trace contraction by mistake, this ratio would jump by ~1/|e|^2.
    """
    d2 = _transport(M, SIGMA_X_ANISO, 1.0, 2.0) - _transport(M, SIGMA_X_ANISO, 1.0)
    d4 = (_transport(M, SIGMA_X_ANISO, 1.0, 1.0, 2.0)
          - _transport(M, SIGMA_X_ANISO, 1.0))
    e2 = float((M[2] ** 2 + M[3] ** 2) / M[1] ** 2)
    ratio = float(jnp.abs(d4[2]) / jnp.abs(d2[2]))
    assert 0.05 * e2 < ratio < 20.0 * e2, (ratio, e2)


def test_spin2_coeff_keeps_the_rotation_equivariance():
    """The coefficient sees only rotation-invariant inputs, so it cannot
    introduce a direction."""
    layer = CentroidMarginalize(mean=MEAN, std=STD)
    import equinox as _eqx
    layer = _eqx.tree_at(lambda l: l.coeff.layers[-1].bias, layer,
                         jnp.full((1,), 0.4))
    _assert_rotation_equivariant(layer, Z, SIGMA_X_ANISO)


def test_flux_sas_is_the_identity_by_default_and_inverts_when_set():
    """The flux warp is off unless asked for, and is a bijection when on.

    `flux_sas=None` must reproduce the plain log10 chart EXACTLY, so every
    checkpoint written before the warp existed still means what it used to.
    With it on, the layer's frozen chart copy has to invert to float precision
    -- `raw_from_standard` and `standard_from_raw` are hand-mirrored from
    `RawMomentStandardize`, so a warp added to one and not the other is a
    silent, population-wide chart mismatch.
    """
    from models.centroid import raw_from_standard, standard_from_raw
    P = (3.4094, 0.5104, 0.708531, 0.487389)

    # off by default: byte-identical round trip through the mirrored chart
    z_plain = standard_from_raw(M, MEAN, STD)
    assert float(jnp.abs(standard_from_raw(M, MEAN, STD, None) - z_plain).max()) == 0.0

    # on: still an exact bijection, and actually different from the plain chart
    z = standard_from_raw(M, MEAN, STD, P)
    assert float(jnp.abs(z - z_plain).max()) > 1e-3
    back = raw_from_standard(z, MEAN, STD, P)
    assert float(jnp.abs(back / M - 1.0).max()) < 1e-5

    # and the layer carrying it round-trips like the bare one does
    layer = CentroidMarginalize(mean=MEAN, std=STD, flux_sas=P)
    for sx in (SIGMA_X, SIGMA_X_ANISO):
        y, ld = layer.inverse_and_log_det(z, sx)
        x, ld2 = layer.transform_and_log_det(y, sx)
        assert float(jnp.abs(x - z).max()) < 1e-4, sx
        assert abs(float(ld + ld2)) < 1e-3, sx


def test_flux_sas_log_det_matches_autodiff():
    """The chart's hand-written log-det must equal the Jacobian's, warp on.

    The warp adds one analytic term to `lad_geom`; getting it wrong would not
    break any round trip, it would just quietly bias every density the flow
    reports.  Autodiff is the only honest check.
    """
    from models.bijections import RawMomentStandardize
    P = (3.4094, 0.5104, 0.708531, 0.487389)
    ch = RawMomentStandardize(mean=MEAN, std=STD, flux_sas=P)
    for scale in (0.5, 1.0, 2.0):
        m = M * jnp.array([scale, scale, scale, scale, scale])
        _, ld = ch.transform_and_log_det(m)
        j = jax.jacfwd(lambda v: ch.transform_and_log_det(v)[0])(m)
        assert abs(float(jnp.linalg.slogdet(j)[1] - ld)) < 1e-6, scale


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
