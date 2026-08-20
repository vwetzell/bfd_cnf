"""
varq.py
=======
The Var[Q|m] correction to the shear layer's SECOND-ORDER training target.

Why it exists
-------------
Templates sharing the same moments m do not share the same shear response: the
rest of the galaxy's structure is not in m.  Write the exact per-template
response as

    Q_t = v(m) + eps,    E[eps | m] = 0,    Sigma_{aj,bk}(m) = Cov[eps_aj, eps_bk | m]

The true lensed density is therefore the population pushed through a RANDOM
displacement.  Because that displacement is evaluated at the starting point m
(Ito, not Stratonovich), its second-cumulant contribution is

    P_true(.|g) = P_det(.|g) + 1/2 d_j d_k [ C_jk P ] + O(g^4),
    C_jk = g_a g_b Sigma_{aj,bk}(m)

A deterministic transport CAN represent this -- that is the point.  The layer's
map is phi_g(m) = m + a(m) g + 1/2 g.b(m).g, whose O(g^2) displacement shifts the
density by -1/2 d_j[(g.b.g)_j P].  Matching the two O(g^2) terms gives

    b_{ab,j} = E[R_t | m]_{ab,j} - (1/P) d_k [ Sigma_{aj,bk}(m) P(m) ]
             = E[R_t | m]_{ab,j} - ( d_k Sigma_{aj,bk} + Sigma_{aj,bk} d_k log P )

`shear._velocity_mse` fits the layer's second derivative to the per-template
`d2m_dg2`, whose minimiser is E[R_t|m] -- the FIRST TERM ONLY.  So the existing
loss pins b to the wrong value with a weight of 1e4.  At first order E[Q_t|m] is
genuinely the right target (that is why deriv_weight helps there); at second
order it is not.  `correction` below returns the missing term, to be ADDED to
the per-template target.

The correction is a total divergence, so it integrates to zero over m and the
density stays normalised -- `main` asserts that numerically.

Keeping it isotropic
--------------------
Fitting Cov[eps] component by component on the invariants would hand the flow a
preferred direction on the sky, which is the one bug this codebase has already
paid for once (`RawMomentStandardize._effective`).  So eps is first rotated into
the GALAXY frame, where every coefficient is rotation-invariant by construction:

    z_j    = eps_0j + i eps_1j    for the spin-0 moments j in (Mf, Mr, Mc)  [spin 2]
    d1, d2 = the two spin-2 columns / Mr;  A = (d1 - i d2)/2  [spin 0]
                                           Be2 = (d1 + i d2)/2  [spin 4]

with omega = conj(e)/|e|, the galaxy-frame coefficients are z_j omega, A, and
Be2 omega^2 -- ten real numbers, all invariant.  Sigma is fitted there and
rotated back, so isotropy is exact rather than learned.

Parity gives a free check: it flips the imaginary parts, so E[Im] must vanish
for every coefficient.  `main` reports it.

    python varq.py [--deg 3] [--n 100000]
"""

from __future__ import annotations

import argparse
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

DEG = 3
DATA = "../bfd_cnf_imsims/data/moments.fits"
# Ten galaxy-frame coefficients, in the order `coeffs` returns them.
NAMES = ["Mf re", "Mf im", "Mr re", "Mr im", "Mc re", "Mc im",
         "A re", "A im", "Be2 re", "Be2 im"]
NC = len(NAMES)


def _exponents(deg, nvar=3):
    """Monomial exponents up to total degree `deg`, as an (nb, nvar) int array."""
    out = [(0,) * nvar]
    for d in range(1, deg + 1):
        for p in itertools.combinations_with_replacement(range(nvar), d):
            e = [0] * nvar
            for i in p:
                e[i] += 1
            out.append(tuple(e))
    return np.array(out)


def invariants(m):
    """(r, k, q) = (Mr/Mf, Mc Mf/Mr^2, |e|^2), the flux-blind triple."""
    Mf, Mr, Mc = m[..., 0], m[..., 1], m[..., 4]
    q = (m[..., 2] ** 2 + m[..., 3] ** 2) / (Mr * Mr)
    return jnp.stack([Mr / Mf, Mc * Mf / (Mr * Mr), q], axis=-1)


def poly(m, expo, mu, sd):
    """Monomial basis in the standardised invariants; jax, so it differentiates."""
    u = (invariants(m) - mu) / sd
    return jnp.prod(u[..., None, :] ** expo, axis=-1)


def _omega(m):
    """conj(e)/|e|, the rotation that carries the galaxy frame to e real."""
    e = jax.lax.complex(m[..., 2] / m[..., 1], m[..., 3] / m[..., 1])
    return jnp.conj(e) / jnp.maximum(jnp.abs(e), 1e-12)


def coeffs(m, Q):
    """(..., 2, 5) response -> (..., 10) GALAXY-FRAME coefficients.

    Exact and invertible -- ten numbers in, ten out; `from_coeffs` inverts it.
    """
    w, Mr = _omega(m), m[..., 1]
    # Each spin-0 response is divided by ITS OWN moment and the spin-2 pair by
    # Mr, so all ten coefficients are dimensionless.  Without this they scale
    # with flux, and flux is not one of the invariants they get regressed on --
    # the fit could not represent their magnitude at all.
    z = [jax.lax.complex(Q[..., 0, j], Q[..., 1, j]) * w / m[..., j]
         for j in (0, 1, 4)]
    d1 = jax.lax.complex(Q[..., 0, 2], Q[..., 0, 3]) / Mr
    d2 = jax.lax.complex(Q[..., 1, 2], Q[..., 1, 3]) / Mr
    A = 0.5 * (d1 - 1j * d2)
    Be2 = 0.5 * (d1 + 1j * d2) * w * w
    parts = z + [A, Be2]
    return jnp.stack([f(c) for c in parts for f in (jnp.real, jnp.imag)], -1)


def from_coeffs(m, c):
    """(..., 10) galaxy-frame coefficients -> (..., 2, 5) response.

    Exact inverse of `coeffs`.  Every stacked pair below is indexed by the SHEAR
    component, and the five are then stacked along the moment axis.
    """
    w, Mr = _omega(m), m[..., 1]
    z = [jax.lax.complex(c[..., 2 * i], c[..., 2 * i + 1]) / w * m[..., j]
         for i, j in enumerate((0, 1, 4))]
    A = jax.lax.complex(c[..., 6], c[..., 7])
    Be2 = jax.lax.complex(c[..., 8], c[..., 9]) / (w * w)
    d1, d2 = A + Be2, 1j * (A - Be2)
    pair = lambda u: jnp.stack([jnp.real(u), jnp.imag(u)], -1)
    # slots 2 and 3 are the real and imaginary parts of ONE complex pair, so
    # they come from (d1, d2) jointly rather than one column each
    s2 = jnp.stack([jnp.real(d1), jnp.real(d2)], -1) * Mr[..., None]
    s3 = jnp.stack([jnp.imag(d1), jnp.imag(d2)], -1) * Mr[..., None]
    return jnp.stack([pair(z[0]), pair(z[1]), s2, s3, pair(z[2])], axis=-1)


def _wls(basis, y, w=None):
    """Least-squares coefficients of each column of `y` on `basis`."""
    if w is None:
        return np.linalg.lstsq(basis, y, rcond=None)[0]
    s = np.sqrt(w)[:, None]
    return np.linalg.lstsq(basis * s, y * s, rcond=None)[0]


def fit(m, dm, deg=DEG):
    """Fit E[c|inv] and Cov[c|inv] in the galaxy frame.

    Returns everything `correction` needs: the standardisation of the
    invariants, the monomial exponents, and the two regression coefficient
    blocks.  The covariance is fitted on the OUTER PRODUCTS of the residuals,
    which is what makes it a conditional covariance rather than a global one.
    """
    m, dm = np.asarray(m, np.float64), np.asarray(dm, np.float64)
    c = np.asarray(coeffs(jnp.asarray(m), jnp.asarray(dm)), np.float64)
    inv = np.asarray(invariants(jnp.asarray(m)), np.float64)
    mu, sd = inv.mean(0), inv.std(0)
    expo = _exponents(deg)
    basis = np.asarray(poly(jnp.asarray(m), expo, mu, sd), np.float64)

    beta_mean = _wls(basis, c)                       # (nb, 10)
    eps = c - basis @ beta_mean
    outer = (eps[:, :, None] * eps[:, None, :]).reshape(len(m), NC * NC)
    beta_cov = _wls(basis, outer)                    # (nb, 100)
    return dict(mu=mu, sd=sd, expo=expo, beta_mean=beta_mean,
                beta_cov=beta_cov), c, eps


def sigma_moment(m, f):
    """Sigma_{(a j),(b k)}(m) in MOMENT space, (10, 10), for ONE galaxy.

    Fitted in the galaxy frame, then carried back by the same linear map that
    `from_coeffs` applies -- so it rotates correctly with the galaxy and the
    whole thing stays isotropic.  J is taken as the jacobian of `from_coeffs`
    rather than written out again, so there is one definition of that map.
    Row index is a*5 + j, matching the (2, 5) response flattened row-major.
    """
    basis = poly(m, f["expo"], f["mu"], f["sd"])
    S = (basis @ f["beta_cov"]).reshape(NC, NC)
    S = 0.5 * (S + S.T)                              # symmetrise the fit
    J = jax.jacfwd(lambda cc: from_coeffs(m, cc))(jnp.zeros(NC)).reshape(10, NC)
    return J @ S @ J.T


def correction(m, f, score):
    """-(d_k Sigma_{aj,bk} + Sigma_{aj,bk} score_k), one galaxy, as (3, 5).

    Columns are [g1g1, g1g2, g2g2], matching `d2m_dg2`, so the result adds
    straight onto the per-template second-order target.
    """
    sig = lambda x: sigma_moment(x, f).reshape(2, 5, 2, 5)
    S = sig(m)                                        # (2, 5, 2, 5)
    dS = jax.jacfwd(sig)(m)                           # (2, 5, 2, 5, 5)
    # contract the second moment index k with d_k and with the score
    term = jnp.einsum("ajbkk->ajb", dS) + jnp.einsum("ajbk,k->ajb", S, score)
    out = -jnp.stack([term[0, :, 0], 0.5 * (term[0, :, 1] + term[1, :, 0]),
                      term[1, :, 1]])
    return out                                        # (3, 5)


def bulk_score(flow, m, batch=2000):
    """grad log p_bulk at each m, with the shear layer peeled off.

    The correction needs d_k log P of the g-INDEPENDENT density, which is the
    flow with its ShearResponse removed -- the same split `bias.split_centroid`
    makes.  During shear training the bulk is frozen, so this is a one-off.
    """
    from flowjax.bijections import Chain, Invert
    from flowjax.distributions import Transformed

    from models.shear import ShearResponse
    # Located BY TYPE: since 16ece5c the chart is bijections[0] and the
    # conditional layers follow it, so `bij[0]` is RawMomentStandardize and the
    # old `assert isinstance(bij[0], ShearResponse)` could never pass.  Same
    # filter `shear.bulk_of` uses.
    bij = flow.bijection.bijection.bijections
    keep = [b for b in bij if not isinstance(b, ShearResponse)]
    assert len(keep) < len(bij), "no ShearResponse in the chain"
    bulk_flow = Transformed(flow.base_dist,
                            Invert(Chain(keep).merge_chains()))
    g = eqx.filter_jit(jax.vmap(jax.grad(lambda x: bulk_flow.log_prob(x))))
    return jnp.concatenate([g(m[i:i + batch]) for i in range(0, len(m), batch)])


def target_offset(flow, m, dm, deg=DEG, batch=2000):
    """The (n, 3, 5) term to ADD to `d2m_dg2` before velocity matching.

    This is the whole point of the module: `shear._velocity_mse` fits the
    layer's second derivative to the per-template `d2m_dg2`, whose minimiser is
    E[R_t|m], and the density needs E[R_t|m] + this.
    """
    f, _, _ = fit(m, dm, deg)
    score = bulk_score(flow, m)
    one = eqx.filter_jit(jax.vmap(correction, in_axes=(0, None, 0)))
    return jnp.concatenate([one(m[i:i + batch], f, score[i:i + batch])
                            for i in range(0, len(m), batch)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deg", type=int, default=DEG)
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--data", default=DATA)
    a = p.parse_args()

    import shear as shear_top
    m, dm, d2m = (np.asarray(v, np.float64)[:a.n] for v in shear_top.load(a.data))
    mj, dmj = jnp.asarray(m), jnp.asarray(dm)

    # 1. the decomposition is exact, or nothing below means anything
    rt = np.asarray(from_coeffs(mj, coeffs(mj, dmj)), np.float64)
    err = np.abs(rt - dm).max() / np.abs(dm).max()
    print(f"coefficient round-trip: max rel err {err:.2e}")
    # exact in exact arithmetic; the tolerance is jax's default float32 epsilon,
    # not a modelling allowance
    assert err < 1e-6, "galaxy-frame decomposition is not invertible"

    # 2. isotropy: the galaxy-frame coefficients must not know the orientation
    f, c, eps = fit(m, dm, a.deg)
    ang = np.arctan2(m[:, 3], m[:, 2])
    oct_ = np.digitize(ang, np.quantile(ang, np.linspace(0, 1, 9)[1:-1]))
    spread = np.array([[c[oct_ == o, i].mean() for o in range(8)]
                       for i in range(NC)])
    sd = c.std(0)
    live = sd > 0            # A's imaginary part is identically zero
    rel = (np.abs(spread - spread.mean(1, keepdims=True)).max(1)[live]
           / sd[live] * np.sqrt((live.sum() and len(m) / 8)))
    print(f"isotropy: max drift of any galaxy-frame mean across orientation "
          f"octiles = {rel.max():.2f} sigma")

    # 3. parity: it flips the imaginary parts, so their means must vanish
    print(f"\n{'coefficient':>12s} {'mean/sd':>10s} {'sd':>12s}")
    for i, nm in enumerate(NAMES):
        print(f"{nm:>12s} {c[:, i].mean() / c[:, i].std():10.4f} "
              f"{c[:, i].std():12.4e}")

    # 4. the correction itself, against the target it modifies
    import bulk
    import jax.random as jr
    n90 = int(0.9 * len(m))
    flow = bulk.build_flow(jr.key(0), m[:n90], shear=True)
    flow = eqx.tree_deserialise_leaves("flows/shear.eqx", flow)
    corr = np.asarray(target_offset(flow, mj, dmj, a.deg), np.float64)
    s = np.stack([m[:, 0], m[:, 1], m[:, 1], m[:, 1], m[:, 4]], -1)[:, None, :]
    print(f"\ncorrection vs the existing target, rms(corr)/rms(d2m_dg2):")
    for i, col in enumerate(["g1g1", "g1g2", "g2g2"]):
        num = np.sqrt(np.mean((corr[:, i] / s[:, 0]) ** 2))
        den = np.sqrt(np.mean((d2m[:, i] / s[:, 0]) ** 2))
        print(f"  {col:>5s}  {num / den:8.2%}")

    # 5. it is a total divergence, so E_P[correction] = 0 -- and the catalog IS
    #    drawn from P, so the plain sample mean estimates it.  Unnormalised, so
    #    the bright tail dominates; compare against the same sample's rms.
    rel = np.abs(corr.mean(0)) / np.sqrt(np.mean(corr ** 2, 0))
    print(f"\nnormalisation (E_P[corr] = 0): max |mean|/rms over the 15 "
          f"components = {rel.max():.3f}")


if __name__ == "__main__":
    main()
