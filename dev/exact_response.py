"""What does `rho` have to be, and does the bound at _COEFF_MAX cost anything?

`models.shear.response` models the physical shape eps = (M1 + i M2) / Mr as

    d eps = A g + B eps^2 gbar                      (first order)
          + mu eps_bar g^2 + nu eps |g|^2 + rho eps^3 gbar^2   (second order)

Those are the only structures spin and parity allow, so the model is not a
choice.  But the three SECOND-order structures carry different powers of eps
(|eps|, |eps|, |eps|^3), and that is the suspicion this script tests: as a
galaxy gets round the rho column of the design matrix shrinks like |eps|^3
while the other two shrink like |eps|, so rho is the ratio of a small residual
to a much smaller number -- unidentified, and driven to the bound by whatever
noise is in the target.

There is no network here.  The catalog's own exact dm/dg and d2m/dg2 determine
all five coefficients per galaxy in closed form (3 complex second-order
structures, 3 complex data), so this measures the TARGET, not a fit.

Reported:
  * |rho| required, against |eps|;
  * the CONTRIBUTION of each second-order term, |c| = |coeff * eps^k|, which is
    what the response actually is and is free of the ill-conditioned division;
  * Im/|.| of each coefficient -- exactly zero if the real-coefficient (parity)
    ansatz holds per galaxy, non-zero where a single galaxy's curvature has
    structure no equivariant model can carry;
  * what clipping rho to _COEFF_MAX costs, as a fraction of the total
    second-order spin-2 response.

    python -m dev.rho_exact --data ../bfd_cnf_imsims/data/gauss2_g0_1M.fits
"""
import argparse

import numpy as np
from astropy.io import fits

from models.shear import _COEFF_MAX

# raw moment slots: [Mf, Mr, M1, M2, Mc]
MR, M1, M2 = 1, 2, 3


def coefficients(m, q, r):
    """(A, B, mu, nu, rho) and the three second-order contributions, per galaxy.

    `q` is (n, 2, 5) = dM_m/dg_i and `r` is (n, 3, 5) = d2M_m/dg2 in
    [g1g1, g1g2, g2g2], with `shear.lens`'s 1/2 convention on the diagonal
    entries.  THE G INDEX COMES FIRST -- that is the layout `shear.load` and
    `models.shear.dm_dg` both use; reshaping to (n, 5, 2) silently transposes
    the data and every number downstream is garbage.
    """
    d0 = m[:, MR]
    n0 = m[:, M1] + 1j * m[:, M2]
    eps = n0 / d0

    n1 = q[:, :, M1] + 1j * q[:, :, M2]                  # (n, 2)
    d1 = q[:, :, MR]                                     # (n, 2)
    N = r[:, :, M1] + 1j * r[:, :, M2]                   # (n, 3)
    D = r[:, :, MR]                                      # (n, 3)

    # First order: eps^(1) = [n1.g - eps d1.g] / d0, a complex real-linear form
    # B1 g1 + B2 g2.
    B1 = (n1[:, 0] - eps * d1[:, 0]) / d0
    B2 = (n1[:, 1] - eps * d1[:, 1]) / d0
    c_p = 0.5 * (B1 - 1j * B2)                           # coefficient of g
    c_m = 0.5 * (B1 + 1j * B2)                           # coefficient of gbar

    # Second order: eps^(2) = half_n2/d0 - (n1.g)(d1.g)/d0^2
    #                         - eps half_d2/d0 + eps (d1.g)^2/d0^2,
    # collected as A11 g1^2 + A12 g1 g2 + A22 g2^2.
    A11 = (0.5 * N[:, 0] / d0 - n1[:, 0] * d1[:, 0] / d0**2
           - eps * 0.5 * D[:, 0] / d0 + eps * d1[:, 0] ** 2 / d0**2)
    A12 = (N[:, 1] / d0 - (n1[:, 0] * d1[:, 1] + n1[:, 1] * d1[:, 0]) / d0**2
           - eps * D[:, 1] / d0 + eps * 2 * d1[:, 0] * d1[:, 1] / d0**2)
    A22 = (0.5 * N[:, 2] / d0 - n1[:, 1] * d1[:, 1] / d0**2
           - eps * 0.5 * D[:, 2] / d0 + eps * d1[:, 1] ** 2 / d0**2)

    # g^2, |g|^2, gbar^2 basis: g1^2 -> (+1, +1, +1), g2^2 -> (-1, +1, -1),
    # g1 g2 -> (+2i, 0, -2i).
    c_z = 0.5 * (A11 + A22)                              # eps |g|^2   -> nu
    half_diff = 0.5 * (A11 - A22)
    c_pp = 0.5 * (half_diff + A12 / (2j))                # eps_bar g^2 -> mu
    c_mm = 0.5 * (half_diff - A12 / (2j))                # eps^3 gbar^2 -> rho

    return eps, c_p, c_m / eps**2, c_pp / np.conj(eps), c_z / eps, c_mm / eps**3, \
        (c_pp, c_z, c_mm)


def _self_check(m, q, r, eps, A, B, mu, nu, rho):
    """The extracted coefficients must rebuild the catalog's own lensed shape.

    Guards the two things that silently break this: the (n, 2, 5) derivative
    layout, and the complex bookkeeping in `coefficients`.  What is left over is
    the O(g^3) truncation, so the tolerance is loose on purpose.
    """
    for g1, g2 in ((0.02, 0.0), (0.0, 0.015), (-0.01, 0.017)):
        quad = np.array([0.5 * g1 * g1, g1 * g2, 0.5 * g2 * g2])
        M = (m + np.einsum("i,nim->nm", np.array([g1, g2]), q)
             + np.einsum("i,nim->nm", quad, r))
        true = (M[:, M1] + 1j * M[:, M2]) / M[:, MR]
        gc = g1 + 1j * g2
        model = eps + (A * gc + B * eps**2 * np.conj(gc)
                       + mu * np.conj(eps) * gc**2 + nu * eps * abs(gc) ** 2
                       + rho * eps**3 * np.conj(gc) ** 2)
        rel = np.abs(model - true).max() / np.abs(true - eps).max()
        assert rel < 1e-3, f"extraction wrong at g=({g1},{g2}): {rel:.2e}"
    print("self-check: coefficients rebuild the exact lensed shape to <1e-3 "
          "of the shift (O(g^3) truncation)\n")


def _pct(x, label):
    p = np.nanpercentile(np.abs(x), [50, 90, 99])
    return f"{label:>10s}{p[0]:12.3g}{p[1]:12.3g}{p[2]:12.3g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../bfd_cnf_imsims/data/gauss2_g0_1M.fits")
    p.add_argument("--n", type=int, default=200000)
    a = p.parse_args()

    d = fits.open(a.data)[1].data[:a.n]
    m = np.asarray(d["moments"], np.float64)
    q = np.asarray(d["dm_dg"], np.float64)
    r = np.asarray(d["d2m_dg2"], np.float64)

    eps, A, B, mu, nu, rho, (c_pp, c_z, c_mm) = coefficients(m, q, r)
    _self_check(m, q, r, eps, A, B, mu, nu, rho)
    e = np.abs(eps)
    print(f"{a.data}: {len(m)} galaxies, _COEFF_MAX = {_COEFF_MAX}\n")

    print("coefficient magnitude          median         p90         p99")
    for x, n_ in ((A, "A"), (B, "B"), (mu, "mu"), (nu, "nu"), (rho, "rho")):
        print(_pct(x, n_))

    print("\nnon-equivariance  |Im| / |.|    median         p90         p99")
    for x, n_ in ((B, "B"), (mu, "mu"), (nu, "nu"), (rho, "rho")):
        print(_pct(np.imag(x) / np.abs(x), n_))

    print("\nsecond-order CONTRIBUTION |coeff * eps^k|   (what the response is)")
    for x, n_ in ((c_pp, "mu term"), (c_z, "nu term"), (c_mm, "rho term")):
        print(_pct(x, n_))

    print("\n|rho| required, by |eps| quintile")
    edges = np.quantile(e, np.linspace(0, 1, 6))
    print(f"{'|eps|':>18s}{'med |rho|':>12s}{'frac > CMAX':>13s}"
          f"{'rho term / total':>18s}")
    tot = np.abs(c_pp) + np.abs(c_z) + np.abs(c_mm)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (e >= lo) & (e < hi)
        print(f"{lo:8.4f}-{hi:<9.4f}{np.median(np.abs(rho[s])):12.3g}"
              f"{(np.abs(rho[s]) > _COEFF_MAX).mean():12.1%}"
              f"{np.median(np.abs(c_mm[s]) / tot[s]):18.1%}")

    # What the bound actually costs: clip |rho| and re-form its contribution.
    clipped = rho * np.minimum(1.0, _COEFF_MAX / np.abs(rho))
    lost = np.abs((rho - clipped) * eps**3)
    print(f"\nclipping |rho| at {_COEFF_MAX}: fires on "
          f"{(np.abs(rho) > _COEFF_MAX).mean():.1%} of galaxies")
    print(f"  lost second-order response / total, median {np.median(lost / tot):.2%}"
          f", p90 {np.percentile(lost / tot, 90):.2%}"
          f", p99 {np.percentile(lost / tot, 99):.2%}")
    print(f"  population-weighted sum(lost)/sum(total) = {lost.sum() / tot.sum():.2%}")


if __name__ == "__main__":
    main()


def conditional_means(eps, c_pp, c_z, c_mm, m, nbin=5):
    """Least-squares (mu, nu, rho) per bin -- the coefficient training targets.

    The per-galaxy solve above divides by eps^3 and so is 1/|eps|^3-amplified
    noise; and a single galaxy's curvature need not be equivariant at all (the
    argument only constrains the conditional MEAN at fixed invariants).  What
    the deriv-supervision MSE actually drives each coefficient to is the real
    weighted least squares over the galaxies sharing those invariants,

        rho_LS = sum Re(c_mm eps_bar^3) / sum |eps|^6

    and likewise mu (power 1, conjugated) and nu (power 1).  That is the number
    _COEFF_MAX has to be able to hold, not the per-galaxy one.
    """
    e2 = np.abs(eps) ** 2
    ls = lambda c, w, p: np.sum(np.real(c * np.conj(w))) / np.sum(np.abs(w) ** 2)
    return (ls(c_pp, np.conj(eps), 1), ls(c_z, eps, 1), ls(c_mm, eps**3, 3), e2)


# --------------------------------------------------------------------------
# spin-0.  The layer's claim is  dX/dg = X c_X Re(eps_bar g)  for X in
# (Mf, Mr, Mc), i.e. the spin-0 response vector must be PARALLEL to eps.  That
# is a falsifiable statement about the catalog, not about any fit.
# --------------------------------------------------------------------------
SPIN0 = ((0, "Mf"), (1, "Mr"), (4, "Mc"))


def spin0_first_order(m, q):
    """Exact per-galaxy c_X, and the part of dX/dg no equivariant model can make.

    Writes dX/dg (a 2-vector in g) as X0 c_X eps + perp.  `c` is the projection
    onto eps; `frac_perp` is |perp| / |dX/dg|, which is zero if and only if the
    response really is a function of the moments through this one structure.
    """
    eps = (m[:, M1] + 1j * m[:, M2]) / m[:, MR]
    ev = np.stack([eps.real, eps.imag], axis=-1)                  # (n, 2)
    e2 = np.sum(ev * ev, axis=-1)
    out = {}
    for slot, name in SPIN0:
        dX = q[:, :, slot]                                        # (n, 2)
        c = np.sum(dX * ev, axis=-1) / (m[:, slot] * e2)
        perp = dX - (m[:, slot] * c)[:, None] * ev
        out[name] = (c, np.linalg.norm(perp, axis=-1)
                     / np.linalg.norm(dX, axis=-1))
    return eps, out
