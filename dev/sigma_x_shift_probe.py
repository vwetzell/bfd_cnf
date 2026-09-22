"""What does centroid marginalisation actually change in moment space?

Ground truth, no flow/layer involved: `weighted_copy_mean` on the real copy
catalog IS eq. (36)'s marginalisation, computed directly from data. Measure
each galaxy's moment shift Delta(Sigma_X) = weighted_copy_mean(Sigma_X) - m0
at a spread of Sigma_X points (isotropic scale + anisotropic at several
angles), then test the leading-order prediction from eq. (36) itself:

    u ~ N(0, Sigma_u),  Sigma_u = J0^-1 Sigma_X J0^-T   (linear change of
    variables X(u) ~ J0 u near u=0, since X(0)=0 by construction)
    m(u) ~ m0 + g.u + 1/2 u^T H u   (Taylor around u=0)
    => E[Delta] ~ 1/2 tr(H Sigma_u)   (odd term vanishes under E[u]=0)

which is LINEAR in Sigma_X's three components (C00, C01, C11) for a FIXED
galaxy (H, J0 don't depend on Sigma_X) -- i.e. per galaxy, per moment, there
should exist one symmetric-2x2 "response tensor" T_a such that

    Delta_a(Sigma_X) ~= T_a : Sigma_X = A_a*C00 + B_a*C01 + D_a*C11

fit that tensor per galaxy per moment from the measured points (least
squares, no intercept -- Delta_a(0)=0 exactly, no smearing with no noise),
report how good the fit is (R^2, i.e. is the leading-order LINEAR-IN-SIGMA_X
picture actually right, including for the anisotropic/off-diagonal C01
term the current SigmaXBlockLayer ansatz was never derived from), and how
the fitted tensor's own components (A,B,D per moment) depend on the
galaxy's own moments -- which is what a from-scratch layer should condition
its own linear-response map on, rather than the current architecture's
specific (and apparently incomplete) functional form.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import centroid as C  # noqa: E402


MOMENT_LABELS = ["Mf", "Mr", "M1", "M2", "Mc"]


def build_grid(sigma_x0):
    """Sigma_X test points: isotropic scale + anisotropic at several angles.

    All near-nominal (within the catalog's validated [1/1.4, 1.4] scale
    margin, |e| <= 0.08) -- this tests LOCAL linearity in Sigma_X around the
    operating point, which is what a per-galaxy Taylor expansion predicts,
    not global behaviour far outside where the copy grid is even valid.
    """
    pts = {
        "iso1.00": np.asarray(sigma_x0, dtype=float),
        "iso0.85": np.asarray(sigma_x0, dtype=float) * 0.85,
        "iso1.15": np.asarray(sigma_x0, dtype=float) * 1.15,
    }
    for e_mag, theta, label in [
        (0.08, 0.0, "e08_t0"),
        (0.08, np.pi / 4, "e08_t45"),
        (0.08, np.pi / 2, "e08_t90"),
        (0.05, np.pi / 8, "e05_t22"),
        (0.05, 3 * np.pi / 8, "e05_t67"),
        (0.03, 5 * np.pi / 8, "e03_t112"),
    ]:
        pts[label] = C._aniso_sigma_x(sigma_x0, e_mag, theta)
    return pts


def fit_linear_tensor(C_pts, deltas):
    """Per-row (galaxy, moment) least-squares fit of Delta ~ A*C00+B*C01+D*C11.

    `C_pts` (n_pts, 3), `deltas` (n_gal, n_pts, 5). Returns coeffs
    (n_gal, 5, 3) and R^2 (n_gal, 5) -- 1 - SS_res/SS_tot (tot around 0,
    since the model has no intercept and Delta(0)=0 is the physical origin).
    """
    # Same design matrix for every galaxy/moment -> one lstsq solve, batched
    # over the (n_gal*5) right-hand sides.
    A = C_pts  # (n_pts, 3)
    rhs = deltas.transpose(1, 0, 2).reshape(len(A), -1)  # (n_pts, n_gal*5)
    coeffs, res, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
    pred = A @ coeffs
    ss_res = ((rhs - pred) ** 2).sum(axis=0)
    ss_tot = (rhs ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)
    n_gal = deltas.shape[0]
    return (coeffs.T.reshape(n_gal, 5, 3), r2.reshape(n_gal, 5))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--copies", default="../bfd_cnf_imsims/data/copies_gauss2_fwd_g2v3d.fits")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--sigma-scale", type=float, default=None,
                    help="override nominal Sigma_X trace; default reads the catalog's own")
    a = p.parse_args()

    copies, galaxies = C.load_copies(a.copies)
    n_gal = len(galaxies)
    # Same convention centroid.py's own main() uses (line ~816):
    # `sigma_x = galaxies["cov_odd"][0] * scale**2`.
    scale = a.sigma_scale if a.sigma_scale is not None else 1.0
    sigma_x0 = np.asarray(galaxies["cov_odd"][0], dtype=float) * scale ** 2
    print(f"{n_gal} galaxies, {len(copies)} copies, Sigma_X0 = {sigma_x0}")

    grid = build_grid(sigma_x0)
    labels = list(grid.keys())
    sx_list = [grid[k] for k in labels]
    print(f"computing {len(sx_list)} Sigma_X points in parallel...")
    results = C.weighted_copy_mean_many(copies, galaxies, sx_list)

    keep = np.ones(n_gal, dtype=bool)
    for _, k in results:
        keep &= k
    idx = np.flatnonzero(keep)[: a.n]
    print(f"{len(idx)} galaxies with copies at every grid point (using first {a.n})")

    m0 = np.asarray(galaxies["moments"][idx], dtype=np.float64)
    deltas = np.empty((len(idx), len(sx_list), 5))
    for j, (tgt, k) in enumerate(results):
        # tgt is already indexed to `k`; map back to `idx`'s positions within k
        pos = np.cumsum(k) - 1
        deltas[:, j, :] = tgt[pos[idx]] - m0

    C_pts = np.stack(sx_list, axis=0)  # (n_pts, 3)
    coeffs, r2 = fit_linear_tensor(C_pts, deltas)

    print("\nLinear-in-Sigma_X fit quality (R^2, no intercept), by moment:")
    for a_, lab in enumerate(MOMENT_LABELS):
        r = r2[:, a_]
        print(f"  {lab:3s}  p5 {np.percentile(r, 5):.4f}  p50 {np.median(r):.4f}  "
              f"p95 {np.percentile(r, 95):.4f}  frac<0.9 {np.mean(r < 0.9):.3f}")

    # How does the fitted M1/M2 response tensor relate to the galaxy's own
    # ellipticity? (A-D)/2 and B are the tensor's own "ellipticity"-like
    # components; compare to the galaxy's (e1, e2) = (M1, M2)/Mr.
    Mr = m0[:, 1]
    e1, e2 = m0[:, 2] / Mr, m0[:, 3] / Mr
    for a_, lab in zip((2, 3), ("M1", "M2")):
        A_, B_, D_ = coeffs[:, a_, 0], coeffs[:, a_, 1], coeffs[:, a_, 2]
        tens_e1, tens_e2 = 0.5 * (A_ - D_), B_
        corr1 = np.corrcoef(tens_e1, e1)[0, 1]
        corr2 = np.corrcoef(tens_e2, e2)[0, 1]
        print(f"\n{lab} response tensor's own (e1,e2)-like split vs galaxy "
              f"(e1,e2): corr(tens_e1,gal_e1)={corr1:.3f}  "
              f"corr(tens_e2,gal_e2)={corr2:.3f}")

    out = Path("scratchpad_sigma_x_probe.npz")
    np.savez(out, idx=idx, m0=m0, deltas=deltas, C_pts=C_pts, coeffs=coeffs, r2=r2,
             labels=np.array(labels))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
