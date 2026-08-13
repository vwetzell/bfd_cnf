"""check_functional_form.py
===========================
Tests whether ShearPerturbative/ShearTaylorLast's ASSUMED functional form can
even represent the true per-galaxy shear response -- entirely independent of
any trained flow. Uses only the exact analytic Pqr derivatives (dm_dg) from
the templates FITS.

The parametric form (see bijections.py ShearPerturbative/ShearTaylorLast
docstrings) claims:
  1. Flux/size (spin-0): dMf/dg, dMr/dg (as 2-vectors over (g1,g2)) must be
     exactly PARALLEL to the galaxy's own (e1,e2) -- the only invariant is
     cA*(e.g), forcing the response direction, not just magnitude.
  2. Ellipticity (spin-2): the (dM1,dM2)/(dg1,dg2) 2x2 Jacobian must be
     SYMMETRIC (dM1/dg2 == dM2/dg1) -- the "c_g*I + c_ge2*(e^2 term)"
     structure has no antisymmetric/rotational component.

Both are checkable directly from dm_dg (in standardised coordinates, via the
same chain-rule conversion used in check_shear_response_accuracy.py) with NO
trained flow involved -- if either fails on real data, the parametric family
itself cannot represent the truth, no matter how it's trained.

Usage:
    cd bfd_cnf
    BFD_TRAIN_FITS=<fits> PYTHONPATH=. python dev/check_functional_form.py
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bfd_cnf.config import TRAIN_FITS_PATH
from bfd_cnf.data import load_training_dataset


def _std_jac_hess(raw2standard, m):
    f = lambda mm: raw2standard.transform_and_log_det(mm)[0]
    J = jax.jacfwd(f)(m)
    H = jax.jacfwd(jax.jacfwd(f))(m)
    return J, H


def _raw_to_std_response(raw2standard, m, A_raw, B_raw):
    J, H = _std_jac_hess(raw2standard, m)
    A_std = J @ A_raw
    B_std = jnp.einsum("ij,jab->iab", J, B_raw) + jnp.einsum(
        "ijk,ja,kb->iab", H, A_raw, A_raw
    )
    return A_std, B_std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--mf-min", type=float, default=1500.0)
    ap.add_argument("--mf-max", type=float, default=90000.0)
    args = ap.parse_args()

    print(f"Training set: {TRAIN_FITS_PATH}")
    data = load_training_dataset(key=jr.key(args.seed), subsample=1)
    raw2standard = data["raw2standard"]
    m_all = np.asarray(data["moments_jnp"])
    dm_dg = np.asarray(data["dm_dg_jnp"])
    weights = np.asarray(data["weights"])
    N = m_all.shape[0]

    sel = (m_all[:, 0] > args.mf_min) & (m_all[:, 0] < args.mf_max)
    rng = np.random.default_rng(args.seed + 1)
    sel_idx = np.flatnonzero(sel)
    p = weights[sel_idx] / weights[sel_idx].sum()
    idx = rng.choice(sel_idx, size=min(args.n, sel_idx.size), replace=True, p=p)

    d2m_dg2 = np.asarray(data["d2m_dg2_jnp"])
    m = jnp.asarray(m_all[idx])
    A_raw = jnp.asarray(dm_dg[idx])[:, :4]     # (n,4,2), raw units
    B_raw = jnp.asarray(d2m_dg2[idx])[:, :4]   # (n,4,2,2), raw units

    def per_example(m_i, A_raw_i, B_raw_i):
        A_std, B_std = _raw_to_std_response(raw2standard, m_i, A_raw_i, B_raw_i)
        return A_std, B_std, m_i

    A_std, B_std, m_used = jax.vmap(per_example)(m, A_raw, B_raw)
    A_std = np.asarray(A_std)  # (n, 4, 2)
    B_std = np.asarray(B_std)  # (n, 4, 2, 2)
    m_used = np.asarray(m_used)

    e1_std = np.asarray(jax.vmap(lambda mm: raw2standard.transform_and_log_det(mm)[0])(m))[:, 2]
    e2_std = np.asarray(jax.vmap(lambda mm: raw2standard.transform_and_log_det(mm)[0])(m))[:, 3]

    print(f"\nn={len(idx):,} galaxies in selection region ({args.mf_min:g} < Mf < {args.mf_max:g})")

    # -- Test 1: is dMf_std/dg (and dMr_std/dg) parallel to (e1_std, e2_std)? --
    for c, name in [(0, "Mf"), (1, "Mr")]:
        Ax, Ay = A_std[:, c, 0], A_std[:, c, 1]
        e_norm = np.hypot(e1_std, e2_std)
        A_norm = np.hypot(Ax, Ay)
        cross = Ax * e2_std - Ay * e1_std          # 0 if parallel
        dot = Ax * e1_std + Ay * e2_std
        sin_angle = np.abs(cross) / (A_norm * e_norm + 1e-12)
        cos_angle = dot / (A_norm * e_norm + 1e-12)
        print(f"\n-- d{name}_std/dg vs (e1_std,e2_std) parallelism --")
        print(f"  median |sin(angle)| = {np.median(sin_angle):.4f}  "
              f"(0 = exactly parallel, as the model requires)")
        print(f"  median cos(angle)   = {np.median(cos_angle):+.4f}  "
              f"median |A| = {np.median(A_norm):.4g}  median |e| = {np.median(e_norm):.4g}")

    # -- Test 2: is the (dM1,dM2)/(dg1,dg2) Jacobian symmetric? --
    dM1_dg1, dM1_dg2 = A_std[:, 2, 0], A_std[:, 2, 1]
    dM2_dg1, dM2_dg2 = A_std[:, 3, 0], A_std[:, 3, 1]
    asym = dM1_dg2 - dM2_dg1
    scale = 0.5 * (np.abs(dM1_dg2) + np.abs(dM2_dg1)) + 1e-12
    frac_asym = np.abs(asym) / scale
    print(f"\n-- (dM1,dM2)/(dg1,dg2) Jacobian symmetry (dM1/dg2 vs dM2/dg1) --")
    print(f"  median dM1/dg2 = {np.median(dM1_dg2):+.4g}   median dM2/dg1 = {np.median(dM2_dg1):+.4g}")
    print(f"  median |asymmetry| / typical scale = {np.median(frac_asym)*100:.2f}%   "
          f"(0% = exactly symmetric, as the model requires)")
    print(f"  correlation(dM1/dg2, dM2/dg1) = {np.corrcoef(dM1_dg2, dM2_dg1)[0,1]:.4f}")

    # Also check trace/anti-trace structure: dM1/dg1 vs dM2/dg2 (model allows
    # these to differ via the c_ge2*p term, so no symmetry predicted here --
    # printed for context only).
    print(f"\n  (for context) dM1/dg1={np.median(dM1_dg1):+.4g}  dM2/dg2={np.median(dM2_dg2):+.4g}")

    # -- Test 3: the anisotropic part of the (M1,M2) Jacobian must align with
    # e^2 = (e1^2-e2^2, 2e1e2): J = c_g*I + c_ge2*[[p,q],[q,-p]], p=e1^2-e2^2,
    # q=2e1e2 -- so (J11-J22, 2*J12) must be parallel to (p, q). --
    p = e1_std ** 2 - e2_std ** 2
    q = 2.0 * e1_std * e2_std
    aniso_x, aniso_y = dM1_dg1 - dM2_dg2, 2.0 * dM1_dg2
    cross2 = aniso_x * q - aniso_y * p
    dot2 = aniso_x * p + aniso_y * q
    n_aniso = np.hypot(aniso_x, aniso_y)
    n_e2 = np.hypot(p, q)
    sin2 = np.abs(cross2) / (n_aniso * n_e2 + 1e-12)
    cos2 = dot2 / (n_aniso * n_e2 + 1e-12)
    print(f"\n-- (M1,M2) Jacobian anisotropic part vs e^2=(e1^2-e2^2,2e1e2) alignment --")
    print(f"  median |sin(angle)| = {np.median(sin2):.4f}  (0 = exactly aligned, as the model requires)")
    print(f"  median cos(angle)   = {np.median(cos2):+.4f}")

    # -- Test 4: flux/size 2nd-order Hessian must be c0*I + c1*(e outer e) --
    # i.e. (H11-H22, 2*H12) parallel to (e1^2-e2^2, 2*e1*e2) = (p,q) (same
    # e-direction as above, NOT e^2 -- outer product of e with itself). --
    ee_x, ee_y = e1_std ** 2 - e2_std ** 2, 2.0 * e1_std * e2_std
    for c, name in [(0, "Mf"), (1, "Mr")]:
        H11, H12, H22 = B_std[:, c, 0, 0], B_std[:, c, 0, 1], B_std[:, c, 1, 1]
        hx, hy = H11 - H22, 2.0 * H12
        cross3 = hx * ee_y - hy * ee_x
        dot3 = hx * ee_x + hy * ee_y
        n_h = np.hypot(hx, hy)
        n_ee = np.hypot(ee_x, ee_y)
        sin3 = np.abs(cross3) / (n_h * n_ee + 1e-12)
        cos3 = dot3 / (n_h * n_ee + 1e-12)
        print(f"\n-- d2{name}_std/dg2 anisotropic part vs (e1^2-e2^2, 2e1e2) alignment --")
        print(f"  median |sin(angle)| = {np.median(sin3):.4f}  (0 = exactly aligned, as the model requires)")
        print(f"  median cos(angle)   = {np.median(cos3):+.4f}   median |H_aniso| = {np.median(n_h):.4g}")


if __name__ == "__main__":
    main()
