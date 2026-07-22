"""
check_template_shear_response.py
================================
Option (a): treat training TEMPLATES as targets and compare the flow's implied
shear response to each template's EXACT analytic shear derivatives (``dm_dg``,
``d2m_dg2``) that the training table already carries on disk.

Why this form (and not ``∂_g log p`` vs an advection term).  The flow is trained
(flows.py:815) so that ``p(x | g)`` is the density of templates advected along
their analytic trajectory ``M(g) = M0 + dM/dg·g + ½ d²M/dg²·gg``.  Comparing the
flow's condition-derivative ``∂_g log p`` to ``−(dM/dg)·∇_x log p`` leaves a
template-dependent velocity-divergence term ``∇_x·(dM/dg)`` (the standardiser
Jacobian is not constant), which would swamp the residual with geometry rather
than flow error.  Instead we read the flow's OWN implied moment shear response,
divergence-free, via its invertibility:

    z0        = flow⁻¹(x_std | g=0, sx)          # freeze the base point
    x_flow(g) = flow(z0 | g, sx)                 # transport it under shear g
    dx_flow/dg, d²x_flow/dg²  = ∂_g x_flow(g)|₀  # flow's shear derivatives

and compare against the analytic trajectory pushed through the SAME standardiser:

    x_an(g)   = standardise(M0 + dM/dg·g + ½ d²M/dg²·gg)
    dx_an/dg, d²x_an/dg²

Both live in standardised moment space [log10 Mf, Mr/Mf, M1/Mr, M2/Mr], so the
residual is dimensionless and directly interpretable.  A well-trained flow that
learned shear as coherent transport has dx_flow/dg ≈ dx_an/dg; where it fails —
especially the bright/large sparse-template corner — is the interpolation claim
under test.

The flow's OWN standardiser (sidecar stats) is used throughout, not the one the
loader rebuilds from the subsample.

Run:
    python check_template_shear_response.py --subsample 2000 --n-templates 5000
    python check_template_shear_response.py --selfcheck
"""

from __future__ import annotations

import argparse

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr


# ---------------------------------------------------------------------------
# Derivative extractors (pure; standardiser is injected so selfcheck needs no flow)
# ---------------------------------------------------------------------------

def _analytic_std_derivs(M0, dmdg, d2mdg2, std_transform):
    """(x0, dx/dg (4,2), d²x/dg² (4,2,2)) of standardise(analytic trajectory) at g=0.

    M0 (4,) raw moments; dmdg (4,2); d2mdg2 (4,2,2).  Autodiff carries the
    standardiser Jacobian, so no manual chain rule.
    """
    def f(g):
        M = M0 + dmdg @ g + 0.5 * jnp.einsum("k,mkl,l->m", g, d2mdg2, g)
        return std_transform(M)
    z = jnp.zeros(2)
    return f(z), jax.jacfwd(f)(z), jax.jacfwd(jax.jacfwd(f))(z)


def _flow_std_derivs(prior_flow, x_std0, sx):
    """(x0, dx/dg (4,2), d²x/dg² (4,2,2)) of the flow's base-frozen transport at g=0."""
    z = jnp.zeros(2)
    z0 = prior_flow.bijection.inverse(x_std0, condition=jnp.concatenate([z, sx]))

    def f(g):
        return prior_flow.bijection.transform(z0, condition=jnp.concatenate([g, sx]))

    return f(z), jax.jacfwd(f)(z), jax.jacfwd(jax.jacfwd(f))(z)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report(log10mf, J_flow, J_an, H_flow, H_an) -> None:
    """Per-flux-bin median relative residual of the first/second shear derivatives."""
    def relnorm(a, b):  # ||a-b||_F / (||b||_F + eps) per template
        num = np.sqrt(((a - b) ** 2).reshape(a.shape[0], -1).sum(1))
        den = np.sqrt((b ** 2).reshape(b.shape[0], -1).sum(1))
        return num / (den + 1e-12)

    rel1 = relnorm(J_flow, J_an)
    rel2 = relnorm(H_flow, H_an)
    print(f"  overall   n={len(rel1):,}   rel|dx/dg|  median={np.median(rel1):.3f} "
          f"p90={np.percentile(rel1, 90):.3f}   "
          f"rel|d2x/dg2| median={np.median(rel2):.3f} p90={np.percentile(rel2, 90):.3f}")

    edges = np.quantile(log10mf, np.linspace(0, 1, 6))
    print("  by log10 Mf (bright end is the sparse-template corner):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (log10mf >= lo) & (log10mf <= hi)
        if m.sum() < 20:
            continue
        print(f"    [{lo:5.2f},{hi:5.2f}]  n={m.sum():>6,}  "
              f"rel1 med={np.median(rel1[m]):.3f}  rel2 med={np.median(rel2[m]):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsample", type=int, default=2000,
                    help="1/subsample of the 72M-row template table read before cuts (default 2000).")
    ap.add_argument("--n-templates", type=int, default=5000,
                    help="Cap the flow evaluation to this many templates after load (default 5000).")
    ap.add_argument("--prior", type=str, default=None, help="Prior-flow .eqx (default: config canonical).")
    ap.add_argument("--out", type=str, default="data/template_shear_response.npz")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        demo()
        return

    import equinox as eqx
    from bfd_cnf.config import PRIOR_FLOW_PATH, Q_FLOW_PATH, key as base_key
    from bfd_cnf.data import load_training_dataset
    from bfd_cnf.models.bijections import load_stats
    from bfd_cnf.models.flows import cx_to_sx_cond, even_cov_to_CX
    from bfd_cnf.integrate_grid import load_prior_flow

    prior_path = args.prior or PRIOR_FLOW_PATH
    r2s = load_stats(prior_path)                       # the FLOW's own standardiser
    std_transform = lambda M: r2s.transform_and_log_det(M)[0]

    ds = load_training_dataset(subsample=args.subsample)
    M0 = np.asarray(ds["moments_jnp"])                 # (N,4) raw
    dmdg = np.asarray(ds["dm_dg_jnp"])[:, :4, :]        # (N,4,2) even moments
    d2mdg2 = np.asarray(ds["d2m_dg2_jnp"])[:, :4, :, :] # (N,4,2,2)
    cov = np.asarray(ds["cov_jnp"])                     # (N,4,4) raw

    N = M0.shape[0]
    if args.n_templates < N:
        idx = np.asarray(jr.choice(jr.PRNGKey(0), N, (args.n_templates,), replace=False))
        M0, dmdg, d2mdg2, cov = M0[idx], dmdg[idx], d2mdg2[idx], cov[idx]
    print(f"Evaluating {M0.shape[0]:,} templates.")

    key, k_flow = jr.split(base_key)
    prior_flow = load_prior_flow(k_flow, prior_path, Q_FLOW_PATH)

    M0j = jnp.asarray(M0, jnp.float32)
    x_std0 = jax.vmap(std_transform)(M0j)                          # (N,4) std
    sx = jax.vmap(cx_to_sx_cond)(even_cov_to_CX(jnp.asarray(cov, jnp.float32)))  # (N,3)

    an = jax.vmap(_analytic_std_derivs, in_axes=(0, 0, 0, None))(
        M0j, jnp.asarray(dmdg, jnp.float32), jnp.asarray(d2mdg2, jnp.float32), std_transform)
    fl = eqx.filter_jit(eqx.filter_vmap(_flow_std_derivs, in_axes=(None, 0, 0)))(
        prior_flow, x_std0, sx)

    _, J_an, H_an = (np.asarray(a) for a in an)
    x0_fl, J_fl, H_fl = (np.asarray(a) for a in fl)

    # round-trip sanity: flow transport at g=0 must return the input point
    rt = np.max(np.abs(x0_fl - np.asarray(x_std0)))
    print(f"round-trip max|flow(z0|0) - x_std0| = {rt:.2e}  (should be ~0)")

    log10mf = np.log10(M0[:, 0])
    print("\ntemplate shear response  flow vs analytic (standardised moment space):")
    _report(log10mf, J_fl, J_an, H_fl, H_an)

    np.savez(args.out, M0=M0, log10mf=log10mf, mr_mf=M0[:, 1] / M0[:, 0],
             sx=np.asarray(sx), J_flow=J_fl, J_analytic=J_an, H_flow=H_fl, H_analytic=H_an)
    print(f"\nSaved per-template derivatives to {args.out}")


def demo() -> None:
    """Self-check: the analytic std-derivative extractor recovers dm_dg / d2m_dg2.

    With an identity standardiser, dx/dg == dm_dg and d²x/dg² == d2m_dg2 exactly.
    With a nonlinear standardiser (log10 on Mf) the flux row picks up the chain
    rule d/dg log10(Mf) = (dMf/dg)/(Mf ln10) — checked against the closed form.
    """
    rng = np.random.default_rng(0)
    M0 = jnp.asarray(np.abs(rng.normal(size=4)) + 2.0)
    dmdg = jnp.asarray(rng.normal(size=(4, 2)))
    d2mdg2 = jnp.asarray(rng.normal(size=(4, 2, 2)))
    d2mdg2 = 0.5 * (d2mdg2 + d2mdg2.transpose(0, 2, 1))  # symmetric in the shear pair

    _, J, H = _analytic_std_derivs(M0, dmdg, d2mdg2, std_transform=lambda M: M)
    assert np.allclose(J, dmdg, atol=1e-5), np.abs(J - dmdg).max()
    assert np.allclose(H, d2mdg2, atol=1e-5), np.abs(H - d2mdg2).max()

    log10 = lambda M: M.at[0].set(jnp.log10(M[0]))
    _, Jl, _ = _analytic_std_derivs(M0, dmdg, d2mdg2, std_transform=log10)
    want0 = dmdg[0] / (M0[0] * jnp.log(10.0))          # chain rule on the flux row
    assert np.allclose(Jl[0], want0, atol=1e-5), (Jl[0], want0)
    assert np.allclose(Jl[1:], dmdg[1:], atol=1e-5)     # other rows untouched
    print("OK: analytic derivative extractor recovers dm_dg / d2m_dg2 (identity + log10 std)")


if __name__ == "__main__":
    main()
