"""
check_template_shear_recovery.py
================================
Option-(a), decision-relevant form: apply a KNOWN shear to the templates using
their exact analytic derivatives, then ask the flow's own estimator to recover
it.  Unlike the base-transport test, this uses the flow's NATIVE marginal
(∂_g log p) — the same quantity shear recovery consumes — so there is no
base-transport/advection ambiguity and no velocity-divergence term.

For a chosen g_true we build two paired, ring-like arms from the SAME templates:

    M±(g) = M0 ± dM/dg·g_true + ½ d²M/dg²·g_true⊗g_true          (analytic, on disk)

standardise, and form each template's NOISELESS flow PQR at applied shear g'=0
(Σ→0 ⇒ P(g')=p_flow(M|g'), no integration):

    P   = p_flow(x|0)
    Qi  = P·∂_{g'_i} log p_flow(x|g')|₀
    Rij = P·(∂²_{ij} log p + ∂_i log p·∂_j log p)

Summing the per-object log-PQR over the ensemble and running BFD's meanShear
(reused from check_mean_shear_bias) gives the recovered ĝ per arm, hence

    m1 = (ĝ1(+) − ĝ1(−)) / (2 g_true) − 1        multiplicative bias
    c1,c2 = (ĝ(+) + ĝ(−)) / 2                     additive bias

A well-trained flow recovers the shear encoded in dm_dg ⇒ m1≈0, c≈0.  A nonzero,
flux-dependent m1 is a real calibratable bias measured against EXACT ground truth
— the sparse-vs-dense question with a number you can trust.  The g=0 arm gives
the additive bias on unsheared (in-distribution) templates directly.

Run:
    python check_template_shear_recovery.py --g 0.02
    python check_template_shear_recovery.py --selfcheck
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

# validated meanShear / bias helpers (repo root); import is side-effect free
from check_mean_shear_bias import _logpqr, _g_cov, _bias, _clean


def _shear_moments(M0, dmdg, d2mdg2, g):
    """M0 (N,4) ± dm_dg·g + ½ d2m_dg2·gg, all raw moments.  g is a (2,) array."""
    g = np.asarray(g, np.float64)
    return M0 + dmdg @ g + 0.5 * np.einsum("k,nmkl,l->nm", g, d2mdg2, g)


def _noiseless_pqr(prior_flow, x_std, sx, batch_size=10_000):
    """Per-template flow PQR [P,Q1,Q2,R11,R22,R12] (flow order) at applied g'=0.

    Batched over templates (the vmapped MAF Hessian OOMs at ~1M in one shot)."""
    import equinox as eqx

    def one(flow, xi, si):
        def lp(g1, g2):
            cond = jnp.concatenate([jnp.stack([g1, g2]), si])
            return flow.log_prob(xi, condition=cond)

        z = jnp.float32(0.0)
        l = lp(z, z)
        d1 = jax.grad(lp, 0)(z, z)
        d2 = jax.grad(lp, 1)(z, z)
        d11 = jax.grad(jax.grad(lp, 0), 0)(z, z)
        d22 = jax.grad(jax.grad(lp, 1), 1)(z, z)
        d12 = jax.grad(jax.grad(lp, 0), 1)(z, z)
        P = jnp.exp(l)
        return jnp.stack(
            [P, P * d1, P * d2, P * (d11 + d1 * d1), P * (d22 + d2 * d2), P * (d12 + d1 * d2)]
        )

    fn = eqx.filter_jit(eqx.filter_vmap(one, in_axes=(None, 0, 0)))
    out = [np.asarray(fn(prior_flow, x_std[s:s + batch_size], sx[s:s + batch_size]))
           for s in range(0, x_std.shape[0], batch_size)]
    return np.concatenate(out, axis=0)


def _recover(pqr_p, pqr_m, g, w=None):
    """(m1,c1,c2, analytic sigma) from paired ± arms via summed log-PQR meanShear."""
    LPp, LPm = _logpqr(pqr_p), _logpqr(pqr_m)
    if w is not None:
        Sp, Sm = (w[:, None] * LPp).sum(0), (w[:, None] * LPm).sum(0)
    else:
        Sp, Sm = LPp.sum(0), LPm.sum(0)
    gp, covp = _g_cov(Sp)
    gm, covm = _g_cov(Sm)
    m1, c1, c2 = _bias(gp, gm, g)
    a_g11 = np.sqrt(covp[0, 0] + covm[0, 0])
    return m1, c1, c2, a_g11 / (2 * g), gp, gm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g", type=float, nargs="+", default=[0.02],
                    help="Applied |g_true| value(s); pass several to sweep (default 0.02). "
                         "Constant m1 across g ⇒ genuine 1st-order bias; m1∝g ⇒ higher-order.")
    ap.add_argument("--subsample", type=int, default=2000, help="1/subsample of the template table (default 2000).")
    ap.add_argument("--n-templates", type=int, default=None, help="Cap templates after load (default: all).")
    ap.add_argument("--nda-weight", action="store_true", help="Weight the ensemble sum by nda (sky density).")
    ap.add_argument("--prior", type=str, nargs="+", default=None,
                    help="Prior-flow .eqx checkpoint(s); pass several to sweep training convergence "
                         "(default: config canonical). FITS is read once and reused across them.")
    ap.add_argument("--stats-file", type=str, default=None,
                    help="Standardiser .stats.npz to use for ALL checkpoints (for backups lacking a "
                         "sidecar; the standardiser is data-derived so shared across checkpoints).")
    ap.add_argument("--save-npz", type=str, default=None,
                    help="Save per-g ± arms in grid-npz format (pqr_p/pqr_m/targets_p/targets_m) "
                         "to {stem}_g{g}.npz, for plot_mbias_hexbin_independent (needs g=0.02 ⇒ Δg=0.04).")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        demo()
        return

    from bfd_cnf.config import PRIOR_FLOW_PATH, Q_FLOW_PATH, key as base_key
    from bfd_cnf.data import load_training_dataset
    from bfd_cnf.models.bijections import load_stats, RawMomentStandardize
    from bfd_cnf.models.flows import cx_to_sx_cond, even_cov_to_CX
    from bfd_cnf.integrate_grid import load_prior_flow

    priors = args.prior or [PRIOR_FLOW_PATH]
    stats_override = None
    if args.stats_file:
        s = np.load(args.stats_file)  # a .stats.npz; used for ALL checkpoints
        stats_override = RawMomentStandardize(mean=jnp.asarray(s["mean"]), std=jnp.asarray(s["std"]))

    ds = load_training_dataset(subsample=args.subsample)
    M0 = np.asarray(ds["moments_jnp"], np.float64)
    dmdg = np.asarray(ds["dm_dg_jnp"], np.float64)[:, :4, :]
    d2mdg2 = np.asarray(ds["d2m_dg2_jnp"], np.float64)[:, :4, :, :]
    cov = np.asarray(ds["cov_jnp"])
    nda = np.asarray(ds["nda"], np.float64)

    N = M0.shape[0]
    if args.n_templates and args.n_templates < N:
        idx = np.asarray(jr.choice(jr.PRNGKey(0), N, (args.n_templates,), replace=False))
        M0, dmdg, d2mdg2, cov, nda = M0[idx], dmdg[idx], d2mdg2[idx], cov[idx], nda[idx]
    print(f"Evaluating {M0.shape[0]:,} templates,  g_true sweep={args.g}"
          f"{'  (nda-weighted)' if args.nda_weight else ''}")

    # sx (Σ_X condition) is derived from cov only ⇒ checkpoint-independent, compute once.
    sx = np.asarray(jax.vmap(cx_to_sx_cond)(even_cov_to_CX(jnp.asarray(cov, jnp.float32))))

    for prior_path in priors:
        r2s = stats_override if stats_override is not None else load_stats(prior_path)
        std_transform = lambda M, _r=r2s: _r.transform_and_log_det(M)[0]
        key, k_flow = jr.split(base_key)
        prior_flow = load_prior_flow(k_flow, prior_path, Q_FLOW_PATH, raw2standard=r2s)
        print(f"\n=== checkpoint: {os.path.basename(prior_path)} ===")

        def pqr_for(g, _flow=prior_flow, _std=std_transform):  # g: (2,) applied shear
            M = _shear_moments(M0, dmdg, d2mdg2, g)
            x_std = np.asarray(jax.vmap(_std)(jnp.asarray(M, jnp.float32)))
            return _noiseless_pqr(_flow, jnp.asarray(x_std), jnp.asarray(sx, jnp.float32))

        pqr_0 = pqr_for([0.0, 0.0])                    # g-independent additive-bias arm
        fin0 = np.all(np.isfinite(pqr_0), 1) & (pqr_0[:, 0] > 1e-10)

        for g in args.g:
            pqr_p = pqr_for([g, 0.0])
            pqr_m = pqr_for([-g, 0.0])
            # keep finite, P>0 rows common to all three arms so they stay row-paired
            good = (fin0 & np.all(np.isfinite(pqr_p), 1) & (pqr_p[:, 0] > 1e-10)
                    & np.all(np.isfinite(pqr_m), 1) & (pqr_m[:, 0] > 1e-10))
            pp, pm, p0 = pqr_p[good], pqr_m[good], pqr_0[good]
            log10mf, ndag = np.log10(M0[good, 0]), nda[good]
            w = ndag if args.nda_weight else None

            if args.save_npz:
                base, ext = os.path.splitext(args.save_npz)
                path = f"{base}_g{g:g}{ext or '.npz'}"
                tgt = M0[good][:, :4]  # paired arms: same templates, [Mf,Mr,M1,M2]
                np.savez(path, pqr_p=pp, pqr_m=pm, targets_p=tgt, targets_m=tgt)
                print(f"  saved {path}  (Δg={2 * g:g})")

            m1, c1, c2, a_m1, _, _ = _recover(pp, pm, g, w)
            g0, _ = _g_cov((ndag[:, None] * _logpqr(p0)).sum(0) if args.nda_weight else _logpqr(p0).sum(0))
            print(f"  g_true={g:g}  (n={pp.shape[0]:,})")
            print(f"    g=0 arm  ĝ = [{g0[0]:+.5f}, {g0[1]:+.5f}]   <- additive bias")
            print(f"    ± lever  m1 = {m1:+.5f} (a{a_m1:.5f})   c1 = {c1:+.5f}   c2 = {c2:+.5f}")

            print("    by log10 Mf:")
            edges = np.quantile(log10mf, np.linspace(0, 1, 6))
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (log10mf >= lo) & (log10mf <= hi)
                if mask.sum() < 50:
                    continue
                wm = ndag[mask] if args.nda_weight else None
                bm1, bc1, bc2, ba, _, _ = _recover(pp[mask], pm[mask], g, wm)
                print(f"      [{lo:5.2f},{hi:5.2f}]  n={mask.sum():>6,}  "
                      f"m1={bm1:+.4f} (a{ba:.4f})  c1={bc1:+.4f}  c2={bc2:+.4f}")


def demo() -> None:
    """Self-check: a linear-in-g toy flow log p = a·(x−g) recovers g_true with m1≈0.

    With log p(x|g) = -½|x-μ-Ag|² the ensemble MLE shear of points drawn at
    x=μ+A·g_true is exactly g_true ⇒ m1=0, and the ± lever helpers invert cleanly.
    """
    rng = np.random.default_rng(0)
    N, gt = 4000, 0.02
    A = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5]])  # dM/dg (4,2)
    mu = rng.normal(size=(N, 4))

    def toy_pqr(g):
        x = mu + (A @ g)  # analytic shear of each "template"
        # log p = -½|x-μ-A g'|²  ⇒ at g'=0: dlp = Aᵀ(x-μ), d2lp = -AᵀA
        d = x - mu
        dlp = d @ A  # (N,2)
        d2 = -(A.T @ A)  # (2,2) constant
        P = np.ones(N)
        Q = P[:, None] * dlp
        R = P[:, None, None] * (d2[None] + dlp[:, :, None] * dlp[:, None, :])
        return np.column_stack([P, Q, R[:, 0, 0], R[:, 1, 1], R[:, 0, 1]])

    pqr_p, pqr_m = toy_pqr(np.array([gt, 0.0])), toy_pqr(np.array([-gt, 0.0]))
    m1, c1, c2, _, _, _ = _recover(_clean(pqr_p), _clean(pqr_m), gt)
    assert abs(m1) < 1e-3, m1
    assert abs(c1) < 1e-3 and abs(c2) < 1e-3, (c1, c2)
    print(f"OK: toy linear flow recovers g_true (m1={m1:+.2e}, c=({c1:+.1e},{c2:+.1e}))")


if __name__ == "__main__":
    main()
