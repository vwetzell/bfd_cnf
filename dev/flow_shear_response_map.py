"""Map the trained prior flow's LEARNED shear response vs analytic BFD, over the
(log10 Mf, Mr/Mf) plane, with explicit isolation of the SigmaXCoupling layer's
g-modulation.

Facts this rests on (from the code, verified — not from other dev/ scripts):
  * g conditions ONLY ExplicitPolyLast; the early layers are unconditional.
  * data->base order: early -> ExplicitPolyLast(g) -> SigmaX(Sigma_X) -> base.
    SigmaX maps ellipticity (m2,m3) -> (1/kappa)[(I+cE)(m2,m3)+D e], kappa=exp(g_s).
    ExplicitPolyLast injects the shear shift shared_c*g on (m2,m3); SigmaX then
    divides it by kappa, so the base score d/dg1 log p picks up ~1/kappa^2.
    => the effective shear response is modulated by a flux/Sigma_X-dependent factor
       even though SigmaX ignores g in its coefficients.

Pure point-response (autodiff of log p at g=0); NO kernel integration, so this
isolates the flow's intrinsic learned response from the RQMC integrator.

  K   = d^2 log p / (dg1 dM1_std)  at ellipticity 0   -> the linear (Q) response slope
  Rc  = d^2 log p / dg1^2          at ellipticity 0   -> the curvature (R) response
  _noSX: same with SigmaX neutralized to identity     -> ExplicitPolyLast alone
  g_s : the SigmaX size log-scale (kappa=exp(g_s)); modulation ~ exp(-2 g_s)

Run:  JAX_PLATFORMS=cpu PYTHONPATH=. python dev/flow_shear_response_map.py
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from bfd_cnf.integrate_grid import load_prior_flow, load_stats
from bfd_cnf.models.bijections import SigmaXCouplingLayer

FLOW = "flows/prior_flow_xy_nll.eqx"
QFLOW = "flows/q_flow_xy.eqx"
NPZ = "data/pqr_grid_100k_nll.npz"

flow = load_prior_flow(jax.random.PRNGKey(0), FLOW, QFLOW)
r2s = load_stats(FLOW)
mean = np.asarray(r2s.mean); std = np.asarray(r2s.std)


# ---- locate + neutralize the SigmaX layer (identity) ----------------------
def _is_sx(m):
    return isinstance(m, SigmaXCouplingLayer)


def _get_sx(tree):
    return next(l for l in jax.tree_util.tree_leaves(tree, is_leaf=_is_sx) if _is_sx(l))


def _zero_last(net):
    last = net.layers[-1]
    return eqx.tree_at(
        lambda n: (n.layers[-1].weight, n.layers[-1].bias),
        net,
        (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)),
    )


def _neuter(sx):
    return eqx.tree_at(
        lambda L: (L.net_flux, L.net_size, L.net_dipquad),
        sx,
        tuple(_zero_last(n) for n in (sx.net_flux, sx.net_size, sx.net_dipquad)),
    )


sx_layer = _get_sx(flow)
flow_noSX = jax.tree_util.tree_map(
    lambda m: _neuter(m) if _is_sx(m) else m, flow, is_leaf=_is_sx
)


# ---- response functionals (autodiff of log p at g=0) ----------------------
def _logp(f, x, g1, g2, sx):
    return f.log_prob(x, condition=jnp.concatenate([jnp.array([g1, g2]), sx]))


def responses(f, x, sx):
    # K = d/dM1 ( d/dg1 log p )  at g=0   (linear/Q response slope in ellipticity)
    dg1 = lambda xx: jax.grad(lambda g: _logp(f, xx, g, 0.0, sx))(0.0)
    K = jax.grad(lambda xx: dg1(xx))(x)[2]                    # d(score)/dM1_std
    # Rc = d^2/dg1^2 log p  at g=0   (curvature/R response)
    Rc = jax.grad(lambda g: jax.grad(lambda gg: _logp(f, x, gg, 0.0, sx))(g))(0.0)
    return K, Rc


K_full = jax.jit(lambda x, sx: responses(flow, x, sx))
K_nosx = jax.jit(lambda x, sx: responses(flow_noSX, x, sx))


def g_s_of(x0, x1, sx):
    ls_n, e1, e2, e2sq, e2sq_n, T_n = sx_layer._unpack(jnp.array([0.0, 0.0, *sx]))
    _, g_s, D, c = sx_layer._coeffs(x0, x1, ls_n, e2sq_n, T_n)
    return float(g_s), float(jnp.exp(g_s))


# ---- analytic reference: bin grid pqr_sim Q1/P over (flux,size) ------------
d = np.load(NPZ)
tg = d["targets_p"].astype(float)
ps = d["pqr_sim_p"].astype(float)                    # analytic (integrated)
pf = d["pqr_p"].astype(float)                        # flow     (integrated)
okv = np.isfinite(ps).all(1) & (ps[:, 0] > 1e-10) & np.isfinite(pf).all(1) & (pf[:, 0] > 1e-10)
fl = np.log10(tg[:, 0]); sz = tg[:, 1] / tg[:, 0]


def cell_int(f_lo, f_hi, s_lo, s_hi):
    """Integrated Q_tot and R_tot (the quantities that enter g_hat=R^-1 Q) summed
    over the targets in the cell, for analytic (BFD) and flow — apples-to-apples."""
    m = okv & (fl >= f_lo) & (fl < f_hi) & (sz >= s_lo) & (sz < s_hi)
    if m.sum() < 30:
        return (np.nan,) * 4 + (0,)

    def tot(pqr):
        P, Q1, R11 = pqr[m, 0], pqr[m, 1], pqr[m, 3]
        Qt = np.sum(Q1 / P)                       # sum d logP/dg1
        Rt = np.sum((Q1 / P) ** 2 - R11 / P)      # sum -d^2 logP/dg1^2
        return Qt, Rt

    Qa, Ra = tot(ps)
    Qf, Rf = tot(pf)
    return Qa, Qf, Ra, Rf, int(m.sum())


# ---- grid: full selection box 1500<Mf<90000, 2.2<Mr/Mf<3.5 -----------------
# NOTE Fourier moments: SMALL Mr/Mf = LARGE (extended) galaxy; LARGE Mr/Mf = compact.
FEDGES = np.linspace(np.log10(1500.0), np.log10(90000.0), 7)      # 6 flux bins
SEDGES = np.linspace(2.2, 3.5, 5)                                 # 4 size bins
FLUX = 0.5 * (FEDGES[:-1] + FEDGES[1:])
SIZE = 0.5 * (SEDGES[:-1] + SEDGES[1:])
LOGSCALES = [11.0, 11.4, 12.0, 12.7]

print(f"standardiser mean={mean}  std={std}")
print(f"c1 (SigmaX locked-size loc) = mean1/std1 = {mean[1]/std[1]:.3f}")
print("cols: SXmod=SigmaX g-modulation of the Q slope; Kpt/Rpt=flow point response "
      "(slope, curvature); then INTEGRATED sums flow/analytic for Q_tot and R_tot,\n"
      "      net=(Qf/Qa)/(Rf/Ra)=local g_hat over-response (net-1 ~ local m).  "
      "Mr/Mf small = large galaxy.\n")

for ls in LOGSCALES:
    sx = jnp.array([ls, 0.0, 0.0])
    print(f"============ log_scale = {ls}  (sigma_xy = {np.exp(ls/2):.0f}) ============")
    print(f"{'log Mf':>6} {'Mr/Mf':>6} | {'SXmod':>6} {'kappa':>6} | "
          f"{'Kpt':>7} {'Rpt':>7} | {'Qf/Qa':>6} {'Rf/Ra':>6} {'net':>6} {'n':>6}")
    Qa_t = Qf_t = Ra_t = Rf_t = 0.0
    for i, fc in enumerate(FLUX):
        f_lo, f_hi = FEDGES[i], FEDGES[i + 1]
        x0 = (fc - mean[0]) / std[0]
        for j, sc in enumerate(SIZE):
            s_lo, s_hi = SEDGES[j], SEDGES[j + 1]
            x1 = (sc - mean[1]) / std[1]
            x = jnp.array([x0, x1, 0.0, 0.0])
            Kf, Rcf = K_full(x, sx); Kf = float(Kf); Rcf = float(Rcf)
            Kn, _ = K_nosx(x, sx); Kn = float(Kn)
            _, kap = g_s_of(x0, x1, (ls, 0.0, 0.0))
            Qa, Qf, Ra, Rf, n = cell_int(f_lo, f_hi, s_lo, s_hi)
            sxmod = Kf / Kn if abs(Kn) > 1e-12 else np.nan
            if n > 0:
                Qa_t += Qa; Qf_t += Qf; Ra_t += Ra; Rf_t += Rf
            qr = Qf / Qa if (n > 0 and abs(Qa) > 1e-12) else np.nan
            rr = Rf / Ra if (n > 0 and abs(Ra) > 1e-12) else np.nan
            net = qr / rr if (n > 0 and abs(rr) > 1e-12) else np.nan
            print(f"{fc:6.2f} {sc:6.2f} | {sxmod:6.3f} {kap:6.3f} | "
                  f"{Kf:7.2f} {Rcf:7.1f} | {qr:6.3f} {rr:6.3f} {net:6.3f} {n:6d}")
    if abs(Qa_t) > 0 and abs(Ra_t) > 0:
        net_all = (Qf_t / Qa_t) / (Rf_t / Ra_t)
        print(f"  ENSEMBLE over this box: Qf/Qa={Qf_t/Qa_t:.3f}  Rf/Ra={Rf_t/Ra_t:.3f}  "
              f"net response={net_all:.3f}  (net-1 = local m ~ {net_all-1:+.3f})")
    print()
