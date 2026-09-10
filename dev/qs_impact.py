"""What does `Q_s != 0` actually cost, and what does forcing it to zero buy?

`ghat` is affine in the selection terms

    Q = sum q - n_ns Q_s/(1 - P_s)
    R = -sum r + n_ns (Q_s Q_s^T/(1 - P_s)^2 + R_s/(1 - P_s))

so any selection variant is a re-solve off `pqr/*.npz`, with no GPU.  Three
variants, all on the SAME per-target Q, R and the same window:

    naive       P_s, Q_s, R_s exactly as `bias.py` computed them
    Q_s = 0     isotropy forces it (see `dev/isotropy_check.py`)
    symmetrised isotropy also forces R_s = (tr R_s / 2) I

The selection terms themselves are recomputed here at the run's own draw seed
(`jr.key(seed + 31)`) so the full R_s matrix is on the record -- `bias.py`
prints only P_s and Q_s.

    python -u dev/qs_impact.py --draws 1048576
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

import bias as B
import bulk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_v10.eqx")
    p.add_argument("--pqr", default="pqr/v10_score.npz")
    p.add_argument("--pop", default="bulgedisc_v3")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--draws", type=int, default=1048576)
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(2500.0, 50000.0))
    p.add_argument("--g", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import fitsio
    train = f"{a.data_dir}/moments_{a.pop}.fits"
    targets = f"{a.data_dir}/targets_{a.pop.replace('bulgedisc_', '')}_g0_200k.fits"
    cov = B.load_cov(targets)
    sigma_x = jnp.asarray(
        np.asarray(fitsio.read(targets)["cov_odd"], dtype=np.float64)[0])

    m_train = bulk.load_moments(train)
    flow = bulk.build_flow(jr.key(a.seed), m_train[:int(0.9 * len(m_train))],
                           shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    flow = bulk.SupportedFlow(flow, eps=0.0)

    size, flux = tuple(a.window_size), tuple(a.window_flux)
    zs = flow.base_dist.sample(jr.key(a.seed + 31), (a.draws,)).astype(jnp.float64)
    tr = eqx.filter_jit(jax.vmap(
        lambda z1: flow.bijection.transform(z1, B.condition(jnp.zeros(2), sigma_x))))
    m_draw = jnp.concatenate([tr(zs[i:i + 16384]) for i in range(0, len(zs), 16384)])
    ps, qs, rs, qs_err = B.selection_terms_score(flow, m_draw, cov, size, flux,
                                                 sigma_x=sigma_x)
    print(f"\n{a.draws} draws, seed {a.seed}")
    print(f"P_s  = {ps:.4f}")
    print(f"Q_s  = ({qs[0]:+.4e}, {qs[1]:+.4e})  +/- ({qs_err[0]:.1e}, "
          f"{qs_err[1]:.1e})   z = ({qs[0]/qs_err[0]:+.2f}, "
          f"{qs[1]/qs_err[1]:+.2f})")
    print(f"R_s  = [[{rs[0,0]:+.5f} {rs[0,1]:+.5f}] [{rs[1,0]:+.5f} "
          f"{rs[1,1]:+.5f}]]")
    tr2 = 0.5 * (rs[0, 0] + rs[1, 1])
    print(f"       isotropy forces R_s11 = R_s22 (split {2*abs(rs[0,0]-rs[1,1])/abs(rs[0,0]+rs[1,1]):.1%}) "
          f"and R_s12 = 0 ({rs[0,1]/abs(tr2):+.2%} of the diagonal)")

    d = np.load(a.pqr)
    qp, rp = d["plus_q"], d["plus_r"]
    qm, rm = d["minus_q"], d["minus_r"]
    sp = B.window_mask(d["obs_plus"], size, flux)
    sm = B.window_mask(d["obs_minus"], size, flux)
    n_p, n_m = int((~sp).sum()), int((~sm).sum())
    print(f"\n{sp.sum()}/{len(sp)} plus, {sm.sum()}/{len(sm)} minus in window "
          f"(n_ns = {n_p}, {n_m})")

    variants = {
        "naive (as run)": (qs, rs),
        "Q_s = 0": (np.zeros(2), rs),
        "Q_s = 0, R_s isotropised": (np.zeros(2), tr2 * np.eye(2)),
        "uncorrected (no ns)": None,
    }
    print(f"\n{'variant':>28s}{'m1':>12s}{'c1':>12s}{'c2':>12s}")
    for name, v in variants.items():
        ns = None if v is None else ((n_p, ps, v[0], v[1]),
                                     (n_m, ps, v[0], v[1]))
        m1, c1, c2 = B.bias(qp, rp, qm, rm, a.g, sel=(sp, sm), ns=ns)
        print(f"{name:>28s}{m1:+12.5f}{c1:+12.2e}{c2:+12.2e}")
    print("\n(Bootstrap bars are unchanged by this; the per-target Q, R are "
          "identical.\n An identical rebuild moves noisy m1 by sd 0.0008.)")


if __name__ == "__main__":
    main()
