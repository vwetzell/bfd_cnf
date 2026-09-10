"""Are the selection terms converged, and at what draw count?

The 2026-09-04 null run found that the ENTIRE Monte-Carlo error budget of a
windowed corrected number lives here, not in the per-target integral: at
2^20 prior draws a seed change moves `Q_s2` by 3.6e-03, the `R_s` traceless
part by 10x, and the corrected `m1` by 4.1e-03 -- four times a `tau` of 1e-3 --
while the per-target `Q`, `R` sum is converged to 1e-08.  No galaxy bootstrap
can see any of that.

`P_s`, `Q_s`, `R_s` depend only on the flow, `C_M`, `Sigma_X` and the window,
never on the targets, so this needs none of `bias.py`'s 20k x 8192 target work.
And `ghat` is affine in them, so a better `R_s` re-solves the bias offline with
no GPU (`dev/qs_impact.py`).

Two exact identities make this self-monitoring, both forced by the window
cutting only spin-0 quantities so that `P_s` can depend on `|g|^2` alone:

    R_s11 = R_s22        ->  the traceless part IS the MC error
    R_s12 = 0
    Q_s   = 0            ->  additionally needs an isotropic population and a
                             circular PSF; true for the `psfe00` config

so a single run reports its own convergence, and the seed-to-seed spread
confirms it.

    python -u dev/selection_convergence.py --draws 20 22 24 --seeds 0 7
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

import bias as B
import bulk
import shear


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_v11.eqx")
    p.add_argument("--pop", default="bulgedisc_v3_psfe00")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--draws", type=int, nargs="*", default=[20, 22, 24],
                   help="log2 of the prior draw count")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 7])
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(2500.0, 50000.0))
    p.add_argument("--batch", type=int, default=4096)
    a = p.parse_args()

    cat = B.CATALOGS[a.pop]
    targets = f"{a.data_dir}/{cat['zero']}.fits"
    cov = B.load_cov(targets)
    sigma_x = jnp.asarray(
        np.asarray(fitsio.read(targets)["cov_odd"], dtype=np.float64)[0])

    # Same construction as `bias.py`: the centroid flow was standardised on the
    # FULL population, so no 0.9 slice here.
    m_train = shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    flow = bulk.SupportedFlow(flow, eps=0.0, support=True)

    size, flux = tuple(a.window_size), tuple(a.window_flux)
    print(f"flow {a.flow}   pop {a.pop}")
    print(f"window: size {size} (Mr/Mf), flux {flux} (Mf)")
    print(f"Sigma_X = {np.asarray(sigma_x)}\n")
    print(f"{'log2 draws':>10} {'seed':>5} {'P_s':>9} {'Q_s1':>11} {'Q_s2':>11} "
          f"{'R_s11':>10} {'R_s22':>10} {'R_s12':>10} {'traceless':>11}")

    zero2 = jnp.zeros(2)
    tr = eqx.filter_jit(jax.vmap(
        lambda z1: flow.bijection.transform(z1, B.condition(zero2, sigma_x))))

    out = {}
    for lg in a.draws:
        n = 1 << lg
        for s in a.seeds:
            # `jr.key(seed + 31)` and the 16384 chunking are bias.py's, so a
            # given (draws, seed) reproduces its run exactly.
            zs = flow.base_dist.sample(jr.key(s + 31), (n,)).astype(jnp.float64)
            m0 = np.concatenate(
                [np.asarray(tr(zs[i:i + 16384])) for i in range(0, n, 16384)])
            del zs
            ps, qs, rs, _ = B.selection_terms_score(
                flow, m0, cov, size, flux, sigma_x=sigma_x, batch=a.batch)
            del m0
            tl = 0.5 * (rs[0, 0] - rs[1, 1])
            out[(lg, s)] = (ps, np.asarray(qs), np.asarray(rs))
            print(f"{lg:>10} {s:>5} {ps:>9.5f} {qs[0]:>+11.3e} {qs[1]:>+11.3e} "
                  f"{rs[0, 0]:>+10.5f} {rs[1, 1]:>+10.5f} {rs[0, 1]:>+10.5f} "
                  f"{tl:>+11.3e}", flush=True)

    if len(a.seeds) > 1:
        print(f"\nseed-to-seed spread (max - min over seeds), the term no "
              f"galaxy bootstrap can see")
        print(f"{'log2 draws':>10} {'d P_s':>10} {'d Q_s2':>11} {'d R_s11':>11} "
              f"{'|traceless|max':>15}")
        for lg in a.draws:
            v = [out[(lg, s)] for s in a.seeds]
            dps = max(x[0] for x in v) - min(x[0] for x in v)
            dq = max(x[1][1] for x in v) - min(x[1][1] for x in v)
            dr = max(x[2][0, 0] for x in v) - min(x[2][0, 0] for x in v)
            tl = max(abs(0.5 * (x[2][0, 0] - x[2][1, 1])) for x in v)
            print(f"{lg:>10} {dps:>10.2e} {dq:>11.2e} {dr:>11.2e} {tl:>15.2e}")

    print("\nThe traceless part is forced to zero by the window being spin-0, "
          "so it\nis a free per-run convergence monitor -- read it before "
          "trusting any\nwindowed CORRECTED number.  Q_s = 0 is forced too "
          "wherever the PSF is\ncircular and the population isotropic.")


if __name__ == "__main__":
    main()
