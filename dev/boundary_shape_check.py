"""Is the reachable-set boundary (where log_p_theta flips from finite to
-inf) a simple function of the OBSERVED shape invariants alone -- i.e. is it
something a data-driven support mask (built only from moments, the way
`build_support_density` already is) could in principle learn -- or does
reachability depend on directions in moment-space that a `(Mr/Mf, Mc/Mr)`-
style projection can't see?

Draws a broad sample from the trained flow (naturally concentrated near the
real population, so it straddles the boundary), inverts each to theta via the
EXACT two-Gaussian model, and checks whether membership (log_p_theta finite
or -inf) is predictable from `(Mr/Mf, Mc/Mr)` alone, or whether points with
the same `(Mr/Mf, Mc/Mr)` split both ways depending on ellipticity/flux.

Run: python dev/boundary_shape_check.py --log2-draws 20
"""
import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402
# NOT `truth` here: it flips jax_enable_x64 at IMPORT time, before the flow
# skeleton (float32, matching the on-disk checkpoint) gets built below.


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pop", default="gauss2_v3d")
    p.add_argument("--flow", default="flows/shear_g2v3d_full2.eqx")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--log2-draws", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    n = 1 << a.log2_draws
    m_train = shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=False)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    import truth  # noqa: E402

    zs = flow.base_dist.sample(jr.key(a.seed + 31), (n,)).astype(jnp.float64)
    tr = eqx.filter_jit(jax.vmap(lambda z1: flow.bijection.transform(
        z1, B.condition(jnp.zeros(2), None))))
    m0 = np.concatenate([np.asarray(tr(zs[i:i + 16384]))
                         for i in range(0, n, 16384)])
    del zs

    ratio, conc = m0[:, 1] / m0[:, 0], m0[:, 4] / m0[:, 1]
    band = (ratio > 2.7) & (ratio < 3.5)
    m0, ratio, conc = m0[band], ratio[band], conc[band]
    print(f"{len(m0)} draws with Mr/Mf in (2.7, 3.5)")

    th, resid = truth.analytic.theta_of_m_batch(jnp.asarray(m0), batch_size=2048)
    resid = np.asarray(resid)
    lpt = np.asarray(truth.log_p_theta(th))
    reachable = np.isfinite(lpt) & (resid < 1e-8)
    print(f"reachable fraction overall: {reachable.mean():.3f}")

    ell = np.hypot(m0[:, 2], m0[:, 3]) / m0[:, 1]     # |e|
    logf = np.log10(m0[:, 0])

    # Bin the 2D (Mr/Mf, Mc/Mr) plane; a bin is HOMOGENEOUS if it's
    # (nearly) all-reachable or all-unreachable, MIXED if both appear --
    # mixed bins are where (Mr/Mf, Mc/Mr) alone fails to predict reachability.
    nb = 40
    r_edges = np.linspace(2.7, 3.5, nb + 1)
    c_edges = np.linspace(np.percentile(conc, 1), np.percentile(conc, 99), nb + 1)
    ri = np.clip(np.digitize(ratio, r_edges) - 1, 0, nb - 1)
    ci = np.clip(np.digitize(conc, c_edges) - 1, 0, nb - 1)
    bin_id = ri * nb + ci

    mixed_frac_list = []
    n_bins_used = 0
    ell_gap_list = []
    for b in np.unique(bin_id):
        sel = bin_id == b
        if sel.sum() < 15:
            continue
        n_bins_used += 1
        frac = reachable[sel].mean()
        purity = max(frac, 1 - frac)
        mixed_frac_list.append(1 - purity)
        if 0 < frac < 1:
            # Within this (ratio, conc) bin, does reachability separate by
            # ellipticity or flux instead?
            e_r, e_u = ell[sel][reachable[sel]], ell[sel][~reachable[sel]]
            f_r, f_u = logf[sel][reachable[sel]], logf[sel][~reachable[sel]]
            ell_gap_list.append((np.median(e_r) - np.median(e_u),
                                 np.median(f_r) - np.median(f_u)))

    mixed_frac = np.array(mixed_frac_list)
    print(f"\n{n_bins_used} (Mr/Mf, Mc/Mr) bins with >=15 draws")
    print(f"bin impurity (0=pure, 0.5=maximally mixed): "
          f"mean {mixed_frac.mean():.3f}  median {np.median(mixed_frac):.3f}  "
          f"frac of bins with impurity>0.1: {(mixed_frac > 0.1).mean():.3f}")

    if ell_gap_list:
        eg = np.array(ell_gap_list)
        print(f"\nin the {len(eg)} genuinely mixed bins, median(|e|_reachable - "
              f"|e|_unreachable) = {np.median(eg[:, 0]):+.4f}  "
              f"median(log10 Mf gap) = {np.median(eg[:, 1]):+.4f}")
        print("(a nonzero, consistently-signed gap means ellipticity/flux "
              "DOES carry information (Mr/Mf, Mc/Mr) alone misses)")


if __name__ == "__main__":
    main()
