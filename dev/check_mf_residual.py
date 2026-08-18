"""Where does the dMf/dg discrepancy live?

Inside the target window (`bias.SIZE_WINDOW`, `bias.FLUX_WINDOW`) the first-order
fit sits at these ratios to its own measured Var[Q|m] floor:

    dMf/dg   7.1% vs 1.86%   3.8x     <- this script
    dMr/dg  20.0% vs 15.2%   1.31x
    dMc/dg  33.1% vs 25.6%   1.29x
    dM1/dg1  2.3% vs 1.31%   1.75x

so Mf is the outlier by a factor of three, and it is the column the chain
reorder moved.  Three questions, in order:

1. SYSTEMATIC OR SCATTER?  `resid^2 = (1-slope)^2 RMS_truth^2 + scatter^2`.
   A slope error is a calibration bug; scatter above the floor is a fit that is
   not using information the moments contain.  They have different fixes.

2. WHERE IN THE POPULATION?  Profile the residual against the four quantities
   the coefficient network actually sees -- z0 (flux), Mr/Mf, Mc Mf/Mr^2, |e|^2
   -- reporting per bin the MEAN residual (systematic) separately from its RMS,
   and each bin's share of the total squared residual.

3. IS IT TRADED AGAINST THE OTHER MOMENTS?  The spin-0 block writes one 3x3
   `s0` for (z0, z1, z2) = (log Mf, logit Mr/Mf, logit Mc/Mr), so an error in
   the flux row also moves Mr and Mc through the ratios.  If the layer is
   trading them off, the residuals correlate.  That was the first explanation
   offered for this discrepancy and it was never actually tested.

    python dev/check_mf_residual.py [--flow flows/shear_cfo.eqx]

ANSWER (2026-08-17), per-partial-supervised flow, inside the target window.

1. SCATTER, NOT CALIBRATION.  slope 1.0348, total resid 6.99%, of which 3.48% is
   the slope and 6.06% is scatter -- 3.3x the 1.86% floor.  The layer is not
   merely miscalibrated; it is not using information the moments carry.

2. IT IS A POINT-SOURCE-END FAILURE, and the floor makes it far worse than the
   raw residual suggests, because the floor moves the OTHER WAY:

       Mr/Mf <     resid    floor    ratio    share of sum d^2
         2.880      4.7%    2.37%     2.0x        7.5%
         3.130      3.9%    1.82%     2.1x        5.3%
         3.271      5.8%    1.36%     4.3x       11.6%
         3.361      8.1%    1.05%     7.7x       22.5%
         3.430      9.0%    0.92%     9.8x       27.5%
         3.500      8.7%    0.94%     9.3x       25.7%

   At the point-source end the Mf response is MORE determined by the moments
   than anywhere else, and that is exactly where the layer is worst -- ~10x its
   achievable limit against 2x at the diffuse end.  The top third by size carries
   53% of the squared residual while having the easiest response to predict.

   Flat in flux (5.6% to 7.3% across the whole window), which closes the
   flux-blindness thread for good.  Concentrated at LOW k = Mc Mf/Mr^2, which is
   correlated with high Mr/Mf and probably the same galaxies.

3. NOT TRADED AGAINST THE OTHER MOMENTS.  Residual correlations: Mf-Mr +0.146,
   Mf-Mc +0.128, Mf-M1 +0.005, Mf-M2 +0.007.  The "one 3x3 s0 shared across the
   spin-0 block, so Mf absorbs the compromise" story was the first explanation
   offered for this discrepancy, was never tested, and is not supported.
"""
import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402
from models.shear import dm_dg                       # noqa: E402

LAB = ["Mf", "Mr", "M1", "M2", "Mc"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear_cfo.eqx")
    p.add_argument("--data", default="../bfd_cnf_imsims/data/moments.fits")
    p.add_argument("-n", type=int, default=40000)
    p.add_argument("--nbin", type=int, default=6)
    a = p.parse_args()

    full = shear_top.load(a.data)
    m, q_true, _ = (np.asarray(x[:a.n], np.float64) for x in full)
    flow = bulk.build_flow(jr.key(0), full[0], shear=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)

    r_mf = m[:, 1] / m[:, 0]
    keep = ((r_mf >= bias.SIZE_WINDOW[0]) & (r_mf <= bias.SIZE_WINDOW[1])
            & (m[:, 0] >= bias.FLUX_WINDOW[0]) & (m[:, 0] <= bias.FLUX_WINDOW[1]))
    m, q_true = m[keep], q_true[keep]
    print(f"{a.flow}: {keep.sum()} of {len(keep)} templates in the target window")

    q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(
        shear_top._shear_layer(flow), jnp.asarray(m, jnp.float32),
        shear_top._chart(flow))
    q = np.asarray(q, np.float64)
    s = np.stack([m[:, 0], m[:, 1], m[:, 1], m[:, 1], m[:, 4]], -1)

    # residual on dMf/dg, both components pooled (they are symmetric)
    d = np.concatenate([(q[:, 0, 0] - q_true[:, 0, 0]) / s[:, 0],
                        (q[:, 1, 0] - q_true[:, 1, 0]) / s[:, 0]])
    t = np.concatenate([q_true[:, 0, 0] / s[:, 0], q_true[:, 1, 0] / s[:, 0]])
    rms_t = np.sqrt(np.mean(t * t))
    slope = np.mean(np.concatenate([q[:, 0, 0] / s[:, 0], q[:, 1, 0] / s[:, 0]]) * t) \
        / np.mean(t * t)

    # ---- 1. systematic vs scatter
    resid = np.sqrt(np.mean(d * d)) / rms_t
    syst = abs(1.0 - slope)
    scat = np.sqrt(max(0.0, resid ** 2 - syst ** 2))
    floor = shear_top.response_scatter.__doc__ and None
    print("\n=== 1. systematic vs scatter, dMf/dg ===")
    print(f"  slope                {slope:.4f}")
    print(f"  total resid/RMS      {resid:.2%}")
    print(f"  from the slope       {syst:.2%}   (calibration)")
    print(f"  scatter              {scat:.2%}   (information not used)")
    print(f"  measured floor       1.86%        (`shear.py scatter`, in-window)")
    print(f"  -> scatter is {scat / 0.0186:.1f}x the floor")

    # ---- 2. where
    k = m[:, 4] * m[:, 0] / m[:, 1] ** 2
    e2 = ((m[:, 2] ** 2 + m[:, 3] ** 2) / m[:, 1] ** 2)
    axes = [("log10 Mf", np.log10(m[:, 0])), ("Mr/Mf", m[:, 1] / m[:, 0]),
            ("k = McMf/Mr^2", k), ("|e|", np.sqrt(e2))]
    print("\n=== 2. where in the population ===")
    print("  mean = systematic part in that bin; rms = total; share = of sum d^2")
    for name, v in axes:
        vv = np.concatenate([v, v])
        ed = np.quantile(vv, np.linspace(0, 1, a.nbin + 1))
        ed[0], ed[-1] = -np.inf, np.inf
        print(f"\n  {name:>16s} {'n':>7s} {'mean':>10s} {'rms':>10s} "
              f"{'rms/RMSt':>9s} {'share':>7s}")
        for i in range(a.nbin):
            sel = (vv >= ed[i]) & (vv < ed[i + 1])
            if sel.sum() < 50:
                continue
            print(f"  {'<' + f'{min(ed[i+1], vv.max()):.3f}':>16s} {sel.sum():7d} "
                  f"{d[sel].mean():+10.2e} {np.sqrt(np.mean(d[sel]**2)):10.2e} "
                  f"{np.sqrt(np.mean(d[sel]**2))/rms_t:9.1%} "
                  f"{np.sum(d[sel]**2)/np.sum(d**2):7.1%}")

    # ---- 3. traded against the other moments?
    print("\n=== 3. correlation with the other moments' residuals ===")
    print("  the spin-0 block writes one 3x3 s0 for (log Mf, logit Mr/Mf,")
    print("  logit Mc/Mr), so a flux-row error moves Mr and Mc through the ratios")
    dj = {}
    for j, lab in enumerate(LAB):
        dj[lab] = np.concatenate([(q[:, 0, j] - q_true[:, 0, j]) / s[:, j],
                                  (q[:, 1, j] - q_true[:, 1, j]) / s[:, j]])
    print(f"  {'':6s}" + "".join(f"{l:>9s}" for l in LAB))
    for l1 in LAB:
        print(f"  {l1:6s}" + "".join(
            f"{np.corrcoef(dj[l1], dj[l2])[0,1]:+9.3f}" for l2 in LAB))


if __name__ == "__main__":
    main()
