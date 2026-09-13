"""Is Mr/Mf ~3.0-3.3, Mc/Mr ~6.0-6.2 a genuine fold of the TRUE two-Gaussian
moment map, or only an artifact of the trained flow?

Checkable exactly on gauss2: invert real (noisy, in-population) training
moments in that band back to theta via truth.analytic.theta_of_m_batch, then
look at the analytic forward Jacobian d(moments)/d(theta) there. A genuine
fold shows up as a small singular value / large condition number -- the map
is close to non-invertible independent of any flow. If the analytic Jacobian
is well-conditioned throughout the band, the fold is a flow-only artifact.

Run: python dev/fold_check.py
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
import truth  # noqa: E402
import shear  # noqa: E402

BAND_RATIO = (3.0, 3.3)
BAND_CONC = (6.0, 6.2)
DATA = "../bfd_cnf_imsims/data/moments_gauss2_fwd_g2v3d.fits"


def main():
    m = np.asarray(shear.load(DATA)[0], dtype=np.float64)
    ratio, conc = m[:, 1] / m[:, 0], m[:, 4] / m[:, 1]
    in_band = ((ratio > BAND_RATIO[0]) & (ratio < BAND_RATIO[1])
              & (conc > BAND_CONC[0]) & (conc < BAND_CONC[1]))
    out_band = ~in_band
    print(f"{in_band.sum()} templates in band, {out_band.sum()} outside")

    def one(theta):
        J = jax.jacfwd(lambda t: truth.analytic.moments(t))(theta)
        return jnp.linalg.svd(J, compute_uv=False)

    batched = jax.jit(jax.vmap(one))

    def jac_singvals(m_batch, chunk=512):
        th, resid = truth.analytic.theta_of_m_batch(jnp.asarray(m_batch))
        ok = np.asarray(resid) < 1e-8
        sv = np.concatenate([
            np.asarray(batched(th[i:i + chunk]))
            for i in range(0, len(th), chunk)])
        return sv, ok

    rng = np.random.default_rng(0)
    idx_in = np.where(in_band)[0]
    idx_out = rng.choice(np.where(out_band)[0], size=min(5000, out_band.sum()),
                         replace=False)

    for name, idx in [("IN BAND (leader locus)", idx_in),
                      ("OUTSIDE BAND (control)", idx_out)]:
        sv, ok = jac_singvals(m[idx])
        sv, cond = sv[ok], sv[ok, 0] / sv[ok, -1]
        print(f"\n{name}: {ok.sum()}/{len(idx)} Newton-converged")
        print(f"  smallest singular value: p1={np.percentile(sv[:, -1], 1):.4g} "
              f"p50={np.percentile(sv[:, -1], 50):.4g} "
              f"p99={np.percentile(sv[:, -1], 99):.4g}")
        print(f"  condition number: p50={np.percentile(cond, 50):.4g} "
              f"p99={np.percentile(cond, 99):.4g} max={cond.max():.4g}")


if __name__ == "__main__":
    main()
