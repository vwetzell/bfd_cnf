"""Bisect GAUSS2_FWD_SIZE_LOGMEDIAN so the noiseless gauss2_fwd population's
median Mr/Mf matches the OLD calibrated population's; also
prints Mc/Mr and log10 Mf.  Run after changing GAUSS2_FWD_ELLIP_SIGMA.
Usage: JAX_PLATFORMS=cpu python dev/gauss2_calib.py [ellip_sigma]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "bfd_cnf_imsims"))
import numpy as np, jax.numpy as jnp
from imsims import sim, analytic

N = 40000
OLD = (-0.642148, 0.869685)   # the calibrated population being preserved


def stats(logmed, se, seed=0):
    p = sim.sample_population_gauss2_fwd(N, np.random.default_rng(seed), logmed, se)
    th = np.stack([np.log(p["flux"]), np.log(p["sigma"]),
                   np.log(p["bulge_ratio"]) - np.log1p(-p["bulge_ratio"]),
                   p["e1"], p["e2"]], 1)
    m = np.asarray(analytic.moments_batch(jnp.asarray(th)))
    return dict(mr_mf=np.median(m[:, 1] / m[:, 0]), mc_mr=np.median(m[:, 4] / m[:, 1]),
                log10_mf=np.median(np.log10(m[:, 0])))


if __name__ == "__main__":
    se = float(sys.argv[1]) if len(sys.argv) > 1 else sim.GAUSS2_FWD_ELLIP_SIGMA
    new_e = sim._ellipticity
    sim._ellipticity = sim._ellipticity_wide      # the sampler used to draw this
    TARGET = stats(*OLD)   # noiseless medians of the OLD population, same method
    sim._ellipticity = new_e
    print("target (old population):", TARGET)
    lo, hi = -1.5, 0.5   # Mr/Mf FALLS as size grows
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if stats(mid, se)["mr_mf"] > TARGET["mr_mf"]: lo = mid
        else: hi = mid
    mid = 0.5 * (lo + hi)
    print(f"sigma_e={se}  size_logmedian={mid:.6f}  ->", stats(mid, se, seed=1))
