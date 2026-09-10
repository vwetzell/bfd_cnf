"""How noisy is a 100k-template `R_s11`?  Split the 1M catalog into blocks.

`dev/rs_compare.py` quotes the templates' error as their own half-split, which
at 100k templates came out at 0.0004.  If a block-of-100k spread is much larger
than that, the half-split is understating the reference's error and the
"11.8% weak at ~30 sigma" gap has to be requoted.
"""
import os
import numpy as np, jax, jax.numpy as jnp
import bias, shear
from bias import window_prob, load_cov

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
TREF = os.environ.get("TREF", f"{D}/moments_bulgedisc_v2_1M.fits")
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
HI = float(os.environ.get("HI", 9e9))
FD = float(os.environ.get("FD", 0.02))
BLOCK = int(os.environ.get("BLOCK", 100000))

if __name__ == "__main__":
    _flow = None
    if os.environ.get("FLOW"):                 # must be built under x32, as saved
        import equinox as eqx, jax.random as jr, bulk
        _flow = eqx.tree_deserialise_leaves(
            os.environ["FLOW"],
            bulk.build_flow(jr.key(0),
                            shear.load(os.environ.get(
                                "TRAIN", f"{D}/{bias.TRAIN_DATA[POP]}"))[0],
                            shear=True, centroid=True))
    jax.config.update("jax_enable_x64", True)
    cov = load_cov(f"{D}/{bias.CATALOGS[POP]['zero']}.fits")
    size, flux = (-np.inf, HI), (FLUX_LO, 1e9)
    tm, tdm, td2m = (jnp.asarray(v, jnp.float64) for v in shear.load(TREF))
    e = jnp.array([FD, 0.0])

    @jax.jit
    def d2(i):
        f = lambda g: window_prob(
            shear.lens(tm[i], tdm[i], td2m[i], jnp.broadcast_to(g, (len(i), 2))),
            cov, size, flux)
        return (f(e) + f(-e) - 2 * f(jnp.zeros(2))) / FD**2

    idx = jnp.arange(len(tm))
    if os.environ.get("FLOW"):
        import fitsio
        from bias import condition
        flow = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x,
            _flow)
        sx = jnp.asarray(fitsio.read(f"{D}/{bias.CATALOGS[POP]['zero']}.fits",
                                     rows=[0])["cov_odd"][0], dtype=jnp.float64)
        idx = flow.base_dist.sample(
            jr.key(int(os.environ.get("SEED", 31))),
            (int(os.environ.get("NDRAW", 1000000)),)).astype(jnp.float64)

        @jax.jit
        def d2(zz):
            f = lambda g: window_prob(
                jax.vmap(lambda z1: flow.bijection.transform(
                    z1, condition(g, sx)))(zz), cov, size, flux)
            return (f(e) + f(-e) - 2 * f(jnp.zeros(2))) / FD**2

    v = np.concatenate([np.asarray(d2(idx[i:i + 16384]))
                        for i in range(0, len(idx), 16384)])
    nan = int(np.isnan(v).sum())               # a few flow draws leave the domain
    v = v[~np.isnan(v)]
    b = v[:len(v) // BLOCK * BLOCK].reshape(-1, BLOCK).mean(axis=1)
    if nan:
        print(f"  dropped {nan} NaN draws ({nan / (len(v) + nan):.2%})")
    print(f"{os.environ.get('FLOW') or os.path.basename(TREF)}: "
          f"flux >= {FLUX_LO:g}, Mr/Mf <= {HI:g}, fd = {FD}")
    print(f"  full R_s11 = {v.mean():.4f}   (n = {len(v)})")
    print(f"  {len(b)} blocks of {BLOCK}: " + " ".join(f"{x:.4f}" for x in b))
    print(f"  block sd = {b.std(ddof=1):.4f}   sem of full = {v.std(ddof=1) / np.sqrt(len(v)):.4f}")
    h = len(v) // 2
    print(f"  half-split of the full catalog = {abs(v[:h].mean() - v[h:].mean()) / 2:.4f}")
