"""Joint per-galaxy test at the size edges: does the flow get each galaxy's
window-selection response right?

Real-vs-closed windowed m1 differs only at the Mr/Mf window edges, but every
ingredient checked alone (spin-0 transport, density, e^2, Mr/Mf noise) matches
([[real-vs-closed-gap-is-size-edge]]).  So test the whole chain on the SAME
galaxies.  Real galaxies whose noiseless Mr/Mf sits near 2.2 or 3.2 are
rendered at g1 = +/-DELTA with CRN noise fields, recentred as imsims does, and
the window applied to the observed moments.  Per galaxy:
    dP/dg    = d/dg1 P(in window)
    dS/dg    = d/dg1 E[e1 * 1_window]        e1 = M1/Mr   (what enters <e1>)
Model predictions for the same galaxy, with Gaussian C_M noise (CRN too):
    flow  : chart^-1 layer^-1(shear^-1(u, g), g),  u = shear(chart(m_nl0), 0)
    exact : chart^-1 layer^-1(chart(m_nl(g)), g)   (exact shear, flow layer)
flow vs exact isolates the shear layer; exact vs real the layer + noise model.
"""
import multiprocessing as mp
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "dev")
sys.path.insert(0, "../bfd_cnf_imsims")
import numpy as np

D = "../bfd_cnf_imsims/data"
DELTA = 0.02
GS = [(DELTA, 0.0), (-DELTA, 0.0)]
N_UP, N_LO, N_REAL = (int(a) for a in (sys.argv[1:4] if len(sys.argv) > 3 else (128, 64, 400)))
N_MODEL = 20000
SIZE, FLUX = (2.2, 3.2), (3000.0, 20000.0)


def one(args):
    p, seed = args
    import bfd
    from imsims import sim
    _, draw = sim.POPULATIONS["gauss2_fwd"]
    wt = bfd.KBlackmanHarris(weightSigma=sim.WEIGHT_SIGMA)
    imgs = [draw(p, shear=g) for g in GS]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_REAL):
        n = rng.normal(0.0, sim.NOISE_SIGMA, imgs[0].shape)
        r = []
        for im in imgs:
            row = np.empty(1, dtype=sim.ROW_DTYPE)[0]
            sim.fill_row(row, sim._measure(im + n, wt, sim.NOISE_SIGMA))
            r.append(row["moments"] if not row["badcenter"] else np.full(5, np.nan))
        out.append(r)
    return np.array(out)                                   # (N_REAL, 2 arms, 5)


def stats(m):
    """m (..., n, 2 arms, 5) -> per-galaxy (dP/dg, dS/dg), NaN rows = outside."""
    import bias as B
    ok = np.isfinite(m).all(-1)
    mm = np.where(ok[..., None], m, 1.0)
    w = B.window_mask(mm.reshape(-1, 5), SIZE, FLUX).reshape(ok.shape) & ok
    e1 = mm[..., 2] / mm[..., 1]
    dP = (w[..., 0].astype(float) - w[..., 1]).mean(-1) / (2 * DELTA)
    dS = (e1[..., 0] * w[..., 0] - e1[..., 1] * w[..., 1]).mean(-1) / (2 * DELTA)
    return dP, dS


def main():
    import fitsio
    import bias as B
    path = f"{D}/{B.CATALOGS['gauss2_v4n']['zero']}.fits"
    t = fitsio.read(path)
    pop = fitsio.read(path, ext="POPULATION")
    pop = pop.astype(pop.dtype.newbyteorder("="))
    # noiseless g = 0 moments to pick edge galaxies, in a child so JAX x64 stays out of here
    with mp.get_context("spawn").Pool(1) as p:
        m0 = p.apply(analytic_moments, (pop, (0.0, 0.0)))
    r0 = m0[:, 1] / m0[:, 0]
    fin = (m0[:, 0] > 3300) & (m0[:, 0] < 18000)
    rng = np.random.default_rng(0)
    up = rng.choice(np.flatnonzero(fin & (np.abs(r0 - 3.2) < 0.12)), N_UP, replace=False)
    lo = rng.choice(np.flatnonzero(fin & (np.abs(r0 - 2.2) < 0.12)), N_LO, replace=False)
    pick = np.concatenate([up, lo])
    with mp.get_context("spawn").Pool(12) as p:
        real = np.array(p.map(one, [(pop[i], 5000 + k) for k, i in enumerate(pick)]))
        mg = [p.apply(analytic_moments, (pop[pick], g)) for g in GS]
    np.savez("dev/edge_selection_real.npz", real=real, pick=pick)

    import jax
    import jax.numpy as jnp
    import centroid as C
    from closed_loop import load_flow
    flow = load_flow("flows/centroid_g2v4n_Ke.eqx")
    chain = flow.bijection.bijection.bijections
    chart, layer = chain[0], C._sigmax_layer(flow)
    sh = next(b for b in chain if type(b).__name__ == "ShearResponse")
    sx = jnp.asarray(t["cov_odd"][0], jnp.float32)
    c = lambda g: B.condition(jnp.asarray(g, jnp.float32), sx)

    @jax.jit
    def full(m, g):
        u = sh.transform(chart.transform(m), c((0.0, 0.0)))
        return chart.inverse(layer.inverse(sh.inverse(u, c(g)), c(g)))

    @jax.jit
    def exact(m, g):
        return chart.inverse(layer.inverse(chart.transform(m), c(g)))

    f32 = lambda a: jnp.asarray(a, jnp.float32)
    mu_full = np.stack([np.asarray(jax.vmap(lambda m: full(m, g))(f32(m0[pick])), np.float64) for g in GS], 1)
    mu_ex = np.stack([np.asarray(jax.vmap(lambda m: exact(m, g))(f32(mg[a])), np.float64)
                      for a, g in enumerate(GS)], 1)
    noise = np.random.default_rng(1).multivariate_normal(np.zeros(5), B.load_cov(path), N_MODEL)
    model = {k: stats(mu[:, None] + noise[None, :, None]) for k, mu in (("flow", mu_full), ("exact", mu_ex))}
    model["real"] = stats(real)

    for name, sl in (("upper edge 3.2", slice(0, N_UP)), ("lower edge 2.2", slice(N_UP, None))):
        n = sl.stop - sl.start if sl.stop else len(pick) - sl.start
        print(f"\n{name}: {n} galaxies, noiseless Mr/Mf within 0.12, {N_REAL} real / {N_MODEL} model noise fields")
        rr = {k: (v[0][sl], v[1][sl]) for k, v in model.items()}
        for j, lab in ((0, "sum dP/dg1"), (1, "sum dS/dg1 = d E[e1 1_w]/dg1")):
            d_rf = rr["real"][j] - rr["flow"][j]
            d_re = rr["real"][j] - rr["exact"][j]
            d_ef = rr["exact"][j] - rr["flow"][j]
            se = lambda x: x.std() * np.sqrt(len(x))
            print(f"  {lab:30s} real {rr['real'][j].sum():+9.2f}  exact {rr['exact'][j].sum():+9.2f}  "
                  f"flow {rr['flow'][j].sum():+9.2f}   real-flow {d_rf.sum():+7.2f}+/-{se(d_rf):.2f}  "
                  f"real-exact {d_re.sum():+7.2f}+/-{se(d_re):.2f}  exact-flow {d_ef.sum():+7.2f}+/-{se(d_ef):.2f}")


def analytic_moments(pop, g):
    import jax.numpy as jnp
    from imsims import analytic
    rho = pop["bulge_ratio"]
    th = jnp.stack([jnp.log(pop["flux"]), jnp.log(pop["sigma"]), jnp.log(rho) - jnp.log1p(-rho),
                    pop["e1"], pop["e2"]], -1)
    return np.asarray(analytic.moments_batch(th, g), np.float64)


if __name__ == "__main__":
    main()
