"""Per-galaxy mean shear response: real noisy recentring vs the flow's layer.

The model says a target's observed moments are
    M_obs = chart^-1( layer^-1( chart(m_nl(g)), [g, Sigma_X] ) ) + N(0, C_M)
with m_nl(g) the galaxy's noiseless moments about its true centre and layer the
SigmaXBlockLayer (K(M) + Sigma_eff), fitted on noiseless copies.  So per galaxy
    E[M_obs | g] = F(m_nl(g), g)
and dE/dg is fixed.  Here the truth: in-window real target galaxies rendered at
g = +/-delta on each axis with the SAME noise fields across arms (CRN),
recentred as imsims does, averaged.  Compared with F's finite difference on the
same galaxy's noiseless moments.  Faint vs bright to see the flux trend.

Summary number per set: the C_M^-1-metric projection slope
    s = sum R_obs . R_flow / sum R_flow . R_flow   (1 = flow right)
which weights each moment by how much shear information it carries.
"""
import sys
from multiprocessing import Pool

sys.path.insert(0, ".")
sys.path.insert(0, "dev")
sys.path.insert(0, "../bfd_cnf_imsims")
import fitsio
import numpy as np

import bias as B
from imsims import sim

D = "../bfd_cnf_imsims/data"
FLOW = "flows/centroid_g2v4n_Ke.eqx"
DELTA = 0.02
GS = [(0.0, 0.0), (DELTA, 0.0), (-DELTA, 0.0), (0.0, DELTA), (0.0, -DELTA)]
N_FAINT, N_BRIGHT = (int(a) for a in (sys.argv[1:3] if len(sys.argv) > 2 else (96, 48)))
N_REAL = int(sys.argv[3]) if len(sys.argv) > 3 else 400


def moments(mc):
    row = np.empty(1, dtype=sim.ROW_DTYPE)[0]
    sim.fill_row(row, mc)
    return row["moments"], bool(row["badcenter"])


def one(args):
    p, seed = args
    import bfd
    _, draw = sim.POPULATIONS["gauss2_fwd"]
    wt = bfd.KBlackmanHarris(weightSigma=sim.WEIGHT_SIGMA)
    imgs = [draw(p, shear=g) for g in GS]
    nl = np.array([moments(sim._measure(im, wt, sim.NOISE_SIGMA, recenter=False))[0] for im in imgs])
    rng = np.random.default_rng(seed)
    obs = []
    for _ in range(N_REAL):
        n = rng.normal(0.0, sim.NOISE_SIGMA, imgs[0].shape)
        r = [moments(sim._measure(im + n, wt, sim.NOISE_SIGMA)) for im in imgs]
        if not any(b for _, b in r):          # drop the whole CRN tuple
            obs.append([m for m, _ in r])
    return nl, np.array(obs)                  # (5 arms, 5), (n_ok, 5 arms, 5)


def main():
    path = f"{D}/{B.CATALOGS['gauss2_v4n']['zero']}.fits"
    t = fitsio.read(path)
    pop = fitsio.read(path, ext="POPULATION")
    win = np.flatnonzero(B.window_mask(t["moments"], (2.2, 3.2), (3000.0, 20000.0)) & ~t["badcenter"])
    q20, q80 = np.quantile(t["moments"][win, 0], [0.2, 0.8])
    rng = np.random.default_rng(0)
    faint = rng.choice(win[t["moments"][win, 0] < q20], N_FAINT, replace=False)
    bright = rng.choice(win[t["moments"][win, 0] > q80], N_BRIGHT, replace=False)
    pick = np.concatenate([faint, bright])
    with Pool(12) as p:
        res = p.map(one, [(pop[i], 1000 + k) for k, i in enumerate(pick)])

    # flow side, after the pool so JAX never forks
    import jax
    import jax.numpy as jnp
    import centroid as C
    from closed_loop import load_flow
    flow = load_flow(FLOW)
    chart, layer = C._chart(flow), C._sigmax_layer(flow)
    sx = jnp.asarray(t["cov_odd"][0], jnp.float32)

    @jax.jit
    def F(m, g):
        return chart.inverse(layer.inverse(chart.transform(m), B.condition(g, sx)))

    CMi = np.linalg.inv(B.load_cov(path))
    rows = []
    for (nl, obs), i in zip(res, pick):
        fl = np.array([np.asarray(F(jnp.asarray(m, jnp.float32), jnp.asarray(g, jnp.float32)), np.float64)
                       for m, g in zip(nl, GS)])
        fd = lambda a: np.stack([(a[..., 1, :] - a[..., 2, :]), (a[..., 3, :] - a[..., 4, :])], -2) / (2 * DELTA)
        R_obs_k = fd(obs)                          # (n_ok, 2, 5) paired per noise field
        R_obs, R_obs_se = R_obs_k.mean(0), R_obs_k.std(0) / np.sqrt(len(obs))
        R_fl, R_nl = fd(fl), fd(nl)
        ip = lambda a, b: np.einsum("ai,ij,aj->", a, CMi, b)
        e = np.hypot(*nl[0, 2:4]) / nl[0, 1]
        rows.append(dict(Mf=nl[0, 0], e=e, n=len(obs),
                         oo=ip(R_obs, R_fl), ff=ip(R_fl, R_fl), nn=ip(R_nl, R_fl),
                         d11=(R_obs[0, 2], R_fl[0, 2], R_nl[0, 2], R_obs_se[0, 2]),
                         d22=(R_obs[1, 3], R_fl[1, 3], R_nl[1, 3], R_obs_se[1, 3]),
                         off0=(obs[:, 0].mean(0) - fl[0]) @ CMi @ (obs[:, 0].mean(0) - fl[0]),
                         R_obs=R_obs, R_fl=R_fl, R_nl=R_nl, R_se=R_obs_se))
    np.save("dev/faint_response.npy", rows, allow_pickle=True)
    # per-noise-field moments for dev/faint_noise_shape.py
    np.savez("dev/faint_response_obs.npz", n_faint=N_FAINT, delta=DELTA, CM=np.linalg.inv(CMi),
             **{f"obs{k}": o for k, (_, o) in enumerate(res)},
             R_fl=np.array([r["R_fl"] for r in rows]), Mf=np.array([r["Mf"] for r in rows]))

    for name, sl in (("faint", slice(0, N_FAINT)), ("bright", slice(N_FAINT, None))):
        rs = rows[sl]
        s_obs = np.array([r["oo"] / r["ff"] for r in rs])
        s_nl = np.array([r["nn"] / r["ff"] for r in rs])
        S = sum(r["oo"] for r in rs) / sum(r["ff"] for r in rs)
        d11 = np.array([r["d11"] for r in rs] + [r["d22"] for r in rs])
        print(f"\n{name}: {len(rs)} galaxies, Mf {np.median([r['Mf'] for r in rs]):.0f} median, "
              f"{np.mean([r['n'] for r in rs]):.0f} ok noise fields each")
        print(f"  pooled slope R_obs on R_flow (C_M^-1 metric)  {S:.4f}   "
              f"per-galaxy mean {s_obs.mean():.4f} +/- {s_obs.std() / np.sqrt(len(rs)):.4f}")
        print(f"  noiseless-fixed-centre on R_flow               {np.mean(s_nl):.4f}")
        print(f"  dM1/dg1 & dM2/dg2  obs/flow {d11[:, 0].sum() / d11[:, 1].sum():.4f}   "
              f"nl/flow {d11[:, 2].sum() / d11[:, 1].sum():.4f}   "
              f"MC se of obs sum {np.sqrt((d11[:, 3] ** 2).sum()) / abs(d11[:, 1].sum()):.4f}")
        print(f"  g=0 mean offset obs-flow, chi2 per galaxy (x N_REAL)  "
              f"{np.mean([r['off0'] for r in rs]) * N_REAL:.2f}")


if __name__ == "__main__":
    main()
