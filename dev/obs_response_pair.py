"""Observed in-window shear response, flow vs the real DES-matched target arms.  (2026-10-06)  GPU.

The first-order formula (memory corrected-m-first-order-bias-formula) makes corrected m the
Fisher-weighted relative error of the flow's in-window mean response.  The +/-0.02 target arms
measure the real response directly, with no copies anywhere.  Each flow is sampled at g = +/-0.02
with common base draws, each synthetic galaxy given a real target's own C_M and Sigma_X (rows
cycled), plus common noise; the observed d<e1>/dg1 and d<M1>/dg1 under the run's tapered window
is compared to the real arms'.  Both flows share draws and noise, so flow - flow is paired.
Errors: bootstrap over 200 random row groups (real and synthetic separately).
Usage: TRAIN_POP=bulgedisc_g2n_176k_des python dev/obs_response_pair.py FLOW [FLOW ...]
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "dev")
import equinox as eqx, fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np, bfd
import bias as B
from closed_loop import D, load_flow

B.WINDOW_TAPER = (0.05, 150.0)
G, NG, NREP, MULT, BATCH = 0.02, 200, 1000, 2, 65536
WINS = {"Mf 2200-20k s 2.2-3.5": ((2.2, 3.5), (2200, 20000)), "Mf 3000-20k s 2.2-3.2": ((2.2, 3.2), (3000, 20000)),
        "  2200-3000": ((2.2, 3.5), (2200, 3000)), "  3000-4500": ((2.2, 3.5), (3000, 4500)),
        "  4500-8000": ((2.2, 3.5), (4500, 8000)), "  8000-20k": ((2.2, 3.5), (8000, 20000)),
        "  s 3.2-3.5 (Mf 2200-20k)": ((3.2, 3.5), (2200, 20000)), "Mf 2200-90k s 2.2-3.5": ((2.2, 3.5), (2200, 90000))}


def sums(mp, mm, grp):
    """per window: group sums (NG, 6) of w+, w+ e1+, w+ M1+ and the same for the minus arm."""
    out = {}
    for k, (s, f) in WINS.items():
        cols = []
        for m in (mp, mm):
            w = np.asarray(B.window_mask(m, s, f), float)
            cols += [w, w * m[:, 2] / m[:, 1], w * m[:, 2]]
        out[k] = np.stack([np.bincount(grp, c, NG) for c in cols], -1)
    return out


def resp(gs, rep):
    s = rep @ gs                                                    # (R, 6)
    return np.stack([(s[:, 1] / s[:, 0] - s[:, 4] / s[:, 3]) / (2 * G),
                     (s[:, 2] / s[:, 0] - s[:, 5] / s[:, 3]) / (2 * G)], -1)   # (R, 2): e1, M1


if __name__ == "__main__":
    cat = {a: fitsio.read(f"{D}/targets_bdg2_{a}_176k_des.fits", columns=["moments", "cov", "cov_odd", "badcenter"])
           for a in ("g1p02", "g1m02", "g0")}
    ok = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
    rp, rm = (np.asarray(cat[a]["moments"][ok], float) for a in ("g1p02", "g1m02"))
    rng = np.random.default_rng(0)
    rep = np.vstack([np.ones(NG), rng.multinomial(NG, np.full(NG, 1 / NG), NREP)])
    real = {k: resp(v, rep) for k, v in sums(rp, rm, rng.integers(0, NG, len(rp))).items()}
    # synthetic: real rows' noise properties (zero arm, as bias.py's sigma_x), cycled
    n = MULT * len(rp)
    row = np.flatnonzero(ok)[np.arange(n) % ok.sum()]
    L = np.linalg.cholesky(np.asarray(bfd.MomentCovariance.bulkUnpack(np.asarray(cat["g0"]["cov"][row], float)), float))
    noise = np.einsum("nij,nj->ni", L, rng.standard_normal((n, 5)))
    sx = jnp.asarray(np.asarray(cat["g0"]["cov_odd"][row], np.float32))
    grp = rng.integers(0, NG, n)
    flows = {}
    for path in sys.argv[1:]:
        flow = load_flow(path)

        @eqx.filter_jit
        def draw(z, g, s):
            return jax.vmap(lambda zi, si: flow.bijection.transform(zi, B.condition(g, si)))(z, s)
        z = flow.base_dist.sample(jr.key(7), (n,))
        arms = [np.concatenate([np.asarray(draw(z[i:i + BATCH], jnp.array([g1, 0.0]), sx[i:i + BATCH]), np.float64)
                                for i in range(0, n, BATCH)]) + noise for g1 in (G, -G)]
        bad = ~(np.isfinite(arms[0]).all(1) & np.isfinite(arms[1]).all(1))
        for a in arms:
            a[bad] = 1.0       # Mr/Mf = 1: outside every window
        print(f"{path}: {n} draws, {bad.sum()} non-finite", flush=True)
        flows[path] = {k: resp(v, rep) for k, v in sums(*arms, grp).items()}
    for j, stat in enumerate(("d<e1>/dg1", "d<M1>/dg1")):
        print(f"\n{stat}  (flow/real ratio; paired flow difference = last - first)")
        for k in WINS:
            r = real[k][:, j]
            line = f"{k:28s} real {r[0]:+.4e} +/- {r[1:].std():.1e}"
            ratios = []
            for path, f in flows.items():
                q = f[k][:, j]
                ratios.append(q / r[0])
                line += f" | {path.split('/')[-1].replace('.eqx', '')} {q[0] / r[0]:.4f} +/- {np.hypot(q[1:].std() / r[0], q[0] * r[1:].std() / r[0] ** 2):.4f}"
            if len(ratios) > 1:
                d = ratios[-1] - ratios[0]
                line += f" | diff {d[0]:+.4f} +/- {d[1:].std():.4f}"
            print(line)
