"""Solve for `ghat` with the selection term evaluated AT the operating shear.

eq. (45)-(46) stands in for the whole selection term with two derivatives at
g = 0, `Q_s` and `R_s`.  Both are the badly behaved objects here:

  * `R_s` from autodiff has no usable mean for a size window (per-draw
    `d2F/dg2 ~ Mf^2` against a boundary occupancy falling as 1/Mf) -- measured
    656% carried by one template at `Mr/Mf <= 3.0`, half-split spread 354%;
  * a finite central difference fixes THAT, but `R_s` is then genuinely
    g-dependent, because `P_s(g)` is not quadratic once the boundary sweeps a
    dense part of the population.  `2(P_s(g)+P_s(-g)-2P_s(0))/g^2` is flat to
    0.2% over 16x in g for a flux-only window and moves 15-18% between
    g = 0.02 and 0.04 at a 3.2/3.0 ceiling.

Neither problem touches `P_s` itself: a mean of `window_prob`, a probability in
[0, 1].  Bounded terms, finite variance, 1/sqrt(N).

So never differentiate `F` per draw, and never Taylor `P_s` from zero.  Split
`P_s(g) = P_even(u) + P_odd(g)`, `u = |g|^2`, and use the EXACT score

    d/dg log(1 - P_s) = -[2 g P_even'(u) + Q_s] / (1 - P_s)

giving the stationary condition of `Q.g + g.R.g/2 + N_ns log(1 - P_s(g))`

    (c(u) I - R) g = Q - N_ns Q_s / (1 - P_s),   c = 2 N_ns P_even'(u) / (1 - P_s)

a fixed point in `u`.  `c(0) = N_ns R_s / (1 - P_s)` exactly, so eq. (45)-(46)
IS this at `u -> 0` -- the difference is that `c` is evaluated where the
solution sits.  The `Q_s^2` term the paper carries appears here on its own, out
of the exact `log`, rather than as a separate expansion term.

`P_even'` is a LOCAL central difference in `u` about the operating point, not a
global fit: a polynomial over |g| in [0, 0.05] has to absorb the large-|g|
structure and returns a slope 17% wrong at the one place it is used (checked
against the flux-only window, where the answer is known).

CONTROL: where `R_eff` is flat in g -- the flux-only window -- this must
reproduce eq. (45)-(46).  `2 P_even'` must come back at -0.2306 there.  The
check runs first and refuses to report the other windows if it fails.
"""
import os, sys
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
import fitsio

import bias, bulk, shear
from bias import condition, window_prob, window_mask, load_cov, selection_terms

D = "../bfd_cnf_imsims/data"
POP = os.environ.get("POP", "bulgedisc_deep_v2")
FLOW = os.environ.get("FLOW", "flows/centroid_bulgedisc_v2.eqx")
PQR = os.environ.get("PQR", "dev/pqr_200k.npz")
FLUX_LO = float(os.environ.get("FLUX_LO", 1600))
HIS = [float(x) for x in os.environ.get("HIS", "9e9 3.6 3.45 3.3 3.2 3.0").split()]
NDRAW = int(os.environ.get("NDRAW", 1048576))
BOOT = int(os.environ.get("BOOT", 200))
G = float(os.environ.get("G", 0.02))
SEED = int(os.environ.get("SEED", 31))
BATCH = 16384
# u nodes as multiples of the operating point G^2.  The derivative stencil is
# +/- 0.5 G^2 about u0 ~ G^2, so 0.5 and 1.5 must be nodes; 0 anchors R_eff.
NODES = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
# Control value: 2 P_even'(u) at the flux-only window, from the independent
# P_s(+/-g) ladder (dev/ps_ladder.py), flat to 0.2% over 16x in g.
CONTROL = -0.2306
CONTROL_TOL = float(os.environ.get("CONTROL_TOL", 0.05))


def curve(fvals, z, size, keep_pool=None):
    """`(P_even(u), Q_s1(u), u_nodes)` on the node grid, one fixed draw set.

    Both signs of g1 at every node: the even part carries the isotropic
    response the metric needs, the odd part IS `Q_s1` and goes to the score.
    A draw non-finite at ANY node is dropped everywhere, so all the differences
    stay paired -- one dropped row moves `P_s` by 1/N, which is the size of the
    whole signal at the smallest node.
    """
    us = NODES * G ** 2
    gs = np.sqrt(us)
    cols = {}
    keep = np.ones(len(z), bool) if keep_pool is None else keep_pool.copy()
    for gg in gs:
        for sgn in (+1.0, -1.0):
            v = np.concatenate([np.asarray(fvals(jnp.array([sgn * gg, 0.0]),
                                                 z[i:i + BATCH], size))
                                for i in range(0, len(z), BATCH)])
            cols[(gg, sgn)] = v
            keep &= np.isfinite(v)
    even = np.array([0.5 * (cols[(gg, +1.0)][keep].mean()
                            + cols[(gg, -1.0)][keep].mean()) for gg in gs])
    odd = np.array([0.5 * (cols[(gg, +1.0)][keep].mean()
                           - cols[(gg, -1.0)][keep].mean()) for gg in gs])
    # Q_s1(u) = odd / |g|; undefined at u = 0, carried over from the next node.
    qs = np.where(gs > 0, odd / np.where(gs > 0, gs, 1.0), 0.0)
    qs[0] = qs[1]
    return even, qs, us, int((~keep).sum())


def local(even, qs, us, u0, delta=None):
    """`(P_s(u0), P_even'(u0), Q_s1(u0))` -- a LOCAL central difference in u.

    Step defaults to half the operating point, i.e. exactly the 0.5/1.5 nodes,
    so no extrapolation to u = 0 and no global fit is involved.
    """
    d = delta if delta is not None else 0.5 * G ** 2
    p = lambda u: float(np.interp(u, us, even))
    dp = (p(u0 + d) - p(u0 - d)) / (2 * d)
    return p(u0), dp, float(np.interp(u0, us, qs))


def solve(Q, R, n_ns, even, qs, us, tol=1e-12, it=64):
    """Stationary point of `Q.g + g.R.g/2 + n_ns log(1 - P_s(g))`."""
    g = np.linalg.solve(-R, Q)
    for _ in range(it):
        ps, dps, qs1 = local(even, qs, us, float(g @ g))
        c = 2 * n_ns * dps / (1 - ps)
        rhs = Q - n_ns * np.array([qs1, 0.0]) / (1 - ps)
        gn = np.linalg.solve(c * np.eye(2) - R, rhs)
        if np.max(np.abs(gn - g)) < tol:
            return gn
        g = gn
    return g


if __name__ == "__main__":
    train = f"{D}/{bias.TRAIN_DATA[POP]}"
    flow = bulk.build_flow(jr.key(0), shear.load(train)[0], shear=True,
                           centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    sx = jnp.asarray(fitsio.read(f"{D}/{bias.CATALOGS[POP]['zero']}.fits",
                                 rows=[0])["cov_odd"][0], dtype=jnp.float64)
    cov = jnp.asarray(load_cov(f"{D}/{bias.CATALOGS[POP]['zero']}.fits"),
                      dtype=jnp.float64)
    z = flow.base_dist.sample(jr.key(SEED), (NDRAW,)).astype(jnp.float64)

    @eqx.filter_jit
    def fvals(g, zz, size):
        m = jax.vmap(lambda z1: flow.bijection.transform(
            z1, condition(jnp.asarray(g, jnp.float64), sx)))(zz)
        return window_prob(m, cov, size, (FLUX_LO, 1e9))

    draw = lambda g, zz: jax.vmap(
        lambda z1: flow.bijection.transform(z1, condition(g, sx)))(zz)

    d = np.load(PQR)
    ok = np.ones(len(d["plus_q"]), dtype=bool)
    for k in ("plus", "minus"):
        ok &= np.isfinite(d[f"{k}_q"]).all(1)
        ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    arm = {k: (d[f"{k}_q"][ok], d[f"{k}_r"][ok]) for k in ("plus", "minus")}
    obs = {k: d[f"obs_{k}"][ok] for k in ("plus", "minus")}
    n = int(ok.sum())

    print(f"{n} targets, flux >= {FLUX_LO:g}, {NDRAW} prior draws, "
          f"u nodes {NODES} x g^2")

    # CONTROL FIRST.  If the flux-only window does not reproduce the known
    # -0.2306, nothing below it is worth printing.
    ev, qs, us, drop = curve(fvals, z, (-np.inf, np.inf))
    _, dps0, _ = local(ev, qs, us, G ** 2)
    got = 2 * dps0
    print(f"\ncontrol (flux-only window): 2 P_even'(g^2) = {got:+.4f}  "
          f"against {CONTROL:+.4f} from the P_s(+/-g) ladder  "
          f"[{abs(got / CONTROL - 1):.1%}]  ({drop} draws dropped)")
    if abs(got / CONTROL - 1) > CONTROL_TOL:
        sys.exit(f"CONTROL FAILED (> {CONTROL_TOL:.0%}); not reporting windows")

    print(f"\n{'Mr/Mf hi':>10s}{'kept':>8s}{'P_s(0)':>9s}"
          f"{'m1 (profile P_s)':>24s}{'m1 (eq.45-46 fd)':>19s}"
          f"{'2Pe(g^2)':>10s}{'R_s11(fd)':>11s}")
    for hi in HIS:
        size = (-np.inf, hi)
        ev, qs, us, _ = curve(fvals, z, size)
        sel = {k: window_mask(obs[k], size, (FLUX_LO, 1e9)) for k in arm}
        rng = np.random.default_rng(0)

        def m1_of(idx):
            gh = {}
            for k, (q, r) in arm.items():
                s = sel[k][idx]
                gh[k] = solve(q[idx][s].sum(0), r[idx][s].sum(0),
                              int((~s).sum()), ev, qs, us)
            return (gh["plus"][0] - gh["minus"][0]) / (2 * G) - 1

        base = m1_of(np.arange(n))
        sd = np.std([m1_of(rng.integers(0, n, n)) for _ in range(BOOT)], ddof=1)
        pse, qse, rse, _ = selection_terms(draw, z, cov, size,
                                           (FLUX_LO, 1e9), fd=0.02)
        ns = tuple((int((~sel[k]).sum()), pse, qse, rse)
                   for k in ("plus", "minus"))
        old = bias.bias(*arm["plus"], *arm["minus"], G,
                        sel=(sel["plus"], sel["minus"]), ns=ns)[0]
        _, dps, _ = local(ev, qs, us, G ** 2)
        print(f"{hi:10.2f}{sel['plus'].mean():8.1%}{ev[0]:9.4f}"
              f"{base:+15.5f} +/- {sd:.5f}{old:+19.5f}"
              f"{2 * dps:+10.3f}{rse[0, 0]:+11.3f}", flush=True)
