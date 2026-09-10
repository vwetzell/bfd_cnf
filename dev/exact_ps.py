"""Does dropping eq. (45)-(46)'s quadratic close the size-ceiling gap?

`ghat` takes ONE Newton step: it expands `N_ns log(1 - P_s(g))` to second order
about `g = 0` and solves.  Nothing checks that the expansion still holds at the
`|g| ~ 0.02` the estimator actually solves at -- and for an `Mr/Mf` ceiling it
does not: the ladder in `dev/rs_agree.py` is flat to 0.004 across a flux floor,
a flux ceiling and a size floor, then jumps +0.035 when the size ceiling goes
on.

`P_s(g)` is cheap to evaluate EXACTLY at any `g` -- it is a mean of
`window_prob` over prior draws -- so the fix is to keep the targets' quadratic
(which is all `--save-pqr` stores) and stop truncating the selection term:

    dL/dg1 = SUM_i q_i1 + g1 SUM_i r_i11 - N_ns P_s'(g1) / (1 - P_s(g1)) = 0

solved for `g1` per arm instead of linearised about zero.  Scalar in `g1`, with
`g2 = 0`: the window is spin-0 so `Q_s2 ~ 0` and the arms carry no `g2`.

The push-forward of the prior at each grid `g1` is window-INDEPENDENT, so one
grid serves every window -- the same trick `stencil_moments` uses.

Built-in check: LINEARISE the same solve about zero and it must reproduce
`bias.ghat`'s scalar answer.  Printed as `newton` below.

    PYTHONPATH=. NDRAW=4194304 python dev/exact_ps.py
"""
import os

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

import bias
from bias import load_cov, window_mask, window_prob
import dev.size_bands as sb

D = sb.D
G = sb.G
BOOT = int(os.environ.get("BOOT", 200))
GRID = np.array([float(x) for x in os.environ.get(
    "GRID", "-0.03 -0.025 -0.02 -0.015 -0.01 -0.005 0 "
            "0.005 0.01 0.015 0.02 0.025 0.03").split()])

INF = float("inf")
WINDOWS = [
    ("Mf>=1600", (-INF, INF), (1600.0, 1e9)),
    ("Mf>=2500", (-INF, INF), (2500.0, 1e9)),
    ("Mf 2500-50000", (-INF, INF), (2500.0, 50000.0)),
    ("+ size >2.2", (2.2, INF), (2500.0, 50000.0)),
    ("nominal (+ <3.2)", (2.2, 3.2), (2500.0, 50000.0)),
    ("size 2.2-3.2 only", (2.2, 3.2), (1600.0, 1e9)),
]


def grid_moments(draw, z, batch=16384):
    """Prior draws lensed at every grid `g1` (g2 = 0), `(len(GRID), n, 5)`.

    Window-independent, kept on the host -- see `sb.stencil_moments` for why
    both of those matter.
    """
    import jax

    @jax.jit
    def one(idx):
        ms = jnp.stack([draw(jnp.array([g, 0.0]), idx) for g in GRID])
        return ms, jnp.isfinite(ms).all(-1).all(0)

    out = [one(jnp.asarray(z[i:i + batch])) for i in range(0, len(z), batch)]
    ms = np.concatenate([np.asarray(o[0], np.float64) for o in out], axis=1)
    good = np.concatenate([np.asarray(o[1]) for o in out])
    if not good.all():
        print(f"  dropped {int((~good).sum())}/{len(good)} non-finite draws")
    return ms[:, good]


def ps_curve(ms, cov, size, flux, batch=16384):
    """`P_s` at every grid `g1`, as a cubic spline in `g1`."""
    import jax

    @jax.jit
    def one(mg):
        return jnp.stack([window_prob(mg[k], cov, size, flux)
                          for k in range(len(GRID))]).mean(-1)

    n = ms.shape[1]
    acc = np.zeros(len(GRID))
    for i in range(0, n, batch):
        b = ms[:, i:i + batch]
        acc += np.asarray(one(jnp.asarray(b)), np.float64) * b.shape[1]
    return CubicSpline(GRID, acc / n)


def solve(sq, sr, n_ns, ps, exact):
    """`g1` from `SUM q + g SUM r - n_ns d/dg log(1-P_s) = 0`.

    `exact=False` linearises `P_s` about zero, which is `bias.ghat`'s scalar
    case and is what the printed `newton` column checks against.
    """
    if not exact:
        p0, q0, r0 = ps(0.0), ps(0.0, 1), ps(0.0, 2)
        num = sq - n_ns * q0 / (1 - p0)
        den = -sr + n_ns * (q0 ** 2 / (1 - p0) ** 2 + r0 / (1 - p0))
        return num / den
    f = lambda g: sq + g * sr - n_ns * ps(g, 1) / (1 - ps(g))
    lo, hi = GRID[0] * 0.9, GRID[-1] * 0.9
    if f(lo) * f(hi) > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-9)


def m1(qp, rp, qm, rm, sp, sm, ps, exact):
    a = solve(qp[sp, 0].sum(), rp[sp, 0, 0].sum(), int((~sp).sum()), ps, exact)
    b = solve(qm[sm, 0].sum(), rm[sm, 0, 0].sum(), int((~sm).sum()), ps, exact)
    return (a - b) / (2 * G) - 1


if __name__ == "__main__":
    d = np.load(sb.PQR)
    ok = np.ones(len(d["plus_q"]), dtype=bool)
    for k in ("plus", "minus"):
        ok &= np.isfinite(d[f"{k}_q"]).all(1)
        ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    qp, rp = d["plus_q"][ok], d["plus_r"][ok]
    qm, rm = d["minus_q"][ok], d["minus_r"][ok]
    obs_p, obs_m = d["obs_plus"][ok], d["obs_minus"][ok]
    cov = load_cov(f"{D}/{bias.CATALOGS[sb.POP]['zero']}.fits")

    sb.TERMS = "flow"
    draw, z = sb.prior_draw()
    print(f"flow: {len(z)} draws, grid {GRID[0]:+.3f} .. {GRID[-1]:+.3f} "
          f"({len(GRID)} points)", flush=True)
    ms = grid_moments(draw, z)
    del draw, z

    rng = np.random.default_rng(0)
    n = len(qp)
    idx = [rng.integers(0, n, n) for _ in range(BOOT)]
    print(f"\n{'window':>20s}{'kept':>7s}{'P_s(0)':>9s}"
          f"{'P_s(.02)':>10s}{'quadratic':>11s}{'rel err':>9s}"
          f"{'R_s spline':>12s}{'R_s fd.02':>11s}"
          f"{'ghat 2x2':>10s}{'newton 1d':>11s}{'EXACT':>19s}")
    for name, size, flux in WINDOWS:
        sp, sm = window_mask(obs_p, size, flux), window_mask(obs_m, size, flux)
        ps = ps_curve(ms, cov, size, flux)
        p0, q0, r0 = ps(0.0), ps(0.0, 1), ps(0.0, 2)
        quad = p0 + q0 * G + 0.5 * r0 * G ** 2      # what eq. (45)-(46) assumes
        # `ghat`'s own 2x2 answer, from the same spline's derivatives
        full = (p0, np.array([q0, 0.0]),
                np.array([[r0, 0.0], [0.0, r0]]))
        ns = ((int((~sp).sum()), *full), (int((~sm).sum()), *full))
        g2x2 = bias.bias(qp, rp, qm, rm, G, sel=(sp, sm), ns=ns)[0]
        mn = m1(qp, rp, qm, rm, sp, sm, ps, False)
        me = m1(qp, rp, qm, rm, sp, sm, ps, True)
        de = np.std([m1(qp[i], rp[i], qm[i], rm[i], sp[i], sm[i], ps, True)
                     for i in idx], ddof=1)
        rfd = (ps(G) + ps(-G) - 2 * p0) / G ** 2
        print(f"{name:>20s}{sp.mean():7.1%}{p0:9.4f}{ps(G):10.4f}"
              f"{quad:11.4f}{(quad - ps(G)) / ps(G):9.2%}"
              f"{r0:12.4f}{rfd:11.4f}"
              f"{g2x2:+10.4f}{mn:+11.4f}"
              f"{me:+12.4f} +/- {de:.4f}", flush=True)
    print("\n`quadratic` is P_s(0)+Q_s g+R_s g^2/2 against the true P_s(0.02) "
          "next to it:\nthe size ceiling is where they part.")
