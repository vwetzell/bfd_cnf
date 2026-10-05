"""Does a tapered (soft) window cut the edge noise in windowed m1?  (2026-10-03)  Offline.

Hard window: weight 1 inside, 0 outside, on each arm's own measured moments.  Galaxies
whose size/flux straddles an edge differently in the +g / -g arms (or in sn8r / varobs)
enter one sum unpaired, with their whole q -- ~96% of the varobs - sn8r gap variance.
Tapered: each edge becomes a smoothstep ramp of half-width h centred on the hard edge,
so a small move in measured moments changes the weight by a small amount.  Estimator per
arm: ghat = [sum w (-R)]^-1 sum w Q.  Uncorrected only (target side; the selection terms
for a weighted window need a bias.py change) -- the point is the ERROR, not the mean.
Per-target Q, R from the full cv6 s2 runs; the pre-filter pad (x0.8 / x1.2) bounds h.
Usage: python dev/taper_check.py
"""
import os, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D, G, S, F = "../bfd_cnf_imsims/data/", 0.02, (2.2, 3.2), (3000, 20000)
RUNS = {"A": ("pqr/cv6_s2_sn8r_nopool.npz", "sn8r"), "B": ("pqr/cv6_s2.npz", "varobs_sn8")}
# (size half-width, flux half-width as a fraction of the edge)
TAPERS = [(0, 0), (0.05, 0), (0.1, 0), (0.2, 0), (0, 0.1), (0, 0.2), (0.1, 0.1), (0.2, 0.2)]


def ramp(x, lo, hi, h):   # 1 inside (lo, hi), smoothstep over [edge - h, edge + h]
    if h == 0:
        return ((x > lo) & (x < hi)).astype(float)
    s = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
    return s((x - lo + h) / (2 * h)) * s((hi + h - x) / (2 * h))


def weight(m, hs, hf):
    f, r = m[:, 0], m[:, 1] / np.maximum(m[:, 0], 1)
    return ramp(r, *S, hs) * ramp(f / F[0], 1, F[1] / F[0], hf)   # flux ramp in units of the low edge


def load(pqr, tag):
    d = np.load(pqr)
    cat = {a: fitsio.read(D + f"targets_bdg2_{a}_176k_{tag}.fits", columns=["moments", "badcenter"])
           for a in ("g1p02", "g1m02", "g0")}
    n = len(cat["g0"])
    keep = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
    pos = {tuple(r): i for i, r in enumerate(np.asarray(cat["g1p02"]["moments"], np.float64))}
    row = np.array([pos.get(tuple(r), -1) for r in d["obs_plus"]])
    assert (row >= 0).all()
    x = {k: np.zeros((n,) + d[k].shape[1:]) for k in ("plus_q", "plus_r", "minus_q", "minus_r")}
    for k in x:
        x[k][row] = d[k]
    integ = np.zeros(n, bool); integ[row] = True
    x["m"] = {"plus": np.asarray(cat["g1p02"]["moments"], np.float64),
              "minus": np.asarray(cat["g1m02"]["moments"], np.float64)}
    x["ok"], x["integ"] = keep & integ, integ
    return x


def m1(x, hs, hf, c):
    """Uncorrected m1 for bootstrap count matrix c (n_rep, n) -- or None for the data."""
    g = {}
    for arm in ("plus", "minus"):
        w = weight(x["m"][arm], hs, hf)
        assert ((w > 1e-6) & ~x["integ"]).sum() < 50, "taper reaches past the pre-filter pad"
        w = w * x["ok"]
        q, r = x[arm + "_q"] * w[:, None], x[arm + "_r"] * w[:, None, None]
        if c is None:
            g[arm] = np.linalg.solve(-r.sum(0), q.sum(0))[0]
        else:
            Q, Rm = c @ q, np.einsum("bn,nij->bij", c, r)
            g[arm] = np.linalg.solve(-Rm, Q[..., None])[:, 0, 0]
    return (g["plus"] - g["minus"]) / (2 * G) - 1, (w > 0).sum(), w.sum()


R = {k: load(*v) for k, v in RUNS.items()}
n = len(R["A"]["ok"])
use = np.flatnonzero(R["A"]["integ"] | R["B"]["integ"])   # rows outside contribute nothing
for x in R.values():
    for k in ("plus_q", "plus_r", "minus_q", "minus_r", "ok", "integ"):
        x[k] = x[k][use]
    x["m"] = {a: v[use] for a, v in x["m"].items()}
c = np.random.default_rng(0).poisson(1.0, (300, len(use))).astype(float)   # Poisson row bootstrap, paired
print(f"{'size h':>6s} {'flux h':>6s} {'sum w':>8s} {'m1 A (sn8r)':>20s} {'m1 B (varobs)':>20s} {'gap B-A':>20s}"
      f"   err x sqrt(sum w / base)")
base = None
for hs, hf in TAPERS:
    (va, _, sw), (vb, _, _) = m1(R["A"], hs, hf, None), m1(R["B"], hs, hf, None)
    ba, bb = m1(R["A"], hs, hf, c)[0], m1(R["B"], hs, hf, c)[0]
    e = np.array([ba.std(), bb.std(), (bb - ba).std()])
    base = base or (sw, e)
    k = e * np.sqrt(sw / base[0]) / base[1]   # error relative to hard, at matched effective count
    print(f"{hs:6.2f} {hf:6.2f} {sw:8.0f} {va:+.5f}+/-{e[0]:.5f} {vb:+.5f}+/-{e[1]:.5f} "
          f"{vb - va:+.5f}+/-{e[2]:.5f}   {k[0]:.2f} {k[1]:.2f} {k[2]:.2f}")
print("(uncorrected: means move with the taper because the selection bias does; compare ERRORS.\n"
      " last columns: each error relative to the hard window's, scaled to equal sum w)")
