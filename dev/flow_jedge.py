"""Does the flow vanish like J at the J -> 0 edge, as the copies do? (2026-09-29)

The consistent model divides each target draw by J(m) = Mr^2 t / 4, t = 1-|e|^2,
so it needs p_flow/J -- finite for the copies (their weight carries J), but only
as good as the flow's density at small t.  Compares flow samples (g=0, the sn8r
Sigma_X) against the J*L-weighted sn8 copies, per (Mf band, size 2.0-3.4):
the band's mass, the t histogram, and E[1/t].
"""
import sys
import numpy as np, fitsio, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
sys.path.insert(0, "."); sys.path.insert(0, "../bfd_cnf_imsims")
import bias as B, bulk, shear
from imsims.copies import log_weights

D = "../bfd_cnf_imsims/data/"
FLOW, NS = "flows/cv2/centroid_s0.eqx", 8_000_000
BANDS = [(2000, 3000), (3000, 3500), (3500, 4000), (4000, 5000), (5000, 8000), (8000, 20000)]
TB = np.array([0, .05, .1, .2, .35, .6, 1.0001])
SIZE = (2.0, 3.4)


def acc(m, w, out):
    t = 1 - (m[:, 2] ** 2 + m[:, 3] ** 2) / m[:, 1] ** 2
    s = m[:, 1] / m[:, 0]
    for b, (lo, hi) in enumerate(BANDS):
        k = (m[:, 0] >= lo) & (m[:, 0] < hi) & (s > SIZE[0]) & (s < SIZE[1]) & (t > 0)
        out[b, 0] += w[k].sum()
        out[b, 1] += (w[k] / t[k]).sum()
        out[b, 2:] += np.histogram(t[k], TB, weights=w[k])[0]


sx = fitsio.read(D + "targets_bdg2_g0_176k_sn8r.fits", columns=["cov_odd"], rows=[0])["cov_odd"][0]
flow = eqx.tree_deserialise_leaves(FLOW, bulk.build_flow(
    jr.key(0), shear.load(D + "moments_bulgedisc_g2_bdg2n.fits")[0], shear=True, centroid=True))
cond = B.condition(jnp.zeros(2), jnp.asarray(sx, jnp.float32))
samp = eqx.filter_jit(lambda k: flow.sample(k, (500_000,), condition=cond))
fo = np.zeros((len(BANDS), 2 + len(TB) - 1)); nf = 0
for i in range(NS // 500_000):
    m = np.asarray(samp(jr.key(100 + i)), np.float64)
    m = m[np.isfinite(m).all(1)]
    nf += len(m); acc(m, np.ones(len(m)), fo)
fo /= nf

f = fitsio.FITS(D + "copies_bulgedisc_g2_bdg2n_sn8.fits")["COPIES"]
co = np.zeros_like(fo); wsum = 0.0
for i in range(0, f.get_nrows(), 4_000_000):
    c = f.read(columns=["moments", "xy", "da"], rows=np.arange(i, min(i + 4_000_000, f.get_nrows())))
    w = np.exp(log_weights(c, sx)); wsum += w.sum()
    acc(np.asarray(c["moments"], np.float64), w, co)
co /= wsum

print(f"flow {nf} samples; copies normalised by total weight.  size {SIZE}")
print(f"{'Mf band':>11s} {'mass flow/cop':>14s} {'E[1/t] flow':>12s} {'copies':>7s}   "
      "t-hist ratio flow/copies for t in " + " ".join(f"{a:.2f}-" for a in TB[:-1]))
for b, (lo, hi) in enumerate(BANDS):
    hf, hc = fo[b, 2:] / fo[b, 0], co[b, 2:] / co[b, 0]
    print(f"{lo:>5d}-{hi:<5d} {fo[b, 0] / co[b, 0]:>14.3f} {fo[b, 1] / fo[b, 0]:>12.4f} "
          f"{co[b, 1] / co[b, 0]:>7.4f}   " + " ".join(f"{x:5.2f}" for x in hf / hc)
          + f"   copies t-mass {' '.join(f'{x:.3f}' for x in hc[:3])}")
