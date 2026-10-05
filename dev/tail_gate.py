"""Tail gate: does a trained flow put mass where the training data has none, at high |e|?
(2026-09-30)

cv2 seed 2's bulk put 2e-5 of its mass at |e| = 1.00000 (training max 0.753).
No training metric saw it (no data there), but the paper model's 1/J weight,
J = (Mr^2 - M1^2 - M2^2)/4, diverges there and that sliver halved R_s.  The
E_MAX chart ceiling now caps 1/J; this gate catches a flow piling mass against
the ceiling anyway.  Samples the flow at g = 0 and compares with the training set:
  - fraction of samples with |e| > training max + MARGIN   (FAIL if > FRAC_MAX)
  - mean Mr^2/(4J) = E[1/(1-|e|^2)], flow / data          (FAIL if off by > WT_TOL)

  python dev/tail_gate.py --kind bulk|shear|centroid FLOW.eqx [--data M.fits] [--sx-from T.fits]
Exit status 1 on FAIL.  GPU; run alone.
"""
import argparse
import sys
sys.path.insert(0, ".")

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import bias as B
import bulk
import shear

MARGIN, FRAC_MAX, WT_TOL = 0.05, 1e-6, 0.02

p = argparse.ArgumentParser()
p.add_argument("flow")
p.add_argument("--kind", choices=["bulk", "shear", "centroid"], required=True)
p.add_argument("--data", default="../bfd_cnf_imsims/data/moments_bulgedisc_g2_bdg2n.fits")
p.add_argument("--sx-from", default=None, help="catalog whose row-0 cov_odd conditions a centroid flow")
p.add_argument("--n", type=int, default=1 << 21)
a = p.parse_args()

m = np.asarray(shear.load(a.data)[0], np.float64)
fl = bulk.load_flow(a.flow, m[:20000], key=jr.key(0),
                    shear=a.kind != "bulk", centroid=a.kind == "centroid")
if a.kind == "centroid":
    sx = jnp.asarray(fitsio.read(a.sx_from, rows=[0])["cov_odd"][0], jnp.float32)
    c = B.condition(jnp.zeros(2), sx)
else:
    c = jnp.zeros(2) if a.kind == "shear" else None
f = eqx.filter_jit(lambda z: jax.vmap(lambda zi: fl.bijection.transform(zi, c))(z))
z = fl.base_dist.sample(jr.key(1), (a.n,))
x = np.concatenate([np.asarray(f(z[i:i + (1 << 18)]), np.float64) for i in range(0, a.n, 1 << 18)])

e = lambda v: np.hypot(v[:, 2], v[:, 3]) / v[:, 1]
wt = lambda v: np.mean(1.0 / (1.0 - e(v) ** 2))
ed, ex = e(m), e(x)
ok = np.isfinite(ex)
frac = np.mean(ex[ok] > ed.max() + MARGIN)
ratio = wt(x[ok]) / wt(m)
fail = frac > FRAC_MAX or abs(ratio - 1) > WT_TOL or (~ok).any()
print(f"tail gate {a.flow}: max|e| flow {ex[ok].max():.4f} data {ed.max():.4f}  "
      f"frac > data max + {MARGIN} {frac:.1e} (max {FRAC_MAX:.0e})  "
      f"E[1/(1-e^2)] flow/data {ratio:.4f} (tol {WT_TOL})  nonfinite {(~ok).sum()}  "
      f"{'FAIL' if fail else 'PASS'}")
sys.exit(1 if fail else 0)
