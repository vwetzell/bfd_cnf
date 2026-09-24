"""Acceptance check for the K(M) effective-kernel centroid layer
(`SigmaXBlockLayer._A` / `_sigma_eff`, `centroid.fit_K`).

a. fit_K on copies 0:4M; held-out 60M:61M R^2 of dX/dg, d2X/dg2 and the
   eq.-36 weight term l_a = -X^T C^-1 dX/dg_a (binned-table reference
   0.957 / 0.903 / 0.933, dev/fit_centroid_K.py)
b. g = 0: Sigma_eff == C exactly
c. rotation equivariance of A X and Sigma_eff
d. layer round trip + log-det consistency at g != 0
e. flow.log_prob and its g-gradient finite
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "dev")
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import bias as B
import bulk
import centroid as C
from fit_centroid_K import r2

COPIES = "../bfd_cnf_imsims/data/copies_gauss2_fwd_g2v4n.fits"
TARGETS = "../bfd_cnf_imsims/data/targets_g2v4_g1p02_22k.fits"
F = fitsio.FITS(COPIES)["COPIES"]
cols = ["moments", "xy", "dxy_dg", "d2xy_dg2"]
tr = F.read(columns=cols, rows=np.arange(0, 4_000_000))
te = F.read(columns=cols, rows=np.arange(60_000_000, 61_000_000))
sx = np.asarray(fitsio.read(TARGETS, rows=[0])["cov_odd"][0], np.float64)
CX = np.array([[sx[0], sx[1]], [sx[1], sx[2]]])

flow = bulk.build_flow(jr.key(0), np.asarray(tr["moments"][:20000], np.float64),
                       shear=True, centroid=True)
flow = C.fit_K(flow, tr)
layer, chart = C._sigmax_layer(flow), C._chart(flow)
ok = True


def check(name, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}: {msg}")


# a. held-out response fit
z = jax.vmap(chart.transform)(jnp.asarray(te["moments"], jnp.float32))
X = jnp.asarray(te["xy"], jnp.float32)
iu = jnp.array([0, 0, 1]), jnp.array([0, 1, 1])


def pred(zi, Xi):
    f = lambda g: layer._A(zi, g) @ Xi
    return jax.jacfwd(f)(jnp.zeros(2)).T, jax.hessian(f)(jnp.zeros(2))[:, iu[0], iu[1]].T


p1, p2 = [np.asarray(v, np.float64) for v in jax.jit(jax.vmap(pred))(z, X)]
d1, d2 = np.asarray(te["dxy_dg"], np.float64), np.asarray(te["d2xy_dg2"], np.float64)
sX = np.asarray(X, np.float64) @ np.linalg.inv(CX)
la = lambda D: -np.einsum("nk,nak->na", sX, D)
r_1, r_2, r_l = r2(d1, p1), r2(d2, p2), r2(la(d1), la(p1))
check("a", r_1 >= 0.93 and r_2 >= 0.85 and r_l >= 0.90,
      f"R^2 dX {r_1:.3f}  d2X {r_2:.3f}  l_a {r_l:.3f}  (binned 0.957/0.903/0.933)")

# b. g = 0 is the raw kernel
cond0 = jnp.concatenate([jnp.zeros(2), jnp.asarray(sx, jnp.float32)])
se0 = jax.vmap(lambda zi: layer._sigma_eff(zi, cond0))(z[:512])
err_b = float(jnp.max(jnp.abs(se0 - jnp.asarray(CX, jnp.float32))))
check("b", err_b < 1e-6 * float(np.abs(CX).max()), f"max |Sigma_eff - C| at g=0 = {err_b:.2e}")

# c. rotation equivariance
rng = np.random.default_rng(0)
R = lambda t: jnp.array([[jnp.cos(t), -jnp.sin(t)], [jnp.sin(t), jnp.cos(t)]])
worst = 0.0
for th in (0.3, 1.1, 2.7):
    for i in range(64):
        zi, Xi = z[i], X[i]
        g = jnp.asarray(rng.normal(0, 0.03, 2), jnp.float32)
        zr = zi.at[3:5].set(R(2 * th) @ zi[3:5])
        lhs = layer._A(zr, R(2 * th) @ g) @ (R(th) @ Xi)
        rhs = R(th) @ (layer._A(zi, g) @ Xi)
        c_ = jnp.concatenate([g, jnp.asarray(sx, jnp.float32)])
        CXr = R(th) @ jnp.asarray(CX, jnp.float32) @ R(th).T
        cr = jnp.concatenate([R(2 * th) @ g, jnp.array([CXr[0, 0], CXr[0, 1], CXr[1, 1]])])
        se_l = layer._sigma_eff(zr, cr)
        se_r = R(th) @ layer._sigma_eff(zi, c_) @ R(th).T
        worst = max(worst,
                    float(jnp.linalg.norm(lhs - rhs) / jnp.linalg.norm(rhs)),
                    float(jnp.linalg.norm(se_l - se_r) / jnp.linalg.norm(se_r)))
check("c", worst < 1e-5, f"worst rel err {worst:.2e}")

# d. round trip at g != 0
cond = jnp.concatenate([jnp.array([0.03, -0.02]), jnp.asarray(sx, jnp.float32)])
zb = z[:512]
y, ldf = jax.vmap(lambda zi: layer.transform_and_log_det(zi, cond))(zb)
xb, ldi = jax.vmap(lambda yi: layer.inverse_and_log_det(yi, cond))(y)
e_x, e_ld = float(jnp.max(jnp.abs(xb - zb))), float(jnp.max(jnp.abs(ldf + ldi)))
check("d", e_x < 1e-4 and e_ld < 1e-4, f"max |x' - x| {e_x:.2e}  max |ld_f + ld_i| {e_ld:.2e}")

# e. log_prob and g-gradient finite
m = jnp.asarray(te["moments"][:2048], jnp.float32)
sxj = jnp.asarray(sx, jnp.float32)
lp = lambda g: jax.vmap(lambda mi: flow.log_prob(mi, condition=B.condition(g, sxj)))(m).sum()
g0 = jnp.array([0.02, 0.0])
v, gr = jax.jit(jax.value_and_grad(lp))(g0)
check("e", bool(jnp.isfinite(v) & jnp.all(jnp.isfinite(gr))),
      f"sum log_prob {float(v):.1f}  d/dg {np.asarray(gr)}")

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
