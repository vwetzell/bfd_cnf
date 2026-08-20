"""How steep are the TRUE response-coefficient functions?

The log-det leak is `-1/2 g^2 tr((dv/dz)^2)`, quadratic in the first-order
response's Jacobian, so it is set by how fast the coefficient network varies
with z -- not by any coefficient's order in g (`/tmp/ablate.py`: zeroing A,B
removes 1.609 of a 1.581 even log-det; every genuinely second-order block
removes ~0.002).  Bounding that steepness is the permanent fix, and the bound
has to come from the data rather than be chosen.

`response` is LINEAR in the coefficients, and at first order in g only five
matter, so bfd's exact per-template dm/dg inverts for them directly:

  spin-0:  dz_i/dg = a_i * dp1/dg,  p1 = Re(ebar.g),  e = e_scale*(z3 + i z4)
           => dp1/dg = e_scale*(z3, z4),  so a_i = (v_i . (z3,z4)) / (e_scale |e|^2)

  spin-2:  de/dg1 = (A + B e^2)/e_scale,  de/dg2 = i(A - B e^2)/e_scale
           => A = e_scale (w1 - i w2)/2,   B e^2 = e_scale (w1 + i w2)/2
           with w_k = dz3/dg_k + i dz4/dg_k.  A and B are real; the imaginary
           parts are a consistency check on the whole inversion.

Per-template values carry Var[Q|m] scatter and blow up as |e| -> 0 (the 1/|e|^2),
so they are smoothed by a weighted polynomial regression on the four invariants
(z0, z1, z2, q=|e|^2) that the network itself sees, weight |e|^2.  The fitted
function is then differentiated w.r.t. z to give the physical steepness.

DO NOT TRUST THIS SCRIPT'S NUMBERS (2026-08-20).  The weighted polynomial
smoothing below underfits badly: it reports the physical |d coeff / dz| as
0.00 (p99 0.01-0.02) for every coefficient, while a ratio-of-sums binned
estimate of the very same conditional mean gives ~15 for a_Mr and shows the
coefficient itself running from 1.4 to 40 across z1 deciles.  The design matrix
carries q^3 terms with q up to 85 and is then weighted by q again, so a handful
of extreme-ellipticity rows determine the fit.  Its |a1| p99 = 6.4 is what
sized `_COEFF_MAX`, and it is wrong -- see `dev/flexibility_audit.py`.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bulk, shear

D = "../bfd_cnf_imsims/data"
m_all, dmdg, _ = shear.load(f"{D}/moments.fits")
N = 100_000
m = jnp.asarray(m_all[:N]); dmdg_j = jnp.asarray(dmdg[:N])

flow = bulk.build_flow(jr.key(0), np.asarray(m_all), shear=True)
chart = shear._chart(flow)
e_scale = float(jax.numpy.asarray(shear._shear_layer(flow).e_scale.tree
              if hasattr(shear._shear_layer(flow).e_scale, "tree")
              else shear._shear_layer(flow).e_scale))

z = jax.vmap(chart.transform)(m)
J = jax.vmap(jax.jacfwd(chart.transform))(m)          # dz/dm
v = jnp.einsum("nij,nkj->nki", J, dmdg_j)             # (N,2,5) dz/dg

z3, z4 = z[:, 3], z[:, 4]
q = z3 ** 2 + z4 ** 2
# spin-0 coefficients
num = v[:, 0, :3] * z3[:, None] + v[:, 1, :3] * z4[:, None]
a = num / (e_scale * q)[:, None]                       # (N,3)
# spin-2 coefficients
w1 = v[:, 0, 3] + 1j * v[:, 0, 4]
w2 = v[:, 1, 3] + 1j * v[:, 1, 4]
A = e_scale * (w1 - 1j * w2) / 2
e2 = (z3 + 1j * z4) ** 2
B = e_scale * (w1 + 1j * w2) / (2 * e2 * e_scale ** 2)
print(f"inversion check -- |Im|/|Re| should be ~0:  A {float(jnp.median(jnp.abs(A.imag)/(jnp.abs(A.real)+1e-12))):.2e}"
      f"   B {float(jnp.median(jnp.abs(B.imag)/(jnp.abs(B.real)+1e-12))):.2e}")
coef = jnp.stack([a[:, 0], a[:, 1], a[:, 2], A.real, B.real], -1)   # (N,5)

# weighted polynomial regression on the invariants the network sees
u = jnp.stack([z[:, 0], z[:, 1], z[:, 2], q], -1)
umu, usd = u.mean(0), u.std(0)
un = (u - umu) / usd
def basis(x):
    t = [jnp.ones(())]
    for i in range(4):
        t.append(x[i])
    for i in range(4):
        for j in range(i, 4):
            t.append(x[i] * x[j])
    for i in range(4):
        t.append(x[i] ** 3)
    return jnp.stack(t)
Phi = jax.vmap(basis)(un)
wt = q                                            # downweight ill-determined round galaxies
ok = jnp.isfinite(coef).all(1) & (q > jnp.quantile(q, 0.05))
Pw = Phi[ok] * wt[ok, None]
beta = jnp.linalg.lstsq(Pw, coef[ok] * wt[ok, None], rcond=None)[0]   # (nb,5)

def fitted(zi):
    qq = zi[3] ** 2 + zi[4] ** 2
    x = (jnp.stack([zi[0], zi[1], zi[2], qq]) - umu) / usd
    return basis(x) @ beta                                            # (5,)

G = jax.vmap(jax.jacfwd(fitted))(z[:20000])                           # (n,5,5) dcoef/dz
nrm = jnp.sqrt(jnp.sum(G ** 2, -1))                                   # (n,5)
names = ["a0(Mf)", "a1(Mr)", "a2(Mc)", "A", "B"]
print(f"\nPHYSICAL |d coeff / dz|, from bfd's own dm/dg:")
print(f"{'coeff':>9}{'median':>10}{'p99':>10}{'max':>10}")
for i, n_ in enumerate(names):
    c = np.asarray(nrm[:, i])
    print(f"{n_:>9}{np.median(c):10.2f}{np.percentile(c,99):10.2f}{c.max():10.1f}")
print(f"\nfor comparison, TRAINED flows (median |dc/dz|):")
print(f"   g_max 0.02 leaky: A 4.63   B 10.65   s0 40.63   max 117159")
print(f"   g_max 0.10 clean: A 0.85   B 58.17   s0 53.63   max   8264")
