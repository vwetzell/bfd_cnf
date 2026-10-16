"""Cost split of bias.log_conv_is_taylor's g-independent coefficients (fp32, GPU).  Diagnostic only.
Usage: python dev/profile_taylor_pre.py FLOW"""
import sys, time; sys.path.insert(0, ".")
import fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
D_ = "../bfd_cnf_imsims/data"
m_train = shear.load(f"{D_}/{B.TRAIN_DATA['bulgedisc_g2n_1m_desv']}")[0]
sx = jnp.asarray(np.asarray(fitsio.read(f"{D_}/{B.CATALOGS['bulgedisc_g2n_1m_desv']['zero']}.fits",
                                        columns=["cov_odd"], rows=[0])["cov_odd"])[0], jnp.float32)
N = 65536
flow = bulk.load_flow(sys.argv[1], m_train, key=jr.key(0), shear=True, centroid=True)
flow_g, layer = B.split_centroid(flow)
ad, sh, rest = B.taylor_layers(flow_g)
rows = np.random.default_rng(0).choice(len(m_train), N, replace=False)
x, _ = B.centroid_transform(layer, jnp.asarray(m_train[rows], jnp.float32), np.broadcast_to(np.asarray(sx), (N, 3)), to_host=False)
E = jnp.eye(2)
w = lambda z, a: ad._w(z, jnp.concatenate([E[a], sx]))
Wf = lambda z: jnp.stack([w(z, 0), w(z, 1)], -1)
W = jax.jit(jax.vmap(Wf))(x)


def timed(name, f, *a):
    f = jax.jit(jax.vmap(f))
    jax.block_until_ready(f(*a)); t = time.perf_counter()
    for _ in range(10): jax.block_until_ready(f(*a))
    print(f"{name:55s} {(time.perf_counter() - t) / 10 / N * 8192 * 1e3:.3f} ms/target-arm", flush=True)


timed("adapter W", Wf, x)
timed("adapter W + K (2 jacfwd)", lambda z: (Wf(z), jnp.stack([jax.jacfwd(w)(z, 0), jax.jacfwd(w)(z, 1)])), x)
timed("adapter W + K (1 jacfwd of W)", lambda z: (Wf(z), jax.jacfwd(Wf)(z)), x)
timed("shear Q, R", sh.response_tensors, x)
timed("shear Q, R + jacfwd (A, dR)", lambda z: (sh.response_tensors(z), jax.jacfwd(sh.response_tensors)(z)), x)
divQ = lambda z: jnp.einsum("iai->a", jax.jacfwd(lambda u: sh.response_tensors(u)[0])(z))
timed("D alone (vmap_b jvp(divQ))", lambda z, Wz: jax.vmap(lambda v: jax.jvp(divQ, (z,), (v,))[1], in_axes=1, out_axes=1)(Wz), x, W)
both = lambda z: (sh.response_tensors(z), jax.jacfwd(sh.response_tensors)(z))
timed("Q,R,A,dR + D in one jvp of (QR, jacfwd QR)",
      lambda z, Wz: jax.vmap(lambda v: jax.jvp(both, (z,), (v,)), in_axes=1)(Wz), x, W)
timed("full pre as in bias (all)", lambda z: z, x)

# D split: projection algebra with the coefficients frozen vs the coefficient net's own 2nd derivatives
from models.shear import project_to_physics, _invariants
from models.bijections import unwrap
es, cl, cs = unwrap(sh.e_scale), unwrap(sh.chart_loc), unwrap(sh.chart_scale)
coeff = lambda z: sh.coeffs(*_invariants(z)[:4])
Qc = lambda z, c: project_to_physics(c, z, jnp.zeros(2), es, cl, cs)[0]
def D_of(qfun):
    div = lambda z: jnp.einsum("iai->a", jax.jacfwd(qfun)(z))
    return lambda z, Wz: jax.vmap(lambda v: jax.jvp(div, (z,), (v,))[1], in_axes=1, out_axes=1)(Wz)
timed("D, coefficients frozen (algebra only)", lambda z, Wz: D_of(lambda u: Qc(u, jax.lax.stop_gradient(coeff(z))))(z, Wz), x, W)
timed("2nd derivs of coefficient net alone (jvp_b of jacfwd)", lambda z, Wz: jax.vmap(lambda v: jax.jvp(jax.jacfwd(coeff), (z,), (v,))[1], in_axes=1)(Wz), x, W)
timed("invariants alone, 2nd order", lambda z, Wz: jax.vmap(lambda v: jax.jvp(jax.jacfwd(lambda u: jnp.stack(_invariants(u)[:4])), (z,), (v,))[1], in_axes=1)(Wz), x, W)
timed("coefficient net value only", coeff, x)
timed("coefficient net + jacfwd_x (5 tangents)", lambda z: jax.jacfwd(coeff)(z), x)
U = jax.jit(jax.vmap(lambda z: jnp.stack(_invariants(z)[:4])))(x)
cu = lambda u: sh.coeffs(u[0], u[1], u[2], u[3])
dU = jax.jit(jax.vmap(lambda z, Wz: jax.jacfwd(lambda t: jnp.stack(_invariants(t)[:4]))(z) @ Wz))(x, W)   # (N, 4, 2)
timed("u-space: jacfwd_u (4 tangents)", lambda u: jax.jacfwd(cu)(u), U)
timed("u-space: jvp_v of jacfwd_u (2 dirs x 4)", lambda u, v: jax.vmap(lambda vb: jax.jvp(jax.jacfwd(cu), (u,), (vb,))[1], in_axes=1)(v), U, dU)
timed("u-space: hessian (jacfwd jacfwd, 16)", lambda u: jax.jacfwd(jax.jacfwd(cu))(u), U)
