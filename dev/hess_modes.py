"""Autodiff ordering for bias.py's per-target (value, grad, Hessian) in g.  Diagnostic only.
Same call shape as `one_t`: T targets x Sd peeled draws, f = log_conv_is on the peeled flow.
Modes: rev = `_val_grad_hess` (forward-over-reverse, current); fwd = jacfwd(jacfwd);
dir = d2/dt2 along e1, e2, e1+e2 (nested jvp; H12 by polarization).  Reports ms per target-arm
(8192 draws), compiled temp MiB per draw, and max |diff| vs rev.
Usage: python dev/hess_modes.py FLOW [T Sd]"""
import sys, time; sys.path.insert(0, ".")
import fitsio, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import bias as B, bulk, shear
D = "../bfd_cnf_imsims/data"
m_train = shear.load(f"{D}/{B.TRAIN_DATA['bulgedisc_g2n_1m_desv']}")[0]
sx = jnp.asarray(np.asarray(fitsio.read(f"{D}/{B.CATALOGS['bulgedisc_g2n_1m_desv']['zero']}.fits",
                                        columns=["cov_odd"], rows=[0])["cov_odd"])[0], jnp.float32)
path = sys.argv[1]
T, Sd = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (16, 4096)
flow = bulk.load_flow(path, m_train, key=jr.key(0), shear=True, centroid=True)
flow_g, layer = B.split_centroid(flow)
rows = np.random.default_rng(0).choice(len(m_train), T * Sd, replace=False)
d, ld = B.centroid_transform(layer, jnp.asarray(m_train[rows], jnp.float32),
                             np.broadcast_to(np.asarray(sx), (T * Sd, 3)), to_host=False)
d = d.reshape(T, Sd, 5)
lw = jnp.asarray(np.random.default_rng(1).normal(0, 2, (T, Sd)), jnp.float32)
ok = jnp.ones((T, Sd), bool)
m0 = jnp.zeros((T, 5))
z0, E = jnp.zeros(2), jnp.eye(2)


def rev(f):
    return B._val_grad_hess(f)


def fwd(f):
    def grad_aux(g):
        J = jax.jacfwd(lambda h: (f(h), f(h)), has_aux=True)(g)   # (grad, val)
        return J[0], J
    H, (gr, v) = jax.jacfwd(grad_aux, has_aux=True)(z0)
    return v, gr, H


def dirn(f):
    U = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
    def d2(u):
        df = lambda g: jax.jvp(f, (g,), (u,))[1]
        (d1, dd) = jax.jvp(df, (z0,), (u,))
        return d1, dd
    v = f(z0)
    d1, dd = jax.vmap(d2)(U)
    h12 = 0.5 * (dd[2] - dd[0] - dd[1])
    return v, d1[:2], jnp.array([[dd[0], h12], [h12, dd[1]]])


def make(mode):
    def one(m_i, d_i, lw_i, ok_i):
        f = lambda g: B.log_conv_is(flow_g, m_i, d_i, lw_i, B.condition(g, sx), ok_i)
        return mode(f)
    return jax.jit(jax.vmap(one))


# taylor: adapter x1 = x0 + W(x0) g (exactly linear), shear y = x1 + Q(x1) g + g.R(x1).g/2 (exactly
# quadratic), so everything bias.py reads at g = 0 is fixed by g-independent coefficients:
#   y = x0 + V g + g.Rh.g/2,  V = W + Q,  Rh_iab = R_iab + C_iab + C_iba,  C_iab = dQ_ia/dx_j W_jb
#   ld_shear = t1.g + g.T2.g/2,  t1_a = tr A_a,  T2_ab = tr dR_ab - tr A_a A_b + D_ab + D_ba,
#   A_a = dQ_a/dx, D_ab = d(tr A_a)/dx . W_b;  ld_adapter = slogdet(I + K.g) exactly, K_a = dW_a/dx.
# Only the bulk (unconditional) stays under the g-autodiff.
bij = list(flow_g.bijection.bijection.bijections)
ad, sh, rest = bij[0], bij[1], bij[2:]
base = flow_g.base_dist


def pre(x0):
    w = lambda x, a: ad._w(x, jnp.concatenate([E[a], sx]))
    W = jnp.stack([w(x0, 0), w(x0, 1)], -1)                                      # (5, 2)
    K = jnp.stack([jax.jacfwd(w)(x0, 0), jax.jacfwd(w)(x0, 1)])                  # (2, 5, 5)
    (Q, R), (A, dR) = sh.response_tensors(x0), jax.jacfwd(sh.response_tensors)(x0)  # A (5,2,5)
    divQ = lambda x: jnp.einsum("iai->a", jax.jacfwd(lambda z: sh.response_tensors(z)[0])(x))
    Dm = jax.vmap(lambda v: jax.jvp(divQ, (x0,), (v,))[1], in_axes=1, out_axes=1)(W)  # D[a, b]
    C = jnp.einsum("iaj,jb->iab", A, W)
    t1 = jnp.einsum("iai->a", A)
    T2 = (jnp.einsum("iabi->ab", dR) - jnp.einsum("iaj,jbi->ab", A, A) + Dm + Dm.T)
    return W + Q, R + C + C.transpose(0, 2, 1), t1, T2, K


def bulk_lp(y):
    tot = 0.0
    for l in rest:
        y, ld = l.transform_and_log_det(y, None)
        tot = tot + ld
    return tot + base.log_prob(y)


def one_taylor(m_i, d_i, lw_i, ok_i):
    x = jnp.where(ok_i[:, None], d_i, 0.0)
    V, Rh, t1, T2, K = jax.vmap(pre)(x)
    def lp(g):
        y = x + V @ g + 0.5 * jnp.einsum("siab,a,b->si", Rh, g, g)
        ld_ad = jnp.linalg.slogdet(jnp.eye(5) + jnp.einsum("saij,a->sij", K, g))[1]
        return jax.vmap(bulk_lp)(y) + ld_ad + t1 @ g + 0.5 * jnp.einsum("sab,a,b->s", T2, g, g)
    f = lambda g: (jax.nn.logsumexp(jnp.where(ok_i, lp(g) + lw_i, -jnp.inf)) - jnp.log(d_i.shape[0]))
    return B._val_grad_hess(f)


args = (m0, d, lw, ok)
ref = None
for name, mode in [("rev", rev), ("taylor", None), ("fwd", fwd), ("dir", dirn)]:
    fn = jax.jit(jax.vmap(one_taylor)) if mode is None else make(mode)
    try:
        mem = fn.lower(*args).compile().memory_analysis().temp_size_in_bytes / 2**20 / (T * Sd)
        out = jax.block_until_ready(fn(*args))
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True); continue
    t = time.perf_counter()
    for _ in range(5): jax.block_until_ready(fn(*args))
    ms = (time.perf_counter() - t) / 5 / (T * Sd) * 8192 * 1e3
    if ref is None: ref = out
    err = [float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30)) for a, b in zip(out, ref)]
    print(f"{name}: {ms:7.3f} ms/target-arm  temp {mem:.4f} MiB/draw  rel diff vs rev val/grad/hess "
          f"{err[0]:.1e} {err[1]:.1e} {err[2]:.1e}", flush=True)

# split of the taylor mode: coefficients alone vs the whole kernel (the rest is the bulk's g-Hessian)
fp = jax.jit(jax.vmap(jax.vmap(pre)))
jax.block_until_ready(fp(d)); t = time.perf_counter()
for _ in range(5): jax.block_until_ready(fp(d))
print(f"taylor coefficients alone (adapter W, K + shear Q, R, dQ, dR, D): "
      f"{(time.perf_counter() - t) / 5 / (T * Sd) * 8192 * 1e3:.3f} ms/target-arm", flush=True)

# Option-1 emulation (adapter AFTER the shear layer): shear terms at the g-independent x0 with no D;
# the adapter then needs W, K at x0 plus their derivative along Q (its own 2nd order, 32-wide net).
def pre_swap(x0):
    w = lambda z, a: ad._w(z, jnp.concatenate([E[a], sx]))
    Wf = lambda z: jnp.stack([w(z, 0), w(z, 1)], -1)
    (Q, R), (A, dR) = sh.response_tensors(x0), jax.jacfwd(sh.response_tensors)(x0)
    Kf = lambda z: jnp.stack([jax.jacfwd(w)(z, 0), jax.jacfwd(w)(z, 1)])
    (W, K), (dW, dK) = jax.vmap(lambda v: jax.jvp(lambda z: (Wf(z), Kf(z)), (x0,), (v,)), in_axes=1, out_axes=0)(Q)
    W, K = W[0], K[0]
    C = jnp.einsum("bia->iab", dW)
    t1 = jnp.einsum("iai->a", A)
    T2 = jnp.einsum("iabi->ab", dR) - jnp.einsum("iaj,jbi->ab", A, A) + jnp.einsum("baii->ab", dK)
    return W + Q, R + C + C.transpose(0, 2, 1), t1, T2, K


def one_swap(m_i, d_i, lw_i, ok_i):
    x = jnp.where(ok_i[:, None], d_i, 0.0)
    V, Rh, t1, T2, K = jax.vmap(pre_swap)(x)
    def lp(g):
        y = x + V @ g + 0.5 * jnp.einsum("siab,a,b->si", Rh, g, g)
        ld_ad = jnp.linalg.slogdet(jnp.eye(5) + jnp.einsum("saij,a->sij", K, g))[1]
        return jax.vmap(bulk_lp)(y) + ld_ad + t1 @ g + 0.5 * jnp.einsum("sab,a,b->s", T2, g, g)
    f = lambda g: (jax.nn.logsumexp(jnp.where(ok_i, lp(g) + lw_i, -jnp.inf)) - jnp.log(d_i.shape[0]))
    return B._val_grad_hess(f)


for name, fn in (("taylor (now)", jax.jit(jax.vmap(one_taylor))), ("swapped order (option 1)", jax.jit(jax.vmap(one_swap))),
                 ("taylor (now)", jax.jit(jax.vmap(one_taylor))), ("swapped order (option 1)", jax.jit(jax.vmap(one_swap)))):
    jax.block_until_ready(fn(*args)); t = time.perf_counter()
    for _ in range(10): jax.block_until_ready(fn(*args))
    print(f"{name}: {(time.perf_counter() - t) / 10 / (T * Sd) * 8192 * 1e3:.3f} ms/target-arm", flush=True)
