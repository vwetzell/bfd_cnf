"""Why does pure NLL not recover the spin-0 response, and what restores it?

`dev/selfconsistency_runaway.py` already removed every excuse: draw the
templates from a flow's OWN bulk and give them that flow's OWN shear response,
and the target is exactly representable, the frozen bulk is exactly the data
density, Var[Q|m] is exactly zero, and bfd's derivatives are not involved.
Pure NLL still recovered only alpha = 0.43 / 0.36 / 0.32 on Mf / Mr / Mc at 96k
steps (spin-2: 1.01), and 400k made it worse, -0.86.  So the failure is in the
OBJECTIVE, not in the data, the bulk, the chart or the bound.

The mechanism it points at: the spin-0 response is odd in e, so its effect on an
isotropic population's density cancels at O(g) and first appears at O(g^2),
while the spin-2 response is coherent and appears at O(g).  At g_max = 0.02 that
is a ~400x weaker signal.  `dev/spin0_profile.py` measures what is left: a clean
convex minimum at the truth with Fisher information 0.011/galaxy.  Identified,
but with a curvature two orders below spin-2's -- a nearly flat direction that
the optimiser drifts along rather than converging in.

That control ran before today's chart-Jacobian reparameterisation, the e_scale
fix and the precision pin, so it is re-run here, with the two levers that could
plausibly restore the signal:

  nll g0.02   the baseline, and a check that the failure survives the fixes
  nll g0.10   spin-0 curvature grows as g^2 while spin-2's grows as g, so this
              is 25x the leverage.  Legitimate even though real lensing is not
              a second-order Taylor map at |g| = 0.1: `lens` GENERATES with the
              exact same second-order form the layer FITS, and only the g = 0
              derivatives are ever used downstream, so the wider fit recovers
              the same coefficients with more information and no bias.
  nll+deriv   the supervision removed in fa50e81, restored at its old weight.
              bfd's per-template dm/dg is exact for the templates, and the
              layer's minimiser under it is E[dm/dg | m] -- what the layer
              represents.  Here the "bfd" derivatives are the generator's own,
              so this arm asks whether the term can hold the direction at all.

alpha is measured against the GENERATOR's response, which is the exact target.
"""
import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
import optax
import bulk, shear
from models.bijections import in_domain
from models.shear import ShearResponse, dm_dg

D_ = "../bfd_cnf_imsims/data"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 96_000
N = 100_000
GEN = "flows/armDL_s1.eqx"

real = shear.load(f"{D_}/moments.fits")[0]
G = bulk.build_flow(jr.key(0), np.asarray(real[:int(0.9 * len(real))]), shear=True)
G = eqx.tree_deserialise_leaves(GEN, G)
layerG, chart = shear._shear_layer(G), shear._chart(G)

m_syn = G.sample(jr.key(7), (N,), condition=jnp.zeros(2))
qs, rs = [], []
for i in range(0, N, 20000):
    q, r = jax.vmap(dm_dg, in_axes=(None, 0, None))(layerG, m_syn[i:i + 20000], chart)
    qs.append(q); rs.append(r)
q_all, r_all = jnp.concatenate(qs), jnp.concatenate(rs)
# Templates that stay ON CHART when lensed at the WIDEST g any arm trains at,
# so every arm sees the same set.  Real templates sit well below the ceiling
# (`dev/gmax_sweep.py` checked them out to g_max = 0.2), but these are drawn
# from the flow, whose support runs all the way up to it -- lensing those at
# |g| = 0.1 puts Mr/Mf past POINT_SOURCE, the chart is undefined there, and the
# arm trains on `nll inf` from step 0.  That is what killed the first attempt.
G_WIDE = 0.10
ok = jnp.ones(N, bool)
for k in range(16):
    ang = 2 * jnp.pi * k / 16
    gk = jnp.tile(jnp.array([G_WIDE * jnp.cos(ang), G_WIDE * jnp.sin(ang)]), (N, 1))
    ok &= in_domain(shear.lens(m_syn, q_all, r_all, gk))
keep = np.asarray(ok)
m_syn, q_all, r_all = m_syn[keep], q_all[keep], r_all[keep]
data = (np.asarray(m_syn), np.asarray(q_all), np.asarray(r_all))
print(f"synthetic templates from {GEN}: {data[0].shape} "
      f"({100 * (1 - keep.mean()):.2f}% dropped as off-chart at |g| = {G_WIDE})",
      flush=True)

m_e, q_e = m_syn[:20000], q_all[:20000]


def partial_norms(m, q_true, r_true):
    """Per-partial RMS of the truth -- fa58e81's `partial_norms` at its nbin=1
    default, reimplemented here because that function went with the term.

    Not optional: dM1/dg1 has RMS 0.367 against dM1/dg2's 0.0134, so a plain
    mean over the (2, 5) array supervises them 750:1, and the second-order
    block swamps the first outright.  Leaving it out is what made the first
    attempt's supervised arm meaningless.
    """
    s = shear._scale(m)[:, None, :]
    return (jnp.sqrt(jnp.mean((q_true / s) ** 2, axis=0)),
            jnp.sqrt(jnp.mean((r_true / s) ** 2, axis=0)))


NQ, NR = partial_norms(jnp.asarray(data[0]), jnp.asarray(data[1]),
                       jnp.asarray(data[2]))


def fresh():
    f = bulk.build_flow(jr.key(0), np.asarray(real[:int(0.9 * len(real))]), shear=True)
    keep = [i for i, b in enumerate(f.bijection.bijection.bijections)
            if not isinstance(b, ShearResponse)]
    src = [b for b in G.bijection.bijection.bijections
           if not isinstance(b, ShearResponse)]
    return eqx.tree_at(lambda ff: [ff.bijection.bijection.bijections[i] for i in keep],
                       f, src)


def train_supervised(flow, data, key, steps, batch=1024, lr=3e-3, g_max=0.02,
                     deriv_weight=1e4):
    """`shear.train` plus fa50e81's removed velocity term.  Same everything else."""
    m, q, r = (jnp.asarray(a) for a in data)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(optax.cosine_decay_schedule(lr, steps)))
    params, static = eqx.partition(flow, shear._trainable(flow, True))
    state = opt.init(params)

    def one(carry, _):
        params, state, key = carry
        key, sk, gk = jr.split(key, 3)
        idx = jr.randint(sk, (batch // 2,), 0, m.shape[0])
        g, take = shear._antithetic(idx, gk, g_max)
        x = shear.lens(m[take], q[take], r[take], g)

        def loss_fn(p):
            model = eqx.combine(p, static)
            nll = -jnp.mean(model.log_prob(x, condition=g))
            lay, ch = shear._shear_layer(model), shear._chart(model)
            mi = m[idx]
            qq, rr = jax.vmap(dm_dg, in_axes=(None, 0, None))(lay, mi, ch)
            s = shear._scale(mi)[:, None, :]
            mse = (jnp.mean(((qq - q[idx]) / s / NQ) ** 2)
                   + jnp.mean(((rr - r[idx]) / s / NR) ** 2))
            return nll + deriv_weight * mse

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = opt.update(grads, state, params)
        return (eqx.apply_updates(params, updates), state, key), loss

    @eqx.filter_jit
    def run(params, state, key, n):
        return jax.lax.scan(one, (params, state, key), None, length=n)

    done = 0
    while done < steps:
        n = min(shear.REPORT, steps - done)
        (params, state, key), losses = run(params, state, key, n)
        done += n
        print(f"step {done - 1:6d}  loss {float(losses[-1]):.4f}", flush=True)
    return eqx.combine(params, static)


def measure(f, g_eval=0.02):
    lay, ch = shear._shear_layer(f), shear._chart(f)
    z = jax.vmap(ch.transform)(m_e)
    ldp = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([g_eval, 0.]))[1])(z)
    ldm = jax.vmap(lambda zi: lay.transform_and_log_det(zi, jnp.array([-g_eval, 0.]))[1])(z)
    q, _ = jax.vmap(dm_dg, in_axes=(None, 0, None))(lay, m_e, ch)
    s = shear._scale(m_e)[:, None, :]
    p, t = np.asarray(q / s), np.asarray(q_e / s)
    a = ((np.mean((p + t) ** 2, (0, 1)) - np.mean((p - t) ** 2, (0, 1)))
         / (4 * np.mean(t ** 2, (0, 1))))
    return float(jnp.mean(ldp + ldm) / 2), a


ARMS = (("nll g0.02", 0.02, False), ("nll g0.10", 0.10, False),
        ("nll+deriv", 0.02, True))
print(f"\n{'arm':>10}{'seed':>5}{'even':>10}"
      + "".join(f"{n:>8}" for n in ["Mf", "Mr", "M1", "M2", "Mc"])
      + "     (alpha vs the GENERATOR, 1.00 = recovered)", flush=True)
out = {}
for name, gm, sup in ARMS:
    rows = []
    for seed in (1, 2):
        f = (train_supervised(fresh(), data, jr.key(seed), STEPS, g_max=gm) if sup
             else shear.train(fresh(), data, jr.key(seed), steps=STEPS, batch=1024,
                              g_max=gm))
        even, a = measure(f)
        rows.append(a)
        print(f"{name:>10}{seed:>5}{even:>10.5f}"
              + "".join(f"{x:8.2f}" for x in a), flush=True)
    out[name] = np.array(rows)
print()
for name, al in out.items():
    print(f"{name:>10}  mean" + "".join(f"{al[:,k].mean():8.2f}" for k in range(5)))
