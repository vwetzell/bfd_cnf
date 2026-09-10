"""Is R broken because the flow's density-derivative is wrong, or because the
estimator differentiates in the wrong GAUGE?

`bias.log_conv_is` holds the draws fixed and moves the PRIOR:

    Phat(g) = (1/S) sum_s w_s p(d_s | g)      w_s fixed          [gauge P]

so every g-derivative lands on `d_g log p(d|g)`, which by the chain rule is
`grad_m log p . dm/dg` -- the flow's own SCORE, an unsmoothed prior gradient.

BFD's template sum does the opposite: the g-dependence sits in the KERNEL,

    P(M|g) = sum_t w_t N(M - m_t(g); C)                          [gauge K]

so its derivatives are `C^-1 (M - m_t) . dm_t/dg` -- bounded by construction.

Both are the same P(M|g).  The shear layer is a PUSHFORWARD (models/shear.py:
"P(m|g) is the pushforward of the unlensed p(m) through it"), so substituting
m = Psi_g(u) inside the integral moves every g out of the density and into the
kernel exactly:

    P(M|g) = INT p(m|g) L(M-m) dm = INT p(u|0) L(M - Psi_g(u)) du

with Psi_g = Phi^-1 . shear(.,g) . Phi and Phi = centroid.transform . chart
(the two g-independent layers ahead of the shear layer).  Estimated on the SAME
draws with the SAME weights:

    Phat'(g) = (1/S) sum_s [p(d_s|0)/q(d_s)] L(M - Psi_g(d_s))    [gauge K]

Identical to gauge P at g = 0, unbiased for P(M|g) at every g, and its
derivatives never touch the flow's score.

If R agrees between the gauges, the flow's density derivative really is wrong.
If gauge K gives a sane R where gauge P gives +146 and a negative Fisher ratio,
then what is broken is the ESTIMATOR's conditioning, and the "score variance is
29x too large" comparison (rsplit.py in gauge P against tsplit.py in gauge K)
was comparing two different decompositions -- A and B are gauge-dependent, only
R = A + B is not.
"""
import sys
import numpy as np, fitsio, jax, jax.numpy as jnp, equinox as eqx
import jax.random as jr

import bias, bulk, shear
from bias import (mixture_draws, in_domain, condition, load_cov, safe_point)

D = "../bfd_cnf_imsims/data"
import os
NT = int(os.environ.get("NT", 2000))
CHUNK = int(os.environ.get("CHUNK", 4096))
SEED = int(os.environ.get("SEED", 0))
ALPHA = float(os.environ.get("ALPHA", 0.5))
TBATCH = int(os.environ.get("TBATCH", 8))


def prep(pop, flow_path):
    cat = bias.CATALOGS[pop]
    rows = fitsio.read(f"{D}/{cat['zero']}.fits")[:NT]
    rows = rows[~rows["badcenter"]]
    m = np.asarray(rows["moments"], np.float64)
    sx = np.asarray(rows["cov_odd"], np.float64)
    cov = load_cov(f"{D}/{cat['zero']}.fits")
    m_train = shear.load(f"{D}/{bias.TRAIN_DATA[pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    return eqx.tree_deserialise_leaves(flow_path, flow), m, sx, cov


def make_psi(flow):
    """Psi_g: the raw-moment diffeomorphism the shear layer pushes the prior by.

    data -> base the chain is [raw2standard, centroid, shear, *bulk], so the two
    layers ahead of `shear` are g-independent and Psi_g conjugates the shear
    layer's own base -> data map (`ShearResponse.shear`) through them.
    """
    bij = flow.bijection.bijection.bijections
    chart, cen, sh = bij[0], bij[1], bij[2]

    def psi(m_raw, g, sxi):
        z = cen.transform(chart.transform(m_raw), jnp.concatenate([g * 0, sxi]))
        y = sh.shear(z, g)
        return chart.inverse(cen.inverse(y, jnp.concatenate([g * 0, sxi])))

    return psi


def check_pushforward(flow, psi, m, sx):
    """The one thing gauge K assumes: log p(Psi_g(u)|g) + log|dPsi/du| = log p(u|0).

    Holds identically if the shear layer is a pushforward.  `ShearResponse.shear`
    inverts `unshear`'s g-series to O(g^3), so this is exact at g = 0 and in the
    first two g-derivatives -- which is all Q and R use.
    """
    # On healthy targets only: a noisy moment can sit past the chart's
    # point-source ceiling, where log p is -4e10 and the comparison is noise.
    keep = np.asarray(in_domain(jnp.asarray(m)))
    g = jnp.array([0.02, -0.01])
    u = jnp.asarray(m[keep][:32], jnp.float32)
    sxi = jnp.asarray(sx[keep][:32], jnp.float32)

    def one(ui, si):
        f = lambda x: psi(x, g, si)
        ld = jnp.linalg.slogdet(jax.jacfwd(f)(ui))[1]
        return (flow.log_prob(f(ui), condition=condition(g, si)) + ld
                - flow.log_prob(ui, condition=condition(jnp.zeros(2), si)))

    r = np.asarray(jax.vmap(one)(u, sxi))
    r = r[np.abs(r) < 1e4]
    print(f"pushforward residual at g=(0.02,-0.01): median |d log p| = "
          f"{np.median(np.abs(r)):.2e}, max {np.abs(r).max():.2e} nats "
          f"over {len(r)} targets", flush=True)
    return np.median(np.abs(r))


def decompose(flow, psi, m, sx, cov):
    """Per-target (Q, A, B) in both gauges, from one shared chunk of draws."""
    Cinv = jnp.asarray(np.linalg.inv(cov), jnp.float32)
    zero = jnp.zeros(2)

    def logL(off):                       # up to a g-independent constant
        return -0.5 * jnp.einsum("...i,ij,...j->...", off, Cinv, off)

    def split(gr, he, u):
        pi = jax.nn.softmax(u)
        Q = jnp.einsum("s,sa->a", pi, gr)
        A = jnp.einsum("s,sab->ab", pi, he)
        B = jnp.einsum("s,sa,sb->ab", pi, gr, gr) - jnp.outer(Q, Q)
        return Q, A, B

    @eqx.filter_jit
    def one(m_i, draws_i, log_wt_i, sxi, ok_i):
        d = jnp.where(ok_i[:, None], draws_i, safe_point(m_i))

        # gauge P: g moves the prior, draws fixed.
        fp = lambda z: (lambda g: flow.log_prob(z, condition=condition(g, sxi)))
        lp0, grP, heP = jax.vmap(
            lambda z: (fp(z)(zero), jax.grad(fp(z))(zero),
                       jax.hessian(fp(z))(zero)))(d)

        # gauge K: g moves the kernel, prior frozen at its g = 0 value.
        # The additive `lp0 + log_wt - logL(M-d)` is g-independent, so it drops
        # out of grad and hess entirely -- only logL(M - Psi_g(d)) survives.
        fk = lambda z: (lambda g: logL(m_i - psi(z, g, sxi)))
        grK, heK = jax.vmap(
            lambda z: (jax.grad(fk(z))(zero), jax.hessian(fk(z))(zero)))(d)

        u = jnp.where(ok_i, lp0 + log_wt_i, -jnp.inf)
        # `Psi_g`'s Jacobian runs through `_chart_spin0_jac`'s 1/(1-u), which
        # diverges as a draw approaches the point-source ceiling.  Those draws
        # are the ones `r-collapse-is-score-variance` already found carry pi = 0
        # exactly; zero them and REPORT the weight lost, so the drop is checked
        # rather than assumed.
        fin = jnp.isfinite(grK).all(-1) & jnp.isfinite(heK).reshape(len(u), -1).all(-1)
        lost = jnp.sum(jnp.where(fin, 0.0, jax.nn.softmax(u)))
        grK = jnp.where(fin[:, None], grK, 0.0)
        heK = jnp.where(fin[:, None, None], heK, 0.0)
        return (split(grP, heP, u), split(grK, heK, u),
                1.0 / jnp.sum(jax.nn.softmax(u) ** 2), lost)

    out = []
    for i in range(0, len(m), TBATCH):
        m_b, sx_b = m[i:i + TBATCH], sx[i:i + TBATCH]
        d, lw = mixture_draws(flow, m_b, cov, CHUNK, ALPHA,
                              SEED + 104729 * (i // TBATCH),
                              batch=len(m_b), sigma_x=sx_b)
        ok = in_domain(jnp.asarray(d))
        res = jax.vmap(one)(jnp.asarray(m_b, jnp.float32),
                            jnp.asarray(d, jnp.float32),
                            jnp.asarray(lw, jnp.float32),
                            jnp.asarray(sx_b, jnp.float32), ok)
        out.append(jax.tree.map(lambda a: np.asarray(a, np.float64), res))
        if i % (TBATCH * 25) == 0:
            print(f"  {i}/{len(m)}", flush=True)
    return jax.tree.map(lambda *a: np.concatenate(a), *out)


def report(name, m, Q, A, B, ess):
    R = A + B
    q1, a11, b11, r11 = Q[:, 0], A[:, 0, 0], B[:, 0, 0], R[:, 0, 0]
    ed = np.percentile(m[:, 0], [0, 20, 40, 60, 80, 100])
    print(f"\n--- {name}   median ESS {np.median(ess):.0f}")
    print(f"{'Mf quintile':>12s}{'A':>13s}{'B':>13s}{'R=A+B':>13s}{'Fisher R':>10s}")
    for k in range(5):
        s = (m[:, 0] >= ed[k]) & (m[:, 0] <= ed[k + 1])
        print(f"{'q'+str(k+1):>12s}{a11[s].mean():13.3e}{b11[s].mean():13.3e}"
              f"{r11[s].mean():13.3e}{(q1[s]**2).sum()/(-r11[s]).sum():10.3f}")
    print(f"{'ALL':>12s}{a11.mean():13.3e}{b11.mean():13.3e}{r11.mean():13.3e}"
          f"{(q1**2).sum()/(-r11).sum():10.3f}")


if __name__ == "__main__":
    pop = sys.argv[1] if len(sys.argv) > 1 else "bulgedisc_deep_v2"
    fp = sys.argv[2] if len(sys.argv) > 2 else "flows/centroid_bulgedisc_v2.eqx"
    flow, m, sx, cov = prep(pop, fp)
    psi = make_psi(flow)
    assert check_pushforward(flow, psi, m, sx) < 1e-2, "not a pushforward"
    (QP, AP, BP), (QK, AK, BK), ess, lost = decompose(flow, psi, m, sx, cov)
    print(f"\ngauge-K weight dropped for a non-finite dPsi/dg: "
          f"median {np.median(lost):.2e}, max {lost.max():.2e} of 1.0")
    print(f"\n=== {pop}  {fp}   n={len(m)}, {CHUNK} draws, alpha={ALPHA}")
    report("gauge P (current: g moves the prior)", m, QP, AP, BP, ess)
    report("gauge K (g moves the kernel, as BFD)", m, QK, AK, BK, ess)
    print(f"\nQ agreement between gauges (should be exact-ish): "
          f"median |dQ1/Q1| = {np.median(np.abs(QK[:,0]/QP[:,0] - 1)):.3e}")
