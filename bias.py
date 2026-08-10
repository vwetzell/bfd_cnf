"""
bias.py
=======
Multiplicative and additive shear bias of a trained flow, measured on the imsims
targets -- noiseless, or with a noise realization integrated over.

The BFD ensemble estimator (Bernstein & Armstrong 2014 sec. 3) maximises

    log L(g) = sum_i log P(M_i | g)

over g, expanded to second order about g = 0,

    ghat = -[ sum_i d2 logP_i/dg2 ]^-1 [ sum_i d logP_i/dg ].

Those two derivatives ARE BFD's PQR combinations -- d logP/dg = Q/P and
d2 logP/dg2 = R/P - Q Q^T/P^2 (paper eq. 12-13, 45-46) -- so differentiating
`log P` with respect to g gives them without ever forming P, Q and R
separately, and `ghat` above is the paper's eq. (19).

What `log P(M_i|g)` is depends on the target's noise:

* **noiseless targets** (`--samples 0`): the noise kernel is a delta and the
  integral collapses to a point evaluation of the prior at the target's own
  moments -- no sampling, no weights, no ESS.
* **noisy targets** (`--samples S`): the target carries M = M^G + M^n with
  M^n ~ N(0, C_M) (paper eq. 5), and P is the prior convolved with that noise,

      P(M_i | g) = INT dm P(m | g) L(M_i - m),   L = N(., C_M)

  which is the continuum limit of the paper's template sum, eq. (38).  It is
  estimated by drawing S points from the noise kernel itself -- see `log_conv`.

Bias comes from the +g/-g pair, which is the SAME galaxies sheared both ways
(same seed, and the same noise realization), so the paired difference carries
no shape noise:

    m1 = (ghat1[+] - ghat1[-]) / (2 g) - 1        c = (ghat[+] + ghat[-]) / 2

Only m1 is measurable here: the catalogs are sheared along g1 only, so there is
no lever arm on m2.  The unsheared catalog gives c independently.

No selection is applied -- every target is used, at any flux -- so P(s|g) = 1
and the non-detection terms of paper eq. (45)-(46) are absent.  Cut on a noisy
flux and they stop being absent.

Usage:
    python bias.py --flow flows/shear.eqx
    python bias.py --flow flows/shear.eqx --samples 1024 --n-targets 100000
"""

from __future__ import annotations

import argparse

import bfd
import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import bulk
import shear

# Catalogs, as (label, filename stem), per population.  The +/- pair share a
# seed and so are the same galaxies; the g=0 run is the same galaxies again.
CATALOGS = {
    "bulgedisc": {"plus": "targets_g1p02_1M", "minus": "targets_g1m02_1M",
                  "zero": "targets_g0_1M"},
    "sersic": {"plus": "sersic_tg1p_20k", "minus": "sersic_tg1m_20k",
               "zero": "sersic_tg1z_20k"},
}


def load_cov(path):
    """C_M, the even-moment noise covariance of a target (paper eq. 9).

    It is a pure noise covariance -- set by the PSF, the weight function and the
    depth, not by the galaxy -- so in these sims every target shares one C_M and
    one Cholesky factor serves for all of them.  Heteroscedastic catalogs would
    only mean carrying a factor per target.
    """
    c = bfd.MomentCovariance.bulkUnpack(fitsio.read(path)["cov"])
    assert np.allclose(c, c[0]), "C_M varies between targets; see load_cov"
    return np.asarray(c[0], dtype=np.float64)


def add_noise(m, cov, seed):
    """One noise realization on the targets: M = M^G + M^n, M^n ~ N(0, C_M).

    Drawing the noise in moment space rather than redrawing noisy stamps is not
    an approximation.  The moments are a linear functional of the image at a
    FIXED centre (paper eq. 5, 7-8) and the pixel noise is Gaussian, so M^n is
    exactly multivariate normal with the covariance eq. (9) already stores.
    What it does assume is the known centre: letting the detection re-find
    X = 0 in the presence of noise is the centroid marginalisation, deferred.
    """
    L = np.linalg.cholesky(cov)
    return m + np.random.default_rng(seed).standard_normal(m.shape) @ L.T


def kernel_draws(cov, n, samples, seed):
    """`samples` offsets per target, drawn from the noise kernel N(0, C_M).

    Antithetic in pairs: eps and -eps both appear, which cancels the leading
    term of the estimator's error where the prior varies linearly across the
    kernel -- which is most of it, the kernel being narrow next to the
    population.  Independent draws PER TARGET, deliberately: sharing one set
    across targets would correlate their Monte Carlo errors so they no longer
    average down in the ensemble sums of eq. (45)-(46).
    """
    L = np.linalg.cholesky(cov)
    half = np.random.default_rng(seed).standard_normal((n, samples // 2, 5)) @ L.T
    return np.concatenate([half, -half], axis=1)


def log_conv(flow, m_i, eps, g):
    """log INT dm P(m|g) L(M_i - m), by Monte Carlo over the noise kernel.

    L is symmetric, so drawing m = M_i + eps with eps ~ N(0, C_M) makes

        Phat = (1/S) sum_s P(M_i + eps_s | g)

    an unbiased estimator of the convolution -- the continuum limit of the
    paper's sum over templates, eq. (38), with the flow standing in for
    sum_G p_G delta(m - M^G).  The draws do not depend on g, so Q and R
    (eq. 12-13) come from differentiating THIS estimator: one estimator
    differentiated, with common random numbers, not three noisy ones divided.
    """
    x = m_i + eps
    # The prior's chart is [log10 Mf, Mr/Mf, ...], so a draw with Mf <= 0 or
    # Mr <= 0 cannot be evaluated -- and needs no evaluating: no galaxy has a
    # negative flux or size, the true prior is zero there, and zero weight is
    # the right answer.  The flow is still called at an in-domain dummy point
    # for those rows so that no NaN can reach the gradient.
    ok = (x[:, 0] > 0) & (x[:, 1] > 0)
    lp = flow.log_prob(jnp.where(ok[:, None], x, m_i), condition=g)
    return jax.nn.logsumexp(jnp.where(ok, lp, -jnp.inf)) - jnp.log(eps.shape[0])


def _over_targets(one, m, eps, batch):
    """Run a per-target function over the catalog in batches, in float64 out.

    The flow is float32 (it was trained that way and the checkpoint stores f32),
    but a 1M-galaxy sum in float32 would lose more than the bias being measured,
    so the per-galaxy values are promoted before they are ever accumulated.
    """
    batched = eqx.filter_jit(jax.vmap(one))
    f32 = lambda a: None if a is None else jnp.asarray(a, dtype=jnp.float32)
    out = []
    for i in range(0, len(m), batch):
        out.append(jax.tree.map(lambda a: np.asarray(a, dtype=np.float64),
                                batched(f32(m[i:i + batch]),
                                        f32(None if eps is None else eps[i:i + batch]))))
    return jax.tree.map(lambda *a: np.concatenate(a), *out)


def pqr(flow, m, eps=None, batch=20000):
    """Per-target (d logP/dg, d2 logP/dg2) at g = 0, as float64.

    With `eps` the target is noisy and P is the convolution above; without it
    the target is noiseless and P is a point evaluation of the prior.
    """
    zero = jnp.zeros(2)

    def one(m_i, eps_i):
        f = (lambda g: flow.log_prob(m_i, condition=g)) if eps is None else (
            lambda g: log_conv(flow, m_i, eps_i, g))
        return jax.grad(f)(zero), jax.hessian(f)(zero)

    return _over_targets(one, m, eps, batch)


def ess(flow, m, eps, batch=20000):
    """Effective number of draws behind Phat, (sum w)^2 / sum w^2, w_s = P(M+eps_s).

    This is the paper's own diagnostic for the one bias its estimator carries:
    sec. 2.5 notes that dividing Q and R by a noisy Phat biases the result
    inversely with "the number of template galaxies contributing significantly
    to the P_i sums", and here that number is the ESS.  Read it as: the induced
    m is of order 1/ESS, so it wants to be well past 1e3 to be irrelevant.
    """
    def one(m_i, eps_i):
        x = m_i + eps_i
        ok = (x[:, 0] > 0) & (x[:, 1] > 0)
        lp = flow.log_prob(jnp.where(ok[:, None], x, m_i), condition=jnp.zeros(2))
        lp = jnp.where(ok, lp, -jnp.inf)
        return jnp.exp(2 * jax.nn.logsumexp(lp) - jax.nn.logsumexp(2 * lp))

    return _over_targets(one, m, eps, batch)


def ghat(q, r, sel=None):
    """The BFD ensemble shear estimate over the selected targets."""
    if sel is not None:
        q, r = q[sel], r[sel]
    return -np.linalg.solve(r.sum(0), q.sum(0))


def bias(qp, rp, qm, rm, g=0.02, sel=None):
    """(m1, c1, c2) from the +g/-g pair."""
    gp, gm = ghat(qp, rp, sel), ghat(qm, rm, sel)
    return (gp[0] - gm[0]) / (2 * g) - 1, *(0.5 * (gp + gm))


def bootstrap(qp, rp, qm, rm, g=0.02, n=200, seed=0, sel=None):
    """Paired bootstrap over galaxies: the same resampled index into BOTH
    catalogs, so the shape noise that the +/- pairing cancels stays cancelled."""
    rng = np.random.default_rng(seed)
    idx0 = np.arange(len(qp)) if sel is None else np.flatnonzero(sel)
    out = [bias(qp[i], rp[i], qm[i], rm[i], g)
           for i in (rng.choice(idx0, len(idx0)) for _ in range(n))]
    return np.std(np.array(out), axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--pop", choices=sorted(CATALOGS), default="bulgedisc")
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--train-data", default=None,
                   help="catalog the flow was trained on (sets the flow's "
                        "standardisation; defaults to the matching moments file)")
    p.add_argument("--g", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--samples", type=int, default=0,
                   help="Monte Carlo draws per target from its noise kernel. "
                        "0 leaves the targets noiseless and P a point "
                        "evaluation; >0 adds a noise realization and integrates "
                        "the prior under C_M.")
    p.add_argument("--noise-seed", type=int, default=1,
                   help="the targets' noise realization; shared by the +g, -g "
                        "and unsheared catalogs so the pairing still cancels "
                        "shape noise")
    p.add_argument("--n-targets", type=int, default=None,
                   help="use only the first N targets (the integration is "
                        "`samples` flow evaluations per target)")
    a = p.parse_args()

    train_data = a.train_data or (
        f"{a.data_dir}/moments{'_sersic' if a.pop == 'sersic' else ''}.fits")
    # Rebuild exactly as `shear.py train` did -- the flow's RawMomentStandardize
    # is fixed by the training split, so the same 90% has to go in here.
    m_train = shear.load(train_data)[0]
    flow = bulk.build_flow(jr.key(a.seed), m_train[:int(0.9 * len(m_train))],
                           shear=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    print(f"{a.flow} on the {a.pop} targets")

    cat = CATALOGS[a.pop]
    path = lambda v: f"{a.data_dir}/{v}.fits"
    n = slice(None) if a.n_targets is None else slice(a.n_targets)
    m = {k: np.asarray(fitsio.read(path(v))["moments"], dtype=np.float64)[n]
         for k, v in cat.items()}
    truth = m["zero"]           # noiseless, unsheared -- for binning only

    eps, batch = None, 20000
    if a.samples:
        cov = load_cov(path(cat["zero"]))
        # The SAME noise realization and the SAME kernel draws for a galaxy in
        # all three catalogs, so what the +/- difference cancels stays cancelled.
        m = {k: add_noise(v, cov, a.noise_seed) for k, v in m.items()}
        eps = kernel_draws(cov, len(truth), a.samples, a.noise_seed + 1000)
        # The Hessian is forward-over-reverse, so a batch costs several times
        # `batch * samples` flow activations -- keep that product bounded.
        batch = max(1, 131_072 // a.samples)
        e = ess(flow, m["zero"][:2000], eps[:2000], 4 * batch)
        print(f"integrating under C_M with {a.samples} draws/target: "
              f"ESS = {np.median(e):.0f} (median), {np.percentile(e, 5):.0f} (5th pct)")

    qr = {k: pqr(flow, v, eps, batch) for k, v in m.items()}
    qp, rp = qr["plus"]
    qm, rm = qr["minus"]

    m1, c1, c2 = bias(qp, rp, qm, rm, a.g)
    dm1, dc1, dc2 = bootstrap(qp, rp, qm, rm, a.g, a.boot, a.seed)
    g0 = ghat(*qr["zero"])
    print(f"\n{len(qp)} targets, g1 = +/-{a.g}")
    print(f"  m1 = {m1:+.5f} +/- {dm1:.5f}")
    print(f"  c1 = {c1:+.2e} +/- {dc1:.1e}   c2 = {c2:+.2e} +/- {dc2:.1e}")
    print(f"  unsheared catalog: ghat = ({g0[0]:+.2e}, {g0[1]:+.2e})")

    # Flux quintiles: the bias is expected to vary far more across the
    # population than its mean does, so the mean alone can hide a lot.  Binned
    # on the NOISELESS unsheared flux, which is independent of both the noise
    # realization and the shear -- binning on the measured flux would be a
    # selection on a noisy quantity, and would owe eq. (40) and (46).
    flux = truth[:, 0]
    edges = np.percentile(flux, [0, 20, 40, 60, 80, 100])
    print(f"\n{'Mf quintile':>14s}{'m1':>12s}{'c1':>12s}{'c2':>12s}")
    for i in range(5):
        sel = (flux >= edges[i]) & (flux <= edges[i + 1])
        b = bias(qp, rp, qm, rm, a.g, sel)
        d = bootstrap(qp, rp, qm, rm, a.g, max(a.boot // 4, 50), a.seed, sel)
        print(f"{f'q{i + 1}':>14s}{b[0]:>+12.4f}{b[1]:>+12.2e}{b[2]:>+12.2e}"
              f"   (+/- {d[0]:.4f})")


if __name__ == "__main__":
    main()
