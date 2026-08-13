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
There are two kinds of noisy target, and the catalog's `IMGNOISE` header says
which:

* **noise added in moment space** (`add_noise`, the older catalogs).  Exact only
  at a KNOWN centre, where the moments are a linear functional of the image.
* **noise added in the image**, then measured with a real `recenter()` (the
  `*_noisy_*` catalogs).  Here the centroid is FOUND, so the moments carry a
  centroid error and the prior must marginalise over it -- which is the centroid
  layer, switched on automatically for such a catalog.  `add_noise` is not
  applied to these: the noise is already in them, in the only place it can be
  and still let the detection respond to it.

* **noisy targets** (`--samples S`): the target carries M = M^G + M^n with
  M^n ~ N(0, C_M) (paper eq. 5), and P is the prior convolved with that noise,

      P(M_i | g) = INT dm P(m | g) L(M_i - m),   L = N(., C_M)

  which is the continuum limit of the paper's template sum, eq. (38).  It is
  estimated by importance sampling S draws from a proposal (`log_conv_is`):
  the pure noise kernel (`log_conv`, `mixture_draws` at alpha = 1, the default)
  or a defensive mixture of the kernel and the prior (`mixture_draws`,
  `--alpha < 1`).

  WHICH PROPOSAL WINS DEPENDS ON DEPTH, and the earlier blanket verdict against
  prior-aware proposals held only where the kernel was already adequate:

      noise_sigma      alpha = 1            alpha = 0.5
      1.0              median ESS 129       median ESS 77        (mixture worse)
      2.73             median ESS 25        median ESS 103       (mixture better)
                       frac(ESS<10) 0.30    frac(ESS<10) 0.013

  At noise_sigma 2.73 the kernel is genuinely starved -- C_M is 7.45x wider in
  variance, so draws centred on M_i rarely land on the prior's support -- and
  that is exactly the regime `mixture_draws` was written for.  The tail is what
  moves, not just the median.

  Two caveats before reaching for it.  It costs ~15x per draw (it samples from
  the flow, running the shear Newton solve and the centroid fixed point), so
  brute-force alpha = 1 at 16x the draws is ~3x cheaper for the same ESS.  And
  alpha < 1 draws part of the proposal FROM THE FLOW, so two runs with different
  flows no longer share random numbers and `--compare`'s pairing weakens.

  A base-space Laplace/Student-t proposal was tried and was worse than either
  (frac(ESS < 10) 0.52 against the kernel's 0.09, unrescued by widening, tail
  weight or mode-quality); it has been removed.

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
    # image noise + recentring; the centroid layer switches on from the header
    python bias.py --flow flows/centroid.eqx --pop bulgedisc_noisy --samples 2048
    # the same targets without it -- what the marginalisation is worth
    python bias.py --flow flows/shear.eqx --no-centroid --pop bulgedisc_noisy --samples 2048
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

from flowjax.bijections import Chain, Invert
from flowjax.distributions import Transformed

import bulk
import shear
from models.bijections import POINT_SOURCE, in_domain
from models.centroid import CentroidMarginalize


def safe_point(m):
    """An in-domain stand-in for rows the mask will discard anyway.

    The flow is still CALLED on masked rows -- `jnp.where` evaluates both
    branches, and a NaN in the discarded one still poisons the gradient -- so
    they need coordinates the chart can actually evaluate, comfortably inside
    the point-source ceiling rather than merely below it.
    """
    mf = jnp.maximum(m[..., 0], 1e-6)
    mr = jnp.clip(m[..., 1], 1e-6, 0.5 * POINT_SOURCE * mf)
    return m.at[..., 0].set(mf).at[..., 1].set(mr)

# Full float32 matmuls, not the TF32 the GPU defaults to.  TF32 keeps 10
# mantissa bits, so a log-density that accumulates through a ~12-layer flow
# comes out with ~1 nat of error -- and XLA picks TF32 kernels by autotuning,
# which is a per-PROCESS decision, so two runs of the SAME code disagreed by
# 1.2 nats on draws carrying real weight.  That is fatal here twice over: the
# paired +g/-g comparison assumes two processes compute identical weights from
# identical draws, and 1 nat is a factor e in a weight.  Measured on this
# machine: cross-process max |dlog_wt| 1.2 -> 1.4e-3, for 7% throughput.
# Only the inference path needs this; training noise at TF32 is harmless.
jax.config.update("jax_default_matmul_precision", "highest")

# Catalogs, as (label, filename stem), per population.  The +/- pair share a
# seed and so are the same galaxies; the g=0 run is the same galaxies again.
CATALOGS = {
    "bulgedisc": {"plus": "targets_g1p02_1M", "minus": "targets_g1m02_1M",
                  "zero": "targets_g0_1M"},
    "sersic": {"plus": "sersic_tg1p_20k", "minus": "sersic_tg1m_20k",
               "zero": "sersic_tg1z_20k"},
    # Pixel noise in the IMAGE and a real recenter(), so the centroid is found
    # rather than assumed.  These need the centroid layer -- and they must NOT
    # be handed to `add_noise`, which is only valid at a fixed centre.  The
    # IMGNOISE header says which kind a catalog is; `main` reads it rather than
    # trusting the name.
    "bulgedisc_noisy": {"plus": "targets_noisy_g1p02_200k",
                        "minus": "targets_noisy_g1m02_200k",
                        "zero": "targets_noisy_g0_200k"},
    # Same population at noise_sigma = 2.73, which puts the 5th percentile of
    # flux S/N at 10 and the median at 19.  That is where the centroid
    # marginalisation is worth +2e-3 to +1.4e-2 rather than the +3.5e-4 it is at
    # the shallower depth -- i.e. where a with/without comparison can resolve
    # it.  Needs flows trained on copies_bulgedisc_deep.fits: the copy grid is
    # only valid within a factor 1.4 in sigma_XY, and this is 2.73x.
    "bulgedisc_deep": {"plus": "targets_deep_g1p02_200k",
                       "minus": "targets_deep_g1m02_200k",
                       "zero": "targets_deep_g0_200k"},
    # The analytic population: two co-elliptical Gaussians whose moments were
    # DRAWN from a chosen density rather than pushed forward from galaxy
    # parameters, so P(m|g), Q and R are known in closed form -- `truth.py`.
    # This is the only population where a measured m or c can be compared
    # against what it should have been, rather than only against zero.
    "gauss2": {"plus": "gauss2_g1p02_1M", "minus": "gauss2_g1m02_1M",
               "zero": "gauss2_g0_1M"},
    # A 2k version of the same, for smoke-testing the pipeline end to end
    # without waiting on a 1M-galaxy render.  Far too small to measure a bias
    # with -- it is there so `check_flow_vs_truth` can be exercised in minutes.
    "gauss2_2k": {"plus": "gauss2_g1p02_2k", "minus": "gauss2_g1m02_2k",
                  "zero": "gauss2_g0_2k"},
}


def condition(g, sigma_x):
    """The flow's condition vector, given the shear and (optionally) Sigma_X.

    Without the centroid layer it is just g.  With it the chain carries
    [g1, g2, C00, C01, C11] -- shear reads the first two, centroid the last
    three.  Everything downstream differentiates with respect to `g` ALONE and
    treats `sigma_x` as a fixed per-target constant, which is what makes Q and R
    (eq. 12-13) still mean what the estimator needs: the target's noise
    properties are data, not something lensing moves.
    """
    return g if sigma_x is None else jnp.concatenate([g, sigma_x])


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


def log_conv_is(flow, m_i, draws, log_wt, g):
    """log (1/S) sum_s exp(log_wt_s) P(draws_s | g) -- the general importance-
    sampled estimator of the convolution P(M_i|g) = INT dm P(m|g) L(M_i - m),
    the continuum limit of the paper's sum over templates, eq. (38).

    `g` is the flow's full condition vector -- just the shear without the
    centroid layer, or [g1, g2, C00, C01, C11] with it (see `condition`).  It is
    forwarded verbatim, so callers differentiating with respect to shear vary
    only its leading two entries.

    `draws` are absolute points in moment space (not offsets from M_i) and
    `log_wt_s` is the log ratio of L(M_i - draws_s) to whatever proposal
    density produced draws_s, fixed once and reused unchanged as g varies --
    so Q and R (eq. 12-13) still come from differentiating ONE estimator with
    common random numbers, exactly as in `log_conv`.  `log_conv`'s pure-kernel
    draws are the alpha = 1 special case, log_wt = 0 everywhere: the kernel is
    its own proposal, so L cancels out of the ratio entirely.
    """
    # The prior's chart cannot evaluate a draw with Mf <= 0, Mr <= 0, or
    # Mr/Mf at or above the point-source ceiling -- and needs no evaluating:
    # no galaxy has a negative flux or size or is larger than the PSF is small,
    # the true prior is zero there, and zero weight is the right answer.  The
    # ceiling is the half that matters here: the kernel is wide enough that a
    # well-resolved target still puts draws past it, and before the chart was
    # bounded the flow answered those with a positive density.
    ok = in_domain(draws)
    dummy = safe_point(m_i)
    lp = flow.log_prob(jnp.where(ok[:, None], draws, dummy), condition=g)
    return (jax.nn.logsumexp(jnp.where(ok, lp + log_wt, -jnp.inf))
            - jnp.log(draws.shape[0]))


def log_conv(flow, m_i, eps, g):
    """log INT dm P(m|g) L(M_i - m), by Monte Carlo over the noise kernel.

    L is symmetric, so drawing m = M_i + eps with eps ~ N(0, C_M) makes

        Phat = (1/S) sum_s P(M_i + eps_s | g)

    an unbiased estimator of the convolution, with the flow standing in for
    sum_G p_G delta(m - M^G).  The draws do not depend on g, so Q and R
    (eq. 12-13) come from differentiating THIS estimator: one estimator
    differentiated, with common random numbers, not three noisy ones divided.
    This is `log_conv_is` with the kernel as its own proposal (log_wt = 0);
    see `mixture_draws` for the general, defensive-mixture proposal.
    """
    return log_conv_is(flow, m_i, m_i + eps, jnp.zeros(eps.shape[0]), g)


def mixture_draws(flow, m, cov, samples, alpha, seed, batch=None, sigma_x=None):
    """Draws and log-importance-weights from the defensive mixture proposal

        q(m) = alpha * N(m; M_i, C_M) + (1 - alpha) * P(m | g=0)

    `flow` here is the PROPOSAL -- it need not be the flow whose density is
    being integrated.  A proposal only has to COVER the posterior, not match
    it, so callers may pass a flow shared across runs (`--proposal-flow`) to
    keep the flow-component draws common random numbers even when the flow
    actually being evaluated differs between runs.

    Pure-kernel sampling (`kernel_draws`) starves whenever C_M is wide next to
    the prior: the ESS can sit below 10 for a real fraction of targets no
    matter how many draws are spent, because every draw comes from the SAME
    place and none of them land where the prior actually has mass.  Mixing in
    draws from the prior itself fixes that: since q >= alpha * L(M_i - m)
    everywhere, the importance weight w = P(m|g)/q(m) is bounded by
    sup_m P(m|g) / alpha, so the estimator has finite variance regardless of
    how badly L and the prior overlap.

    Both components are drawn and their densities evaluated at g = 0, NOT at
    the g the caller will eventually want -- the proposal only has to COVER
    the posterior, not track it.  Sampling from P(.|g) would put the draws
    inside the g-autodiff and add a pathwise gradient term through the
    sampler; freezing the proposal at g = 0 keeps every g-dependence in the
    numerator of `log_conv_is`, preserving the "one estimator differentiated,
    common random numbers" property the module docstring describes.

    Returns (draws, log_wt), float64 numpy arrays of shape (n, samples, 5) and
    (n, samples), matching `kernel_draws`'s convention except that `draws` are
    absolute moment-space points, not offsets (log_conv_is needs the flow-
    component draws, which are not offsets from anything).
    """
    n = len(m)
    # Sampling runs the flow BASE -> DATA, which for the shear layer is a Newton
    # solve plus a jacfwd log-det per draw (models/shear.py) -- several times the
    # cost and memory of the data -> base log_prob direction, even at g = 0 where
    # the layer is the identity.  So the chunk is sized on `batch * samples`, the
    # same budget `pqr` uses for its forward-over-reverse Hessian.
    batch = max(1, 131_072 // samples) if batch is None else batch
    n_k = round(alpha * samples)
    n_k -= n_k % 2                       # kernel half stays antithetic (pairs)
    n_f = samples - n_k
    # The alpha >= 1 branch below skips the flow entirely, so the evenness trim
    # turning an ODD `samples` into n_f = 1 would silently hand back one draw
    # fewer than asked -- a shape change, not a weight one.  Every caller passes
    # a power of two; this is the net under that.
    if alpha >= 1.0 and n_f:
        raise ValueError(f"alpha >= 1 needs an even `samples`; got {samples}")

    m64 = np.asarray(m, dtype=np.float64)
    kernel = kernel_draws(cov, n, n_k, seed)     # (n, n_k, 5) offsets, empty if n_k=0

    L = jnp.asarray(np.linalg.cholesky(cov))
    log_diag = jnp.sum(jnp.log(jnp.diag(L)))

    # Guard log(alpha) and log(1 - alpha) at the 0/1 endpoints so neither
    # produces a NaN; -inf * (finite log-density) is fine, it just drops that
    # component out of the logaddexp mixture density below.
    log_alpha = -jnp.inf if alpha <= 0.0 else float(np.log(alpha))
    log_1ma = -jnp.inf if alpha >= 1.0 else float(np.log1p(-alpha))

    key = jr.key(seed) if n_f else None
    draws_out, wt_out = [], []
    for i in range(0, n, batch):
        m_i = jnp.asarray(m64[i:i + batch])
        kern_i = jnp.asarray(kernel[i:i + batch])
        # The proposal is evaluated at g = 0, but at each target's OWN Sigma_X:
        # the centroid layer is part of the prior the proposal has to cover, and
        # unlike g it is not something the estimator differentiates.
        cond0 = (jnp.zeros(2) if sigma_x is None else
                 jax.vmap(condition, in_axes=(None, 0))(
                     jnp.zeros(2), jnp.asarray(sigma_x[i:i + batch],
                                               dtype=jnp.float32)))

        if alpha >= 1.0:
            # q IS the kernel, so log_wt = log_k - log_q is identically zero and
            # log_k need never be formed -- it is a triangular solve over every
            # draw whose result is then subtracted from itself.  There is no
            # flow work in this branch at all, so `_mixture_chunk` -- built to
            # hoist exactly that work -- is never called here.
            x = m_i[:, None, :] + kern_i
            draws_out.append(np.asarray(x, dtype=np.float32))
            wt_out.append(np.zeros(x.shape[:-1], dtype=np.float32))
            continue

        sub = None
        if n_f:
            key, sub = jr.split(key)
        x, log_wt = _mixture_chunk(flow, m_i, kern_i, cond0, sub, L, log_diag,
                                   log_alpha, log_1ma, n_f)
        # float32 on the host: these are (n, samples, 5) and at 32k draws/target
        # float64 would be 26 GB per catalog.  Nothing is lost -- `_over_targets`
        # casts to f32 on the way to the device anyway, because the flow is f32.
        draws_out.append(np.asarray(x, dtype=np.float32))
        wt_out.append(np.asarray(log_wt, dtype=np.float32))

    return np.concatenate(draws_out), np.concatenate(wt_out)


def split_centroid(flow):
    """Peel the data-adjacent centroid layer off the front of the chain.

    Returns `(rest, layer)`, or `(flow, None)` if there is no centroid layer.

    This is worth doing because of WHERE the layer sits.  In data -> base order
    the chain is [centroid, shear, raw2standard, bulk], so the centroid layer is
    applied FIRST, to the raw target moments, conditioned only on Sigma_X.  Both
    its input and its condition are independent of g, so

        log p(x | g, Sigma) = log p_rest(centroid(x, Sigma) | g)
                              + log |det d centroid / dx|

    and that second term is exactly g-independent: it contributes NOTHING to Q
    or R (eq. 12-13).  Evaluating it once, outside the autodiff, keeps the
    forward-over-reverse Hessian from ever traversing the layer's jacfwd log-det
    -- which is 5 extra JVPs of the coefficient network per draw, and what makes
    the Hessian run out of memory at the batch sizes the shear-only flow used.

    The peel is exact, not an approximation; see `centroid_transform`.
    """
    bij = flow.bijection.bijection.bijections
    if not isinstance(bij[0], CentroidMarginalize):
        return flow, None
    rest = Invert(Chain(list(bij[1:])).merge_chains())
    return Transformed(flow.base_dist, rest), bij[0]


@eqx.filter_jit
def _centroid_apply(layer, x, cond):
    """vmapped centroid transform + log-det, jitted ONCE at module level.

    Building the `eqx.filter_jit` wrapper inside `centroid_transform` instead
    made a fresh wrapper per call, so JAX recompiled on every chunk -- 1181 ms
    against 18 ms for the traced computation, and 91% of the whole integration
    step.  Equinox splits `layer` into its arrays and its structure, so retracing
    happens only if the structure or the shapes change, not when weights do.
    """
    return jax.vmap(layer.transform_and_log_det)(x, cond)


@eqx.filter_jit
def _mixture_chunk(flow, m_i, kern, cond0, key, L, log_diag, log_alpha, log_1ma, n_f):
    """One batch of `mixture_draws`' flow work, jitted ONCE at module level.

    `flow.sample` and `flow.log_prob` each traverse a ~12-layer flow whose
    every layer holds an MLP -- hundreds of individual kernel launches,
    dispatched one at a time from Python when called eagerly.  Measured at 8
    targets x 512 draws: eager `flow.sample` 1123.8 ms, eager `flow.log_prob`
    717.0 ms, both jitted together 2.0 ms -- 922x, and the difference between
    8% and full GPU utilisation on a production run.  As with `_centroid_apply`,
    building the wrapper inside `mixture_draws` would retrace on every batch;
    built here it retraces only when a batch's shapes actually change.

    Callers must not invoke this at alpha >= 1.0: with the kernel as its own
    proposal there is no flow work to hoist (log_wt is identically zero), so
    `mixture_draws` keeps that branch outside this function entirely.
    """
    x = m_i[:, None, :] + kern
    if n_f:
        if cond0.ndim == 1:
            flow_x = flow.sample(key, (m_i.shape[0], n_f), condition=cond0)
        else:
            # flowjax prepends sample_shape to the CONDITION's batch shape,
            # so a per-target condition already supplies the target axis --
            # asking for (B, n_f) on top of it would nest a second one.
            flow_x = jnp.moveaxis(flow.sample(key, (n_f,), condition=cond0), 0, 1)
        x = jnp.concatenate([x, flow_x], axis=1)

    def log_normal(offset):
        # log N(offset; 0, cov), via a triangular solve rather than inverting
        # cov -- the numerically stable way to get the quadratic form.
        shp = offset.shape[:-1]
        y = jax.scipy.linalg.solve_triangular(L, offset.reshape(-1, 5).T, lower=True)
        quad = jnp.sum(y * y, axis=0).reshape(shp)
        return -0.5 * quad - log_diag - 2.5 * jnp.log(2 * jnp.pi)

    log_k = log_normal(x - m_i[:, None, :])
    ok = in_domain(x)
    dummy = safe_point(m_i)
    safe = jnp.where(ok[..., None], x, dummy[:, None, :])
    # log_prob vectorises as (5),(5)->(), so a per-target condition has to
    # be broadcast along the DRAW axis as well as the target axis.
    cond_f = (cond0 if cond0.ndim == 1 else
              jnp.broadcast_to(cond0[:, None, :],
                               safe.shape[:-1] + cond0.shape[-1:]))
    log_f = jnp.where(ok, flow.log_prob(safe, condition=cond_f), -jnp.inf)
    log_q = jnp.logaddexp(log_alpha + log_k, log_1ma + log_f)

    # Where a draw is far enough out that BOTH densities underflow float32,
    # log_wt is -inf - (-inf) = NaN -- on a draw that is still nominally
    # in-domain (Mf, Mr > 0), so `log_conv_is`'s domain mask does not catch
    # it, and ONE such draw NaNs that target's whole logsumexp and with it
    # its Q and R.  A draw out where L has underflowed carries no weight, so
    # -inf is not a patch over the NaN, it IS the answer.
    log_wt = log_k - log_q
    return x, jnp.where(jnp.isfinite(log_wt), log_wt, -jnp.inf)


def centroid_transform(layer, x, sigma_x, batch=200_000, to_host=True):
    """Apply the centroid layer to `x`, returning (transformed, log-det).

    `x` is (..., 5) and `sigma_x` broadcasts against its leading axes.  The
    log-det is returned separately so the caller can fold it into an importance
    weight, which is where it belongs: it multiplies the density of that draw
    and never depends on g.  It cannot be skipped: it varies by ~18 nats across
    a chunk of kernel draws (they reach the poorly resolved region where the
    layer's T is large), and dropping it moves R by 8%.

    Everything is kept on device and the Sigma_X broadcast is done there too.
    Doing it in host numpy instead -- `np.broadcast_to(...).reshape()` plus a
    few `astype` copies of a (131072, 5) array per chunk -- cost 1181 ms against
    18 ms for the actual transform, i.e. 98% of this function and 91% of the
    whole integration step.  `to_host=False` also leaves the result on device
    for callers that feed it straight back into a jitted function.
    """
    shp = x.shape[:-1]
    xf = jnp.asarray(x, dtype=jnp.float32).reshape(-1, 5)
    sj = jnp.asarray(sigma_x, dtype=jnp.float32)
    # The layer keeps the chain's condition width even after being peeled off,
    # and reads Sigma_X as the LAST three entries -- so pad the shear slots.
    if layer.cond_shape[0] > 3:
        sj = jnp.concatenate(
            [jnp.zeros(sj.shape[:-1] + (layer.cond_shape[0] - 3,)), sj], axis=-1)
    sf = jnp.broadcast_to(sj.reshape((-1,) + (1,) * (len(shp) - 1) + sj.shape[-1:]),
                          shp + sj.shape[-1:]).reshape(-1, sj.shape[-1])

    z_out, ld_out = [], []
    for i in range(0, xf.shape[0], batch):
        z, ld = _centroid_apply(layer, xf[i:i + batch], sf[i:i + batch])
        z_out.append(z)
        ld_out.append(ld)
    z = jnp.concatenate(z_out).reshape(shp + (5,))
    ld = jnp.concatenate(ld_out).reshape(shp)
    if to_host:
        return np.asarray(z, dtype=np.float32), np.asarray(ld, dtype=np.float32)
    return z, ld


def _over_targets(one, arrays, batch):
    """Run a per-target function over the catalog in batches, in float64 out.

    `arrays` is a tuple of per-target arrays (or None, for an argument `one`
    doesn't use); vmap treats a None leaf as an empty pytree, so it is simply
    passed through unbatched -- the same trick the old (m, eps) version used.

    The flow is float32 (it was trained that way and the checkpoint stores f32),
    but a 1M-galaxy sum in float32 would lose more than the bias being measured,
    so the per-galaxy values are promoted before they are ever accumulated.
    """
    batched = eqx.filter_jit(jax.vmap(one))
    f32 = lambda a: None if a is None else jnp.asarray(a, dtype=jnp.float32)
    n = len(arrays[0])
    out = []
    for i in range(0, n, batch):
        chunk = tuple(f32(None if a is None else a[i:i + batch]) for a in arrays)
        out.append(jax.tree.map(lambda a: np.asarray(a, dtype=np.float64),
                                batched(*chunk)))
    return jax.tree.map(lambda *a: np.concatenate(a), *out)


def pqr(flow, m, draws=None, log_wt=None, batch=20000, sigma_x=None):
    """Per-target (d logP/dg, d2 logP/dg2) at g = 0, as float64.

    With `draws`/`log_wt` the target is noisy and P is the convolution above;
    without them the target is noiseless and P is a point evaluation of the
    prior -- a single flow call, no sampling, no weights.

    `sigma_x` switches on the centroid layer.  It enters the condition but NOT
    the differentiation: grad and hessian are taken with respect to the 2-vector
    g while Sigma_X is held fixed, because the target's noise properties are
    data, not something lensing moves.  Q and R therefore keep their eq. (12-13)
    meaning with the centroid marginalisation folded into P.
    """
    zero = jnp.zeros(2)

    def one(m_i, draws_i, log_wt_i, sigma_x_i):
        cond = lambda g: condition(g, sigma_x_i)
        f = (lambda g: flow.log_prob(m_i, condition=cond(g))) if draws is None else (
            lambda g: log_conv_is(flow, m_i, draws_i, log_wt_i, cond(g)))
        return jax.grad(f)(zero), jax.hessian(f)(zero)

    return _over_targets(one, (m, draws, log_wt, sigma_x), batch)


def pqr_streamed(flow, m, cov, samples, alpha, seed, sigma_x=None,
                 batch=64, chunk=2048, report=None,
                 proposal=None, proposal_sigma_x=None):
    """Per-target (d logP/dg, d2 logP/dg2) WITHOUT materialising the draws.

    `Phat = (1/S) sum_s w_s p(x_s|g)` is a plain sum over samples, and so are its
    two g-derivatives, so a target needs only three running accumulators

        A = sum_s w_s p_s      B = sum_s w_s grad p_s     C = sum_s w_s hess p_s

    from which `grad log P = B/A` and `hess log P = C/A - (B/A)(B/A)^T`.  Memory
    is then O(targets) instead of O(targets x samples): the (n, S, 5) draw array
    is 26 GB per catalog at n = 20000, S = 32768, and it never exists here.  That
    is what makes a large S affordable, which is the only thing that fixes the
    ESS starvation at depth.

    Each chunk is evaluated with the SAME `log_conv_is` the non-streamed path
    uses, then combined: a chunk returns `f = log(A_c / n_c)`, `grad f = B_c/A_c`
    and `hess f = C_c/A_c - (B_c/A_c)(B_c/A_c)^T`, which is enough to recover
    `A_c`, `B_c/A_c` and `C_c/A_c` and merge them as an A-weighted mean.  The
    merge runs in log space against a running max, so no chunk can overflow.

    Chunks are seeded per (target batch, chunk index), so the draws are common
    random numbers across catalogs exactly as before -- at alpha = 1.  With
    alpha < 1 part of the proposal comes from the flow itself, so two runs with
    DIFFERENT flows no longer share draws and the pairing weakens; that is a
    property of the defensive mixture, not of the streaming.  `proposal` (with
    `proposal_sigma_x`) restores it: `mixture_draws` draws its flow component
    and evaluates `log_q` from `proposal`/`proposal_sigma_x` instead of
    `flow`/`sigma_x`, while everything downstream of the draws --
    `split_centroid(flow)`, the centroid transform of draws and targets, and
    `log_conv_is` -- keeps evaluating `flow` at `sigma_x`.  `proposal is None`
    (the default) is exactly today's behavior.

    Accuracy, measured against a single logsumexp over the SAME draws:

        chunks   dQ/Q     dR/R
             1   3.3e-7   8.0e-5     (no merge -- exact)
             2   5.3e-3   9.8e-3
             4   3.7e-3   4.2e-3

    The merge is algebraically exact; the residual is float32 in the per-chunk
    `d2f`, which `c_over_a` has to un-cancel.  It is conditioned on the
    per-chunk ESS, and the table above is a worst case: 256 draws where the ESS
    is 24, i.e. ~4 effective samples per chunk.  Use the largest chunk that
    fits.  The perturbation is random per target, so it enters the eq. (45)-(46)
    ensemble sums as ~dR/sqrt(N) -- 4e-5 at N = 20000 -- and a component common
    to all targets cancels between +g and -g anyway.
    """
    zero = jnp.zeros(2)
    # `flow` still carries the centroid layer, because `mixture_draws` needs the
    # FULL prior to build its proposal.  Everything downstream of the transform
    # must use the peeled flow instead -- evaluating `flow` on already
    # transformed draws would apply the layer a second time.
    flow_g, layer = split_centroid(flow)

    def one(m_i, draws_i, log_wt_i, sigma_x_i):
        f = lambda g: log_conv_is(flow_g, m_i, draws_i, log_wt_i,
                                  condition(g, sigma_x_i))
        # One forward-over-reverse pass per g direction yields the value, the
        # gradient AND a column of the Hessian.  Calling f, jax.grad(f) and
        # jax.hessian(f) separately -- as this did -- evaluates the flow three
        # times over, and `hessian` is `jacfwd(grad)` so it already computed the
        # first two internally and discarded them.
        vg = jax.value_and_grad(f)
        (val, df), (_, h0) = jax.jvp(vg, (zero,), (jnp.array([1.0, 0.0]),))
        _, (_, h1) = jax.jvp(vg, (zero,), (jnp.array([0.0, 1.0]),))
        return val, df, jnp.stack([h0, h1], axis=-1)

    batched = eqx.filter_jit(jax.vmap(one))
    n_chunks = max(1, samples // chunk)
    q_out, r_out = [], []
    n_empty = 0

    for i in range(0, len(m), batch):
        m_b = m[i:i + batch]
        sx_b = None if sigma_x is None else sigma_x[i:i + batch]
        psx_b = None if proposal_sigma_x is None else proposal_sigma_x[i:i + batch]
        # The merge runs in float64 on the host.  It has to: `c_over_a` below is
        # rebuilt as `hess + (B/A)(B/A)^T`, and `hess` was itself computed as
        # `C/A - (B/A)(B/A)^T`, so at low ESS -- where B/A is large -- that is a
        # cancellation undone.  In float32 it costs 1.7% on R; in float64 it is
        # exact to 1e-7 against a single logsumexp over the same draws.
        log_a = np.full(len(m_b), -np.inf)
        bhat = np.zeros((len(m_b), 2))
        chat = np.zeros((len(m_b), 2, 2))
        # The targets' own transform does not depend on the chunk, so do it once
        # per target batch rather than once per chunk.
        m_z = (jnp.asarray(m_b, dtype=jnp.float32) if layer is None else
               centroid_transform(layer, m_b, sx_b, to_host=False)[0])

        for c in range(n_chunks):
            # The seed carries BOTH the chunk and the target batch.  Without the
            # batch term every target batch would replay the same offsets, so
            # targets in different batches would share a noise realization --
            # the correlation `kernel_draws` is documented to avoid, because
            # correlated per-target errors stop averaging down in the eq.
            # (45)-(46) ensemble sums.  Note this makes `batch` part of the
            # random stream: two runs to be compared must use the same one.
            d, lw = mixture_draws(
                flow if proposal is None else proposal, m_b, cov, chunk, alpha,
                seed + 7919 * c + 104729 * (i // batch),
                batch=len(m_b), sigma_x=sx_b if proposal is None else psx_b)
            lw = jnp.asarray(lw, dtype=jnp.float32)
            if layer is not None:
                # Stay on device: the transformed draws go straight back into a
                # jitted function, so a host round-trip here is pure loss.
                d, ld = centroid_transform(layer, d, sx_b, to_host=False)
                lw = lw + ld
            else:
                d = jnp.asarray(d, dtype=jnp.float32)
            f, df, d2f = batched(
                m_z, d, lw,
                None if sx_b is None else jnp.asarray(sx_b, dtype=jnp.float32))
            f = np.asarray(f, dtype=np.float64)
            df = np.asarray(df, dtype=np.float64)
            d2f = np.asarray(d2f, dtype=np.float64)
            # f = log(A_c / n_c); recover C_c/A_c from hess = C/A - (B/A)(B/A)^T
            la = f + np.log(chunk)
            c_over_a = d2f + np.einsum("ia,ib->iab", df, df)

            # A chunk contributes only if it carries weight and its derivatives
            # are finite.  A chunk whose draws all underflow gives la = -inf,
            # and with the accumulator also still -inf the max trick computes
            # -inf - -inf = NaN.  That is not hypothetical: at 20000 targets it
            # hit 1 (the faintest in the catalog, flux 755) and a single NaN row
            # takes out R_tot and with it m1, c1 and c2 for the WHOLE ensemble.
            good = (np.isfinite(la) & np.isfinite(df).all(1)
                    & np.isfinite(c_over_a).reshape(len(la), -1).all(1))
            df = np.where(good[:, None], df, 0.0)
            c_over_a = np.where(good[:, None, None], c_over_a, 0.0)

            mx = np.maximum(log_a, np.where(good, la, -np.inf))
            mxs = np.where(np.isfinite(mx), mx, 0.0)
            wa = np.where(np.isfinite(log_a), np.exp(log_a - mxs), 0.0)
            wc = np.where(good, np.exp(la - mxs), 0.0)
            tot = wa + wc
            upd = tot > 0.0
            den = np.where(upd, tot, 1.0)
            bhat = np.where(upd[:, None],
                            (bhat * wa[:, None] + df * wc[:, None]) / den[:, None],
                            bhat)
            chat = np.where(upd[:, None, None],
                            (chat * wa[:, None, None]
                             + c_over_a * wc[:, None, None]) / den[:, None, None],
                            chat)
            log_a = np.where(upd, mxs + np.log(den), log_a)

        q_out.append(bhat)
        r_out.append(chat - np.einsum("ia,ib->iab", bhat, bhat))
        n_empty += int((~np.isfinite(log_a)).sum())
        if report and (i // batch) % report == 0:
            print(f"    {i + len(m_b)}/{len(m)} targets", flush=True)

    if n_empty:
        # Q = R = 0 for these, so they drop out of the eq. (45)-(46) sums rather
        # than poisoning them -- but a rising count means the proposal is
        # failing, not that the targets are uninformative.
        print(f"    {n_empty} targets had no draw with any weight; "
              f"they contribute nothing", flush=True)
    return np.concatenate(q_out), np.concatenate(r_out)


def ess(flow, m, draws, log_wt, batch=20000, sigma_x=None):
    """Effective number of draws behind Phat, (sum w)^2 / sum w^2,
    w_s = exp(log_wt_s) * P(draws_s|g=0).

    This is the paper's own diagnostic for the one bias its estimator carries:
    sec. 2.5 notes that dividing Q and R by a noisy Phat biases the result
    inversely with "the number of template galaxies contributing significantly
    to the P_i sums", and here that number is the ESS.  Read it as: the induced
    m is of order 1/ESS, so it wants to be well past 1e3 to be irrelevant.
    With log_wt = 0 (the pure-kernel case) this is exactly the old w_s = P(M+eps_s).
    """
    def one(m_i, draws_i, log_wt_i, sigma_x_i):
        ok = in_domain(draws_i)
        dummy = safe_point(m_i)
        cond = condition(jnp.zeros(2), sigma_x_i)
        lp = flow.log_prob(jnp.where(ok[:, None], draws_i, dummy), condition=cond)
        lw = jnp.where(ok, lp + log_wt_i, -jnp.inf)
        return jnp.exp(2 * jax.nn.logsumexp(lw) - jax.nn.logsumexp(2 * lw))

    return _over_targets(one, (m, draws, log_wt, sigma_x), batch)


def ghat(q, r, sel=None):
    """The BFD ensemble shear estimate over the selected targets."""
    if sel is not None:
        q, r = q[sel], r[sel]
    return -np.linalg.solve(r.sum(0), q.sum(0))


def bias(qp, rp, qm, rm, g=0.02, sel=None):
    """(m1, c1, c2) from the +g/-g pair."""
    gp, gm = ghat(qp, rp, sel), ghat(qm, rm, sel)
    return (gp[0] - gm[0]) / (2 * g) - 1, *(0.5 * (gp + gm))


def save_pqr(path, qr, flux):
    """Write per-target Q and R so two runs can be differenced later.

    The point is the PAIRING.  Two runs over the same galaxies, the same noise
    realization and the same kernel draws differ only in the flow, so their
    difference in m1 is far better determined than either run's own bootstrap
    error -- the population scatter that dominates both cancels.  Recovering
    that needs the per-target values, which are otherwise summed away.
    """
    cols = {f"{k}_{n}": v for k, (q, r) in qr.items()
            for n, v in (("q", q), ("r", r))}
    np.savez_compressed(path, flux=flux, **cols)
    print(f"wrote {path}")


def compare(path_a, path_b, g=0.02, n=200, seed=0, labels=("A", "B")):
    """Paired difference in (m1, c1, c2) between two saved runs.

    Resamples galaxies ONCE per bootstrap draw and evaluates both runs on that
    same resample, so the difference keeps the pairing.  Reports each run's own
    bias with its unpaired error, then the difference with its paired one --
    which is the number that says whether the two flows actually differ.
    """
    a, b = np.load(path_a), np.load(path_b)
    if len(a["plus_q"]) != len(b["plus_q"]):
        raise SystemExit("the two runs cover different numbers of targets")
    # Drop non-finite targets from BOTH runs jointly, so the pairing -- the
    # whole point of this comparison -- stays galaxy for galaxy.
    ok = np.ones(len(a["plus_q"]), dtype=bool)
    for d in (a, b):
        for k in ("plus", "minus"):
            ok &= np.isfinite(d[f"{k}_q"]).all(1)
            ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    if not ok.all():
        print(f"  dropping {int((~ok).sum())} targets non-finite in either run")
    a = {k: v[ok] for k, v in a.items()}
    b = {k: v[ok] for k, v in b.items()}
    get = lambda d: (d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"])
    rng = np.random.default_rng(seed)
    flux = a["flux"]
    edges = np.percentile(flux, [0, 20, 40, 60, 80, 100])

    def row(name, sel):
        ba, bb = bias(*get(a), g, sel), bias(*get(b), g, sel)
        idx0 = np.flatnonzero(sel) if sel is not None else np.arange(len(flux))
        d = [np.subtract(bias(*get(b), g, i), bias(*get(a), g, i))
             for i in (rng.choice(idx0, len(idx0)) for _ in range(n))]
        sd = np.std(np.array(d), axis=0)
        print(f"  {name:>14s} {ba[0]:>+10.4f} {bb[0]:>+10.4f} "
              f"{bb[0] - ba[0]:>+10.4f} +/- {sd[0]:.4f}")
        return sd

    print(f"\n{'':>14s} {labels[0]:>10s} {labels[1]:>10s} "
          f"{'difference (paired)':>22s}")
    row("all", None)
    for i in range(5):
        row(f"Mf q{i + 1}", (flux >= edges[i]) & (flux <= edges[i + 1]))


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
    p.add_argument("--centroid", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="use a flow with the centroid-marginalisation layer. "
                        "Defaults to whatever the catalog's IMGNOISE header "
                        "says; --no-centroid forces it off, which measures what "
                        "the marginalisation is worth.")
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
    p.add_argument("--alpha", type=float, default=1.0,
                   help="the kernel's share of the defensive mixture proposal "
                        "(mixture_draws); 1.0, the default, is pure kernel "
                        "sampling. WHETHER IT HELPS DEPENDS ON DEPTH: at "
                        "noise_sigma 1 it measured worse (median ESS 129 -> 77), "
                        "but at 2.73 it is the opposite and the tail is what "
                        "moves -- frac(ESS<10) 30%% -> 1.3%% at alpha 0.5. Note "
                        "alpha < 1 draws part of the proposal FROM THE FLOW, so "
                        "two runs with different flows lose their common random "
                        "numbers and --compare weakens.")
    p.add_argument("--chunk", type=int, default=None,
                   help="draws evaluated at once; streams the integral in "
                        "chunks of this size and keeps only per-target running "
                        "sums, so `--samples` is not limited by memory. Defaults "
                        "to min(samples, 8192).")
    p.add_argument("--noise-seed", type=int, default=1,
                   help="the targets' noise realization; shared by the +g, -g "
                        "and unsheared catalogs so the pairing still cancels "
                        "shape noise")
    p.add_argument("--batch-budget", type=int, default=None,
                   help="targets x draws held on the device at once; lower it "
                        "if the Hessian runs out of memory")
    p.add_argument("--save-pqr", default=None,
                   help="write per-target Q and R to this .npz, for --compare")
    p.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"), default=None,
                   help="paired difference between two --save-pqr runs, and "
                        "nothing else; the only way to resolve a shift smaller "
                        "than either run's own error bar")
    p.add_argument("--n-targets", type=int, default=None,
                   help="use only the first N targets (the integration is "
                        "`samples` flow evaluations per target)")
    p.add_argument("--proposal-flow", default=None,
                   help="DANGEROUS unless it EQUALS --flow: at alpha < 1 the "
                        "weights carry P_eval/q_proposal, and wherever the "
                        "evaluated flow has mass the proposal does not cover "
                        "that ratio explodes. Measured: sharing one proposal "
                        "across two flows put 73%% of the ensemble sum|R11| on "
                        "a SINGLE target and returned m1 = -0.84 where the "
                        "self-proposed run gave +0.023, with the ESS "
                        "diagnostic and every training metric looking normal. "
                        "Use it only for the alpha=1 kernel, where the "
                        "proposal does not involve a flow at all. "
                        "Otherwise: draw the defensive mixture's flow component from THIS"
                        "flow instead of the one being evaluated, so two runs "
                        "with different --flow still share draws and --compare "
                        "keeps its pairing. Only matters when --alpha < 1.")
    a = p.parse_args()

    if a.compare:
        compare(*a.compare, a.g, a.boot, a.seed,
                labels=[s.split("/")[-1][:10] for s in a.compare])
        return

    cat = CATALOGS[a.pop]
    path = lambda v: f"{a.data_dir}/{v}.fits"
    # IMGNOISE says the noise is already in the image, put there before a real
    # recenter().  Such a catalog must not be handed to `add_noise` (which is
    # only exact at a fixed centre) and needs the centroid layer, because its
    # moments carry a centroid error the prior has to marginalise over.
    # ext=1: imsims writes its header on the table HDU, not the primary one, so
    # a bare read_header would silently report every catalog as noiseless.
    img_noise = bool(
        fitsio.read_header(path(cat["zero"]), ext=1).get("IMGNOISE", False))
    use_centroid = a.centroid if a.centroid is not None else img_noise

    train_data = a.train_data or (
        f"{a.data_dir}/moments{'_sersic' if a.pop == 'sersic' else ''}.fits")
    # Rebuild exactly as `shear.py train` did -- the flow's RawMomentStandardize
    # is fixed by the training split, so the same 90% has to go in here.  The
    # centroid flow was standardised on the copy catalog's GALAXIES table
    # instead, which is the full population; --train-data selects it.
    m_train_full = shear.load(train_data)[0]
    slice90 = lambda arr: arr[:int(0.9 * len(arr))]
    m_train = m_train_full if use_centroid else slice90(m_train_full)
    flow = bulk.build_flow(jr.key(a.seed), m_train, shear=True,
                           centroid=use_centroid)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    print(f"{a.flow} on the {a.pop} targets"
          + (" (image noise, recentred; centroid layer on)" if use_centroid
             else ""))

    proposal = None
    if a.proposal_flow:
        # Keyed on IMGNOISE, not `use_centroid`/`--no-centroid`: --no-centroid
        # selects what is being MEASURED, and must not degrade the proposal,
        # which only has to cover the posterior the CATALOG actually has.  Its
        # training slice is keyed the same way, for the same reason the eval
        # flow's is keyed on `use_centroid` above -- get this wrong and the
        # proposal's RawMomentStandardize is silently off.
        m_train_prop = m_train_full if img_noise else slice90(m_train_full)
        proposal = bulk.build_flow(jr.key(a.seed), m_train_prop, shear=True,
                                   centroid=img_noise)
        proposal = eqx.tree_deserialise_leaves(a.proposal_flow, proposal)
        print(f"  proposal flow: {a.proposal_flow} "
              f"(draws shared across eval flows)")

    if img_noise and not a.samples:
        raise SystemExit(
            "these targets carry image noise, so P(M|g) is the prior CONVOLVED "
            "with C_M -- a point evaluation of the prior at a noisy M lands in "
            "its tails and returns nonsense. Pass --samples > 0.")

    n = slice(None) if a.n_targets is None else slice(a.n_targets)
    rows = {k: fitsio.read(path(v))[n] for k, v in cat.items()}
    m = {k: np.asarray(v["moments"], dtype=np.float64) for k, v in rows.items()}
    truth = m["zero"]           # unsheared -- for binning only

    # A target whose recentring did not converge is not at a detection point, so
    # it is outside the formalism entirely.  Drop it from ALL THREE catalogs so
    # the +/- pairing still lines up galaxy for galaxy.
    keep = np.ones(len(truth), dtype=bool)
    if img_noise:
        for v in rows.values():
            keep &= ~v["badcenter"]
        if not keep.all():
            print(f"  dropping {int((~keep).sum())} targets "
                  f"({(~keep).mean():.1e}) whose recentring did not converge")
        rows = {k: v[keep] for k, v in rows.items()}
        m = {k: v[keep] for k, v in m.items()}
        truth = m["zero"]

    # Sigma_X per target: the centroid layer's condition.  Constant in these
    # sims, read per row anyway so a varying-depth catalog needs no change here.
    # Read unconditionally: the proposal flow's centroid layer is keyed on
    # IMGNOISE (see the proposal-flow build above), so it needs Sigma_X even
    # when the evaluation flow -- keyed on `use_centroid` -- does not.
    sigma_x_all = np.asarray(rows["zero"]["cov_odd"], dtype=np.float64)
    sigma_x = sigma_x_all if use_centroid else None
    proposal_sigma_x = sigma_x_all if img_noise else None
    # The three mixture_draws call sites below all draw the flow component
    # from EITHER the proposal, if one was given, or the flow itself --
    # collapse that choice once instead of three times.  `pqr_streamed` does
    # the equivalent internally, so it is passed `proposal`/`proposal_sigma_x`
    # directly rather than through these.
    draw_flow = flow if proposal is None else proposal
    draw_sigma_x = sigma_x if proposal is None else proposal_sigma_x

    draws = log_wt = None
    batch = 20000
    chunk = None
    if a.samples:
        cov = load_cov(path(cat["zero"]))
        if not img_noise:
            # The SAME noise realization for a galaxy in all three catalogs, so
            # what the +/- difference cancels stays cancelled.
            m = {k: add_noise(v, cov, a.noise_seed) for k, v in m.items()}
        # The Hessian is forward-over-reverse, so a batch costs several times
        # `batch * samples` flow activations -- keep that product bounded.  The
        # centroid layer does NOT inflate this, because it is peeled off below
        # and never enters the autodiff.
        chunk = min(a.samples, a.chunk or 8192)
        batch = max(1, (a.batch_budget or 131_072) // chunk)
        if a.samples > chunk:
            # Streamed: the draws are generated, used and discarded per chunk,
            # so nothing of size (targets x samples) is ever held.  This is the
            # only path that reaches large `samples` -- at 20000 x 32768 the draw
            # array alone would be 26 GB per catalog.
            print(f"  streaming {a.samples} draws/target in chunks of {chunk} "
                  f"({batch} targets at a time)")
        else:
            # Each catalog needs its own draws, centred on its own noisy M_i --
            # but the SAME seed for all three, so the kernel offsets and the
            # flow's defensive draws are the same common random numbers in each,
            # the same role `eps` played when it was shared directly.
            draws, log_wt = {}, {}
            for k, v in m.items():
                draws[k], log_wt[k] = mixture_draws(
                    draw_flow, v, cov, a.samples, a.alpha, a.noise_seed + 1000,
                    sigma_x=draw_sigma_x)

    # Peel the centroid layer off and apply it once.  Its log-det is a property
    # of the draw and of Sigma_X, never of g, so it belongs in the importance
    # weight rather than inside the g-autodiff -- see `split_centroid`.  The
    # proposal weights themselves stay as they are: they are ratios in the
    # ORIGINAL moment space, which is where the kernel lives.
    # `pqr_streamed` and the ESS diagnostic below build their own draws, so they
    # need the moments in the ORIGINAL space, before the peel rewrites `m`.
    m_raw = dict(m)
    flow_g, layer = split_centroid(flow)
    if layer is not None:
        if draws is not None:
            for k in draws:
                draws[k], ld = centroid_transform(layer, draws[k], sigma_x)
                log_wt[k] = log_wt[k] + ld
        # `m` itself only ever enters as the kernel's centre (already used, in
        # the original space) or, with no draws, as a point evaluation -- where
        # the log-det is an additive g-independent constant and so drops out of
        # both Q and R.  Either way it is discarded here rather than tracked.
        m = {k: centroid_transform(layer, v, sigma_x)[0] for k, v in m.items()}
        print("  centroid layer applied outside the g-autodiff "
              "(its log-det is g-independent, so Q and R are unchanged)")

    if a.samples:
        # ESS on one chunk's worth of draws, over a slice of targets -- a
        # diagnostic, so it does not need the full sample count.  Report it
        # scaled to the full S, since ESS grows linearly with draws.
        n_e = min(2000, len(m["zero"]))
        d_e, w_e = mixture_draws(draw_flow, m_raw["zero"][:n_e], cov, chunk,
                                 a.alpha, a.noise_seed + 1000, sigma_x=None
                                 if draw_sigma_x is None else draw_sigma_x[:n_e])
        if layer is not None:
            d_e, ld_e = centroid_transform(layer, d_e, sigma_x[:n_e])
            w_e = w_e + ld_e
        e = ess(flow_g, m["zero"][:n_e], d_e, w_e, 4 * batch,
                None if sigma_x is None else sigma_x[:n_e])
        scale = a.samples / chunk
        print(f"integrating under C_M with {a.samples} draws/target "
              f"(alpha = {a.alpha}): ESS = {np.median(e) * scale:.0f} (median), "
              f"{np.percentile(e, 5) * scale:.0f} (5th pct), "
              f"frac<10 = {float((e * scale < 10).mean()):.3f}")

    if a.samples and a.samples > chunk:
        qr = {}
        for k, v in m_raw.items():
            print(f"  {k}:", flush=True)
            qr[k] = pqr_streamed(flow, v, cov, a.samples, a.alpha,
                                 a.noise_seed + 1000, sigma_x,
                                 batch=batch, chunk=chunk,
                                 report=max(1, len(v) // (batch * 10)),
                                 proposal=proposal, proposal_sigma_x=proposal_sigma_x)
    else:
        qr = {k: pqr(flow_g, v, None if draws is None else draws[k],
                     None if log_wt is None else log_wt[k], batch, sigma_x)
              for k, v in m.items()}
    # A single non-finite target, or one with |R| many orders of magnitude
    # above the rest, takes out the whole eq. (45)-(46) sum -- so drop both
    # kinds from ALL catalogs together and keep the +/- pairing aligned.  The
    # magnitude cut catches flow density-curvature spikes: a handful of fully
    # in_domain targets (median |R| ~1e2) where the flow's local Hessian
    # blows up to |R| ~1e6-1e9, a normalising-flow generalisation artifact in
    # a sparsely-trained pocket, not a support-boundary effect.  Measured:
    # dropping the worst 5 of 200000 in one quintile took ghat[0] on a g=0
    # null test from +8.7e-3 to -3.6e-4.  With the `pqr_streamed` merge guard
    # non-finite entries should not arise; this is the net under it.
    finite = np.ones(len(qr["plus"][0]), dtype=bool)
    for q, r in qr.values():
        finite &= np.isfinite(q).all(1) & np.isfinite(r).reshape(len(r), -1).all(1)
    rnorm = {k: np.linalg.norm(r.reshape(len(r), -1), axis=1) for k, (q, r) in qr.items()}
    med = np.median(np.concatenate([rn[finite] for rn in rnorm.values()]))
    sane = finite.copy()
    for rn in rnorm.values():
        sane &= rn < 1000 * med
    if not finite.all():
        print(f"  dropping {int((~finite).sum())} targets with a non-finite "
              f"Q or R ({(~finite).mean():.1e})")
    if (finite & ~sane).any():
        n = int((finite & ~sane).sum())
        print(f"  dropping {n} targets with |R| > 1000x the population "
              f"median ({n / len(finite):.1e}); a flow density-curvature "
              f"spike, not a domain effect")
    if not sane.all():
        qr = {k: (q[sane], r[sane]) for k, (q, r) in qr.items()}
        truth = truth[sane]

    qp, rp = qr["plus"]
    qm, rm = qr["minus"]

    if a.save_pqr:
        save_pqr(a.save_pqr, qr, truth[:, 0])

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
