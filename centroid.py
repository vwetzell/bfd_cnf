"""
centroid.py
===========
Phase 3: fit and validate `models.centroid.CentroidMarginalize` against a
catalog of shifted template copies.  The layer is a closed-form transport (see
its module docstring) with ONE trained coefficient, the k^4 spin-2 bracket, so
`train` runs the NLL loop over that and `check` reports the shift it produces
against the catalog's own weighted copy means.  Use `--holdout` to keep a
fraction of the galaxies out of training and report `check` on those.

The validation set is not the galaxies -- it is their *copies*.
`imsims.copies` replicates every template galaxy over a grid of coordinate
origins u and records the moments m(u) and first moments X(u) of each, which is
the left-hand side of the paper's eq. (36).  The weight of a copy under a target
whose first moments have covariance Sigma_X is

    w(u) = d2u . |J(u)| . N(X(u); 0, Sigma_X)

so the marginalised prior is the w-weighted distribution of the m(u).  That is
what `check` compares `CentroidMarginalize`'s analytic transport against.

Why draw rather than weight
---------------------------
Copy weights span e^-8 across a galaxy's grid, so a flat batch of copies has an
effective sample size of ~13% of its length -- the classic reason this needed
self-normalised importance sampling before.  It does not need it here.  The
weights of ONE galaxy's copies sum to that galaxy's detection probability, and
`imsims` measures that as 1.00000 for every galaxy when sn_min = 0.  So drawing
galaxies uniformly and then one copy each in proportion to w is exact
stratification at full ESS, with no importance weights anywhere.

That equality is a property of the no-selection case, not a general one.  Turn on
a flux cut and the per-galaxy sum becomes P(s|G) < 1 and genuinely varies between
galaxies; `CopySampler` would then have to carry it as a per-galaxy weight
instead of normalising it away.  `tests/test_centroid.py::test_sampler_is_the
_weighted_distribution` and imsims' own
`test_copy_weights_sum_to_the_detection_probability` are what license it today.

Staging
-------
Sigma_X is the same for every galaxy in the current sims (fixed noise level,
circular PSF), so this stage asks only whether the layer can represent the
marginalisation at all -- the conditioning is on a constant.  `--sigma-scale`
reweights the same catalog to a different Sigma_X without regenerating it, which
is what the catalog's factor-1.4 grid margin is for, and is the cheap way to see
the layer's Sigma_X dependence before the sims grow a varying noise level and
then an elliptical PSF.

Usage:
    python centroid.py train --copies ../bfd_cnf_imsims/data/copies_bulgedisc.fits
    python centroid.py check --copies ../bfd_cnf_imsims/data/copies_bulgedisc.fits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import bulk
from models.bijections import SigmaXBlockLayer, _bound_coeff_input
from models.centroid import dm_dsigma

# The copy-weight convention lives in imsims and must not be duplicated here: it
# is the one place the |J| choice of eq. (35) vs (36) is made, and two copies of
# that would drift.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bfd_cnf_imsims"))
try:
    from imsims.copies import log_weights
except ImportError as exc:                                  # pragma: no cover
    raise SystemExit("centroid.py needs ../bfd_cnf_imsims on the path: "
                     f"{exc}") from None

MOMENT_LABELS = ["Mf", "Mr", "M1", "M2", "Mc"]


def load_copies(path):
    """Read an `imsims.copies` catalog -> (copies, galaxies)."""
    with fitsio.FITS(path) as f:
        return f["COPIES"].read(), f["GALAXIES"].read()


class CopySampler:
    """Draw one shifted copy per galaxy, in proportion to its eq.-36 weight.

    Galaxies are drawn in proportion to the SUM of their copy weights, not
    uniformly.  That sum is the galaxy's detection probability, and it is not
    always 1: on a shallow catalog it is 1.00000 for every galaxy, but at
    noise_sigma = 2.73 the faintest ~1% fall to 0.98 and the worst to 0.69,
    because their centroid distribution reaches the `det J <= 0` boundary that
    `makeTemplates` cuts on -- the paper's positive-Jacobian assumption (sec.
    5.3) failing at low S/N, not a grid that is too small (none of them come
    near `XY_MAX`).  Those galaxies really are less likely to be detected at all,
    so normalising the sum away would over-weight exactly the objects the
    formalism represents worst.

    Within a galaxy the copies are then drawn by weight, which is exact
    stratification at full ESS -- the point of doing it this way rather than
    self-normalised importance weighting over a flat batch of copies, where the
    e^-8 spread of the weights leaves an ESS of ~13% of the batch.

    Copies must be grouped by `gal`, which `imsims.copies` writes them as.
    """

    def __init__(self, copies, sigma_x):
        w = np.exp(log_weights(copies, sigma_x))
        self.m = np.ascontiguousarray(copies["moments"], dtype=np.float64)
        # Each copy's OWN exact shear derivatives, when the catalog carries them.
        # A copy is the parent measured about a shifted origin, so the parent's
        # derivatives do NOT transfer -- these have to come from the catalog, and
        # `imsims.copies` now stores them (bfd's `makeTemplates` computed them all
        # along; only slot D0 used to be kept).  Without them the layer can only
        # be trained at g = 0.
        names = copies.dtype.names
        self.dm = (np.ascontiguousarray(copies["dm_dg"], dtype=np.float64)
                   if "dm_dg" in names else None)
        self.d2m = (np.ascontiguousarray(copies["d2m_dg2"], dtype=np.float64)
                    if "d2m_dg2" in names else None)
        # One running cumulative sum serves every galaxy: within a group the CDF
        # is just cw shifted by the group's base, so a single searchsorted over
        # the whole array does the per-galaxy draw for the entire batch at once.
        self.cw = np.cumsum(w)
        gal = copies["gal"]
        # FITS hands back big-endian ints, which JAX will not accept.
        self.ids = np.unique(gal).astype(np.int64)
        self.starts = np.searchsorted(gal, self.ids, side="left")
        self.ends = np.searchsorted(gal, self.ids, side="right")
        self.base = self.cw[self.starts] - w[self.starts]
        self.total = self.cw[self.ends - 1] - self.base
        # P(s|G) as a galaxy-selection probability -- see the class docstring.
        self._gal_cdf = np.cumsum(self.total)
        self._gal_cdf /= self._gal_cdf[-1]

    @property
    def n_galaxies(self):
        return len(self.ids)

    def ess(self):
        """Effective sample size per galaxy -- the paper's own sec.-2.5 diagnostic."""
        w = np.diff(np.concatenate([[0.0], self.cw]))
        num = np.add.reduceat(w, self.starts) ** 2
        den = np.add.reduceat(w * w, self.starts)
        return num / den

    def draw(self, rng, batch):
        """`batch` galaxies drawn by detection probability, one copy each by weight.

        Returns (copy moments, galaxy row indices, dm/dg, d2m/dg2) -- the second
        so the caller can line the batch up with per-galaxy quantities, and the
        last two so it can LENS the drawn copies.  The derivatives are None on a
        catalog rendered before `imsims.copies` began storing them.
        """
        gi = np.searchsorted(self._gal_cdf, rng.random(batch))
        gi = np.clip(gi, 0, len(self.ids) - 1)
        target = self.base[gi] + rng.random(batch) * self.total[gi]
        idx = np.searchsorted(self.cw, target, side="left")
        # Guard the group edges against float error in the cumulative sum.
        idx = np.clip(idx, self.starts[gi], self.ends[gi] - 1)
        return (self.m[idx], self.ids[gi],
                None if self.dm is None else self.dm[idx],
                None if self.d2m is None else self.d2m[idx])

    @property
    def has_derivatives(self):
        return self.dm is not None and self.d2m is not None


def _centroid_layer(flow):
    """The CentroidMarginalize inside the built flow.

    Located BY TYPE: the chart is bijections[0] now, so an index here would
    return the chart and train it instead of this layer.
    """
    from models.centroid import CentroidMarginalize
    for b in flow.bijection.bijection.bijections:
        if isinstance(b, CentroidMarginalize):
            return b
    raise ValueError("no CentroidMarginalize in this flow")


def _chart(flow):
    """The RawMomentStandardize, which `dm_dsigma` needs to convert its z-space
    shift back to the raw moments the copy catalog is measured in."""
    return flow.bijection.bijection.bijections[0]


def _trainable(flow):
    """Filter spec selecting only the centroid layer.

    Same reasoning as `shear._trainable`: the Sigma_X dependence is a small part
    of the total likelihood, so a free bulk will absorb batch noise and drown it.
    """
    spec = jax.tree.map(lambda _: False, flow)
    return eqx.tree_at(
        lambda f: _centroid_layer(f), spec,
        replace=jax.tree.map(eqx.is_inexact_array, _centroid_layer(flow)))


def condition(sigma_x, batch, g=(0.0, 0.0)):
    """The chain's condition vector [g1, g2, C00, C01, C11], tiled over a batch.

    Training used to run only at g = 0, because the copies were unlensed and
    there was nothing to lens them with.  There is now: `imsims.copies` stores
    each copy's own exact dm/dg and d2m/dg2, so `train` samples g over the shear
    disc and lenses the drawn copies by their own derivatives, exactly as
    `shear.train` does with templates.  That is what gives the layer's
    g-conditioned coefficients (`models/centroid.g_invariants`) a gradient --
    at g = 0 their inputs are identically zero and they never move.
    """
    row = jnp.concatenate([jnp.asarray(g, dtype=jnp.float32),
                           jnp.asarray(sigma_x, dtype=jnp.float32)])
    return jnp.tile(row, (batch, 1))


def channels(m0, d):
    """The two responses one k^4 spin-2 coefficient controls, scale-free.

    `d` is a raw-moment shift (N, 5).  Returns `dMr/Mr` and the spin-2 response
    projected on each galaxy's OWN ellipticity -- the same projection
    `check`'s `ellipticity response` line sums, taken per galaxy.  Projecting
    is what makes the spin-2 target usable: M1 and M2 average to ~0 over an
    isotropic population, so an unprojected residual is dominated by a
    direction that carries no population-level signal.

    Works on numpy or jax arrays.
    """
    Mr = m0[:, 1]
    e1, e2 = m0[:, 2] / Mr, m0[:, 3] / Mr
    dr = d[:, 1] / Mr
    return dr, (d[:, 2] / Mr - e1 * dr) * e1 + (d[:, 3] / Mr - e2 * dr) * e2


def relative_channels(m0, d):
    """`channels` extended with the two channels it left out: `dMf/Mf` and
    `dMc/Mc` -- so `train_sigmax`'s loss is entirely scale-free and no single
    bright galaxy's contribution to a batch outscales the rest by magnitude
    alone (the mechanism behind the near-inert `net_size`/`net_dipquad` a
    plain-NLL/absolute-moment loss produced -- see `train_sigmax`'s
    docstring). `dMc/Mc` matches `check`'s own scale convention (and
    `shear._scale`'s), not `_transport`'s ansatz-relative scale -- that one is
    internal to the OLD layer's bracket, not a convention any loss uses.

    Works on numpy or jax arrays, like `channels`.
    """
    dr, pr = channels(m0, d)
    return d[:, 0] / m0[:, 0], dr, d[:, 4] / m0[:, 4], pr


def train(flow, m0, truth, sigma_x, key, steps=3000, batch=8192, lr=1e-2):
    """Fit the k^4 spin-2 coefficient against the catalog's copy-weighted shifts.

    NOT the likelihood.  The copy NLL cannot see this coefficient at all:
    scanning a constant `c` from -1 to 3 moves the spin-2 response ratio from
    +6.6 to -5.5 and moves the NLL of 65536 copies by 0.7 nats with no
    structure -- less than the batch-to-batch noise of the loop this replaces,
    which duly wandered for 16000 steps and left the layer at a response ratio
    of -4.4 (`dev/c_scan.py`).  That is not a tuning failure: the whole
    marginalisation changes |e| by ~0.5%, so it is a tiny perturbation of a
    density and an enormous one of a shear response, and only an estimator
    aimed at the response can resolve it.

    So the target is `imsims.copies`' own weighted copy means -- the exact
    marginalisation of this population's own templates, which is ground truth
    for the map, not a proxy for it.  Least squares against a noisy per-galaxy
    target estimates its CONDITIONAL MEAN, which is precisely what a
    deterministic transport can carry (the `Var[.|m]` floor is the residual,
    not a bias).

    Both channels the coefficient touches are fitted, each normalised by its
    own catalog spread so neither's units decide the weighting.  They pull
    different ways -- see HANDOFF -- and letting one net see both is the point:
    a constant cannot satisfy both, a function of the invariants might.
    """
    tr_dr, tr_pr = channels(m0, truth)
    s_r, s_p = float(tr_dr.std()), float(tr_pr.std())
    print(f"targets: dMr/Mr rms {s_r:.3e}, spin-2 projection rms {s_p:.3e}")

    m0j = jnp.asarray(m0, dtype=jnp.float32)
    drj = jnp.asarray(tr_dr, dtype=jnp.float32)
    prj = jnp.asarray(tr_pr, dtype=jnp.float32)
    sx = jnp.asarray(sigma_x, dtype=jnp.float32)

    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(optax.cosine_decay_schedule(lr, steps)))
    params, static = eqx.partition(flow, _trainable(flow))
    state = opt.init(params)
    rng = np.random.default_rng(int(jr.randint(key, (), 0, 2**30)))

    @eqx.filter_jit
    def step(params, state, idx):
        def loss_fn(p):
            model = eqx.combine(p, static)
            mb = m0j[idx]
            pred = jax.vmap(dm_dsigma, in_axes=(None, 0, None, None))(
                _centroid_layer(model), mb, sx, _chart(model))
            dr, pr = channels(mb, pred)
            size = jnp.mean(((dr - drj[idx]) / s_r) ** 2)
            spin2 = jnp.mean(((pr - prj[idx]) / s_p) ** 2)
            return size + spin2, (size, spin2)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = opt.update(grads, state, params)
        return eqx.apply_updates(params, updates), state, aux

    for i in range(steps):
        idx = jnp.asarray(rng.integers(0, len(m0), batch))
        params, state, (size, spin2) = step(params, state, idx)
        if i % 250 == 0 or i == steps - 1:
            print(f"step {i:5d}  size {float(size):.5f}  spin2 {float(spin2):.5f}")
    flow = eqx.combine(params, static)
    L = _centroid_layer(flow)
    c = np.asarray(jax.vmap(L.bracket_coeffs, in_axes=(0, None, None))(
        m0j[:20000], *L.chart()))
    for j, lab in enumerate(("c_spin2", "c_spin4")):
        print(f"fitted {lab}: p5 {np.percentile(c[:, j], 5):.3f}  p50 "
              f"{np.median(c[:, j]):.3f}  p95 {np.percentile(c[:, j], 95):.3f}")
    return flow


def weighted_copy_mean(copies, galaxies, sigma_x):
    """Per-galaxy w-weighted mean of the copy moments -- the layer's target.

    This is what the marginalisation does to each galaxy, straight from the
    catalog and with no flow involved.
    """
    w = np.exp(log_weights(copies, sigma_x))
    n = len(galaxies)
    den = np.bincount(copies["gal"], weights=w, minlength=n)
    num = np.stack([np.bincount(copies["gal"], weights=w * copies["moments"][:, j],
                                minlength=n) for j in range(5)], axis=1)
    keep = den > 0
    return num[keep] / den[keep, None], keep


def check(flow, copies, galaxies, sigma_x, n=4000, skip=0):
    """Compare the layer's own shift with the catalog's weighted copy means.

    Diagnostic only, exactly as `shear.check` is: the layer is a transport that
    matches DENSITIES, so it is not obliged to reproduce any single galaxy's
    marginalisation -- only the conditional mean of it, which is what the columns
    below measure.  The scatter that no deterministic transport can carry is the
    `Var[.|m]` floor of the module docstring.
    """
    target, keep = weighted_copy_mean(copies, galaxies, sigma_x)
    m0 = galaxies["moments"][keep][skip:skip + n]
    target = target[skip:skip + n]
    layer, chart = _centroid_layer(flow), _chart(flow)
    sx = jnp.asarray(sigma_x, dtype=jnp.float32)
    pred = np.asarray(jax.vmap(dm_dsigma, in_axes=(None, 0, None, None))(
        layer, jnp.asarray(m0, dtype=jnp.float32), sx, chart))
    truth = target - m0

    # Normalise by each moment's own magnitude; the spin-2 pair by Mr, since M1
    # and M2 pass through zero (same convention as shear._scale).
    scale = np.stack([m0[:, 0], m0[:, 1], m0[:, 1], m0[:, 1], m0[:, 4]], axis=1)
    print(f"  {'':4s} {'layer shift':>13s} {'catalog shift':>14s} {'resid':>10s}")
    for j, lab in enumerate(MOMENT_LABELS):
        p, t = pred[:, j] / scale[:, j], truth[:, j] / scale[:, j]
        print(f"  {lab:4s} {p.mean():+13.3e} {t.mean():+14.3e} "
              f"{np.sqrt(np.mean((p - t) ** 2)):10.3e}")

    # The number that decides whether this was worth doing: how much of the
    # ellipticity amplification the layer reproduces.
    def e(m):
        return (m[:, 2] + 1j * m[:, 3]) / m[:, 1]

    e0 = e(m0)
    resp = lambda mm: ((e(mm) * np.conj(e0)).real.sum()
                       / (e0 * np.conj(e0)).real.sum() - 1.0)
    rc, rl = resp(target), resp(m0 + pred)
    print(f"  ellipticity response  catalog {rc:+.4e}   layer {rl:+.4e}"
          f"   ratio {rl / rc:.4f}")


# ---------------------------------------------------------------------------
# SigmaXBlockLayer training (NLL, not the CentroidMarginalize regression above)
# ---------------------------------------------------------------------------
#
# SigmaXBlockLayer has five independent coefficient nets, trained by
# `train_sigmax` -- see its docstring for what the loss is and why (a
# flux-relative regression against `weighted_copy_mean`'s own eq.-36 target,
# not NLL: an NLL/log_prob-based attempt was tried first and measured to fail
# for reasons that motivated this replacement).
REPORT = 500

# `--anisotropic`'s training grid: (|e|, position angle in radians).  Two
# DIFFERENT angles are required -- a single angle would let the nets learn a
# direction-specific shortcut (e.g. "M1 only") that happens to work there and
# nowhere else, rather than the general e1/e2-covariant dependence Sigma_X's
# own ellipticity actually needs.  Magnitudes match the class docstring's
# calibration bound (`e_max=0.1` in `bulk.build_flow`'s SigmaXBlockLayer
# construction) with margin on both sides.
ANISO_TRAIN_POINTS = ((0.05, 0.0), (0.07, np.pi / 4))


def _aniso_sigma_x(sigma_x, e_mag, theta):
    """An anisotropic Sigma_X with the SAME trace as isotropic `sigma_x`
    (`[C00, C01, C11]`) but ellipticity `(e1, e2) = e_mag*(cos 2theta, sin
    2theta)`, in the identical `log_scale/e1/e2` parameterisation
    `SigmaXBlockLayer._unpack` -> `cx_to_sx_cond` uses internally
    (`models.bijections._sx_cond_to_CX`'s inverse convention) -- so a test
    point built here lands at exactly the `T_n`/anisotropy the layer itself
    would compute from it, not merely SOME anisotropic covariance.
    """
    C00, C01, C11 = (float(v) for v in sigma_x)
    log_scale = 0.5 * float(np.linalg.slogdet(np.array(
        [[C00, C01], [C01, C11]]))[1])
    e1, e2 = e_mag * np.cos(2.0 * theta), e_mag * np.sin(2.0 * theta)
    half_T = np.exp(log_scale) / np.sqrt(max(1.0 - e_mag**2, 1e-8))
    return np.array([half_T * (1.0 + e1), half_T * e2, half_T * (1.0 - e1)])

def _sigmax_layer(flow):
    """The SigmaXBlockLayer inside the built flow.  Located BY TYPE, same
    reasoning as `_centroid_layer`."""
    for b in flow.bijection.bijection.bijections:
        if isinstance(b, SigmaXBlockLayer):
            return b
    raise ValueError("no SigmaXBlockLayer in this flow")


def _sigmax_trainable(flow):
    """Filter spec selecting only the SigmaXBlockLayer's five coefficient nets.

    Same mechanism as `shear._trainable`/`_trainable` above: freezing bulk AND
    shear matters for the same reason shear's docstring gives for freezing the
    bulk -- a free earlier layer will happily absorb batch noise and drown the
    small conditional signal this layer is supposed to carry.
    """
    spec = jax.tree.map(lambda _: False, flow)
    return eqx.tree_at(
        lambda f: _sigmax_layer(f), spec,
        replace=jax.tree.map(eqx.is_inexact_array, _sigmax_layer(flow)))


def sigmax_dm_dsigma(layer, m, full_cond, chart):
    """`SigmaXBlockLayer`'s generative shift of `m` at this condition, in RAW
    moments -- the `SigmaXBlockLayer` analogue of `models.centroid.dm_dsigma`.

    `dm_dsigma` calls `CentroidMarginalize.marginalize` directly; this layer
    has no such method (only the `AbstractBijection` transform/inverse pair),
    and its condition is the full `[g1, g2, C00, C01, C11]` vector rather than
    a bare Sigma_X, so it needs its own version rather than reusing that one.
    `inverse_and_log_det` is the base -> data step in this chain (see
    `SigmaXBlockLayer`'s docstring: `transform_and_log_det` composes
    `_raw_transform`, which is the data -> base direction the chain runs
    top-down), i.e. the "marginalize" side -- the one `dm_dsigma` also uses.
    """
    z = chart.transform(m)
    z2 = layer.inverse_and_log_det(z, full_cond)[0]
    return chart.inverse(z2) - m


def train_sigmax(flow, m0, truth, sigma_x, key, steps=6000, batch=8192, lr=3e-3,
                 extra_scales=None):
    """Fit `SigmaXBlockLayer`'s five coefficient nets by SUPERVISED regression
    against `imsims.copies`' own weighted copy mean, directly in the CHART's
    z-space -- not plain NLL, and not `relative_channels`' raw-moment ratios
    either (both tried first, both superseded; see below).

    `extra_scales`: optional list of `(sigma_x_i, truth_i)` pairs (same `m0`,
    a DIFFERENT Sigma_X and its own `weighted_copy_mean` target) to train
    across alongside the base `(sigma_x, truth)`, one drawn uniformly per
    batch.  Added because a layer trained at a single Sigma_X was measured to
    NOT generalise across the catalog's own documented valid `sigma_scale`
    margin ([1/1.4, 1.4], `main`'s `--sigma-scale` help text): the ellipticity
    -response ratio went 0.79 (trained scale) -> 1.63 (0.8x) -> -0.09 (1.2x),
    i.e. the analytic `T_n` prefactor alone does not carry the nets across
    scales they never saw.  Cheap to fix because `weighted_copy_mean` is
    exact and instant (~3s/scale) -- no new simulation, just a few more
    calls to it at different Sigma_X, done ONCE up front by the caller
    (`main`), not resampled per step.

    First attempt was plain NLL of the frozen bulk+shear's `flow.log_prob` on
    lensed copy draws.  Measured result: NLL flat at ~80 nats for 6000 steps,
    `net_size`/`net_dipquad` near-inert, the ellipticity-response ratio -0.04
    against a target of ~1.  Root cause: the loss scored ABSOLUTE-moment
    log-density, so a rare bright galaxy in the population's own flux tail
    produced a gradient orders of magnitude larger than a typical galaxy's,
    and `optax.clip_by_global_norm` crushed everyone else's signal to match
    it, batch to batch, until the net accumulated signal averaged toward ~0.

    Second attempt regressed `relative_channels` (dMf/Mf, dMr/Mr, dMc/Mc,
    spin-2 projected on the galaxy's own e) against the raw-moment target
    shift.  That fixed the flux-outlier problem but reintroduced a SUBTLER
    one: the chart's `_inverse_transform` is MULTIPLICATIVE (`Mc = z2*Mr`,
    `M1 = z3*Mr`, `M2 = z4*Mr`, `Mr = z1*Mf`), so e.g.
    `dMc/Mc ~= (kappa-1) + mc_shift/z2` -- `net_size`'s `kappa` leaks into the
    "Mc" loss term even though `net_mc`'s own z-space correction is genuinely
    independent, coupling nets that should be free to move separately.

    Fix: skip the chart's raw-moment ratios entirely and compare the z-space
    shift the layer actually produces (`dz_pred = layer.inverse_and_log_det
    (z, cond)[0] - z`) against the chart applied to the catalog's own target
    (`dz_true = chart.transform(target) - chart.transform(m0)`), one MSE term
    per z-slot, each normalised by that slot's own std across the training
    population.  Each z-slot is already an appropriately-scaled, roughly-O(1)
    coordinate by the chart's own construction, so this needs no ratios and
    no `channels()` projection -- `relative_channels`/`check_sigmax` stay as
    reporting-only diagnostics in human-readable (raw-moment) units, but the
    training loss itself is in z-space.

    No lensing/g-sampling either: `weighted_copy_mean` is the UNLENSED g=0
    marginalisation, and `SigmaXBlockLayer` has no g-dependence at all
    (`_unpack` never reads `condition[:2]`).
    """
    chart = _chart(flow)
    tf = lambda a: np.asarray(jax.vmap(chart.transform)(jnp.asarray(a, dtype=jnp.float32)))
    z = tf(m0)
    grid = [(sigma_x, truth)] + list(extra_scales or [])
    dz_true_grid = [tf(m0 + t) - z for _, t in grid]
    # One shared normalisation (mean over scales), so a step at any grid
    # scale is on the same loss footing as the base-scale run this was
    # tuned against.
    scales = np.mean([np.std(dz, axis=0) for dz in dz_true_grid], axis=0)
    print("targets (z-space dz): " +
          "  ".join(f"z{i} rms {scales[i]:.3e}" for i in range(5)) +
          (f"  ({len(grid)} Sigma_X scales)" if len(grid) > 1 else ""))

    zj = jnp.asarray(z, dtype=jnp.float32)
    dz_true_j = jnp.stack([jnp.asarray(dz, dtype=jnp.float32) for dz in dz_true_grid])
    scales_j = jnp.asarray(scales, dtype=jnp.float32)
    cond_grid_j = jnp.stack([jnp.concatenate([jnp.zeros(2), jnp.asarray(sx, dtype=jnp.float32)])
                             for sx, _ in grid])

    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(optax.cosine_decay_schedule(lr, steps)))
    params, static = eqx.partition(flow, _sigmax_trainable(flow))
    state = opt.init(params)
    rng = np.random.default_rng(int(jr.randint(key, (), 0, 2**30)))
    n_grid = len(grid)

    @eqx.filter_jit
    def step(params, state, idx, g):
        def loss_fn(p):
            model = eqx.combine(p, static)
            layer = _sigmax_layer(model)
            zb = zj[idx]
            cond = jnp.tile(cond_grid_j[g], (zb.shape[0], 1))
            z_pred = jax.vmap(lambda zz, cc: layer.inverse_and_log_det(zz, cc)[0])(zb, cond)
            dz_pred = z_pred - zb
            terms = jnp.mean(((dz_pred - dz_true_j[g, idx]) / scales_j) ** 2, axis=0)
            return jnp.sum(terms), terms

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = opt.update(grads, state, params)
        return eqx.apply_updates(params, updates), state, loss, aux

    for i in range(steps):
        idx = jnp.asarray(rng.integers(0, len(m0), batch))
        g = int(rng.integers(0, n_grid))
        params, state, loss, aux = step(params, state, idx, g)
        if i % REPORT == 0 or i == steps - 1:
            a = [float(x) for x in aux]
            print(f"step {i:6d}  scale#{g}  loss {float(loss):.5f}  " +
                  "  ".join(f"z{j} {a[j]:.5f}" for j in range(5)), flush=True)
    return eqx.combine(params, static)


def check_sigmax(flow, copies, galaxies, sigma_x, n=4000, skip=0):
    """`SigmaXBlockLayer` analogue of `check`: layer shift vs. catalog copy
    mean, both at g=0 -- the catalog's weighted copy means are unlensed."""
    target, keep = weighted_copy_mean(copies, galaxies, sigma_x)
    m0 = galaxies["moments"][keep][skip:skip + n]
    target = target[skip:skip + n]
    layer, chart = _sigmax_layer(flow), _chart(flow)
    full_cond = jnp.tile(
        jnp.concatenate([jnp.zeros(2), jnp.asarray(sigma_x, dtype=jnp.float32)]),
        (len(m0), 1))
    pred = np.asarray(jax.vmap(sigmax_dm_dsigma, in_axes=(None, 0, 0, None))(
        layer, jnp.asarray(m0, dtype=jnp.float32), full_cond, chart))
    truth = target - m0

    scale = np.stack([m0[:, 0], m0[:, 1], m0[:, 1], m0[:, 1], m0[:, 4]], axis=1)
    print(f"  {'':4s} {'layer shift':>13s} {'catalog shift':>14s} {'resid':>10s}")
    for j, lab in enumerate(MOMENT_LABELS):
        p, t = pred[:, j] / scale[:, j], truth[:, j] / scale[:, j]
        print(f"  {lab:4s} {p.mean():+13.3e} {t.mean():+14.3e} "
              f"{np.sqrt(np.mean((p - t) ** 2)):10.3e}")

    def e(m):
        return (m[:, 2] + 1j * m[:, 3]) / m[:, 1]

    e0 = e(m0)
    resp = lambda mm: ((e(mm) * np.conj(e0)).real.sum()
                       / (e0 * np.conj(e0)).real.sum() - 1.0)
    rc, rl = resp(target), resp(m0 + pred)
    print(f"  ellipticity response  catalog {rc:+.4e}   layer {rl:+.4e}"
          f"   ratio {rl / rc:.4f}")


def net_saturation(flow, m0, sigma_x, n=20000):
    """Fraction of the tanh-bounded outputs (`g_s`, `s0`'s own-e correction)
    sitting near their bound, vs. a small/sane fraction of their range -- the
    task's own convergence check (2b).  Both bounds are `SigmaXBlockLayer`'s
    `g_s_max`/`s0_e_max`; if the layer saturates them the bound is too tight
    and is biasing the fit, not merely tracking a genuinely large response.

    `layer` operates on the CHART's z-coordinates (`x0,x1,x2,x3,x4` =
    `log10 Mf, Mr/Mf, Mc/Mr, M1/Mr, M2/Mr` -- see `SigmaXBlockLayer`'s
    docstring), not raw moments, so `m0` (raw) must go through `chart.
    transform` first -- this used to feed raw moments straight into
    `_ellipticity`/`_s0`, which was silently wrong on both counts (wrong
    coordinate space, and, before the slot fix, the wrong ellipticity
    indices too).
    """
    layer, chart = _sigmax_layer(flow), _chart(flow)
    m_raw = jnp.asarray(m0[:n], dtype=jnp.float32)
    full_cond = jnp.tile(
        jnp.concatenate([jnp.zeros(2), jnp.asarray(sigma_x, dtype=jnp.float32)]),
        (m_raw.shape[0], 1))

    def one(row, cond):
        log_scale_n, e1, e2, e_mag_sq, e_mag_sq_n, T_n = layer._unpack(cond)
        x0, x1, x2, x3, x4 = chart.transform(row)
        y3, y4, kappa, _, _ = layer._ellipticity(
            x0, x1, x3, x4, e1, e2, log_scale_n, e_mag_sq_n, T_n)
        s0 = layer._s0(x0, y3, y4, log_scale_n, e_mag_sq_n, T_n)
        s0_base = T_n * layer.net_flux(
            jnp.array([_bound_coeff_input(x0), log_scale_n, e_mag_sq_n]))[0]
        # These ARE the tanh outputs (g_s/g_s_max, (s0-s0_base)/s0_e_max), each
        # in [-1, 1] by construction -- "saturated" means sitting near +-1.
        return jnp.stack([jnp.log(kappa) / layer._g_s_max,
                          (s0 - s0_base) / layer._s0_e_max])

    out = np.asarray(jax.vmap(one)(m_raw, full_cond))
    for name, col in [("g_s (net_size)", out[:, 0]),
                      ("s0 own-e (net_flux_e)", out[:, 1])]:
        frac95 = float(np.mean(np.abs(col) > 0.95))
        print(f"  {name:24s} |tanh output| p50 {np.median(np.abs(col)):.3f}  "
              f"p99 {np.percentile(np.abs(col), 99):.3f}  "
              f"frac > 0.95 (saturated) {frac95:.3%}")


def _flux_sas(s):
    """Parse "mu,sig,a,b" for `--flux-sas`; see `models.bijections.sas`."""
    v = tuple(float(x) for x in s.split(","))
    if len(v) != 4:
        raise argparse.ArgumentTypeError("--flux-sas needs mu,sig,a,b")
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "check", "train-sigmax", "check-sigmax"],
                   help="train/check fit CentroidMarginalize's k^4 bracket by "
                        "regression (see the module docstring); train-sigmax/"
                        "check-sigmax fit SigmaXBlockLayer's five coefficient "
                        "nets by a flux-relative regression instead (see "
                        "`train_sigmax`'s docstring).")
    p.add_argument("--copies", default="../bfd_cnf_imsims/data/copies_bulgedisc.fits")
    p.add_argument("--flow", default="flows/centroid.eqx")
    p.add_argument("--init", default="flows/shear.eqx",
                   help="shear checkpoint to warm-start bulk+shear from")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=None,
                   help="train-sigmax only; defaults to 3e-3 if unset.")
    p.add_argument("--sigma-scale", type=float, default=1.0,
                   help="rescale Sigma_X by this factor squared; the catalog "
                        "grid is valid over roughly [1/1.4, 1.4]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flux-sas", type=_flux_sas, default=None,
                   help="fitted sinh-arcsinh warp of the flux axis as \"mu,sig,a,b\"; omit for the plain log10 this chart has always used. Gaussianises log10 Mf (skew 1.78 -> 0 on bulgedisc_v2). MUST match across bulk/shear/centroid/bias or the charts disagree.")
    p.add_argument("--holdout", type=float, default=0.0,
                   help="fraction of the galaxies to keep OUT of training and "
                        "report `check` on. The split is WITHIN one population "
                        "-- never across populations, which is the thing this "
                        "method must never do.")
    p.add_argument("--multi-scale", action="store_true",
                   help="train-sigmax only: also train against Sigma_X * "
                        "(1/1.4)^2 and Sigma_X * 1.4^2 (the catalog's own "
                        "documented valid margin, see --sigma-scale's help), "
                        "not just the base --sigma-scale value. A layer "
                        "trained at one Sigma_X was measured to NOT "
                        "generalise across this margin on its own -- "
                        "ellipticity-response ratio 0.79 -> 1.63 -> -0.09 "
                        "at scale 1.0 -> 0.8 -> 1.2 -- so this is not "
                        "optional if the layer needs to see varying noise.")
    p.add_argument("--anisotropic", action="store_true",
                   help="train-sigmax only: also train against a few "
                        "ANISOTROPIC Sigma_X points (ANISO_TRAIN_POINTS), "
                        "same trace as --sigma-scale's base but with |e| in "
                        "[0.03, 0.08] at two different position angles. "
                        "Training so far has always been at e1=e2=0 exactly, "
                        "so the layer never saw Sigma_X's own anisotropy; "
                        "see SigmaXBlockLayer's net_flux/net_flux_e docstring "
                        "and train_sigmax's `extra_scales` mechanism, which "
                        "this generalises the identical way --multi-scale "
                        "does for pure noise-level changes.")
    a = p.parse_args()
    sigmax = a.mode.endswith("-sigmax")

    copies, galaxies = load_copies(a.copies)
    sigma_x = galaxies["cov_odd"][0] * a.sigma_scale**2
    # Held-out template split, WITHIN the population -- never across
    # populations, which is the one thing this method must never do.
    n_train = int(round((1.0 - a.holdout) * len(galaxies)))
    print(f"{len(galaxies)} galaxies, {len(copies)} copies, "
          f"Sigma_X = {np.round(sigma_x, 1)}")

    # Standardise the bulk against the unmarginalised galaxy moments, which is
    # the population the frozen bulk was trained on.
    m_train = np.asarray(galaxies["moments"], dtype=np.float64)
    flow = bulk.build_flow(jr.key(a.seed), m_train, shear=True, centroid=True,
                           flux_sas=a.flux_sas)

    if a.mode in ("train", "train-sigmax"):
        if a.init:
            prior = eqx.tree_deserialise_leaves(
                a.init, bulk.build_flow(jr.key(a.seed), m_train, shear=True,
                                        flux_sas=a.flux_sas))
            pb = prior.bijection.bijection.bijections
            # prior is [raw2standard, shear, *bulk]; this chain inserts the
            # centroid layer after the chart, so it is
            # [raw2standard, centroid, shear, *bulk].  Graft the chart and the
            # bulk wholesale, and the shear layer's PARAMETERS rather than the
            # layer itself -- the prior's copy carries cond_shape (2,) and this
            # chain needs (5,).
            flow = eqx.tree_at(lambda f: f.bijection.bijection.bijections[0],
                               flow, pb[0])
            flow = eqx.tree_at(lambda f: f.bijection.bijection.bijections[3:],
                               flow, pb[2:])
            flow = eqx.tree_at(lambda f: f.bijection.bijection.bijections[2].coeffs,
                               flow, pb[1].coeffs)
            # Both grafted layers' frozen chart copies must match the chart
            # they now sit behind -- `bulk.train` moves the chart's mean/std,
            # so what `build_flow` copied from the raw sample is stale.  This
            # used to fix only the centroid layer; leaving the SHEAR layer
            # stale was the whole of its dm/dg error peak at Mr/Mf ~ 3.1.
            # It also re-points the centroid layer's own frozen COPY of this
            # shear layer (`models.centroid.shear_delta_invariants`) at the
            # just-grafted one -- otherwise it would go on delensing with the
            # PRE-graft (freshly initialised) shear coefficients instead of
            # the warm-started ones the chain itself now uses.  Passing `m_train`
            # also resyncs the shear layer's coefficient-net whitening stats
            # (`coeffs.u_mean/.u_white`), stale by the same mechanism.
            flow = bulk.sync_chart_constants(flow, m_train=m_train)
            print(f"warm started bulk + shear from {a.init}")
        target, keep = weighted_copy_mean(copies, galaxies, sigma_x)
        m0 = np.asarray(galaxies["moments"][keep], dtype=np.float64)
        print(f"fitting on galaxies [0, {n_train}) of {len(m0)}")
        if sigmax:
            extra_scales = None
            if a.multi_scale or a.anisotropic:
                extra_scales = []

            def _add_point(sx_, label):
                tgt_, keep_ = weighted_copy_mean(copies, galaxies, sx_)
                # `keep` (den > 0, i.e. detected) should be scale-invariant
                # in the no-selection sims this trains on (every galaxy's
                # copy weights sum to detection prob 1.00000, see
                # `CopySampler`'s docstring) -- assert rather than risk a
                # silent row misalignment against the base scale's `m0`.
                assert np.array_equal(keep_, keep), \
                    f"keep mask differs at {label} -- rows would misalign"
                m0_ = np.asarray(galaxies["moments"][keep_], dtype=np.float64)
                extra_scales.append((sx_, (tgt_ - m0_)[:n_train]))

            if a.multi_scale:
                for s_ in (1.0 / 1.4, 1.4):
                    _add_point(galaxies["cov_odd"][0] * s_**2, f"scale {s_}")
                print(f"training across {1 + len(extra_scales)} Sigma_X scales "
                      f"(base + {[round(s_, 3) for s_ in (1/1.4, 1.4)]})")
            if a.anisotropic:
                n_before = len(extra_scales)
                for e_mag, theta in ANISO_TRAIN_POINTS:
                    _add_point(_aniso_sigma_x(sigma_x, e_mag, theta),
                              f"e={e_mag}, theta={theta:.3f}")
                print(f"training across {len(extra_scales) - n_before} "
                      f"anisotropic Sigma_X points {ANISO_TRAIN_POINTS}")
            flow = train_sigmax(flow, m0[:n_train], (target - m0)[:n_train],
                                sigma_x, jr.key(a.seed + 1),
                                steps=a.steps, batch=a.batch,
                                lr=a.lr if a.lr is not None else 3e-3,
                                extra_scales=extra_scales)
        else:
            flow = train(flow, m0[:n_train], (target - m0)[:n_train], sigma_x,
                        jr.key(a.seed + 1), steps=a.steps, batch=a.batch)
        eqx.tree_serialise_leaves(a.flow, flow)
        print(f"wrote {a.flow}")
    else:
        flow = eqx.tree_deserialise_leaves(a.flow, flow)
    if a.holdout:
        print(f"\ntrained on galaxies [0, {n_train}); "
              f"CHECK below is the HELD-OUT [{n_train}, {len(galaxies)})")
    if sigmax:
        m_check = np.asarray(galaxies["moments"], dtype=np.float64)
        net_saturation(flow, m_check[n_train if a.holdout else 0:], sigma_x)
        check_sigmax(flow, copies, galaxies, sigma_x,
                    skip=n_train if a.holdout else 0)
    else:
        check(flow, copies, galaxies, sigma_x, skip=n_train if a.holdout else 0)


if __name__ == "__main__":
    main()
