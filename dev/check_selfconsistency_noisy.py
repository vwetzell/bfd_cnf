"""Is the noisy size-profile structure real, or an artefact of the binning?

The noisy m1(Mr/Mf) profile is binned on the TRUE Mr/Mf, which is a HIDDEN
variable once the target is noisy: the estimator only ever sees M = m + noise.
BFD's prior is correct for the population; a subsample selected on something
the estimator cannot see has a DIFFERENT moment density, so the population
prior is the wrong prior for it and per-bin m1 is biased even when the density
is perfect.  (This is a different objection from the one
`dev/check_bias_by_selection.py` handles -- that one is about cutting on a
SHEARED quantity, and is avoided by cutting on the g=0 catalog.  Binning on the
g=0 catalog does not help here: the issue is the estimator's blindness to m,
not the arms disagreeing.)

The suspicion is concrete: gauss2 shows a +0.019 -> -0.064 profile spread while
its population m1 is +0.001, which is what a pure binning artefact would look
like.

The control makes the prior EXACTLY right by construction: draw the targets
FROM THE FLOW at +/-g, so the flow is its own truth, then run the identical
noisy pipeline and the identical binning.  Pairing comes from sharing the base
draw z between the two conditions, which is the flow's own notion of "the same
galaxy transported to +/-g".

    m1 ~ 0 in every bin  -> the binning is sound and the hump is real
    a hump appears       -> the hump is the method, not the flow

    python dev/check_selfconsistency_noisy.py [-n 20000] [--samples 8192]
"""
import argparse
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402
from models.bijections import in_domain              # noqa: E402

NBOOT = 300
DATA = "../bfd_cnf_imsims/data"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear.eqx")
    p.add_argument("--centroid", action="store_true",
                   help="mirror bias.py's image-noise path: build the flow with "
                        "the centroid layer, draw and evaluate at the targets' "
                        "Sigma_X (read from --cov-from, as bias.py reads it from "
                        "the catalog). The control's model is then exactly what "
                        "that path assumes -- centroid-marginalised moments plus "
                        "INDEPENDENT C_M noise -- so it measures method bias "
                        "under that model, not whether the model is right.")
    p.add_argument("--train-data", default=f"{DATA}/moments.fits")
    p.add_argument("--cov-from", default=f"{DATA}/targets_g0_1M.fits")
    p.add_argument("-n", type=int, default=20000)
    p.add_argument("--samples", type=int, default=8192)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--batch", type=int, default=64,
                   help="targets per `pqr_streamed` batch. Memory goes as "
                        "batch x chunk, so raising --chunk to kill the merge "
                        "means lowering this by the same factor.")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--n-iter", type=int, default=None,
                   help="override the centroid layer's fixed-point inverse "
                        "iteration count. `sample` uses that inverse while "
                        "`log_prob` uses the exact forward map, so if 3 passes "
                        "have not converged the proposal is drawn from one "
                        "distribution and weighted by another -- which breaks "
                        "the IS identity. At the deep Sigma_X the round-trip "
                        "p99 is 4.8e-4 at n_iter=3 against 7.7e-5 at 6.")
    p.add_argument("--sigma-x-scale", type=float, default=1.0,
                   help="multiply Sigma_X (the CENTROID covariance) by this, "
                        "leaving C_M alone. The layer's displacement is O(T) "
                        "with T proportional to Sigma_X, so this dials the "
                        "layer's strength while the control stays exactly "
                        "self-consistent at whatever Sigma_X it is told.")
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="multiply C_M by this factor, matching bias.py --noise-scale")
    p.add_argument("--g", type=float, default=0.02,
                   help="shear of the +/- arms. m1 keeps ODD orders in g, "
                        "so an O(g^3) error in ghat -- the PQR expansion's own "
                        "truncation -- shows up as m1 proportional to g^2.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--draw-seed", type=int, default=None,
                   help="seed for the KERNEL/PROPOSAL draws alone, leaving the "
                        "targets and their noise realisation fixed. Defaults to "
                        "--seed + 100, i.e. every run before this flag existed. "
                        "Two runs differing only here isolate the Monte-Carlo "
                        "integral: their per-target Qhat differ by 2 Var_MC(Qhat), "
                        "which is the term that biases Rhat HIGH (Rhat contains "
                        "(B/A)^2, the square of an MC estimate) and so biases m1 "
                        "LOW by sum Var_MC(Qhat) / sum j -- flat in g, "
                        "a pure response-scale error.  See bias.pqr_streamed's "
                        "`crossfit`, which removes it. It is also the only way to "
                        "get a DRAW-realisation error bar: the bootstrap "
                        "everywhere in HANDOFF.md resamples TARGETS and leaves "
                        "every target's draws untouched, so it cannot see this.")
    p.add_argument("--nbin", type=int, default=8)
    p.add_argument("--save-pqr", default=None,
                   help="write the control PQR (+ its size axis) here, "
                        "for check_noisy_size_profile.py --control")
    a = p.parse_args()

    m_train = shear_top.load(a.train_data)[0]
    if not a.centroid:
        m_train = m_train[:int(0.9 * len(m_train))]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=a.centroid)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    if a.n_iter is not None:
        from models.centroid import CentroidMarginalize
        old = flow.bijection.bijection.bijections[0]
        new = eqx.tree_at(lambda l: l.coeffs,
                          CentroidMarginalize(jr.key(0), n_iter=a.n_iter,
                                              cond_dim=old.cond_shape[0]),
                          old.coeffs)
        flow = eqx.tree_at(lambda f: f.bijection.bijection.bijections[0],
                           flow, new)
        print(f"centroid layer inverse: n_iter = {a.n_iter}")

    # Sigma_X is constant across these catalogs, so one row conditions the draws
    # and a repeat of it feeds `pqr_streamed`'s per-target argument.
    sigma_x = None
    if a.centroid:
        sigma_x = np.repeat(np.asarray(fitsio.read(a.cov_from)["cov_odd"][:1],
                                       dtype=np.float64), a.n, axis=0)
        sigma_x *= a.sigma_x_scale
        if a.sigma_x_scale != 1.0:
            print(f"Sigma_X scaled by {a.sigma_x_scale}")

    # `transform` is the base -> data direction for this chain (`sample` agrees
    # with it exactly); a shared z at three conditions gives paired arms.
    z = flow.base_dist.sample(jr.key(a.seed), (a.n,))
    cond = lambda g: bias.condition(
        jnp.asarray(g), None if sigma_x is None else jnp.asarray(sigma_x[0]))
    draw = lambda g: np.asarray(
        jax.vmap(lambda zz: flow.bijection.transform(zz, cond(g)))(z),
        dtype=np.float64)
    G = a.g
    m0, mp, mm = draw([0.0, 0.0]), draw([G, 0.0]), draw([-G, 0.0])

    ok = np.isfinite(m0).all(1) & np.isfinite(mp).all(1) & np.isfinite(mm).all(1)
    ok &= np.asarray(in_domain(m0) & in_domain(mp) & in_domain(mm))
    print(f"{a.flow}: {a.n} flow draws, {int((~ok).sum())} dropped "
          f"(non-finite or off-chart)")
    m0, mp, mm = m0[ok], mp[ok], mm[ok]
    if sigma_x is not None:
        sigma_x = sigma_x[ok]

    cov = bias.load_cov(a.cov_from) * a.noise_scale ** 2
    # Same noise realisation in both arms, as `bias.main` does for its paired
    # catalogs -- otherwise shape noise swamps m1 at this n.
    Mp = bias.add_noise(mp, cov, a.seed + 1)
    Mm = bias.add_noise(mm, cov, a.seed + 1)

    dseed = a.seed + 100 if a.draw_seed is None else a.draw_seed
    if a.draw_seed is not None:
        print(f"draw seed: {dseed} (targets and noise unchanged)")

    qr = {}
    for k, M in (("plus", Mp), ("minus", Mm)):
        q, r = bias.pqr_streamed(flow, jnp.asarray(M), cov, a.samples, a.alpha,
                                 dseed, sigma_x=sigma_x, chunk=a.chunk,
                                 batch=a.batch)
        qr[k] = (np.asarray(q, np.float64), np.asarray(r, np.float64))
        print(f"  {k} arm done")

    fin = np.ones(len(m0), bool)
    for q, r in qr.values():
        fin &= np.isfinite(q).all(1) & np.isfinite(r).reshape(len(r), -1).all(1)
    rn = [np.linalg.norm(r.reshape(len(r), -1), axis=1) for _, r in qr.values()]
    med = np.median(np.concatenate([x[fin] for x in rn]))
    for x in rn:
        fin &= x < 1000 * med           # bias.py's own curvature-spike guard
    print(f"  keeping {int(fin.sum())} of {len(m0)} after finite + |R| guard")

    (qp, rp), (qm, rm) = qr["plus"], qr["minus"]
    y = m0[:, 1] / m0[:, 0]
    rng = np.random.default_rng(0)
    m1 = lambda s: bias.bias(qp[s], rp[s], qm[s], rm[s], G)[0]
    boot = lambda s: np.std([m1(i) for i in
                             (rng.choice(np.flatnonzero(s), int(s.sum()))
                              for _ in range(NBOOT))])

    print(f"\n  overall m1 = {m1(fin):+.4f} +/- {boot(fin):.4f}   "
          f"(must be ~0: the flow IS the prior here)")

    if a.save_pqr:
        # `y` travels with the PQR: the control's targets are flow draws, not
        # catalog rows, so its size axis cannot be recovered from a catalog the
        # way `check_noisy_size_profile` recovers the real run's.
        np.savez_compressed(a.save_pqr, plus_q=qp, plus_r=rp, minus_q=qm,
                            minus_r=rm, y=y, x=np.log10(m0[:, 0]), keep=fin)
        print(f"  wrote {a.save_pqr}")

    ed = np.quantile(y[fin], np.linspace(0, 1, a.nbin + 1))
    ed[0], ed[-1] = -np.inf, np.inf
    print(f"\n=== m1 vs TRUE Mr/Mf, {a.nbin} bins -- the same (hidden-variable) "
          f"binning the hump profile uses ===")
    for k in range(a.nbin):
        s = fin & (y >= ed[k]) & (y < ed[k + 1])
        if s.sum() < 200:
            continue
        print(f"  Mr/Mf < {min(ed[k+1], y[fin].max()):6.3f}  n={int(s.sum()):6d}  "
              f"m1 = {m1(s):+.4f} +/- {boot(s):.4f}")

    st, sb = fin & (y >= 3.15) & (y < 3.40), fin & (y >= 2.00) & (y < 2.60)
    if st.sum() > 200 and sb.sum() > 200:
        it, ib = np.flatnonzero(st), np.flatnonzero(sb)
        v = m1(st) - m1(sb)
        e = np.std([m1(rng.choice(it, len(it))) - m1(rng.choice(ib, len(ib)))
                    for _ in range(NBOOT)])
        print(f"\n  HUMP (same windows as the real run) = {v:+.4f} +/- {e:.4f}")
        print(f"  real bulgedisc run gave             = +0.0240 +/- 0.0033")


if __name__ == "__main__":
    main()
