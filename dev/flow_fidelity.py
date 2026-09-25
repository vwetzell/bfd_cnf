"""How well does the flow reproduce the distribution its templates came from?

Two stages, so a misfit can be pinned on a layer:

A. bulk + shear (`--shear-flow`, no centroid) against the HELD-OUT last 10% of
   the prior catalog it was trained on (`bulk._split`/`shear.py` hold it
   out), each galaxy lensed by its own exact dm_dg/d2m_dg2.
B. the full flow (`--flow`, with centroid at the targets' Sigma_X) against
   shifted template copies resampled by their eq.-36 weight -- that resample
   IS the centroid-marginalised template density P(M | g, Sigma_X) -- with
   the weight's own lensed centroid X(g) at g != 0 (`log_weights(..., g=)`).
   Every copy galaxy was in training: B measures misfit, not overfit.

Per stage, everything by flux bin:
  1. C2ST: held-out AUC of a gradient-boosted classifier separating flow draws
     from reference draws, next to a BASELINE AUC for two reference samples
     from disjoint galaxy halves.  Only the excess over baseline is misfit.
  2. PIT: reference pushed to the flow's base must be iid N(0,1).  z0-z2 are
     flux-correlated, so only their UNBINNED moments are a test; z3, z4
     (spin 2) are tested per flux bin too.
  3. Shear response d<e1>/dg1 by central difference at g = +/-0.02 (common
     base draws for the flow; the same galaxies/copies re-lensed and
     re-weighted for the reference), with errors.
"""
import argparse
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "dev")
sys.path.insert(0, "../bfd_cnf_imsims")

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import bias as B
import bulk
import shear
from closed_loop import D, REAL
from imsims.copies import log_weights

COPIES = f"{D}/copies_gauss2_fwd_g2v4n.fits"
PRIOR = f"{D}/{B.TRAIN_DATA['gauss2_v4n']}"
BINS = [0, 3000, 5000, 8000, 20000, np.inf]
EBINS = [0, 0.05, 0.08, 0.11, 0.15, np.inf]   # --bin-by e: galaxy |e| = |M1+iM2|/Mr
COLS = ["gal", "moments", "xy", "da", "dm_dg", "d2m_dg2", "dxy_dg", "d2xy_dg2"]
H = 0.02


def feats(m):
    Mf, Mr = m[:, 0], m[:, 1]
    return np.stack([np.log(np.abs(Mf)), Mr / Mf, m[:, 2] / Mr, m[:, 3] / Mr, m[:, 4] / Mr], 1)


def e1(m):
    return m[:, 2] / m[:, 1]


def lens(m, dm, d2m, g):
    q = np.array([0.5 * g[0] ** 2, g[0] * g[1], 0.5 * g[1] ** 2])
    return m + g[0] * dm[:, 0] + g[1] * dm[:, 1] + np.einsum("k,nkj->nj", q, d2m)  # dm (n,2,5), d2m (n,3,5)


def bins():
    return list(zip(BINS[:-1], BINS[1:]))


def c2st(a, b, rng):
    """Held-out AUC separating sample a (label 0) from b (label 1), overall and per flux bin."""
    X = np.concatenate([feats(a), feats(b)]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    p = rng.permutation(len(y)); X, y = X[p], y[p]; h = len(y) // 2
    pr = HistGradientBoostingClassifier(max_iter=300).fit(X[:h], y[:h]).predict_proba(X[h:])[:, 1]
    yt, fl = y[h:], np.exp(X[h:, 0])
    per = [roc_auc_score(yt[k], pr[k]) if k.sum() > 400 and 0 < yt[k].mean() < 1 else np.nan
           for k in ((fl >= lo) & (fl < hi) for lo, hi in bins())]
    return roc_auc_score(yt, pr), per


def report_c2st(name, flow_vs, base):
    print(f"\n  1. C2ST {name}: AUC flow-vs-ref {flow_vs[0]:.4f}   baseline ref-vs-ref {base[0]:.4f}")
    for (lo, hi), f, b in zip(bins(), flow_vs[1], base[1]):
        print(f"     flux {lo:6.0f}-{hi:<7.0f} flow {f:.4f}  baseline {b:.4f}  excess {f - b:+.4f}")


def report_pit(z, flux):
    ok = np.isfinite(z).all(1); z, flux = z[ok], flux[ok]
    print(f"\n  2. PIT ({(~ok).sum()} non-finite dropped); unbinned mean/std per base coord, <|z|^2> (want 5):")
    print("     " + "  ".join(f"z{i} {z[:, i].mean():+.3f}/{z[:, i].std():.3f}" for i in range(5))
          + f"   <|z|^2> {np.mean((z ** 2).sum(1)):.3f}")
    for lo, hi in bins():
        k = (flux >= lo) & (flux < hi)
        print(f"     flux {lo:6.0f}-{hi:<7.0f} n {k.sum():7d}  z3 {z[k, 3].mean():+.3f}/{z[k, 3].std():.3f}"
              f"  z4 {z[k, 4].mean():+.3f}/{z[k, 4].std():.3f}")


def binvar(m):
    return np.hypot(m[:, 2], m[:, 3]) / m[:, 1] if BIN_BY == "e" else m[:, 0]


def report_response(d_flow, flux_flow, ref_fn, flux_ref):
    """d_flow: per-draw CRN difference of e1; ref_fn(mask) -> (value, error)."""
    print(f"\n  3. d<e1>/dg1 by {'|e|' if BIN_BY == 'e' else 'flux'}:   flow                 ref            flow/ref")
    for lo, hi in (zip(EBINS[:-1], EBINS[1:]) if BIN_BY == "e" else bins()):
        kf = (flux_flow >= lo) & (flux_flow < hi) & np.isfinite(d_flow)
        kr = (flux_ref >= lo) & (flux_ref < hi)
        f, fe = d_flow[kf].mean(), d_flow[kf].std() / np.sqrt(kf.sum())
        r, re = ref_fn(kr)
        rat = f / r
        print(f"     {lo:8.4g}-{hi:<8.4g} {f:+.4f}+/-{fe:.4f}   {r:+.4f}+/-{re:.4f}   "
              f"{rat:.4f}+/-{abs(rat) * np.hypot(fe / f, re / r):.4f}")


def draw_all(flow, cond, z, batch):
    f = eqx.filter_jit(jax.vmap(lambda zi, c: flow.bijection.transform(zi, c), in_axes=(0, None)))
    return np.concatenate([np.asarray(f(z[i:i + batch], cond), np.float64) for i in range(0, len(z), batch)])


def to_base(flow, cond, m, batch):
    f = eqx.filter_jit(jax.vmap(lambda mi, c: flow.bijection.inverse(mi, c), in_axes=(0, None)))
    return np.concatenate([np.asarray(f(jnp.asarray(m[i:i + batch], jnp.float32), cond), np.float64)
                           for i in range(0, len(m), batch)])


def flow_response(flow, cond_of, z, batch):
    m0 = draw_all(flow, cond_of(jnp.zeros(2)), z, batch)
    mp = draw_all(flow, cond_of(jnp.array([H, 0.0])), z, batch)
    mm = draw_all(flow, cond_of(jnp.array([-H, 0.0])), z, batch)
    return m0, (e1(mp) - e1(mm)) / (2 * H)


def stage_a(a, rng):
    m, dm, d2m = shear.load(PRIOR)
    n_tr = int(0.9 * len(m))
    flow = eqx.tree_deserialise_leaves(
        a.shear_flow, bulk.build_flow(jr.key(0), m[:n_tr][:20000], shear=True, centroid=False))
    cond_of = lambda g: B.condition(g, None)
    ho = slice(n_tr, None)
    mh, dmh, d2mh = m[ho], dm[ho], d2m[ho]
    print(f"\n=== A. bulk+shear {a.shear_flow} vs held-out prior ({len(mh)} galaxies)")
    n = len(mh)
    z = flow.base_dist.sample(jr.key(a.seed), (n,))
    m0, d_flow = flow_response(flow, cond_of, z, a.batch)
    ok = np.isfinite(m0).all(1)
    tr = m[rng.choice(n_tr, n, replace=False)]      # baseline: train-split real vs held-out real
    if not a.response_only:
        report_c2st("(A)", c2st(mh, m0[ok], rng), c2st(mh, tr, rng))
        report_pit(to_base(flow, cond_of(jnp.zeros(2)), mh, a.batch), mh[:, 0])
    d_ref = (e1(lens(mh, dmh, d2mh, [H, 0])) - e1(lens(mh, dmh, d2mh, [-H, 0]))) / (2 * H)
    report_response(d_flow, binvar(m0), lambda k: (d_ref[k].mean(), d_ref[k].std() / np.sqrt(k.sum())),
                    binvar(mh))


def stage_b(a, rng):
    m_tr = np.asarray(shear.load(PRIOR)[0])[:20000]
    flow = eqx.tree_deserialise_leaves(
        a.flow, bulk.build_flow(jr.key(0), m_tr, shear=True, centroid=True))
    sx = fitsio.read(f"{D}/{REAL['plus']}.fits", rows=[0])["cov_odd"][0]
    sxj = jnp.asarray(sx, jnp.float32)
    cond_of = lambda g: B.condition(g, sxj)
    F = fitsio.FITS(COPIES)["COPIES"]
    c = np.concatenate([F.read(columns=COLS, rows=np.arange(s, s + a.slab_rows)) for s in a.slabs])
    print(f"\n=== B. full flow {a.flow} vs eq.-36-weighted copies "
          f"({len(c)} copies, {len(np.unique(c['gal']))} galaxies)")
    lw0 = log_weights(c, sx)
    w0 = np.exp(lw0 - lw0.max())

    def resample(mask, n):
        p = w0 * mask; p /= p.sum()
        return rng.choice(len(c), n, p=p)

    half = (c["gal"] % 2 == 0)
    ia, ib = resample(half, a.n // 2), resample(~half, a.n // 2)
    idx = resample(np.ones(len(c), bool), a.n)
    z = flow.base_dist.sample(jr.key(a.seed), (a.n,))
    m0, d_flow = flow_response(flow, cond_of, z, a.batch)
    ok = np.isfinite(m0).all(1)
    if not a.response_only:
      report_c2st("(B)", c2st(c["moments"][idx], m0[ok], rng),
                c2st(c["moments"][ia], c["moments"][ib], rng))
      report_pit(to_base(flow, cond_of(jnp.zeros(2)), c["moments"][idx], a.batch), c["moments"][idx, 0])

    mp = lens(c["moments"], c["dm_dg"], c["d2m_dg2"], [H, 0])
    mm = lens(c["moments"], c["dm_dg"], c["d2m_dg2"], [-H, 0])
    wp = np.exp(log_weights(c, sx, g=np.array([H, 0.0])) - lw0.max())
    wm = np.exp(log_weights(c, sx, g=np.array([-H, 0.0])) - lw0.max())
    ep, em, flux = e1(mp), e1(mm), c["moments"][:, 0]
    blk = c["gal"] % 20

    def ref(k):
        est = lambda s: ((wp[s] * ep[s]).sum() / wp[s].sum() - (wm[s] * em[s]).sum() / wm[s].sum()) / (2 * H)
        full = est(k)
        jk = np.array([est(k & (blk != b)) for b in range(20)])
        return full, np.sqrt(19 / 20 * ((jk - jk.mean()) ** 2).sum())

    report_response(d_flow, binvar(m0), ref, binvar(c["moments"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flow", default="flows/centroid_g2v4n_K.eqx")
    p.add_argument("--shear-flow", default="flows/shear_g2v4n.eqx")
    p.add_argument("--stage", choices=["A", "B", "both"], default="both")
    p.add_argument("--n", type=int, default=400_000)
    p.add_argument("--slabs", type=int, nargs="+", default=[0, 40_000_000, 80_000_000])
    p.add_argument("--slab-rows", type=int, default=4_000_000)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--batch", type=int, default=65536)
    p.add_argument("--bin-by", choices=["flux", "e"], default="flux")
    p.add_argument("--response-only", action="store_true")
    a = p.parse_args()
    global BIN_BY
    BIN_BY = a.bin_by
    rng = np.random.default_rng(a.seed)
    if a.stage in ("A", "both"):
        stage_a(a, rng)
    if a.stage in ("B", "both"):
        stage_b(a, rng)


if __name__ == "__main__":
    main()
