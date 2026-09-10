"""Re-quote a saved run at 2^24 selection draws, and scan the window locally.

Three jobs at once, because they share the one expensive thing:

  A1  re-quote a config's corrected `m1`, `c1`, `c2` at 2^24 draws, retiring
      the unquoted +/-4e-03 that sits under every 2^20 number;
  A2  section 3.4's LOCAL window stability -- perturb each boundary by ~5-10%,
      one at a time, and read the SLOPE `d m1 / d(boundary)`;
  A3  `c = 0`, the free absolute test, at the same draw count.

`ghat` is affine in `(P_s, Q_s, R_s)` and `--save-pqr` stored each arm's own
per-target `Q`, `R` and observed moments, so none of this needs a target
integration -- it is a re-solve off `pqr/*.npz`.  And the selection terms
themselves depend only on the flow, `C_M`, `Sigma_X` and the window, so all
the windows share ONE pass over the draws (`selection_terms_score(windows=)`).
That also makes the scan PAIRED: every window sees the identical draw set,
hence the identical MC noise, which is what lets a 7e-04 slope be read at all.

The perturbations are deliberately LOCAL.  General window-independence is not
the requirement and is not tested here -- the cuts exist to exclude regions
governed by systematics this estimator does not model, so a failure far
outside them would refute nothing.  In particular the size floor is NOT pushed
towards 3.0, which walks into the known size-edge support leak, a separate
channel.  See `GUIDING_PRINCIPLES.md` section 3.4.

    python -u dev/window_scan.py --pop bulgedisc_v3 --pqr pqr/v11_plain.npz
    python -u dev/window_scan.py --pop bulgedisc_v3_psfe2p --pqr pqr/v11_psfe2p.npz --no-scan
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

import bias as B
import bulk
import shear

# The size CEILING is 3.2, not the 1e9 this file and `dev/gauss2_*.sh` used
# until 2026-09-06.  It is doing real work, not tidying: without it the window
# admits objects whose measured Mr/Mf is at or past `POINT_SOURCE` (3.692575),
# i.e. SMALLER THAN A POINT SOURCE -- noise-scattered, physically impossible,
# and where `in_domain` rejects most of the kernel cloud so the per-target
# integral barely converges.  Measured on gauss2_v3d: those are 8.05% of the
# no-ceiling window and carry essentially ALL of its draw noise (seed-to-seed
# spread on windowed uncorrected m1, 0.01815 without the ceiling against
# 0.00179 with it).  The ceiling also MOVES THE ANSWER: uncorrected m1 goes
# -0.04698 -> +0.03196 (shallow) and -0.05122 -> +0.02753 (deep).
NOMINAL_SIZE = (2.2, 3.2)
NOMINAL_FLUX = (2500.0, 50000.0)

# (label, which boundary) -> how to build a perturbed window.
BOUNDARIES = {
    "size_lo": lambda s, f, v: ((v, s[1]), f),
    "size_hi": lambda s, f, v: ((s[0], v), f),
    "flux_lo": lambda s, f, v: (s, (v, f[1])),
    "flux_hi": lambda s, f, v: (s, (f[0], v)),
}
NOMINAL_VALUE = {"size_lo": NOMINAL_SIZE[0],
                 "size_hi": NOMINAL_SIZE[1],
                 "flux_lo": NOMINAL_FLUX[0],
                 "flux_hi": NOMINAL_FLUX[1]}


def build_windows(fracs, size=None, flux=None):
    """[(label, boundary, value, frac, (size, flux))], nominal first."""
    size = NOMINAL_SIZE if size is None else tuple(size)
    flux = NOMINAL_FLUX if flux is None else tuple(flux)
    nominal = {"size_lo": size[0], "size_hi": size[1],
               "flux_lo": flux[0], "flux_hi": flux[1]}
    out = [("nominal", None, 0.0, 0.0, (size, flux))]
    for name, make in BOUNDARIES.items():
        v0 = nominal[name]
        if not np.isfinite(v0) or v0 > 1e8:
            continue        # an absent boundary has nothing to perturb
        for fr in fracs:
            for sgn in (-1, +1):
                v = v0 * (1.0 + sgn * fr)
                out.append((f"{name} {sgn * fr:+.0%}", name, v, sgn * fr,
                            make(size, flux, v)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/centroid_v11.eqx")
    p.add_argument("--pop", default="bulgedisc_v3")
    # `plain` and `psfe00` are bit-identical in BOTH C_M and Sigma_X (checked
    # 2026-09-04), so one selection-term pass serves every pqr taken under a
    # circular PSF -- including the MC null `v11_psfe00_s7.npz`.
    p.add_argument("--pqr", nargs="+", default=["pqr/v11_plain.npz"])
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--log2-draws", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--window-guard", type=float, default=0.0,
                   help="drop selection draws whose |Q| or |R| exceeds "
                        "this factor times the pilot median -- the SAME "
                        "knob bias.py passes as --window-guard, which "
                        "this path used to ignore. Off by default like bias.py's, because it is a TRUNCATION: quote the sensitivity, not one "
                        "value. Pass 100 to match the g2v3d/g2v3e runs.")
    p.add_argument("--draw-chunk", type=int, default=0,
                   help="generate the prior draws in chunks of this "
                        "many instead of one allocation. Needed past "
                        "2^26 on a 16 GB card (2^28 asks for 10 GiB). "
                        "Uses fold_in per chunk, so the draw stream "
                        "DIFFERS from an unchunked run of the same seed.")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--no-prefilter-weights", action="store_true",
                   help="accept a PQR with no `prefilter_w`. ONLY for a run that really had no --prefilter-pad")
    p.add_argument("--g", type=float, default=0.02)
    p.add_argument("--size", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help=f"size window (Mr/Mf); default {NOMINAL_SIZE}")
    p.add_argument("--flux", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help=f"flux window (Mf); default {NOMINAL_FLUX}")
    p.add_argument("--perturb", type=float, nargs="*", default=[0.05, 0.10])
    p.add_argument("--no-scan", action="store_true",
                   help="nominal window only (A1/A3 without A2)")
    p.add_argument("--drop-top", type=int, default=0,
                   help="drop the k targets with the largest |R11| before the "
                        "re-solve -- the concentration check; psfe1m needs 1")
    p.add_argument("--cache", default=None,
                   help="npz of selection terms; reused if it exists, written "
                        "if not.  Default is derived from pop/draws/seed.")
    a = p.parse_args()

    wins = build_windows([] if a.no_scan else a.perturb, a.size, a.flux)

    cat = B.CATALOGS[a.pop]
    targets = f"{a.data_dir}/{cat['zero']}.fits"
    cov = B.load_cov(targets)
    sigma_x = jnp.asarray(
        np.asarray(fitsio.read(targets)["cov_odd"], dtype=np.float64)[0])

    n = 1 << a.log2_draws
    print(f"flow {a.flow}   pop {a.pop}   pqr {a.pqr}")
    print(f"{n} prior draws (2^{a.log2_draws}), seed {a.seed}, "
          f"{len(wins)} windows in one pass")
    print(f"Sigma_X = {np.asarray(sigma_x)}\n", flush=True)

    # The terms are the only GPU work here and they cost ~10 min at 2^24, so
    # cache them: they depend on the flow, C_M, Sigma_X, the window and the
    # draw seed, and on NOTHING about the targets.  Every re-analysis of the
    # scan is then a second of numpy -- and the flow is built INSIDE the else
    # branch so a cached re-solve never touches the GPU at all, which is what
    # lets one run while a 2^24 pass is in flight (the 16 GB card OOMs with two
    # JAX processes on it).
    # The chunked path draws a different stream, so it must not collide with
    # an unchunked cache of the same (pop, draws, seed).
    cache = a.cache or (f"logs/phase_a/terms_{a.pop}_{a.log2_draws}_"
                        f"s{a.seed}_{len(wins)}w"
                        f"{f'_c{a.draw_chunk}' if a.draw_chunk else ''}"
                        f"{f'_g{a.window_guard:g}' if a.window_guard else ''}"
                        ".npz")
    if os.path.exists(cache):
        z = np.load(cache)
        assert z["n_windows"] == len(wins), f"{cache} holds a different scan"
        terms = [(np.float64(z["ps"][i]), z["qs"][i], z["rs"][i], z["qerr"][i])
                 for i in range(len(wins))]
        print(f"selection terms loaded from {cache}\n", flush=True)
    else:
        # `bias.py`'s construction exactly: the centroid flow was standardised
        # on the FULL population, so no 0.9 slice.  (The chart constants come
        # from the checkpoint either way, but the base distribution's shape
        # does not.)
        m_train = shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0]
        flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
        flow = eqx.tree_deserialise_leaves(a.flow, flow)
        jax.config.update("jax_enable_x64", True)
        flow = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x,
            flow)
        flow = bulk.SupportedFlow(flow, eps=0.0, support=True)
        # `jr.key(seed + 31)` and the 16384 chunking are bias.py's, so a given
        # (draws, seed) reproduces its run exactly.
        tr = eqx.filter_jit(jax.vmap(lambda z1: flow.bijection.transform(
            z1, B.condition(jnp.zeros(2), sigma_x))))
        if a.draw_chunk:
            # `base_dist.sample(key, (n,))` materialises ALL n base points on
            # the card at once: at 2^28 that is one 10 GiB allocation on a
            # 16 GB card and it dies.  The transform below was already chunked;
            # only the sampling was not.
            #
            # Chunking it needs a DIFFERENT key per chunk -- `sample` at a
            # fixed key NESTS, so k calls with one key return k identical
            # blocks (the same trap as [[pqr-streamed-seed-nests]]).
            # `fold_in` gives independent chunks, which necessarily changes the
            # draw stream, so this is opt-in and carries its own cache name.
            # The estimand is unchanged: these are still n iid draws from the
            # prior, just not the same n.
            # Filled in place, not concatenated: at 2^28 the host array is
            # 10.7 GB and `np.concatenate` would hold the chunks AND the
            # result at once, peaking at 21 GB for no reason.
            # TWO levels, and both are load-bearing.  The outer one bounds
            # the SAMPLE (the 10 GiB allocation that started this); the inner
            # 16384 bounds the TRANSFORM, which is what the unchunked path
            # always did.  Feeding a whole draw-chunk straight into `tr`
            # replaces a 10 GiB failure with a 20 GiB one -- measured, at
            # 2^24 chunked into 4.
            cs = a.draw_chunk
            nc = (n + cs - 1) // cs
            m0 = np.empty((n, 5), dtype=np.float64)
            for j in range(nc):
                lo, hi = j * cs, min((j + 1) * cs, n)
                zs = flow.base_dist.sample(
                    jr.fold_in(jr.key(a.seed + 31), j),
                    (hi - lo,)).astype(jnp.float64)
                for i in range(0, hi - lo, 16384):
                    k = min(i + 16384, hi - lo)
                    m0[lo + i:lo + k] = np.asarray(tr(zs[i:k]))
                del zs
            print(f"  {n} draws in {nc} chunks of {cs} ({m0.nbytes / 2**30:.1f}"
                  f" GB on the host) -- DIFFERENT stream from an unchunked run "
                  f"of the same seed, do not pair them", flush=True)
        else:
            zs = flow.base_dist.sample(
                jr.key(a.seed + 31), (n,)).astype(jnp.float64)
            m0 = np.concatenate(
                [np.asarray(tr(zs[i:i + 16384])) for i in range(0, n, 16384)])
            del zs
        # `bias.py` passes `guard=a.window_guard` here (the runs use 100);
        # this path passed nothing, so the offline re-solve was a DIFFERENT,
        # unguarded estimator from the in-run one.  It survived 2^24 and 2^26
        # on luck: at 2^28 seed 0 a single draw of 268M (Mf = 1999,
        # Mr/Mf = 1.741) took 100% of R_s11, R_s11 = 3.4e8, m1 = -0.61.
        terms = B.selection_terms_score(
            flow, m0, cov, None, None, sigma_x=sigma_x, batch=a.batch,
            windows=[w[4] for w in wins], guard=a.window_guard)
        del m0
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, n_windows=len(wins),
                 ps=np.array([t[0] for t in terms]),
                 qs=np.stack([t[1] for t in terms]),
                 rs=np.stack([t[2] for t in terms]),
                 qerr=np.stack([t[3] for t in terms]))
        print(f"selection terms cached to {cache}\n", flush=True)

    for pqr_path in a.pqr:
        print(f"\n{'=' * 100}\nre-solve {pqr_path}", flush=True)
        d = np.load(pqr_path)
        qp, rp, qm, rm = d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"]
        obs_p, obs_m = d["obs_plus"], d["obs_minus"]
        # N_ns must count the CATALOG's out-of-window population, not the PQR
        # file's rows.  With `--prefilter-pad` the out-of-window rows are a
        # subsample and each stands for 1/sample of them; counting rows
        # under-counts by 64% and moves m1 by 0.015 with no change in the bar.
        # Ones when the key is absent (a file written before bias.py wrote it,
        # or a run with no pre-filter) reproduces the old behaviour exactly.
        if "prefilter_w" not in d.files and not a.no_prefilter_weights:
            raise SystemExit(
                f"{pqr_path} has no `prefilter_w`. If it came from a "
                "--prefilter-pad run, N_ns is under-counted by ~64% and m1 "
                "moves by 0.015 with NO change in the bar -- silently. "
                "Re-run bias.py, or pass --no-prefilter-weights if the run "
                "genuinely had no pre-filter.")
        pw = d["prefilter_w"] if "prefilter_w" in d.files else np.ones(len(qp))

        # `bias.py`'s |Q|/|R| guard fires at 1000x the median, and psfe1m
        # carries a target at 572x that it therefore KEEPS -- one galaxy
        # holding 7.9% of SUM|R11| inside the window, against 0.13% for every
        # other config.  Its corrected m1 is -0.072 +/- 0.069 with that galaxy
        # and about -0.009 +/- 0.005 without, so the number is not a
        # measurement until it is dropped.  Quote both, never just one, and do
        # NOT fix this by loosening the guard's factor -- that is a tuned
        # constant, and the concentration report is the right net.
        if a.drop_top:
            key = np.nan_to_num(np.abs(rp[:, 0, 0]), nan=np.inf)
            keep = np.sort(np.argsort(key)[:-a.drop_top])
            print(f"  dropped the {a.drop_top} target(s) with the largest "
                  f"|R11|: largest {key.max():.4g}, largest kept "
                  f"{key[keep].max():.4g}, median {np.median(key):.4g}")
            qp, rp, qm, rm = qp[keep], rp[keep], qm[keep], rm[keep]
            obs_p, obs_m = obs_p[keep], obs_m[keep]
            pw = pw[keep]

        masks, nss, rows = [], [], []
        for (label, bnd, val, frac, (size, flux)), (ps, qs, rs, _) in zip(wins, terms):
            sp = B.window_mask(obs_p, size, flux)
            sm = B.window_mask(obs_m, size, flux)
            masks.append((sp, sm))
            nss.append(((float(pw[~sp].sum()), ps, qs, rs),
                        (float(pw[~sm].sum()), ps, qs, rs)))
            m1u, _, _ = B.bias(qp, rp, qm, rm, a.g, sel=(sp, sm))
            m1c, c1c, c2c = B.bias(qp, rp, qm, rm, a.g, sel=(sp, sm), ns=nss[-1])
            rows.append(dict(label=label, bnd=bnd, val=val, frac=frac, m1c=m1c,
                             m1u=m1u, c1=c1c, c2=c2c, ps=ps, rs=rs,
                             n_in=int(sp.sum())))

        # One resample index shared by EVERY window, so the difference against the
        # nominal is paired galaxy for galaxy.  That matters more than the absolute
        # bar here: moving a boundary by 10% swaps a few hundred galaxies in or
        # out, and the resulting jump in m1 is ordinary sample variance, not a
        # failure of the selection correction.  The unpaired bar (+/-0.008) cannot
        # tell those apart; the paired one can.
        rng = np.random.default_rng(a.seed)
        n_all = len(qp)
        boots = np.empty((a.boot, len(wins), 3))
        for b in range(a.boot):
            idx = rng.choice(n_all, n_all)
            qpi, rpi, qmi, rmi = qp[idx], rp[idx], qm[idx], rm[idx]
            pwi = pw[idx]
            for w, ((sp, sm), (nsp, nsm)) in enumerate(zip(masks, nss)):
                spi, smi = sp[idx], sm[idx]
                boots[b, w] = B.bias(
                    qpi, rpi, qmi, rmi, a.g, sel=(spi, smi),
                    ns=((float(pwi[~spi].sum()),) + nsp[1:],
                        (float(pwi[~smi].sum()),) + nsm[1:]))
        err = boots.std(axis=0, ddof=1)
        derr = (boots - boots[:, :1]).std(axis=0, ddof=1)

        print(f"\n{'window':>16}{'n_in':>7}{'P_s':>9}{'R_s11':>10}{'R_s12':>10}"
              f"{'tracels':>10}{'m1 unc':>10}{'m1 corr':>10}{'+/-gal':>9}"
              f"{'d vs nom':>10}{'+/-paird':>9}{'c1':>10}{'c2':>10}")
        for w, r in enumerate(rows):
            r["boot"], r["dboot"] = err[w, 0], derr[w, 0]
            r["dm1"] = r["m1c"] - rows[0]["m1c"]
            tl = 0.5 * (r["rs"][0, 0] - r["rs"][1, 1])
            print(f"{r['label']:>16}{r['n_in']:>7}{r['ps']:>9.5f}"
                  f"{r['rs'][0,0]:>+10.5f}{r['rs'][0,1]:>+10.5f}{tl:>+10.2e}"
                  f"{r['m1u']:>+10.5f}{r['m1c']:>+10.5f}{err[w,0]:>9.5f}"
                  f"{r['dm1']:>+10.5f}{derr[w,0]:>9.5f}"
                  f"{r['c1']:>+10.2e}{r['c2']:>+10.2e}", flush=True)

        nom = rows[0]
        print(f"\nA1/A3 at 2^{a.log2_draws}, nominal window "
              f"size {NOMINAL_SIZE} flux {NOMINAL_FLUX}:")
        print(f"  corrected m1 = {nom['m1c']:+.5f} +/- {nom['boot']:.5f} (galaxy)")
        print(f"  c1 = {nom['c1']:+.3e} +/- {err[0,1]:.2e}   "
              f"c2 = {nom['c2']:+.3e} +/- {err[0,2]:.2e}   "
              f"(A3: forced to zero by isotropy + a circular PSF; the bar IS the "
              f"finite sample's own quadrupole)")
        print(f"  R_s traceless monitor = "
              f"{0.5*(nom['rs'][0,0]-nom['rs'][1,1]):+.3e}   "
              f"R_s12 = {nom['rs'][0,1]:+.3e}   (both forced to 0; they ARE the "
              f"selection MC error)")

        if len(rows) > 1:
            print(f"\nA2 local window stability.  The slope is the number a real "
                  f"analysis needs:\nit converts uncertainty in where the cut sits "
                  f"into a systematic on m1.")
            print(f"\n{'boundary':>10}{'nominal':>12}{'per +/-10% of m1':>20}"
                  f"{'sigma':>8}{'vs tau=1e-3':>13}{'paired sd':>11}")
            for name in BOUNDARIES:
                pts = [r for r in rows if r["bnd"] == name]
                if not pts:
                    continue
                v0 = NOMINAL_VALUE[name]
                idx = [0] + [rows.index(r) for r in pts]
                x = np.array([0.0] + [r["val"] - v0 for r in pts])
                slope = np.polyfit(x, np.array([rows[i]["m1c"] for i in idx]),
                                   1)[0]
                # The slope needs its OWN bar, and it is not the per-point one:
                # every window is fitted on the same galaxies, so refit the line
                # inside each paired bootstrap resample.  Without this the scan
                # cannot say whether a trend is real, which is the only thing
                # section 3.4 asks it.
                sd = float(np.std([np.polyfit(x, boots[b, idx, 0], 1)[0]
                                   for b in range(a.boot)], ddof=1))
                per10, per10sd = slope * 0.1 * v0, sd * 0.1 * v0
                bar = float(np.median([r["dboot"] for r in pts]))
                print(f"{name:>10}{v0:>12.6g}"
                      f"{per10:>+13.2e} +/-{per10sd:>6.1e}"
                      f"{abs(slope) / sd if sd else np.inf:>8.1f}"
                      f"{abs(per10) / 1e-3:>12.1f}x{bar:>11.2e}")
            print(f"\n('per +/-10%' is the systematic on m1 from a 10% "
                  f"uncertainty in where that cut sits --\n the number a real "
                  f"analysis needs.  Its bar is a PAIRED refit of the line "
                  f"inside\n each bootstrap resample, so 'sigma' says whether "
                  f"the trend is real at all; a\n slope consistent with zero "
                  f"but with a bar many times tau means the scan has\n NOT "
                  f"tested the boundary, which is a different statement from "
                  f"passing.)")


if __name__ == "__main__":
    main()
