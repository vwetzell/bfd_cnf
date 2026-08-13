"""The Mr/Mf ramp, judged only where a real analysis would keep the galaxies.

Everything above `Mr/Mf = 3.5` is out of scope here: that tail is a known,
separately diagnosed flow-density failure at the point-source boundary
(memory: `resolution-edge-bias-is-real`), and cutting it is legitimate because
the cut is spin-0 and evaluated on the SAME galaxies in the +g and -g arms, so
`dP(s|g)/dg = 0` and the eq. (45)-(46) selection terms stay negligible.  What is
left below 3.5 is a monotone ramp -- m1 running -1% to +3% with resolution --
and that is what this sweeps.

Metric: `dev/check_noiseless_shear_recovery.py` restricted to `Mr/Mf < 3.5`,
re-binned into octiles WITHIN the cut, summarised by the slope of m1 against
Mr/Mf and by RMS(m1) across those bins.  Global m1 is reported but must not be
optimised -- it cancels along the ramp.  This is the noiseless path, so it sees
the flow's density and response and nothing of the noisy-target machinery; it is
the fast selector, and the winner still has to be confirmed on the noisy run.

Two arms beyond the baseline:

* `traincut` -- train on `Mr/Mf < 3.65` only.  If the interior ramp is the
  edge population dragging a globally-fitted response around, dropping the
  galaxies we have already agreed not to measure fixes it for free.  The cut
  sits above the 3.5 evaluation boundary so the truncation's own edge is not
  what we then measure.
* `score` -- the first-order-ONLY score-contracted velocity term
  (`shear.train(score_weight=...)`).  `_velocity_mse` weighs each moment by its
  own response, but Q weighs by the SCORE, and the Mr/Mc columns are 8-58%
  wrong with a cancellation nothing in the loss enforces.  The earlier attempt
  at this summed first and second order and only ever drove the second (391 vs
  1.42); this term is first order alone.

Paired seeds throughout: the seed floor with the bulk retrained is sd 0.0038 on
RMS(m1), as large as any effect ever measured on this metric, so a single-seed
win means nothing.

    python dev/sweep_ramp_cut.py --eval-only        # baseline flow on disk
    python dev/sweep_ramp_cut.py base traincut      # named arms, all seeds
"""
import argparse
import json
import sys
import time

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402

DATA = "../bfd_cnf_imsims/data/moments.fits"
OUT = "dev/sweep_ramp_cut.jsonl"
G = 0.02
CUT = 3.5          # evaluation boundary: nothing above this is measured
NBINS = 8
SEEDS = (0, 1, 2)

#     name         train_cut   score_weight
ARMS = {
    "base":       (None, 0.0),
    "traincut":   (3.65, 0.0),
    "score3":     (None, 3.0),
    "score30":    (None, 30.0),
}


def recovery(flow, m, dm, d2m, n90):
    """m1 in octiles of Mr/Mf BELOW the cut, plus its slope and RMS.

    One shear direction (g1) is enough here: the direction split only dominates
    in the extreme octiles, which the cut has already removed.
    """
    shear_m = lambda s: m + s * G * dm[:, 0, :] + 0.5 * (s * G) ** 2 * d2m[:, 0, :]
    p = [np.asarray(v, np.float64)
         for s in (+1, -1) for v in bias.pqr(flow, shear_m(s))]
    m1 = lambda sel: bias.bias(p[0], p[1], p[2], p[3], G, sel)[0]

    r = m[:, 1] / m[:, 0]
    rng = np.random.default_rng(0)

    def err(sel, n=200):
        i0 = np.flatnonzero(sel)
        return float(np.std([m1(rng.choice(i0, len(i0))) for _ in range(n)]))

    out = {}
    for lab, base in (("train", np.arange(len(m)) < n90),
                      ("heldout", np.arange(len(m)) >= n90)):
        base = base & (r < CUT)
        e = np.quantile(r[base], np.linspace(0, 1, NBINS + 1))
        e[0], e[-1] = -np.inf, np.inf
        binned = [(float(r[s].mean()), float(m1(s)), err(s))
                  for k in range(NBINS)
                  for s in [base & (r >= e[k]) & (r < e[k + 1])]]
        x = np.array([b[0] for b in binned])
        y = np.array([b[1] for b in binned])
        # Slope of the ramp itself -- the number that says whether a tomographic
        # bin with a different size distribution gets a different m1.
        slope = float(np.polyfit(x, y, 1)[0])
        out[lab] = {"all": float(m1(base)), "all_err": err(base),
                    "slope": slope, "rms": float(np.sqrt((y ** 2).mean())),
                    "ptp": float(y.max() - y.min()), "bins": binned}
    return out


def one(name, train_cut, score_weight, seed, data):
    m, dm, d2m = data
    t0 = time.time()
    if train_cut is not None:
        keep = (m[:, 1] / m[:, 0]) < train_cut
        fit = tuple(v[keep] for v in (m, dm, d2m))
    else:
        fit = (m, dm, d2m)
    n90f = int(0.9 * len(fit[0]))
    train_set = tuple(v[:n90f] for v in fit)

    k = jr.key(seed)
    bflow = bulk.train(bulk.build_flow(k, train_set[0]), train_set[0],
                       jr.key(seed + 100), steps=4000)
    bulk_val = float(-jnp.mean(bflow.log_prob(jnp.asarray(fit[0][n90f:]))))

    flow = bulk.build_flow(k, train_set[0], shear=True)
    flow = eqx.tree_at(lambda f: f.bijection.bijection.bijections[1:], flow,
                       bflow.bijection.bijection.bijections)
    flow = shear_top.train(flow, train_set, jr.key(seed + 1), steps=6000,
                           score_weight=score_weight)
    shear_val = float(shear_top.val_nll(
        flow, tuple(v[n90f:] for v in fit), jr.key(99)))

    # Evaluated on the FULL catalog's own 90/10 split, so every arm is scored on
    # the same galaxies whatever it was trained on.
    rec = recovery(flow, m, dm, d2m, int(0.9 * len(m)))
    return {"name": name, "seed": seed, "train_cut": train_cut,
            "score_weight": score_weight, "n_train": int(len(train_set[0])),
            "bulk_val_nll": bulk_val, "shear_val_nll": shear_val,
            "converged": bool(np.isfinite(shear_val) and shear_val < 2 * bulk_val),
            "minutes": (time.time() - t0) / 60, **rec}


def report(res):
    r = res["heldout"]
    print(f"  {res['name']}-s{res['seed']}: m1 {r['all']:+.4f} "
          f"slope {r['slope']:+.4f} RMS {r['rms']:.4f} ptp {r['ptp']:.4f} "
          f"[{res['minutes']:.1f} min]"
          + ("   ** DIVERGED **" if not res["converged"] else ""), flush=True)
    print("     " + "  ".join(f"{b[0]:.2f}:{b[1]:+.4f}" for b in r["bins"]),
          flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arms", nargs="*", default=[])
    p.add_argument("--eval-only", default=None, nargs="?", const="flows/shear.eqx",
                   help="score an existing checkpoint instead of training")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = p.parse_args()

    m, dm, d2m = (np.asarray(v, np.float64) for v in shear_top.load(DATA))

    if a.eval_only:
        n90 = int(0.9 * len(m))
        flow = bulk.build_flow(jr.key(0), m[:n90], shear=True)
        flow = eqx.tree_deserialise_leaves(a.eval_only, flow)
        rec = recovery(flow, m, dm, d2m, n90)
        for lab in ("train", "heldout"):
            print(f"\n{a.eval_only}  {lab}  Mr/Mf < {CUT}")
            print(f"  m1 {rec[lab]['all']:+.4f} +/- {rec[lab]['all_err']:.4f}  "
                  f"slope {rec[lab]['slope']:+.4f}  RMS {rec[lab]['rms']:.4f}  "
                  f"ptp {rec[lab]['ptp']:.4f}")
            for c, v, e in rec[lab]["bins"]:
                print(f"    {c:6.3f}  {v:+.4f} +/- {e:.4f}")
        return

    want = a.arms or list(ARMS)
    if any(w not in ARMS for w in want):
        raise SystemExit(f"unknown arm; have {list(ARMS)}")
    for name in want:
        for seed in a.seeds:
            print(f"\n{'=' * 70}\n{name} seed {seed}\n{'=' * 70}", flush=True)
            res = one(name, *ARMS[name], seed, (m, dm, d2m))
            with open(OUT, "a") as f:
                f.write(json.dumps(res) + "\n")
            report(res)


if __name__ == "__main__":
    main()
