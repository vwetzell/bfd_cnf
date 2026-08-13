"""Does more flow capacity fix the resolution-dependent m1?  One axis at a time.

`dev/check_noiseless_shear_recovery.py` showed the bias is the BULK density: the
shear layer's response is right to 0.3%, but binned on Mr/Mf the noiseless m1
still runs +/-1.7% with an oscillating pattern that train and held-out share --
capacity or fit, not overfitting.  This sweeps the only three things that set
capacity (`bulk.LAYERS`, `NN_WIDTH`, `NN_DEPTH`) and nothing else.

Each point trains a bulk flow from scratch, warm-starts a shear flow from it,
and re-runs the noiseless recovery.  `shear.train` freezes the bulk, so the only
thing these knobs change is p(m) -- which is the whole point.

Read the ranking off RMS(m1) ACROSS BINS, not global m1: the ramp cancels in the
aggregate (memory: global +0.0017 while bins ran -1.7% to +1.7%), so global m1
measures how well the errors happened to offset, not how big they are.

Three baseline seeds are in the config list on purpose.  Every point gets a
different random init, so their spread is the noise floor any other point has to
beat before it means anything.

    python dev/sweep_capacity.py                # every config, in order
    python dev/sweep_capacity.py base width128  # named configs only
"""
import json
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402

DATA = "../bfd_cnf_imsims/data/moments.fits"
OUT = "dev/sweep_capacity.jsonl"
G = 0.02
NBINS = 8

#      name           layers width depth  bulk  shear  seed  shear_lr
CONFIGS = [
    ("base",              8,   64,    2,  4000,  6000, 0, 3e-3),
    ("base-s1",           8,   64,    2,  4000,  6000, 1, 3e-3),
    ("base-s2",           8,   64,    2,  4000,  6000, 2, 3e-3),
    ("layers12",         12,   64,    2,  4000,  6000, 0, 3e-3),
    ("layers16",         16,   64,    2,  4000,  6000, 0, 3e-3),
    ("width128",          8,  128,    2,  4000,  6000, 0, 3e-3),
    ("width256",          8,  256,    2,  4000,  6000, 0, 3e-3),
    ("depth3",            8,   64,    3,  4000,  6000, 0, 3e-3),
    ("depth4",            8,   64,    4,  4000,  6000, 0, 3e-3),
    # Not a capacity point: the control that says whether 4000/6000 steps was
    # already enough.  If this moves as much as the capacity points do, the
    # sweep is measuring the optimiser, not the model.
    ("base-2xsteps",      8,   64,    2,  8000, 12000, 0, 3e-3),

    # Retries at a lower shear lr.  `shear.train` evaluates the FROZEN bulk at
    # sheared -- i.e. off-manifold -- moments, and a deeper bulk extrapolates
    # far worse there: layers12 opened at nll 157 against the baseline's 42.8
    # and lr 3e-3 threw it straight to 1e22, where it stayed (velocity mse
    # 0.57 vs 2.5e-3, so ShearResponse never trained at all).  A diverged run
    # answers nothing about capacity, so the point has to be re-run to count.
    # lr is a TRAINING knob, not a flow knob -- the architecture is untouched.
    ("layers12-lr1e3",   12,   64,    2,  4000,  6000, 0, 1e-3),
    ("layers12-lr3e4",   12,   64,    2,  4000,  6000, 0, 3e-4),
    # layers16 gets no shear-lr retry: its BULK nll is already inf (init 6.5e25,
    # overflowed by step 500), so the shear stage's lr is irrelevant.  The one
    # attempt worth making lowers the bulk lr too -- if the initial forward pass
    # can be walked down before it overflows, the point is recoverable; if not,
    # depth > 12 is out of reach without touching the initialisation, which is
    # an architecture change and out of scope here.
    ("layers16-blr3e4",  16,   64,    2,  4000,  6000, 0, 3e-4, 3e-4),
    # The baseline at the same lowered lr, so a retry is compared against
    # something trained the same way rather than against lr 3e-3.
    ("base-lr1e3",        8,   64,    2,  4000,  6000, 0, 1e-3),
    ("base-lr3e4",        8,   64,    2,  4000,  6000, 0, 3e-4),

    # width128 at the SAME two extra inits the baseline got, so width is judged
    # on a paired comparison (base-s1 vs width128-s1) rather than against a
    # 3-seed mean.  Width at seed 0 came in at RMS 0.0135 against a baseline
    # spread of 0.0143-0.0218 -- consistent with a real gain, and equally
    # consistent with seed 0 simply being a good init at any width.  Two seeds
    # is what separates those.
    ("width128-s1",       8,  128,    2,  4000,  6000, 1, 3e-3),
    ("width128-s2",       8,  128,    2,  4000,  6000, 2, 3e-3),

    # Net DEPTH is the only axis with a monotone trend rather than a plateau:
    # RMS 0.0184 (mean of 3 seeds at depth 2) -> 0.0124 -> 0.0103, and depth4
    # beats width256 with a fifth of the parameters, so it is not parameter
    # count.  Pair it against the baseline's own inits, then push the axis until
    # it saturates or hits the same init ceiling the LAYER count did.
    ("depth4-s1",         8,   64,    4,  4000,  6000, 1, 3e-3),
    ("depth4-s2",         8,   64,    4,  4000,  6000, 2, 3e-3),
    ("depth5",            8,   64,    5,  4000,  6000, 0, 3e-3),
    ("depth6",            8,   64,    6,  4000,  6000, 0, 3e-3),
    # depth6 is the last single-seed point that looks good (0.0100 at seed 0).
    # Three times now a seed-0 win has evaporated on pairing; leaving this one
    # unpaired would just invite the same error a fourth time.
    ("depth6-s1",         8,   64,    6,  4000,  6000, 1, 3e-3),
    ("depth6-s2",         8,   64,    6,  4000,  6000, 2, 3e-3),
]


def recovery(flow, m, dm, d2m, n90):
    """`dev/check_noiseless_shear_recovery.py`, returned instead of printed."""
    shear_m = lambda s: m + s * G * dm[:, 0, :] + 0.5 * (s * G) ** 2 * d2m[:, 0, :]
    qp, rp = bias.pqr(flow, shear_m(+1))
    qm, rm = bias.pqr(flow, shear_m(-1))
    qp, rp, qm, rm = (np.asarray(v, np.float64) for v in (qp, rp, qm, rm))

    r = m[:, 1] / m[:, 0]
    rng = np.random.default_rng(0)
    m1 = lambda s: bias.bias(qp, rp, qm, rm, G, s)[0]

    def err(s, n=200):
        i0 = np.flatnonzero(s)
        return float(np.std([m1(rng.choice(i0, len(i0))) for _ in range(n)]))

    out = {}
    for lab, base in (("train", np.arange(len(m)) < n90),
                      ("heldout", np.arange(len(m)) >= n90)):
        q = np.quantile(r[base], np.linspace(0, 1, NBINS + 1))
        q[0], q[-1] = -np.inf, np.inf
        binned = [(float(r[s].mean()), float(m1(s)), err(s))
                  for k in range(NBINS)
                  for s in [base & (r >= q[k]) & (r < q[k + 1])]]
        vals = np.array([b[1] for b in binned])
        out[lab] = {"all": float(m1(base)), "all_err": err(base),
                    "rms": float(np.sqrt((vals ** 2).mean())),
                    "ptp": float(vals.max() - vals.min()), "bins": binned}
    return out


def one(name, layers, width, depth, bulk_steps, shear_steps, seed, shear_lr,
        data, bulk_lr=1e-3):
    m, dm, d2m = data
    n90 = int(0.9 * len(m))
    t0 = time.time()
    dims = dict(layers=layers, nn_width=width, nn_depth=depth)

    k_build = jr.key(seed)
    bflow = bulk.build_flow(k_build, m[:n90], **dims)
    bflow = bulk.train(bflow, m[:n90], jr.key(seed + 100), steps=bulk_steps,
                       lr=bulk_lr)
    bulk_val = float(-jnp.mean(bflow.log_prob(jnp.asarray(m[n90:]))))

    # Same warm start as `shear.py train --init`: the bulk sublayers are copied
    # in and then frozen, so `shear.train` only ever fits ShearResponse.
    flow = bulk.build_flow(k_build, m[:n90], shear=True, **dims)
    flow = eqx.tree_at(lambda f: f.bijection.bijection.bijections[1:], flow,
                       bflow.bijection.bijection.bijections)
    train_set = tuple(x[:n90] for x in (m, dm, d2m))
    flow = shear_top.train(flow, train_set, jr.key(seed + 1), steps=shear_steps,
                           lr=shear_lr)
    shear_val = float(shear_top.val_nll(
        flow, tuple(x[n90:] for x in (m, dm, d2m)), jr.key(99)))

    n_par = sum(x.size for x in
                jax.tree.leaves(eqx.filter(bflow, eqx.is_inexact_array)))
    rec = recovery(flow, m, dm, d2m, n90)
    # A shear stage that blew up produces a flow whose m1 means nothing.  Judge
    # it on its own val nll, not on a threshold in m1: 41.3 is where every
    # converged run lands, and a diverged one misses by many orders.
    converged = bool(np.isfinite(shear_val) and shear_val < 2.0 * bulk_val)
    return {"name": name, "layers": layers, "width": width, "depth": depth,
            "bulk_steps": bulk_steps, "shear_steps": shear_steps, "seed": seed,
            "shear_lr": shear_lr, "bulk_lr": bulk_lr,
            "bulk_params": int(n_par), "bulk_val_nll": bulk_val,
            "shear_val_nll": shear_val, "converged": converged,
            "minutes": (time.time() - t0) / 60,
            **rec}


def main():
    want = sys.argv[1:]
    todo = [c for c in CONFIGS if not want or c[0] in want]
    if want and len(todo) != len(want):
        raise SystemExit(f"unknown config(s); have {[c[0] for c in CONFIGS]}")

    m, dm, d2m = (np.asarray(v, np.float64) for v in shear_top.load(DATA))
    for cfg in todo:
        print(f"\n{'=' * 70}\n{cfg[0]}: layers={cfg[1]} width={cfg[2]} "
              f"depth={cfg[3]} steps={cfg[4]}/{cfg[5]} seed={cfg[6]} "
              f"shear_lr={cfg[7]:g} bulk_lr={(cfg[8] if len(cfg) > 8 else 1e-3):g}"
              f"\n{'=' * 70}",
              flush=True)
        res = one(*cfg[:8], (m, dm, d2m), *cfg[8:])
        with open(OUT, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"  {res['name']}: bulk nll {res['bulk_val_nll']:.4f}  "
              f"params {res['bulk_params']}  "
              f"heldout m1 {res['heldout']['all']:+.4f}  "
              f"RMS {res['heldout']['rms']:.4f}  "
              f"ptp {res['heldout']['ptp']:.4f}  "
              f"[{res['minutes']:.1f} min]"
              + ("   ** DIVERGED **" if not res["converged"] else ""), flush=True)


if __name__ == "__main__":
    main()
