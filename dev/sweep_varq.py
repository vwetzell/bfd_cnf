"""Does the Var[Q|m] offset to the second-order target move the m1 ramp?

`varq.py` derives it: `shear._velocity_mse` fits the layer's second derivative
to the per-template `d2m_dg2`, whose minimiser is E[R_t|m], but the density
needs E[R_t|m] - (1/P) d_k[Sigma_{aj,bk} P].  The offset is 7.9% / 93.6% / 7.0%
of the existing target on the [g1g1, g1g2, g2g2] columns, so it is not a small
perturbation.

Same paired design as the (invalid) score_weight sweep: the offset only touches
the SHEAR stage, so the bulk is loaded once from flows/bulk.eqx and shared by
every run.  That put the seed noise at sd 0.0015 against the capacity sweep's
0.0038, which is the whole reason three seeds are enough here.

    python dev/sweep_varq.py
"""
import json
import sys
import time

import equinox as eqx
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402
from sweep_capacity import recovery                  # noqa: E402

DATA = "../bfd_cnf_imsims/data/moments.fits"
OUT = "dev/sweep_varq.jsonl"
SEEDS = [0, 1, 2]


def main():
    m, dm, d2m = (np.asarray(v, np.float64) for v in shear_top.load(DATA))
    n90 = int(0.9 * len(m))
    train_set = tuple(x[:n90] for x in (m, dm, d2m))
    val_set = tuple(x[n90:] for x in (m, dm, d2m))

    bulk_only = eqx.tree_deserialise_leaves(
        "flows/bulk.eqx", bulk.build_flow(jr.key(0), m[:n90]))

    for varq in (False, True):
        for seed in SEEDS:
            t0 = time.time()
            print(f"\n{'=' * 70}\nvarq={varq} seed={seed}\n{'=' * 70}", flush=True)
            flow = bulk.build_flow(jr.key(0), m[:n90], shear=True)
            flow = eqx.tree_at(lambda f: f.bijection.bijection.bijections[1:],
                               flow, bulk_only.bijection.bijection.bijections)
            flow = shear_top.train(flow, train_set, jr.key(seed + 1), varq=varq)
            rec = recovery(flow, m, dm, d2m, n90)
            res = {"varq": varq, "seed": seed,
                   "val_nll": shear_top.val_nll(flow, val_set, jr.key(99)),
                   "minutes": (time.time() - t0) / 60, **rec}
            with open(OUT, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"  varq={varq} s={seed}: heldout m1 {rec['heldout']['all']:+.4f}"
                  f"  RMS {rec['heldout']['rms']:.4f}  "
                  f"ptp {rec['heldout']['ptp']:.4f}  "
                  f"[{res['minutes']:.1f} min]", flush=True)

    rows = [json.loads(l) for l in open(OUT)]
    for tag in (False, True):
        v = [r["heldout"]["rms"] for r in rows if r["varq"] is tag]
        print(f"\nvarq={tag}: RMS " + " ".join(f"{x:.4f}" for x in v) +
              f"   mean {np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}")
    d = np.array([r["heldout"]["rms"] for r in rows if r["varq"] is True]) - \
        np.array([r["heldout"]["rms"] for r in rows if r["varq"] is False])
    sem = np.std(d, ddof=1) / np.sqrt(len(d))
    print(f"paired difference {np.mean(d):+.4f} +/- {sem:.4f} "
          f"({abs(np.mean(d)) / sem:.1f} sigma)")


if __name__ == "__main__":
    main()
