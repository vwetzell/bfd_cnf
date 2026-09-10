"""What log-scales does the trained flow actually use, in- and out-of-sample?

`_bounded_log_scale`'s default cap is max_scale=100 -- a factor 100 per layer,
8 layers deep.  That is what lets a small step outside the training envelope
send |base| to 1e7 and log p to -1e14.  Before tightening it, measure what the
CONVERGED flow uses on real data: if every layer sits inside +-1.5, the cap can
come down by an order of magnitude for free.

Records every call by monkeypatching the module-global, so it sees each layer
of the stack without knowing their classes.  Runs eagerly (no vmap) because the
recording is python-side.

    PYTHONPATH=. python dev/ls_census.py [flows/bulk_v3.eqx]
"""
import os
import sys

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.bijections as bij                            # noqa: E402
import bulk                                                # noqa: E402
import shear as sm                                         # noqa: E402

D = "../bfd_cnf_imsims/data"
FLOW = sys.argv[1] if len(sys.argv) > 1 else "flows/bulk_v3.eqx"
N = int(os.environ.get("N", 400))

REC = []
INP = []
_orig = bij._bounded_log_scale
_orig_call = bij.CoeffNet.__call__


def _spy(u, min_scale=1e-2, max_scale=100.0):
    out = _orig(u, min_scale, max_scale)
    REC.append(np.asarray(out).ravel())
    return out


def _spy_call(self, x):
    INP.append(np.asarray(x).ravel())
    return _orig_call(self, x)


def main():
    m, _, _ = sm.load(f"{D}/moments_bulgedisc_v3.fits")
    m = np.asarray(m, np.float64)
    flow = eqx.tree_deserialise_leaves(FLOW, bulk.build_flow(jr.key(0), m))
    chart = flow.bijection.bijection.bijections[0]

    bij._bounded_log_scale = _spy
    bij.CoeffNet.__call__ = _spy_call
    try:
        def census(rows, label):
            REC.clear(); INP.clear()
            for x in rows:
                flow.log_prob(jnp.asarray(x, jnp.float32))
            v = np.concatenate(REC)
            w = np.abs(np.concatenate([a for a in INP if a.size]))
            per = len(v) / len(rows)
            print(f"  {label:<24s} {per:4.0f} c/row  "
                  f"|ls| p50 {np.percentile(np.abs(v), 50):6.3f} "
                  f"p99 {np.percentile(np.abs(v), 99):6.3f} "
                  f"max {np.abs(v).max():7.3f} cap {np.mean(np.abs(v) > 0.99 * np.log(100)):5.1%}"
                  f" | net-in p99 {np.percentile(w, 99):7.3f} "
                  f"max {w.max():9.3g}")
            return v

        print(f"log-scale census, {FLOW}  (cap = +-{np.log(100.0):.3f})")
        idx = np.random.default_rng(0).choice(len(m), N, replace=False)
        census(m[idx], "training prior")

        # the extrapolation probe: the most extreme training row, walked out
        size = m[:, 1] / m[:, 0]
        i = int(np.argmax(size))
        z_i = chart.transform(jnp.asarray(m[i], jnp.float32))
        print(f"\n  most extreme training row: Mr/Mf {size[i]:.4f}, "
              f"z1 {float(z_i[1]):.3f}")
        for d in (0.0, 0.1, 0.5, 1.0):
            mp = chart.inverse(z_i.at[1].add(d))
            v = census([np.asarray(mp)], f"z1 + {d:.1f}")
            lp = float(flow.log_prob(mp))
            print(f"      -> sum ls {v.sum():+12.4g}   log p {lp:12.5g}")
    finally:
        bij._bounded_log_scale = _orig
        bij.CoeffNet.__call__ = _orig_call


if __name__ == "__main__":
    main()
