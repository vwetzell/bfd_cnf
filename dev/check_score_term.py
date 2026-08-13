"""Did `--score-weight` actually move its own term?

A null arm is only informative if the lever moved -- the distinction
[[response-lever-is-exhausted]] drew between a true null (band-weighting, which
improved its target 4-7% and still did nothing to the bias) and a no-op.  The
sweep throws the training curves away, so read the term straight off the
checkpoints instead: `first` is the score-contracted first-order residual the
weight multiplies, evaluated on the held-out 10%.

    python dev/check_score_term.py
"""
import sys

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

sys.path.insert(0, ".")

import bulk                                          # noqa: E402
import shear                                         # noqa: E402

DATA = "../bfd_cnf_imsims/data/moments.fits"
FLOWS = {"ref": "flows/shear_bounded.eqx", "sw1": "flows/shear_sw1.eqx",
         "sw3": "flows/shear_sw3.eqx", "sw10": "flows/shear_sw10.eqx"}


def main():
    data = shear.load(DATA)
    n = int(0.9 * len(data[0]))
    train_m = data[0][:n]
    m, q, r = (jnp.asarray(a[n:]) for a in data)

    print(f"{'arm':>5s} {'score term':>12s} {'velocity mse':>14s} {'val nll':>10s}")
    for name, path in FLOWS.items():
        flow = eqx.tree_deserialise_leaves(
            path, bulk.build_flow(jr.key(0), train_m, shear=True))
        mse, first = shear._velocity_mse(shear._shear_layer(flow), m, q, r,
                                         shear.bulk_score(flow, m))
        nll = shear.val_nll(flow, (m, q, r), jr.key(99))
        print(f"{name:>5s} {float(first):>12.4f} {float(mse):>14.3e} {nll:>10.4f}")


if __name__ == "__main__":
    main()
