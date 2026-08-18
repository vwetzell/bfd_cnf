"""Can the coefficient network represent a0(Mf) at all, outside the flow?

The layer's Mf response coefficient is 19% too large at the point-source end and
the error is a size-dependent systematic.  It survives a 35x reweighting of the
supervision toward that end (`shear.partial_norms(nbin=20)`, null), so it is not
that the loss under-asks.  Three structural suspects are already excluded: at
first order a0_layer is proportional to s0[0,0] alone (p2 and p3 have vanishing
g-derivative at g = 0), the required s0[0,0] is ~2.4 against a _COEFF_MAX of 12
so nothing saturates, and the net's inputs carry the same information as the
(r, k, q) that `response_scatter` fits to a 0.92% floor in that bin.

This removes the flow entirely.  Same architecture (`CoeffNet`, same width and
depth), same inputs (z0, z1, z2, q from the TRAINED chart), same bound, fit
straight to bfd's a0 with its natural |e|^2 weight -- `_spin0_a` shows a0
inverts out of dm/dg with |e|^2 as the weight, since a round galaxy has no
first-order spin-0 response and nothing to say about a0.

    recovers a0 across all size bins -> the net and inputs are fine, and the
                                        problem is the JOINT optimisation
    fails at the point-source end     -> the net or the inputs cannot do it, and
                                        "representable" was wrong

    python dev/check_a0_representable.py

ANSWER (2026-08-18): a0 IS FULLY REPRESENTABLE.  Same architecture, same inputs
from the trained chart, same _COEFF_MAX bound, same |e|^2 weight:

    Mr/Mf <   a0 truth   a0 fit      bias    in-flow bias
      2.880     1.4283   1.4318   +0.0018        +0.0009
      3.130     1.6847   1.6834   +0.0014        +0.0248
      3.271     1.8219   1.8195   +0.0006        +0.1000
      3.361     1.9096   1.9073   -0.0007        +0.2022
      3.430     1.9672   1.9645   -0.0007        +0.2865
      3.500     2.0144   2.0134   +0.0010        +0.3788

Bias <= 0.002 in every bin against +0.379 in-flow -- 380x better at the ceiling.
So the network and the inputs are fine and the failure is the JOINT
OPTIMISATION, not capacity, not conditioning, not the inputs.

AND IT IS NOT THE OBVIOUS HYPERPARAMETERS.  Top-bin a0 bias against the +0.3784
default: lr 1e-3 -> +0.3953, lr 1e-2 -> +0.4194, 24000 steps -> +0.4046,
batch 4096 -> +0.4024.  Three times the learning rate either way, four times the
steps, four times the batch, and the ceiling bin never moves; several are
slightly worse.  (24k steps DOES help the middle bins, 0.100 -> 0.046, while the
ceiling degrades -- so the optimiser is trading one against the other.)

--joint IS BROKEN, TWICE, AND SHOULD NOT BE TRUSTED.  It was meant to test
whether sharing a trunk across coefficients is what costs a0 its accuracy, since
the layer emits fourteen outputs and the working fit above emits one.  Both
attempts return a CONSTANT ~-1.56 offset in a0 with the shape essentially
correct (fit spans 0.60 across the size range against truth's 0.59), and a
weighted mse of 0.22 against 1e-3 for the single-output fit.  A right-shape,
wrong-offset result is the signature of a bug in the harness, not a finding.
The first attempt was dominated by ill-determined B (unweighted |max| 5860
against _COEFF_MAX 12); guarding that changed nothing, so that was not the cause
either.  The trunk-sharing question remains OPEN and this is not the instrument
to answer it with.

Checked in passing and NOT a problem: _COEFF_MAX = 12 is adequate.  Unweighted,
B runs p1 -24.8 to p99 +32.9, which looks alarming -- but B is carried with
weight |e|^4 precisely because it blows up as e -> 0, and under that weight it is
p1 +1.04, p50 +2.22, p99 +5.22 with 0.027% of the weight beyond 12.  a_Mc is
likewise tame (p99 +4.91, 0.004% beyond).  The module docstring's "[-2, 7]"
stands.
"""
import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402
from models.bijections import CoeffNet               # noqa: E402
from models.shear import _COEFF_MAX, _Q_LOC, _Q_SCALE  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear_cfo.eqx")
    p.add_argument("--data", default="../bfd_cnf_imsims/data/moments.fits")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--width", type=int, default=bulk.NN_WIDTH)
    p.add_argument("--depth", type=int, default=bulk.NN_DEPTH)
    p.add_argument("--joint", action="store_true",
                   help="fit all FIVE first-order coefficients (a_Mf, a_Mr, "
                        "a_Mc, A, B) from one shared trunk, as the layer does "
                        "with fourteen. Isolates whether trunk sharing is what "
                        "costs a0 its accuracy at the point-source end.")
    a = p.parse_args()

    full = shear_top.load(a.data)
    m, q_true = np.asarray(full[0], np.float64), np.asarray(full[1], np.float64)
    flow = bulk.build_flow(jr.key(0), full[0], shear=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)
    chart = shear_top._chart(flow)

    r = m[:, 1] / m[:, 0]
    keep = ((r >= bias.SIZE_WINDOW[0]) & (r <= bias.SIZE_WINDOW[1])
            & (m[:, 0] >= bias.FLUX_WINDOW[0]) & (m[:, 0] <= bias.FLUX_WINDOW[1]))
    m, q_true, r = m[keep], q_true[keep], r[keep]

    a0, e_sq = shear_top._spin0_a(m, q_true)          # (n, 3) for Mf, Mr, Mc
    A, B, e4 = shear_top._spin2_AB(m, q_true)
    if a.joint:
        # the five first-order coefficients, each with its own natural weight:
        # |e|^2 for the spin-0 a_X, 1 for A, |e|^4 for B (`_spin2_AB`).
        tgt = np.stack([a0[:, 0], a0[:, 1], a0[:, 2], A, B], -1)
        wgt = np.stack([e_sq, e_sq, e_sq, np.ones_like(e_sq), e4], -1)
    else:
        tgt = a0[:, :1]
        wgt = e_sq[:, None]
    # Drop templates whose target the BOUNDED output cannot produce.  B and
    # a_Mc are `coefficient / |e|^k` and blow up as e -> 0, where they are also
    # meaningless -- unweighted B reaches 5860 against a _COEFF_MAX of 12, and
    # (pred - 5860)^2 swamps a shared trunk even at the small |e|^4 weight those
    # templates carry.  Under the natural weights the coefficients are tame
    # (B: p1 +1.04, p50 +2.22, p99 +5.22; only 0.027% of the weight is beyond
    # 12), so this removes noise, not signal.
    ok = (np.isfinite(tgt).all(1) & np.isfinite(wgt).all(1) & (e_sq > 1e-6)
          & (np.abs(tgt) < _COEFF_MAX).all(1))
    m, r, tgt, e_sq, wgt = m[ok], r[ok], tgt[ok], e_sq[ok], wgt[ok]
    print(f"{a.flow}: {len(m)} in-window templates with a usable a0")

    # the net's ACTUAL inputs, from the trained chart
    z = np.asarray(jax.vmap(chart.transform)(jnp.asarray(m, jnp.float32)), np.float64)
    q = z[:, 3] ** 2 + z[:, 4] ** 2
    X = jnp.asarray(np.stack([z[:, 0], z[:, 1], z[:, 2],
                              (q - _Q_LOC) / _Q_SCALE], -1), jnp.float32)
    Y = jnp.asarray(tgt, jnp.float32)
    # each coefficient's weight normalised to mean 1 so no one of them
    # dominates the shared trunk purely through its units
    W = jnp.asarray(wgt / wgt.mean(0), jnp.float32)

    net = CoeffNet(jr.key(1), 4, Y.shape[1], a.width, a.depth, jax.nn.silu)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(optax.cosine_decay_schedule(a.lr, a.steps)))
    params, static = eqx.partition(net, eqx.is_inexact_array)
    state = opt.init(params)

    @eqx.filter_jit
    def step(params, state, idx):
        def loss(p_):
            f = eqx.combine(p_, static)
            # same bound the layer applies to its coefficients
            pred = _COEFF_MAX * jnp.tanh(jax.vmap(f)(X[idx]) / _COEFF_MAX)
            return jnp.sum(W[idx] * (pred - Y[idx]) ** 2) / jnp.sum(W[idx])
        v, g = jax.value_and_grad(loss)(params)
        upd, state = opt.update(g, state, params)
        return eqx.apply_updates(params, upd), state, v

    key = jr.key(2)
    for i in range(a.steps):
        key, sk = jr.split(key)
        params, state, v = step(params, state,
                                jr.randint(sk, (a.batch,), 0, len(Y)))
        if i % 1000 == 0 or i == a.steps - 1:
            print(f"  step {i:5d}  weighted mse {float(v):.5e}")

    net = eqx.combine(params, static)
    pred = np.asarray(_COEFF_MAX * jnp.tanh(
        jax.vmap(net)(X) / _COEFF_MAX), np.float64)[:, 0]
    tgt = tgt[:, 0]

    print("\na0(Mf) bias by size -- isolated fit vs what the flow achieves")
    print(f"  {'Mr/Mf <':>9s} {'a0 truth':>10s} {'a0 fit':>9s} {'bias':>9s} "
          f"{'in-flow bias':>13s}")
    inflow = [+0.0009, +0.0248, +0.1000, +0.2022, +0.2865, +0.3788]
    ed = np.quantile(r, np.linspace(0, 1, 7))
    for i in range(6):
        s = (r >= ed[i]) & (r < (ed[i + 1] if i < 5 else np.inf))
        print(f"  {ed[i+1]:9.3f} {np.median(tgt[s]):10.4f} {np.median(pred[s]):9.4f} "
              f"{np.median(pred[s] - tgt[s]):+9.4f} {inflow[i]:+13.4f}")


if __name__ == "__main__":
    main()
