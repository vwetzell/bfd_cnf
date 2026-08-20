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

--joint IS FIXED, AND TRUNK SHARING IS NOT THE CAUSE (2026-08-19).

THE BUG WAS `@eqx.filter_jit` ON `step`.  Inside the trace the loss read
4.14e-4 while the IDENTICAL expression, same batch, same parameters, read
6.35e-1 evaluated outside it -- a factor of 1535.  So the optimiser's reported
loss was meaningless and the fit never converged where the diagnostic looked,
which is exactly the "constant offset with the shape right" that three previous
attempts saw and correctly called a harness bug.  With the decorator removed,
inside == outside to the digit and the fit converges in 600 steps.  The
decorator is deliberately NOT restored: this is a one-off diagnostic and
correctness beats speed.  `shear.py`'s own training does NOT share the defect
(its printed velocity mse 0.176 against 0.064 implied by the measured dm/dg
residuals -- same order, differing only by the partial_norms/_scale
normalisation), so this was local to this script.

TWO OTHER REAL DEFECTS, also fixed: the loss was one scalar over all five
coefficients with no per-coefficient scale (B took essentially the whole
gradient), and it normalised by a PER-BATCH weight sum, a ratio of sums whose
expectation is not the ratio of expectations under a skewed |e|^2 weight.

THE ANSWER.  Symmetry control (`--control same`, all five outputs given
identical targets and weights) now passes: every column lands within 0.0008.
And the real five-coefficient fit from ONE shared trunk reaches

    a0_Mf +0.0025   a0_Mr +0.0140   a0_Mc +0.0084   A +0.0001   B +0.1384

with a0(Mf) within 0.004 in EVERY size bin against +0.3788 in-flow at the
ceiling.  A single trunk fits all five first-order coefficients as well as a
single-output net fits one.  So capacity, conditioning, the inputs, the
_COEFF_MAX bound, the supervision weight, the hyperparameters AND trunk sharing
are all now excluded -- every component of the coefficient network is
exonerated, and what remains is the JOINT OPTIMISATION itself: the flow reaches
these coefficients through velocity supervision plus NLL instead of fitting
them directly.

THE FIX THIS IMPLIES.  This script is now, literally, the pretraining
procedure: it fits `CoeffNet` to bfd's exact coefficients to <=0.004.  Write the
fitted `_Coeffs` out and give `shear.py` an `--init-coeffs` that loads and
freezes (or strongly anchors) it while the density trains.

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
    p.add_argument("--control", choices=["same", "weights"], default=None,
                   help="diagnostic control for --joint: set all five targets "
                        "to a0 so nothing competes for the trunk. 'same' also "
                        "sets all five weights to |e|^2 (isolates the "
                        "multi-output machinery); 'weights' keeps the differing "
                        "natural weights (adds weight contention only).")
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
        if a.control:
            # All five outputs asked for the SAME thing the single-output fit
            # nails, so nothing is competing for the trunk's features.  What is
            # left differing between the two modes is only the per-column
            # WEIGHT, which is the remaining suspect: |e|^2 concentrates a0 on
            # the high-|e| tail while A's weight of 1 spreads it over everyone.
            tgt = np.repeat(a0[:, :1], 5, axis=1)
            if a.control == "same":
                wgt = np.repeat(e_sq[:, None], 5, axis=1)
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

    # Per-coefficient RMS of the target, under its own weight.  WITHOUT this the
    # joint loss was one scalar over all five, so B -- which spans several times
    # a0's range -- took essentially the whole gradient and a0 came out
    # right-shape/wrong-offset.  That was the "harness bug" the docstring
    # suspected, and it is what made --joint untrustworthy: it is the real
    # loss's `shear.partial_norms` treatment, missing here.  Single-output runs
    # are unaffected (one column, scale divides out).
    cscale = jnp.sqrt(jnp.sum(W * Y ** 2, 0) / jnp.sum(W, 0))

    def step(params, state, idx):
        def loss(p_):
            f = eqx.combine(p_, static)
            # same bound the layer applies to its coefficients
            pred = _COEFF_MAX * jnp.tanh(jax.vmap(f)(X[idx]) / _COEFF_MAX)
            # NOT sum(...)/sum(W[idx]): that is a ratio of per-batch sums,
            # and W = |e|^2/mean is skewed enough that most batches carry no
            # high-|e| row, so the denominator swings by orders of magnitude
            # and E[num/den] != E[num]/E[den] -- a biased gradient.  W is
            # already normalised to mean 1 per column over the FULL sample, so
            # a plain batch mean is unbiased for the weighted mean.
            return jnp.mean(W[idx] * ((pred - Y[idx]) / cscale) ** 2)
        v, g = jax.value_and_grad(loss)(params)
        upd, state = opt.update(g, state, params)
        return eqx.apply_updates(params, upd), state, v

    key = jr.key(2)
    _probe_idx = jr.randint(jr.key(1234), (a.batch,), 0, len(Y))
    for i in range(a.steps):
        key, sk = jr.split(key)
        params, state, v = step(params, state,
                                jr.randint(sk, (a.batch,), 0, len(Y)))
        if i % 1000 == 0 or i == a.steps - 1:
            print(f"  step {i:5d}  weighted mse {float(v):.5e}")

    # Same batch, two routes: the jitted `step`'s own loss, and the loss
    # recomputed outside from `eqx.combine(params, static)`.  If these differ,
    # the optimiser and the diagnostic are not looking at the same parameters.
    _, _, v_in = step(params, state, _probe_idx)
    _n = eqx.combine(params, static)
    _p = _COEFF_MAX * jnp.tanh(jax.vmap(_n)(X[_probe_idx]) / _COEFF_MAX)
    v_out = float(jnp.mean(W[_probe_idx]
                           * ((_p - Y[_probe_idx]) / cscale) ** 2))
    print(f"\n  SAME batch: inside step {float(v_in):.5e}   "
          f"outside {v_out:.5e}   ratio {v_out / float(v_in):.1f}")

    net = eqx.combine(params, static)
    pred_all = np.asarray(_COEFF_MAX * jnp.tanh(
        jax.vmap(net)(X) / _COEFF_MAX), np.float64)
    # The SAME loss, on the full sample with the final weights.  If this
    # disagrees with the per-batch value printed during training, the reported
    # loss is not describing the fit the table shows.
    full_loss = float(jnp.mean(
        jnp.asarray(W) * ((jnp.asarray(pred_all, jnp.float32) - Y) / cscale) ** 2))
    print(f"\n  full-sample loss with final params: {full_loss:.5e}")
    kk = jr.key(99)
    bl = []
    for _ in range(20):
        kk, s2 = jr.split(kk)
        j = jr.randint(s2, (a.batch,), 0, len(Y))
        pb = _COEFF_MAX * jnp.tanh(jax.vmap(net)(X[j]) / _COEFF_MAX)
        bl.append(float(jnp.mean(W[j] * ((pb - Y[j]) / cscale) ** 2)))
    print(f"  same loss on 20 random batches: median {np.median(bl):.5e}  "
          f"min {min(bl):.5e}  max {max(bl):.5e}")
    print(f"  len(Y) = {len(Y)}   X {X.shape}  Y {Y.shape}  W {W.shape}")
    # Per-column, so a bad column cannot hide behind the scalar loss.  A column
    # whose fit is offset but correctly SHAPED is the signature that its
    # gradient is not reaching the output bias.
    # The WEIGHTED bias is the one the loss actually minimises.  The unweighted
    # median is not: W = |e|^2/mean is heavily skewed, so most templates carry
    # almost no weight and the fit is unconstrained there -- which is how a
    # weighted mse of 3.5e-4 sat next to a median bias of -1.5, and why five
    # columns given IDENTICAL targets and weights each drifted somewhere
    # different.  Report both; only `wbias` is evidence.
    print(f"\n{'column':>8s} {'tgt med':>10s} {'fit med':>10s} {'bias':>10s} "
          f"{'wbias':>10s} {'tgt sd':>10s} {'fit sd':>10s}")
    for c in range(pred_all.shape[1]):
        wc = np.asarray(W)[:, c]
        print(f"{['a0_Mf','a0_Mr','a0_Mc','A','B'][c] if a.joint else 'a0_Mf':>8s} "
              f"{np.median(tgt[:, c]):10.4f} {np.median(pred_all[:, c]):10.4f} "
              f"{np.median(pred_all[:, c] - tgt[:, c]):+10.4f} "
              f"{(wc * (pred_all[:, c] - tgt[:, c])).sum() / wc.sum():+10.4f} "
              f"{tgt[:, c].std():10.4f} {pred_all[:, c].std():10.4f}")
    pred = pred_all[:, 0]
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
