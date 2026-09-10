"""Where does `pqr_streamed`'s wall clock actually go?

The 20k-target matched runs cost 40 min each (0.12 s/target over three arms),
which puts a 1M-target measurement at 33 h.  Before spending that, split the
per-chunk cost into its three stages and see which one is worth attacking:

    mixture   `mixture_draws` -- the proposal.  At alpha < 1 half the draws come
              from the flow BASE -> DATA, i.e. a Newton solve plus a jacfwd
              log-det per draw (bias.py:682).
    centroid  `centroid_transform` of the draws.
    rest      the forward-over-reverse Hessian (`batched`) plus the host merge.

Monkeypatches the two globals `pqr_streamed` looks up rather than
reconstructing the gauge plumbing by hand, so it times the REAL call path.

    python -u dev/bench_pqr.py --pop gauss2_v3 --flow flows/centroid_g2v3.eqx
    python -u dev/bench_pqr.py --alpha 1.0 --batch-budget 262144
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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

p = argparse.ArgumentParser()
p.add_argument("--pop", default="gauss2_v3")
p.add_argument("--flow", default="flows/centroid_g2v3.eqx")
p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
p.add_argument("--n-targets", type=int, default=256)
p.add_argument("--samples", type=int, default=8192)
p.add_argument("--chunk", type=int, default=4096)
p.add_argument("--batch-budget", type=int, default=65536)
p.add_argument("--alpha", type=float, default=0.5)
p.add_argument("--gauge", default="auto")
p.add_argument("--floor-eps", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

path = lambda v: f"{a.data_dir}/{v}.fits"
cat = B.CATALOGS[a.pop]
m_train = shear.load(f"{a.data_dir}/{B.TRAIN_DATA[a.pop]}")[0]
flow = bulk.build_flow(jr.key(a.seed), m_train, shear=True, centroid=True)
flow = eqx.tree_deserialise_leaves(a.flow, flow)
flow = bulk.SupportedFlow(flow, eps=a.floor_eps, broad_std=4, support=True)

rows = fitsio.read(path(cat["zero"]))[: a.n_targets]
rows = rows[~rows["badcenter"]]
m = np.asarray(rows["moments"], dtype=np.float64)
sigma_x = np.asarray(rows["cov_odd"], dtype=np.float64)
cov = B.load_cov(path(cat["zero"]))

chunk = min(a.samples, a.chunk)
batch = max(1, a.batch_budget // chunk)
print(f"{len(m)} targets, {a.samples} draws in chunks of {chunk}, "
      f"{batch} targets/batch, alpha {a.alpha}, gauge {a.gauge}")

# --- timing shims over the two globals `pqr_streamed` looks up -------------
T = {"mixture": 0.0, "centroid": 0.0}
_mix, _ctr = B.mixture_draws, B.centroid_transform


def timed_mixture(*args, **kw):
    t = time.perf_counter()
    out = _mix(*args, **kw)          # returns numpy, so already synced
    T["mixture"] += time.perf_counter() - t
    return out


def timed_centroid(*args, **kw):
    t = time.perf_counter()
    out = _ctr(*args, **kw)
    jax.block_until_ready(out)
    T["centroid"] += time.perf_counter() - t
    return out


B.mixture_draws, B.centroid_transform = timed_mixture, timed_centroid

run = lambda mm, ss: B.pqr_streamed(
    flow, mm, cov, ss, a.alpha, a.seed, sigma_x=sigma_x[: len(mm)],
    batch=batch, chunk=chunk, gauge=a.gauge)

# `pqr_streamed` builds its `eqx.filter_jit` wrappers INSIDE the call, so every
# call is a fresh jit cache entry and pays the full compile.  Run at n and 2n
# and difference: the compile is common to both and drops out, leaving the
# marginal per-target cost -- which is the only thing a 1M run scales.
def measure(mm):
    T["mixture"] = T["centroid"] = 0.0
    t0 = time.perf_counter()
    run(mm, a.samples)
    return time.perf_counter() - t0, T["mixture"], T["centroid"]


print(f"  peel: centroid layer {'FOUND' if B.split_centroid(flow)[1] is not None else 'ABSENT'}")
half = len(m)
t1, _, _ = measure(m)                       # pays the compile
total, mix, ctr = measure(m)                # cache warm: the real cost
rest = total - mix - ctr
per = total / half
print(f"\n  compile   {t1 - total:7.2f} s  (first pqr_streamed call only)")
print(f"  mixture   {mix:7.2f} s  {mix/total:5.1%}")
print(f"  centroid  {ctr:7.2f} s  {ctr/total:5.1%}")
print(f"  rest      {rest:7.2f} s  {rest/total:5.1%}   (Hessian + merge)")
print(f"  MARGINAL  {total:7.2f} s / {half} targets   {per*1000:.1f} ms/target/arm")
print(f"  => {per * 3 * 1e6 / 3600:.1f} h for 1M targets x 3 arms")
